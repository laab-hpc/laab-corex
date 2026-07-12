#ifndef LAAB_CBLAS_UTILS_H
#define LAAB_CBLAS_UTILS_H

#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>    // uint32_t, uint64_t
#include <sys/types.h> // off_t
#include <sys/stat.h>
#include <unistd.h>
#include <math.h>      // sqrt

#define SCRUB_SIZE (50 * 1024 * 1024)

static inline FILE *laab_open_trace_file_with_tag(const char *tag)
{
    char *trace_dir = getenv("LAAB_TRACE_DIR");
    struct stat st = {0};
    char hostname[HOST_NAME_MAX];
    char trace_path[PATH_MAX];
    FILE *trace_file;

    if (!trace_dir) {
        fprintf(stderr, "Environment variable LAAB_TRACE_DIR not set.\n");
        return NULL;
    }

    if (stat(trace_dir, &st) == -1) {
        fprintf(stderr, "LAAB_TRACE_DIR does not exist.\n");
        return NULL;
    }

    if (gethostname(hostname, sizeof(hostname)) != 0) {
        perror("gethostname failed");
        return NULL;
    }

    hostname[sizeof(hostname) - 1] = '\0';
    if (!tag || !tag[0]) tag = "0";

    snprintf(trace_path, sizeof(trace_path), "%s/traces.%s.%s.log", trace_dir, tag, hostname);

    trace_file = fopen(trace_path, "a");
    if (!trace_file) {
        perror("Failed to open trace file");
        return NULL;
    }

    return trace_file;
}

static inline void cache_scrub(void)
{
    double *scrub = (double *)malloc((size_t)SCRUB_SIZE * sizeof(double));
    if (!scrub) return;

    for (size_t i = 0; i < (size_t)SCRUB_SIZE; ++i) scrub[i] = 0.0;
    free(scrub);
}

static inline size_t dtype_size(const char *dtype)
{
    if (strcmp(dtype, "float32") == 0) return 4;
    if (strcmp(dtype, "float64") == 0) return 8;
    if (strcmp(dtype, "complex128") == 0) return 16;
    return 0;
}

static inline int parse_matrix_filename(const char *filename, int *M, int *N, char *dtype)
{
    const char *name = strrchr(filename, '/');
    name = name ? name + 1 : filename;
    return sscanf(name, "M%dx%d-%31[^-]", M, N, dtype) == 3;
}

static void choose_process_grid(int nprocs, int *nprow, int *npcol)
{
    int r = (int)sqrt((double)nprocs);
    while (r > 1 && (nprocs % r) != 0) --r;
    if (r == 1) {
        *nprow = 1;
        *npcol = nprocs;
    } else {
        *nprow = r;
        *npcol = nprocs / r;
    }
}

typedef struct {
    uint64_t lo;
    uint64_t hi;
} laab_u128_t;

static void *load_block_cyclic_matrix(const char *filename,
                                      int myrow, int mycol, int nprow, int npcol,
                                      int block_rows, int block_cols,
                                      int local_rows, int local_cols)
{
    FILE *fp;
    char *rowbuf;
    char *local;
    size_t elem_size;
    int M, N;
    char dtype[32];

    if (!parse_matrix_filename(filename, &M, &N, dtype)) return NULL;
    elem_size = dtype_size(dtype);
    if (!elem_size) return NULL;

    local = (char *)malloc((size_t)local_rows * (size_t)local_cols * elem_size);
    if (!local) return NULL;

    fp = fopen(filename, "rb");
    if (!fp) {
        free(local);
        return NULL;
    }
    rowbuf = (char *)malloc((size_t)block_cols * elem_size);
    if (!rowbuf) {
        fclose(fp);
        free(local);
        return NULL;
    }

    for (int m = 0, local_start_col = 0; local_start_col < local_cols; ++m, local_start_col += block_cols) {
        int global_tile_col = mycol + m * npcol;

        for (int k = 0, local_start_row = 0; local_start_row < local_rows; ++k, local_start_row += block_rows) {
            int global_tile_row = myrow + k * nprow;
            int row_start = global_tile_row * block_rows;
            int col_start = global_tile_col * block_cols;
            int tile_rows, tile_cols;

            if (row_start >= M || col_start >= N) {
                free(rowbuf);
                fclose(fp);
                free(local);
                return NULL;
            }

            tile_rows = (M - row_start < block_rows) ? (M - row_start) : block_rows;
            tile_cols = (N - col_start < block_cols) ? (N - col_start) : block_cols;
            if (local_start_row + tile_rows > local_rows ||
                local_start_col + tile_cols > local_cols) {
                free(rowbuf);
                fclose(fp);
                free(local);
                return NULL;
            }

            for (int row_offset = 0; row_offset < tile_rows; ++row_offset) {
                off_t off = (off_t)(((size_t)(row_start + row_offset) * (size_t)N +
                                     (size_t)col_start) * elem_size);
                if (fseeko(fp, off, SEEK_SET) != 0 ||
                    fread(rowbuf, elem_size, (size_t)tile_cols, fp) != (size_t)tile_cols) {
                    free(rowbuf);
                    fclose(fp);
                    free(local);
                    return NULL;
                }

                for (int col_offset = 0; col_offset < tile_cols; ++col_offset) {
                    int local_r = local_start_row + row_offset;
                    int local_c = local_start_col + col_offset;
                    size_t idx = (size_t)local_r + (size_t)local_c * (size_t)local_rows;

                    if (elem_size == 4) {
                        ((uint32_t *)local)[idx] = ((uint32_t *)rowbuf)[col_offset];
                    } else if (elem_size == 8) {
                        ((uint64_t *)local)[idx] = ((uint64_t *)rowbuf)[col_offset];
                    } else if (elem_size == 16) {
                        ((laab_u128_t *)local)[idx] = ((laab_u128_t *)rowbuf)[col_offset];
                    } else {
                        free(rowbuf);
                        fclose(fp);
                        free(local);
                        return NULL;
                    }
                }
            }
        }
    }

    free(rowbuf);
    fclose(fp);
    return local;
}

#endif