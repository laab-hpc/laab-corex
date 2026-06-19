#ifndef LAAB_UTILS_H
#define LAAB_UTILS_H

#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif

#include <stdio.h>     // fprintf, stderr, FILE, fopen, perror, snprintf
#include <stdlib.h>    // getenv, EXIT_FAILURE
#include <string.h>    // strcmp, strrchr
#include <stdint.h>    // uint32_t, uint64_t
#include <unistd.h>    // gethostname
#include <limits.h>    // HOST_NAME_MAX, PATH_MAX
#include <sys/types.h> // off_t
#include <sys/stat.h>  // struct stat, stat, mkdir

static inline FILE *laab_open_trace_file(void)
{
    // Get environment variable for trace directory
    char *trace_dir = getenv("LAAB_TRACE_DIR");
    if (!trace_dir) {
        fprintf(stderr, "Environment variable LAAB_TRACE_DIR not set.\n");
        return NULL;
    }

    // Ensure LAAB_TRACE_DIR exists
    struct stat st = {0};
    if (stat(trace_dir, &st) == -1) {
        fprintf(stderr,"LAAB_TRACE_DIR does not exist.\n");
        return NULL;
    }

    // Create trace file in append mode
    char hostname[HOST_NAME_MAX];

    if (gethostname(hostname, sizeof(hostname)) != 0) {
        perror("gethostname failed");
        return NULL;
    }

    hostname[sizeof(hostname) - 1] = '\0';

    char trace_path[PATH_MAX];

    snprintf(
        trace_path,
        sizeof(trace_path),
        "%s/traces.%s.log",
        trace_dir,
        hostname
    );

    FILE *trace_file = fopen(trace_path, "a");
    if (!trace_file) {
        perror("Failed to open trace file");
        return NULL;
    }

    return trace_file;
}

static inline FILE *laab_open_matrix_file(int m, int n, const char* prec, const char* prop, const char* type){
    char *matrix_dir = getenv("LAAB_MATRIX_DIR");
    if (!matrix_dir) {
        fprintf(stderr, "Environment variable LAAB_MATRIX_DIR not set.\n");
        return NULL;
    }

    if (!prop) prop = "gen";
    if (!type) type = "dense";

    char filename[4096];
    snprintf(filename, sizeof(filename), "%s/M%dx%d-%s-%s.%s.txt", matrix_dir, m, n, prec, prop, type);

    FILE *matrix_file = fopen(filename, "rb");
    if (!matrix_file) {
        perror("Failed to open matrix file");
        return NULL;
    }

    return matrix_file;
}

#define BILLION 1000000000L
#define SCRUB_SIZE (50 * 1024 * 1024)

static void cache_scrub(void)
{
    double *scrub = (double *)malloc((size_t)SCRUB_SIZE * sizeof(double));
    if (!scrub) return;

    for (size_t i = 0; i < (size_t)SCRUB_SIZE; ++i) scrub[i] = 0.0;
    free(scrub);
}

static void split_dim(int n, int p, int coord, int *start, int *count)
{
    int base = n / p, rem = n % p;
    *count = base + (coord < rem ? 1 : 0);
    *start = coord < rem ? coord * (base + 1) : rem * (base + 1) + (coord - rem) * base;
}

static size_t dtype_size(const char *dtype)
{
    if (strcmp(dtype, "float32") == 0) return 4;
    if (strcmp(dtype, "float64") == 0) return 8;
    if (strcmp(dtype, "complex128") == 0) return 16;
    if (strcmp(dtype, "int32") == 0) return 4;
    if (strcmp(dtype, "int64") == 0) return 8;
    return 0;
}

typedef struct {
    uint64_t lo;
    uint64_t hi;
} laab_u128_t;

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

static int parse_matrix_filename(const char *filename, int *M, int *N, char *dtype)
{
    const char *name = strrchr(filename, '/');
    name = name ? name + 1 : filename;
    return sscanf(name, "M%dx%d-%31[^-]", M, N, dtype) == 3;
}

static void *load_local_matrix(const char *filename, int myrow, int mycol, int nprow, int npcol,
                               int *M, int *N, char *dtype, int *local_rows, int *local_cols)
{
    FILE *fp;
    void *buf;
    size_t elem_size;
    int row0, col0, i;

    if (!parse_matrix_filename(filename, M, N, dtype)) return NULL;
    elem_size = dtype_size(dtype);
    if (!elem_size) return NULL;
    if (myrow < 0 || myrow >= nprow || mycol < 0 || mycol >= npcol) return NULL;

    split_dim(*M, nprow, myrow, &row0, local_rows);
    split_dim(*N, npcol, mycol, &col0, local_cols);

    buf = malloc((size_t)(*local_rows) * (size_t)(*local_cols) * elem_size);
    if (!buf) return NULL;

    fp = fopen(filename, "rb");
    if (!fp) {
        free(buf);
        return NULL;
    }

    for (i = 0; i < *local_rows; ++i) {
        off_t off = (off_t)(((size_t)(row0 + i) * (size_t)(*N) + (size_t)col0) * elem_size);
        if (fseeko(fp, off, SEEK_SET) != 0 ||
            fread((char *)buf + (size_t)i * (size_t)(*local_cols) * elem_size,
                  elem_size, (size_t)(*local_cols), fp) != (size_t)(*local_cols)) {
            fclose(fp);
            free(buf);
            return NULL;
        }
    }

    fclose(fp);
    return buf;
}

static void *load_local_matrix_block(const char *filename, int block_row, int block_col,
                                     int Mb, int Nb, int *M, int *N, char *dtype,
                                     int *local_rows, int *local_cols)
{
    FILE *fp;
    void *buf;
    size_t elem_size;
    int row_start, col_start, i;

    if (!parse_matrix_filename(filename, M, N, dtype)) return NULL;
    elem_size = dtype_size(dtype);
    if (!elem_size) return NULL;
    if (block_row < 0 || block_col < 0) return NULL;
    if (Mb <= 0 || Nb <= 0) return NULL;

    row_start = block_row * Mb;
    col_start = block_col * Nb;
    if (row_start >= *M || col_start >= *N) return NULL;

    *local_rows = (*M - row_start < Mb) ? (*M - row_start) : Mb;
    *local_cols = (*N - col_start < Nb) ? (*N - col_start) : Nb;

    buf = malloc((size_t)(*local_rows) * (size_t)(*local_cols) * elem_size);
    if (!buf) {
        free(buf);
        return NULL;
    }

    fp = fopen(filename, "rb");
    if (!fp) {
        free(buf);
        return NULL;
    }

    for (i = 0; i < *local_rows; ++i) {
        off_t off = (off_t)(((size_t)(row_start + i) * (size_t)(*N) + (size_t)col_start) * elem_size);
        if (fseeko(fp, off, SEEK_SET) != 0 ||
            fread((char *)buf + (size_t)i * (size_t)(*local_cols) * elem_size,
                  elem_size, (size_t)(*local_cols), fp) != (size_t)(*local_cols)) {
            fclose(fp);
            free(buf);
            return NULL;
        }
    }

    fclose(fp);
    return buf;
}

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
