#ifndef LAAB_CBLAS_UTILS_H
#define LAAB_CBLAS_UTILS_H

#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

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

static inline void *load_dense_matrix(const char *filename, int rows, int cols, const char *dtype)
{
    FILE *fp = fopen(filename, "rb");
    size_t elem_size = dtype_size(dtype);
    size_t count = (size_t)rows * (size_t)cols;
    void *buf;

    if (!fp || !elem_size) {
        if (fp) fclose(fp);
        return NULL;
    }

    buf = malloc(count * elem_size);
    if (!buf) {
        fclose(fp);
        return NULL;
    }

    if (fread(buf, elem_size, count, fp) != count) {
        fclose(fp);
        free(buf);
        return NULL;
    }

    fclose(fp);
    return buf;
}

#endif
