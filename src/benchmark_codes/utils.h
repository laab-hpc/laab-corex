#ifndef LAAB_LOG_H
#define LAAB_LOG_H

#include <stdio.h>     // fprintf, stderr, FILE, fopen, perror, snprintf
#include <stdlib.h>    // getenv, EXIT_FAILURE
#include <unistd.h>    // gethostname
#include <limits.h>    // HOST_NAME_MAX, PATH_MAX
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
        if (mkdir(trace_dir, 0755) != 0) {
            perror("Error creating LAAB_TRACE_DIR");
            return NULL;
        }
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

#define BILLION 1000000000L
#define SCRUB_SIZE (50 * 1024 * 1024)

static void cache_scrub(void)
{
    double *scrub = (double *)malloc((size_t)SCRUB_SIZE * sizeof(double));
    if (!scrub) return;

    for (size_t i = 0; i < (size_t)SCRUB_SIZE; ++i) scrub[i] = 0.0;
    free(scrub);
}

#endif