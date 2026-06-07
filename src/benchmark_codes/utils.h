#ifndef LAAB_LOG_H
#define LAAB_LOG_H

#include <stdio.h>     // fprintf, stderr, FILE, fopen, perror, snprintf
#include <stdlib.h>    // getenv, EXIT_FAILURE
#include <unistd.h>    // gethostname
#include <limits.h>    // HOST_NAME_MAX, PATH_MAX
#include <sys/stat.h>  // struct stat, stat, mkdir

static inline FILE *laab_open_log_file(void)
{
    // Get environment variable for log directory
    char *log_dir = getenv("LAAB_LOG_DIR");
    if (!log_dir) {
        fprintf(stderr, "Environment variable LAAB_LOG_DIR not set.\n");
        return NULL;
    }

    // Ensure LAAB_LOG_DIR exists
    struct stat st = {0};
    if (stat(log_dir, &st) == -1) {
        if (mkdir(log_dir, 0755) != 0) {
            perror("Error creating LAAB_LOG_DIR");
            return NULL;
        }
    }

    // Create log file in append mode
    char hostname[HOST_NAME_MAX];

    if (gethostname(hostname, sizeof(hostname)) != 0) {
        perror("gethostname failed");
        return NULL;
    }

    hostname[sizeof(hostname) - 1] = '\0';

    char log_path[PATH_MAX];

    snprintf(
        log_path,
        sizeof(log_path),
        "%s/traces.%s.log",
        log_dir,
        hostname
    );

    FILE *log_file = fopen(log_path, "a");
    if (!log_file) {
        perror("Failed to open log file");
        return NULL;
    }

    return log_file;
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