#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <getopt.h>
#include <sched.h>
#include <string.h>
#include <complex.h>
#ifdef USE_MKL
    #include <mkl.h>
#else
    #include <cblas.h>
#endif
#include "utils.h"

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s -A <matrix_A_file> -B <matrix_B_file> [--reps <count>] [--tag <tag>]\n"
            "\n"
            "Performs GEMM: C = A * B using matrix files named like\n"
            "M100x200-float64-...dense\n"
            "Supported dtypes: float32, float64, complex128\n"
            "Options:\n"
            "  -A <path>       Path to matrix A file\n"
            "  -B <path>       Path to matrix B file\n"
            "  --reps <int>    Number of repetitions (default: 1)\n"
            "  --tag <str>     Trace filename tag (default: 0)\n",
            prog);
}

static const char *kernel_name(const char *dtype)
{
    if (strcmp(dtype, "float32") == 0) return "sgemm";
    if (strcmp(dtype, "float64") == 0) return "dgemm";
    if (strcmp(dtype, "complex128") == 0) return "zgemm";
    return "gemm";
}

static double gemm_flops(int m, int n, int k, const char *dtype)
{
    double factor = 0.0;

    if (strcmp(dtype, "float32") == 0 || strcmp(dtype, "float64") == 0) {
        factor = 2.0;
    } else if (strcmp(dtype, "complex128") == 0) {
        factor = 8.0;
    }

    return factor * (double)m * (double)n * (double)k;
}

static void zero_matrix(void *buf, size_t count, const char *dtype)
{
    if (strcmp(dtype, "float32") == 0) {
        float *ptr = (float *)buf;
        for (size_t i = 0; i < count; ++i) ptr[i] = 0.0f;
        return;
    }

    if (strcmp(dtype, "float64") == 0) {
        double *ptr = (double *)buf;
        for (size_t i = 0; i < count; ++i) ptr[i] = 0.0;
        return;
    }

    if (strcmp(dtype, "complex128") == 0) {
        double complex *ptr = (double complex *)buf;
        for (size_t i = 0; i < count; ++i) ptr[i] = 0.0 + 0.0 * I;
    }
}

static double matrix_l2(const void *buf, size_t count, const char *dtype)
{
    if (strcmp(dtype, "float32") == 0) {
        return (double)cblas_snrm2((int)count, (const float *)buf, 1);
    }

    if (strcmp(dtype, "float64") == 0) {
        return cblas_dnrm2((int)count, (const double *)buf, 1);
    }

    if (strcmp(dtype, "complex128") == 0) {
        return cblas_dznrm2((int)count, buf, 1);
    }

    return 0.0;
}

static void gemm_once(int m, int n, int k, const void *A, const void *B, void *C, const char *dtype)
{
    if (strcmp(dtype, "float32") == 0) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, alpha, A, k, B, n, beta, C, n);
        return;
    }

    if (strcmp(dtype, "float64") == 0) {
        const double alpha = 1.0;
        const double beta = 0.0;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, alpha, A, k, B, n, beta, C, n);
        return;
    }

    if (strcmp(dtype, "complex128") == 0) {
        const double complex alpha = 1.0 + 0.0 * I;
        const double complex beta = 0.0 + 0.0 * I;
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, &alpha, A, k, B, n, &beta, C, n);
    }
}

int main(int argc, char **argv)
{
    const char *afile = NULL;
    const char *bfile = NULL;
    const char *tag = "0";
    int reps = 1;

    static struct option long_opts[] = {
        {"reps", required_argument, 0, 1},
        {"tag", required_argument, 0, 2},
        {0, 0, 0, 0}
    };

    int opt, long_idx;
    while ((opt = getopt_long(argc, argv, "A:B:", long_opts, &long_idx)) != -1) {
        switch (opt) {
            case 'A': afile = optarg; break;
            case 'B': bfile = optarg; break;
            case 1: reps = atoi(optarg); break;
            case 2: tag = optarg; break;
            default:
                usage(argv[0]);
                return EXIT_FAILURE;
        }
    }

    if (!afile || !bfile || reps <= 0) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    FILE *trace_file = laab_open_trace_file_with_tag(tag);
    if (!trace_file) return EXIT_FAILURE;

    int m = 0, k_a = 0, k_b = 0, n = 0;
    int k;
    int np;
    size_t elem_size;
    size_t szC;
    double flops;
    double c_l2;
    void *A;
    void *B;
    void *C;
    char dtype_a[32], dtype_b[32];
    char datetime[32];
    char hostname[HOST_NAME_MAX];
    const char *omp_env;
    const char *kernel;
    int cpu;

    if (!parse_matrix_filename(afile, &m, &k_a, dtype_a) ||
        !parse_matrix_filename(bfile, &k_b, &n, dtype_b)) {
        fprintf(stderr, "failed to parse matrix metadata from filenames\n");
        fclose(trace_file);
        return EXIT_FAILURE;
    }

    if (k_a != k_b || strcmp(dtype_a, dtype_b) != 0) {
        fprintf(stderr, "matrix inputs must satisfy A.cols == B.rows and A.dtype == B.dtype\n");
        fclose(trace_file);
        return EXIT_FAILURE;
    }

    if (strcmp(dtype_a, "float32") != 0 &&
        strcmp(dtype_a, "float64") != 0 &&
        strcmp(dtype_a, "complex128") != 0) {
        fprintf(stderr, "unsupported dtype %s; supported dtypes are float32, float64, complex128\n", dtype_a);
        fclose(trace_file);
        return EXIT_FAILURE;
    }

    k = k_a;
    elem_size = dtype_size(dtype_a);
    szC = (size_t)m * (size_t)n;
    flops = gemm_flops(m, n, k, dtype_a);
    kernel = kernel_name(dtype_a);

    omp_env = getenv("OMP_NUM_THREADS");
    np = omp_env ? atoi(omp_env) : 1;
    if (np <= 0) np = 1;

    A = load_dense_matrix(afile, m, k, dtype_a);
    if (!A) {
        fprintf(stderr, "Failed to load matrix A from %s\n", afile);
        fclose(trace_file);
        return EXIT_FAILURE;
    }

    B = load_dense_matrix(bfile, k, n, dtype_b);
    if (!B) {
        fprintf(stderr, "Failed to load matrix B from %s\n", bfile);
        free(A);
        fclose(trace_file);
        return EXIT_FAILURE;
    }

    C = malloc(szC * elem_size);
    if (!C) {
        fprintf(stderr, "Allocation failed for matrices (m=%d, n=%d, k=%d).\n", m, n, k);
        free(A);
        free(B);
        fclose(trace_file);
        return EXIT_FAILURE;
    }

    zero_matrix(C, szC, dtype_a);
    gemm_once(m, n, k, A, B, C, dtype_a);
    c_l2 = matrix_l2(C, szC, dtype_a);

    {
        time_t now = time(NULL);
        struct tm *tm_info = localtime(&now);
        strftime(datetime, sizeof(datetime), "%Y-%m-%d %H:%M:%S", tm_info);
    }

    if (gethostname(hostname, sizeof(hostname)) != 0) {
        perror("gethostname failed");
        free(A);
        free(B);
        free(C);
        fclose(trace_file);
        return EXIT_FAILURE;
    }
    hostname[sizeof(hostname) - 1] = '\0';
    cpu = sched_getcpu();

    fprintf(trace_file,
            "[LAAB-STEP] cblas/%s | prob_size=A=%dx%d+B=%dx%d | prec=\"%s\" | flops=%.0f | interface=cblas\n",
            kernel, m, k, k, n, dtype_a, flops);
    fflush(trace_file);

    fprintf(trace_file, "[LAAB-RUN] cblas/%s | ts=%s | nthreads=%d | execution=OMP | l2_norm=%.12e\n", kernel, datetime, np, c_l2);
    fflush(trace_file);

    fprintf(trace_file, "[LAAB-HOST] cblas/%s | hostname=%s | core_id=%d\n", kernel, hostname, cpu);
    fflush(trace_file);

    for (int r = 0; r < reps; ++r) {
        zero_matrix(C, szC, dtype_a);

        cache_scrub();

        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        gemm_once(m, n, k, A, B, C, dtype_a);

        clock_gettime(CLOCK_MONOTONIC, &end);

        double elapsed = (double)(end.tv_sec - start.tv_sec) +
                         (double)(end.tv_nsec - start.tv_nsec) / 1000000000.0;
        double gflops = flops / elapsed / 1e9;

        {
            time_t now = time(NULL);
            struct tm *tm_info = localtime(&now);
            strftime(datetime, sizeof(datetime), "%Y-%m-%d %H:%M:%S", tm_info);
        }

        fprintf(trace_file, "[LAAB] cblas/%s | rep=%d | ts=%s | exec_time (s)= %.5f | perf (GFLOP/s)= %.5f\n",
                kernel, r, datetime, elapsed, gflops);
        fflush(trace_file);
    }

    free(A);
    free(B);
    free(C);
    fclose(trace_file);

    return EXIT_SUCCESS;
}
