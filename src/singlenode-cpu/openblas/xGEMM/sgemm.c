#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <getopt.h>
#include <sched.h>
#include <cblas.h>

#define BILLION 1000000000L
#define SCRUB_SIZE (50 * 1024 * 1024)

static void cache_scrub(void)
{
    float *scrub = (float *)malloc((size_t)SCRUB_SIZE * sizeof(float));
    if (!scrub) return;

    for (size_t i = 0; i < (size_t)SCRUB_SIZE; ++i) scrub[i] = 0.0f;
    free(scrub);
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s -m <rows_of_C> -n <cols_of_C> [--reps <count>]\n"
            "\n"
            "Performs SGEMM: C(m x n) = A(m x n) * B(n x n)\n"
            "Options:\n"
            "  -m <int>        Number of rows (m)\n"
            "  -n <int>        Number of cols (n); also used as inner dim k\n"
            "  --reps <int>    Number of repetitions (default: 1)\n",
            prog);
}

int main(int argc, char **argv)
{
    int m = -1;
    int n = -1;
    int reps = 1;

    static struct option long_opts[] = {
        {"reps", required_argument, 0, 1},
        {0, 0, 0, 0}
    };

    int opt, long_idx;
    while ((opt = getopt_long(argc, argv, "m:n:", long_opts, &long_idx)) != -1) {
        switch (opt) {
            case 'm':
                m = atoi(optarg);
                break;
            case 'n':
                n = atoi(optarg);
                break;
            case 1:
                reps = atoi(optarg);
                break;
            default:
                usage(argv[0]);
                return EXIT_FAILURE;
        }
    }

    if (m <= 0 || n <= 0 || reps <= 0) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    const int k = n;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const double flops = 2.0 * (double)m * (double)n * (double)k;
    const char *omp_env = getenv("OMP_NUM_THREADS");
    int nthreads = omp_env ? atoi(omp_env) : 1;
    if (nthreads <= 0) nthreads = 1;

    size_t szA = (size_t)m * (size_t)k;
    size_t szB = (size_t)k * (size_t)n;
    size_t szC = (size_t)m * (size_t)n;

    float *A = (float *)malloc(szA * sizeof(float));
    float *B = (float *)malloc(szB * sizeof(float));
    float *C = (float *)malloc(szC * sizeof(float));

    if (!A || !B || !C) {
        fprintf(stderr, "Allocation failed for matrices (m=%d, n=%d, k=%d).\n", m, n, k);
        free(A);
        free(B);
        free(C);
        return EXIT_FAILURE;
    }

    srand48((long)time(NULL));
    for (size_t i = 0; i < szA; ++i) A[i] = (float)drand48();
    for (size_t i = 0; i < szB; ++i) B[i] = (float)drand48();

    printf("[LAAB-INFO] openblas/sgemm | op_sizes=(m=%d, n=%d) | nthreads=%d | flops=%.0f\n",
           m, n, nthreads, flops);
    fflush(stdout);

    for (size_t i = 0; i < szC; ++i) C[i] = 0.0f;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, A, k, B, n, beta, C, n);

    for (int r = 0; r < reps; ++r) {
        for (size_t i = 0; i < szC; ++i) C[i] = 0.0f;

        cache_scrub();

        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, alpha, A, k, B, n, beta, C, n);

        clock_gettime(CLOCK_MONOTONIC, &end);

        double elapsed = (double)(end.tv_sec - start.tv_sec) +
                         (double)(end.tv_nsec - start.tv_nsec) / (double)BILLION;
        double gflops = flops / elapsed / 1e9;

        time_t now = time(NULL);
        struct tm *tm_info = localtime(&now);
        char datetime[32];
        strftime(datetime, sizeof(datetime), "%Y-%m-%d %H:%M:%S", tm_info);

        int cpu = sched_getcpu();
        printf("[LAAB] openblas/sgemm | rep=%d | dt=%s | dur=%.5f s | perf=%.5f GFLOP/s | lcpu=%d \n",
               r, datetime, elapsed, gflops, cpu);
        fflush(stdout);
    }

    free(A);
    free(B);
    free(C);
    return EXIT_SUCCESS;
}
