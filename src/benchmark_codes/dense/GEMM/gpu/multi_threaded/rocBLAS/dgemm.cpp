#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <getopt.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include "utils.h"

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s -m <rows_of_C> -n <cols_of_C> -k <inner_dim> [--reps <count>]\n"
            "\n"
            "Performs DGEMM: C(m x n) = A(m x k) * B(k x n)\n"
            "Options:\n"
            "  -m <int>        Number of rows of C and A\n"
            "  -n <int>        Number of cols of C and B\n"
            "  -k <int>        Inner dimension; cols of A and rows of B\n"
            "  --reps <int>    Number of repetitions (default: 1)\n",
            prog);
}

int main(int argc, char *argv[])
{
    int m = -1;
    int n = -1;
    int k = -1;
    int reps = 1;

    static struct option long_opts[] = {
        {"reps", required_argument, 0, 1},
        {0, 0, 0, 0}
    };

    int opt, long_idx;
    while ((opt = getopt_long(argc, argv, "m:n:k:", long_opts, &long_idx)) != -1) {
        switch (opt) {
            case 'm':
                m = atoi(optarg);
                break;
            case 'n':
                n = atoi(optarg);
                break;
            case 'k':
                k = atoi(optarg);
                break;
            case 1:
                reps = atoi(optarg);
                break;
            default:
                usage(argv[0]);
                return EXIT_FAILURE;
        }
    }

    if (m <= 0 || n <= 0 || k <= 0 || reps <= 0) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    FILE *trace_file = laab_open_trace_file();
    if (!trace_file) return EXIT_FAILURE;

    int device_id = -1;
    hipGetDevice(&device_id);
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, device_id);

    char *visible_dev = getenv("HIP_VISIBLE_DEVICES");
    if (!visible_dev) visible_dev = getenv("ROCR_VISIBLE_DEVICES");
    printf("[INFO] Running on HIP device: %d (VISIBLE_DEVICES=%s)\n",
           device_id, visible_dev ? visible_dev : "not set");

    char pci_bus_id[16];
    snprintf(pci_bus_id, sizeof(pci_bus_id), "%04x:%02x:%02x.0",
             prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);

    double alpha = 1.0;
    double beta = 0.0;
    const double flops = 2.0 * (double)m * (double)n * (double)k;

    fprintf(trace_file, "[LAAB-INFO] rocblas/dgemm | op_sizes=(m=%d, n=%d, k=%d) | flops=%.0f\n",
            m, n, k, flops);
    fflush(trace_file);

    double *A = (double *)malloc((size_t)m * (size_t)k * sizeof(double));
    double *B = (double *)malloc((size_t)k * (size_t)n * sizeof(double));
    double *C = (double *)malloc((size_t)m * (size_t)n * sizeof(double));

    srand48((unsigned)time(NULL));
    for (int i = 0; i < m * k; i++) A[i] = drand48();
    for (int i = 0; i < k * n; i++) B[i] = drand48();

    double *d_A, *d_B, *d_C;
    hipMalloc((void **)&d_A, (size_t)m * (size_t)k * sizeof(double));
    hipMalloc((void **)&d_B, (size_t)k * (size_t)n * sizeof(double));
    hipMalloc((void **)&d_C, (size_t)m * (size_t)n * sizeof(double));

    hipMemcpy(d_A, A, (size_t)m * (size_t)k * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_B, B, (size_t)k * (size_t)n * sizeof(double), hipMemcpyHostToDevice);

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    for (int it = 0; it < reps; it++) {
        cache_scrub();

        hipMemset(d_C, 0, (size_t)m * (size_t)n * sizeof(double));

        hipEvent_t start, stop;
        hipEventCreate(&start);
        hipEventCreate(&stop);

        hipEventRecord(start, 0);

        rocblas_dgemm(handle,
                      rocblas_operation_none, rocblas_operation_none,
                      n, m, k,
                      &alpha,
                      d_B, n,
                      d_A, k,
                      &beta,
                      d_C, n);

        hipEventRecord(stop, 0);
        hipEventSynchronize(stop);

        float milliseconds = 0.0f;
        hipEventElapsedTime(&milliseconds, start, stop);

        float seconds = milliseconds / 1000.0f;
        double gflops = (2.0 * m * n * (double)k) / 1e9;
        double gflops_per_sec = gflops / seconds;

        time_t now = time(NULL);
        struct tm *tm_info = localtime(&now);
        char datetime[32];
        strftime(datetime, sizeof(datetime), "%Y-%m-%d %H:%M:%S", tm_info);

        fprintf(trace_file, "[LAAB] rocblas/dgemm | rep=%d | dt=%s | dur=%.3f s | perf=%.2f GFLOP/s | GPU=%s | BUS=%s\n",
                it, datetime, seconds, gflops_per_sec, visible_dev, pci_bus_id);
        fflush(trace_file);

        hipEventDestroy(start);
        hipEventDestroy(stop);
    }

    rocblas_destroy_handle(handle);

    free(A);
    free(B);
    free(C);
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    fclose(trace_file);

    return 0;
}
