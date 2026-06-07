#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <getopt.h>
#include <limits.h>    // for HOST_NAME_MAX
#include <cuda_runtime.h>
#include <cublas_v2.h>
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

int main(int argc, char* argv[]) {
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
    cudaGetDevice(&device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    char* visible_dev = getenv("CUDA_VISIBLE_DEVICES");
    //printf("[INFO] Running on CUDA device: %d\n", device_id);
    printf("[INFO] Running on CUDA device: %d (CUDA_VISIBLE_DEVICES=%s)\n", device_id, visible_dev ? visible_dev : "not set");

    // Format PCI Bus ID string (e.g., "0000:81:00.0")
    char pci_bus_id[16];
    snprintf(pci_bus_id, sizeof(pci_bus_id), "%04x:%02x:%02x.0",
            prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);

    double alpha = 1.0;
    double beta = 0.0;
    const double flops = 2.0 * (double)m * (double)n * (double)k;

    fprintf(trace_file, "[LAAB-INFO] cublas/dgemm | op_sizes=(m=%d, n=%d, k=%d) | flops=%.0f\n",
           m, n, k, flops);
    fflush(trace_file);

    double *A = (double*)malloc(m * k * sizeof(double));
    double *B = (double*)malloc(k * n * sizeof(double));
    double *C = (double*)malloc(m * n * sizeof(double));

    srand48((unsigned)time(NULL));
    for (int i = 0; i < m * k; i++) A[i] = drand48();
    for (int i = 0; i < k * n; i++) B[i] = drand48();

    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, m * k * sizeof(double));
    cudaMalloc((void**)&d_B, k * n * sizeof(double));
    cudaMalloc((void**)&d_C, m * n * sizeof(double));

    cudaMemcpy(d_A, A, m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, k * n * sizeof(double), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    for (int it = 0; it < reps; it++) {
        cache_scrub();

        cudaMemset(d_C, 0, m * n * sizeof(double));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);

        // cuBLAS uses column-major → swap m and n
        cublasDgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n, m, k,
                    &alpha,
                    d_B, n,
                    d_A, k,
                    &beta,
                    d_C, n);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        float seconds = milliseconds / 1000.0f;
        double gflops = (2.0 * m * n * (double)k) / 1e9;
        double gflops_per_sec = gflops / seconds;

        time_t now = time(NULL);
        struct tm *tm_info = localtime(&now);
        char datetime[32];
        strftime(datetime, sizeof(datetime), "%Y-%m-%d %H:%M:%S", tm_info);

        fprintf(trace_file, "[LAAB] cublas/dgemm | rep=%d | dt=%s | dur=%.3f s | perf=%.2f GFLOP/s | GPU=%s | BUS=%s\n",
            it, datetime, seconds, gflops_per_sec, visible_dev, pci_bus_id);
        
        fflush(trace_file);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    cublasDestroy(handle);

    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    fclose(trace_file);

    return 0;
}


