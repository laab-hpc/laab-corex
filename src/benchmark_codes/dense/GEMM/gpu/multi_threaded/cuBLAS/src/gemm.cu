#define _GNU_SOURCE
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "utils.h"

#define CUDA_CHECK(expr) do { \
    cudaError_t err__ = (expr); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        return EXIT_FAILURE; \
    } \
} while (0)

#define CUBLAS_CHECK(expr) do { \
    cublasStatus_t err__ = (expr); \
    if (err__ != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: status=%d\n", __FILE__, __LINE__, (int)err__); \
        return EXIT_FAILURE; \
    } \
} while (0)

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

static double matrix_l2(const void *buf, size_t count, const char *dtype)
{
    size_t i;
    double sum = 0.0;

    if (strcmp(dtype, "float32") == 0) {
        const float *ptr = (const float *)buf;
        for (i = 0; i < count; ++i) sum += (double)ptr[i] * (double)ptr[i];
        return sqrt(sum);
    }

    if (strcmp(dtype, "float64") == 0) {
        const double *ptr = (const double *)buf;
        for (i = 0; i < count; ++i) sum += ptr[i] * ptr[i];
        return sqrt(sum);
    }

    if (strcmp(dtype, "complex128") == 0) {
        const cuDoubleComplex *ptr = (const cuDoubleComplex *)buf;
        for (i = 0; i < count; ++i) {
            double real = cuCreal(ptr[i]);
            double imag = cuCimag(ptr[i]);
            sum += real * real + imag * imag;
        }
        return sqrt(sum);
    }

    return 0.0;
}

static int gemm_once(cublasHandle_t handle,
                     cudaStream_t stream,
                     int m,
                     int n,
                     int k,
                     const void *d_A,
                     const void *d_B,
                     void *d_C,
                     size_t bytes_c,
                     const char *dtype,
                     double *elapsed_seconds)
{
    cudaEvent_t start = NULL;
    cudaEvent_t stop = NULL;
    float elapsed_ms = 0.0f;

    CUDA_CHECK(cudaMemsetAsync(d_C, 0, bytes_c, stream));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));

    if (strcmp(dtype, "float32") == 0) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 n, m, k,
                                 &alpha,
                                 (const float *)d_B, n,
                                 (const float *)d_A, k,
                                 &beta,
                                 (float *)d_C, n));
    } else if (strcmp(dtype, "float64") == 0) {
        const double alpha = 1.0;
        const double beta = 0.0;
        CUBLAS_CHECK(cublasDgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 n, m, k,
                                 &alpha,
                                 (const double *)d_B, n,
                                 (const double *)d_A, k,
                                 &beta,
                                 (double *)d_C, n));
    } else if (strcmp(dtype, "complex128") == 0) {
        const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
        CUBLAS_CHECK(cublasZgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 n, m, k,
                                 &alpha,
                                 (const cuDoubleComplex *)d_B, n,
                                 (const cuDoubleComplex *)d_A, k,
                                 &beta,
                                 (cuDoubleComplex *)d_C, n));
    } else {
        fprintf(stderr, "unsupported dtype %s\n", dtype);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    if (elapsed_seconds) *elapsed_seconds = (double)elapsed_ms / 1000.0;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return EXIT_SUCCESS;
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

    FILE *trace_file = NULL;
    void *A = NULL;
    void *B = NULL;
    void *C = NULL;
    void *d_A = NULL;
    void *d_B = NULL;
    void *d_C = NULL;
    cublasHandle_t handle = NULL;
    cudaStream_t stream = NULL;

    int m = 0;
    int n = 0;
    int k = 0;
    int k_a = 0;
    int k_b = 0;
    int np = 1;
    int cpu = -1;
    int device_id = -1;
    int opt;
    int long_idx;
    size_t elem_size;
    size_t szA;
    size_t szB;
    size_t szC;
    size_t bytesA;
    size_t bytesB;
    size_t bytesC;
    double flops;
    double c_l2;
    char dtype_a[32];
    char dtype_b[32];
    char datetime[32];
    char hostname[HOST_NAME_MAX];
    const char *kernel;
    const char *cuda_visible;
    cudaDeviceProp prop;
    char pci_bus_id[16];

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

    trace_file = laab_open_trace_file_with_tag(tag);
    if (!trace_file) return EXIT_FAILURE;

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
    szA = (size_t)m * (size_t)k;
    szB = (size_t)k * (size_t)n;
    szC = (size_t)m * (size_t)n;
    bytesA = szA * elem_size;
    bytesB = szB * elem_size;
    bytesC = szC * elem_size;
    flops = gemm_flops(m, n, k, dtype_a);
    kernel = kernel_name(dtype_a);

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

    C = malloc(bytesC);
    if (!C) {
        fprintf(stderr, "Allocation failed for output matrix C (%s)\n", strerror(errno));
        free(A);
        free(B);
        fclose(trace_file);
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaGetDevice(&device_id));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));
    CUDA_CHECK(cudaMemcpyAsync(d_A, A, bytesA, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B, bytesB, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (gemm_once(handle, stream, m, n, k, d_A, d_B, d_C, bytesC, dtype_a, NULL) != EXIT_SUCCESS) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);
        cudaStreamDestroy(stream);
        free(A);
        free(B);
        free(C);
        fclose(trace_file);
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaMemcpy(C, d_C, bytesC, cudaMemcpyDeviceToHost));
    c_l2 = matrix_l2(C, szC, dtype_a);

    {
        time_t now = time(NULL);
        struct tm *tm_info = localtime(&now);
        strftime(datetime, sizeof(datetime), "%Y-%m-%d %H:%M:%S", tm_info);
    }

    if (gethostname(hostname, sizeof(hostname)) != 0) {
        perror("gethostname failed");
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);
        cudaStreamDestroy(stream);
        free(A);
        free(B);
        free(C);
        fclose(trace_file);
        return EXIT_FAILURE;
    }

    hostname[sizeof(hostname) - 1] = '\0';
    cpu = sched_getcpu();
    cuda_visible = getenv("CUDA_VISIBLE_DEVICES");
    snprintf(pci_bus_id, sizeof(pci_bus_id), "%04x:%02x:%02x.0",
             prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);

    fprintf(trace_file,
            "[LAAB-STEP] cublas/%s | prob_size=A=%dx%d+B=%dx%d | prec=\"%s\" | flops=%.0f | interface=cublas\n",
            kernel, m, k, k, n, dtype_a, flops);
    fflush(trace_file);

    fprintf(trace_file,
            "[LAAB-RUN] cublas/%s | ts=%s | nthreads=%d | execution=CUDA | l2_norm=%.12e\n",
            kernel, datetime, np, c_l2);
    fflush(trace_file);

    fprintf(trace_file,
            "[LAAB-HOST] cublas/%s | hostname=%s | core_id=%d | gpu_id=%d | gpu_bus=%s | visible_devices=%s\n",
            kernel, hostname, cpu, device_id, pci_bus_id, cuda_visible ? cuda_visible : "not set");
    fflush(trace_file);

    for (int r = 0; r < reps; ++r) {
        double elapsed = 0.0;
        double gflops = 0.0;

        cache_scrub();

        if (gemm_once(handle, stream, m, n, k, d_A, d_B, d_C, bytesC, dtype_a, &elapsed) != EXIT_SUCCESS) {
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            cublasDestroy(handle);
            cudaStreamDestroy(stream);
            free(A);
            free(B);
            free(C);
            fclose(trace_file);
            return EXIT_FAILURE;
        }

        gflops = flops / elapsed / 1e9;

        {
            time_t now = time(NULL);
            struct tm *tm_info = localtime(&now);
            strftime(datetime, sizeof(datetime), "%Y-%m-%d %H:%M:%S", tm_info);
        }

        fprintf(trace_file,
                "[LAAB] cublas/%s | rep=%d | ts=%s | exec_time (s)= %.5f | perf (GFLOP/s)= %.5f\n",
                kernel, r, datetime, elapsed, gflops);
        fflush(trace_file);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    free(A);
    free(B);
    free(C);
    fclose(trace_file);

    return EXIT_SUCCESS;
}
