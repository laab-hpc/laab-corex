#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include "utils.h"

static int check_dgemm(void)
{
    const int m = 2;
    const int k = 3;
    const int n = 2;

    const double alpha = 1.0;
    const double beta = 0.0;
    const double tol = 1e-12;

    double A[6] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    };

    double B[6] = {
        7.0, 8.0,
        9.0, 10.0,
        11.0, 12.0
    };

    double C[4] = {0.0, 0.0, 0.0, 0.0};
    double expected[4] = {
        58.0, 64.0,
        139.0, 154.0
    };

    double *d_A = NULL, *d_B = NULL, *d_C = NULL;
    rocblas_handle handle;

    hipMalloc((void **)&d_A, sizeof(A));
    hipMalloc((void **)&d_B, sizeof(B));
    hipMalloc((void **)&d_C, sizeof(C));

    hipMemcpy(d_A, A, sizeof(A), hipMemcpyHostToDevice);
    hipMemcpy(d_B, B, sizeof(B), hipMemcpyHostToDevice);
    hipMemset(d_C, 0, sizeof(C));

    rocblas_create_handle(&handle);
    rocblas_dgemm(handle,
                  rocblas_operation_none, rocblas_operation_none,
                  n, m, k,
                  &alpha,
                  d_B, n,
                  d_A, k,
                  &beta,
                  d_C, n);
    hipMemcpy(C, d_C, sizeof(C), hipMemcpyDeviceToHost);

    rocblas_destroy_handle(handle);
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    int ok = 1;
    for (int i = 0; i < m * n; ++i) {
        if (fabs(C[i] - expected[i]) > tol) {
            ok = 0;
            break;
        }
    }

    FILE *trace_file = laab_open_trace_file();
    if (!trace_file) return EXIT_FAILURE;

    if (ok) {
        fprintf(trace_file, "[LAAB-INFO] rocblas/dgemm | correctness=PASS\n");
        fclose(trace_file);
        return 1;
    }

    fprintf(trace_file, "[LAAB-INFO] rocblas/dgemm | correctness=FAIL\n");
    fclose(trace_file);
    printf("Correctness check failed. Expected:\n");
    printf("[%.1f %.1f]\n", expected[0], expected[1]);
    printf("[%.1f %.1f]\n", expected[2], expected[3]);
    printf("Got:\n");
    printf("[%.12f %.12f]\n", C[0], C[1]);
    printf("[%.12f %.12f]\n", C[2], C[3]);
    return 0;
}

static int check_sgemm(void)
{
    const int m = 2;
    const int k = 3;
    const int n = 2;

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const float tol = 1e-5f;

    float A[6] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    float B[6] = {
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
    };

    float C[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float expected[4] = {
        58.0f, 64.0f,
        139.0f, 154.0f
    };

    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    rocblas_handle handle;

    hipMalloc((void **)&d_A, sizeof(A));
    hipMalloc((void **)&d_B, sizeof(B));
    hipMalloc((void **)&d_C, sizeof(C));

    hipMemcpy(d_A, A, sizeof(A), hipMemcpyHostToDevice);
    hipMemcpy(d_B, B, sizeof(B), hipMemcpyHostToDevice);
    hipMemset(d_C, 0, sizeof(C));

    rocblas_create_handle(&handle);
    rocblas_sgemm(handle,
                  rocblas_operation_none, rocblas_operation_none,
                  n, m, k,
                  &alpha,
                  d_B, n,
                  d_A, k,
                  &beta,
                  d_C, n);
    hipMemcpy(C, d_C, sizeof(C), hipMemcpyDeviceToHost);

    rocblas_destroy_handle(handle);
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    int ok = 1;
    for (int i = 0; i < m * n; ++i) {
        if (fabsf(C[i] - expected[i]) > tol) {
            ok = 0;
            break;
        }
    }

    FILE *trace_file = laab_open_trace_file();
    if (!trace_file) return EXIT_FAILURE;

    if (ok) {
        fprintf(trace_file, "[LAAB-INFO] rocblas/sgemm | correctness=PASS\n");
        fclose(trace_file);
        return 1;
    }

    fprintf(trace_file, "[LAAB-INFO] rocblas/sgemm | correctness=FAIL\n");
    fclose(trace_file);
    printf("Correctness check failed. Expected:\n");
    printf("[%.1f %.1f]\n", expected[0], expected[1]);
    printf("[%.1f %.1f]\n", expected[2], expected[3]);
    printf("Got:\n");
    printf("[%.6f %.6f]\n", C[0], C[1]);
    printf("[%.6f %.6f]\n", C[2], C[3]);
    return 0;
}

int main(void)
{
    int d_ok = check_dgemm();
    int s_ok = check_sgemm();
    return (d_ok && s_ok) ? 0 : 1;
}
