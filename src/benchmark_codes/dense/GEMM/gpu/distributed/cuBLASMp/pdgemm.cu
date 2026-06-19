#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <getopt.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasmp.h>
#include <nccl.h>

extern "C" {
#include "utils.h"
}

#define DIE(...) do { \
    fprintf(stderr, __VA_ARGS__); \
    fprintf(stderr, "\n"); \
    if (mpi_ok) MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); \
    exit(EXIT_FAILURE); \
} while (0)

#define CUDA_CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) DIE("CUDA %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(e)); \
} while (0)

#define CUBLAS_CHECK(x) do { \
    cublasStatus_t e = (x); \
    if (e != CUBLAS_STATUS_SUCCESS) DIE("cuBLAS %s:%d: status=%d", __FILE__, __LINE__, (int)e); \
} while (0)

#define CUBLASMP_CHECK(x) do { \
    cublasMpStatus_t e = (x); \
    if (e != CUBLASMP_STATUS_SUCCESS) DIE("cuBLASMp %s:%d: status=%d", __FILE__, __LINE__, (int)e); \
} while (0)

#define NCCL_CHECK(x) do { \
    ncclResult_t e = (x); \
    if (e != ncclSuccess) DIE("NCCL %s:%d: %s", __FILE__, __LINE__, ncclGetErrorString(e)); \
} while (0)

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s -A <matrix_A_file> -B <matrix_B_file> -b <block_size> [--reps <count>]\n",
            prog);
}

static int get_local_rank(MPI_Comm comm)
{
    MPI_Comm local_comm;
    int local_rank = 0;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);
    return local_rank;
}

int main(int argc, char **argv)
{
    int mpi_ok = 0;

    const char *afile = NULL;
    const char *bfile = NULL;
    int nb = -1;
    int reps = 1;

    int world_rank = 0;
    int world_size = 1;
    int nprow = 1;
    int npcol = 1;
    int myrow = 0;
    int mycol = 0;

    int local_rank = 0;
    int device_count = 0;
    int device_id = 0;

    int m = 0;
    int n = 0;
    int k = 0;
    int k_a = 0;
    int k_b = 0;
    int isrcproc = 0;

    char dtype_a[32];
    char dtype_b[32];
    char datetime[32];

    int64_t one = 1;
    int64_t mloc_a = 0;
    int64_t nloc_a = 0;
    int64_t mloc_b = 0;
    int64_t nloc_b = 0;
    int64_t mloc_c = 0;
    int64_t nloc_c = 0;
    int64_t lld_a = 1;
    int64_t lld_b = 1;
    int64_t lld_c = 1;

    size_t elems_a = 0;
    size_t elems_b = 0;
    size_t elems_c = 0;
    size_t bytes_a = 0;
    size_t bytes_b = 0;
    size_t bytes_c = 0;
    size_t d_work_size = 0;
    size_t h_work_size = 0;

    double alpha = 1.0;
    double beta = 0.0;
    double flops = 0.0;

    FILE *trace_file = NULL;
    double *A = NULL;
    double *B = NULL;
    double *dA = NULL;
    double *dB = NULL;
    double *dC = NULL;
    void *d_work = NULL;
    void *h_work = NULL;

    cudaStream_t stream = NULL;
    cublasHandle_t blas = NULL;
    cublasMpHandle_t mp = NULL;
    cublasMpGrid_t grid = NULL;
    cublasMpMatrixDescriptor_t descA = NULL;
    cublasMpMatrixDescriptor_t descB = NULL;
    cublasMpMatrixDescriptor_t descC = NULL;

    ncclUniqueId nccl_id;
    ncclComm_t nccl_comm = NULL;

    static struct option long_opts[] = {
        {"reps", required_argument, 0, 1},
        {0, 0, 0, 0}
    };

    int opt;
    int long_idx;

    while ((opt = getopt_long(argc, argv, "A:B:b:", long_opts, &long_idx)) != -1) {
        switch (opt) {
            case 'A': afile = optarg; break;
            case 'B': bfile = optarg; break;
            case 'b': nb = atoi(optarg); break;
            case 1: reps = atoi(optarg); break;
            default:
                usage(argv[0]);
                return EXIT_FAILURE;
        }
    }

    if (!afile || !bfile || nb <= 0 || reps <= 0) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    MPI_Init(&argc, &argv);
    mpi_ok = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    choose_process_grid(world_size, &nprow, &npcol);

    if (nprow * npcol != world_size) {
        DIE("bad process grid: %d x %d != %d", nprow, npcol, world_size);
    }

    myrow = world_rank / npcol;
    mycol = world_rank % npcol;

    local_rank = get_local_rank(MPI_COMM_WORLD);

    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) DIE("rank %d: no CUDA device visible", world_rank);

    device_id = local_rank % device_count;

    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaFree(NULL));
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUBLAS_CHECK(cublasCreate(&blas));
    CUBLAS_CHECK(cublasSetStream(blas, stream));

    if (world_rank == 0) NCCL_CHECK(ncclGetUniqueId(&nccl_id));
    MPI_Bcast(&nccl_id, (int)sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
    NCCL_CHECK(ncclCommInitRank(&nccl_comm, world_size, nccl_id, world_rank));

    CUBLASMP_CHECK(cublasMpCreate(&mp, stream));
    CUBLASMP_CHECK(cublasMpGridCreate(
        (int64_t)nprow,
        (int64_t)npcol,
        CUBLASMP_GRID_LAYOUT_ROW_MAJOR,
        nccl_comm,
        &grid));

    trace_file = laab_open_trace_file();
    if (!trace_file) DIE("rank %d: failed to open trace file", world_rank);

    if (!parse_matrix_filename(afile, &m, &k_a, dtype_a) ||
        !parse_matrix_filename(bfile, &k_b, &n, dtype_b)) {
        DIE("rank %d: failed to parse matrix metadata", world_rank);
    }

    if (k_a != k_b || strcmp(dtype_a, "float64") != 0 || strcmp(dtype_b, "float64") != 0) {
        DIE("matrix inputs must satisfy A.cols == B.rows and both dtypes must be float64");
    }

    k = k_a;
    flops = 2.0 * (double)m * (double)n * (double)k;

    if (world_rank == 0) {
        fprintf(trace_file,
                "[LAAB-INFO] cublasmp/gemm | nranks=%d | grid_dim=(%d,%d) | A=(%d,%d,%s) | B=(%d,%d,%s) | block_size=%dx%d | flops=%.0f\n",
                world_size, nprow, npcol, m, k_a, dtype_a, k_b, n, dtype_b, nb, nb, flops);
        fflush(trace_file);
    }

    mloc_a = cublasMpNumroc((int64_t)m, (int64_t)nb, (uint32_t)myrow, (uint32_t)isrcproc, (uint32_t)nprow);
    nloc_a = cublasMpNumroc((int64_t)k, (int64_t)nb, (uint32_t)mycol, (uint32_t)isrcproc, (uint32_t)npcol);
    mloc_b = cublasMpNumroc((int64_t)k, (int64_t)nb, (uint32_t)myrow, (uint32_t)isrcproc, (uint32_t)nprow);
    nloc_b = cublasMpNumroc((int64_t)n, (int64_t)nb, (uint32_t)mycol, (uint32_t)isrcproc, (uint32_t)npcol);
    mloc_c = cublasMpNumroc((int64_t)m, (int64_t)nb, (uint32_t)myrow, (uint32_t)isrcproc, (uint32_t)nprow);
    nloc_c = cublasMpNumroc((int64_t)n, (int64_t)nb, (uint32_t)mycol, (uint32_t)isrcproc, (uint32_t)npcol);

    lld_a = mloc_a > 1 ? mloc_a : 1;
    lld_b = mloc_b > 1 ? mloc_b : 1;
    lld_c = mloc_c > 1 ? mloc_c : 1;

    elems_a = (size_t)lld_a * (size_t)nloc_a;
    elems_b = (size_t)lld_b * (size_t)nloc_b;
    elems_c = (size_t)lld_c * (size_t)nloc_c;

    bytes_a = elems_a * sizeof(double);
    bytes_b = elems_b * sizeof(double);
    bytes_c = elems_c * sizeof(double);

    CUBLASMP_CHECK(cublasMpMatrixDescriptorCreate(
        (int64_t)m, (int64_t)k, (int64_t)nb, (int64_t)nb,
        0, 0, lld_a, CUDA_R_64F, grid, &descA));

    CUBLASMP_CHECK(cublasMpMatrixDescriptorCreate(
        (int64_t)k, (int64_t)n, (int64_t)nb, (int64_t)nb,
        0, 0, lld_b, CUDA_R_64F, grid, &descB));

    CUBLASMP_CHECK(cublasMpMatrixDescriptorCreate(
        (int64_t)m, (int64_t)n, (int64_t)nb, (int64_t)nb,
        0, 0, lld_c, CUDA_R_64F, grid, &descC));

    MPI_Barrier(MPI_COMM_WORLD);
    double io_start = MPI_Wtime();

    A = (double *)load_block_cyclic_matrix(
        afile, myrow, mycol, nprow, npcol, nb, nb, (int)mloc_a, (int)nloc_a);

    B = (double *)load_block_cyclic_matrix(
        bfile, myrow, mycol, nprow, npcol, nb, nb, (int)mloc_b, (int)nloc_b);

    MPI_Barrier(MPI_COMM_WORLD);
    double io_end = MPI_Wtime();

    if ((bytes_a && !A) || (bytes_b && !B)) {
        DIE("rank %d: matrix load failed", world_rank);
    }

    CUDA_CHECK(cudaMalloc((void **)&dA, bytes_a ? bytes_a : 1));
    CUDA_CHECK(cudaMalloc((void **)&dB, bytes_b ? bytes_b : 1));
    CUDA_CHECK(cudaMalloc((void **)&dC, bytes_c ? bytes_c : 1));

    if (bytes_a) CUDA_CHECK(cudaMemcpyAsync(dA, A, bytes_a, cudaMemcpyHostToDevice, stream));
    if (bytes_b) CUDA_CHECK(cudaMemcpyAsync(dB, B, bytes_b, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(dC, 0, bytes_c ? bytes_c : 1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    double io_elapsed = io_end - io_start;
    double ab_mb = (double)(bytes_a + bytes_b) / (1024.0 * 1024.0);

    fprintf(trace_file,
            "[LAAB-INFO] cublasmp/gemm | rank=%d | gpu=%d | grid_id=(%d,%d) | A_local=(%ld,%ld) | B_local=(%ld,%ld) | C_local=(%ld,%ld) | io_time=%.5f s | ab_size=%.2f MB\n",
            world_rank, device_id, myrow, mycol,
            (long)mloc_a, (long)nloc_a,
            (long)mloc_b, (long)nloc_b,
            (long)mloc_c, (long)nloc_c,
            io_elapsed, ab_mb);
    fflush(trace_file);

    CUBLASMP_CHECK(cublasMpGemm_bufferSize(
        mp,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        (int64_t)m,
        (int64_t)n,
        (int64_t)k,
        &alpha,
        dA,
        one,
        one,
        descA,
        dB,
        one,
        one,
        descB,
        &beta,
        dC,
        one,
        one,
        descC,
        CUBLAS_COMPUTE_64F,
        &d_work_size,
        &h_work_size));

    if (d_work_size) CUDA_CHECK(cudaMalloc(&d_work, d_work_size));
    if (h_work_size) {
        h_work = malloc(h_work_size);
        if (!h_work) DIE("rank %d: host workspace allocation failed", world_rank);
    }

#define GEMM() CUBLASMP_CHECK(cublasMpGemm( \
        mp, \
        CUBLAS_OP_N, \
        CUBLAS_OP_N, \
        (int64_t)m, \
        (int64_t)n, \
        (int64_t)k, \
        &alpha, \
        dA, \
        one, \
        one, \
        descA, \
        dB, \
        one, \
        one, \
        descB, \
        &beta, \
        dC, \
        one, \
        one, \
        descC, \
        CUBLAS_COMPUTE_64F, \
        d_work, \
        d_work_size, \
        h_work, \
        h_work_size))

    GEMM();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    {
        double local_l2 = 0.0;
        double local_sumsq = 0.0;
        double global_sumsq = 0.0;

        CUBLAS_CHECK(cublasDnrm2_64(blas, (int64_t)elems_c, dC, 1, &local_l2));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        local_sumsq = local_l2 * local_l2;

        MPI_Reduce(&local_sumsq, &global_sumsq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (world_rank == 0) {
            fprintf(trace_file,
                    "[LAAB-INFO] cublasmp/gemm | C_l2=%.12e\n",
                    sqrt(global_sumsq));
            fflush(trace_file);
        }
    }

    for (int r = 0; r < reps; ++r) {
        CUDA_CHECK(cudaMemsetAsync(dC, 0, bytes_c ? bytes_c : 1, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        cache_scrub();

        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        GEMM();
        CUDA_CHECK(cudaStreamSynchronize(stream));

        MPI_Barrier(MPI_COMM_WORLD);
        double end = MPI_Wtime();

        double elapsed = end - start;
        double max_elapsed = 0.0;

        MPI_Allreduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (world_rank == 0) {
            time_t now = time(NULL);
            struct tm *tm_info = localtime(&now);
            strftime(datetime, sizeof(datetime), "%Y-%m-%d %H:%M:%S", tm_info);

            fprintf(trace_file,
                    "[LAAB] cublasmp/gemm | rep=%d | dt=%s | dur=%.5f s | perf=%.5f GFLOP/s | nranks=%d | grid_dim=(%d,%d)\n",
                    r, datetime, max_elapsed, flops / max_elapsed / 1e9,
                    world_size, nprow, npcol);
            fflush(trace_file);
        }
    }

#undef GEMM

    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (h_work) free(h_work);
    if (d_work) cudaFree(d_work);

    if (A) free(A);
    if (B) free(B);

    if (dA) cudaFree(dA);
    if (dB) cudaFree(dB);
    if (dC) cudaFree(dC);

    if (descA) CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(descA));
    if (descB) CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(descB));
    if (descC) CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(descC));

    if (grid) CUBLASMP_CHECK(cublasMpGridDestroy(grid));
    if (mp) CUBLASMP_CHECK(cublasMpDestroy(mp));
    if (blas) CUBLAS_CHECK(cublasDestroy(blas));
    if (nccl_comm) NCCL_CHECK(ncclCommDestroy(nccl_comm));
    if (stream) CUDA_CHECK(cudaStreamDestroy(stream));

    if (trace_file) fclose(trace_file);

    MPI_Finalize();
    return EXIT_SUCCESS;
}