#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <getopt.h>
#include <math.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasmp.h>
#include <cuComplex.h>
#include <nccl.h>

extern "C" {
#include "utils.h"
}

static int g_mpi_ok = 0;

#define DIE(...) do { \
    fprintf(stderr, __VA_ARGS__); \
    fprintf(stderr, "\n"); \
    if (g_mpi_ok) MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); \
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

typedef struct {
    const char *dtype_name;
    const char *kernel_name;
    const char *interface_name;
    size_t elem_size;
    cudaDataType data_type;
    cublasComputeType_t compute_type;
    double flop_factor;
} dtype_config_t;

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s -A <matrix_A_file> -B <matrix_B_file> -b <block_size> [--reps <count>] [--tag <tag>]\n"
            "\n"
            "Performs distributed GEMM using matrix files named like\n"
            "M100x200-float64-...dense\n"
            "Supported dtypes: float32, float64, complex128\n",
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

static const dtype_config_t *get_dtype_config(const char *dtype)
{
    static const dtype_config_t configs[] = {
        {"float32", "psgemm", "pxgemm", sizeof(float), CUDA_R_32F, CUBLAS_COMPUTE_32F, 2.0},
        {"float64", "pdgemm", "pxgemm", sizeof(double), CUDA_R_64F, CUBLAS_COMPUTE_64F, 2.0},
        {"complex128", "pzgemm", "pxgemm", sizeof(cuDoubleComplex), CUDA_C_64F, CUBLAS_COMPUTE_64F, 8.0}
    };
    size_t i;
    for (i = 0; i < sizeof(configs) / sizeof(configs[0]); ++i) {
        if (strcmp(dtype, configs[i].dtype_name) == 0) return &configs[i];
    }
    return NULL;
}

static double local_l2_squared(cublasHandle_t blas, const void *dC, size_t elems_c, const dtype_config_t *cfg)
{
    if (strcmp(cfg->dtype_name, "float32") == 0) {
        float local_l2 = 0.0f;
        CUBLAS_CHECK(cublasSnrm2_64(blas, (int64_t)elems_c, (const float *)dC, 1, &local_l2));
        return (double)local_l2 * (double)local_l2;
    }

    if (strcmp(cfg->dtype_name, "float64") == 0) {
        double local_l2 = 0.0;
        CUBLAS_CHECK(cublasDnrm2_64(blas, (int64_t)elems_c, (const double *)dC, 1, &local_l2));
        return local_l2 * local_l2;
    }

    if (strcmp(cfg->dtype_name, "complex128") == 0) {
        double local_l2 = 0.0;
        CUBLAS_CHECK(cublasDznrm2_64(blas, (int64_t)elems_c, (const cuDoubleComplex *)dC, 1, &local_l2));
        return local_l2 * local_l2;
    }

    return 0.0;
}

static void gemm_once(cublasMpHandle_t mp,
                      int m,
                      int n,
                      int k,
                      const dtype_config_t *cfg,
                      const void *alpha,
                      const void *dA,
                      cublasMpMatrixDescriptor_t descA,
                      const void *dB,
                      cublasMpMatrixDescriptor_t descB,
                      const void *beta,
                      void *dC,
                      cublasMpMatrixDescriptor_t descC,
                      void *d_work,
                      size_t d_work_size,
                      void *h_work,
                      size_t h_work_size)
{
    const int64_t one = 1;
    CUBLASMP_CHECK(cublasMpGemm(
        mp,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        (int64_t)m,
        (int64_t)n,
        (int64_t)k,
        alpha,
        dA,
        one,
        one,
        descA,
        dB,
        one,
        one,
        descB,
        beta,
        dC,
        one,
        one,
        descC,
        cfg->compute_type,
        d_work,
        d_work_size,
        h_work,
        h_work_size));
}

int main(int argc, char **argv)
{
    const char *afile = NULL;
    const char *bfile = NULL;
    const char *tag = "0";
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
    int opt;
    int long_idx;
    int nthreads = 1;

    char dtype_a[32];
    char dtype_b[32];
    char datetime[32];
    char hostname[HOST_NAME_MAX];
    char pci_bus_id[16];
    char *visible_dev = NULL;
    const char *omp_env;
    const dtype_config_t *cfg = NULL;
    double flops = 0.0;
    double io_elapsed = 0.0;
    double ab_mb = 0.0;

    int64_t mloc_a = 0;
    int64_t nloc_a = 0;
    int64_t mloc_b = 0;
    int64_t nloc_b = 0;
    int64_t mloc_c = 0;
    int64_t nloc_c = 0;
    int64_t lld_a = 1;
    int64_t lld_b = 1;
    int64_t lld_c = 1;
    int64_t one = 1;

    size_t elems_a = 0;
    size_t elems_b = 0;
    size_t elems_c = 0;
    size_t bytes_a = 0;
    size_t bytes_b = 0;
    size_t bytes_c = 0;
    size_t d_work_size = 0;
    size_t h_work_size = 0;

    FILE *trace_file = NULL;
    void *A = NULL;
    void *B = NULL;
    void *dA = NULL;
    void *dB = NULL;
    void *dC = NULL;
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

    float alpha_s = 1.0f;
    float beta_s = 0.0f;
    double alpha_d = 1.0;
    double beta_d = 0.0;
    cuDoubleComplex alpha_z = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta_z = make_cuDoubleComplex(0.0, 0.0);
    const void *alpha = NULL;
    const void *beta = NULL;

    static struct option long_opts[] = {
        {"reps", required_argument, 0, 1},
        {"tag", required_argument, 0, 2},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "A:B:b:", long_opts, &long_idx)) != -1) {
        switch (opt) {
            case 'A': afile = optarg; break;
            case 'B': bfile = optarg; break;
            case 'b': nb = atoi(optarg); break;
            case 1: reps = atoi(optarg); break;
            case 2: tag = optarg; break;
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
    g_mpi_ok = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    choose_process_grid(world_size, &nprow, &npcol);
    if (nprow * npcol != world_size) {
        DIE("bad process grid: %d x %d != %d", nprow, npcol, world_size);
    }

    myrow = world_rank / npcol;
    mycol = world_rank % npcol;
    local_rank = get_local_rank(MPI_COMM_WORLD);

    trace_file = laab_open_trace_file_with_tag(tag);
    if (!trace_file) DIE("rank %d: failed to open trace file", world_rank);

    if (!parse_matrix_filename(afile, &m, &k_a, dtype_a) ||
        !parse_matrix_filename(bfile, &k_b, &n, dtype_b)) {
        DIE("rank %d: failed to parse matrix metadata", world_rank);
    }

    if (k_a != k_b || strcmp(dtype_a, dtype_b) != 0) {
        DIE("matrix inputs must satisfy A.cols == B.rows and A.dtype == B.dtype");
    }

    cfg = get_dtype_config(dtype_a);
    if (!cfg) {
        DIE("unsupported dtype %s; supported dtypes are float32, float64, complex128", dtype_a);
    }

    k = k_a;
    flops = cfg->flop_factor * (double)m * (double)n * (double)k;

    omp_env = getenv("OMP_NUM_THREADS");
    nthreads = omp_env ? atoi(omp_env) : 1;
    if (nthreads <= 0) nthreads = 1;

    switch (cfg->data_type) {
        case CUDA_R_32F:
            alpha = &alpha_s;
            beta = &beta_s;
            break;
        case CUDA_R_64F:
            alpha = &alpha_d;
            beta = &beta_d;
            break;
        case CUDA_C_64F:
            alpha = &alpha_z;
            beta = &beta_z;
            break;
        default:
            DIE("unsupported CUDA data type");
    }

    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) DIE("rank %d: no CUDA device visible", world_rank);

    device_id = local_rank % device_count;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    visible_dev = getenv("CUDA_VISIBLE_DEVICES");
    snprintf(pci_bus_id, sizeof(pci_bus_id), "%04x:%02x:%02x.0",
             prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    if (gethostname(hostname, sizeof(hostname)) != 0) {
        DIE("gethostname failed");
    }
    hostname[sizeof(hostname) - 1] = '\0';

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
    bytes_a = elems_a * cfg->elem_size;
    bytes_b = elems_b * cfg->elem_size;
    bytes_c = elems_c * cfg->elem_size;

    CUBLASMP_CHECK(cublasMpMatrixDescriptorCreate(
        (int64_t)m, (int64_t)k, (int64_t)nb, (int64_t)nb,
        0, 0, lld_a, cfg->data_type, grid, &descA));
    CUBLASMP_CHECK(cublasMpMatrixDescriptorCreate(
        (int64_t)k, (int64_t)n, (int64_t)nb, (int64_t)nb,
        0, 0, lld_b, cfg->data_type, grid, &descB));
    CUBLASMP_CHECK(cublasMpMatrixDescriptorCreate(
        (int64_t)m, (int64_t)n, (int64_t)nb, (int64_t)nb,
        0, 0, lld_c, cfg->data_type, grid, &descC));

    if (world_rank == 0) {
        fprintf(trace_file,
                "[LAAB-STEP] cublasmp/%s | prob_size=A=%dx%d+B=%dx%d+nb=%dx%d | prec=\"%s\" | flops=%.0f | interface=%s\n",
                cfg->kernel_name, m, k, k, n, nb, nb, cfg->dtype_name, flops, cfg->interface_name);
        fflush(trace_file);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    {
        double io_start = MPI_Wtime();
        A = load_block_cyclic_matrix(afile, myrow, mycol, nprow, npcol, nb, nb, (int)mloc_a, (int)nloc_a);
        B = load_block_cyclic_matrix(bfile, myrow, mycol, nprow, npcol, nb, nb, (int)mloc_b, (int)nloc_b);
        MPI_Barrier(MPI_COMM_WORLD);
        io_elapsed = MPI_Wtime() - io_start;
    }

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

    ab_mb = (double)(bytes_a + bytes_b) / (1024.0 * 1024.0);
    fprintf(trace_file,
            "[LAAB-HOST] cublasmp/%s | hostname=%s | rank=%d | nthreads=%d | gpu_id=%d | gpu_bus=%s | visible_devices=%s | grid_id=(%d,%d) | A_local=(%ld,%ld) | B_local=(%ld,%ld) | C_local=(%ld,%ld) | io_time=%.5f s | ab_size=%.2f MB\n",
            cfg->kernel_name, hostname, world_rank, nthreads, device_id, pci_bus_id,
            visible_dev ? visible_dev : "not set", myrow, mycol,
            (long)mloc_a, (long)nloc_a, (long)mloc_b, (long)nloc_b, (long)mloc_c, (long)nloc_c,
            io_elapsed, ab_mb);
    fflush(trace_file);

    CUBLASMP_CHECK(cublasMpGemm_bufferSize(
        mp,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        (int64_t)m,
        (int64_t)n,
        (int64_t)k,
        alpha,
        dA, one, one, descA,
        dB, one, one, descB,
        beta,
        dC, one, one, descC,
        cfg->compute_type,
        &d_work_size,
        &h_work_size));

    if (d_work_size) CUDA_CHECK(cudaMalloc(&d_work, d_work_size));
    if (h_work_size) {
        h_work = malloc(h_work_size);
        if (!h_work) DIE("rank %d: host workspace allocation failed", world_rank);
    }

    gemm_once(mp, m, n, k, cfg, alpha, dA, descA, dB, descB, beta, dC, descC, d_work, d_work_size, h_work, h_work_size);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    {
        double local_sumsq = local_l2_squared(blas, dC, elems_c, cfg);
        double global_sumsq = 0.0;
        CUDA_CHECK(cudaStreamSynchronize(stream));
        MPI_Reduce(&local_sumsq, &global_sumsq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (world_rank == 0) {
            time_t now = time(NULL);
            struct tm *tm_info = localtime(&now);
            strftime(datetime, sizeof(datetime), "%Y-%m-%d %H:%M:%S", tm_info);
            fprintf(trace_file,
                    "[LAAB-RUN] cublasmp/%s | ts=%s | nranks=%d | grid_dim=(%d,%d) | l2_norm=%.12e\n",
                    cfg->kernel_name, datetime, world_size, nprow, npcol, sqrt(global_sumsq));
            fflush(trace_file);
        }
    }

    for (int r = 0; r < reps; ++r) {
        double elapsed = 0.0;
        double max_elapsed = 0.0;

        CUDA_CHECK(cudaMemsetAsync(dC, 0, bytes_c ? bytes_c : 1, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        cache_scrub();

        MPI_Barrier(MPI_COMM_WORLD);
        {
            double start = MPI_Wtime();
            gemm_once(mp, m, n, k, cfg, alpha, dA, descA, dB, descB, beta, dC, descC, d_work, d_work_size, h_work, h_work_size);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            MPI_Barrier(MPI_COMM_WORLD);
            elapsed = MPI_Wtime() - start;
        }

        MPI_Allreduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (world_rank == 0) {
            time_t now = time(NULL);
            struct tm *tm_info = localtime(&now);
            strftime(datetime, sizeof(datetime), "%Y-%m-%d %H:%M:%S", tm_info);
            fprintf(trace_file,
                    "[LAAB] cublasmp/%s | rep=%d | ts=%s | exec_time (s)= %.5f | perf (GFLOP/s)= %.5f\n",
                    cfg->kernel_name, r, datetime, max_elapsed, flops / max_elapsed / 1e9);
            fflush(trace_file);
        }
    }

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
