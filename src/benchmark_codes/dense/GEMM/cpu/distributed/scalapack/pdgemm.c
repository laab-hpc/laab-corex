#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <getopt.h>
#include <sched.h>
#include <math.h>
#include <string.h>
#include <cblas.h>
#include <mpi.h>

#include "utils.h"

#define DEFAULT_NB 256

/* BLACS / ScaLAPACK Fortran interfaces */
extern void blacs_pinfo_(int *mypnum, int *nprocs);
extern void blacs_get_(int *icontxt, int *what, int *val);
extern void blacs_gridinit_(int *icontxt, const char *order, int *nprow, int *npcol);
extern void blacs_gridinfo_(int *icontxt, int *nprow, int *npcol, int *myrow, int *mycol);
extern void blacs_gridexit_(int *icontxt);

extern int numroc_(const int *n, const int *nb, const int *iproc,
                   const int *isrcproc, const int *nprocs);
extern void descinit_(int *desc, const int *m, const int *n,
                      const int *mb, const int *nb,
                      const int *irsrc, const int *icsrc,
                      const int *ictxt, const int *lld, int *info);
extern void pdgemm_(const char *transa, const char *transb,
                    const int *m, const int *n, const int *k,
                    const double *alpha,
                    const double *A, const int *ia, const int *ja, const int *desca,
                    const double *B, const int *ib, const int *jb, const int *descb,
                    const double *beta,
                    double *C, const int *ic, const int *jc, const int *descc);

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s -A <matrix_A_file> -B <matrix_B_file> -b <block_size> [--reps <count>]\n"
            "\n"
            "Performs PDGEMM: C = A * B using matrix files named like\n"
            "M100x200-float64-...dense\n"
            "Options:\n"
            "  -A <path>       Path to matrix A file\n"
            "  -B <path>       Path to matrix B file\n"
            "  -b <int>        ScaLAPACK block size\n"
            "  --reps <int>    Number of repetitions (default: 1)\n",
            prog);
}



int main(int argc, char **argv)
{
    const char *afile = NULL;
    const char *bfile = NULL;
    int nb = -1;
    int reps = 1;
    FILE *trace_file = NULL;

    static struct option long_opts[] = {
        {"reps", required_argument, 0, 1},
        {0, 0, 0, 0}
    };

    int opt, long_idx;
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

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int mypnum, nprocs;
    blacs_pinfo_(&mypnum, &nprocs);

    int nprow, npcol;
    choose_process_grid(nprocs, &nprow, &npcol);

    int ictxt = 0;
    int neg1 = -1;
    int zero = 0;
    blacs_get_(&neg1, &zero, &ictxt);
    blacs_gridinit_(&ictxt, "R", &nprow, &npcol);

    int myrow, mycol;
    blacs_gridinfo_(&ictxt, &nprow, &npcol, &myrow, &mycol);

    trace_file = laab_open_trace_file();
    if (!trace_file) {
        blacs_gridexit_(&ictxt);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int m = 0, k_a = 0, k_b = 0, n = 0;
    char dtype_a[32], dtype_b[32];
    if (!parse_matrix_filename(afile, &m, &k_a, dtype_a) ||
        !parse_matrix_filename(bfile, &k_b, &n, dtype_b)) {
        if (world_rank == 0) fprintf(stderr, "failed to parse matrix metadata from filenames\n");
        fclose(trace_file);
        blacs_gridexit_(&ictxt);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (world_rank == 0) {
        fprintf(trace_file,
                "[LAAB-INFO] scalapack/pdgemm | nranks=%d | grid_dim=(%d,%d) | A=(%d,%d,%s) | B=(%d,%d,%s) | block_size=%dx%d | flops=%.0f\n",
                world_size, nprow, npcol, m, k_a, dtype_a, k_b, n, dtype_b,
                nb, nb, 2.0 * (double)m * (double)n * (double)k_a);
        fflush(trace_file);
    }

    if (k_a != k_b || strcmp(dtype_a, "float64") != 0 || strcmp(dtype_b, "float64") != 0) {
        if (world_rank == 0) {
            fprintf(stderr, "matrix inputs must satisfy A.cols == B.rows and both dtypes must be float64\n");
        }
        fclose(trace_file);
        blacs_gridexit_(&ictxt);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const int k = k_a;
    const int isrcproc = 0;

    int mloc_a = numroc_(&m, &nb, &myrow, &isrcproc, &nprow);
    int nloc_a = numroc_(&k, &nb, &mycol, &isrcproc, &npcol);
    int mloc_b = numroc_(&k, &nb, &myrow, &isrcproc, &nprow);
    int nloc_b = numroc_(&n, &nb, &mycol, &isrcproc, &npcol);
    int mloc_c = numroc_(&m, &nb, &myrow, &isrcproc, &nprow);
    int nloc_c = numroc_(&n, &nb, &mycol, &isrcproc, &npcol);

    int lld_a = (mloc_a > 1) ? mloc_a : 1;
    int lld_b = (mloc_b > 1) ? mloc_b : 1;
    int lld_c = (mloc_c > 1) ? mloc_c : 1;

    int desca[9], descb[9], descc[9];
    int info = 0;
    descinit_(desca, &m, &k, &nb, &nb, &isrcproc, &isrcproc, &ictxt, &lld_a, &info);
    if (info != 0) {
        if (world_rank == 0) fprintf(stderr, "descinit(A) failed, info=%d\n", info);
        fclose(trace_file);
        blacs_gridexit_(&ictxt);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    descinit_(descb, &k, &n, &nb, &nb, &isrcproc, &isrcproc, &ictxt, &lld_b, &info);
    if (info != 0) {
        if (world_rank == 0) fprintf(stderr, "descinit(B) failed, info=%d\n", info);
        fclose(trace_file);
        blacs_gridexit_(&ictxt);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    descinit_(descc, &m, &n, &nb, &nb, &isrcproc, &isrcproc, &ictxt, &lld_c, &info);
    if (info != 0) {
        if (world_rank == 0) fprintf(stderr, "descinit(C) failed, info=%d\n", info);
        fclose(trace_file);
        blacs_gridexit_(&ictxt);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    size_t szC = (size_t)lld_c * (size_t)nloc_c;
    double io_start, io_end, io_elapsed, io_max_elapsed;
    double ab_mb;

    MPI_Barrier(MPI_COMM_WORLD);
    io_start = MPI_Wtime();
    double *A = (double *)load_block_cyclic_matrix(afile, myrow, mycol, nprow, npcol,
                                                   nb, nb, mloc_a, nloc_a);
    double *B = (double *)load_block_cyclic_matrix(bfile, myrow, mycol, nprow, npcol,
                                                   nb, nb, mloc_b, nloc_b);
    MPI_Barrier(MPI_COMM_WORLD);
    io_end = MPI_Wtime();
    double *C = (double *)calloc(szC, sizeof(double));

    if (!A || !B || !C) {
        fprintf(stderr, "Rank %d allocation failed.\n", world_rank);
        free(A);
        free(B);
        free(C);
        fclose(trace_file);
        blacs_gridexit_(&ictxt);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    io_elapsed = io_end - io_start;
    io_max_elapsed = 0.0;
    MPI_Allreduce(&io_elapsed, &io_max_elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    ab_mb = (double)(((size_t)mloc_a * (size_t)nloc_a +
                      (size_t)mloc_b * (size_t)nloc_b) * sizeof(double)) /
            (1024.0 * 1024.0);
    fprintf(trace_file,
            "[LAAB-INFO] scalapack/pdgemm | rank=%d | grid_id=(%d,%d) | A_local=(%d,%d) | B_local=(%d,%d) | io_time=%.5f s | ab_size=%.2f MB\n",
            world_rank, myrow, mycol, mloc_a, nloc_a, mloc_b, nloc_b, io_max_elapsed, ab_mb);
    fflush(trace_file);

    const int one = 1;
    const double alpha = 1.0;
    const double beta = 0.0;
    const double flops = 2.0 * (double)m * (double)n * (double)k;

    pdgemm_("N", "N", &m, &n, &k, &alpha,
            A, &one, &one, desca,
            B, &one, &one, descb,
            &beta,
            C, &one, &one, descc);

    {
        double local_sumsq = 0.0;
        double global_sumsq = 0.0;
        double c_l2 = 0.0;
        double local_l2 = cblas_dnrm2((int)szC, C, 1);
        local_sumsq = local_l2 * local_l2;

        MPI_Reduce(&local_sumsq, &global_sumsq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (world_rank == 0) {
            c_l2 = sqrt(global_sumsq);
            fprintf(trace_file,
                    "[LAAB-INFO] scalapack/pdgemm | C_l2=%.12e\n",
                    c_l2);
            fflush(trace_file);
        }
    }

    for (int r = 0; r < reps; ++r) {
        for (size_t i = 0; i < szC; ++i) C[i] = 0.0;

        cache_scrub();
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        pdgemm_("N", "N", &m, &n, &k, &alpha,
                A, &one, &one, desca,
                B, &one, &one, descb,
                &beta,
                C, &one, &one, descc);

        MPI_Barrier(MPI_COMM_WORLD);
        double end = MPI_Wtime();

        double elapsed = end - start;
        double max_elapsed = 0.0;
        MPI_Allreduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        double gflops = flops / max_elapsed / 1e9;
        time_t now = time(NULL);
        struct tm *tm_info = localtime(&now);
        char datetime[32];
        strftime(datetime, sizeof(datetime), "%Y-%m-%d %H:%M:%S", tm_info);

        if (world_rank == 0) {
            fprintf(trace_file,
                    "[LAAB] scalapack/pdgemm | rep=%d | dt=%s | dur=%.5f s | perf=%.5f GFLOP/s | nranks=%d | grid_dim=(%d,%d)\n",
                    r, datetime, max_elapsed, gflops, world_size, nprow, npcol);
            fflush(trace_file);
        }
    }

    free(A);
    free(B);
    free(C);
    fclose(trace_file);

    blacs_gridexit_(&ictxt);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
