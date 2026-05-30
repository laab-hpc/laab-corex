#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <getopt.h>
#include <sched.h>
#include <math.h>
#include <mpi.h>

#define SCRUB_SIZE (50 * 1024 * 1024)

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

static void cache_scrub(void)
{
    double *scrub = (double *)malloc((size_t)SCRUB_SIZE * sizeof(double));
    if (!scrub) return;

    for (size_t i = 0; i < (size_t)SCRUB_SIZE; ++i) scrub[i] = 0.0;
    free(scrub);
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s -m <rows_of_C> -n <cols_of_C> -b <block_size> [--reps <count>]\n"
            "\n"
            "Performs PDGEMM: C(m x n) = A(m x n) * B(n x n)\n"
            "Options:\n"
            "  -m <int>        Number of rows (m)\n"
            "  -n <int>        Number of cols (n); also used as inner dim k\n"
            "  -b <int>        Block size (nb)\n"
            "  --reps <int>    Number of repetitions (default: 1)\n",
            prog);
}

static void choose_process_grid(int nprocs, int *nprow, int *npcol)
{
    int r = (int)sqrt((double)nprocs);
    while (r > 1 && (nprocs % r) != 0) --r;
    if (r == 1) {
        *nprow = 1;
        *npcol = nprocs;
    } else {
        *nprow = r;
        *npcol = nprocs / r;
    }
}

int main(int argc, char **argv)
{
    int m = -1;
    int n = -1;
    int nb = -1;
    int reps = 1;

    static struct option long_opts[] = {
        {"reps", required_argument, 0, 1},
        {0, 0, 0, 0}
    };

    int opt, long_idx;
    while ((opt = getopt_long(argc, argv, "m:n:b:", long_opts, &long_idx)) != -1) {
        switch (opt) {
            case 'm': m = atoi(optarg); break;
            case 'n': n = atoi(optarg); break;
            case 'b': nb = atoi(optarg); break;
            case 1: reps = atoi(optarg); break;
            default:
                usage(argv[0]);
                return EXIT_FAILURE;
        }
    }

    if (m <= 0 || n <= 0 || nb <= 0 || reps <= 0) {
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
    int zero = 0;
    int neg1 = -1;
    blacs_get_(&neg1, &zero, &ictxt);
    blacs_gridinit_(&ictxt, "R", &nprow, &npcol);

    int myrow, mycol;
    blacs_gridinfo_(&ictxt, &nprow, &npcol, &myrow, &mycol);

    const int k = n;
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
        blacs_gridexit_(&ictxt);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    descinit_(descb, &k, &n, &nb, &nb, &isrcproc, &isrcproc, &ictxt, &lld_b, &info);
    if (info != 0) {
        if (world_rank == 0) fprintf(stderr, "descinit(B) failed, info=%d\n", info);
        blacs_gridexit_(&ictxt);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    descinit_(descc, &m, &n, &nb, &nb, &isrcproc, &isrcproc, &ictxt, &lld_c, &info);
    if (info != 0) {
        if (world_rank == 0) fprintf(stderr, "descinit(C) failed, info=%d\n", info);
        blacs_gridexit_(&ictxt);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    size_t szA = (size_t)lld_a * (size_t)nloc_a;
    size_t szB = (size_t)lld_b * (size_t)nloc_b;
    size_t szC = (size_t)lld_c * (size_t)nloc_c;

    double *A = (double *)malloc(szA * sizeof(double));
    double *B = (double *)malloc(szB * sizeof(double));
    double *C = (double *)malloc(szC * sizeof(double));

    if (!A || !B || !C) {
        fprintf(stderr, "Rank %d allocation failed.\n", world_rank);
        free(A);
        free(B);
        free(C);
        blacs_gridexit_(&ictxt);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    srand48((long)time(NULL) + world_rank);
    for (size_t i = 0; i < szA; ++i) A[i] = drand48();
    for (size_t i = 0; i < szB; ++i) B[i] = drand48();
    for (size_t i = 0; i < szC; ++i) C[i] = 0.0;

    const int one = 1;
    const double alpha = 1.0;
    const double beta = 0.0;
    const double flops = 2.0 * (double)m * (double)n * (double)k;

    if (world_rank == 0) {
        printf("[LAAB-INFO] scalapack/pdgemm | op_sizes=(m=%d, n=%d) | np=%d | flops=%.0f\n",
               m, n, world_size, flops);
        fflush(stdout);
    }

    /* Warmup */
    pdgemm_("N", "N", &m, &n, &k, &alpha,
            A, &one, &one, desca,
            B, &one, &one, descb,
            &beta,
            C, &one, &one, descc);

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

        double local_elapsed = end - start;
        double elapsed = 0.0;
        MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (world_rank == 0) {
            double gflops = flops / elapsed / 1e9;

            time_t now = time(NULL);
            struct tm *tm_info = localtime(&now);
            char datetime[32];
            strftime(datetime, sizeof(datetime), "%Y-%m-%d %H:%M:%S", tm_info);

            int cpu = sched_getcpu();
            printf("[LAAB] scalapack/pdgemm | rep=%d | dt=%s | dur=%.5f s | perf=%.5f GFLOP/s | np=%d | cid=%d \n",
                   r, datetime, elapsed, gflops, world_size, cpu);
            fflush(stdout);
        }
    }

    free(A);
    free(B);
    free(C);

    blacs_gridexit_(&ictxt);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
