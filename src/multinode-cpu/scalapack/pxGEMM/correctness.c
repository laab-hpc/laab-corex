#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

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

static int owner_1d(int gidx_1based, int nb, int nprocs)
{
    return ((gidx_1based - 1) / nb) % nprocs;
}

static int check_pdgemm(void)
{
    const int m = 16;
    const int k = 16;
    const int n = 16;
    const int nb = 4;
    const int isrc = 0;
    const int one = 1;

    const double alpha = 1.0;
    const double beta = 0.0;
    const double tol = 1e-12;

    /* Explicit 16x16 matrices with known expected result:
       B = 2*I  => expected C = A*B = 2*A */
    double A_global[16 * 16];
    double B_global[16 * 16];
    double expected_global[16 * 16];

    int mypnum, nprocs;
    blacs_pinfo_(&mypnum, &nprocs);
    if ((nprocs % 2) != 0) {
        if (mypnum == 0) fprintf(stderr, "Use an even number of MPI ranks for this test.\n");
        return 0;
    }

    int nprow = 2;
    int npcol = nprocs / 2;

    int ictxt = 0;
    int zero = 0;
    int neg1 = -1;
    blacs_get_(&neg1, &zero, &ictxt);
    blacs_gridinit_(&ictxt, "R", &nprow, &npcol);

    int myrow, mycol;
    blacs_gridinfo_(&ictxt, &nprow, &npcol, &myrow, &mycol);

    int mloc_a = numroc_(&m, &nb, &myrow, &isrc, &nprow);
    int nloc_a = numroc_(&k, &nb, &mycol, &isrc, &npcol);
    int mloc_b = numroc_(&k, &nb, &myrow, &isrc, &nprow);
    int nloc_b = numroc_(&n, &nb, &mycol, &isrc, &npcol);
    int mloc_c = numroc_(&m, &nb, &myrow, &isrc, &nprow);
    int nloc_c = numroc_(&n, &nb, &mycol, &isrc, &npcol);

    int lld_a = (mloc_a > 1) ? mloc_a : 1;
    int lld_b = (mloc_b > 1) ? mloc_b : 1;
    int lld_c = (mloc_c > 1) ? mloc_c : 1;

    int desca[9], descb[9], descc[9];
    int info = 0;
    descinit_(desca, &m, &k, &nb, &nb, &isrc, &isrc, &ictxt, &lld_a, &info);
    if (info != 0) {
        if (mypnum == 0) fprintf(stderr, "descinit(A) failed, info=%d\n", info);
        blacs_gridexit_(&ictxt);
        return 0;
    }
    descinit_(descb, &k, &n, &nb, &nb, &isrc, &isrc, &ictxt, &lld_b, &info);
    if (info != 0) {
        if (mypnum == 0) fprintf(stderr, "descinit(B) failed, info=%d\n", info);
        blacs_gridexit_(&ictxt);
        return 0;
    }
    descinit_(descc, &m, &n, &nb, &nb, &isrc, &isrc, &ictxt, &lld_c, &info);
    if (info != 0) {
        if (mypnum == 0) fprintf(stderr, "descinit(C) failed, info=%d\n", info);
        blacs_gridexit_(&ictxt);
        return 0;
    }

    size_t szA = (size_t)lld_a * (size_t)nloc_a;
    size_t szB = (size_t)lld_b * (size_t)nloc_b;
    size_t szC = (size_t)lld_c * (size_t)nloc_c;

    double *A = (double *)calloc(szA, sizeof(double));
    double *B = (double *)calloc(szB, sizeof(double));
    double *C = (double *)calloc(szC, sizeof(double));

    if (!A || !B || !C) {
        if (mypnum == 0) fprintf(stderr, "Allocation failed.\n");
        free(A); free(B); free(C);
        blacs_gridexit_(&ictxt);
        return 0;
    }

    int lrowA[17] = {0};
    int lrowB[17] = {0};
    int lrowC[17] = {0};
    int lcolA[17] = {0};
    int lcolB[17] = {0};
    int lcolC[17] = {0};

    int c = 0;
    for (int i = 1; i <= m; ++i) if (owner_1d(i, nb, nprow) == myrow) lrowA[i] = ++c;
    c = 0;
    for (int i = 1; i <= k; ++i) if (owner_1d(i, nb, nprow) == myrow) lrowB[i] = ++c;
    c = 0;
    for (int i = 1; i <= m; ++i) if (owner_1d(i, nb, nprow) == myrow) lrowC[i] = ++c;

    c = 0;
    for (int j = 1; j <= k; ++j) if (owner_1d(j, nb, npcol) == mycol) lcolA[j] = ++c;
    c = 0;
    for (int j = 1; j <= n; ++j) if (owner_1d(j, nb, npcol) == mycol) lcolB[j] = ++c;
    c = 0;
    for (int j = 1; j <= n; ++j) if (owner_1d(j, nb, npcol) == mycol) lcolC[j] = ++c;

    for (int j = 1; j <= n; ++j) {
        for (int i = 1; i <= m; ++i) {
            double a = (double)(i * 16 + j);
            A_global[(i - 1) + (j - 1) * m] = a;
            B_global[(i - 1) + (j - 1) * m] = (i == j) ? 2.0 : 0.0;
            expected_global[(i - 1) + (j - 1) * m] = 2.0 * a;
        }
    }

    for (int i = 1; i <= m; ++i) {
        if (lrowA[i] == 0) continue;
        for (int j = 1; j <= k; ++j) {
            if (lcolA[j] == 0) continue;
            int li = lrowA[i];
            int lj = lcolA[j];
            A[(size_t)(li - 1) + (size_t)(lj - 1) * (size_t)lld_a] = A_global[(i - 1) + (j - 1) * m];
        }
    }

    for (int i = 1; i <= k; ++i) {
        if (lrowB[i] == 0) continue;
        for (int j = 1; j <= n; ++j) {
            if (lcolB[j] == 0) continue;
            int li = lrowB[i];
            int lj = lcolB[j];
            B[(size_t)(li - 1) + (size_t)(lj - 1) * (size_t)lld_b] = B_global[(i - 1) + (j - 1) * k];
        }
    }

    pdgemm_("N", "N", &m, &n, &k, &alpha,
            A, &one, &one, desca,
            B, &one, &one, descb,
            &beta,
            C, &one, &one, descc);

    int local_ok = 1;
    for (int i = 1; i <= m; ++i) {
        if (lrowC[i] == 0) continue;
        for (int j = 1; j <= n; ++j) {
            if (lcolC[j] == 0) continue;
            int li = lrowC[i];
            int lj = lcolC[j];
            double got = C[(size_t)(li - 1) + (size_t)(lj - 1) * (size_t)lld_c];
            double exp = expected_global[(i - 1) + (j - 1) * m];
            if (fabs(got - exp) > tol) {
                local_ok = 0;
                break;
            }
        }
        if (!local_ok) break;
    }

    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

    if (mypnum == 0) {
        if (global_ok) {
            printf("PDGEMM correctness: PASS\n");
        } else {
            printf("PDGEMM correctness: FAIL\n");
            printf("Expected relation: C = 2*A for explicit 16x16 test matrices.\n");
        }
    }

    free(A); free(B); free(C);
    blacs_gridexit_(&ictxt);
    return global_ok;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (world_size < 4) {
        if (world_rank == 0) {
            fprintf(stderr, "PDGEMM correctness requires at least 4 MPI ranks.\n");
        }
        MPI_Finalize();
        return 1;
    }
    int ok = check_pdgemm();
    MPI_Finalize();
    return ok ? 0 : 1;
}
