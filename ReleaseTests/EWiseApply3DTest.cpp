/*
 * Test for the SpParMat3D element-wise apply (EWiseApply on SpParMat3D).
 * Uses the 2D EWiseApply as the oracle: for each layer count, the 3D result
 * (converted back to 2D) must equal the 2D result exactly.
 *
 * Two cases per layer count:
 *   1. Both operands built via the 2D->3D constructor (canonical, sorted).
 *   2. One operand is a Mult_AnXBn_SUMMA3D product with sortOutput=true -- this
 *      is the intended use (feeding a 3D SpGEMM result into a sorted-merge op),
 *      and requires the fixed PBBS integerSort + the sorted SUMMA3D output.
 *
 * Reuses the rmat_scale16 matrices (same data as SpGEMM3DTest).
 */
#include <mpi.h>
#include <cmath>
#include <iostream>
#include <functional>
#include <sstream>
#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/CommGrid3D.h"
#include "CombBLAS/SpParMat3D.h"
#include "CombBLAS/ParFriends.h"
using namespace std;
using namespace combblas;

#ifdef _OPENMP
int cblas_splits = omp_get_max_threads();
#else
int cblas_splits = 1;
#endif

// Predicate that keeps every combined entry.
struct SelectAll { template <class T1, class T2> bool operator()(const T1&, const T2&) const { return true; } };

static bool isPerfectSquare(int x){ int r = (int)(std::sqrt((double)x) + 0.5); return r*r == x; }

int main(int argc, char* argv[])
{
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    if(argc < 3){
        if(myrank == 0) cout << "Usage: ./EWiseApply3DTest <MatrixA> <MatrixB>" << endl;
        MPI_Finalize();
        return -1;
    }
    string Aname(argv[1]), Bname(argv[2]);
    typedef SpDCCols<int64_t, double>            DER;
    typedef SpParMat<int64_t, double, DER>       Mat2D;
    typedef SpParMat3D<int64_t, double, DER>     Mat3D;
    typedef PlusTimesSRing<double, double>       PTFF;

    shared_ptr<CommGrid> fullWorld;
    fullWorld.reset(new CommGrid(MPI_COMM_WORLD, 0, 0));
    Mat2D A(fullWorld), B(fullWorld);
    A.ParallelReadMM(Aname, true, maximum<double>());
    B.ParallelReadMM(Bname, true, maximum<double>());

    SelectAll selectAll;
    // 2D references (union-sum: missing entries treated as 0).
    Mat2D refAB   = EWiseApply<double, DER>(A, B, plus<double>(), selectAll, true, true, 0.0, 0.0, true);
    Mat2D prod2D  = Mult_AnXBn_Synch<PTFF, double, DER, int64_t, double, double, DER, DER>(A, B, false, false);
    Mat2D refAprod= EWiseApply<double, DER>(A, prod2D, plus<double>(), selectAll, true, true, 0.0, 0.0, true);

    int failed = 0;
    for(int layers = 1; layers <= 16; layers *= 4){
        if(layers > nprocs || nprocs % layers != 0 || !isPerfectSquare(nprocs / layers)){
            if(myrank == 0) fprintf(stderr, "skip %d layers (incompatible with %d procs)\n", layers, nprocs);
            continue;
        }
        if(myrank == 0) fprintf(stderr, "Trying %d layers\n", layers);

        // Case 1: two ctor-built operands (both column-split, canonical, sorted).
        Mat3D A3(A, layers, true, false), B3(B, layers, true, false);
        Mat3D C3 = EWiseApply<double, DER>(A3, B3, plus<double>(), selectAll, true, true, 0.0, 0.0, true);
        Mat2D C3_2D = C3.Convert2D();
        bool ok1 = (refAB == C3_2D);

        // Case 2: EWiseApply3D with a SUMMA3D(sortOutput=true) product as an operand.
        Mat3D Acol(A, layers, true, false), Brow(B, layers, false, false);
        Mat3D prod3D = Mult_AnXBn_SUMMA3D<PTFF, double, DER, int64_t, double, double, DER, DER>(Acol, Brow, /*sortOutput=*/true);
        Mat3D D3 = EWiseApply<double, DER>(A3, prod3D, plus<double>(), selectAll, true, true, 0.0, 0.0, true);
        Mat2D D3_2D = D3.Convert2D();
        bool ok2 = (refAprod == D3_2D);

        if(myrank == 0){
            fprintf(stderr, "  ctor-operands EWiseApply3D  : %s\n", ok1 ? "Correct" : "NOT correct");
            fprintf(stderr, "  SUMMA3D-operand EWiseApply3D: %s\n", ok2 ? "Correct" : "NOT correct");
        }
        if(!ok1 || !ok2) failed = 1;
    }

    int global = 0;
    MPI_Allreduce(&failed, &global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if(myrank == 0) fprintf(stderr, global == 0 ? "EWiseApply3D TEST PASSED\n" : "EWiseApply3D TEST FAILED\n");
    MPI_Finalize();
    return global;
}
