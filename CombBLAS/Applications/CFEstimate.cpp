/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.6 -------------------------------------------------*/
/* date: 6/15/2017 ---------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc  --------------------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2017, The Regents of the University of California

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */

#include <mpi.h>
#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/CommGrid3D.h"
#include "CombBLAS/SpParMat3D.h"
#include "CombBLAS/ParFriends.h"

using namespace std;
using namespace combblas;

#define EPS 0.0001

#ifdef TIMING
double cblas_alltoalltime;
double cblas_allgathertime;
//////////////////////////
double mcl_Abcasttime;
double mcl_Bbcasttime;
double mcl_localspgemmtime;
double mcl_multiwaymergetime;
double mcl_kselecttime;
double mcl_prunecolumntime;
double mcl_symbolictime;
///////////////////////////
double mcl_Abcasttime_prev;
double mcl_Bbcasttime_prev;
double mcl_localspgemmtime_prev;
double mcl_multiwaymergetime_prev;
double mcl_kselecttime_prev;
double mcl_prunecolumntime_prev;
double mcl_symbolictime_prev;
///////////////////////////
double mcl3d_conversiontime;
double mcl3d_symbolictime;
double mcl3d_Abcasttime;
double mcl3d_Bbcasttime;
double mcl3d_SUMMAtime;
double mcl3d_localspgemmtime;
double mcl3d_SUMMAmergetime;
double mcl3d_reductiontime;
double mcl3d_3dmergetime;
double mcl3d_kselecttime;
///////////////////////////
double g_mcl3d_conversiontime;
double g_mcl3d_symbolictime;
double g_mcl3d_Abcasttime;
double g_mcl3d_Bbcasttime;
double g_mcl3d_SUMMAtime;
double g_mcl3d_localspgemmtime;
double g_mcl3d_SUMMAmergetime;
double g_mcl3d_reductiontime;
double g_mcl3d_3dmergetime;
double g_mcl3d_kselecttime;
///////////////////////////
double mcl3d_conversiontime_prev;
double mcl3d_symbolictime_prev;
double mcl3d_Abcasttime_prev;
double mcl3d_Bbcasttime_prev;
double mcl3d_SUMMAtime_prev;
double mcl3d_localspgemmtime_prev;
double mcl3d_SUMMAmergetime_prev;
double mcl3d_reductiontime_prev;
double mcl3d_3dmergetime_prev;
double mcl3d_kselecttime_prev;
#endif

#ifdef _OPENMP
int cblas_splits = omp_get_max_threads();
#else
int cblas_splits = 1;
#endif


// Simple helper class for declarations: Just the numerical type is templated
// The index type and the sequential matrix type stays the same for the whole code
// In this case, they are "int" and "SpDCCols"
template <class NT>
class PSpMat
{
public:
    typedef SpDCCols < int64_t, NT > DCCols;
    typedef SpParMat < int64_t, NT, DCCols > MPI_DCCols;
};

int main(int argc, char* argv[])
{
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

    if(argc < 2){
        if(myrank == 0)
        {
            cout << "Usage: ./<Binary> <MatrixA> " << endl;
        }
        MPI_Finalize();
        return -1;
    }
    else {
        double vm_usage, resident_set;
        string Aname(argv[1]);

        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );
        
        SpParMat<int64_t, double, SpDCCols < int64_t, double >> M(fullWorld);

        M.ParallelReadMM(Aname, true, maximum<double>());
        FullyDistVec<int64_t, int64_t> p( M.getcommgrid() );
        p.iota(M.getnrow(), 0);
        p.RandPerm();
        (M)(p,p,true);// in-place permute to save memory
        //M.ReadGeneralizedTuples(Aname, maximum<double>());
        //MCLPruneRecoverySelect<int64_t, double, SpDCCols < int64_t, double >>(M, 2.0, 102, 105, 0.9, 1);
        int64_t n = M.getnrow();
        int64_t nnz = M.getnnz();
        if(myrank == 0){
            fprintf(stderr, "n after MCLPruneRecoverySelect: %lld\n", n);
            fprintf(stderr, "nnz after MCLPruneRecoverySelect: %lld\n", nnz);
        }

        typedef PlusTimesSRing<double, double> PTFF;
        double t0, t1;
        
        /*
        SpParMat<int64_t, double, SpDCCols < int64_t, double >> X(M);
        SpParMat<int64_t, double, SpDCCols < int64_t, double >> Y(M);
        M.FreeMemory();
        SpParMat<int64_t, double, SpDCCols < int64_t, double >> M2 = Mult_AnXBn_Synch<PTFF, double, SpDCCols<int64_t, double>, int64_t >(X, Y);
        X.FreeMemory();
        Y.FreeMemory();
        FullyDistVec<int64_t, int64_t> p2( M2.getcommgrid() );
        p2.iota(M2.getnrow(), 0);
        p2.RandPerm();
        (M2)(p2,p2,true);// in-place permute to save memory

        
        SpParMat<int64_t, double, SpDCCols < int64_t, double >> X2(M2);
        SpParMat<int64_t, double, SpDCCols < int64_t, double >> Y2(M2);
        M2.FreeMemory();
        SpParMat<int64_t, double, SpDCCols < int64_t, double >> M4 = Mult_AnXBn_Synch<PTFF, double, SpDCCols<int64_t, double>, int64_t >(X2, Y2);
        X2.FreeMemory();
        Y2.FreeMemory();
        FullyDistVec<int64_t, int64_t> p4( M4.getcommgrid());
        p4.iota(M4.getnrow(), 0);
        p4.RandPerm();
        (M4)(p4,p4,true);// in-place permute to save memory
        */

        SpParMat<int64_t, double, SpDCCols < int64_t, double >> X(M);
        SpParMat<int64_t, double, SpDCCols < int64_t, double >> Y(M);
        int64_t flop = EstimateFLOP<PTFF, int64_t, double, double, SpDCCols<int64_t, double>, SpDCCols<int64_t, double> >(X, Y, false, false);
        if(myrank == 0){
            fprintf(stderr, "flop: %lld\n", flop);
        }
        SpParMat<int64_t, double, SpDCCols < int64_t, double >> Z = Mult_AnXBn_Synch<PTFF, double, SpDCCols<int64_t, double>, int64_t >(X, Y);
        int64_t nnzc = Z.getnnz();
        if(myrank == 0){
            fprintf(stderr, "nnzc: %lld\n", nnzc);
        }
        X.FreeMemory();
        Y.FreeMemory();
        Z.FreeMemory();
    }
    MPI_Finalize();
    return 0;
}
