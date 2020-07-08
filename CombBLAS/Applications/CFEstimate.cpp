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
double mcl3d_totaltime;
double mcl3d_floptime;
int64_t mcl3d_layer_flop;
int64_t mcl3d_layer_nnzc;
int64_t mcl3d_nnzc;
int64_t mcl3d_flop;
int mcl3d_max_phase;
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
double g_mcl3d_totaltime;
double g_mcl3d_floptime;
int64_t g_mcl3d_layer_flop;
int64_t g_mcl3d_layer_nnzc;
////////////////////////////
double l_mcl3d_conversiontime;
double l_mcl3d_symbolictime;
double l_mcl3d_Abcasttime;
double l_mcl3d_Bbcasttime;
double l_mcl3d_SUMMAtime;
double l_mcl3d_localspgemmtime;
double l_mcl3d_SUMMAmergetime;
double l_mcl3d_reductiontime;
double l_mcl3d_3dmergetime;
double l_mcl3d_kselecttime;
double l_mcl3d_totaltime;
double l_mcl3d_floptime;
int64_t l_mcl3d_layer_flop;
int64_t l_mcl3d_layer_nnzc;
////////////////////////////
double a_mcl3d_conversiontime;
double a_mcl3d_symbolictime;
double a_mcl3d_Abcasttime;
double a_mcl3d_Bbcasttime;
double a_mcl3d_SUMMAtime;
double a_mcl3d_localspgemmtime;
double a_mcl3d_SUMMAmergetime;
double a_mcl3d_reductiontime;
double a_mcl3d_3dmergetime;
double a_mcl3d_kselecttime;
double a_mcl3d_totaltime;
double a_mcl3d_floptime;
int64_t a_mcl3d_layer_flop;
int64_t a_mcl3d_layer_nnzc;
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
double mcl3d_totaltime_prev;
double mcl3d_floptime_prev;
int64_t mcl3d_layer_flop_prev;
int64_t mcl3d_layer_nnzc_prev;
///////////////////////////
double sym_Abcasttime;
double sym_Bbcasttime;
double sym_estimatefloptime;
double sym_estimatennztime;
double sym_asquarennztime;
double sym_SUMMAnnzreductiontime;
double sym_totaltime;
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
        if(myrank == 0){
            fprintf(stderr, "Data: %s\n", argv[1]);
        }
        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );
        
        double t0, t1;

        SpParMat<int64_t, double, SpDCCols < int64_t, double >> M(fullWorld);
        
        // Read labelled triple files
        t0 = MPI_Wtime();
        M.ReadGeneralizedTuples(Aname, maximum<double>());
        t1 = MPI_Wtime();
        if(myrank == 0) fprintf(stderr, "Time taken to read file: %lf\n", t1-t0);
        
        typedef PlusTimesSRing<double, double> PTFF;

        SpParMat<int64_t, double, SpDCCols < int64_t, double >> A2D(M);
        SpParMat<int64_t, double, SpDCCols < int64_t, double >> B2D(M);
        SpParMat3D<int64_t, double, SpDCCols < int64_t, double >> A3D(A2D, 4, true, false);
        SpParMat3D<int64_t, double, SpDCCols < int64_t, double >> B3D(B2D, 4, false, false);
        SpParMat3D<int64_t, double, SpDCCols < int64_t, double >> C3D = A3D.template MemEfficientSpGEMM3D<PTFF>(B3D,
            10, 2.0, 1100, 1400, 0.9, 1, 0);
        SpParMat<int64_t, double, SpDCCols < int64_t, double >> C2D = C3D.Convert2D();

        SpParMat<int64_t, double, SpDCCols < int64_t, double >> X2D(M);
        SpParMat<int64_t, double, SpDCCols < int64_t, double >> Y2D(M);
        SpParMat<int64_t, double, SpDCCols < int64_t, double >> Z2D = MemEfficientSpGEMM< PTFF, double, SpDCCols<int64_t, double>, int64_t >(X2D, Y2D, 10, 2.0, 1100, 1400, 0.9, 1, 0);
        int64_t C2D_m = C2D.getnrow();
        int64_t C2D_n = C2D.getncol();
        int64_t C2D_nnz = C2D.getnnz();
        int64_t Z2D_m = Z2D.getnrow();
        int64_t Z2D_n = Z2D.getncol();
        int64_t Z2D_nnz = Z2D.getnnz();
        bool flag = false;
        if(Z2D == C2D) flag = true;
        if(myrank == 0) fprintf(stderr, "%d\n", flag);
        if(myrank == 0){
            fprintf(stderr, "m: %lld - %lld\n", C2D_m, Z2D_m);
            fprintf(stderr, "n: %lld - %lld\n", C2D_n, Z2D_n);
            fprintf(stderr, "nnz: %lld - %lld\n", C2D_nnz, Z2D_nnz);
        }
    }
    MPI_Finalize();
    return 0;
}
