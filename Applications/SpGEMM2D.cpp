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
double mcl_totaltime;
double mcl_tt;
int64_t mcl_nnzc;
///////////////////////////
double mcl_Abcasttime_prev;
double mcl_Bbcasttime_prev;
double mcl_localspgemmtime_prev;
double mcl_multiwaymergetime_prev;
double mcl_kselecttime_prev;
double mcl_prunecolumntime_prev;
double mcl_symbolictime_prev;
double mcl_totaltime_prev;
double mcl_tt_prev;
int64_t mcl_nnzc_prev;
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

        // Read matrix market files
        M.ParallelReadMM(Aname, true, maximum<double>());
        FullyDistVec<int64_t, int64_t> p( M.getcommgrid() );
        p.iota(M.getnrow(), 0);
        p.RandPerm();
        (M)(p,p,true);// in-place permute to save memory
        
        // Read labelled triple files
        //M.ReadGeneralizedTuples(Aname, maximum<double>());
        
        //MCLPruneRecoverySelect<int64_t, double, SpDCCols < int64_t, double >>(M, 2.0, 52, 55, 0.9, 1);

        typedef PlusTimesSRing<double, double> PTFF;
        double t0, t1;
        
        for(int layers = 16; layers <= 16; layers = layers * 4){
            SpParMat<int64_t, double, SpDCCols < int64_t, double >> X(M);
            SpParMat<int64_t, double, SpDCCols < int64_t, double >> Y(M);
            if(myrank == 0) fprintf(stderr, "Running 2D\n\n");
            int calculatedPhases = CalculateNumberOfPhases<PTFF, double, SpDCCols<int64_t, double>, int64_t >(X, Y, 
                2.0, 1100, 1400, 0.9, 1, 0);
            if(myrank == 0) fprintf(stderr, "Approximately %d phases required\n", calculatedPhases);
            //int phases = calculatedPhases;
            int phases = 16;
            while(phases <= 16){
#ifdef TIMING
                mcl_symbolictime = 0;
                mcl_Abcasttime = 0;
                mcl_Bbcasttime = 0;
                mcl_localspgemmtime = 0;
                mcl_multiwaymergetime = 0;
                mcl_kselecttime = 0;
                mcl_prunecolumntime = 0;
                mcl_totaltime = 0;
                mcl_tt = 0;
                mcl_nnzc = 0;
#endif
                int it; // Number of iterations to run
                for(it = 0; it < 1; it++){
#ifdef TIMING
                    mcl_Abcasttime_prev = mcl_Abcasttime;
                    mcl_Bbcasttime_prev = mcl_Bbcasttime;
                    mcl_localspgemmtime_prev = mcl_localspgemmtime;
                    mcl_multiwaymergetime_prev = mcl_multiwaymergetime;
                    mcl_kselecttime_prev = mcl_kselecttime;
                    mcl_prunecolumntime_prev = mcl_prunecolumntime;
                    mcl_symbolictime_prev = mcl_symbolictime;
                    mcl_totaltime_prev = mcl_totaltime;
                    mcl_tt_prev = mcl_tt;
                    mcl_nnzc_prev = mcl_nnzc;
#endif
#ifdef TIMING
                    MPI_Barrier(MPI_COMM_WORLD);
                    t0 = MPI_Wtime();
#endif
                    SpParMat<int64_t, double, SpDCCols < int64_t, double >> Z4 = MemEfficientSpGEMM<PTFF, double, SpDCCols<int64_t, double>, int64_t >(X, Y, 
                            phases, 2.0, 1100, 1400, 0.9, 1, 0);
#ifdef TIMING
                    MPI_Barrier(MPI_COMM_WORLD);
                    t1 = MPI_Wtime();
                    mcl_totaltime += (t1-t0);
#endif
#ifdef TIMING
                    double g_mcl_tt = 0;
                    MPI_Allreduce(&mcl_tt, &g_mcl_tt, 1, MPI_DOUBLE, MPI_MAX, X.getcommgrid()->GetWorld());
                    mcl_tt = g_mcl_tt;
                    if(myrank == 0){
                        fprintf(stderr, "[2D: Iteration: %d] Symbolictime: %lf\n", it, (mcl_symbolictime - mcl_symbolictime_prev));
                        fprintf(stderr, "[2D: Iteration: %d] Abcasttime: %lf\n", it, (mcl_Abcasttime - mcl_Abcasttime_prev));
                        fprintf(stderr, "[2D: Iteration: %d] Bbcasttime: %lf\n", it, (mcl_Bbcasttime - mcl_Bbcasttime_prev));
                        fprintf(stderr, "[2D: Iteration: %d] LocalSPGEMM: %lf\n", it, (mcl_localspgemmtime - mcl_localspgemmtime_prev));
                        fprintf(stderr, "[2D: Iteration: %d] SUMMAmerge: %lf\n", it, (mcl_multiwaymergetime - mcl_multiwaymergetime_prev));
                        fprintf(stderr, "[2D: Iteration: %d] SelectionRecovery: %lf\n", it, (mcl_kselecttime - mcl_kselecttime_prev + mcl_prunecolumntime - mcl_prunecolumntime_prev));
                        fprintf(stderr, "[2D: Iteration: %d] Total time: %lf\n", it, (mcl_totaltime - mcl_totaltime_prev));
                        fprintf(stderr, "-----------------------------------------------------\n");
                    }
#endif
                }
#ifdef TIMING
                if(myrank == 0){
                    fprintf(stderr, "%lf, ", (mcl_symbolictime/it));
                    fprintf(stderr, " %lf,", (mcl_Abcasttime/it));
                    fprintf(stderr, " %lf,", (mcl_Bbcasttime/it));
                    fprintf(stderr, " %lf,", (mcl_localspgemmtime/it));
                    fprintf(stderr, " %lf,", (mcl_multiwaymergetime/it));
                    fprintf(stderr, " ,");
                    fprintf(stderr, " ,");
                    fprintf(stderr, " %lf,", ((mcl_kselecttime + mcl_prunecolumntime)/it));
                    fprintf(stderr, " %lf\n", (mcl_totaltime/it));
                }
                if(myrank == 0) fprintf(stderr, "====================================================\n\n");
#endif
                if(myrank == 0) fprintf(stderr, "\n\n++++++++++++++++++++++++++++++++++++++++++++\n\n\n\n");

                int ii = 1;
                while(ii <= phases) ii = ii * 2;
                phases = ii;
            }
#ifdef TIMING
            int64_t mcl_flop = EstimateFLOP<PTFF, int64_t, double, double, SpDCCols<int64_t, double>, SpDCCols<int64_t, double> >(X, Y, false, false);
            if(myrank == 0) fprintf(stderr, "mcl_flop %lld\n", mcl_flop);
            if(myrank == 0) fprintf(stderr, "mcl_nnzc %lld\n", mcl_nnzc);
#endif
            if(myrank == 0) fprintf(stderr, "\n\n\n\n********************************************\n\n\n\n\n\n\n");

        }
        
    }
    MPI_Finalize();
    return 0;
}
