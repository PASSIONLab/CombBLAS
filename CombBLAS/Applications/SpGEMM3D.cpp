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
        
        //// Read labelled triple files
        //M.ReadGeneralizedTuples(Aname, maximum<double>());
        
        //// Sparsify the matrix
        //MCLPruneRecoverySelect<int64_t, double, SpDCCols < int64_t, double >>(M, 2.0, 52, 55, 0.9, 1);

        typedef PlusTimesSRing<double, double> PTFF;
        double t0, t1;
        
        for(int layers = 1; layers <= 64; layers = layers * 4){
#ifdef TIMING
            mcl3d_nnzc = 0;
            mcl3d_flop = 0;
#endif
            SpParMat<int64_t, double, SpDCCols < int64_t, double >> A2(M);
            SpParMat<int64_t, double, SpDCCols < int64_t, double >> B2(M);
            SpParMat3D<int64_t, double, SpDCCols < int64_t, double >> A3D(A2, layers, true, false);
            SpParMat3D<int64_t, double, SpDCCols < int64_t, double >> B3D(B2, layers, false, false);
            if(myrank == 0) fprintf(stderr, "Running 3D with %d layers\n\n", layers);

            int calculatedPhases = A3D.template CalculateNumberOfPhases<PTFF>(B3D,
                2.0, 1100, 1400, 0.9, 1, 24.0);
            if(myrank == 0) fprintf(stderr, "Approximately %d phases required\n", calculatedPhases);

            /**/
            //int phases = calculatedPhases;
            mcl3d_max_phase = 1000;
            int phases = 8;
            while(phases <= 64){
#ifdef TIMING
                mcl3d_conversiontime = 0;
                mcl3d_symbolictime = 0;
                mcl3d_Abcasttime = 0;
                mcl3d_Bbcasttime = 0;
                mcl3d_localspgemmtime = 0;
                mcl3d_SUMMAmergetime = 0;
                mcl3d_reductiontime = 0;
                mcl3d_3dmergetime = 0;
                mcl3d_kselecttime = 0;
                mcl3d_totaltime = 0;
                mcl3d_floptime = 0;
                mcl3d_layer_nnzc = 0;
#endif
                int it; // Number of iterations to run
                for(it = 0; it < 1; it++){
#ifdef TIMING
                    mcl3d_conversiontime_prev = mcl3d_conversiontime;
                    mcl3d_symbolictime_prev = mcl3d_symbolictime;
                    mcl3d_Abcasttime_prev = mcl3d_Abcasttime;
                    mcl3d_Bbcasttime_prev = mcl3d_Bbcasttime;
                    mcl3d_SUMMAtime_prev = mcl3d_SUMMAtime;
                    mcl3d_localspgemmtime_prev = mcl3d_localspgemmtime;
                    mcl3d_SUMMAmergetime_prev = mcl3d_SUMMAmergetime;
                    mcl3d_reductiontime_prev = mcl3d_reductiontime;
                    mcl3d_3dmergetime_prev = mcl3d_3dmergetime;
                    mcl3d_kselecttime_prev = mcl3d_kselecttime;
                    mcl3d_totaltime_prev = mcl3d_totaltime;
                    mcl3d_floptime_prev = mcl3d_floptime;
#endif
#ifdef TIMING
                    MPI_Barrier(MPI_COMM_WORLD);
                    t0 = MPI_Wtime();
#endif
                    SpParMat3D<int64_t, double, SpDCCols < int64_t, double >> C3D = A3D.template MemEfficientSpGEMM3D<PTFF>(B3D,
                        phases, 2.0, 1100, 1400, 0.9, 1, 0);
#ifdef TIMING
                    MPI_Barrier(MPI_COMM_WORLD);
                    t1 = MPI_Wtime();
                    mcl3d_totaltime += (t1-t0);
#endif
#ifdef TIMING
                    //mcl3d_symbolictime = g_mcl3d_symbolictime;
                    //mcl3d_Abcasttime = g_mcl3d_Abcasttime;
                    //mcl3d_Bbcasttime = g_mcl3d_Bbcasttime;
                    //mcl3d_SUMMAtime = g_mcl3d_SUMMAtime;
                    //mcl3d_localspgemmtime = g_mcl3d_localspgemmtime;
                    //mcl3d_SUMMAmergetime = g_mcl3d_SUMMAmergetime;
                    //mcl3d_reductiontime = g_mcl3d_reductiontime;
                    //mcl3d_3dmergetime = g_mcl3d_3dmergetime;
                    //mcl3d_kselecttime = g_mcl3d_kselecttime;
                    //mcl3d_totaltime = g_mcl3d_totaltime;
                    //mcl3d_floptime = g_mcl3d_floptime;
                    //mcl3d_layer_nnzc = g_mcl3d_layer_nnzc;

                    if(myrank == 0){
                        fprintf(stderr, "[3D: Iteration: %d] Symbolictime: %lf\n", it, (mcl3d_symbolictime - mcl3d_symbolictime_prev));
                        fprintf(stderr, "[3D: Iteration: %d] Abcasttime: %lf\n", it, (mcl3d_Abcasttime - mcl3d_Abcasttime_prev));
                        fprintf(stderr, "[3D: Iteration: %d] Bbcasttime: %lf\n", it, (mcl3d_Bbcasttime - mcl3d_Bbcasttime_prev));
                        fprintf(stderr, "[3D: Iteration: %d] LocalSPGEMM: %lf\n", it, (mcl3d_localspgemmtime - mcl3d_localspgemmtime_prev));
                        fprintf(stderr, "[3D: Iteration: %d] SUMMAmerge: %lf\n", it, (mcl3d_SUMMAmergetime - mcl3d_SUMMAmergetime_prev));
                        fprintf(stderr, "[3D: Iteration: %d] Reduction: %lf\n", it, (mcl3d_reductiontime - mcl3d_reductiontime_prev));
                        fprintf(stderr, "[3D: Iteration: %d] 3D Merge: %lf\n", it, (mcl3d_3dmergetime - mcl3d_3dmergetime_prev));
                        fprintf(stderr, "[3D: Iteration: %d] SelectionRecovery: %lf\n", it, (mcl3d_kselecttime - mcl3d_kselecttime_prev));
                        fprintf(stderr, "[3D: Iteration: %d] Total time: %lf\n", it, (mcl3d_totaltime - mcl3d_totaltime_prev));
                        fprintf(stderr, "-----------------------------------------------------\n");
                    }
#endif
                }
#ifdef TIMING
                MPI_Allreduce(&mcl3d_symbolictime, &g_mcl3d_symbolictime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                MPI_Allreduce(&mcl3d_Abcasttime, &g_mcl3d_Abcasttime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                MPI_Allreduce(&mcl3d_Bbcasttime, &g_mcl3d_Bbcasttime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                MPI_Allreduce(&mcl3d_localspgemmtime, &g_mcl3d_localspgemmtime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                MPI_Allreduce(&mcl3d_SUMMAmergetime, &g_mcl3d_SUMMAmergetime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                MPI_Allreduce(&mcl3d_reductiontime, &g_mcl3d_reductiontime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                MPI_Allreduce(&mcl3d_3dmergetime, &g_mcl3d_3dmergetime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                MPI_Allreduce(&mcl3d_kselecttime, &g_mcl3d_kselecttime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                MPI_Allreduce(&mcl3d_totaltime, &g_mcl3d_totaltime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

                MPI_Allreduce(&mcl3d_symbolictime, &l_mcl3d_symbolictime, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
                MPI_Allreduce(&mcl3d_Abcasttime, &l_mcl3d_Abcasttime, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
                MPI_Allreduce(&mcl3d_Bbcasttime, &l_mcl3d_Bbcasttime, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
                MPI_Allreduce(&mcl3d_localspgemmtime, &l_mcl3d_localspgemmtime, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
                MPI_Allreduce(&mcl3d_SUMMAmergetime, &l_mcl3d_SUMMAmergetime, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
                MPI_Allreduce(&mcl3d_reductiontime, &l_mcl3d_reductiontime, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
                MPI_Allreduce(&mcl3d_3dmergetime, &l_mcl3d_3dmergetime, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
                MPI_Allreduce(&mcl3d_kselecttime, &l_mcl3d_kselecttime, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
                MPI_Allreduce(&mcl3d_totaltime, &l_mcl3d_totaltime, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

                MPI_Allreduce(&mcl3d_symbolictime, &a_mcl3d_symbolictime, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(&mcl3d_Abcasttime, &a_mcl3d_Abcasttime, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(&mcl3d_Bbcasttime, &a_mcl3d_Bbcasttime, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(&mcl3d_localspgemmtime, &a_mcl3d_localspgemmtime, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(&mcl3d_SUMMAmergetime, &a_mcl3d_SUMMAmergetime, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(&mcl3d_reductiontime, &a_mcl3d_reductiontime, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(&mcl3d_3dmergetime, &a_mcl3d_3dmergetime, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(&mcl3d_kselecttime, &a_mcl3d_kselecttime, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(&mcl3d_totaltime, &a_mcl3d_totaltime, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                a_mcl3d_symbolictime /= A3D.getcommgrid()->GetSize();
                a_mcl3d_Abcasttime /= A3D.getcommgrid()->GetSize();
                a_mcl3d_Bbcasttime /= A3D.getcommgrid()->GetSize();
                a_mcl3d_localspgemmtime /= A3D.getcommgrid()->GetSize();
                a_mcl3d_SUMMAmergetime /= A3D.getcommgrid()->GetSize();
                a_mcl3d_reductiontime /= A3D.getcommgrid()->GetSize();
                a_mcl3d_3dmergetime /= A3D.getcommgrid()->GetSize();
                a_mcl3d_kselecttime /= A3D.getcommgrid()->GetSize();
                a_mcl3d_totaltime /= A3D.getcommgrid()->GetSize();

                if(myrank == 0){
                    fprintf(stderr, "max: ");
                    fprintf(stderr, "%lf,", ((g_mcl3d_symbolictime*phases)/(it*std::min(mcl3d_max_phase, phases))));
                    fprintf(stderr, " %lf,", ((g_mcl3d_Abcasttime*phases)/(it*std::min(mcl3d_max_phase, phases))));
                    fprintf(stderr, " %lf,", ((g_mcl3d_Bbcasttime*phases)/(it*std::min(mcl3d_max_phase, phases))));
                    fprintf(stderr, " %lf,", ((g_mcl3d_localspgemmtime*phases)/(it*std::min(mcl3d_max_phase, phases))));
                    fprintf(stderr, " %lf,", ((g_mcl3d_SUMMAmergetime*phases)/(it*std::min(mcl3d_max_phase, phases))));
                    fprintf(stderr, " %lf,", ((g_mcl3d_reductiontime*phases)/(it*std::min(mcl3d_max_phase, phases))));
                    fprintf(stderr, " %lf,", ((g_mcl3d_3dmergetime*phases)/(it*std::min(mcl3d_max_phase, phases))));
                    fprintf(stderr, " %lf,", ((g_mcl3d_kselecttime*phases)/(it*std::min(mcl3d_max_phase, phases))));
                    fprintf(stderr, " %lf\n", ((g_mcl3d_totaltime*phases)/(it*std::min(mcl3d_max_phase, phases))));

                    fprintf(stderr, "min: ");
                    fprintf(stderr, "%lf,", ((l_mcl3d_symbolictime*phases)/(it*std::min(mcl3d_max_phase, phases))));
                    fprintf(stderr, " %lf,", ((l_mcl3d_Abcasttime*phases)/(it*std::min(mcl3d_max_phase, phases))));
                    fprintf(stderr, " %lf,", ((l_mcl3d_Bbcasttime*phases)/(it*std::min(mcl3d_max_phase, phases))));
                    fprintf(stderr, " %lf,", ((l_mcl3d_localspgemmtime*phases)/(it*std::min(mcl3d_max_phase, phases))));
                    fprintf(stderr, " %lf,", ((l_mcl3d_SUMMAmergetime*phases)/(it*std::min(mcl3d_max_phase, phases))));
                    fprintf(stderr, " %lf,", ((l_mcl3d_reductiontime*phases)/(it*std::min(mcl3d_max_phase, phases))));
                    fprintf(stderr, " %lf,", ((l_mcl3d_3dmergetime*phases)/(it*std::min(mcl3d_max_phase, phases))));
                    fprintf(stderr, " %lf,", ((l_mcl3d_kselecttime*phases)/(it*std::min(mcl3d_max_phase, phases))));
                    fprintf(stderr, " %lf\n", ((l_mcl3d_totaltime*phases)/(it*std::min(mcl3d_max_phase, phases))));

                    fprintf(stderr, "avg: ");
                    fprintf(stderr, "%lf,", ((a_mcl3d_symbolictime*phases)/(it*std::min(mcl3d_max_phase, phases))));
                    fprintf(stderr, " %lf,", ((a_mcl3d_Abcasttime*phases)/(it*std::min(mcl3d_max_phase, phases))));
                    fprintf(stderr, " %lf,", ((a_mcl3d_Bbcasttime*phases)/(it*std::min(mcl3d_max_phase, phases))));
                    fprintf(stderr, " %lf,", ((a_mcl3d_localspgemmtime*phases)/(it*std::min(mcl3d_max_phase, phases))));
                    fprintf(stderr, " %lf,", ((a_mcl3d_SUMMAmergetime*phases)/(it*std::min(mcl3d_max_phase, phases))));
                    fprintf(stderr, " %lf,", ((a_mcl3d_reductiontime*phases)/(it*std::min(mcl3d_max_phase, phases))));
                    fprintf(stderr, " %lf,", ((a_mcl3d_3dmergetime*phases)/(it*std::min(mcl3d_max_phase, phases))));
                    fprintf(stderr, " %lf,", ((a_mcl3d_kselecttime*phases)/(it*std::min(mcl3d_max_phase, phases))));
                    fprintf(stderr, " %lf\n", ((a_mcl3d_totaltime*phases)/(it*std::min(mcl3d_max_phase, phases))));
                }
                if(myrank == 0) fprintf(stderr, "====================================================\n\n");
#endif
                if(myrank == 0) fprintf(stderr, "\n\n++++++++++++++++++++++++++++++++++++++++++++\n\n\n\n");

                int ii = 1;
                while(ii <= phases) ii = ii * 2;
                phases = ii;
            }
#ifdef TIMING
            mcl3d_layer_flop = EstimateFLOP<PTFF, int64_t, double, double, SpDCCols<int64_t, double>, SpDCCols<int64_t, double> >(
                    *(A3D.GetLayerMat()), 
                    *(B3D.GetLayerMat()), 
                    false, false);
            MPI_Allreduce(&mcl3d_layer_flop, &mcl3d_flop, 1, MPI_LONG_LONG_INT, MPI_SUM, A3D.getcommgrid3D()->GetFiberWorld());
            MPI_Allreduce(&mcl3d_layer_nnzc, &mcl3d_nnzc, 1, MPI_LONG_LONG_INT, MPI_SUM, A3D.getcommgrid3D()->GetWorld());
            if(myrank == 0) fprintf(stderr, "mcl3d_layer_flop %lld\n", mcl3d_layer_flop);
            if(myrank == 0) fprintf(stderr, "mcl3d_layer_nnzc %lld\n", mcl3d_layer_nnzc);
            if(myrank == 0) fprintf(stderr, "mcl3d_nnzc %lld\n", mcl3d_nnzc);
            if(myrank == 0) fprintf(stderr, "mcl3d_flop %lld\n", mcl3d_flop);
#endif
            if(myrank == 0) fprintf(stderr, "\n\n\n\n********************************************\n\n\n\n\n\n\n");
            /**/
        }
        
    }
    MPI_Finalize();
    return 0;
}
