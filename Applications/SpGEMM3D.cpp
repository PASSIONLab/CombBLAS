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
double sym_Abcasttime = 0;
double sym_Bbcasttime = 0;
double sym_estimatefloptime = 0;
double sym_estimatennztime = 0;
double sym_SUMMAnnzreductiontime = 0;
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
double mcl_kselecttime;
double mcl_prunecolumntime;

int64_t mcl3d_layer_nnza;
int64_t mcl3d_nnza;
int64_t mcl3d_proc_flop;
int64_t mcl3d_layer_flop;
int64_t mcl3d_flop;
int64_t mcl3d_proc_nnzc_pre_red;
int64_t mcl3d_layer_nnzc_pre_red;
int64_t mcl3d_nnzc_pre_red;
int64_t mcl3d_proc_nnzc_post_red;
int64_t mcl3d_layer_nnzc_post_red;
int64_t mcl3d_nnzc_post_red;
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
        if(myrank == 0){
            fprintf(stderr, "Data: %s\n", argv[1]);
        }
        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );
        
        double t0, t1;

        SpParMat<int64_t, double, SpDCCols < int64_t, double >> M(fullWorld);

        //// Read matrix market files
        //t0 = MPI_Wtime();
        //M.ParallelReadMM(Aname, true, maximum<double>());
        //t1 = MPI_Wtime();
        //if(myrank == 0) fprintf(stderr, "Time taken to read file: %lf\n", t1-t0);
        //t0 = MPI_Wtime();
        //FullyDistVec<int64_t, int64_t> p( M.getcommgrid() );
        //FullyDistVec<int64_t, int64_t> q( M.getcommgrid() );
        //p.iota(M.getnrow(), 0);
        //q.iota(M.getncol(), 0);
        //p.RandPerm();
        //q.RandPerm();
        //(M)(p,q,true);// in-place permute to save memory
        //t1 = MPI_Wtime();
        //if(myrank == 0) fprintf(stderr, "Time taken to permuatate input: %lf\n", t1-t0);
        
        // Read labelled triple files
        t0 = MPI_Wtime();
        M.ReadGeneralizedTuples(Aname, maximum<double>());
        t1 = MPI_Wtime();
        if(myrank == 0) fprintf(stderr, "Time taken to read file: %lf\n", t1-t0);
        
        //// Sparsify the matrix
        //MCLPruneRecoverySelect<int64_t, double, SpDCCols < int64_t, double >>(M, 2.0, 52, 55, 0.9, 1);

        typedef PlusTimesSRing<double, double> PTFF;
    
        // Run 2D multiplication to compare against
        SpParMat<int64_t, double, SpDCCols < int64_t, double >> A2D(M);
        SpParMat<int64_t, double, SpDCCols < int64_t, double >> B2D(M);
        SpParMat<int64_t, double, SpDCCols < int64_t, double >> C2D = 
            Mult_AnXBn_Synch<PTFF, double, SpDCCols<int64_t, double>, int64_t, double, double, SpDCCols<int64_t, double>, SpDCCols<int64_t, double> >
            (A2D, B2D);
        
        // Increase number of layers 1 -> 4 -> 16
        for(int layers = 1; layers <= 16; layers = layers * 4){
            // Create two copies of input matrix which would be used in multiplication
            SpParMat<int64_t, double, SpDCCols < int64_t, double >> A2(M);
            SpParMat<int64_t, double, SpDCCols < int64_t, double >> B2(M);

            // Convert 2D matrices to 3D
            SpParMat3D<int64_t, double, SpDCCols < int64_t, double >> A3D(A2, layers, true, false);
            SpParMat3D<int64_t, double, SpDCCols < int64_t, double >> B3D(B2, layers, false, false);
            if(myrank == 0) fprintf(stderr, "Running 3D with %d layers\n", layers);

            /**/
            int phases = 1;
            while(phases <= 1){
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
#endif
#ifdef TIMING
                    MPI_Barrier(MPI_COMM_WORLD);
                    t0 = MPI_Wtime();
#endif
                    SpParMat3D<int64_t, double, SpDCCols < int64_t, double >> C3D = 
                        Mult_AnXBn_SUMMA3D<PTFF, double, SpDCCols<int64_t, double>, int64_t, double, double, SpDCCols<int64_t, double>, SpDCCols<int64_t, double> >
                        (A3D, B3D);
#ifdef TIMING
                    MPI_Barrier(MPI_COMM_WORLD);
                    t1 = MPI_Wtime();
                    mcl3d_totaltime += (t1-t0);
#endif

                    SpParMat<int64_t, double, SpDCCols < int64_t, double >> C3D2D = C3D.Convert2D();
                    if(C2D == C3D2D){
                        if(myrank == 0) fprintf(stderr, "Correct!\n");
                    }
                    else{
                        if(myrank == 0) fprintf(stderr, "Not correct!\n");
                    }
#ifdef TIMING
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
                int ii = 1;
                while(ii <= phases) ii = ii * 2;
                phases = ii;
            }
            if(myrank == 0) fprintf(stderr, "\n\n********************************************\n\n");
            /**/
        }
        
    }
    MPI_Finalize();
    return 0;
}
