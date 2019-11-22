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

#ifdef TIMING
double cblas_alltoalltime;
double cblas_allgathertime;
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

        //M.ParallelReadMM(Aname, true, maximum<double>());
        M.ReadGeneralizedTuples(Aname, maximum<double>());
        SpParMat<int64_t, double, SpDCCols < int64_t, double >> A(M);
        SpParMat3D<int64_t, double, SpDCCols < int64_t, double >> A3D(A, 64, true, false);
        SpParMat<int64_t, double, SpDCCols < int64_t, double >> B(M);
        SpParMat3D<int64_t, double, SpDCCols < int64_t, double >> B3D(B, 64, false, false);

        typedef PlusTimesSRing<double, double> PTFF;

        for(int i = 0; i < 100; i++){
            //SpParMat<int64_t, double, SpDCCols < int64_t, double >> C;
            //C = MemEfficientSpGEMM<PTFF, double, SpDCCols<int64_t, double> >(A, B, 10, 2.0, 1100, 1400, 0.9, 1, 0);
            A3D.template MemEfficientSpGEMM3D<PTFF>(B3D,
                10, 2.0, 1100, 1400, 0.9, 1, 0);
            MPI_Barrier(MPI_COMM_WORLD);
            process_mem_usage(vm_usage, resident_set);
            if(myrank == 0) fprintf(stderr, "VmSize after %dth multiplication %lf %lf\n", i+1, vm_usage, resident_set);
        }

        //SpDCCols<int64_t, double> * Alocal = M.seqptr();
        //for(int i = 0; i < 10000; i++) {
            //SpDCCols<int64_t, double> * pp = new SpDCCols<int64_t, double>(*(Alocal));
            //vector<SpDCCols<int64_t, double>* > vv;
            //pp->ColSplit(10, vv);
            //vector<SpDCCols<int64_t, double>* > vt;
            //for(int j = 0; j < vv.size(); j++) {
                //vt.push_back(vv[j]);
                //vt.push_back(new SpDCCols<int64_t, double>(0, vv[j]->getnrow(), vv[j]->getncol(), 0));
            //}
            //pp->ColConcatenate(vt);
            //vector<SpDCCols<int64_t, double>* >().swap(vv);
            //vector<SpDCCols<int64_t, double>* >().swap(vt);
            //pp->ColSplit(20, vt);
            //for(int j = 0; j < vt.size(); j++) delete vt[j];
            //vector<SpDCCols<int64_t, double>* >().swap(vt);
            //process_mem_usage(vm_usage, resident_set);
            //if(myrank == 0) fprintf(stderr, "VmSize after %dth iteration %lf %lf\n", i+1, vm_usage, resident_set);
        //}
    }
    MPI_Finalize();
    return 0;
}
