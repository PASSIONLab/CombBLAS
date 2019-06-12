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

    if(argc < 2)
    {
        if(myrank == 0)
        {
            cout << "Usage: ./<Binary> <MatrixA> " << endl;
        }
        MPI_Finalize(); 
        return -1;
    }				
    {
        string Aname(argv[1]);

        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );

        SpParMat<int64_t,double, SpDCCols < int64_t, double >> A(fullWorld);
        SpParMat<int64_t,double, SpDCCols < int64_t, double >> B(fullWorld);
        SpParMat<int64_t,double, SpDCCols < int64_t, double >> Ap(fullWorld);
        SpParMat<int64_t,double, SpDCCols < int64_t, double >> Bp(fullWorld);
        A.ParallelReadMM(Aname, true, maximum<double>());
        B.ParallelReadMM(Aname, true, maximum<double>());
        Ap.ParallelReadMM(Aname, true, maximum<double>());
        Bp.ParallelReadMM(Aname, true, maximum<double>());

        double t0, t1;

        t0=MPI_Wtime();
        SpParMat3D<int64_t,double, SpDCCols < int64_t, double > > A3D(A, 4, true, true);    // Column split
        MPI_Barrier(MPI_COMM_WORLD);
        t1=MPI_Wtime();
        if(myrank == 0){
            printf("2D->3D Distribution Time: %lf\n", t1-t0);
        }
        printf("myrank: %d\t:\tA2D [%dx%d(%d)]\tA3D [%dx%d(%d)]\n", myrank, 
                Ap.seqptr()->getnrow(), Ap.seqptr()->getncol(), Ap.seqptr()->getnnz(),
                A3D.seqptr()->getnrow(), A3D.seqptr()->getncol(), A3D.seqptr()->getnnz()
                );

        SpParMat3D<int64_t,double, SpDCCols < int64_t, double > > B3D(B, 4, false, true);   // Row split
        t0=MPI_Wtime();
        typedef PlusTimesSRing<double, double> PTFF;
        SpParMat3D<int64_t, double, SpDCCols<int64_t, double> > C3D = A3D.template mult<PTFF>(B3D);
        //A3D.template mult<PTFF>(B3D);
        MPI_Barrier(MPI_COMM_WORLD);
        t1=MPI_Wtime();
        if(myrank == 0){
            printf("3D 1st Multiplication Time: %lf\n", t1-t0);
        }
        //printf("myrank: %d\tA3D: [%dx%d]\tC3D: [%dx%d]\tnnz: %d=%d\n", myrank, 
                //A3D.seqptr()->getnrow(), A3D.seqptr()->getncol(), 
                //C3D.seqptr()->getnrow(), C3D.seqptr()->getncol(), 
                //A3D.seqptr()->getnnz(), C3D.seqptr()->getnnz()
                //);

        t0 = MPI_Wtime();
        SpParMat<int64_t,double, SpDCCols<int64_t, double> > C2D = Mult_AnXBn_DoubleBuff<PTFF, double, SpDCCols < int64_t, double > >(Ap, Bp);
        MPI_Barrier(MPI_COMM_WORLD);
        t1=MPI_Wtime();
        if(myrank == 0){
            printf("2D 1st Multiplication Time: %lf\n", t1-t0);
        }

        //SpParMat<int64_t, double, SpDCCols <int64_t, double> > C3D2D = C3D.Convert2D();
        //bool equal = (C2D == C3D2D);
        //if(myrank == 0){
            //if(equal) printf("Equal\n");
            //else printf("Not Equal\n");
        //}
        //printf("myrank: %d\tC2D: [%dx%d]\tC3D2D: [%dx%d]\tnnz: %d=%d\n", myrank, 
                //C2D.seqptr()->getnrow(), C2D.seqptr()->getncol(), 
                //C3D2D.seqptr()->getnrow(), C3D2D.seqptr()->getncol(), 
                //C2D.seqptr()->getnnz(), C3D2D.seqptr()->getnnz()
                //);

        int C3D_nnz = C3D.getnnz();
        int C2D_nnz = C2D.getnnz();
        if(myrank == 0){
            printf("C3D_nnz: %d C2D_nnz: %d\n", C3D_nnz, C2D_nnz);
        }

        t0=MPI_Wtime();
        C3D = C3D.template mult<PTFF>(B3D);
        MPI_Barrier(MPI_COMM_WORLD);
        t1=MPI_Wtime();
        if(myrank == 0){
            printf("3D 2nd Multiplication Time: %lf\n", t1-t0);
        }

        t0 = MPI_Wtime();
        C2D = Mult_AnXBn_DoubleBuff<PTFF, double, SpDCCols < int64_t, double > >(C2D, Bp);
        MPI_Barrier(MPI_COMM_WORLD);
        t1=MPI_Wtime();
        if(myrank == 0){
            printf("2D 2nd Multiplication Time: %lf\n", t1-t0);
        }

        //SpParMat<int64_t, double, SpDCCols <int64_t, double> > C3D2D = C3D.Convert2D();
        ////equal = (C2D == C3D2D);
        //bool equal = (C2D == C3D2D);
        //if(myrank == 0){
            //if(equal) printf("Equal\n");
            //else printf("Not Equal\n");
        //}

        C3D_nnz = C3D.getnnz();
        C2D_nnz = C2D.getnnz();
        if(myrank == 0){
            printf("C3D_nnz: %d C2D_nnz: %d\n", C3D_nnz, C2D_nnz);
        }
    }
	MPI_Finalize();
	return 0;
}

