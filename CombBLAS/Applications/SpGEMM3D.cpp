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

        //string prefix("3D-stdout-"); 
        //string proc = to_string(myrank); 
        //string filename = prefix + proc;
        //FILE * fp;
        //fp = fopen(filename.c_str(), "w");
        //fclose(fp);

        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );

        SpParMat<int64_t,double, SpDCCols < int64_t, double >> A(fullWorld);
        SpParMat<int64_t,double, SpDCCols < int64_t, double >> B(fullWorld);
        A.ParallelReadMM(Aname, true, maximum<double>());
        B.ParallelReadMM(Aname, true, maximum<double>());
        
        printf("%d\n", A.getnnz());
        double t0, t1;
        
        //fp = fopen(filename.c_str(), "a");
        //fprintf(fp, "---------------------------[COLUMN SPLITTING]----------------------------\n");
        //fclose(fp);

        t0=MPI_Wtime();
        SpParMat3D<int64_t,double, SpDCCols < int64_t, double > > A3D(A, 9, true, true);    // Column split
        MPI_Barrier(MPI_COMM_WORLD);
        t1=MPI_Wtime();
        if(myrank == 0){
            printf("2D->3D Distribution Time: %lf\n", t1-t0);
        }
        SpParMat<int64_t, double, SpDCCols <int64_t, double> > A3D2D = A3D.Convert2D();
        printf("%d\n", A3D2D.getnnz());

        //fp = fopen(filename.c_str(), "a");
        //fprintf(fp, "---------------------------[ROW SPLITTING]----------------------------\n");
        //fclose(fp);

        //SpParMat3D<int64_t,double, SpDCCols < int64_t, double > > B3D(B, 9, false, true);   // Row split
        //t0=MPI_Wtime();
        //typedef PlusTimesSRing<double, double> PTFF;
        //SpParMat<int64_t,double, SpDCCols<int64_t, double> > C3D2D = A3D.template mult<PTFF>(B3D);
        //t1=MPI_Wtime();
        //int bli = C3D2D.getnnz();
        ////printf("[3D] myrank %d\tnrow %d\tncol %d\tnnz %d\n", myrank, C3D2D.seqptr()->getnrow(), C3D2D.seqptr()->getncol(), C3D2D.seqptr()->getnnz());
        //if(myrank == 0){
            //printf("3D Multiplication Time: %lf\n", t1-t0);
            ////printf("[3D] myrank %2d\tnnz %d\n", myrank, bli);
        //}

        //SpParMat<int64_t,double, SpDCCols < int64_t, double >> Ap(fullWorld);
        //SpParMat<int64_t,double, SpDCCols < int64_t, double >> Bp(fullWorld);
        //Ap.ParallelReadMM(Aname, true, maximum<double>());
        //Bp.ParallelReadMM(Aname, true, maximum<double>());
        //t0 = MPI_Wtime();
        //SpParMat<int64_t,double, SpDCCols<int64_t, double> > C2D = Mult_AnXBn_DoubleBuff<PTFF, double, SpDCCols < int64_t, double > >(Ap, Bp);
        //int ibl = C2D.getnnz();
        ////printf("[ohoh] myrank %d\tnrow %d\tncol %d\tnnz %d\n", myrank, C2D.seqptr()->getnrow(), C2D.seqptr()->getncol(), C2D.seqptr()->getnnz());
        //t1 = MPI_Wtime();
        //if(myrank == 0){
            //printf("%lf\n", t1-t0);
            //printf("[ihih] myrank %2d\tnnz %d\n", myrank, ibl);
        //}
        //SpParMat<int64_t,double, SpDCCols < int64_t, double >> Aq(fullWorld);
        //SpParMat<int64_t,double, SpDCCols < int64_t, double >> Bq(fullWorld);
        //Aq.ParallelReadMM(Aname, true, maximum<double>());
        //Bq.ParallelReadMM(Aname, true, maximum<double>());
        //t0 = MPI_Wtime();
        //SpParMat<int64_t,double, SpDCCols<int64_t, double> > Cq2D = Mult_AnXBn_Synch<PTFF, double, SpDCCols < int64_t, double > >(Aq, Bq);
        //int iqbl = Cq2D.getnnz();
        ////printf("[ohoh] myrank %d\tnrow %d\tncol %d\tnnz %d\n", myrank, C2D.seqptr()->getnrow(), C2D.seqptr()->getncol(), C2D.seqptr()->getnnz());
        //t1 = MPI_Wtime();
        //if(myrank == 0){
            //printf("%lf\n", t1-t0);
            //printf("[ohoh] myrank %2d\tnnz %d\n", myrank, iqbl);
        //}
    }
	MPI_Finalize();
	return 0;
}

