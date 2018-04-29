/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.3 -------------------------------------------------*/
/* date: 2/1/2013 ----------------------------------------------*/
/* authors: Aydin Buluc (abuluc@lbl.gov), Adam Lugowski --------*/
/****************************************************************/
/*
 Copyright (c) 2010-, Aydin Buluc
 
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

using namespace std;
using namespace combblas;
#define ITERATIONS 10

// Simple helper class for declarations: Just the numerical type is templated 
// The index type and the sequential matrix type stays the same for the whole code
// In this case, they are "int" and "SpDCCols"
template <class NT>
class PSpMat 
{ 
public: 
	typedef SpDCCols < int, NT > DCCols;
	typedef SpParMat < int, NT, DCCols > MPI_DCCols;
};


int main(int argc, char* argv[])
{
	int nprocs, myrank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

	if(argc < 4)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./Galerkin <Matrix> <S> <STranspose>" << endl;
			cout << "<Matrix>,<S>,<STranspose> are absolute addresses, and files should be in triples format" << endl;
		}
		MPI_Finalize();
		return -1;
	}				
	{
		string Aname(argv[1]);		
		string Sname(argv[2]);
		string STname(argv[3]);		

		MPI_Barrier(MPI_COMM_WORLD);
		typedef PlusTimesSRing<double, double> PTDOUBLEDOUBLE;	

		PSpMat<double>::MPI_DCCols A, S, ST;	// construct objects
		
		A.ReadDistribute(Aname, 0);
		S.ReadDistribute(Sname, 0);
		ST.ReadDistribute(STname, 0);
		SpParHelper::Print("Data read\n");

		// force the calling of C's destructor
		{
			PSpMat<double>::MPI_DCCols C = Mult_AnXBn_DoubleBuff<PTDOUBLEDOUBLE, double, SpDCCols<int,double> >(A, ST);
			PSpMat<double>::MPI_DCCols D = Mult_AnXBn_DoubleBuff<PTDOUBLEDOUBLE, double, SpDCCols<int,double> >(S, C);
			SpParHelper::Print("Warmed up for DoubleBuff (right evaluate)\n");
		}	
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Pcontrol(1,"SpGEMM_DoubleBuff_right");
		double t1 = MPI_Wtime(); 	// initilize (wall-clock) timer
		for(int i=0; i<ITERATIONS; i++)
		{
			PSpMat<double>::MPI_DCCols C = Mult_AnXBn_DoubleBuff<PTDOUBLEDOUBLE, double, SpDCCols<int,double> >(A, ST);
			PSpMat<double>::MPI_DCCols D = Mult_AnXBn_DoubleBuff<PTDOUBLEDOUBLE, double, SpDCCols<int,double> >(S, C);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		double t2 = MPI_Wtime(); 	
		MPI_Pcontrol(-1,"SpGEMM_DoubleBuff_right");
		if(myrank == 0)
		{
			cout<<"Double buffered multiplications (right evaluate) finished"<<endl;	
			printf("%.6lf seconds elapsed per iteration\n", (t2-t1)/(double)ITERATIONS);
		}

		// force the calling of C's destructor
		{
			PSpMat<double>::MPI_DCCols C = Mult_AnXBn_DoubleBuff<PTDOUBLEDOUBLE, double, SpDCCols<int,double> >(S, A);
			PSpMat<double>::MPI_DCCols D = Mult_AnXBn_DoubleBuff<PTDOUBLEDOUBLE, double, SpDCCols<int,double> >(C, ST);
			SpParHelper::Print("Warmed up for DoubleBuff (left evaluate)\n");
		}	
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Pcontrol(1,"SpGEMM_DoubleBuff_left");
		t1 = MPI_Wtime(); 	// initilize (wall-clock) timer
		for(int i=0; i<ITERATIONS; i++)
		{
			PSpMat<double>::MPI_DCCols C = Mult_AnXBn_DoubleBuff<PTDOUBLEDOUBLE, double, SpDCCols<int,double> >(S, A);
			PSpMat<double>::MPI_DCCols D = Mult_AnXBn_DoubleBuff<PTDOUBLEDOUBLE, double, SpDCCols<int,double> >(C, ST);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		t2 = MPI_Wtime(); 	
		MPI_Pcontrol(-1,"SpGEMM_DoubleBuff_left");
		if(myrank == 0)
		{
			cout<<"Double buffered multiplications (left evaluate) finished"<<endl;	
			printf("%.6lf seconds elapsed per iteration\n", (t2-t1)/(double)ITERATIONS);
		}

		// force the calling of C's destructor
		{	
			PSpMat<double>::MPI_DCCols C = Mult_AnXBn_Synch<PTDOUBLEDOUBLE, double, SpDCCols<int,double> >(A, ST);
			PSpMat<double>::MPI_DCCols D = Mult_AnXBn_Synch<PTDOUBLEDOUBLE, double, SpDCCols<int,double> >(S, C);
		}
		SpParHelper::Print("Warmed up for Synch (right evaluate)\n");
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Pcontrol(1,"SpGEMM_Synch_right");
		t1 = MPI_Wtime(); 	// initilize (wall-clock) timer
		for(int i=0; i<ITERATIONS; i++)
		{
			PSpMat<double>::MPI_DCCols C = Mult_AnXBn_Synch<PTDOUBLEDOUBLE, double, SpDCCols<int,double> >(A, ST);
			PSpMat<double>::MPI_DCCols D = Mult_AnXBn_Synch<PTDOUBLEDOUBLE, double, SpDCCols<int,double> >(S, C);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Pcontrol(-1,"SpGEMM_Synch_right");
		t2 = MPI_Wtime(); 	
		if(myrank == 0)
		{
			cout<<"Synchronous multiplications (right evaluate) finished"<<endl;	
			printf("%.6lf seconds elapsed per iteration\n", (t2-t1)/(double)ITERATIONS);
		}

		// force the calling of C's destructor
		{	
			PSpMat<double>::MPI_DCCols C = Mult_AnXBn_Synch<PTDOUBLEDOUBLE, double, SpDCCols<int,double> >(S, A);
			PSpMat<double>::MPI_DCCols D = Mult_AnXBn_Synch<PTDOUBLEDOUBLE, double, SpDCCols<int,double> >(C, ST);
		}
		SpParHelper::Print("Warmed up for Synch (left evaluate)\n");
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Pcontrol(1,"SpGEMM_Synch_left");
		t1 = MPI_Wtime(); 	// initilize (wall-clock) timer
		for(int i=0; i<ITERATIONS; i++)
		{
			PSpMat<double>::MPI_DCCols C = Mult_AnXBn_Synch<PTDOUBLEDOUBLE, double, SpDCCols<int,double> >(S, A);
			PSpMat<double>::MPI_DCCols D = Mult_AnXBn_Synch<PTDOUBLEDOUBLE, double, SpDCCols<int,double> >(C, ST);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Pcontrol(-1,"SpGEMM_Synch_left");
		t2 = MPI_Wtime(); 	
		if(myrank == 0)
		{
			cout<<"Synchronous multiplications (left evaluate) finished"<<endl;	
			printf("%.6lf seconds elapsed per iteration\n", (t2-t1)/(double)ITERATIONS);
		}
	}
	MPI_Finalize();
	return 0;
}

