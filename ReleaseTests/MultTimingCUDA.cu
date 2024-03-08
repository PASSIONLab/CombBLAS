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

// #include <cuda.h>

#ifdef __CUDACC__

#include <mpi.h>
#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include "CombBLAS/CombBLAS.h"
// #include "../include/GALATIC/source/device/Multiply.cuh"

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

#define ElementType double
#define ITERATIONS 100

// Simple helper class for declarations: Just the numerical type is templated
// The index type and the sequential matrix type stays the same for the whole code
// In this case, they are "int" and "SpDCCols"
template <class NT>
class PSpMat
{
public:
	typedef SpDCCols<uint32_t, NT> DCCols;
	typedef SpParMat<uint32_t, NT, DCCols> MPI_DCCols;
};

// Outline of debug stages
// stage = 0: LocalHybrid does not run/immediately returns
// stage = 1: LocalHybrid mallocs and transposes as needed, but returns immediately after
// stage = 2: LocalHybrid runs the kernel, but does not perform cleanup
// stage = 3: Full run of LocalHybrid
// stages 1 & 2 may lead to memory leaks, be aware on memory limited systems
int main(int argc, char *argv[])
{
#ifdef GPU_ENABLED
// SpParHelper::Print("GPU ENABLED\n");
#endif
	int nprocs, myrank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	if (argc < 4)
	{
		if (myrank == 0)
		{
			cout << "Usage: ./MultTest <MatrixA> <MatrixB> <MatrixC>" << endl;
			cout << "<MatrixA>,<MatrixB>,<MatrixC> are absolute addresses, and files should be in triples format" << endl;
		}
		MPI_Finalize();
		return -1;
	}
	{
		string Aname(argv[1]);
		string Bname(argv[2]);
		string Cname(argv[3]);

		MPI_Barrier(MPI_COMM_WORLD);
		typedef PlusTimesSRing<double, double> PTDOUBLEDOUBLE;
		typedef SelectMaxSRing<bool, int64_t> SR;

		shared_ptr<CommGrid> fullWorld;
		fullWorld.reset(new CommGrid(MPI_COMM_WORLD, 0, 0));

		// construct objects
		PSpMat<double>::MPI_DCCols A(fullWorld);
		PSpMat<double>::MPI_DCCols B(fullWorld);
		PSpMat<double>::MPI_DCCols C(fullWorld);
		PSpMat<double>::MPI_DCCols CControl(fullWorld);

		A.ParallelReadMM(Aname, true, maximum<double>());
#ifndef NOGEMM
		B.ParallelReadMM(Bname, true, maximum<double>());

		CControl.ParallelReadMM(Cname, true, maximum<double>());
#endif
		A.PrintInfo();

#ifndef NOGEMM
		double t3 = MPI_Wtime();
		C = Mult_AnXBn_DoubleBuff_CUDA<PTDOUBLEDOUBLE, double, PSpMat<double>::DCCols>(A, B);
		double t4 = MPI_Wtime();
		std::cout << "Time taken: " << t4 - t3 << std::endl;
		C.PrintInfo();
		if (CControl == C)
		{
			SpParHelper::Print("Double buffered multiplication working correctly\n");
		}
		else
		{
			SpParHelper::Print("ERROR in double CUDA  buffered multiplication, go fix it!\n");
		}
		{ // force the calling of C's destructor
			t3 = MPI_Wtime();
			C = Mult_AnXBn_DoubleBuff<PTDOUBLEDOUBLE, ElementType, PSpMat<ElementType>::DCCols>(A, B);
			t4 = MPI_Wtime();
			std::cout << "Time taken: " << t4 - t3 << std::endl;
			C.PrintInfo();
			if (CControl == C)
			{
				SpParHelper::Print("Double buffered multiplication working correctly\n");
			}
			else
			{
				SpParHelper::Print("ERROR in double non-CUDA  buffered multiplication, go fix it!\n");
			}
			// int64_t cnnz = C.getnnz();
			// ostringstream tinfo;
			// tinfo << "C has a total of " << cnnz << " nonzeros" << endl;
			// SpParHelper::Print(tinfo.str());
			SpParHelper::Print("Warmed up for DoubleBuff\n");
			
		}
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Pcontrol(1, "SpGEMM_DoubleBuff");
		double t1 = MPI_Wtime(); // initilize (wall-clock) timer
		for (int i = 0; i < ITERATIONS; i++)
		{
			C = Mult_AnXBn_DoubleBuff<PTDOUBLEDOUBLE, ElementType, PSpMat<ElementType>::DCCols>(A, B);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		double t2 = MPI_Wtime();
		MPI_Pcontrol(-1, "SpGEMM_DoubleBuff");
		if (myrank == 0)
		{
			cout << "Double buffered multiplications finished" << endl;
			printf("%.6lf seconds elapsed per iteration\n", (t2 - t1) / (double)ITERATIONS);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Pcontrol(1, "SpGEMM_DoubleBuff");
		t1 = MPI_Wtime(); // initilize (wall-clock) timer
		for (int i = 0; i < ITERATIONS; i++)
		{
			C = Mult_AnXBn_DoubleBuff_CUDA<PTDOUBLEDOUBLE, ElementType, PSpMat<ElementType>::DCCols>(A, B);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		t2 = MPI_Wtime();
		MPI_Pcontrol(-1, "SpGEMM_DoubleBuff");
		if (myrank == 0)
		{
			cout << "Double buffered CUDA multiplications finished" << endl;
			printf("%.6lf seconds elapsed per iteration\n", (t2 - t1) / (double)ITERATIONS);
		}
#endif
	}
	MPI_Finalize();
	return 0;
}

#endif