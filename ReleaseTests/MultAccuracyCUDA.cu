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
int ITERATIONS = 50;

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
	int host_rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	typedef PlusTimesSRing<ElementType, ElementType> PTDOUBLEDOUBLE;

	if (argc < 3)
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

		if (myrank == 0 || nprocs == 1)
		{
			std::cout << Aname << std::endl;
			std::cout << Bname << std::endl;
		}
		typedef PlusTimesSRing<double, double> MinPlusSRing;
		typedef SelectMaxSRing<bool, int64_t> SR;

		shared_ptr<CommGrid> fullWorld;
		fullWorld.reset(new CommGrid(MPI_COMM_WORLD, 0, 0));

		std::cout << "Constructing objects:" << std::endl;
		// construct objects
		PSpMat<double>::MPI_DCCols A(fullWorld);
		PSpMat<double>::MPI_DCCols B(fullWorld);
		PSpMat<double>::MPI_DCCols C(fullWorld);
		PSpMat<double>::MPI_DCCols CControl(fullWorld);

		A.ParallelReadMM(Aname, true, maximum<double>());
#ifndef NOGEMM
		B.ParallelReadMM(Bname, true, maximum<double>());

#endif
		A.PrintInfo();

#ifndef NOGEMM
		C = Mult_AnXBn_DoubleBuff_CUDA<PTDOUBLEDOUBLE, double, PSpMat<double>::DCCols>(A, B);
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGetLastError());
		C.PrintInfo();
		cudaDeviceSynchronize();
		{
			CControl = Mult_AnXBn_DoubleBuff<PTDOUBLEDOUBLE, ElementType, PSpMat<ElementType>::DCCols>(A, B);
			C.PrintInfo();
			if (CControl == C)
			{
				SpParHelper::Print("Double buffered multiplication working correctly\n");
			}
			else
			{
				SpParHelper::Print("ERROR in double CUDA  buffered multiplication, from CPU!\n");
				A.PrintInfo();
				C.PrintInfo();
				CControl.PrintInfo();
				SpDCCols<uint32_t, double> spdcsc = C.seq();
				Dcsc<uint32_t, double> *dcsc = C.seq().GetDCSC();
				double maxdiff = 0;
				double a = 0;
				double b = 0;
				for (int i = 0; i < spdcsc.getnnz(); ++i)
				{
					if (abs(dcsc->numx[i] - CControl.seq().GetDCSC()->numx[i]) > maxdiff)
					{
						maxdiff = abs(dcsc->numx[i] - CControl.seq().GetDCSC()->numx[i]);
						a = dcsc->numx[i];
						b = CControl.seq().GetDCSC()->numx[i];
					}
				}
				std::cout << "MAX DIFF = " << maxdiff << std::endl;
				std::cout << a << std::endl;
				std::cout << b << std::endl;
			}
		}
	}
#endif

MPI_Finalize();
return 0;
}
#endif