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
		string ITERS(argv[1]);
		string COMMTEST(argv[2]);
		string Aname(argv[3]);
		string Bname(argv[4]);
		string Cname(argv[5]);

		if(myrank == 0 || nprocs == 1) {
		std::cout << Aname << std::endl;
		std::cout << Bname << std::endl;
		std::cout << Cname << std::endl;
		std::cout << nprocs << std::endl;
		std::string filename = "output" + Aname.substr(0, Aname.length() - 4) + ".txt";

		FILE *f = fopen(filename.c_str(), "a");
			if(f==NULL){printf("failed to open file: permission issue ?\n");exit(1);}
			// cout << "Double buffered CUDA multiplications finished" << endl;
			fprintf(f, "Input A: %s, with NPROCS: %i\n", Aname.c_str(), nprocs);
			fclose(f);
		}
		ITERATIONS = std::stoi(ITERS);

		bool COMMTESTON = std::stoi(COMMTEST) > 0;
		//if(!COMMTESTON) GPUTradeoff = 1024 * 100 * 500;
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
		//A.PrintInfo();

#ifndef NOGEMM
		double t3 = MPI_Wtime();
		C = Mult_AnXBn_DoubleBuff_CUDA<PTDOUBLEDOUBLE, double, PSpMat<double>::DCCols>(A, B);
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGetLastError());

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
		//std::cout << "MAX DIFF = " << maxdiff << std::endl;
		//std::cout << a << std::endl;
		//std::cout << b << std::endl;
		double t4 = MPI_Wtime();
		//std::cout << "Time taken: " << t4 - t3 << std::endl;
		C.PrintInfo();
		cudaDeviceSynchronize();
		if (CControl == C)
		{
		//	SpParHelper::Print("Double buffered multiplication working correctly\n");
		}
		else
		{
			SpParHelper::Print("ERROR in double CUDA  buffered multiplication, go fix it!\n");
		}
		{ // force the calling of C's destructor
			t3 = MPI_Wtime();
			C = Mult_AnXBn_DoubleBuff<PTDOUBLEDOUBLE, ElementType, PSpMat<ElementType>::DCCols>(A, B);
			t4 = MPI_Wtime();
			//std::cout << "Time taken: " << t4 - t3 << std::endl;
			C.PrintInfo();
			if (CControl == C)
			{
			//	SpParHelper::Print("Double buffered multiplication working correctly\n");
			}
			else
			{
				SpParHelper::Print("ERROR in double non-CUDA  buffered multiplication, go fix it!\n");
			}
			// int64_t cnnz = C.getnnz();
			// ostringstream tinfo;
			// tinfo << "C has a total of " << cnnz << " nonzeros" << endl;
			// SpParHelper::Print(tinfo.str());
			//SpParHelper::Print("Warmed up for DoubleBuff\n");
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
		if (myrank == 0 || nprocs == 1)
			{
				std::string filename = "output" + Aname.substr(0,Aname.length() - 4) + ".txt";
				//std::cout << filename.c_str() << std::endl;
				FILE *f = fopen(filename.c_str(), "a");
				if(f==NULL){printf("failed to open file: permission issue ?\n");exit(1);}
				// cout << "Double buffered CUDA multiplications finished" << endl;
				fprintf(f, "CPU Time: %.6lf\n", (t2 - t1) / ((double) ITERATIONS));
				fclose(f);
			}
			int maxhits = 0;
		for (int j = 0; j < 500; ++j)
		{
			//if(!COMMTESTON) j = 500;
			//std::cout << j << std::endl;
			size_t free, total;
			int id;
			MPI_Comm_rank(MPI_COMM_WORLD, &id);
			cudaMemGetInfo(&free, &total);
			// cout << "GPU " << id << " memory: free=" << free << ", total=" << total << endl;

			commtime = 0;
			comms = 0;
			datahits = 0;
			rowshits = 0;
			colhits = 0;
			cudaDeviceSynchronize();
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Pcontrol(1, "SpGEMM_DoubleBuff");
			{
				C = Mult_AnXBn_DoubleBuff_CUDA<PTDOUBLEDOUBLE, double, PSpMat<double>::DCCols>(A, B);
			}
			
			int svdhits = datahits + rowshits + colhits;
			int commper = comms;
			comms = 0;
			datahits = 0;
			rowshits = 0;
			colhits = 0;
			GPUTradeoff = 1024 * 100 * j;
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Pcontrol(1, "SpGEMM_DoubleBuff");
			{
				C = Mult_AnXBn_DoubleBuff_CUDA<PTDOUBLEDOUBLE, double, PSpMat<double>::DCCols>(A, B);
			}
			
			bool allt;
			int nnprocs;
			MPI_Comm_size(MPI_COMM_WORLD, &nnprocs);
			int newhits = datahits + rowshits + colhits;
			if (myrank == 0) {
				for(int i = 1; i < nnprocs; ++i) {
					MPI_Status idc;
					int recv;
					MPI_Recv(&recv, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &idc);
					svdhits += recv;
					MPI_Recv(&recv, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &idc);
					newhits += recv;
				}
			} else {
				MPI_Send(&svdhits, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
				MPI_Send(&newhits, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
			}
			allt = j > 0 && svdhits == newhits;
			if(j == 0) maxhits = newhits;
			MPI_Bcast(&allt, 1, MPI_INT, 0, MPI_COMM_WORLD);
			if(allt) {
				continue;
			}
			comms = 0;
			datahits = 0;
			rowshits = 0;
			colhits = 0;
			commtime = 0;
			comptime = 0;
			checkingTime = 0;
			// std::cout << "Running with tradeoff of " << 100 * j << "KB" << std::endl;
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Pcontrol(1, "SpGEMM_DoubleBuff");
			t1 = MPI_Wtime(); // initilize (wall-clock) timer
			
			for (int i = 0; i < ITERATIONS; i++)
			{
				// std::cout << "--------------NEW ITER------------" << std::endl;
				C = Mult_AnXBn_DoubleBuff_CUDA<PTDOUBLEDOUBLE, double, PSpMat<double>::DCCols>(A, B);
			}
			MPI_Barrier(MPI_COMM_WORLD);
			t2 = MPI_Wtime();
			MPI_Pcontrol(-1, "SpGEMM_DoubleBuff");
			commper = 3 * nnprocs * nnprocs;
			if (myrank == 0 || nprocs == 1)
			{
				std::string filename = "output" + Aname.substr(0,Aname.length() - 4) + ".txt";
				//std::cout << filename.c_str() << std::endl;
				FILE *f = fopen(filename.c_str(), "a");
				if(f==NULL){printf("failed to open file: permission issue ?\n");exit(1);}
				// cout << "Double buffered CUDA multiplications finished" << endl;
				printf("%i,%i,%i,%.6lf,%.6lf,%.6lf,%.6lf\n", GPUTradeoff / 1024, newhits,maxhits, (t2 - t1) / (double)ITERATIONS, (commtime) / (double)ITERATIONS,comptime / (double) ITERATIONS, checkingTime / (double) ITERATIONS);
				fprintf(f, "%i,%i,%i,%.6lf,%.6lf,%.6lf\n", GPUTradeoff / 1024, newhits,maxhits, (t2 - t1) / (double)ITERATIONS, (commtime) / (double)ITERATIONS,comptime / (double) ITERATIONS);
				fclose(f);
			}
			if(!COMMTESTON) break;
			if(!newhits) break;
			if(nprocs == 1) break;
		}
#endif
	}
	MPI_Finalize();
	return 0;
}

#endif