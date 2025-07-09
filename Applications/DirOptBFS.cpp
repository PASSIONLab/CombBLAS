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


#define DETERMINISTIC
#define BOTTOMUPTIME
#include <mpi.h>
#include <sys/time.h> 
#include <iostream>
#include <iomanip>
#include <functional>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>
#ifdef THREADED
	#ifndef _OPENMP
	#define _OPENMP
	#endif
	#include <omp.h>
#endif

// These macros should be defined before stdint.h is included
#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif
#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
#endif
#include <stdint.h>

double cblas_alltoalltime;
double cblas_allgathertime;
double cblas_mergeconttime;
double cblas_transvectime;
double cblas_localspmvtime;
double cblas_ewisemulttime;

double bottomup_sendrecv;
double bottomup_allgather;
double bottomup_total;
double bottomup_convert;

double bu_local;
double bu_update;
double bu_rotate;
int cblas_splits;


#include "CombBLAS/CombBLAS.h"

using namespace combblas;
using namespace std;

#define ITERS 64
#define EDGEFACTOR 16

// 64-bit floor(log2(x)) function 
// note: least significant bit is the "zeroth" bit
// pre: v > 0
unsigned int highestbitset(uint64_t v)
{
	// b in binary is {10,1100, 11110000, 1111111100000000 ...}  
	const uint64_t b[] = {0x2ULL, 0xCULL, 0xF0ULL, 0xFF00ULL, 0xFFFF0000ULL, 0xFFFFFFFF00000000ULL};
	const unsigned int S[] = {1, 2, 4, 8, 16, 32};
	int i;

	unsigned int r = 0; // result of log2(v) will go here
	for (i = 5; i >= 0; i--) 
	{
		if (v & b[i])	// highestbitset is on the left half (i.e. v > S[i] for sure)
		{
			v >>= S[i];
			r |= S[i];
		} 
	}
	return r;
}

template <class T>
bool from_string(T & t, const string& s, ios_base& (*f)(ios_base&))
{
        istringstream iss(s);
        return !(iss >> f >> t).fail();
}


template <typename PARMAT>
void Symmetricize(PARMAT & A)
{
	// boolean addition is practically a "logical or"
	// therefore this doesn't destruct any links
	PARMAT AT = A;
	AT.Transpose();
	A += AT;
}

/**
 * Binary function to prune the previously discovered vertices from the current frontier 
 * When used with EWiseApply(SparseVec V, DenseVec W,...) we get the 'exclude = false' effect of EWiseMult
**/
struct prunediscovered
{
  	int64_t operator()(int64_t x, const int64_t & y) const
	{
		return ( y == -1 ) ? x: -1;
	}
};

int main(int argc, char* argv[])
{
#ifdef THREADED
    cblas_splits = omp_get_max_threads(); 
#else
    cblas_splits = 1;
#endif


    int nprocs, myrank;
#ifdef _OPENMP
    int provided, flag, claimed;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided );
    MPI_Is_thread_main( &flag );
    if (!flag)
        SpParHelper::Print("This thread called init_thread but Is_thread_main gave false\n");
    MPI_Query_thread( &claimed );
    if (claimed != provided)
        SpParHelper::Print("Query thread gave different thread level than requested\n");
#else
	MPI_Init(&argc, &argv);
#endif
    
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
	if(argc < 2)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./dobfs <Scale>" << endl;
			cout << "Example: ./dobfs 25" << endl;
		}
		MPI_Finalize();
		return -1;
	}	
	{
		typedef SpParMat < int64_t, bool, SpDCCols<int64_t,bool> > PSpMat_Bool;
		typedef SpParMat < int64_t, bool, SpDCCols<int32_t,bool> > PSpMat_s32p64;	// sequentially use 32-bits for local matrices, but parallel semantics are 64-bits
		typedef SpParMat < int64_t, int, SpDCCols<int32_t,int> > PSpMat_s32p64_Int;	// similarly mixed, but holds integers as upposed to booleans

		// Declare objects
		PSpMat_Bool A;	
		PSpMat_s32p64 Aeff;
		PSpMat_s32p64 ALocalT;
		shared_ptr<CommGrid> fullWorld;
		fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );
		FullyDistVec<int64_t, int64_t> degrees(fullWorld);	// degrees of vertices (including multi-edges and self-loops)
		FullyDistVec<int64_t, int64_t> nonisov(fullWorld);	// id's of non-isolated (connected) vertices
		unsigned scale;
		OptBuf<int32_t, int64_t> optbuf;	// let indices be 32-bits

		scale = static_cast<unsigned>(atoi(argv[1]));
		ostringstream outs;
		outs << "Forcing scale to : " << scale << endl;
		SpParHelper::Print(outs.str());

		SpParHelper::Print("Using fast vertex permutations; skipping edge permutations (like v2.1)\n");	
		
		// this is an undirected graph, so A*x does indeed BFS
		double initiator[4] = {.57, .19, .19, .05};

		double t01 = MPI_Wtime();
		double t02;
		DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>();
		DEL->GenGraph500Data(initiator, scale, EDGEFACTOR, true, true );	// generate packed edges
		SpParHelper::Print("Generated renamed edge lists\n");
		t02 = MPI_Wtime();
		ostringstream tinfo;
		tinfo << "Generation took " << t02-t01 << " seconds" << endl;
		SpParHelper::Print(tinfo.str());
	

		// Start Kernel #1
		MPI_Barrier(MPI_COMM_WORLD);
		double t1 = MPI_Wtime();

		// conversion from distributed edge list, keeps self-loops, sums duplicates
		PSpMat_s32p64_Int * G = new PSpMat_s32p64_Int(*DEL, false); 
		delete DEL;	// free memory before symmetricizing
		SpParHelper::Print("Created Sparse Matrix (with int32 local indices and values)\n");

		MPI_Barrier(MPI_COMM_WORLD);
		double redts = MPI_Wtime();
		G->Reduce(degrees, Row, plus<int64_t>(), static_cast<int64_t>(0));	// Identity is 0 
		MPI_Barrier(MPI_COMM_WORLD);
		double redtf = MPI_Wtime();

		ostringstream redtimeinfo;
		redtimeinfo << "Calculated degrees in " << redtf-redts << " seconds" << endl;
		SpParHelper::Print(redtimeinfo.str());
		A =  PSpMat_Bool(*G);			// Convert to Boolean
		delete G;
		int64_t removed  = A.RemoveLoops();

		ostringstream loopinfo;
		loopinfo << "Converted to Boolean and removed " << removed << " loops" << endl;
		SpParHelper::Print(loopinfo.str());
		A.PrintInfo();

		FullyDistVec<int64_t, int64_t> * ColSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid());
		FullyDistVec<int64_t, int64_t> * RowSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid());
		A.Reduce(*ColSums, Column, plus<int64_t>(), static_cast<int64_t>(0)); 	
		A.Reduce(*RowSums, Row, plus<int64_t>(), static_cast<int64_t>(0)); 	
		SpParHelper::Print("Reductions done\n");
		ColSums->EWiseApply(*RowSums, plus<int64_t>());
		SpParHelper::Print("Intersection of colsums and rowsums found\n");
		delete RowSums;
    	// only the indices of non-isolated vertices
		nonisov = ColSums->FindInds([](int64_t val){return val > 0;});
		delete ColSums;

		nonisov.RandPerm();	// so that A(v,v) is load-balanced (both memory and time wise)
		SpParHelper::Print("Found non-isolated vertices\n");	
		A.PrintInfo();
		
#ifndef NOPERMUTE
		A(nonisov, nonisov, true);	// in-place permute to save memory	
		SpParHelper::Print("Dropped isolated vertices from input\n");	
		A.PrintInfo();
#endif

		Aeff = PSpMat_s32p64(A);	// Convert to 32-bit local integers
		A.FreeMemory();
		SpParHelper::Print("Converted to 32-bit integers\n");	
		
		Symmetricize(Aeff);	// A += A';
		SpParHelper::Print("Symmetricized\n");	
		
		Aeff.OptimizeForGraph500(optbuf);		// Should be called before threading is activated
		ALocalT = PSpMat_s32p64(Aeff.seq().TransposeConstPtr(), Aeff.getcommgrid());	// this should be copied before the threading is activated
	#ifdef THREADED
		tinfo << "Threading activated with " << cblas_splits << " threads" << endl;
		SpParHelper::Print(tinfo.str());
		Aeff.ActivateThreading(cblas_splits);	
	#endif
		Aeff.PrintInfo();
			
		MPI_Barrier(MPI_COMM_WORLD);
		double t2=MPI_Wtime();
			
		ostringstream k1timeinfo;
		k1timeinfo << (t2-t1) - (redtf-redts) << " seconds elapsed for Kernel #1" << endl;
		SpParHelper::Print(k1timeinfo.str());

		Aeff.PrintInfo();
		float balance = Aeff.LoadImbalance();
		ostringstream lbout;
		lbout << "Load balance: " << balance << endl;
		SpParHelper::Print(lbout.str());

		MPI_Barrier(MPI_COMM_WORLD);
		t1 = MPI_Wtime();

		// Now that every remaining vertex is non-isolated, randomly pick ITERS many of them as starting vertices
	#ifndef NOPERMUTE
		degrees = degrees(nonisov);	// fix the degrees array too
		degrees.PrintInfo("Degrees array");
	#endif
		// degrees.DebugPrint();
		FullyDistVec<int64_t, int64_t> Cands(A.getcommgrid(), ITERS, 0);
		double nver = (double) degrees.TotalLength();
		
	#ifdef DETERMINISTIC
		uint64_t seed = 1383098845;
	#else
		uint64_t seed= time(NULL);
	#endif
		MTRand M(seed);	// generate random numbers with Mersenne Twister 
		
		vector<double> loccands(ITERS);
		vector<int64_t> loccandints(ITERS);
		if(myrank == 0)
		{
			for(int i=0; i<ITERS; ++i) {
				loccands[i] = M.rand();
			}
			copy(loccands.begin(), loccands.end(), ostream_iterator<double>(cout," ")); cout << endl;
			transform(loccands.begin(), loccands.end(), loccands.begin(), [nver](double val){return val * nver;});
			
			for(int i=0; i<ITERS; ++i) {
				loccandints[i] = static_cast<int64_t>(loccands[i]);
			}
			copy(loccandints.begin(), loccandints.end(), ostream_iterator<double>(cout," ")); cout << endl;
		}

		MPI_Bcast(&(loccandints[0]), ITERS, MPIType<int64_t>(),0,MPI_COMM_WORLD);
		for(int i=0; i<ITERS; ++i)
		{
			Cands.SetElement(i,loccandints[i]);
		}

		#define MAXTRIALS 1
		for(int trials =0; trials < MAXTRIALS; trials++)	// try different algorithms for BFS if MAXTRIALS > 1
		{
			cblas_allgathertime = 0;
			cblas_alltoalltime = 0;
			cblas_mergeconttime = 0;
			cblas_transvectime = 0;
			cblas_localspmvtime = 0;
			cblas_ewisemulttime = 0;
			bottomup_sendrecv = 0;
			bottomup_allgather  = 0;
			bottomup_total = 0;
			bottomup_convert = 0;
			
			bu_local = 0;
			bu_update = 0;
			bu_rotate = 0;

			MPI_Pcontrol(1,"BFS");

			double MTEPS[ITERS]; double INVMTEPS[ITERS]; double TIMES[ITERS]; double EDGES[ITERS];

			for(int i=0; i<ITERS; ++i)
			{
				SpParHelper::Print("A BFS iteration is starting\n");
				
				// FullyDistVec ( shared_ptr<CommGrid> grid, IT globallen, NT initval);
				FullyDistVec<int64_t, int64_t> parents ( Aeff.getcommgrid(), Aeff.getncol(), (int64_t) -1);	// identity is -1

				// FullyDistSpVec ( shared_ptr<CommGrid> grid, IT glen);
				FullyDistSpVec<int64_t, int64_t> fringe(Aeff.getcommgrid(), Aeff.getncol());	// numerical values are stored 0-based
					
				ostringstream devout;
				devout.setf(ios::fixed);

				MPI_Barrier(MPI_COMM_WORLD);
				double t1 = MPI_Wtime();

				int64_t num_edges = Aeff.getnnz();
				int64_t num_nodes = Aeff.getncol();
				int64_t up_cutoff = num_edges / 20;
				int64_t down_cutoff = (((double) num_nodes) * ((double)num_nodes)) / ((double) num_edges * 12.0);

				devout << "param " << num_nodes << " vertices with " << num_edges << " edges" << endl;
				devout << up_cutoff << " up and " << down_cutoff << " down" << endl;

				fringe.SetElement(Cands[i], Cands[i]);
				parents.SetElement(Cands[i], Cands[i]);
				int iterations = 0;

				BitMapFringe<int64_t,int64_t> bm_fringe(fringe.getcommgrid(), fringe);
				BitMapCarousel<int64_t,int64_t> done(Aeff.getcommgrid(), parents.TotalLength(), bm_fringe.GetSubWordDisp());
				SpDCCols<int,bool>::SpColIter *starts = CalcSubStarts(ALocalT, fringe, done);
				int64_t fringe_size = fringe.getnnz();
				int64_t last_fringe_size = 0;
				double pred_start = MPI_Wtime();
				fringe.Apply(myset<int64_t>(1));
				int64_t pred = EWiseMult(fringe, degrees, false, (int64_t) 0).Reduce(plus<int64_t>(), (int64_t) 0);
				double pred_end = MPI_Wtime();
				devout << "  s" << setw(15) << pred << setw(15) << setprecision(5) << (pred_end - pred_start) << endl;
				cblas_ewisemulttime += (pred_end - pred_start); 

				while(fringe_size > 0) 
				{
					if ((pred > up_cutoff) && (last_fringe_size < fringe_size)) 
					{   // Bottom-up
						MPI_Barrier(MPI_COMM_WORLD);
						double conv_start = MPI_Wtime();
						done.LoadVec(parents);
						bm_fringe.LoadFromSpVec(fringe);
						double conv_end = MPI_Wtime();
						devout << "  c" << setw(30) << setprecision(5) << (conv_end - conv_start) << endl;
						bottomup_convert += (conv_end - conv_start);

						while (fringe_size > 0) 		
						{
							double step_start = MPI_Wtime();
							BottomUpStep(ALocalT, fringe, bm_fringe, parents, done, starts);
							double step_end = MPI_Wtime();

							devout << setw(2) << iterations << "u" << setw(15) << fringe_size << setprecision(5) << setw(15) << (step_end-step_start) << endl;
							bottomup_total += (step_end-step_start);
							iterations++;
							last_fringe_size = fringe_size;
							fringe_size = bm_fringe.GetNumSet();
							if ((fringe_size < down_cutoff) && (last_fringe_size > fringe_size)) 
							{
								conv_start = MPI_Wtime();
								bm_fringe.UpdateSpVec(fringe);
								conv_end = MPI_Wtime();
								devout << "  c" << setw(30) << setprecision(5) << (conv_end - conv_start) << endl;
								bottomup_convert += (conv_end - conv_start);
								break;
							}
						}
					} 
					else 
					{   // Top-down
						double step_start = MPI_Wtime();
						fringe.setNumToInd();
						fringe = SpMV(Aeff, fringe,optbuf);
						double ewise_start = MPI_Wtime();
						fringe = EWiseMult(fringe, parents, true, (int64_t) -1);
						parents.Set(fringe);
						double step_end = MPI_Wtime();
						devout << setw(2) << iterations << "d" << setw(15) << fringe.getnnz() << setw(15) << setprecision(5) << (step_end-step_start) << endl;
						cblas_ewisemulttime += (step_end - ewise_start); 

						pred_start = MPI_Wtime();
						fringe.Apply(myset<int64_t>(1));
						pred = EWiseMult(fringe, degrees, false, (int64_t) 0).Reduce(plus<int64_t>(), (int64_t) 0);
						pred_end = MPI_Wtime();
						devout << "  s" << setw(15) << pred << setw(15) << setprecision(5) << (pred_end - pred_start) << endl;
						cblas_ewisemulttime += (pred_end - pred_start); 
						iterations++;
						last_fringe_size = fringe_size;
						fringe_size = fringe.getnnz();
					}
				}
				MPI_Barrier(MPI_COMM_WORLD);
				double t2 = MPI_Wtime();
				delete[] starts;
				SpParHelper::Print(devout.str());
				
				FullyDistSpVec<int64_t, int64_t> parentsp = parents.Find([](int64_t val){return val > -1;});
				parentsp.Apply(myset<int64_t>(1));	
				// we use degrees on the directed graph, so that we don't count the reverse edges in the teps score
				int64_t nedges = EWiseMult(parentsp, degrees, false, (int64_t) 0).Reduce(plus<int64_t>(), (int64_t) 0);
				int64_t nverts = parentsp.Reduce(plus<int64_t>(), (int64_t) 0);
				
				ostringstream outnew;
				outnew << i << "th starting vertex was " << Cands[i] << endl;
				outnew << "Number iterations: " << iterations << endl;
				outnew << "Number of vertices found: " << nverts << endl; 
				outnew << "Number of edges traversed: " << nedges << endl;
				outnew << "BFS time: " << t2-t1 << " seconds" << endl;
				outnew << "MTEPS: " << static_cast<double>(nedges) / (t2-t1) / 1000000.0 << endl;
				outnew << "Total communication (average so far): " << (cblas_allgathertime + cblas_alltoalltime) / (i+1) << endl;
				TIMES[i] = t2-t1;
				EDGES[i] = nedges;
				MTEPS[i] = static_cast<double>(nedges) / (t2-t1) / 1000000.0;
				SpParHelper::Print(outnew.str());
			}
			MPI_Pcontrol(-1,"BFS");
			SpParHelper::Print("Finished\n");
#ifdef TIMING
			double * bu_total, *bu_ag_all, *bu_sr_all, *bu_convert, *td_ag_all, *td_a2a_all, *td_tv_all, *td_mc_all, *td_spmv_all, *td_ewm_all;
			if(myrank == 0)
			{
				bu_total = new double[nprocs];
				bu_ag_all = new double[nprocs];
				bu_sr_all = new double[nprocs];
				bu_convert = new double[nprocs];
				td_ag_all = new double[nprocs];
				td_a2a_all = new double[nprocs];
				td_tv_all = new double[nprocs];
				td_mc_all = new double[nprocs];
				td_spmv_all = new double[nprocs];
				td_ewm_all = new double[nprocs];
			}
			bottomup_allgather /= static_cast<double>(ITERS);
			bottomup_sendrecv /= static_cast<double>(ITERS);
			bottomup_total /= static_cast<double>(ITERS);
			bottomup_convert /= static_cast<double>(ITERS);	// conversion not included in total time
			
			cblas_allgathertime /= static_cast<double>(ITERS);
			cblas_alltoalltime /= static_cast<double>(ITERS);
			cblas_transvectime /= static_cast<double>(ITERS);
			cblas_mergeconttime /= static_cast<double>(ITERS);
			cblas_localspmvtime /= static_cast<double>(ITERS);
			cblas_ewisemulttime /= static_cast<double>(ITERS);
			
			MPI_Gather(&bottomup_convert, 1, MPI_DOUBLE, bu_convert, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Gather(&bottomup_total, 1, MPI_DOUBLE, bu_total, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Gather(&bottomup_allgather, 1, MPI_DOUBLE, bu_ag_all, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Gather(&bottomup_sendrecv, 1, MPI_DOUBLE, bu_sr_all, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Gather(&cblas_allgathertime, 1, MPI_DOUBLE, td_ag_all, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Gather(&cblas_alltoalltime, 1, MPI_DOUBLE, td_a2a_all, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Gather(&cblas_transvectime, 1, MPI_DOUBLE, td_tv_all, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Gather(&cblas_mergeconttime, 1, MPI_DOUBLE, td_mc_all, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Gather(&cblas_localspmvtime, 1, MPI_DOUBLE, td_spmv_all, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Gather(&cblas_ewisemulttime, 1, MPI_DOUBLE, td_ewm_all, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

			double bu_local_total = 0;
			double bu_update_total = 0;
			double bu_rotate_total = 0;

			MPI_Allreduce(&bu_local, &bu_local_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce(&bu_update, &bu_update_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce(&bu_rotate, &bu_rotate_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

			if(myrank == 0)
			{
				cout << "BU Local: " << bu_local_total/nprocs << endl;
				cout << "BU Update: " << bu_update_total/nprocs << endl;
				cout << "BU Rotate: " << bu_rotate_total/nprocs << endl;

				vector<double> total_time(nprocs, 0);
				for(int i=0; i< nprocs; ++i) 				// find the mean performing guy
					total_time[i] += bu_total[i] + bu_convert[i] + td_ag_all[i] +  td_a2a_all[i] + td_tv_all[i] + td_mc_all[i] + td_spmv_all[i] + td_ewm_all[i];
                
				vector<size_t> permutation = SpHelper::find_order(total_time);
				size_t smallest = permutation[0];
				size_t largest = permutation[nprocs-1];
				size_t median = permutation[nprocs/2];
				
				cout << "TOTAL (accounted) MEAN: " << accumulate( total_time.begin(), total_time.end(), 0.0 )/ static_cast<double> (nprocs) << endl;
				cout << "TOTAL (accounted) MAX: " << total_time[0] << endl;
				cout << "TOTAL (accounted) MIN: " << total_time[nprocs-1]  << endl;
				cout << "TOTAL (accounted) MEDIAN: " << total_time[nprocs/2] << endl;
				cout << "-------------------------------" << endl;
				
				cout << "Convert median: " << bu_convert[median] << endl;
				cout << "Bottom-up allgather median: " << bu_ag_all[median] << endl;
				cout << "Bottom-up send-recv median: " << bu_sr_all[median] << endl;
				cout << "Bottom-up compute median: " << bu_total[median] - (bu_ag_all[median] + bu_sr_all[median]) << endl;
				cout << "Top-down allgather median: " << td_ag_all[median] << endl;
				cout << "Top-down all2all median: " << td_a2a_all[median] << endl;
				cout << "Top-down transposevector median: " << td_tv_all[median] << endl;
				cout << "Top-down mergecontributions median: " << td_mc_all[median] << endl;
				cout << "Top-down spmsv median: " << td_spmv_all[median] << endl;
				cout << "-------------------------------" << endl;
				
				cout << "Convert MEAN: " << accumulate( bu_convert, bu_convert+nprocs, 0.0 )/ static_cast<double> (nprocs) << endl;
				cout << "Bottom-up total MEAN: " << accumulate( bu_total, bu_total+nprocs, 0.0 )/ static_cast<double> (nprocs) << endl;
				cout << "Bottom-up allgather MEAN: " << accumulate( bu_ag_all, bu_ag_all+nprocs, 0.0 )/ static_cast<double> (nprocs) << endl;
				cout << "Bottom-up send-recv MEAN: " << accumulate( bu_sr_all, bu_sr_all+nprocs, 0.0 )/ static_cast<double> (nprocs) << endl;
				cout << "Top-down allgather MEAN: " << accumulate( td_ag_all, td_ag_all+nprocs, 0.0 )/ static_cast<double> (nprocs) << endl;
				cout << "Top-down all2all MEAN: " << accumulate( td_a2a_all, td_a2a_all+nprocs, 0.0 )/ static_cast<double> (nprocs) << endl;
				cout << "Top-down transposevector MEAN: " << accumulate( td_tv_all, td_tv_all+nprocs, 0.0 )/ static_cast<double> (nprocs) << endl;
				cout << "Top-down mergecontributions MEAN: " << accumulate( td_mc_all, td_mc_all+nprocs, 0.0 )/ static_cast<double> (nprocs) << endl;
				cout << "Top-down spmsv MEAN: " << accumulate( td_spmv_all, td_spmv_all+nprocs, 0.0 )/ static_cast<double> (nprocs) << endl;
				cout << "-------------------------------" << endl;

				
				cout << "Bottom-up allgather fastest: " << bu_ag_all[smallest] << endl;
				cout << "Bottom-up send-recv fastest: " << bu_sr_all[smallest] << endl;
				cout << "Bottom-up compute fastest: " << bu_total[smallest] - (bu_ag_all[smallest] + bu_sr_all[smallest]) << endl;
				cout << "Top-down allgather fastest: " << td_ag_all[smallest] << endl;
				cout << "Top-down all2all fastest: " << td_a2a_all[smallest] << endl;
				cout << "Top-down transposevector fastest: " << td_tv_all[smallest] << endl;
				cout << "Top-down mergecontributions fastest: " << td_mc_all[smallest] << endl;
				cout << "Top-down spmsv fastest: " << td_spmv_all[smallest] << endl;
				cout << "-------------------------------" << endl;

				
				cout << "Bottom-up allgather slowest: " << bu_ag_all[largest] << endl;
				cout << "Bottom-up send-recv slowest: " << bu_sr_all[largest] << endl;
				cout << "Bottom-up compute slowest: " << bu_total[largest] - (bu_ag_all[largest] + bu_sr_all[largest]) << endl;
				cout << "Top-down allgather slowest: " << td_ag_all[largest] << endl;
				cout << "Top-down all2all slowest: " << td_a2a_all[largest] << endl;
				cout << "Top-down transposevector slowest: " << td_tv_all[largest] << endl;
				cout << "Top-down mergecontributions slowest: " << td_mc_all[largest] << endl;
				cout << "Top-down spmsv slowest: " << td_spmv_all[largest] << endl;
			}
#endif
			ostringstream os;
			sort(EDGES, EDGES+ITERS);
			os << "--------------------------" << endl;
			os << "Min nedges: " << EDGES[0] << endl;
			os << "First Quartile nedges: " << (EDGES[(ITERS/4)-1] + EDGES[ITERS/4])/2 << endl;
			os << "Median nedges: " << (EDGES[(ITERS/2)-1] + EDGES[ITERS/2])/2 << endl;
			os << "Third Quartile nedges: " << (EDGES[(3*ITERS/4) -1 ] + EDGES[3*ITERS/4])/2 << endl;
			os << "Max nedges: " << EDGES[ITERS-1] << endl;
 			double mean = accumulate( EDGES, EDGES+ITERS, 0.0 )/ ITERS;
			vector<double> zero_mean(ITERS);	// find distances to the mean
			transform(EDGES, EDGES+ITERS, zero_mean.begin(), [mean](double val){return val - mean;});
			// self inner-product is sum of sum of squares
			double deviation = inner_product( zero_mean.begin(),zero_mean.end(), zero_mean.begin(), 0.0 );
   			deviation = sqrt( deviation / (ITERS-1) );
   			os << "Mean nedges: " << mean << endl;
			os << "STDDEV nedges: " << deviation << endl;
			os << "--------------------------" << endl;
	
			sort(TIMES,TIMES+ITERS);
			os << "Min time: " << TIMES[0] << " seconds" << endl;
			os << "First Quartile time: " << (TIMES[(ITERS/4)-1] + TIMES[ITERS/4])/2 << " seconds" << endl;
			os << "Median time: " << (TIMES[(ITERS/2)-1] + TIMES[ITERS/2])/2 << " seconds" << endl;
			os << "Third Quartile time: " << (TIMES[(3*ITERS/4)-1] + TIMES[3*ITERS/4])/2 << " seconds" << endl;
			os << "Max time: " << TIMES[ITERS-1] << " seconds" << endl;
 			mean = accumulate( TIMES, TIMES+ITERS, 0.0 )/ ITERS;
			transform(TIMES, TIMES+ITERS, zero_mean.begin(), [mean](double val){return val - mean;});
			deviation = inner_product( zero_mean.begin(),zero_mean.end(), zero_mean.begin(), 0.0 );
   			deviation = sqrt( deviation / (ITERS-1) );
   			os << "Mean time: " << mean << " seconds" << endl;
			os << "STDDEV time: " << deviation << " seconds" << endl;
			os << "--------------------------" << endl;

			sort(MTEPS, MTEPS+ITERS);
			os << "Min MTEPS: " << MTEPS[0] << endl;
			os << "First Quartile MTEPS: " << (MTEPS[(ITERS/4)-1] + MTEPS[ITERS/4])/2 << endl;
			os << "Median MTEPS: " << (MTEPS[(ITERS/2)-1] + MTEPS[ITERS/2])/2 << endl;
			os << "Third Quartile MTEPS: " << (MTEPS[(3*ITERS/4)-1] + MTEPS[3*ITERS/4])/2 << endl;
			os << "Max MTEPS: " << MTEPS[ITERS-1] << endl;
			transform(MTEPS, MTEPS+ITERS, INVMTEPS, safemultinv<double>()); 	// returns inf for zero teps
			double hteps = static_cast<double>(ITERS) / accumulate(INVMTEPS, INVMTEPS+ITERS, 0.0);	
			os << "Harmonic mean of MTEPS: " << hteps << endl;
			transform(INVMTEPS, INVMTEPS+ITERS, zero_mean.begin(),
				[hteps](double val){return val - 1/hteps;});
			deviation = inner_product( zero_mean.begin(),zero_mean.end(), zero_mean.begin(), 0.0 );
   			deviation = sqrt( deviation / (ITERS-1) ) * (hteps*hteps);	// harmonic_std_dev
			os << "Harmonic standard deviation of MTEPS: " << deviation << endl;
			SpParHelper::Print(os.str());
		}
	}
	MPI_Finalize();
	return 0;
}

