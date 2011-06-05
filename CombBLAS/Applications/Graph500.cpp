#include <mpi.h>
#include <sys/time.h> 
#include <iostream>
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
#ifdef NOTR1
        #include <boost/tr1/memory.hpp>
#else
        #include <tr1/memory>
#endif

//#include "include/pat_api.h"
double cblas_alltoalltime;
double cblas_allgathertime;
#ifdef _OPENMP
int cblas_splits = omp_get_max_threads(); 
#else
int cblas_splits = 1;
#endif


#include "../SpTuples.h"
#include "../SpDCCols.h"
#include "../SpParMat.h"
#include "../FullyDistVec.h"
#include "../FullyDistSpVec.h"
#include "../ParFriends.h"
#include "../DistEdgeList.h"

#define ITERS 16
#define EDGEFACTOR 16
using namespace std;

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
bool from_string(T & t, const string& s, std::ios_base& (*f)(std::ios_base&))
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

int main(int argc, char* argv[])
{
	MPI::Init(argc, argv);
	//MPI::COMM_WORLD.Set_errhandler ( MPI::ERRORS_THROW_EXCEPTIONS );
	int nprocs = MPI::COMM_WORLD.Get_size();
	int myrank = MPI::COMM_WORLD.Get_rank();
	
	if(argc < 3)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./Graph500 <Auto,Force,Input> <Available RAM in MB (per core) | Scale Forced | Input Name>" << endl;
			cout << "Example: ./Graph500 Auto 1024" << endl;
		}
		MPI::Finalize(); 
		return -1;
	}		
	{
		typedef SelectMaxSRing<bool, int64_t> SR;	
		typedef SpParMat < int64_t, bool, SpDCCols<int64_t,bool> > PSpMat_Bool;
		typedef SpParMat < int64_t, int, SpDCCols<int64_t,int> > PSpMat_Int;
		typedef SpParMat < int64_t, int64_t, SpDCCols<int64_t,int64_t> > PSpMat_Int64;
		typedef SpParMat < int32_t, int32_t, SpDCCols<int32_t,int32_t> > PSpMat_Int32;

		// Declare objects
		PSpMat_Bool A;	
		FullyDistVec<int64_t, int64_t> degrees;	// degrees of vertices (including multi-edges and self-loops)
		FullyDistVec<int64_t, int64_t> nonisov;	// id's of non-isolated (connected) vertices
		unsigned scale;
		OptBuf<int64_t, int64_t> optbuf;
		bool scramble = false;

		if(string(argv[1]) == string("Input")) // input option
		{
			ifstream input(argv[2]);
			A.ReadDistribute(input, 0);	// read it from file
			SpParHelper::Print("Read input");

			PSpMat_Int64 * G = new PSpMat_Int64(A); 
			G->Reduce(degrees, Row, plus<int64_t>(), static_cast<int64_t>(0));	// identity is 0 
			delete G;

			Symmetricize(A);	// A += A';
			FullyDistVec<int64_t, int64_t> * ColSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid(), 0);
			A.Reduce(*ColSums, Column, plus<int64_t>(), static_cast<int64_t>(0)); 	// plus<int64_t> matches the type of the output vector
			nonisov = ColSums->FindInds(bind2nd(greater<int64_t>(), 0));	// only the indices of non-isolated vertices
			delete ColSums;
			A = A(nonisov, nonisov);
		}
		else if(string(argv[1]) == string("Binary"))
		{
			uint64_t n, m;
			from_string(n,string(argv[3]),std::dec);
			from_string(m,string(argv[4]),std::dec);
			
			ostringstream outs;
			outs << "Reading " << argv[2] << " with " << n << " vertices and " << m << " edges" << endl;
			SpParHelper::Print(outs.str());
			DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>(argv[2], n, m);
			SpParHelper::Print("Read binary input to distributed edge list\n");

			PermEdges(*DEL);
			SpParHelper::Print("Permuted Edges\n");

			RenameVertices(*DEL);	
			//DEL->Dump32bit("graph_permuted");
			SpParHelper::Print("Renamed Vertices\n");

			// conversion from distributed edge list, keeps self-loops, sums duplicates
			PSpMat_Int64 * G = new PSpMat_Int64(*DEL, false); 
			delete DEL;	// free memory before symmetricizing
			SpParHelper::Print("Created Int64 Sparse Matrix\n");

			G->Reduce(degrees, Row, plus<int64_t>(), static_cast<int64_t>(0));	// Identity is 0 

			A =  PSpMat_Bool(*G);			// Convert to Boolean
			delete G;
			int64_t removed  = A.RemoveLoops();

			ostringstream loopinfo;
			loopinfo << "Converted to Boolean and removed " << removed << " loops" << endl;
			SpParHelper::Print(loopinfo.str());
			A.PrintInfo();

			FullyDistVec<int64_t, int64_t> * ColSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid(), 0);
			FullyDistVec<int64_t, int64_t> * RowSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid(), 0);
			A.Reduce(*ColSums, Column, plus<int64_t>(), static_cast<int64_t>(0)); 	
			A.Reduce(*RowSums, Row, plus<int64_t>(), static_cast<int64_t>(0)); 	
			ColSums->EWiseApply(*RowSums, plus<int64_t>());
			delete RowSums;

			nonisov = ColSums->FindInds(bind2nd(greater<int64_t>(), 0));	// only the indices of non-isolated vertices
			delete ColSums;

			SpParHelper::Print("Found (and permuted) non-isolated vertices\n");	
			nonisov.RandPerm();	// so that A(v,v) is load-balanced (both memory and time wise)
			A.PrintInfo();
			A(nonisov, nonisov, true);	// in-place permute to save memory
			SpParHelper::Print("Dropped isolated vertices from input\n");	
			A.PrintInfo();

			Symmetricize(A);	// A += A';
			SpParHelper::Print("Symmetricized\n");	
			//A.Dump("graph_symmetric");

		#ifdef THREADED	
			ostringstream tinfo;
			tinfo << "Threading activated with " << cblas_splits << " threads" << endl;
			SpParHelper::Print(tinfo.str());
			A.ActivateThreading(cblas_splits);	
		#endif
		}
		else 
		{	
			if(string(argv[1]) == string("Auto"))	
			{
				// calculate the problem size that can be solved
				// number of nonzero columns are at most the matrix dimension (for small p)
				// for large p, though, nzc = nnz since each subcolumn will have a single nonzero 
				// so assume (1+8+8+8)*nedges for the uint64 case and (1+4+4+4)*nedges for uint32
				uint64_t raminbytes = static_cast<uint64_t>(atoi(argv[2])) * 1024 * 1024;	
				uint64_t peredge = 1+3*sizeof(int64_t);
				uint64_t maxnedges = raminbytes / peredge;
				uint64_t maxvertices = maxnedges / 32;	
				unsigned maxscale = highestbitset(maxvertices * nprocs);

				string name;
				if(maxscale > 36)	// at least 37 so it fits comfortably along with vectors 
				{
					name = "Medium";	
					scale = 36;
				}
				else if(maxscale > 32)
				{
					name = "Small";
					scale = 32;
				}
				else if(maxscale > 29)
				{
					name = "Mini";
					scale = 29;
				}
				else if(maxscale > 26)
				{
					name = "Toy";
					scale = 26;
				}
				else
				{
					name = "Debug";
					scale = 20;	// fits even to single processor
				}

				ostringstream outs;
				outs << "Max scale allowed : " << maxscale << endl;
				outs << "Using the " << name << " problem" << endl;
				SpParHelper::Print(outs.str());
			}
			else if(string(argv[1]) == string("Force"))	
			{
				scale = static_cast<unsigned>(atoi(argv[2]));
				ostringstream outs;
				outs << "Forcing scale to : " << scale << endl;
				SpParHelper::Print(outs.str());

				if(argc > 3 && string(argv[3]) == string("FastGen"))
				{
					SpParHelper::Print("Using fast vertex permutations; skipping edge permutations (like v2.1)\n");	
					scramble = true;
				}
			}
			else
			{
				SpParHelper::Print("Unknown option\n");
				MPI::Finalize(); 
				return -1;	
			}
			// this is an undirected graph, so A*x does indeed BFS
 			double initiator[4] = {.57, .19, .19, .05};

			double t01 = MPI_Wtime();
			double t02;
			DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>();
			if(!scramble)
			{
				DEL->GenGraph500Data(initiator, scale, EDGEFACTOR);
				SpParHelper::Print("Generated edge lists\n");
				t02 = MPI_Wtime();
				ostringstream tinfo;
				tinfo << "Generation took " << t02-t01 << " seconds" << endl;
				SpParHelper::Print(tinfo.str());
		
				PermEdges(*DEL);
				SpParHelper::Print("Permuted Edges\n");
				//DEL->Dump64bit("edges_permuted");
				//SpParHelper::Print("Dumped\n");

				RenameVertices(*DEL);	// intermediate: generates RandPerm vector, using MemoryEfficientPSort
				SpParHelper::Print("Renamed Vertices\n");
			}
			else	// fast generation
			{
				DEL->GenGraph500Data(initiator, scale, EDGEFACTOR, true, true );	// generate packed edges
				SpParHelper::Print("Generated renamed edge lists\n");
				t02 = MPI_Wtime();
				ostringstream tinfo;
				tinfo << "Generation took " << t02-t01 << " seconds" << endl;
				SpParHelper::Print(tinfo.str());
			}

			// Start Kernel #1
			MPI::COMM_WORLD.Barrier();
			double t1 = MPI_Wtime();

			// conversion from distributed edge list, keeps self-loops, sums duplicates
			PSpMat_Int32 * G = new PSpMat_Int32(*DEL, false); 
			delete DEL;	// free memory before symmetricizing
			SpParHelper::Print("Created Sparse Matrix (with int32 local indices and values)\n");

			MPI::COMM_WORLD.Barrier();
			double redts = MPI_Wtime();
			G->Reduce(degrees, Row, plus<int64_t>(), static_cast<int64_t>(0));	// Identity is 0 
			MPI::COMM_WORLD.Barrier();
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

			FullyDistVec<int64_t, int64_t> * ColSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid(), 0);
			FullyDistVec<int64_t, int64_t> * RowSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid(), 0);
			A.Reduce(*ColSums, Column, plus<int64_t>(), static_cast<int64_t>(0)); 	
			A.Reduce(*RowSums, Row, plus<int64_t>(), static_cast<int64_t>(0)); 	
			SpParHelper::Print("Reductions done\n");
			ColSums->EWiseApply(*RowSums, plus<int64_t>());
			SpParHelper::Print("Intersection of colsums and rowsums found\n");
			delete RowSums;

			// TODO: seg fault in FindInds for scale 33 
			nonisov = ColSums->FindInds(bind2nd(greater<int64_t>(), 0));	// only the indices of non-isolated vertices
			delete ColSums;

			SpParHelper::Print("Found (and permuted) non-isolated vertices\n");	
			nonisov.RandPerm();	// so that A(v,v) is load-balanced (both memory and time wise)
			A.PrintInfo();
			A(nonisov, nonisov, true);	// in-place permute to save memory	
			SpParHelper::Print("Dropped isolated vertices from input\n");	
			A.PrintInfo();

			Symmetricize(A);	// A += A';
			SpParHelper::Print("Symmetricized\n");	

		#ifdef THREADED	
			ostringstream tinfo;
			tinfo << "Threading activated with " << cblas_splits << " threads" << endl;
			SpParHelper::Print(tinfo.str());
			A.ActivateThreading(cblas_splits);	
		#endif
			A.PrintInfo();
			
			MPI::COMM_WORLD.Barrier();
			double t2=MPI_Wtime();
			
			ostringstream k1timeinfo;
			k1timeinfo << (t2-t1) - (redtf-redts) << " seconds elapsed for Kernel #1" << endl;
			SpParHelper::Print(k1timeinfo.str());
		}
		A.PrintInfo();
		float balance = A.LoadImbalance();
		ostringstream outs;
		outs << "Load balance: " << balance << endl;
		SpParHelper::Print(outs.str());

		MPI::COMM_WORLD.Barrier();
		double t1 = MPI_Wtime();

		// TODO: Threaded code crashes in FullyDistVec()
		// Now that every remaining vertex is non-isolated, randomly pick ITERS many of them as starting vertices
		degrees = degrees(nonisov);	// fix the degrees array too
		degrees.PrintInfo("Degrees array");
		// degrees.DebugPrint();
		FullyDistVec<int64_t, int64_t> Cands(ITERS, 0, 0);
		double nver = (double) degrees.TotalLength();

		MTRand M;	// generate random numbers with Mersenne Twister
		vector<double> loccands(ITERS);
		vector<int64_t> loccandints(ITERS);
		if(myrank == 0)
		{
			for(int i=0; i<ITERS; ++i)
				loccands[i] = M.rand();
			copy(loccands.begin(), loccands.end(), ostream_iterator<double>(cout," ")); cout << endl;
			transform(loccands.begin(), loccands.end(), loccands.begin(), bind2nd( multiplies<double>(), nver ));
			
			for(int i=0; i<ITERS; ++i)
				loccandints[i] = static_cast<int64_t>(loccands[i]);
			copy(loccandints.begin(), loccandints.end(), ostream_iterator<double>(cout," ")); cout << endl;
		}

		MPI::COMM_WORLD.Barrier();
		MPI::COMM_WORLD.Bcast(&(loccandints[0]), ITERS, MPIType<int64_t>(),0);
		MPI::COMM_WORLD.Barrier();
		for(int i=0; i<ITERS; ++i)
		{
			Cands.SetElement(i,loccandints[i]);
		}

		#ifdef THREADED
		#define MAXTRIALS 1
		#else
		#define MAXTRIALS 2
		#endif		
		for(int trials =0; trials < MAXTRIALS; trials++)	// try different algorithms for BFS
		{
			cblas_allgathertime = 0;
			cblas_alltoalltime = 0;
			if(trials == 1)		// second run for multithreaded turned off
			{
	                       	A.OptimizeForGraph500(optbuf);	
				MPI_Pcontrol(1,"BFS_SPA_Buf");
			}
			else
				MPI_Pcontrol(1,"BFS");

			double MTEPS[ITERS]; double INVMTEPS[ITERS]; double TIMES[ITERS]; double EDGES[ITERS];
			for(int i=0; i<ITERS; ++i)
			{
				// FullyDistVec (shared_ptr<CommGrid> grid, IT globallen, NT initval, NT id);
				FullyDistVec<int64_t, int64_t> parents ( A.getcommgrid(), A.getncol(), (int64_t) -1, (int64_t) -1);	// identity is -1

				// FullyDistSpVec ( shared_ptr<CommGrid> grid, IT glen);
				FullyDistSpVec<int64_t, int64_t> fringe(A.getcommgrid(), A.getncol());	// numerical values are stored 0-based

				MPI::COMM_WORLD.Barrier();
				double t1 = MPI_Wtime();

				fringe.SetElement(Cands[i], Cands[i]);
				int iterations = 0;
				while(fringe.getnnz() > 0)
				{
					fringe.setNumToInd();
					//fringe.PrintInfo("fringe before SpMV");
					fringe = SpMV<SR>(A, fringe,true, optbuf);	// SpMV with sparse vector (with indexisvalue flag set), optimization enabled
					// fringe.PrintInfo("fringe after SpMV");
	
					#ifdef TIMING
					MPI::COMM_WORLD.Barrier();
					double t_a1 = MPI_Wtime();
					#endif
					fringe = EWiseMult(fringe, parents, true, (int64_t) -1);	// clean-up vertices that already has parents 
					#ifdef TIMING
					MPI::COMM_WORLD.Barrier();
					double t_a2 = MPI_Wtime();
					ostringstream ewisemtime;
					ewisemtime << "EWiseMult took " << t_a2-t_a1 << " seconds" << endl;
					SpParHelper::Print(ewisemtime.str());
					#endif
					// fringe.PrintInfo("fringe after cleanup");
					parents += fringe;
					// parents.PrintInfo("Parents after addition");
					iterations++;
					MPI::COMM_WORLD.Barrier();
				}
				MPI::COMM_WORLD.Barrier();
				double t2 = MPI_Wtime();
	
				FullyDistSpVec<int64_t, int64_t> parentsp = parents.Find(bind2nd(greater<int64_t>(), -1));
				parentsp.Apply(set<int64_t>(1));
	
				// we use degrees on the directed graph, so that we don't count the reverse edges in the teps score
				int64_t nedges = EWiseMult(parentsp, degrees, false, (int64_t) 0).Reduce(plus<int64_t>(), (int64_t) 0);
	
				ostringstream outnew;
				outnew << i << "th starting vertex was " << Cands[i] << endl;
				outnew << "Number iterations: " << iterations << endl;
				outnew << "Number of vertices found: " << parentsp.Reduce(plus<int64_t>(), (int64_t) 0) << endl; 
				outnew << "Number of edges traversed: " << nedges << endl;
				outnew << "BFS time: " << t2-t1 << " seconds" << endl;
				outnew << "MTEPS: " << static_cast<double>(nedges) / (t2-t1) / 1000000.0 << endl;
				outnew << "Total communication (average so far): " << (cblas_allgathertime + cblas_alltoalltime) / (i+1) << endl;
				TIMES[i] = t2-t1;
				EDGES[i] = nedges;
				MTEPS[i] = static_cast<double>(nedges) / (t2-t1) / 1000000.0;
				SpParHelper::Print(outnew.str());
			}
			SpParHelper::Print("Finished\n");
			ostringstream os;
			if(trials == 1)
				MPI_Pcontrol(-1,"BFS_SPA_Buf");
			else
				MPI_Pcontrol(-1,"BFS");
			

			os << "Per iteration communication times: " << endl;
			os << "AllGatherv: " << cblas_allgathertime / ITERS << endl;
			os << "AlltoAllv: " << cblas_alltoalltime / ITERS << endl;

			sort(EDGES, EDGES+ITERS);
			os << "--------------------------" << endl;
			os << "Min nedges: " << EDGES[0] << endl;
			os << "First Quartile nedges: " << (EDGES[(ITERS/4)-1] + EDGES[ITERS/4])/2 << endl;
			os << "Median nedges: " << (EDGES[(ITERS/2)-1] + EDGES[ITERS/2])/2 << endl;
			os << "Third Quartile nedges: " << (EDGES[(3*ITERS/4) -1 ] + EDGES[3*ITERS/4])/2 << endl;
			os << "Max nedges: " << EDGES[ITERS-1] << endl;
 			double mean = accumulate( EDGES, EDGES+ITERS, 0.0 )/ ITERS;
			vector<double> zero_mean(ITERS);	// find distances to the mean
			transform(EDGES, EDGES+ITERS, zero_mean.begin(), bind2nd( minus<double>(), mean )); 	
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
			transform(TIMES, TIMES+ITERS, zero_mean.begin(), bind2nd( minus<double>(), mean )); 	
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
			transform(INVMTEPS, INVMTEPS+ITERS, zero_mean.begin(), bind2nd(minus<double>(), 1/hteps));
			deviation = inner_product( zero_mean.begin(),zero_mean.end(), zero_mean.begin(), 0.0 );
   			deviation = sqrt( deviation / (ITERS-1) ) * (hteps*hteps);	// harmonic_std_dev
			os << "Harmonic standard deviation of MTEPS: " << deviation << endl;
			SpParHelper::Print(os.str());
		}
	}
	MPI::Finalize();
	return 0;
}

