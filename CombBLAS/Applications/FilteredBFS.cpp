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

double cblas_alltoalltime;
double cblas_allgathertime;
#ifdef _OPENMP
int cblas_splits = omp_get_max_threads(); 
#else
int cblas_splits = 1;
#endif

#include "../CombBLAS.h"
#include "TwitterEdge.h"

#define ITERS 16
#define EDGEFACTOR 16
using namespace std;

template <class T>
bool from_string(T & t, const string& s, std::ios_base& (*f)(std::ios_base&))
{
        istringstream iss(s);
        return !(iss >> f >> t).fail();
}


int main(int argc, char* argv[])
{
	MPI::Init(argc, argv);
	MPI::COMM_WORLD.Set_errhandler ( MPI::ERRORS_THROW_EXCEPTIONS );
	int nprocs = MPI::COMM_WORLD.Get_size();
	int myrank = MPI::COMM_WORLD.Get_rank();
	
	if(argc < 3)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./FilteredBFS <Text, Binary, Gen> <Text Input Name | Binary Input Name | Scale Forced>" << endl;
			cout << "Example: ./FilteredBFS Text twitter_small.txt" << endl;
		}
		MPI::Finalize(); 
		return -1;
	}		
	{
		typedef SpParMat < int64_t, TwitterEdge, SpDCCols<int64_t, TwitterEdge > > PSpMat_Twitter;
		typedef SpParMat < int64_t, bool, SpDCCols<int64_t, bool > > PSpMat_Bool;

		// Declare objects
		PSpMat_Twitter A;	
		FullyDistVec<int64_t, int64_t> degrees;	// degrees of vertices (including multi-edges and self-loops)
		FullyDistVec<int64_t, int64_t> nonisov;	// id's of non-isolated (connected) vertices

		if(string(argv[1]) == string("Text")) // text input option
		{
			ifstream input(argv[2]);
			// ReadDistribute (ifstream& infile, int master, bool nonum, HANDLER handler, bool transpose)
			// if nonum is true, then numerics are not supplied and they are assumed to be all 1's
			A.ReadDistribute(input, 0, false, TwitterReadSaveHandler<int64_t>(), true);	// read it from file (and transpose on the fly)
			A.PrintInfo();
			SpParHelper::Print("Read input\n");
			return -1;

			PSpMat_Bool * ABool = new PSpMat_Bool(A);
			FullyDistVec<int64_t, int64_t> * ColSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid());
			FullyDistVec<int64_t, int64_t> * RowSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid());
			ABool->Reduce(*ColSums, Column, plus<int64_t>(), static_cast<int64_t>(0)); 	
			ABool->Reduce(*RowSums, Row, plus<int64_t>(), static_cast<int64_t>(0)); 	
			ColSums->DebugPrint();
			RowSums->DebugPrint();
			ColSums->EWiseApply(*RowSums, plus<int64_t>());
			delete RowSums;

			nonisov = ColSums->FindInds(bind2nd(greater<int64_t>(), 0));	// only the indices of non-isolated vertices
			delete ColSums;

			SpParHelper::Print("Found (and permuted) non-isolated vertices\n");	
			nonisov.RandPerm();	// so that A(v,v) is load-balanced (both memory and time wise)
			A.PrintInfo();
			A(nonisov, nonisov, true);	// in-place permute to save memory
			SpParHelper::Print("Dropped isolated vertices from input\n");	
/*
			Symmetricize(A);	// A += A';
*/
		}
		else 
		{	
			SpParHelper::Print("Not supported yet\n");
			return 0;
		}
		A.PrintInfo();
		float balance = A.LoadImbalance();
		ostringstream outs;
		outs << "Load balance: " << balance << endl;
		SpParHelper::Print(outs.str());

		MPI::COMM_WORLD.Barrier();
		double t1 = MPI_Wtime();

		degrees = degrees(nonisov);	// fix the degrees array too
		degrees.PrintInfo("Degrees array");

		FullyDistVec<int64_t, int64_t> Cands(ITERS);
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

		#define MAXTRIALS 1
		for(int trials =0; trials < MAXTRIALS; trials++)	// try different algorithms for BFS
		{
			cblas_allgathertime = 0;
			cblas_alltoalltime = 0;
			MPI_Pcontrol(1,"BFS");

			double MTEPS[ITERS]; double INVMTEPS[ITERS]; double TIMES[ITERS]; double EDGES[ITERS];
			for(int i=0; i<ITERS; ++i)
			{
				// FullyDistVec ( shared_ptr<CommGrid> grid, IT globallen, NT initval);
				FullyDistVec<int64_t, ParentType> parents ( A.getcommgrid(), A.getncol(), ParentType());	

				// FullyDistSpVec ( shared_ptr<CommGrid> grid, IT glen);
				FullyDistSpVec<int64_t, ParentType> fringe(A.getcommgrid(), A.getncol());	

				MPI::COMM_WORLD.Barrier();
				double t1 = MPI_Wtime();

				fringe.SetElement(Cands[i], Cands[i]);
				int iterations = 0;
				while(fringe.getnnz() > 0)
				{
					fringe.ApplyInd(NumSetter);
					//fringe.PrintInfo("fringe before SpMV");

					// SpMV with sparse vector, optimizations disabled for generality
					SpMV<LatestRetwitterBFS>(A, fringe, fringe, false);	
					// fringe.PrintInfo("fringe after SpMV");
				
					//  EWiseApply (const FullyDistSpVec<IU,NU1> & V, const FullyDistVec<IU,NU2> & W, 
					//		_BinaryOperation _binary_op, _BinaryPredicate _doOp, bool allowVNulls, NU1 Vzero)
					// ABAB: Parallelize EWiseApply?
					// ABAB: allowVNulls currently true, until we embed its logic to SpMV	
					// #define prunediscovered_f(x,y) return ( y == ParentType() ) ? x: ParentType()

					fringe = EWiseApply<ParentType>(fringe, parents, prunediscovered_f(), bintotality<ParentType,ParentType>(), true, ParentType());
					// fringe.PrintInfo("fringe after cleanup");
					parents += fringe;
					// parents.PrintInfo("Parents after addition");
					iterations++;
					MPI::COMM_WORLD.Barrier();
				}
				MPI::COMM_WORLD.Barrier();
				double t2 = MPI_Wtime();
	
			/*
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
			*/
			}
			SpParHelper::Print("Finished\n");
			ostringstream os;
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

