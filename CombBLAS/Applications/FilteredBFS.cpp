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
#ifdef TAU_PROFILE
	#include <Profile/Profiler.h>
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

#define MAX_ITERS 1024
#define ITERS 16 
#define CC_LIMIT 5
#define PERMUTEFORBALANCE
using namespace std;


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
#ifdef TAU_PROFILE
	TAU_PROFILE_TIMER(maintimer, "main()", "int (int, char **)", TAU_DEFAULT);
    	TAU_PROFILE_INIT(argc, argv);
    	TAU_PROFILE_START(maintimer);
#endif

	MPI::Init(argc, argv);
	MPI::COMM_WORLD.Set_errhandler ( MPI::ERRORS_THROW_EXCEPTIONS );
	int nprocs = MPI::COMM_WORLD.Get_size();
	int myrank = MPI::COMM_WORLD.Get_rank();
	
	if(argc < 3)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./FilteredBFS <File, Gen> <Input Name | Scale>" << endl;
			cout << "Example: ./FilteredBFS File twitter_small.txt" << endl;
		}
		MPI::Finalize(); 
		return -1;
	}		
	{
		typedef SpParMat < int64_t, TwitterEdge, SpDCCols<int64_t, TwitterEdge > > PSpMat_Twitter;
		typedef SpParMat < int64_t, bool, SpDCCols<int64_t, bool > > PSpMat_Bool;

		// Declare objects
		PSpMat_Twitter A, B;	
		FullyDistVec<int64_t, int64_t> indegrees;	// in-degrees of vertices (including multi-edges and self-loops)
		FullyDistVec<int64_t, int64_t> oudegrees;	// out-degrees of vertices (including multi-edges and self-loops)
		FullyDistVec<int64_t, int64_t> degrees;	// combined degrees of vertices (including multi-edges and self-loops)
		FullyDistVec<int64_t, int64_t> nonisov;	// id's of non-isolated (connected) vertices

		double t01 = MPI_Wtime();
		if(string(argv[1]) == string("File")) // text input option
		{
			// ReadDistribute (const string & filename, int master, bool nonum, HANDLER handler, bool transpose, bool pario)
			// if nonum is true, then numerics are not supplied and they are assumed to be all 1's
			A.ReadDistribute(string(argv[2]), 0, false, TwitterReadSaveHandler<int64_t>(), true, true);	// read it from file (and transpose on the fly)
		}
		else 
		{	
			SpParHelper::Print("Not supported yet\n");
			return 0;
		}
		double t02 = MPI_Wtime();			
		ostringstream tinfo;
		tinfo << "I/O took " << t02-t01 << " seconds" << endl;                
		SpParHelper::Print(tinfo.str());

		A.PrintInfo();
		SpParHelper::Print("Read input\n");

		PSpMat_Bool * ABool = new PSpMat_Bool(A);
		ABool->PrintInfo();
		ABool->Reduce(oudegrees, Column, plus<int64_t>(), static_cast<int64_t>(0)); 	
		ABool->Reduce(indegrees, Row, plus<int64_t>(), static_cast<int64_t>(0)); 	

//		indegrees.DebugPrint();
		degrees = indegrees;	
		degrees.EWiseApply(oudegrees, plus<int64_t>());
		SpParHelper::Print("All degrees calculated\n");
		delete ABool;

#ifdef PERMUTEFORBALANCE
		nonisov = degrees.FindInds(bind2nd(greater<int64_t>(), 0));	// only the indices of non-isolated vertices
		SpParHelper::Print("Found (and permuted) non-isolated vertices\n");	
		nonisov.RandPerm();	// so that A(v,v) is load-balanced (both memory and time wise)
		// nonisov.DebugPrint();
		A(nonisov, nonisov, true);	// in-place permute to save memory
		SpParHelper::Print("Dropped isolated vertices from input\n");	

		indegrees = indegrees(nonisov);	// fix the degrees arrays too
		oudegrees = oudegrees(nonisov);	
		degrees =degrees(nonisov);
		indegrees.PrintInfo("In degrees array");
		oudegrees.PrintInfo("Out degrees array");
#endif
			
		A.PrintInfo();
		Symmetricize(A);	// A += A';
		A.PrintInfo();
		float balance = A.LoadImbalance();
		ostringstream outs;
		outs << "Load balance: " << balance << endl;
		SpParHelper::Print(outs.str());

		MPI::COMM_WORLD.Barrier();
		double t1 = MPI_Wtime();

		FullyDistVec<int64_t, int64_t> Cands(MAX_ITERS);
		double nver = (double) degrees.TotalLength();

		MTRand M;	// generate random numbers with Mersenne Twister
		vector<double> loccands(MAX_ITERS);
		vector<int64_t> loccandints(MAX_ITERS);
		if(myrank == 0)
		{
			for(int i=0; i<MAX_ITERS; ++i)
				loccands[i] = M.rand();
			//copy(loccands.begin(), loccands.end(), ostream_iterator<double>(cout," ")); cout << endl;
			transform(loccands.begin(), loccands.end(), loccands.begin(), bind2nd( multiplies<double>(), nver ));
			
			for(int i=0; i<MAX_ITERS; ++i)
				loccandints[i] = static_cast<int64_t>(loccands[i]);
			// copy(loccandints.begin(), loccandints.end(), ostream_iterator<double>(cout," ")); cout << endl;
		}

		MPI::COMM_WORLD.Barrier();
		MPI::COMM_WORLD.Bcast(&(loccandints[0]), MAX_ITERS, MPIType<int64_t>(),0);
		MPI::COMM_WORLD.Barrier();
		for(int i=0; i<MAX_ITERS; ++i)
		{
			Cands.SetElement(i,loccandints[i]);
		}

		#define MAXTRIALS 1
		for(int trials =0; trials < MAXTRIALS; trials++)	// try different algorithms for BFS
		{
			cblas_allgathertime = 0;
			cblas_alltoalltime = 0;

			double MTEPS[ITERS]; double INVMTEPS[ITERS]; double TIMES[ITERS]; double EDGES[ITERS];
			int sruns = 0;		// successful runs
			for(int i=0; i<MAX_ITERS && sruns < ITERS; ++i)
			{

				// FullyDistVec ( shared_ptr<CommGrid> grid, IT globallen, NT initval);
				FullyDistVec<int64_t, ParentType> parents ( A.getcommgrid(), A.getncol(), ParentType());	

				// FullyDistSpVec ( shared_ptr<CommGrid> grid, IT glen);
				FullyDistSpVec<int64_t, ParentType> fringe(A.getcommgrid(), A.getncol());	

				MPI::COMM_WORLD.Barrier();
				double t1 = MPI_Wtime();

				fringe.SetElement(Cands[i], Cands[i]);
				parents.SetElement(Cands[i], ParentType(Cands[i]));	// make root discovered
				int iterations = 0;
				while(fringe.getnnz() > 0)
				{
					fringe.ApplyInd(NumSetter);
					//fringe.PrintInfo("fringe before SpMV");
					//fringe.DebugPrint();

					// SpMV with sparse vector, optimizations disabled for generality
					SpMV<LatestRetwitterBFS>(A, fringe, fringe, false);	
				#ifdef DEBUG
					if(fringe.getnnz() > 1)
					{
						fringe.PrintInfo("fringe after SpMV");
						fringe.DebugPrint();
					}
				#endif

					//  EWiseApply (const FullyDistSpVec<IU,NU1> & V, const FullyDistVec<IU,NU2> & W, 
					//		_BinaryOperation _binary_op, _BinaryPredicate _doOp, bool allowVNulls, NU1 Vzero)
					fringe = EWiseApply<ParentType>(fringe, parents, getfringe(), keepinfrontier_f(), true, ParentType());
					//fringe.PrintInfo("fringe after cleanup");
					//fringe.DebugPrint();

					
					parents += fringe;
					//parents.PrintInfo("Parents after addition");
					//parents.DebugPrint();
					iterations++;
					MPI::COMM_WORLD.Barrier();
				}
				MPI::COMM_WORLD.Barrier();
				double t2 = MPI_Wtime();
	
			
				FullyDistSpVec<int64_t, ParentType> parentsp = parents.Find(isparentset());
				parentsp.Apply(set<ParentType>(ParentType(1)));
	
				FullyDistSpVec<int64_t, int64_t> intraversed = EWiseApply<int64_t>(parentsp, indegrees, seldegree(), passifthere(), true, ParentType());
				FullyDistSpVec<int64_t, int64_t> outraversed = EWiseApply<int64_t>(parentsp, oudegrees, seldegree(), passifthere(), true, ParentType());
				
				int64_t in_nedges = intraversed.Reduce(plus<int64_t>(), (int64_t) 0);
				int64_t ou_nedges = outraversed.Reduce(plus<int64_t>(), (int64_t) 0);
				int64_t nedges = in_nedges + ou_nedges;	// count birectional edges twice
	
				if(parentsp.getnnz() > CC_LIMIT)
				{
					// intraversed.PrintInfo("Incoming edges traversed per vertex");
					// intraversed.DebugPrint();
					// outraversed.PrintInfo("Outgoing edges traversed per vertex");
					// outraversed.DebugPrint();
					
				#ifdef DEBUG
					parents.PrintInfo("Final parents array");
					parents.DebugPrint();
				#endif
					
					ostringstream outnew;
					outnew << i << "th starting vertex was " << Cands[i] << endl;
					outnew << "Number iterations: " << iterations << endl;
					outnew << "Number of vertices found: " << parentsp.getnnz() << endl; 
					outnew << "Number of edges traversed in both directions: " << nedges << endl;
					outnew << "Number of edges traversed in one direction: " << ou_nedges << endl;
					outnew << "BFS time: " << t2-t1 << " seconds" << endl;
					outnew << "MTEPS (bidirectional): " << static_cast<double>(nedges) / (t2-t1) / 1000000.0 << endl;
					outnew << "MTEPS (unidirectional): " << static_cast<double>(ou_nedges) / (t2-t1) / 1000000.0 << endl;
					outnew << "Total communication (average so far): " << (cblas_allgathertime + cblas_alltoalltime) / (i+1) << endl;

					TIMES[sruns] = t2-t1;
					EDGES[sruns] = ou_nedges;
					MTEPS[sruns++] = static_cast<double>(ou_nedges) / (t2-t1) / 1000000.0;
					SpParHelper::Print(outnew.str());
				}
			}
			if (sruns < 2)
			{
				SpParHelper::Print("Not enough valid runs done\n");
				MPI::Finalize();
			}
			ostringstream os;
			
			os << sruns << " valid runs done" << endl;
			os << "Connected component lower limite was " << CC_LIMIT << endl;
			os << "Per iteration communication times: " << endl;
			os << "AllGatherv: " << cblas_allgathertime / sruns << endl;
			os << "AlltoAllv: " << cblas_alltoalltime / sruns << endl;

			sort(EDGES, EDGES+sruns);
			os << "--------------------------" << endl;
			os << "Min nedges: " << EDGES[0] << endl;
			os << "Median nedges: " << (EDGES[(sruns/2)-1] + EDGES[sruns/2])/2 << endl;
			os << "Max nedges: " << EDGES[sruns-1] << endl;
 			double mean = accumulate( EDGES, EDGES+sruns, 0.0 )/ sruns;
			vector<double> zero_mean(sruns);	// find distances to the mean
			transform(EDGES, EDGES+sruns, zero_mean.begin(), bind2nd( minus<double>(), mean )); 	
			// self inner-product is sum of sum of squares
			double deviation = inner_product( zero_mean.begin(),zero_mean.end(), zero_mean.begin(), 0.0 );
   			deviation = sqrt( deviation / (sruns-1) );
   			os << "Mean nedges: " << mean << endl;
			os << "STDDEV nedges: " << deviation << endl;
			os << "--------------------------" << endl;
	
			sort(TIMES,TIMES+sruns);
			os << "Min time: " << TIMES[0] << " seconds" << endl;
			os << "Median time: " << (TIMES[(sruns/2)-1] + TIMES[sruns/2])/2 << " seconds" << endl;
			os << "Max time: " << TIMES[sruns-1] << " seconds" << endl;
 			mean = accumulate( TIMES, TIMES+sruns, 0.0 )/ sruns;
			transform(TIMES, TIMES+sruns, zero_mean.begin(), bind2nd( minus<double>(), mean )); 	
			deviation = inner_product( zero_mean.begin(),zero_mean.end(), zero_mean.begin(), 0.0 );
   			deviation = sqrt( deviation / (sruns-1) );
   			os << "Mean time: " << mean << " seconds" << endl;
			os << "STDDEV time: " << deviation << " seconds" << endl;
			os << "--------------------------" << endl;

			sort(MTEPS, MTEPS+sruns);
			os << "Min MTEPS: " << MTEPS[0] << endl;
			os << "Median MTEPS: " << (MTEPS[(sruns/2)-1] + MTEPS[sruns/2])/2 << endl;
			os << "Max MTEPS: " << MTEPS[sruns-1] << endl;
			transform(MTEPS, MTEPS+sruns, INVMTEPS, safemultinv<double>()); 	// returns inf for zero teps
			double hteps = static_cast<double>(sruns) / accumulate(INVMTEPS, INVMTEPS+sruns, 0.0);	
			os << "Harmonic mean of MTEPS: " << hteps << endl;
			transform(INVMTEPS, INVMTEPS+sruns, zero_mean.begin(), bind2nd(minus<double>(), 1/hteps));
			deviation = inner_product( zero_mean.begin(),zero_mean.end(), zero_mean.begin(), 0.0 );
   			deviation = sqrt( deviation / (sruns-1) ) * (hteps*hteps);	// harmonic_std_dev
			os << "Harmonic standard deviation of MTEPS: " << deviation << endl;
			SpParHelper::Print(os.str());
		}
	}
#ifdef TAU_PROFILE
    	TAU_PROFILE_STOP(maintimer);
#endif
	MPI::Finalize();
	return 0;
}

