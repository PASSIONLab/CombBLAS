#include <mpi.h>
#include <sys/time.h> 
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include "../CombBLAS.h"

using namespace std;

#define ITERATIONS 10
#define EDGEFACTOR 8

template <class T>
bool from_string(T & t, const string& s, std::ios_base& (*f)(std::ios_base&))
{
        istringstream iss(s);
        return !(iss >> f >> t).fail();
}

int main(int argc, char* argv[])
{
	MPI::Init(argc, argv);
	int nprocs = MPI::COMM_WORLD.Get_size();
	int myrank = MPI::COMM_WORLD.Get_rank();

	if(argc < 2)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./IndexingTiming <Scale>" << endl;
		}
		MPI::Finalize(); 
		return -1;
	}				
	{
		typedef SpParMat <int, double, SpDCCols<int,double> > PARDBMAT;
		PARDBMAT *A, *B;		// declare objects
 		double initiator[4] = {.6, .4/3, .4/3, .4/3};
		DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>();

		int scale = static_cast<unsigned>(atoi(argv[1]));
		ostringstream outs;
		outs << "Forcing scale to : " << scale << endl;
		SpParHelper::Print(outs.str());
		DEL->GenGraph500Data(initiator, scale, EDGEFACTOR, true, true );        // generate packed edges
		SpParHelper::Print("Generated renamed edge lists\n");
		
		// conversion from distributed edge list, keeps self-loops, sums duplicates
		A = new PARDBMAT(*DEL, false); 
		delete DEL;	// free memory before symmetricizing
		SpParHelper::Print("Created double Sparse Matrix\n");
		A->PrintInfo();	

		for(unsigned i=1; i<3; i++)
		{
			DEL->GenGraph500Data(initiator, scale-i, EDGEFACTOR, true, true );        // "i" scale smaller
			B = new PARDBMAT(*DEL, false);
			delete DEL;
			SpParHelper::Print("Created RHS Matrix\n");
			B->PrintInfo();
			FullyDistVec<int,int> p;
			p.iota(A->getnrow(), 0);
			p.RandPerm();
			p = p(B->getnrow());  // just get the first B->getnrow() entries of the permutation
			p.PrintInfo();
		
			PARDBMAT ATemp = *A; 	
			ATemp.SpAsgn(p,p,*B);
		
			double t1 = MPI::Wtime(); 	// initilize (wall-clock) timer
			for(int j=0; j< ITERATIONS; ++j)
				PARDBMAT ATemp = *A;
			double t2 = MPI::Wtime(); 	
			double copytime = t2-t1;

			t1 = MPI::Wtime();
			for(int j=0; j< ITERATIONS; ++j)
			{
				PARDBMAT ATemp = *A;
				ATemp.SpAsgn(p,p,*B);
			}	
			t2 = MPI::Wtime();

			if(myrank == 0)
			{
				cout<< "Scale " << scale-i << " assignment iterations finished"<<endl;	
				printf("%.6lf seconds elapsed per iteration\n", ((t2-t1)-copytime)/(double)ITERATIONS);
			
			}	
			delete B;
		}
	}
	MPI::Finalize();
	return 0;
}
