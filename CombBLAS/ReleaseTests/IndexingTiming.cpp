#include <mpi.h>
#include <sys/time.h> 
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include "../SpParVec.h"
#include "../SpTuples.h"
#include "../SpDCCols.h"
#include "../SpParMat.h"
#include "../DenseParMat.h"
#include "../DenseParVec.h"


using namespace std;

#define ITERATIONS 10

int main(int argc, char* argv[])
{
	MPI::Init(argc, argv);
	int nprocs = MPI::COMM_WORLD.Get_size();
	int myrank = MPI::COMM_WORLD.Get_rank();

	if(argc < 2)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./IndexingTest <Inputfile>" << endl;
		}
		MPI::Finalize(); 
		return -1;
	}				
	{
		ifstream input(argv[1]);
		if(myrank == 0)
		{	
			if(input.fail())
			{
				cout << "One of the input files do not exist, aborting" << endl;
				MPI::COMM_WORLD.Abort(NOFILE);
				return -1;
			}
		}
		MPI::COMM_WORLD.Barrier();
		typedef SpParMat <int, double, SpDCCols<int,double> > PARDBMAT;
		PARDBMAT A;		// declare objects
		A.ReadDistribute(input, 0);	

		A.PrintInfo();	
		SpParVec<int,int> p;
		RandPerm(p,A.getlocalrows());
		SpParHelper::Print("Permutation Generated\n");
		PARDBMAT B = A(p,p);
		B.PrintInfo();

		float oldbalance = A.LoadImbalance();
		float newbalance = B.LoadImbalance();
		ostringstream outs;
		outs << "Old balance: " << oldbalance << endl;
		outs << "New balance: " << newbalance << endl;
		SpParHelper::Print(outs.str());
		SpParHelper::Print(outs.str());

		MPI::COMM_WORLD.Barrier();
		double t1 = MPI::Wtime(); 	// initilize (wall-clock) timer
	
		for(int i=0; i<ITERATIONS; i++)
		{
			B = A(p,p);
		}
		
		MPI::COMM_WORLD.Barrier();
		double t2 = MPI::Wtime(); 	

		if(myrank == 0)
		{
			cout<<"Indexing Iterations finished"<<endl;	
			printf("%.6lf seconds elapsed per iteration\n", (t2-t1)/(double)ITERATIONS);
		}

		/**  Test #2
   		**   
		int nclust = 10;
		vector< SpParVec<int,int> > clusters(nclust);

		for(int i = 0; i< nclust; i++)
		{
			int k = std::min(A.getnrow() / nclust, A.getnrow() - nclust * i);
			clusters[i].iota(k, nclust * i + 1);
			clusters[i] = p(clusters[i]);
		}

		for(int i=0; i< nclust; i++)
		{
			B = A(clusters[i], clusters[i]);
			B.PrintInfo();
		}  
		**/
			

		input.clear();
		input.close();
	}
	MPI::Finalize();
	return 0;
}
