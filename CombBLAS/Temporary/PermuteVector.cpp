#include <mpi.h>
#include <sys/time.h> 
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include "../CombBLAS.h"

using namespace std;


int main(int argc, char* argv[])
{
	int nprocs, myrank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

	if(argc < 2)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./PermuteVectors <VectorOne> <VectorTwo> ... (as many vectors as needed)\n" << endl;
			cout << "Permutes all the vector inputs with the same random order\n" << endl;
		}
		MPI_Finalize(); 
		return -1;
	}				
	{
		bool randpermed =  false;
		FullyDistVec<int,int> randperm;
		for(int i=1; i < argc; ++i)
		{ 
			string vecname(argv[i]);
			ifstream inputvec(vecname.c_str());

			if(myrank == 0)
			{	
				if(inputvec.fail())
				{
					cout << "One of the input vector files do not exist, aborting" << endl;
					MPI_Abort(MPI_COMM_WORLD, NOFILE);
					return -1;
				}
			}
			FullyDistVec<int,int> vec;
			vec.ReadDistribute(inputvec, 0);

			if(!randpermed)		
			{
				randperm.iota(vec.TotalLength(), 0);
				randperm.RandPerm();	// can't we seed this (to avoid indexing and iota generation?)
				randpermed = true;
			}
				
			inputvec.clear();
			inputvec.close();
		}
	}
	MPI_Finalize();
	return 0;
}
