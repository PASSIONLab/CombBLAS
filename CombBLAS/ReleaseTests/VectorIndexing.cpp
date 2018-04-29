#include <mpi.h>
#include <sys/time.h> 
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include "CombBLAS/CombBLAS.h"

using namespace std;
using namespace combblas;

int main(int argc, char* argv[])
{
	int nprocs, myrank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

	if(argc < 4)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./VectorIndexing <BASEADDRESS> <VectorOne> <VectorTwo>" << endl;
			cout << "Example: ./VectorIndexing ../TESTDATA sp10outta100.txt sp30outta100.txt" << endl;
			cout << "Input files should be under <BASEADDRESS> in tuples format" << endl;
		}
		MPI_Finalize(); 
		return -1;
	}				
	{
		string directory(argv[1]);
		string vec1name(argv[2]);
		string vec2name(argv[3]);
		vec1name = directory+"/"+vec1name;
		vec2name = directory+"/"+vec2name;

		ifstream inputvec1(vec1name.c_str());
		ifstream inputvec2(vec2name.c_str());

		if(myrank == 0)
		{	
			if(inputvec1.fail() || inputvec2.fail())
			{
				cout << "One of the input vector files do not exist, aborting" << endl;
				MPI_Abort(MPI_COMM_WORLD, NOFILE);
				return -1;
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
        FullyDistSpVec<int,int> vec1, vec2;
			
		vec1.ReadDistribute(inputvec1, 0);
		vec2.ReadDistribute(inputvec2, 0);

        vec1.PrintInfo("vec1");
        vec1.DebugPrint();
        vec2.PrintInfo("vec2");
        vec2.DebugPrint();
        
        FullyDistVec<int,int> dvec;
        dvec.iota(100, 1001);   // with 100 entries, first being 1001
        dvec.PrintInfo("dvec");
        dvec.DebugPrint();
        
        auto subvec1  = dvec(vec1);
        subvec1.DebugPrint();
        
        auto subvec2  = dvec(vec2);
        subvec2.DebugPrint();
        
        FullyDistSpVec<int,int> vecA(12);
        for(int i=0; i<12; i+=3)  vecA.SetElement(i,i);
        MPI_Barrier(MPI_COMM_WORLD);
        vecA.DebugPrint();
        FullyDistVec<int,int> dvecA;
        dvecA.iota(12, 0);   // with 12 entries, first being 0
        
        auto subvecA  = dvecA(vecA);
        subvecA.DebugPrint();

        
		inputvec1.clear();
		inputvec1.close();
		inputvec2.clear();
		inputvec2.close();
	}
	MPI_Finalize();
	return 0;
}
