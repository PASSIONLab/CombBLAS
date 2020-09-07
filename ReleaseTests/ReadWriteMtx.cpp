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

	if(argc < 3)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./ReadWriteMtx <InputMTX> <OutputMTX>" << endl;
			cout << "<InputMTX> can have reverse edges missing or with different values, but the reader will choose the edge with maximum value when that happens (common in dna/protein sequence search)" << endl;
			cout << "<OutputMTX> is just the name of the output after such incompatibilities in MTX file are removed and all bidirectional edges now have the same value" << endl;
			
		}
		MPI_Finalize(); 
		return -1;
	}				
	{
		string Aname(argv[1]);	
		string Bname(argv[2]);
	
		SpParMat<int64_t, double, SpDCCols<int64_t,double>> A;
		
        A.ParallelReadMM(Aname, true, maximum<double>());
        A.ParallelWriteMM(Bname,true);
	}
	MPI_Finalize();
	return 0;
}

