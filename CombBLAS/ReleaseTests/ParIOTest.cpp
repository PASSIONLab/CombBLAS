#include <mpi.h>
#include <sys/time.h> 
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include "../CombBLAS.h"

using namespace std;



// Simple helper class for declarations: Just the numerical type is templated 
// The index type and the sequential matrix type stays the same for the whole code
// In this case, they are "int" and "SpDCCols"
template <class NT>
class PSpMat 
{ 
public: 
	typedef SpDCCols < int64_t, NT > DCCols;
	typedef SpParMat < int64_t, NT, DCCols > MPI_DCCols;
};

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
			cout << "Usage: ./ParIOTest <MatrixA>" << endl;
			cout << "<MatrixA> is an absolute address, and file should be in Matrix Market format" << endl;
		}
		MPI_Finalize(); 
		return -1;
	}				
	{
		string Aname(argv[1]);		
	
		typedef PlusTimesSRing<double, double> PTDOUBLEDOUBLE;	
		typedef SelectMaxSRing<bool, int64_t> SR;	

        PSpMat<double>::MPI_DCCols A, AControl;
		
        A.ParallelReadMM(Aname);
		AControl.ReadDistribute(Aname, 0);

		if (A == AControl)
		{
			SpParHelper::Print("Parallel Matrix Market I/O working correctly\n");
		}
		else
		{
			SpParHelper::Print("ERROR in Parallel Matrix Market I/O");
			A.SaveGathered("A_Error.txt");
		}
	}
	MPI_Finalize();
	return 0;
}

