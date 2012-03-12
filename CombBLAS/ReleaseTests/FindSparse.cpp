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
	MPI::Init(argc, argv);
	int nprocs = MPI::COMM_WORLD.Get_size();
	int myrank = MPI::COMM_WORLD.Get_rank();

	if(argc < 3)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./FindSparse <BASEADDRESS> <Matrix>" << endl;
			cout << "Input files should be under <BASEADDRESS> in appropriate format" << endl;
		}
		MPI::Finalize(); 
		return -1;
	}				
	{
		string directory(argv[1]);		
		string matrixname(argv[2]);
		matrixname = directory+"/"+matrixname;
	
		typedef SpParMat <int, double, SpDCCols<int,double> > PARDBMAT;
		PARDBMAT A;		// declare objects
		FullyDistVec<int,int> crow, ccol;
		FullyDistVec<int,double> cval;

		A.ReadDistribute(matrixname, 0);	

		A.Find(crow, ccol, cval);
		PARDBMAT B(A.getnrow(), A.getncol(), crow, ccol, cval); // Sparse()

		if (A ==  B)
		{
			SpParHelper::Print("Find and Sparse working correctly\n");	
		}
		else
		{
			SpParHelper::Print("ERROR in Find(), go fix it!\n");	

			SpParHelper::Print("Rows array: \n");
			crow.DebugPrint();

			SpParHelper::Print("Columns array: \n");
			ccol.DebugPrint();

			SpParHelper::Print("Values array: \n");
			cval.DebugPrint();
		}
	}
	MPI::Finalize();
	return 0;
}
