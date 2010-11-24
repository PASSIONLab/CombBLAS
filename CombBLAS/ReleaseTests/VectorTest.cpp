#include <sys/time.h> 
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include "../SpParVec.h"

using namespace std;


int main(int argc, char* argv[])
{

	MPI::Init(argc, argv);
	int nprocs = MPI::COMM_WORLD.Get_size();
	int myrank = MPI::COMM_WORLD.Get_rank();

	{
		SpParVec<int64_t, int64_t> SPV_A(1024);
		SPV_A.SetElement(2,2);
		SPV_A.SetElement(83,-83);
		SPV_A.SetElement(284,284);
		SPV_A.DebugPrint();
		
		SpParVec<int64_t, int64_t> SPV_B(1024);
		SPV_B.SetElement(2,4);
		SPV_B.SetElement(184,368);
		SPV_B.SetElement(83,-1);
		SPV_B.DebugPrint();
	}
	MPI::Finalize();
	return 0;
}

