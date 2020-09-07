#include "CombBLAS.h"
#include <mpi.h>
#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>

int main(int argc, char* argv[])
{
	int nprocs, myrank;
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
        MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

	SpParMat < int64_t, bool, SpDCCols<int64_t,bool> > A;
	A.ReadDistribute(string(argv[1]), 0);   // read it from file	
	SpParMat < int64_t, bool, SpDCCols<int32_t,bool> > Aeff = A;

	OptBuf<int32_t, int64_t> optbuf; 
	Aeff.OptimizeForGraph500(optbuf); 

	FullyDistVec<int64_t, int64_t> fringe1(Aeff.getcommgrid(), Aeff.getncol(), (int64_t) 0); // anything is fine
	FullyDistSpVec<int64_t, int64_t> fringe(Aeff.getcommgrid(), Aeff.getncol());
        fringe = fringe1; // initial frontier, copy every column vertices
        fringe.setNumToInd();
        fringe.DebugPrint();                
        fringe = SpMV(Aeff, fringe,optbuf);    
        fringe.DebugPrint();
        fringe = fringe.Uniq();
        fringe.DebugPrint();
	return 1;
}
