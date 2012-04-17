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

// Simple helper class for declarations: Just the numerical type is templated 
// The index type and the sequential matrix type stays the same for the whole code
// In this case, they are "int" and "SpDCCols"
template <class NT>
class PSpMat 
{ 
public: 
	typedef SpDCCols < int, NT > DCCols;
	typedef SpParMat < int, NT, DCCols > MPI_DCCols;
};


int main(int argc, char* argv[])
{
	MPI::Init(argc, argv);
	int nprocs = MPI::COMM_WORLD.Get_size();
	int myrank = MPI::COMM_WORLD.Get_rank();

	if(argc < 5)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./GalerkinNew <Matrix> <OffDiagonal> <Diagonal> <T(right hand side restriction matrix)>" << endl;
			cout << "<Matrix> <OffDiagonal> <Diagonal> <T> are absolute addresses, and files should be in triples format" << endl;
			cout << "Example: ./GalerkinNew TESTDATA/grid3d_k5.txt TESTDATA/offdiag_grid3d_k5.txt TESTDATA/diag_grid3d_k5.txt TESTDATA/restrict_T_grid3d_k5.txt" << endl;
		}
		MPI::Finalize(); 
		return -1;
	}				
	{
		string Aname(argv[1]);		
		string Aoffd(argv[2]);
		string Adiag(argv[3]);
		string Tname(argv[4]);		

		// A = L+D
		// A*T = L*T + D*T;
		// S*(A*T) = S*L*T + S*D*T;
		ifstream inputD(Adiag.c_str());

		MPI::COMM_WORLD.Barrier();
		typedef PlusTimesSRing<double, double> PTDD;	

		PSpMat<double>::MPI_DCCols A, L, T;	// construct objects
		FullyDistVec<int,double> dvec;
		
		// For matrices, passing the file names as opposed to fstream objects
		A.ReadDistribute(Aname, 0);
		L.ReadDistribute(Aoffd, 0);
		T.ReadDistribute(Tname, 0);
		dvec.ReadDistribute(inputD,0);
		SpParHelper::Print("Data read\n");

		PSpMat<double>::MPI_DCCols S = T;
		S.Transpose();

		// force the calling of C's destructor; warm up instruction cache - also check correctness
		{
			PSpMat<double>::MPI_DCCols AT = PSpGEMM<PTDD>(A, T);
			PSpMat<double>::MPI_DCCols SAT = PSpGEMM<PTDD>(S, AT);

			PSpMat<double>::MPI_DCCols LT = PSpGEMM<PTDD>(L, T); 
			PSpMat<double>::MPI_DCCols SLT = PSpGEMM<PTDD>(S, LT);
			PSpMat<double>::MPI_DCCols SD = S;
			SD.DimApply(Column, dvec, multiplies<double>());	// scale columns of S to get SD
			PSpMat<double>::MPI_DCCols SDT = PSpGEMM<PTDD>(SD, T);
			SLT += SDT;	// now this is SAT

			if(SLT == SAT)
			{
				SpParHelper::Print("Splitting approach is correct\n");
			}
			else
			{
				SpParHelper::Print("Error in splitting, go fix it\n");
			}
		}	
		MPI::COMM_WORLD.Barrier();
		double t1 = MPI::Wtime(); 	// initilize (wall-clock) timer
		for(int i=0; i<ITERATIONS; i++)
		{
			PSpMat<double>::MPI_DCCols AT = PSpGEMM<PTDD>(A, T);
			PSpMat<double>::MPI_DCCols SAT = PSpGEMM<PTDD>(S, AT);
		}
		MPI::COMM_WORLD.Barrier();
		double t2 = MPI::Wtime(); 	
		if(myrank == 0)
		{
			cout<<"Full restriction (without splitting) finished"<<endl;	
			printf("%.6lf seconds elapsed per iteration\n", (t2-t1)/(double)ITERATIONS);
		}

		MPI::COMM_WORLD.Barrier();
		t1 = MPI::Wtime(); 	// initilize (wall-clock) timer
		for(int i=0; i<ITERATIONS; i++)
		{

			PSpMat<double>::MPI_DCCols LT = PSpGEMM<PTDD>(L, T); 
			PSpMat<double>::MPI_DCCols SLT = PSpGEMM<PTDD>(S, LT);
			PSpMat<double>::MPI_DCCols SD = S;
			SD.DimApply(Column, dvec, multiplies<double>());	// scale columns of S to get SD
			PSpMat<double>::MPI_DCCols SDT = PSpGEMM<PTDD>(SD, T);
			SLT += SDT;
		}
		MPI::COMM_WORLD.Barrier();
		t2 = MPI::Wtime(); 	
		if(myrank == 0)
		{
			cout<<"Full restriction (with splitting) finished"<<endl;	
			printf("%.6lf seconds elapsed per iteration\n", (t2-t1)/(double)ITERATIONS);
		}
		inputD.clear();inputD.close();
	}
	MPI::Finalize();
	return 0;
}

