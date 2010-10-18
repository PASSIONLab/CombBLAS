#include <mpi.h>
#include <sys/time.h> 
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#ifdef NOTR1
        #include <boost/tr1/memory.hpp>
#else
        #include <tr1/memory>
#endif
#include "../SpParVec.h"
#include "../SpTuples.h"
#include "../SpDCCols.h"
#include "../SpParMat.h"
#include "../DenseParMat.h"
#include "../DenseParVec.h"


using namespace std;

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

	if(argc < 4)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./MultTest <MatrixA> <MatrixB> <MatrixC>" << endl;
			cout << "<MatrixA>,<MatrixB>,<MatrixC> are absolute addresses, and files should be in triples format" << endl;
		}
		MPI::Finalize(); 
		return -1;
	}				
	{
		string Aname(argv[1]);		
		string Bname(argv[2]);
		string Cname(argv[3]);

		ifstream inputA(Aname.c_str());
		ifstream inputB(Bname.c_str());
		ifstream inputC(Cname.c_str());

		MPI::COMM_WORLD.Barrier();
	
		typedef PlusTimesSRing<double, double> PTDOUBLEDOUBLE;	

		PSpMat<double>::MPI_DCCols A, B, C, CControl;	// construct objects
		
		A.ReadDistribute(inputA, 0);
		B.ReadDistribute(inputB, 0);
		CControl.ReadDistribute(inputC, 0);

		C = Mult_AnXBn_PassiveTarget<PTDOUBLEDOUBLE>(A, B);

		if (CControl == C)
		{
			SpParHelper::Print("Passive Target Multiplication working correctly\n");	
		}
		else
		{
			SpParHelper::Print("ERROR in Passive Target Multiplication, go fix it!\n");	
		}


		C = Mult_AnXBn_Synch<PTDOUBLEDOUBLE>(A,B);
		if (CControl == C)
		{
			SpParHelper::Print("Synchronous Multiplication working correctly\n");	
		}
		else
		{
			SpParHelper::Print("ERROR in Synchronous Multiplication, go fix it!\n");	
		}


		inputA.clear();
		inputA.close();
		inputB.clear();
		inputB.close();
		inputC.clear();
		inputC.close();
	}
	MPI::Finalize();
	return 0;
}

