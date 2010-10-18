#include <mpi.h>
#include <sys/time.h> 
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#ifdef NOTR1
        #include <boost/tr1/tuple.hpp>
#else
        #include <tr1/tuple>
#endif
#include "../SpParVec.h"
#include "../SpTuples.h"
#include "../SpDCCols.h"
#include "../SpParMat.h"
#include "../DenseParMat.h"
#include "../DenseParVec.h"


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

	if(argc < 3)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./MultTest <MatrixA> <MatrixB>" << endl;
			cout << "<MatrixA>,<MatrixB> are absolute addresses, and files should be in triples format" << endl;
		}
		MPI::Finalize(); 
		return -1;
	}				
	{
		string Aname(argv[1]);		
		string Bname(argv[2]);

		ifstream inputA(Aname.c_str());
		ifstream inputB(Bname.c_str());

		MPI::COMM_WORLD.Barrier();
	
		typedef PlusTimesSRing<double, double> PTDOUBLEDOUBLE;	

		PSpMat<double>::MPI_DCCols A, B,C;	// construct objects
		
		A.ReadDistribute(inputA, 0);
		B.ReadDistribute(inputB, 0);

		SpParHelper::Print("Data read\n");
		
		C = Mult_AnXBn_Synch<PTDOUBLEDOUBLE>(A, B);
		C = Mult_AnXBn_Synch<PTDOUBLEDOUBLE>(A, B);
		SpParHelper::Print("Warmed up for Synch\n");
		
		MPI::COMM_WORLD.Barrier();
		double t1 = MPI::Wtime(); 	// initilize (wall-clock) timer
	
		for(int i=0; i<ITERATIONS; i++)
		{
			C = Mult_AnXBn_Synch<PTDOUBLEDOUBLE>(A, B);
		}
		
		MPI::COMM_WORLD.Barrier();
		double t2 = MPI::Wtime(); 	

		if(myrank == 0)
		{
			cout<<"Synchronous multiplications finished"<<endl;	
			printf("%.6lf seconds elapsed per iteration\n", (t2-t1)/(double)ITERATIONS);
		}

		C = Mult_AnXBn_PassiveTarget<PTDOUBLEDOUBLE>(A, B);
		C = Mult_AnXBn_PassiveTarget<PTDOUBLEDOUBLE>(A, B);
		SpParHelper::Print("Warmed up for PassiveTarget\n");

		MPI::COMM_WORLD.Barrier();
		t1 = MPI::Wtime(); 	// initilize (wall-clock) timer
	
		for(int i=0; i<ITERATIONS; i++)
		{
			C = Mult_AnXBn_PassiveTarget<PTDOUBLEDOUBLE>(A, B);
		}
		
		MPI::COMM_WORLD.Barrier();
		t2 = MPI::Wtime(); 	

		if(myrank == 0)
		{
			cout<<"Passive target multiplications finished"<<endl;	
			printf("%.6lf seconds elapsed per iteration\n", (t2-t1)/(double)ITERATIONS);
		}		

		inputA.clear();
		inputA.close();
		inputB.clear();
		inputB.close();
	}
	MPI::Finalize();
	return 0;
}

