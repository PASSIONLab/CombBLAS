/****************************************************************/
/* Sequential and Parallel Sparse Matrix Multiplication Library */
/* version 2.3 --------------------------------------------------/
/* date: 01/18/2009 ---------------------------------------------/
/* author: Aydin Buluc (aydin@cs.ucsb.edu) ----------------------/
/****************************************************************/

#include <mpi.h>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>  // Required for stringstreams
#include <ctime>
#include <cmath>
#include "../SpParVec.h"
#include "../SpTuples.h"
#include "../SpDCCols.h"
#include "../SpParMat.h"
#include "../DenseParMat.h"
#include "../DenseParVec.h"

using namespace std;

#define EPS 0.00001

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


double Chaos(const PSpMat<double>::MPI_DCCols & A)
{
	return 1.0;
}



void Interpret(const PSpMat<double>::MPI_DCCols & A)
{
	// Placeholder
}


int main(int argc, char* argv[])
{
	MPI::Init(argc, argv);
	int nprocs = MPI::COMM_WORLD.Get_size();
	int myrank = MPI::COMM_WORLD.Get_rank();
	
	typedef PlusTimesSRing<double, double> PTDOUBLEDOUBLE;

	if(argc < 4)
        {
		if(myrank == 0)
		{	
                	cout << "Usage: ./mcl <BASEADDRESS> <INFLATION> <PRUNELIMIT>" << endl;
                	cout << "Example: ./mcl Data/ 2 0.00001" << endl;
                	cout << "Input file input.txt should be under <BASEADDRESS> in triples format" << endl;
                }
		MPI::Finalize(); 
		return -1;
        }

	{
		double inflation = atof(argv[2]);
		double prunelimit = atof(argv[3]);

		string directory(argv[1]);		
		string ifilename = "input.txt";
		ifilename = directory+"/"+ifilename;

		ifstream input(ifilename.c_str());
		if( !input ) 
		{
		    	SpParHelper::Print( "Error opening input stream\n");
    			return -1;
  		}
		MPI::COMM_WORLD.Barrier();
	
		PSpMat<double>::MPI_DCCols A;	// construct object
		A.ReadDistribute(input, 0);	// read it from file
	
		input.clear();
		input.close();
		
		double oldchaos = Chaos(A); 
		double newchaos = oldchaos ;
		// while there is an epsilon improvement
		while(( oldchaos - newchaos) > EPS)
		{
			A.Square<PTDOUBLEDOUBLE>() ;		// expand 
			A.Inflate(inflation);	// inflate (and renormalize)

			SpParHelper::Print("Before pruning...");
			A.PrintInfo();

			A.Prune(bind2nd(less<double>(), prunelimit));

			SpParHelper::Print("After pruning...");
			A.PrintInfo();

			oldchaos = newchaos; 
			newchaos = Chaos(A);
		}
		Interpret(A);	
	}	

	// make sure the destructors for all objects are called before MPI::Finalize()
	MPI::Finalize();	
	return 0;
}

