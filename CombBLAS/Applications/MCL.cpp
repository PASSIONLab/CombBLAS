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
#include "../Operations.h"

using namespace std;

#define EPS 0.001

// Simple helper class for declarations: Just the numerical type is templated 
// The index type and the sequential matrix type stays the same for the whole code
// In this case, they are "int" and "SpDCCols"
template <class NT>
class Dist
{ 
public: 
	typedef SpDCCols < int, NT > DCCols;
	typedef SpParMat < int, NT, DCCols > MPI_DCCols;
	typedef DenseParVec < int, NT> MPI_DenseVec;
};


void Interpret(const Dist<double>::MPI_DCCols & A)
{
	// Placeholder
}


double Inflate(Dist<double>::MPI_DCCols & A, double power)
{		
	A.Apply(bind2nd(exponentiate(), power));
	{
		// Reduce (Column): pack along the columns, result is a vector of size n
		Dist<double>::MPI_DenseVec colsums = A.Reduce(Column, plus<double>(), 0.0);			
		colsums.Apply(safemultinv<double>());
		A.DimScale(colsums, Column);	// scale each "Column" with the given vector

#ifdef DEBUG
		colsums = A.Reduce(Column, plus<double>(), 0.0);			
		colsums.PrintToFile("colnormalizedsums"); 
#endif		
	}

	// After normalization, each column of A is now a stochastic vector
	Dist<double>::MPI_DenseVec colssqs = A.Reduce(Column, plus<double>(), 0.0, bind2nd(exponentiate(), 2));	// sums of squares of columns

	// Matrix entries are non-negative, so max() can use zero as identity
	Dist<double>::MPI_DenseVec colmaxs = A.Reduce(Column, maximum<double>(), 0.0);

	colmaxs -= colssqs;	// chaos indicator
	return colmaxs.Reduce(maximum<double>(), 0.0);
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
	
		Dist<double>::MPI_DCCols A;	// construct object
		A.ReadDistribute(input, 0);	// read it from file
	
		input.clear();
		input.close();

		// chaos doesn't make sense for non-stochastic matrices	
		// it is in the range {0,1} for stochastic matrices
		double chaos = 1000;

		// while there is an epsilon improvement
		while( chaos > EPS)
		{
			double t1 = MPI_Wtime();
			A.Square<PTDOUBLEDOUBLE>() ;		// expand 
			
			chaos = Inflate(A, inflation);	// inflate (and renormalize)

			stringstream s;
			s << "New chaos: " << chaos << '\n';
			SpParHelper::Print(s.str());
			
#ifdef DEBUG	
			SpParHelper::Print("Before pruning...\n");
			A.PrintInfo();
#endif
			A.Prune(bind2nd(less<double>(), prunelimit));
			
			double t2=MPI_Wtime();
			if(myrank == 0)
				printf("%.6lf seconds elapsed for this iteration\n", (t2-t1));

#ifdef DEBUG	
			SpParHelper::Print("After pruning...\n");
			A.PrintInfo();
#endif
		}
		Interpret(A);	
	}	

	// make sure the destructors for all objects are called before MPI::Finalize()
	MPI::Finalize();	
	return 0;
}
