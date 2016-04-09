/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.5 -------------------------------------------------*/
/* date: 10/09/2015 ---------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc, Adam Lugowski ------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2015, The Regents of the University of California
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */

#include <mpi.h>

// These macros should be defined before stdint.h is included
#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif
#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
#endif
#include <stdint.h>

#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>  // Required for stringstreams
#include <ctime>
#include <cmath>
#include "../CombBLAS.h"

using namespace std;

#define EPS 0.001

// Simple helper class for declarations: Just the numerical type is templated 
// The index type and the sequential matrix type stays the same for the whole code
// In this case, they are "int" and "SpDCCols"
template <class NT>
class Dist
{ 
public: 
	typedef SpDCCols < int64_t, NT > DCCols;
	typedef SpParMat < int64_t, NT, DCCols > MPI_DCCols;
	typedef FullyDistVec < int64_t, NT> MPI_DenseVec;
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
		A.DimApply(Column, colsums, multiplies<double>());	// scale each "Column" with the given vector

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
	int nprocs, myrank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
	typedef PlusTimesSRing<double, double> PTDOUBLEDOUBLE;
	if(argc < 4)
        {
		if(myrank == 0)
		{	
                	cout << "Usage: ./mcl <FILENAME_MATRIX_MARKET> <INFLATION> <PRUNELIMIT> <BASE_OF_MM>" << endl;
                	cout << "Example: ./mcl input.mtx 2 0.0001 0" << endl;
                }
		MPI_Finalize(); 
		return -1;
        }

	{
		double inflation = atof(argv[2]);
		double prunelimit = atof(argv[3]);

		string ifilename(argv[1]);		

		Dist<double>::MPI_DCCols A;	// construct object
		if(argv[4] == "0")
		{
			A.ParallelReadMM(ifilename, false);	// use zero-based indexing for matrix-market file
		}
		else
		{
			A.ParallelReadMM(ifilename);
		}
		
		SpParHelper::Print("File Read\n");
		float balance = A.LoadImbalance();
		int64_t nnz = A.getnnz();
		ostringstream outs;
		outs << "Load balance: " << balance << endl;
		outs << "Nonzeros: " << nnz << endl;
		SpParHelper::Print(outs.str());

		A.AddLoops(1.0);	// matrix_add_loops($mx); // with weight 1.0
		Inflate(A, 1); 		// matrix_make_stochastic($mx);

	
		// chaos doesn't make sense for non-stochastic matrices	
		// it is in the range {0,1} for stochastic matrices
		double chaos = 1;

		// while there is an epsilon improvement
		while( chaos > EPS)
		{
			double t1 = MPI_Wtime();
			A.Square<PTDOUBLEDOUBLE>() ;		// expand 
			// Dist<double>::MPI_DCCols TA = A;
			// A = PSpGEMM<PTDOUBLEDOUBLE>(TA, A);
			
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
	MPI_Finalize();	
	return 0;
}
