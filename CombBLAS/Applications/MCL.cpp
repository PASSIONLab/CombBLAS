/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.6 -------------------------------------------------*/
/* date: 05/15/2016 --------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc, Adam Lugowski ------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2016, The Regents of the University of California
 
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


class Dist
{ 
public: 
	typedef SpDCCols < int64_t, float > DCCols;
	typedef SpParMat < int64_t, float, DCCols > MPI_DCCols;
	typedef FullyDistVec < int64_t, float> MPI_DenseVec;
};


void Interpret(const Dist::MPI_DCCols & A)
{
	// Placeholder
}

void MakeColStochastic(Dist::MPI_DCCols & A)
{
    Dist::MPI_DenseVec colsums = A.Reduce(Column, plus<float>(), 0.0);
    colsums.Apply(safemultinv<float>());
    A.DimApply(Column, colsums, multiplies<float>());	// scale each "Column" with the given vector
}

float Chaos(Dist::MPI_DCCols & A)
{
    // sums of squares of columns
    Dist::MPI_DenseVec colssqs = A.Reduce(Column, plus<float>(), 0.0, bind2nd(exponentiate(), 2));
    // Matrix entries are non-negative, so max() can use zero as identity
    Dist::MPI_DenseVec colmaxs = A.Reduce(Column, maximum<float>(), 0.0);
    colmaxs -= colssqs;
    
    // multiplu by number of nonzeros in each column
    Dist::MPI_DenseVec nnzPerColumn = A.Reduce(Column, plus<float>(), 0.0, [](float val){return 1.0;});
    colmaxs.EWiseApply(nnzPerColumn, multiplies<float>());
    
    return colmaxs.Reduce(maximum<float>(), 0.0);
}


void Inflate(Dist::MPI_DCCols & A, float power)
{
	A.Apply(bind2nd(exponentiate(), power));
}


int main(int argc, char* argv[])
{
    
    int provided;
    
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (provided < MPI_THREAD_SERIALIZED)
    {
        printf("ERROR: The MPI library does not have MPI_THREAD_SERIALIZED support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    int nthreads;
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
    

	int nprocs, myrank;
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    if(myrank == 0)
    {
        cout << "Process Grid (p x p x t): " << sqrt(nprocs) << " x " << sqrt(nprocs) << " x " << nthreads << endl;
    }
    
	typedef PlusTimesSRing<float, float> PTFF;
    if(argc < 9)
    {
        if(myrank == 0)
        {
            cout << "Usage: ./mcl <FILENAME_MATRIX_MARKET> <INFLATION> <PRUNELIMIT> <KSELECT> <RECOVER NUMBER> <RECOVER PCT> <BASE_OF_MM> <RANDPERMUTE> [PHASES]" << endl;
            cout << "Example (0-indexed mtx and random permutation on): ./mcl input.mtx 2 0.0001 500 600 0.9 0 1" << endl;
            cout << "Example with two phases in SpGEMM: ./mcl input.mtx 2 0.0001 500 600 0.9 0 1 2" << endl;
        }
        MPI_Finalize();
        return -1;
    }
	{
		float inflation = atof(argv[2]);
		float prunelimit = atof(argv[3]);
        int64_t select = atoi(argv[4]);
        int64_t recover_num = atoi(argv[5]);
        float recover_pct = atoi(argv[6]);
        
        int phases = 1;
        if(argc > 9)
        {
            phases = atoi(argv[9]);
        }
        int randpermute = atoi(argv[8]);
        
        ostringstream runinfo;
        runinfo << "Running with... " << endl;
        runinfo << "Inflation: " << inflation << endl;
        runinfo << "Prunelimit: " << prunelimit << endl;
        runinfo << "Recover number: " << recover_num << endl;
        runinfo << "Recover percent: " << recover_pct << endl;
        runinfo << "Maximum column nonzeros: " << select << " in " << phases << " phases "<< endl;
        SpParHelper::Print(runinfo.str());

		string ifilename(argv[1]);

        double tIO = MPI_Wtime();
		Dist::MPI_DCCols A;	// construct object
		if(string(argv[7]) == "0")
		{
            SpParHelper::Print("Treating input zero based\n");
            A.ParallelReadMM(ifilename, false, maximum<float>());	// use zero-based indexing for matrix-market file
		}
		else
		{
            A.ParallelReadMM(ifilename, true, maximum<float>());
		}
		
        ostringstream outs;
        outs << "File Read time: " << MPI_Wtime() - tIO << endl;
		SpParHelper::Print(outs.str());
        
        if(randpermute)
        {
            // randomly permute for load balance
            if(A.getnrow() == A.getncol())
            {
                FullyDistVec<int64_t, int64_t> p( A.getcommgrid());
                p.iota(A.getnrow(), 0);
                p.RandPerm();
                (A)(p,p,true);// in-place permute to save memory
                SpParHelper::Print("Applied symmetric permutation.\n");
            }
            else
            {
                SpParHelper::Print("Rectangular matrix: Can not apply symmetric permutation.\n");
            }
        }
        
        
		float balance = A.LoadImbalance();
		int64_t nnz = A.getnnz();
        outs.str("");
        outs.clear();
		outs << "Load balance: " << balance << endl;
		outs << "Nonzeros: " << nnz << endl;
		SpParHelper::Print(outs.str());
        
        
        double tstart = MPI_Wtime();
        
        Dist::MPI_DenseVec colmaxs = A.Reduce(Column, maximum<float>(), 1.0);
        A.AddLoops(colmaxs);
		//A.AddLoops(1.0);	// matrix_add_loops($mx); // with weight 1.0
        outs.str("");
        outs.clear();
        outs << "Added loops" << endl;
        SpParHelper::Print(outs.str());
        A.PrintInfo();
        
        Inflate(A, 1); 		// matrix_make_stochastic($mx);
        float initChaos = Chaos(A);
        outs.str("");
        outs.clear();
        outs << "Made stochastic" << endl;
        outs << "Initial chaos = " << initChaos << endl;
        SpParHelper::Print(outs.str());
        A.PrintInfo();
        

	
		// chaos doesn't make sense for non-stochastic matrices	
		// it is in the range {0,1} for stochastic matrices
		float chaos = 1;
        int it=1;

		// while there is an epsilon improvement
		while( chaos > EPS)
		{
			double t1 = MPI_Wtime();
			//A.Square<PTFF>() ;		// expand
            A = MemEfficientSpGEMM<PTFF, float, Dist::DCCols>(A, A, phases, prunelimit,select, recover_num, recover_pct);
            MakeColStochastic(A);
            double t2 = MPI_Wtime();
            stringstream ss;
            ss << "Squared in " << (t2-t1) << " seconds" << endl;
            SpParHelper::Print(ss.str());
            A.PrintInfo();

            chaos = Chaos(A);
            
			Inflate(A, inflation);	// inflate (and renormalize)
            MakeColStochastic(A);
            
            stringstream sss;
            sss << "Inflated in " << (MPI_Wtime()-t2) << " seconds" << endl;
            SpParHelper::Print(sss.str());
            A.PrintInfo();
            
			
            // Prunning is performed inside MemEfficientSpGEMM
            /*
#ifdef DEBUG	
			SpParHelper::Print("Before pruning...\n");
			A.PrintInfo();
#endif
			A.Prune(bind2nd(less<float>(), prunelimit));
             */
            
            float newbalance = A.LoadImbalance();
			double t3=MPI_Wtime();
            stringstream s;
            s << "Iteration: " << it << " chaos: " << chaos << "  load-balance: "<< newbalance << " time: " << (t3-t1) << endl;
            SpParHelper::Print(s.str());
            it++;

#ifdef DEBUG	
			SpParHelper::Print("After pruning...\n");
			A.PrintInfo();
#endif
		}
		Interpret(A);
        
        double tend = MPI_Wtime();
        stringstream s2;
        s2 << "=====================================\n" ;
        s2 << "Total time: " << (tend-tstart) << endl;
        SpParHelper::Print(s2.str());

	}	

    
    
	// make sure the destructors for all objects are called before MPI::Finalize()
	MPI_Finalize();	
	return 0;
}
