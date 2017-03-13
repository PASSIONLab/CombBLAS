/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.6 -------------------------------------------------*/
/* date: 11/15/2016 --------------------------------------------*/
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
#include "CC.h"

using namespace std;

#define EPS 0.0001

double mcl_Abcasttime;
double mcl_Bbcasttime;
double mcl_localspgemmtime;
double mcl_multiwaymergetime;
double mcl_kselecttime;
double mcl_prunecolumntime;
int64_t mcl_memory;



class Dist
{ 
public: 
	typedef SpDCCols < int64_t, double > DCCols;
	typedef SpParMat < int64_t, double, DCCols > MPI_DCCols;
	typedef FullyDistVec < int64_t, double> MPI_DenseVec;
};


void Interpret(Dist::MPI_DCCols & A)
{
    int64_t nCC;
    // A is a directed graph
    // Since we need weekly connected components, we symmetricize A
    Dist::MPI_DCCols AT = A;
    AT.Transpose();
    A += AT;
    FullyDistVec<int64_t, int64_t> cclabels = CC(A, nCC);
}

void MakeColStochastic(Dist::MPI_DCCols & A)
{
    Dist::MPI_DenseVec colsums = A.Reduce(Column, plus<double>(), 0.0);
    colsums.Apply(safemultinv<double>());
    A.DimApply(Column, colsums, multiplies<double>());	// scale each "Column" with the given vector
}

double Chaos(Dist::MPI_DCCols & A)
{
    // sums of squares of columns
    Dist::MPI_DenseVec colssqs = A.Reduce(Column, plus<double>(), 0.0, bind2nd(exponentiate(), 2));
    // Matrix entries are non-negative, so max() can use zero as identity
    Dist::MPI_DenseVec colmaxs = A.Reduce(Column, maximum<double>(), 0.0);
    colmaxs -= colssqs;
    
    // multiplu by number of nonzeros in each column
    Dist::MPI_DenseVec nnzPerColumn = A.Reduce(Column, plus<double>(), 0.0, [](double val){return 1.0;});
    colmaxs.EWiseApply(nnzPerColumn, multiplies<double>());
    
    return colmaxs.Reduce(maximum<double>(), 0.0);
}


void Inflate(Dist::MPI_DCCols & A, double power)
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
    
    int nthreads = 1;
#ifdef THREADED
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
#endif

	int nprocs, myrank;
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    if(myrank == 0)
    {
        cout << "Process Grid (p x p x t): " << sqrt(nprocs) << " x " << sqrt(nprocs) << " x " << nthreads << endl;
    }
    
	typedef PlusTimesSRing<double, double> PTFF;
    if(argc < 3)
    {
        if(myrank == 0)
        {
            cout << "Usage: ./mcl -M <FILENAME_MATRIX_MARKET> (required)\n";
            cout << "-I <INFLATION> (default: 2)\n";
            cout << "-p <CUTOFF> (default: 1/10000)\n";
            cout << "-S <SELECTION NUMBER> (default: 1100)\n";
            cout << "-R <RECOVER NUMBER> (default: 900)\n";
            cout << "-pct <RECOVER PCT> (default: 90)\n";
            cout << "-base <BASE OF MATRIX MARKET> (default:1)\n";
            cout << "-rand <RANDOMLY PERMUTE VERTICES> (default:0)\n";
            cout << "-phases <NUM PHASES in SPGEMM> (default:1)\n";
            cout << "Example (0-indexed mtx and random permutation on): ./mcl -M input.mtx -I 2 -p 0.0001 -S 1100 -R 1400 -pct 0.9 -base 0 -rand 1 -phases 1" << endl;
        }
        MPI_Finalize();
        return -1;
    }
	{
        // default parameters of mac can be found by #mcl -z
        string ifilename = "";
        double inflation = 2.0;
        double prunelimit = 1.0/10000.0;
        int64_t select = 1100;
        int64_t recover_num = 1400;
        double recover_pct = .9; // TODO: make it consistent with mcl by representing it as percentage
        int base = 1;
        int randpermute = 0;
        int phases = 1;
        bool show = false;
        bool keep_isolated = false; // mcl removes isolated vertices by default
        
        for (int i = 1; i < argc; i++)
        {
            if (strcmp(argv[i],"-M")==0){
                ifilename = string(argv[i+1]);
                if(myrank == 0) printf("filename: %s",ifilename.c_str());}
            else if (strcmp(argv[i],"--show")==0){
                show = true;
                if(myrank == 0) printf("\nShow matrices after major steps");
            }
            else if (strcmp(argv[i],"--keep-isolated")==0){
                keep_isolated = true;
                if(myrank == 0) printf("\nKeep isolated vertices at the beginning");
            }
            else if (strcmp(argv[i],"-I")==0){
                inflation = atof(argv[i + 1]);
                if(myrank == 0) printf("Inflation: %f",inflation);
            } else if (strcmp(argv[i],"-p")==0) {
                prunelimit = atof(argv[i + 1]);
                if(myrank == 0) printf("\nCutoff:%f",prunelimit);
            } else if (strcmp(argv[i],"-S")==0) {
                select = atoi(argv[i + 1]);
                if(myrank == 0) printf("\nSelection Number:%lld",select);
            } else if (strcmp(argv[i],"-R")==0) {
                recover_num = atoi(argv[i + 1]);
                if(myrank == 0) printf("\nRecovery Number:%lld",recover_num);
            } else if (strcmp(argv[i],"-pct")==0) {
                recover_pct = atof(argv[i + 1]);
                if(myrank == 0) printf("\nRecovery Percentage:%f",recover_pct);
            } else if (strcmp(argv[i],"-base")==0) {
                base = atoi(argv[i + 1]);
                if(myrank == 0) printf("\nBase of MM (1 or 0):%d",base);
            }
            else if (strcmp(argv[i],"-rand")==0) {
                randpermute = atoi(argv[i + 1]);
                if(myrank == 0) printf("\nRandomly permute the matrix? (1 or 0):%d",randpermute);
            }
            else if (strcmp(argv[i],"-phases")==0) {
                phases = atoi(argv[i + 1]);
                if(myrank == 0) printf("\nNumber of SpGEMM phases:%d",phases);
            }
        }

        ostringstream runinfo;
        runinfo << "\nRunning HipMCL with... " << endl;
        runinfo << "Inflation: " << inflation << endl;
        runinfo << "Prunelimit: " << prunelimit << endl;
        runinfo << "Recover number: " << recover_num << endl;
        runinfo << "Recover percent: " << recover_pct << endl;
        runinfo << "Selection number: " << select << " in " << phases << " phases "<< endl;
        SpParHelper::Print(runinfo.str());

        
        double tIO = MPI_Wtime();
		Dist::MPI_DCCols A;	// construct object
        A.ParallelReadMM(ifilename, base, maximum<double>());	// if base=0, then it is implicitly converted to Boolean false
						
        ostringstream outs;
        outs << "File Read time: " << MPI_Wtime() - tIO << endl;
		SpParHelper::Print(outs.str());
        if(show)
            A.PrintInfo();
        
        if(!keep_isolated)
        {
            FullyDistVec<int64_t,double> ColSums = A.Reduce(Column, plus<double>(), 0.0);
            FullyDistVec<int64_t, int64_t> nonisov = ColSums.FindInds(bind2nd(greater<double>(), 0));
            A(nonisov, nonisov, true);
            SpParHelper::Print("Removed isolated vertices.\n");
            if(show)
            {
                A.PrintInfo();
            }
        }
        
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
        
        
	double balance = A.LoadImbalance();
	int64_t nnz = A.getnnz();
        outs.str("");
        outs.clear();
	outs << "Load balance: " << balance << endl;
	outs << "Nonzeros: " << nnz << endl;
	SpParHelper::Print(outs.str());
        
        
        double tstart = MPI_Wtime();
        
        // Precossing: default adjustloop setting
        // 1. Remove loops
        // 2. set loops to max of all arc weights
        A.RemoveLoops();
        Dist::MPI_DenseVec colmaxs = A.Reduce(Column, maximum<double>(), numeric_limits<double>::min());
        A.Apply([](double val){return val==numeric_limits<double>::min() ? 1.0 : val;}); // for isolated vertices
        A.AddLoops(colmaxs);
        outs.str("");
        outs.clear();
        outs << "Adjusted loops according to default mcl parameters" << endl;
        SpParHelper::Print(outs.str());
        if(show)
        {
            A.PrintInfo();
        }
        
        MakeColStochastic(A);
        //Inflate(A, 1); 		// matrix_make_stochastic($mx);
        //double initChaos = Chaos(A);
        //outs << "Initial chaos = " << initChaos << endl;
        SpParHelper::Print("Made stochastic\n");
        if(show)
        {
            A.PrintInfo();
        }
        

        
#ifdef TIMING
        mcl_Abcasttime = 0;
        mcl_Bbcasttime = 0;
        mcl_localspgemmtime = 0;
        mcl_multiwaymergetime = 0;
        mcl_kselecttime = 0;
        mcl_prunecolumntime = 0;
#endif

		// chaos doesn't make sense for non-stochastic matrices
		// it is in the range {0,1} for stochastic matrices
		double chaos = 1;
        int it=1;

		// while there is an epsilon improvement
		while( chaos > EPS)
		{
            /*
#ifdef TIMING
            double mcl_Abcasttime1=mcl_Abcasttime;
            double mcl_Bbcasttime1=mcl_Bbcasttime;
            double mcl_localspgemmtime1=mcl_localspgemmtime;
            double mcl_multiwaymergetime1 = mcl_multiwaymergetime;
            double mcl_kselecttime1=mcl_kselecttime;
            double mcl_prunecolumntime1=mcl_prunecolumntime;
#endif
             */
			double t1 = MPI_Wtime();
			//A.Square<PTFF>() ;		// expand
            A = MemEfficientSpGEMM<PTFF, double, Dist::DCCols>(A, A, phases, prunelimit,select, recover_num, recover_pct, 64); // 64 GB memory

            MakeColStochastic(A);
            //double t2 = MPI_Wtime();
            //stringstream ss;
            //ss << "=================================================" << endl;
            //ss << "Squared in " << (t2-t1) << " seconds" << endl;
            //SpParHelper::Print(ss.str());
/*
#ifdef TIMING
            if(myrank==0)
            {
                cout << "Breakdown of squaring time: \n mcl_Abcast= " << mcl_Abcasttime - mcl_Abcasttime1 << "\n mcl_Bbcast= " << mcl_Bbcasttime - mcl_Bbcasttime1 << "\n mcl_localspgemm= " << mcl_localspgemmtime-mcl_localspgemmtime1 << "\n mcl_multiwaymergetime= "<< mcl_multiwaymergetime-mcl_multiwaymergetime1 << "\n mcl_kselect= " << mcl_kselecttime-mcl_kselecttime1 << "\n mcl_prunecolumn= " << mcl_prunecolumntime - mcl_prunecolumntime1 << endl;
                //cout << "=================================================" << endl;
            }
#endif
 */

            if(show)
            {
                SpParHelper::Print("After expansion\n");
                A.PrintInfo();
            }
            chaos = Chaos(A);
            
	    
            Inflate(A, inflation);	// inflate (and renormalize)
            MakeColStochastic(A);
            
            //stringstream sss;
            //sss << "Inflated in " << (MPI_Wtime()-t2) << " seconds" << endl;
            //SpParHelper::Print(sss.str());
            if(show)
            {
                SpParHelper::Print("After inflation\n");
                A.PrintInfo();
            }
            
			
            
            double newbalance = A.LoadImbalance();
			double t3=MPI_Wtime();
            stringstream s;
            s << "Iteration: " << std::setw(3) << it << " chaos: " << setprecision(3) << chaos << " nnz: " << A.getnnz() << "  load-balance: "<< newbalance << " Total time: " << (t3-t1) << endl;
            SpParHelper::Print(s.str());
            it++;
            


		}
        
        double tcc1 = MPI_Wtime();
		Interpret(A);
        double tcc = MPI_Wtime() - tcc1;
        
        double tend = MPI_Wtime();
        stringstream s2;
        s2 << "=====================================\n" ;
        s2 << "Total time: " << (tend-tstart) << endl;
        SpParHelper::Print(s2.str());
        
        if(myrank==0)
        {
            cout << "Breakdown of squaring time: \n mcl_Abcast= " << mcl_Abcasttime << "\n mcl_Bbcast= " << mcl_Bbcasttime << "\n mcl_localspgemm= " << mcl_localspgemmtime << "\n mcl_multiwaymergetime= "<< mcl_multiwaymergetime << "\n mcl_kselect= " << mcl_kselecttime << "\n mcl_prunecolumn= " << mcl_prunecolumntime << endl;
            cout << "Connected Component time: " << tcc << endl;
            cout << "=================================================" << endl;
        }

	}

    
    
	// make sure the destructors for all objects are called before MPI::Finalize()
	MPI_Finalize();	
	return 0;
}
