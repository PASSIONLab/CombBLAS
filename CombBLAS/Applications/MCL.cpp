/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.6 -------------------------------------------------*/
/* date: 6/15/2017 ---------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc  --------------------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2017, The Regents of the University of California
 
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
double cblas_allgathertime;	// for compilation (TODO: fix this dependency)
int64_t mcl_memory;
double tIO;



class Dist
{
    public:
    typedef SpDCCols < int64_t, double > DCCols;
    typedef SpParMat < int64_t, double, DCCols > MPI_DCCols;
    typedef FullyDistVec < int64_t, double> MPI_DenseVec;
};


typedef struct
{
    //Input/Output file
    string ifilename;
    int base;
    string ofilename;
    
    //Preprocessing
    int randpermute;
    bool remove_isolated;
    
    //inflation
    double inflation;
    
    //pruning
    double prunelimit;
    int64_t select;
    int64_t recover_num;
    double recover_pct;
    int kselectVersion; // 0: adapt based on k, 1: kselect1, 2: kselect2
    
    //HipMCL optimization
    int phases;
    int perProcessMem;
    
    //debugging
    bool show;
    
    
}HipMCLParam;


void InitParam(HipMCLParam & param)
{
    //Input/Output file
    param.ifilename = "";
    param.ofilename = "";
    param.base = 1;
    
    //Preprocessing
    // mcl removes isolated vertices by default,
    // we don't do this because it will create different ordering of vertices!
    param.remove_isolated = false;
    param.randpermute = 0;
    
    //inflation
    param.inflation = 0.0;
    
    //pruning
    param.prunelimit = 1.0/10000.0;
    param.select = 1100;
    param.recover_num = 1400;
    param.recover_pct = .9; // we allow both 90 or .9 as input. Internally, we keep it 0.9
    param.kselectVersion = 1;
    
    //HipMCL optimization
    param.phases = 1;
    param.perProcessMem = 0;
    
    //debugging
    param.show = false;
}

void ShowParam(HipMCLParam & param)
{
    ostringstream runinfo;
    runinfo << "\n======================================" << endl;
    runinfo << "Running HipMCL with the parameters: " << endl;
    runinfo << "======================================" << endl;
    runinfo << "Input/Output file" << endl;
    runinfo << "    input filename: " << param.ifilename << endl;
    runinfo << "    Base of the input matrix: " << param.base << endl;
    runinfo << "    Output filename: " << param.ofilename << endl;
    
    
    runinfo << "Preprocessing" << endl;
    runinfo << "    Remove isolated vertices? : ";
    if (param.remove_isolated) runinfo << "yes";
    else runinfo << "no" << endl;
    
    runinfo << "    Randomly permute vertices? : ";
    if (param.randpermute) runinfo << "yes";
    else runinfo << "no" << endl;
    
    runinfo << "Inflation: " << param.inflation << endl;
    
    runinfo << "Pruning" << endl;
    runinfo << "    Prunelimit: " << param.prunelimit << endl;
    runinfo << "    Recover number: " << param.recover_num << endl;
    runinfo << "    Recover percent: " << ceil(param.recover_pct*100) << endl;
    runinfo << "    Selection number: " << param.select << endl;
    // do not expose selection option at this moment
    //runinfo << "Selection algorithm: ";
    //if(kselectVersion==1) runinfo << "tournament select" << endl;
    //else if(kselectVersion==2) runinfo << "quickselect" << endl;
    //else runinfo << "adaptive based on k" << endl;
    
    
    
    runinfo << "HiMCL optimization" << endl;
    runinfo << "    Number of phases: " << param.phases << endl;
    runinfo << "    Memory avilable per process: ";
    if(param.perProcessMem>0) runinfo << param.perProcessMem << "GB" << endl;
    else runinfo << "not provided" << endl;
    
    runinfo << "Debugging" << endl;
    runinfo << "    Show matrices after major steps? : ";
    if (param.show) runinfo << "yes";
    else runinfo << "no" << endl;
    runinfo << "======================================" << endl;
    SpParHelper::Print(runinfo.str());
}

void ProcessParam(int argc, char* argv[], HipMCLParam & param)
{
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i],"-M")==0){
            param.ifilename = string(argv[i+1]);
        }
        else if (strcmp(argv[i],"-o")==0){
            param.ofilename = string(argv[i+1]);
        }
        else if (strcmp(argv[i],"--show")==0){
            param.show = true;
        }
        else if (strcmp(argv[i],"--remove-isolated")==0){
            param.remove_isolated = true;
        }
        else if (strcmp(argv[i],"--tournament-select")==0){
            param.kselectVersion = 1;
        }
        else if (strcmp(argv[i],"--quick-select")==0){
            param.kselectVersion = 2;
            
        }
        else if (strcmp(argv[i],"-I")==0){
            param.inflation = atof(argv[i + 1]);
            
        } else if (strcmp(argv[i],"-p")==0) {
            param.prunelimit = atof(argv[i + 1]);
            
        } else if (strcmp(argv[i],"-S")==0) {
            param.select = atoi(argv[i + 1]);
            
        } else if (strcmp(argv[i],"-R")==0) {
            param.recover_num = atoi(argv[i + 1]);
            
        } else if (strcmp(argv[i],"-pct")==0)
        {
            param.recover_pct = atof(argv[i + 1]);
            if(param.recover_pct>1) param.recover_pct/=100.00;
        } else if (strcmp(argv[i],"-base")==0) {
            param.base = atoi(argv[i + 1]);
        }
        else if (strcmp(argv[i],"-rand")==0) {
            param.randpermute = atoi(argv[i + 1]);
        }
        else if (strcmp(argv[i],"-phases")==0) {
            param.phases = atoi(argv[i + 1]);
        }
        else if (strcmp(argv[i],"-per-process-mem")==0) {
            param.perProcessMem = atoi(argv[i + 1]);
        }
    }
    
    if(param.ofilename=="") // construct output file name if it is not provided
    {
        param.ofilename = param.ifilename + ".hipmcl";
    }
    
}


void ShowOptions()
{
    ostringstream runinfo;
    
    runinfo << "Usage: ./hipmcl -M <input filename> -I <inlfation> (required)" << endl;
    
    runinfo << "======================================" << endl;
    runinfo << "     Detail parameter options    " << endl;
    runinfo << "======================================" << endl;
    
    
    
    runinfo << "Input/Output file" << endl;
    runinfo << "    -M <input file name> (mandatory)" << endl;
    runinfo << "    -base <index of the first vertex, 0|1> (default: 1) " << endl;
    runinfo << "    -o <output filename> (default: input_file_name.hipmcl )" << endl;
    
    runinfo << "Inflation" << endl;
    runinfo << "-I <inflation> (mandatory)\n";
    
    runinfo << "Preprocessing" << endl;
    runinfo << "    -rand <randomly permute vertices> (default:0)\n";
    runinfo << "    --remove-isolated : if provided, remove isolated vertices (default: don't remove isolated vertices)\n";
    
    
    runinfo << "Pruning" << endl;
    runinfo << "    -p <cutoff> (default: 1/10000)\n";
    runinfo << "    -R <recovery number> (default: 1400)\n";
    runinfo << "    -pct <recovery pct> (default: 90)\n";
    runinfo << "    -S <selection number> (default: 1100)\n";
    
    
    runinfo << "HipMCL optimization" << endl;
    runinfo << "    -phases <number of phases> (default:1)\n";
    runinfo << "    -per-process-mem <memory (GB) available per process> (default:0, number of phases is not estimated)\n" << endl;
    
    runinfo << "Debugging" << endl;
    runinfo << "    --show: show matrices after major steps (default: do not show matrices)" << endl;


    
    runinfo << "======================================" << endl;
    runinfo << "     Few examples    " << endl;
    runinfo << "======================================" << endl;
    runinfo << "Example on a laption with 0-indexed matrix and random permutation on:\n./hipmcl -M graph.mtx -I 2 -base 0 -rand 1 -phases 1 -o graph.hipmcl" << endl;
    runinfo << "Example on the NERSC/Edison system with 16 nodes and 24 threads per node: \nsrun -N 16 -n 16 -c 24  ./hipmcl -M graph.mtx -per-process-mem 64 -o graph.hipmcl" << endl;
    SpParHelper::Print(runinfo.str());
}


// base: base of items
// clusters are always numbered 0-based
FullyDistVec<int64_t, int64_t> Interpret(Dist::MPI_DCCols & A)
{
    int64_t nCC;
    // A is a directed graph
    // Since we need weekly connected components, we symmetricize A
    Dist::MPI_DCCols AT = A;
    AT.Transpose();
    A += AT;
    FullyDistVec<int64_t, int64_t> cclabels = CC(A, nCC);
    return cclabels;
    //cclabels.ParallelWrite(ofilename, base); // clusters are always numbered 0-based
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

// default adjustloop setting
// 1. Remove loops
// 2. set loops to max of all arc weights
void AdjustLoops(Dist::MPI_DCCols & A)
{

    A.RemoveLoops();
    Dist::MPI_DenseVec colmaxs = A.Reduce(Column, maximum<double>(), numeric_limits<double>::min());
    A.Apply([](double val){return val==numeric_limits<double>::min() ? 1.0 : val;}); // for isolated vertices
    A.AddLoops(colmaxs);
    ostringstream outs;
    outs << "Adjusted loops according to default mcl parameters" << endl;
    SpParHelper::Print(outs.str());
}

void RemoveIsolated(Dist::MPI_DCCols & A, HipMCLParam & param)
{
    ostringstream outs;
    FullyDistVec<int64_t,double> ColSums = A.Reduce(Column, plus<double>(), 0.0);
    FullyDistVec<int64_t, int64_t> nonisov = ColSums.FindInds(bind2nd(greater<double>(), 0));
    int64_t numIsolated = A.getnrow() - nonisov.TotalLength();
    outs << "Number of isolated vertices: " << numIsolated << endl;
    SpParHelper::Print(outs.str());
    
    A(nonisov, nonisov, true);
    SpParHelper::Print("Removed isolated vertices.\n");
    if(param.show)
    {
        A.PrintInfo();
    }
    
}

//TODO: handle reordered cluster ids
void RandPermute(Dist::MPI_DCCols & A, HipMCLParam & param)
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

FullyDistVec<int64_t, int64_t> HipMCL(Dist::MPI_DCCols & A, HipMCLParam & param)
{
    if(param.remove_isolated)
        RemoveIsolated(A, param);
    
    if(param.randpermute)
        RandPermute(A, param);

    // Adjust self loops
    AdjustLoops(A);

    // Make stochastic
    MakeColStochastic(A);
    SpParHelper::Print("Made stochastic\n");

    if(param.show)
    {
        A.PrintInfo();
    }
    

    // chaos doesn't make sense for non-stochastic matrices
    // it is in the range {0,1} for stochastic matrices
    double chaos = 1;
    int it=1;
    double tInflate = 0;
    double tExpand = 0;
     typedef PlusTimesSRing<double, double> PTFF;
    // while there is an epsilon improvement
    while( chaos > EPS)
    {
        double t1 = MPI_Wtime();
        //A.Square<PTFF>() ;		// expand
        A = MemEfficientSpGEMM<PTFF, double, Dist::DCCols>(A, A, param.phases, param.prunelimit,param.select, param.recover_num, param.recover_pct, param.kselectVersion, param.perProcessMem);
        
        MakeColStochastic(A);
        tExpand += (MPI_Wtime() - t1);
        
        if(param.show)
        {
            SpParHelper::Print("After expansion\n");
            A.PrintInfo();
        }
        chaos = Chaos(A);
        
        double tInflate1 = MPI_Wtime();
        Inflate(A, param.inflation);
        MakeColStochastic(A);
        tInflate += (MPI_Wtime() - tInflate1);
        
        if(param.show)
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
    FullyDistVec<int64_t, int64_t> cclabels = Interpret(A);
    double tcc = MPI_Wtime() - tcc1;
    
    
#ifdef TIMING
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    if(myrank==0)
    {
        cout << "Squaring: " << mcl_Abcasttime + mcl_Bbcasttime + mcl_localspgemmtime + mcl_multiwaymergetime << endl;
        cout << "       Abcast= " << mcl_Abcasttime << endl;
        cout << "       Bbcast= " << mcl_Bbcasttime << endl;
        cout << "       localspgemm= " << mcl_localspgemmtime << endl;
        cout << "       multiwaymergetime= "<< mcl_multiwaymergetime << endl;
        cout << "Pruning: " << mcl_kselecttime + mcl_prunecolumntime << endl;
        cout << "       kselect= " << mcl_kselecttime << endl;
        cout << "       prunecolumn= " << mcl_prunecolumntime << endl;
        cout << "Inflation " << tInflate << endl;
        cout << "Component: " << tcc << endl;
        cout << "File I/O: " << tIO << endl;
        cout << "=================================================" << endl;
    }
    
#endif
    
    return cclabels;


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
    
    HipMCLParam param;
    InitParam(param); // initialize parameters to default values
    ProcessParam(argc, argv, param);
    if(param.ifilename=="" || param.inflation == 0.0)
    {
        SpParHelper::Print("Required options are missing.\n");
        ShowOptions();
        MPI_Finalize();
        return -1;
    }
    
    
    if(myrank == 0)
    {
        cout << "Process Grid (p x p x t): " << sqrt(nprocs) << " x " << sqrt(nprocs) << " x " << nthreads << endl;
    }
    ShowParam(param);
    
    
    {
        
        double tIO1 = MPI_Wtime();
        Dist::MPI_DCCols A;	// construct object
        
        //TODO: Put a command line option for them
        //A.ParallelReadMM(param.ifilename, param.base, maximum<double>());	// if base=0, then it is implicitly converted to Boolean false
        
        FullyDistVec<int64_t, array<char, MAXVERTNAME> > perm = A.ReadGeneralizedTuples(param.ifilename,  maximum<double>());
        tIO = MPI_Wtime() - tIO1;
        ostringstream outs;
        outs << "File Read time: " << tIO  << endl;
        SpParHelper::Print(outs.str());
        
        double balance = A.LoadImbalance();
        int64_t nnz = A.getnnz();
        outs.str("");
        outs.clear();
        outs << "Load balance: " << balance << endl;
        outs << "Nonzeros: " << nnz << endl;
        SpParHelper::Print(outs.str());
        
        if(param.show)
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
        
        
        
        double tstart = MPI_Wtime();
        
        // Run HipMCL
        FullyDistVec<int64_t, int64_t> culstLabels = HipMCL(A, param);
        culstLabels.ParallelWrite(param.ofilename, param.base); // clusters are always numbered 0-based
        
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
