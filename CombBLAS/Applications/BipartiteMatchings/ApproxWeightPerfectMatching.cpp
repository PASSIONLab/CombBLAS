
#ifdef THREADED
#ifndef _OPENMP
#define _OPENMP
#endif

#include <omp.h>
int cblas_splits = 1;
#endif

#include "CombBLAS/CombBLAS.h"
#include <mpi.h>
#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>
#include <limits>


#include "BPMaximalMatching.h"
#include "BPMaximumMatching.h"
#include "ApproxWeightPerfectMatching.h"

using namespace std;
using namespace combblas;

typedef SpParMat < int64_t, bool, SpDCCols<int64_t,bool> > Par_DCSC_Bool;
typedef SpParMat < int64_t, int64_t, SpDCCols<int64_t, int64_t> > Par_DCSC_int64_t;
typedef SpParMat < int64_t, double, SpDCCols<int64_t, double> > Par_DCSC_Double;
typedef SpParMat < int64_t, double, SpCCols<int64_t, double> > Par_CSC_Double;
typedef SpParMat < int64_t, bool, SpCCols<int64_t,bool> > Par_CSC_Bool;

void ShowUsage()
{
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    if(myrank == 0)
    {
        cout << "\n-------------- usage --------------\n";
        cout << "Usage: ./awpm -input <filename>\n";
        cout << "Optional parameters: -randPerm: randomly permute the matrix for load balance (default: no random permutation)\n";
        cout << "                     -optsum: Optimize the sum of diagonal (default: Optimize the product of diagonal)\n";
        cout << "                     -noWeightedCard: do not use weighted cardinality matching (default: use weighted cardinality matching)\n";
        cout << "                     -output <output file>: output file name \n";
        //cout << "                     -saveMCM <output file>: output file where maximum cardinality matching is saved \n";
        cout << " \n-------------- examples ----------\n";
        cout << "Example: mpirun -np 4 ./awpm -input cage12.mtx \n" << endl;
        cout << "(output matching is saved to cage12.mtx.awpm.txt)\n" << endl;
    }
}







int main(int argc, char* argv[])
{
    
    // ------------ initialize MPI ---------------
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (provided < MPI_THREAD_SERIALIZED)
    {
        printf("ERROR: The MPI library does not have MPI_THREAD_SERIALIZED support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    if(argc < 3)
    {
        ShowUsage();
        MPI_Finalize();
        return -1;
    }
    
    bool randPerm = false;
    bool optimizeProd = true; // by default optimize sum_log_abs(aii) (after equil)
    
    bool weightedCard = true;
    string ifilename = "";
    string ofname = "";
    //string ofnameMCM = "";
    for(int i = 1; i<argc; i++)
    {
        if (string(argv[i]) == string("-input")) ifilename = argv[i+1];
        if (string(argv[i]) == string("-output")) ofname = argv[i+1];
        //if (string(argv[i]) == string("-saveMCM")) ofnameMCM = argv[i+1];
        if (string(argv[i]) == string("-optsum")) optimizeProd = false;
        if (string(argv[i]) == string("-noWeightedCard")) weightedCard = false;
        if (string(argv[i]) == string("-randPerm")) randPerm = true;
    }

    
    
    
    // ------------ Process input arguments and build matrix ---------------
    {
        Par_DCSC_Double * AWeighted;
        ostringstream tinfo;
        tinfo << fixed;
        cout << fixed;
        double t01, t02;
        if(ifilename!="")
        {
            AWeighted = new Par_DCSC_Double(MPI_COMM_WORLD);
            t01 = MPI_Wtime();
            AWeighted->ParallelReadMM(ifilename, true, maximum<double>()); // one-based matrix market file
            t02 = MPI_Wtime();
       
            if(AWeighted->getnrow() != AWeighted->getncol())
            {
                 SpParHelper::Print("Rectangular matrix: Can not compute a perfect matching.\n");
                MPI_Finalize();
                return -1;
            }
            
            tinfo.str("");
            tinfo << "Input file name: " << ifilename << endl;
            tinfo << "Reading the input matrix in" << t02-t01 << " seconds" << endl;
            SpParHelper::Print(tinfo.str());
            
            SpParHelper::Print("Pruning explicit zero entries\n");
            AWeighted->Prune([](double val){return fabs(val)==0;}, true);
            
            AWeighted->PrintInfo();
        }
        else
        {
            ShowUsage();
            MPI_Finalize();
            return -1;
        }
        
        
        
        // ***** careful: if you permute the matrix, you have the permute the matching vectors as well!!
        // randomly permute for load balance
        
        FullyDistVec<int64_t, int64_t> randp( AWeighted->getcommgrid());
        if(randPerm)
        {
            if(AWeighted->getnrow() == AWeighted->getncol())
            {
                randp.iota(AWeighted->getnrow(), 0);
                randp.RandPerm();
		double oldbalance = AWeighted->LoadImbalance();
                (*AWeighted)(randp,randp,true);
 		double newbalance = AWeighted->LoadImbalance();
                SpParHelper::Print("Matrix is randomly permuted for load balance.\n");
		stringstream s;
        	s << "load-balance: before:"  << oldbalance << " after:" << newbalance << endl;
        	SpParHelper::Print(s.str());
            }
            else
            {
                SpParHelper::Print("Rectangular matrix: Can not apply symmetric permutation.\n");
            }
        }
        
       
        Par_DCSC_Bool A = *AWeighted; //just to compute degree
        // Reduce is not multithreaded, so I am doing it here
        FullyDistVec<int64_t, int64_t> degCol(A.getcommgrid());
        A.Reduce(degCol, Column, plus<int64_t>(), static_cast<int64_t>(0));
      
        int64_t maxdeg = degCol.Reduce(maximum<int64_t>(), static_cast<int64_t>(0));
        tinfo.str("");
        tinfo << "Maximum degree: " << maxdeg << endl;
        SpParHelper::Print(tinfo.str());

        SpParHelper::Print("Preprocessing is done.\n");
        SpParHelper::Print("----------------------------------------\n");
        
        FullyDistVec<int64_t, int64_t> mateRow2Col ( A.getcommgrid(), A.getnrow(), (int64_t) -1);
        FullyDistVec<int64_t, int64_t> mateCol2Row ( A.getcommgrid(), A.getncol(), (int64_t) -1);
        
        double startT = MPI_Wtime();
        AWPM(*AWeighted, mateRow2Col, mateCol2Row,  optimizeProd, weightedCard);
        double endT = MPI_Wtime();
        

        
        if(optimizeProd)
            TransformWeight(*AWeighted, true);
        else
            TransformWeight(*AWeighted, false);
        int64_t diagnnz;
        double origWeight = Trace(*AWeighted, diagnnz);
        double mWeight =  MatchingWeight( *AWeighted, mateRow2Col, mateCol2Row) ;
        tinfo.str("");
        tinfo  << "Matching is computed " << endl;
        tinfo  << "Sum of Diagonal (with transformation)" << endl;
        tinfo << "      After matching: "<< mWeight  << endl;
        tinfo << "      Before matching: " << origWeight << endl;
        
        tinfo  << "Time: " << endT - startT << endl;
        tinfo  << "----------------------------------------\n";
        SpParHelper::Print(tinfo.str());
        
        //revert random permutation if applied before
        if(randPerm==true && randp.TotalLength() >0)
        {
            // inverse permutation
            FullyDistVec<int64_t, int64_t>invRandp = randp.sort();
            mateRow2Col = mateRow2Col(invRandp);
        }
        if(ofname!="")
        {
            mateRow2Col.ParallelWrite(ofname,false,false);
        }
        
        
    }
    MPI_Finalize();
    return 0;
}


