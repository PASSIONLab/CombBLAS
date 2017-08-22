
#ifdef THREADED
#ifndef _OPENMP
#define _OPENMP
#endif

#include <omp.h>
int cblas_splits = 1;
#endif

#include "../CombBLAS.h"
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

// algorithmic options
bool prune,randMM, moreSplit;
int init;
bool randMaximal;
bool fewexp;
bool randPerm;
bool saveMatching;
string ofname;


typedef SpParMat < int64_t, bool, SpDCCols<int64_t,bool> > Par_DCSC_Bool;
typedef SpParMat < int64_t, int64_t, SpDCCols<int64_t, int64_t> > Par_DCSC_int64_t;
typedef SpParMat < int64_t, double, SpDCCols<int64_t, double> > Par_DCSC_Double;
typedef SpParMat < int64_t, bool, SpCCols<int64_t,bool> > Par_CSC_Bool;

template <class IT, class NT, class DER>
void TransformWeight(SpParMat < IT, NT, DER > & A)
{
	//A.Apply([](NT val){return log(1+abs(val));});
	// if the matrix has explicit zero entries, we can still have problem.
	// One solution is to remove explicit zero entries before cardinality matching (to be tested)
	//A.Apply([](NT val){if(val==0) return log(numeric_limits<NT>::min()); else return log(fabs(val));});
	A.Apply([](NT val){return (fabs(val));});
	
	FullyDistVec<IT, NT> maxvRow(A.getcommgrid());
	A.Reduce(maxvRow, Row, maximum<NT>(), static_cast<NT>(numeric_limits<NT>::lowest()));
	A.DimApply(Row, maxvRow, [](NT val, NT maxval){return val/maxval;});
	
	FullyDistVec<IT, NT> maxvCol(A.getcommgrid());
	A.Reduce(maxvCol, Column, maximum<NT>(), static_cast<NT>(numeric_limits<NT>::lowest()));
	A.DimApply(Column, maxvCol, [](NT val, NT maxval){return val/maxval;});
	
	A.Apply([](NT val){return log(val);});
	
	//FullyDistVec<IT, NT> maxv(A.getcommgrid());
	//A.Reduce(maxv, Column, maximum<NT>(), static_cast<NT>(numeric_limits<NT>::lowest()));
	//A.DimApply(Column, maxv, [](NT val, NT maxval){return val - maxval;});
	//minv.Reduce(minimum<NT>(), static_cast<NT>(999999999999.0));
	//pair<IT, NT> x = minv.MinElement();
	//cout << "***** minimum value in the matrix: " << get<1>(x) << endl;
	//A.PrintInfo();

}
void ShowUsage()
{
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    if(myrank == 0)
    {
        cout << "\n-------------- usage --------------\n";
        cout << "Usage (input matrix): ./awpm input <matrix> saveMatching\n\n";
        cout << " \n-------------- examples ----------\n";
        cout << "Example: mpirun -np 4 ./bpmm input cage12.mtx saveMatching\n" << endl;
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
    
    init = DMD;
    randMaximal = false;
    prune = false;
    randMM = true;
    moreSplit = false;
    fewexp=false;
    saveMatching = true;
    ofname = "";
    randPerm = false;
    
    SpParHelper::Print("***** I/O and other preprocessing steps *****\n");
    // ------------ Process input arguments and build matrix ---------------
    {
        
        Par_DCSC_Bool * ABool;
        Par_DCSC_Double * AWighted;
        ostringstream tinfo;
        double t01, t02;
        if(string(argv[1]) == string("input")) // input option
        {
            AWighted = new Par_DCSC_Double();
            
            string filename(argv[2]);
            ofname = filename + ".matching.out";
            tinfo.str("");
            
            tinfo << "\n**** Reading input matrix: " << filename << " ******* " << endl;
            SpParHelper::Print(tinfo.str());
            t01 = MPI_Wtime();
            AWighted->ParallelReadMM(filename, true, maximum<double>()); // one-based matrix market file
            t02 = MPI_Wtime();
            AWighted->PrintInfo();
            tinfo.str("");
            tinfo << "Reader took " << t02-t01 << " seconds" << endl;
            SpParHelper::Print(tinfo.str());
            
            SpParHelper::Print("Pruning explicit zero entries....\n");
            AWighted->Prune([](double val){return fabs(val)==0;}, true);
            
            AWighted->PrintInfo();
            //GetOptions(argv+3, argc-3);
            /*
            FullyDistVec<int64_t, int64_t> prow(AWighted->getcommgrid());
            FullyDistVec<int64_t, int64_t> pcol(AWighted->getcommgrid());
            prow.iota(AWighted->getnrow(), 0);
            pcol.iota(AWighted->getncol(), 0);
            prow.RandPerm();
            pcol.RandPerm();
            (*AWighted)(prow, prow, true);
            
            AWighted->SaveGathered("test.rand.txt");
             */
        }
        else if(argc < 4)
        {
            ShowUsage();
            MPI_Finalize();
            return -1;
        }
        else
        {
            
            unsigned scale = (unsigned) atoi(argv[2]);
            unsigned EDGEFACTOR = (unsigned) atoi(argv[3]);
            double initiator[4];
            if(string(argv[1]) == string("er"))
            {
                initiator[0] = .25;
                initiator[1] = .25;
                initiator[2] = .25;
                initiator[3] = .25;
                if(myrank==0)
                cout << "Randomly generated ER matric\n";
            }
            else if(string(argv[1]) == string("g500"))
            {
                initiator[0] = .57;
                initiator[1] = .19;
                initiator[2] = .19;
                initiator[3] = .05;
                if(myrank==0)
                cout << "Randomly generated G500 matric\n";
            }
            else if(string(argv[1]) == string("ssca"))
            {
                initiator[0] = .6;
                initiator[1] = .4/3;
                initiator[2] = .4/3;
                initiator[3] = .4/3;
                if(myrank==0)
                cout << "Randomly generated SSCA matric\n";
            }
            else
            {
                if(myrank == 0)
                printf("The input type - %s - is not recognized.\n", argv[2]);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            
            SpParHelper::Print("Generating input matrix....\n");
            t01 = MPI_Wtime();
            DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>();
            DEL->GenGraph500Data(initiator, scale, EDGEFACTOR, true, true);
            AWighted = new Par_DCSC_Double(*DEL, false);
            // Add random weight ??
            delete DEL;
            t02 = MPI_Wtime();
            AWighted->PrintInfo();
            tinfo.str("");
            tinfo << "Generator took " << t02-t01 << " seconds" << endl;
            SpParHelper::Print(tinfo.str());
            
            Symmetricize(*AWighted);
            //removeIsolated(*ABool);
            SpParHelper::Print("Generated matrix symmetricized....\n");
            AWighted->PrintInfo();
            
            //GetOptions(argv+4, argc-4);
            
            
        }
        
        
        // ***** careful: if you permute the matrix, you have the permute the matching vectors as well!!
        // randomly permute for load balance
        if(randPerm)
        {
            SpParHelper::Print("Performing random permutation of matrix.\n");
            FullyDistVec<int64_t, int64_t> prow(AWighted->getcommgrid());
            FullyDistVec<int64_t, int64_t> pcol(AWighted->getcommgrid());
            prow.iota(AWighted->getnrow(), 0);
            pcol.iota(AWighted->getncol(), 0);
            prow.RandPerm();
            pcol.RandPerm();
            (*AWighted)(prow, pcol, true);
            SpParHelper::Print("Performed random permutation of matrix.\n");
        }
        
        Par_DCSC_Bool A = *AWighted;
        Par_DCSC_Bool AT = A;
        AT.Transpose();
        
        // Reduce is not multithreaded, so I am doing it here
        FullyDistVec<int64_t, int64_t> degCol(A.getcommgrid());
        A.Reduce(degCol, Column, plus<int64_t>(), static_cast<int64_t>(0));
        
        int nthreads;
#ifdef THREADED
#pragma omp parallel
        {
            int splitPerThread = 1;
            if(moreSplit) splitPerThread = 4;
            nthreads = omp_get_num_threads();
            cblas_splits = nthreads*splitPerThread;
        }
        tinfo.str("");
        tinfo << "Threading activated with " << nthreads << " threads, and matrix split into "<< cblas_splits <<  " parts" << endl;
        SpParHelper::Print(tinfo.str());
        A.ActivateThreading(cblas_splits); // note: crash on empty matrix
        AT.ActivateThreading(cblas_splits);
#endif
        
        
        SpParHelper::Print("**************************************************\n\n");
        
        // compute the maximum cardinality matching
        FullyDistVec<int64_t, int64_t> mateRow2Col ( A.getcommgrid(), A.getnrow(), (int64_t) -1);
        FullyDistVec<int64_t, int64_t> mateCol2Row ( A.getcommgrid(), A.getncol(), (int64_t) -1);
        
        // using best options for the maximum cardinality matching
        /*
         init = DMD; randMaximal = false; randMM = true; prune = true;
         MaximalMatching(A, AT, mateRow2Col, mateCol2Row, degCol, init, randMaximal);
         maximumMatching(A, mateRow2Col, mateCol2Row, prune, randMM);
         */
        
        double ts = MPI_Wtime();
		TransformWeight(*AWighted);
		
        init = DMD; randMaximal = false; randMM = false; prune = true;
        MaximalMatching(A, AT, mateRow2Col, mateCol2Row, degCol, init, randMaximal);
		//MaximalMatching(A, AT, mateRow2Col, mateCol2Row, degCol, GREEDY, randMaximal);
		//WeightedGreedy(*AWighted, mateRow2Col, mateCol2Row, degCol);
		cout << "Weight: " << MatchingWeight( *AWighted, mateRow2Col, mateCol2Row) << endl;
		CheckMatching(mateRow2Col,mateCol2Row);
		
		
        maximumMatching(A, mateRow2Col, mateCol2Row, prune, randMM);
		//maximumMatching(*AWighted, mateRow2Col, mateCol2Row, prune, false, true);
        cout << "Weight: " << MatchingWeight( *AWighted, mateRow2Col, mateCol2Row) << endl;
        double tcard = MPI_Wtime() - ts;
        CheckMatching(mateRow2Col,mateCol2Row);
        ts = MPI_Wtime();
        
		
        TwoThirdApprox(*AWighted, mateRow2Col, mateCol2Row);
        
        double tweighted = MPI_Wtime() - ts;
        
        CheckMatching(mateRow2Col,mateCol2Row);
        
    
        AWighted->PrintInfo();
        tinfo.str("");
        tinfo << "Total time: " << tcard + tweighted << " [ card: " << tcard << " weighted: " << tweighted << " ]" << endl;
        SpParHelper::Print(tinfo.str());
        
        if(saveMatching && ofname!="")
        {
            mateRow2Col.ParallelWrite(ofname,false,false);
        }
		
        

        
        
    }
    MPI_Finalize();
    return 0;
}

