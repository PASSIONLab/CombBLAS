
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


#include "BPMaximalMatching.h"
#include "BPMaximumMatching.h"

using namespace std;
using namespace combblas;


typedef SpParMat < int64_t, bool, SpDCCols<int64_t,bool> > Par_DCSC_Bool;
typedef SpParMat < int64_t, int64_t, SpDCCols<int64_t, int64_t> > Par_DCSC_int64_t;
typedef SpParMat < int64_t, double, SpDCCols<int64_t, double> > Par_DCSC_Double;
typedef SpParMat < int64_t, bool, SpCCols<int64_t,bool> > Par_CSC_Bool;

// algorithmic options
bool prune, randMM, moreSplit;
int init;
bool randMaximal;
bool fewexp;
bool randPerm;
bool saveMatching;
string ofname;



/*
 Remove isolated vertices and purmute
 */
void removeIsolated(Par_DCSC_Bool & A)
{
    
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    
    
    FullyDistVec<int64_t, int64_t> * ColSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid());
    FullyDistVec<int64_t, int64_t> * RowSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid());
    FullyDistVec<int64_t, int64_t> nonisoRowV;	// id's of non-isolated (connected) Row vertices
    FullyDistVec<int64_t, int64_t> nonisoColV;	// id's of non-isolated (connected) Col vertices
    FullyDistVec<int64_t, int64_t> nonisov;	// id's of non-isolated (connected) vertices
    
    A.Reduce(*ColSums, Column, plus<int64_t>(), static_cast<int64_t>(0));
    A.Reduce(*RowSums, Row, plus<int64_t>(), static_cast<int64_t>(0));
    
    // this steps for general graph
    /*
     ColSums->EWiseApply(*RowSums, plus<int64_t>()); not needed for bipartite graph
     nonisov = ColSums->FindInds(bind2nd(greater<int64_t>(), 0));
     nonisov.RandPerm();	// so that A(v,v) is load-balanced (both memory and time wise)
     A.operator()(nonisov, nonisov, true);	// in-place permute to save memory
     */
    
    // this steps for bipartite graph
    nonisoColV = ColSums->FindInds(bind2nd(greater<int64_t>(), 0));
    nonisoRowV = RowSums->FindInds(bind2nd(greater<int64_t>(), 0));
    delete ColSums;
    delete RowSums;
    
    
    {
        nonisoColV.RandPerm();
        nonisoRowV.RandPerm();
    }
    
    
    int64_t nrows1=A.getnrow(), ncols1=A.getncol(), nnz1 = A.getnnz();
    double avgDeg1 = (double) nnz1/(nrows1+ncols1);
    
    
    A.operator()(nonisoRowV, nonisoColV, true);
    
    int64_t nrows2=A.getnrow(), ncols2=A.getncol(), nnz2 = A.getnnz();
    double avgDeg2 = (double) nnz2/(nrows2+ncols2);
    
    
    if(myrank == 0)
    {
        cout << "ncol nrows  nedges deg \n";
        cout << nrows1 << " " << ncols1 << " " << nnz1 << " " << avgDeg1 << " \n";
        cout << nrows2 << " " << ncols2 << " " << nnz2 << " " << avgDeg2 << " \n";
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    
}


void ShowUsage()
{
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    if(myrank == 0)
    {
        cout << "\n-------------- usage --------------\n";
        cout << "Usage (random matrix): ./bpmm <er|g500|ssca> <Scale> <EDGEFACTOR> <init><diropt><prune><graft>\n";
        cout << "Usage (input matrix): ./bpmm <input> <matrix> <init><diropt><prune><graft>\n\n";
        
        cout << " \n-------------- meaning of arguments ----------\n";
        cout << "** er: Erdos-Renyi, g500: Graph500 benchmark, ssca: SSCA benchmark\n";
        cout << "** scale: matrix dimention is 2^scale\n";
        cout << "** edgefactor: average degree of vertices\n";
        cout << "** (optional) init : maximal matching algorithm used to initialize\n ";
        cout << "      none: noinit, greedy: greedy init , ks: Karp-Sipser, dmd: dynamic mindegree\n";
        cout << "       default: none\n";
        cout << "** (optional) randMaximal: random parent selection in greedy/Karp-Sipser\n" ;
        //cout << "** (optional) diropt: employ direction-optimized BFS\n" ;
        cout << "** (optional) prune: discard trees as soon as an augmenting path is found\n" ;
        //cout << "** (optional) graft: employ tree grafting\n" ;
        cout << "** (optional) moreSplit: more splitting of Matrix.\n" ;
        cout << "** (optional) randPerm: Randomly permute the matrix for load balance.\n" ;
        cout << "** (optional) saveMatching: Save the matching vector in a file (filename: inputfile_matching.txt).\n" ;
        cout << "(order of optional arguments does not matter)\n";
        
        
        cout << " \n-------------- examples ----------\n";
        cout << "Example: mpirun -np 4 ./bpmm g500 18 16" << endl;
        cout << "Example: mpirun -np 4 ./bpmm g500 18 16 ks diropt graft" << endl;
        cout << "Example: mpirun -np 4 ./bpmm input cage12.mtx randPerm ks diropt graft\n" << endl;
    }
}

void GetOptions(char* argv[], int argc)
{
    string allArg="";
    for(int i=0; i<argc; i++)
    {
        allArg += string(argv[i]);
    }
    
    if(allArg.find("prune")!=string::npos)
        prune = true;
    if(allArg.find("fewexp")!=string::npos)
        fewexp = true;
    if(allArg.find("moreSplit")!=string::npos)
        moreSplit = true;
    if(allArg.find("saveMatching")!=string::npos)
        saveMatching=true;
    if(allArg.find("randMM")!=string::npos)
        randMM = true;
    if(allArg.find("randMaximal")!=string::npos)
        randMaximal = true;
    if(allArg.find("randPerm")!=string::npos)
        randPerm = true;
    if(allArg.find("greedy")!=string::npos)
        init = GREEDY;
    else if(allArg.find("ks")!=string::npos)
        init = KARP_SIPSER;
    else if(allArg.find("dmd")!=string::npos)
        init = DMD;
    else
        init = NO_INIT;
    
}

void showCurOptions()
{
    ostringstream tinfo;
    tinfo.str("");
    tinfo << "\n---------------------------------\n";
    tinfo << "Calling maximum-cardinality matching with options: " << endl;
    tinfo << " init: ";
    if(init == NO_INIT) tinfo << " no-init ";
    if(init == KARP_SIPSER) tinfo << " Karp-Sipser, ";
    if(init == DMD) tinfo << " dynamic mindegree, ";
    if(init == GREEDY) tinfo << " greedy, ";
    if(randMaximal) tinfo << " random parent selection in greedy/Karp-Sipser, ";
    if(prune) tinfo << " tree pruning, ";
    if(moreSplit) tinfo << " moreSplit ";
    if(randPerm) tinfo << " Randomly permute the matrix for load balance ";
    if(saveMatching) tinfo << " Write the matcing in a file";
    tinfo << "\n---------------------------------\n\n";
    SpParHelper::Print(tinfo.str());
    
}

void experiment(Par_DCSC_Bool & A, Par_DCSC_Bool & AT, FullyDistVec<int64_t, int64_t> degCol)
{
    FullyDistVec<int64_t, int64_t> mateRow2Col ( A.getcommgrid(), A.getnrow(), (int64_t) -1);
    FullyDistVec<int64_t, int64_t> mateCol2Row ( A.getcommgrid(), A.getncol(), (int64_t) -1);
    
    // best option
    init = DMD; randMaximal = false; randMM = true; prune = true;
    showCurOptions();
    MaximalMatching(A, AT, mateRow2Col, mateCol2Row, degCol, init, randMaximal);
    maximumMatching(A, mateRow2Col, mateCol2Row,prune, randMM);
    mateRow2Col.Apply([](int64_t val){return (int64_t) -1;});
    mateCol2Row.Apply([](int64_t val){return (int64_t) -1;});
    
    // best option + KS
    init = KARP_SIPSER; randMaximal = true; randMM = true; prune = true;
    showCurOptions();
    MaximalMatching(A, AT, mateRow2Col, mateCol2Row, degCol, init, randMaximal);
    maximumMatching(A, mateRow2Col, mateCol2Row, prune,  randMM);
    mateRow2Col.Apply([](int64_t val){return (int64_t) -1;});
    mateCol2Row.Apply([](int64_t val){return (int64_t) -1;});
    
    
    // best option + Greedy
    init = GREEDY; randMaximal = true; randMM = true; prune = true;
    showCurOptions();
    MaximalMatching(A, AT, mateRow2Col, mateCol2Row, degCol, init, randMaximal);
    maximumMatching(A, mateRow2Col, mateCol2Row, prune,  randMM);
    mateRow2Col.Apply([](int64_t val){return (int64_t) -1;});
    mateCol2Row.Apply([](int64_t val){return (int64_t) -1;});
    
    // best option + No init
    init = NO_INIT; randMaximal = false; randMM = true; prune = true;
    showCurOptions();
    maximumMatching(A, mateRow2Col, mateCol2Row, prune,  randMM);
    mateRow2Col.Apply([](int64_t val){return (int64_t) -1;});
    mateCol2Row.Apply([](int64_t val){return (int64_t) -1;});
    
    
    // best option - randMM
    init = DMD; randMaximal = false; randMM = false; prune = true;
    showCurOptions();
    MaximalMatching(A, AT, mateRow2Col, mateCol2Row, degCol, init, randMaximal);
    maximumMatching(A, mateRow2Col, mateCol2Row, prune,  randMM);
    mateRow2Col.Apply([](int64_t val){return (int64_t) -1;});
    mateCol2Row.Apply([](int64_t val){return (int64_t) -1;});
    
    
    // best option - prune
    init = DMD; randMaximal = false; randMM = true; prune = false;
    showCurOptions();
    MaximalMatching(A, AT, mateRow2Col, mateCol2Row, degCol, init, randMaximal);
    maximumMatching(A, mateRow2Col, mateCol2Row, prune,  randMM);
    mateRow2Col.Apply([](int64_t val){return (int64_t) -1;});
    mateCol2Row.Apply([](int64_t val){return (int64_t) -1;});
    
}


// default experiment
void defaultExp(Par_DCSC_Bool & A, Par_DCSC_Bool & AT, FullyDistVec<int64_t, int64_t> degCol)
{
    FullyDistVec<int64_t, int64_t> mateRow2Col ( A.getcommgrid(), A.getnrow(), (int64_t) -1);
    FullyDistVec<int64_t, int64_t> mateCol2Row ( A.getcommgrid(), A.getncol(), (int64_t) -1);
    
    // best option
    init = DMD; randMaximal = false; randMM = true; prune = true;
    showCurOptions();
    MaximalMatching(A, AT, mateRow2Col, mateCol2Row, degCol, init, randMaximal);
    maximumMatching(A, mateRow2Col, mateCol2Row, prune, randMM);
    if(saveMatching && ofname!="")
    {
        mateRow2Col.ParallelWrite(ofname,false,false);
    }
    mateRow2Col.Apply([](int64_t val){return (int64_t) -1;});
    mateCol2Row.Apply([](int64_t val){return (int64_t) -1;});
    
}







void experiment_maximal(Par_DCSC_Bool & A, Par_DCSC_Bool & AT, FullyDistVec<int64_t, int64_t> degCol)
{
    
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    
    FullyDistVec<int64_t, int64_t> mateRow2Col ( A.getcommgrid(), A.getnrow(), (int64_t) -1);
    FullyDistVec<int64_t, int64_t> mateCol2Row ( A.getcommgrid(), A.getncol(), (int64_t) -1);
    
    double time_start;
    
    // best option
    init = DMD; randMaximal = false; randMM = true; prune = true;
    time_start=MPI_Wtime();
    MaximalMatching(A, AT, mateRow2Col, mateCol2Row, degCol, init, randMaximal);
    double time_dmd = MPI_Wtime()-time_start;
    int64_t cardDMD = mateRow2Col.Count([](int64_t mate){return mate!=-1;});
    
    time_start=MPI_Wtime();
    maximumMatching(A, mateRow2Col, mateCol2Row, prune, randMM);
    double time_mm_dmd = MPI_Wtime()-time_start;
    int64_t mmcardDMD = mateRow2Col.Count([](int64_t mate){return mate!=-1;});
    mateRow2Col.Apply([](int64_t val){return (int64_t) -1;});
    mateCol2Row.Apply([](int64_t val){return (int64_t) -1;});
    
    // best option + KS
    init = KARP_SIPSER; randMaximal = true; randMM = true; prune = true;
    time_start=MPI_Wtime();
    MaximalMatching(A, AT, mateRow2Col, mateCol2Row, degCol, init, randMaximal);
    double time_ks = MPI_Wtime()-time_start;
    int64_t cardKS = mateRow2Col.Count([](int64_t mate){return mate!=-1;});
    
    time_start=MPI_Wtime();
    maximumMatching(A, mateRow2Col, mateCol2Row, prune, randMM);
    double time_mm_ks = MPI_Wtime()-time_start;
    int64_t mmcardKS = mateRow2Col.Count([](int64_t mate){return mate!=-1;});
    mateRow2Col.Apply([](int64_t val){return (int64_t) -1;});
    mateCol2Row.Apply([](int64_t val){return (int64_t) -1;});
    
    
    // best option + Greedy
    init = GREEDY; randMaximal = true; randMM = true; prune = true;
    time_start=MPI_Wtime();
    MaximalMatching(A, AT, mateRow2Col, mateCol2Row, degCol, init, randMaximal);
    double time_greedy = MPI_Wtime()-time_start;
    int64_t cardGreedy = mateRow2Col.Count([](int64_t mate){return mate!=-1;});
    
    time_start=MPI_Wtime();
    maximumMatching(A, mateRow2Col, mateCol2Row, prune, randMM);
    double time_mm_greedy = MPI_Wtime()-time_start;
    int64_t mmcardGreedy = mateRow2Col.Count([](int64_t mate){return mate!=-1;});
    mateRow2Col.Apply([](int64_t val){return (int64_t) -1;});
    mateCol2Row.Apply([](int64_t val){return (int64_t) -1;});
    
    if(myrank == 0)
    {
        cout << "\n maximal matching experiment \n";
        cout << cardGreedy << " " << mmcardGreedy << " " << time_greedy << " " << time_mm_greedy << " " << cardKS << " " << mmcardKS << " " << time_ks << " " << time_mm_ks << " " << cardDMD << " " << mmcardDMD << " " << time_dmd << " " << time_mm_dmd << " \n";
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
    saveMatching = false;
    ofname = "";
    randPerm = false;
    
    SpParHelper::Print("***** I/O and other preprocessing steps *****\n");
    // ------------ Process input arguments and build matrix ---------------
    {
        
        Par_DCSC_Bool * ABool;
        ostringstream tinfo;
        double t01, t02;
        if(string(argv[1]) == string("input")) // input option
        {
            ABool = new Par_DCSC_Bool();
            
            string filename(argv[2]);
            tinfo.str("");
            tinfo << "\n**** Reading input matrix: " << filename << " ******* " << endl;
            SpParHelper::Print(tinfo.str());
            t01 = MPI_Wtime();
            ABool->ParallelReadMM(filename, true, maximum<bool>()); // one-based matrix market file
            t02 = MPI_Wtime();
            ABool->PrintInfo();
            tinfo.str("");
            tinfo << "Reader took " << t02-t01 << " seconds" << endl;
            SpParHelper::Print(tinfo.str());
            GetOptions(argv+3, argc-3);
            if(saveMatching)
            {
                ofname = filename + ".matching.out";
            }
            
            
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
            ABool = new Par_DCSC_Bool(*DEL, false);
            delete DEL;
            t02 = MPI_Wtime();
            ABool->PrintInfo();
            tinfo.str("");
            tinfo << "Generator took " << t02-t01 << " seconds" << endl;
            SpParHelper::Print(tinfo.str());
            
            Symmetricize(*ABool);
            //removeIsolated(*ABool);
            SpParHelper::Print("Generated matrix symmetricized....\n");
            ABool->PrintInfo();
            
            GetOptions(argv+4, argc-4);
            
        }
        
        
        if(randPerm)
        {
            // randomly permute for load balance
            SpParHelper::Print("Performing random permutation of matrix.\n");
            FullyDistVec<int64_t, int64_t> prow(ABool->getcommgrid());
            FullyDistVec<int64_t, int64_t> pcol(ABool->getcommgrid());
            prow.iota(ABool->getnrow(), 0);
            pcol.iota(ABool->getncol(), 0);
            prow.RandPerm();
            pcol.RandPerm();
            (*ABool)(prow, pcol, true);
            SpParHelper::Print("Performed random permutation of matrix.\n");
        }
        
        
        Par_DCSC_Bool A = *ABool;
        Par_DCSC_Bool AT = A;
        AT.Transpose();
        
        // Reduce is not multithreaded, so I am doing it here
        FullyDistVec<int64_t, int64_t> degCol(A.getcommgrid());
        A.Reduce(degCol, Column, plus<int64_t>(), static_cast<int64_t>(0));
        
        // TODO: Follow the AWPM guideline to use CSC matrices.
        // Currently this file does not use multithreading in SpMSpV
/*
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
        //A.ActivateThreading(cblas_splits); // note: crash on empty matrix
        //AT.ActivateThreading(cblas_splits);
#endif
 */
        
        
        SpParHelper::Print("**************************************************\n\n");
        defaultExp(A, AT, degCol);
    }
    MPI_Finalize();
    return 0;
}


