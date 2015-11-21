#include "../CombBLAS.h"
#include <mpi.h>
#include <sys/time.h> 
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>
#include "BPMaximalMatching.h"


#ifdef THREADED
	#ifndef _OPENMP
	#define _OPENMP
	#endif

	#include <omp.h>
    int cblas_splits = 1;
#endif

using namespace std;

bool prune, mvInvertMate, randMM, moreSplit;
int init;
bool randMaximal;
bool fewexp;


template <typename PARMAT>
void Symmetricize(PARMAT & A)
{
	// boolean addition is practically a "logical or"
	// therefore this doesn't destruct any links
	PARMAT AT = A;
	AT.Transpose();
	A += AT;
}


struct VertexType
{
public:
    VertexType(int64_t p=-1, int64_t r=-1, int16_t pr=0){parent=p; root = r; prob = pr;};
    
    friend bool operator<(const VertexType & vtx1, const VertexType & vtx2 )
    {
        if(vtx1.prob==vtx2.prob) return vtx1.parent<vtx2.parent;
        else return vtx1.prob<vtx2.prob;
    };
    friend bool operator==(const VertexType & vtx1, const VertexType & vtx2 ){return vtx1.parent==vtx2.parent;};
    friend ostream& operator<<(ostream& os, const VertexType & vertex ){os << "(" << vertex.parent << "," << vertex.root << ")"; return os;};
    //private:
    int64_t parent;
    int64_t root;
    int16_t prob; // probability of selecting an edge
    
};




typedef SpParMat < int64_t, bool, SpDCCols<int64_t,bool> > PSpMat_Bool;
typedef SpParMat < int64_t, bool, SpDCCols<int32_t,bool> > PSpMat_s32p64;
typedef SpParMat < int64_t, int64_t, SpDCCols<int64_t,int64_t> > PSpMat_Int64;
typedef SpParMat < int64_t, float, SpDCCols<int64_t,float> > PSpMat_float;
void maximumMatching(PSpMat_s32p64 & Aeff, FullyDistVec<int64_t, int64_t>& mateRow2Col,
                     FullyDistVec<int64_t, int64_t>& mateCol2Row);
template <class IT, class NT>
bool isMaximalmatching(PSpMat_Int64 & A, FullyDistVec<IT,NT> & mateRow2Col, FullyDistVec<IT,NT> & mateCol2Row,
                       FullyDistSpVec<int64_t, int64_t> unmatchedRow, FullyDistSpVec<int64_t, int64_t> unmatchedCol);




/*
 Remove isolated vertices and purmute
 */
void removeIsolated(PSpMat_Bool & A)
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
        cout << "Usage (random matrix): ./maximal <er|g500|ssca> <Scale> <EDGEFACTOR> <algo><rand><moreSplit>\n";
        cout << "Usage (input matrix): ./maximal <input> <matrix> <algo><rand><moreSplit>\n\n";
        
        cout << " \n-------------- meaning of arguments ----------\n";
        cout << "** er: Erdos-Renyi, g500: Graph500 benchmark, ssca: SSCA benchmark\n";
        cout << "** scale: matrix dimention is 2^scale\n";
        cout << "** edgefactor: average degree of vertices\n";
        cout << "** algo : maximal matching algorithm used to initialize\n ";
        cout << "      greedy: greedy init , ks: Karp-Sipser, dmd: dynamic mindegree\n";
        cout << "       default: dynamic mindegree\n";
        cout << "** (optional) rand: random parent selection in greedy/Karp-Sipser\n" ;
        cout << "** (optional) moreSplit: more splitting of Matrix.\n" ;
        cout << "(order of optional arguments does not matter)\n";
        
        cout << " \n-------------- examples ----------\n";
        cout << "Example: mpirun -np 4 ./maximal g500 18 16 ks rand" << endl;
        cout << "Example: mpirun -np 4 ./maximal input cage12.mtx dmd\n" << endl;
    }
}

void GetOptions(char* argv[], int argc)
{
    string allArg="";
    for(int i=0; i<argc; i++)
    {
        allArg += string(argv[i]);
    }
    
    if(allArg.find("moreSplit")!=string::npos)
        moreSplit = true;
    if(allArg.find("randMaximal")!=string::npos)
        randMaximal = true;
    if(allArg.find("greedy")!=string::npos)
        init = GREEDY;
    else if(allArg.find("ks")!=string::npos)
        init = KARP_SIPSER;
    else if(allArg.find("dmd")!=string::npos)
        init = DMD;
    else
        init = DMD;
    
}

void showCurOptions()
{
    ostringstream tinfo;
    tinfo.str("");
    tinfo << "\n---------------------------------\n";
    tinfo << " Maximal matching algorithm options: ";
    if(init == KARP_SIPSER) tinfo << " Karp-Sipser, ";
    if(init == DMD) tinfo << " dynamic mindegree, ";
    if(init == GREEDY) tinfo << " greedy, ";
    if(randMaximal) tinfo << " random parent selection in greedy/Karp-Sipser, ";
    if(moreSplit) tinfo << " moreSplit ";
    tinfo << "\n---------------------------------\n\n";
    SpParHelper::Print(tinfo.str());
    
}

void experiment(PSpMat_s32p64 & A, PSpMat_s32p64 & AT, FullyDistVec<int64_t, int64_t> degCol)
{
    FullyDistVec<int64_t, int64_t> mateRow2Col ( A.getcommgrid(), A.getnrow(), (int64_t) -1);
    FullyDistVec<int64_t, int64_t> mateCol2Row ( A.getcommgrid(), A.getncol(), (int64_t) -1);
    
    // best option
    init = DMD; randMaximal = false;
    //showCurOptions();
    MaximalMatching(A, AT, mateRow2Col, mateCol2Row, degCol, init, randMaximal);
    mateRow2Col.Apply([](int64_t val){return (int64_t) -1;});
    mateCol2Row.Apply([](int64_t val){return (int64_t) -1;});
    
    // best option + KS
    init = KARP_SIPSER; randMaximal = true;
    //showCurOptions();
    MaximalMatching(A, AT, mateRow2Col, mateCol2Row, degCol, init, randMaximal);
    mateRow2Col.Apply([](int64_t val){return (int64_t) -1;});
    mateCol2Row.Apply([](int64_t val){return (int64_t) -1;});
    
    
    // best option + Greedy
    init = GREEDY; randMaximal = true;
    //showCurOptions();
    MaximalMatching(A, AT, mateRow2Col, mateCol2Row, degCol, init, randMaximal);
    mateRow2Col.Apply([](int64_t val){return (int64_t) -1;});
    mateCol2Row.Apply([](int64_t val){return (int64_t) -1;});
    
    // best option + KS
    init = KARP_SIPSER; randMaximal = false;
    //showCurOptions();
    MaximalMatching(A, AT, mateRow2Col, mateCol2Row, degCol, init, randMaximal);
    mateRow2Col.Apply([](int64_t val){return (int64_t) -1;});
    mateCol2Row.Apply([](int64_t val){return (int64_t) -1;});
    
    
    // best option + Greedy
    init = GREEDY; randMaximal = false;
    //showCurOptions();
    MaximalMatching(A, AT, mateRow2Col, mateCol2Row, degCol, init, randMaximal);
    mateRow2Col.Apply([](int64_t val){return (int64_t) -1;});
    mateCol2Row.Apply([](int64_t val){return (int64_t) -1;});
    
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
    moreSplit = false;
    
    // ------------ Process input arguments and build matrix ---------------
	{
        
        PSpMat_Bool * ABool;
        PSpMat_s32p64 ALocalT;
        ostringstream tinfo;
        double t01, t02;
        if(string(argv[1]) == string("input")) // input option
        {
            ABool = new PSpMat_Bool();
            
            string filename(argv[2]);
            matrix_name = filename;
            tinfo.str("");
            tinfo << "**** Reading input matrix: " << filename << " ******* " << endl;
            SpParHelper::Print(tinfo.str());
            t01 = MPI_Wtime();
            ABool->ParallelReadMM(filename);
            t02 = MPI_Wtime();
            ABool->PrintInfo();
            tinfo.str("");
            tinfo << "Reader took " << t02-t01 << " seconds" << endl;
            SpParHelper::Print(tinfo.str());
            GetOptions(argv+3, argc-3);

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
                matrix_name = "ER";
            }
            else if(string(argv[1]) == string("g500"))
            {
                initiator[0] = .57;
                initiator[1] = .19;
                initiator[2] = .19;
                initiator[3] = .05;
                matrix_name = "G500";
            }
            else if(string(argv[1]) == string("ssca"))
            {
                initiator[0] = .6;
                initiator[1] = .4/3;
                initiator[2] = .4/3;
                initiator[3] = .4/3;
                matrix_name = "SSCA";
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
            ABool = new PSpMat_Bool(*DEL, false);
            delete DEL;
            t02 = MPI_Wtime();
            ABool->PrintInfo();
            tinfo.str("");
            tinfo << "Generator took " << t02-t01 << " seconds" << endl;
            SpParHelper::Print(tinfo.str());
            
            //Symmetricize(*ABool);
            //removeIsolated(*ABool);
            //SpParHelper::Print("Generated matrix symmetricized....\n");
            ABool->PrintInfo();
            
            GetOptions(argv+4, argc-4);

        }

  
        // randomly permute for load balance
        SpParHelper::Print("Performing random permuation of matrix.\n");
        FullyDistVec<int64_t, int64_t> prow(ABool->getcommgrid());
        FullyDistVec<int64_t, int64_t> pcol(ABool->getcommgrid());
        prow.iota(ABool->getnrow(), 0);
        pcol.iota(ABool->getncol(), 0);
        prow.RandPerm();
        pcol.RandPerm();
        (*ABool)(prow, pcol, true);
        SpParHelper::Print("Performed random permuation of matrix.\n");

     
        PSpMat_s32p64 A = *ABool;
        PSpMat_s32p64 AT = A;
        if(ABool->getnrow() > ABool->getncol())
            AT.Transpose();
        else
            A.Transpose();
       
        
        // Reduce is not multithreaded, so I am doing it here
        FullyDistVec<int64_t, int64_t> degCol(A.getcommgrid());
        A.Reduce(degCol, Column, plus<int64_t>(), static_cast<int64_t>(0));
        
        int nthreads;
#ifdef _OPENMP
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

        SpParHelper::Print(" #####################################################\n");
        SpParHelper::Print(" ################## Run 1 ############################\n");
        SpParHelper::Print(" #####################################################\n");
        experiment(A, AT, degCol);
        
        SpParHelper::Print(" #####################################################\n");
        SpParHelper::Print(" ################## Run 2 ############################\n");
        SpParHelper::Print(" #####################################################\n");
        experiment(A, AT, degCol);
        
        
        SpParHelper::Print(" #####################################################\n");
        SpParHelper::Print(" ################## Run 3 ############################\n");
        SpParHelper::Print(" #####################################################\n");
        experiment(A, AT, degCol);
	}
	MPI_Finalize();
	return 0;
}


