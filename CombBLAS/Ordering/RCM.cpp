#define DETERMINISTIC
#include "../CombBLAS.h"
#include <mpi.h>
#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>


#define EDGEFACTOR 16  /// changed to 8
using namespace std;


#ifdef THREADED
#ifndef _OPENMP
#define _OPENMP
#endif

#include <omp.h>
int cblas_splits = 1;
#endif


template <typename PARMAT>
void Symmetricize(PARMAT & A)
{
    // boolean addition is practically a "logical or"
    // therefore this doesn't destruct any links
    PARMAT AT = A;
    AT.Transpose();
    AT.RemoveLoops(); // not needed for boolean matrices, but no harm in keeping it
    A += AT;
}



struct VertexType
{
public:
    VertexType(int64_t ord=-1, int64_t deg=-1){order=ord; degree = deg;};
    
    friend bool operator<(const VertexType & vtx1, const VertexType & vtx2 )
    {
        if(vtx1.order==vtx2.order) return vtx1.degree < vtx2.degree;
        else return vtx1.order<vtx2.order;
    };
    friend bool operator==(const VertexType & vtx1, const VertexType & vtx2 ){return vtx1.order==vtx2.order & vtx1.degree==vtx2.degree;};
    friend ostream& operator<<(ostream& os, const VertexType & vertex ){os << "(" << vertex.order << "," << vertex.degree << ")"; return os;};
    //private:
    int64_t order;
    int64_t degree;
};



struct SelectMinSR
{
    typedef int64_t T_promote;
    static T_promote id(){ return -1; };
    static bool returnedSAID() { return false; }
    //static MPI_Op mpi_op() { return MPI_MIN; };
    
    static T_promote add(const T_promote & arg1, const T_promote & arg2)
    {
        return std::min(arg1, arg2);
    }
    
    static T_promote multiply(const bool & arg1, const T_promote & arg2)
    {
        return arg2;
    }
    
    static void axpy(bool a, const T_promote & x, T_promote & y)
    {
        y = std::min(y, x);
    }
};


typedef SpParMat < int64_t, bool, SpDCCols<int64_t,bool> > Par_DCSC_Bool;
typedef SpParMat < int64_t, bool, SpCCols<int64_t,bool> > Par_CSC_Bool;
FullyDistVec<int64_t, int64_t> RCM(Par_DCSC_Bool & A, FullyDistVec<int64_t, int64_t> degrees);

int main(int argc, char* argv[])
{
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    if(argc < 3)
    {
        if(myrank == 0)
        {
            cout << "Usage: ./rcm <rmat|er|input> <scale|filename> <splitPerThread>" << endl;
            cout << "Example: mpirun -np 4 ./rcm rmat 20" << endl;
            cout << "Example: mpirun -np 4 ./rcm er 20" << endl;
            cout << "Example: mpirun -np 4 ./rcm input a.mtx" << endl;
            
        }
        MPI_Finalize();
        return -1;
    }
    {
        Par_DCSC_Bool * ABool;
        ostringstream tinfo;
        
        if(string(argv[1]) == string("input")) // input option
        {
            ABool = new Par_DCSC_Bool();
            string filename(argv[2]);
            tinfo.str("");
            tinfo << "**** Reading input matrix: " << filename << " ******* " << endl;
            SpParHelper::Print(tinfo.str());
            double t01 = MPI_Wtime();
            ABool->ParallelReadMM(filename);
            double t02 = MPI_Wtime();
            ABool->PrintInfo();
            tinfo.str("");
            tinfo << "Reader took " << t02-t01 << " seconds" << endl;
            SpParHelper::Print(tinfo.str());
            
            if(ABool->getnrow() == ABool->getncol())
            {
                FullyDistVec<int64_t, int64_t> p( ABool->getcommgrid());
                p.iota(ABool->getnrow(), 0);
                p.RandPerm();
                (*ABool)(p,p,true);// in-place permute to save memory
                SpParHelper::Print("Applied symmetric permutation.\n");
            }
            else
            {
                SpParHelper::Print("Rectangular matrix: Can not apply symmetric permutation.\n");
            }
            
            Symmetricize(*ABool); //***** because RCM makes sense only to symmetric matrices
        }
        else if(string(argv[1]) == string("rmat"))
        {
            unsigned scale;
            scale = static_cast<unsigned>(atoi(argv[2]));
            double initiator[4] = {.57, .19, .19, .05};
            DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>();
            DEL->GenGraph500Data(initiator, scale, EDGEFACTOR, true, false );
            MPI_Barrier(MPI_COMM_WORLD);
            
            ABool = new Par_DCSC_Bool(*DEL, false);
            Symmetricize(*ABool);
            delete DEL;
        }
        else if(string(argv[1]) == string("er"))
        {
            unsigned scale;
            scale = static_cast<unsigned>(atoi(argv[2]));
            double initiator[4] = {.25, .25, .25, .25};
            DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>();
            DEL->GenGraph500Data(initiator, scale, EDGEFACTOR, true, false );
            MPI_Barrier(MPI_COMM_WORLD);
            
            ABool = new Par_DCSC_Bool(*DEL, false);
            Symmetricize(*ABool);
            delete DEL;
        }
        else
        {
            SpParHelper::Print("Unknown input option\n");
            MPI_Finalize();
            return -1;
        }
        
       
        ABool->PrintInfo();
        float balance = ABool->LoadImbalance();
        ostringstream outs;
        outs << "Load balance: " << balance << endl;
        SpParHelper::Print(outs.str());
        
        // Reduce is not multithreaded, so I am doing it here
        FullyDistVec<int64_t, int64_t> degrees ( ABool->getcommgrid());
        ABool->Reduce(degrees, Column, plus<int64_t>(), static_cast<int64_t>(0));
        
        
        int splitPerThread = 1;
        if(argc==4)
            splitPerThread = atoi(argv[3]);
        int nthreads;

#ifdef THREADED
#pragma omp parallel
        {
            nthreads = omp_get_num_threads();
            cblas_splits = nthreads*splitPerThread;
        }
        tinfo.str("");
        tinfo << "Threading activated with " << nthreads << " threads, and matrix split into "<< cblas_splits <<  " parts" << endl;
        SpParHelper::Print(tinfo.str());
        ABool->ActivateThreading(cblas_splits); // note: crash on empty matrix
        
#endif
        

        // compute bandwidth
        FullyDistVec<int64_t, int64_t> rcmorder = RCM(*ABool, degrees);
        // note: threaded matrix can not be permuted
        // That is why I am not a supporter of split matrix.
        // ABAB: Any suggestions in lieu of split matrix?
        //(*ABool)(rcmorder,rcmorder,true);// in-place permute to save memory
        // compute bandwidth

        
        /*
        
        FullyDistSpVec<int64_t, int64_t> fringe(ABool->getcommgrid(),  int64_t(5) );
        //fringe.SetElement(0, 10);
        fringe.SetElement(1, 11);
        fringe.SetElement(3, 0);
        fringe.SetElement(4, 2);
        FullyDistSpVec<int64_t, int64_t> sorted=  fringe.sort();
        FullyDistVec<int64_t, int64_t> idx = sorted.FindVals([](int64_t x){return true;});
        FullyDistVec<int64_t, int64_t> val(idx.getcommgrid());
        val.iota(idx.TotalLength(),10);
        FullyDistSpVec<int64_t, int64_t> sorted1 (fringe.TotalLength(), idx, val);
        
        FullyDistSpVec<int64_t, int64_t> sortedi= sorted.Invert(5);
        sorted.DebugPrint();
        sortedi.DebugPrint();
        sorted1.DebugPrint();
        */
         
        
    }
    MPI_Finalize();
    return 0;
}


// perform ordering from a source vertex
void RCMOrder(Par_DCSC_Bool & A, int64_t source, FullyDistVec<int64_t, int64_t>& order, int64_t startOrder)
{
 
    double tSpMV=0, tOrder, tOther, tSpMV1;
    tOrder1 = MPI_Wtime();
    
    FullyDistVec<int64_t, int64_t> degrees ( A.getcommgrid());
    A.Reduce(degrees, Column, plus<int64_t>(), static_cast<int64_t>(0));    // HERE
    
    int64_t nv = A.getnrow();
    FullyDistSpVec<int64_t, int64_t> fringe(A.getcommgrid(),  nv );
    order.SetElement(source, startOrder);
    fringe.SetElement(source, startOrder);
    int64_t curOrder = startOrder+1;
    
    
    
    while(fringe.getnnz() > 0) // continue until the frontier is empty
    {
        
        fringe = EWiseApply<int64_t>(fringe, order,
                                    [](int64_t parent_order, int64_t ord){return ord;},
                                    [](int64_t parent_order, int64_t ord){return true;},
                                    false, (int64_t) -1);
        
        tSpMV1 = MPI_Wtime();
        SpMV<SelectMinSR>(A, fringe, fringe, false);
        tSpMV += MPI_Wtime() - tSpMV1;
        fringe = EWiseMult(fringe, order, true, (int64_t) -1);
       
        
        //fringe.DebugPrint();
        FullyDistSpVec<int64_t, VertexType> fringeRow = EWiseApply<VertexType>(fringe, degrees,
                                           [](int64_t parent_order, int64_t degree){return VertexType(parent_order, degree);},
                                           [](int64_t parent_order, int64_t degree){return true;},
                                           false, (int64_t) -1);
        //fringeRow.ApplyInd([](VertexType vtx, int64_t idx){return VertexType(vtx.order, vtx.degree, idx);});
        
        /*
        //FullyDistSpVec::sort returns (i,j) index pairs such that
        // jth entry before sorting becomes ith entry after sorting.
        // Here i/j is the index of the elements relative to the dense containter
        // Alternatively, j's consist a permutation that would premute the undorted vector to sorted vector
         */
        
        FullyDistSpVec<int64_t, int64_t> sorted =  fringeRow.sort();
        // idx is the index  of fringe in sorted order
        FullyDistVec<int64_t, int64_t> idx = sorted.FindVals([](int64_t x){return true;});
        FullyDistVec<int64_t, int64_t> val(idx.getcommgrid());
        // val is the index  of fringe in sorted order (relative to each other starting with  1)
        val.iota(idx.TotalLength(),curOrder);
        curOrder += idx.TotalLength();
        FullyDistSpVec<int64_t, int64_t> levelOrder (fringe.TotalLength(), idx, val);
        order.Set(levelOrder);
        
    }
    
    tOrder = MPI_Wtime() - tOrder;
    tOther = tOrder - tSpMV;
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    if(myrank == 0)
    {
        cout << "Ordering time: " << endl;
        cout << "SpMV time: " <<  tSpMV << " Other time: " << tOther << endl;
        cout << "Total time: " <<  tOrder << " seconds." << endl;
        
    }
    
    //order.DebugPrint();
    
}




FullyDistVec<int64_t, int64_t> RCM(Par_DCSC_Bool & A, FullyDistVec<int64_t, int64_t> degrees)
{
    /*
     list of current unvisited vertices
     After discovering a pseudo peripheral vertex in the ith connected component
     all vertices in the ith component are removed from this vector.
     I am keeping degrees of the vertices as well so that we can select the starting vertex
     in each connected component based on the degrees.
     degrees can be replaced by random numbers or other things
    */
    FullyDistSpVec<int64_t, int64_t> unvisited ( A.getcommgrid(),  A.getnrow());
    unvisited.iota(A.getnrow(), (int64_t) 0); // index and values become the same
    FullyDistSpVec<int64_t, pair<int64_t, int64_t>> unvisitedVertices =
    EWiseApply<pair<int64_t, int64_t>>(unvisited, degrees,
                                       [](int64_t vtx, int64_t deg){return make_pair(deg, vtx);},
                                       [](int64_t vtx, int64_t deg){return true;},
                                       false, (int64_t) -1);
    
   
    FullyDistVec<int64_t, int64_t> rcmorder ( A.getcommgrid(),  A.getnrow(), (int64_t) -1);
    int cc = 1;
    
    while(unvisitedVertices.getnnz()>0) // for each connected component
    {

        // Select a minimum-degree unvisited vertex as the initial source
        
        pair<int64_t, int64_t> mindegree_vertex = unvisitedVertices.Reduce(minimum<pair<int64_t, int64_t> >(), make_pair(LLONG_MAX, (int64_t)-1));
        int64_t source = mindegree_vertex.second;
        
        int64_t prevLevel=-1, curLevel=0; // initialized just to make the first iteration going
        // level structure in the current BFS tree
        // we are not using this information. Currently it is serving as visited flag
        FullyDistVec<int64_t, int64_t> level ( A.getcommgrid(),  A.getnrow(), (int64_t) -1);
        
        int iterations = 0;
        double tstart = MPI_Wtime();
        double tSpMV=0, tBFS=0, tOther=0, tSpMV1, tBFS1, tOther1;
        while(curLevel > prevLevel)
        {
            prevLevel = curLevel;
            FullyDistSpVec<int64_t, int64_t> fringe(A.getcommgrid(),  A.getnrow() );
            level = (int64_t)-1; // reset level structure in every iteration
            level.SetElement(source, 1); // place source at level 1
            fringe.SetElement(source, source); // include source to the initial fringe
            curLevel = 2;
            tBFS1 = MPI_Wtime();
            while(fringe.getnnz() > 0) // continue until the frontier is empty
            {
                fringe.setNumToInd(); // unncessary since we don't care about the parent
                
                tSpMV1 = MPI_Wtime();
                SpMV<SelectMinSR>(A, fringe, fringe, false); // HERE
                tSpMV += MPI_Wtime() - tSpMV1;
                fringe = EWiseMult(fringe, level, true, (int64_t) -1);
                // set value to the current level
                fringe=curLevel;
                curLevel++;
                level.Set(fringe);
            }
            tBFS += MPI_Wtime() - tBFS1;
            curLevel = curLevel-2;
            
            
            // last non-empty level
            tOther1 = MPI_Wtime();
            fringe = level.Find(curLevel);
            fringe.setNumToInd();
            
            // this can be done by combingn a reduce and find
            FullyDistSpVec<int64_t, pair<int64_t, int64_t>> fringe_degree =
            EWiseApply<pair<int64_t, int64_t>>(fringe, degrees,
                                               [](int64_t vtx, int64_t deg){return make_pair(deg, vtx);},
                                               [](int64_t vtx, int64_t deg){return true;},
                                               false, (int64_t) -1);
            
            //for(int i=0; i<fringe_degree.getnnz(); i++)
            //   cout << "(" << fringe_degree[i].first << ", " << fringe_degree[i].second << ")";
            
            mindegree_vertex = fringe_degree.Reduce(minimum<pair<int64_t, int64_t> >(), make_pair(LLONG_MAX, (int64_t)-1));
            //cout << "\n ** (" << mindegree_vertex.first << ", " << mindegree_vertex.second << ")" << endl;
            if (curLevel > prevLevel)
                source = mindegree_vertex.second;
            iterations++;
            tOther += MPI_Wtime() - tOther1;
            
        }
        
        int myrank;
        MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
        if(myrank == 0)
        {
            cout << "Pseudo-peripheral vertex identification time" << endl;
            cout << "Connected component # " << cc++ << endl;
            cout << "vertex " << source+1 << " is a pseudo peripheral vertex" << endl;
            cout << "pseudo diameter: " << curLevel << " iterations: "<< iterations <<  endl;
            cout << "SpMV time: " <<  tSpMV << " BFS time: " << tBFS << " Other time: " << tOther << endl;
            cout << "Total time: " <<  MPI_Wtime() - tstart << " seconds." << endl;
           
        }
        
        // order vertices in this connected component
        int64_t curOrder =  A.getnrow() - unvisitedVertices.getnnz();
        RCMOrder(A, source, rcmorder, curOrder);
        
        
        // remove vertices in the current connected component
        //unvisitedVertices = EWiseMult(unvisitedVertices, level, true, (int64_t) -1);
        unvisitedVertices = EWiseApply<pair<int64_t, int64_t>>(unvisitedVertices, level,
                                           [](pair<int64_t, int64_t> vtx, int64_t visited){return vtx;},
                                           [](pair<int64_t, int64_t> vtx, int64_t visited){return visited==-1;},
                                           false, make_pair((int64_t)-1, (int64_t)0));
        
        if(myrank == 0)
            cout << "remaining vertices: " << unvisitedVertices.getnnz() << endl;
    }
    
    return rcmorder;
}


