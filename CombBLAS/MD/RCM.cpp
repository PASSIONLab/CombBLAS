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


typedef SpParMat < int64_t, bool, SpDCCols<int64_t,bool> > PSpMat_Bool;
typedef SpParMat < int64_t, int64_t, SpDCCols<int64_t,int64_t> > PSpMat_Int64;
void RCM(PSpMat_Bool & A);

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
            cout << "Usage: ./rcm <rmat|er|input> <scale|filename>" << endl;
            cout << "Example: mpirun -np 4 ./rcm rmat 20" << endl;
            cout << "Example: mpirun -np 4 ./rcm er 20" << endl;
            cout << "Example: mpirun -np 4 ./rcm input a.mtx" << endl;
            
        }
        MPI_Finalize();
        return -1;
    }
    {
        PSpMat_Bool * ABool;
        ostringstream tinfo;
        
        if(string(argv[1]) == string("input")) // input option
        {
            ABool = new PSpMat_Bool();
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
        }
        else if(string(argv[1]) == string("rmat"))
        {
            unsigned scale;
            scale = static_cast<unsigned>(atoi(argv[2]));
            double initiator[4] = {.57, .19, .19, .05};
            DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>();
            DEL->GenGraph500Data(initiator, scale, EDGEFACTOR, true, false );
            MPI_Barrier(MPI_COMM_WORLD);
            
            ABool = new PSpMat_Bool(*DEL, false);
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
            
            ABool = new PSpMat_Bool(*DEL, false);
            Symmetricize(*ABool);
            delete DEL;
        }
        else
        {
            SpParHelper::Print("Unknown input option\n");
            MPI_Finalize();
            return -1;
        }
        
        Symmetricize(*ABool);
        RCM(*ABool);
        
        /*
        FullyDistSpVec<int64_t, int64_t> fringe(ABool->getcommgrid(),  int64_t(5) );
        //fringe.SetElement(0, 10);
        fringe.SetElement(1, 11);
        fringe.SetElement(3, 0);
        fringe.SetElement(4, 2);
        FullyDistSpVec<int64_t, int64_t> sorted=  fringe.sort();
        FullyDistVec<int64_t, int64_t> idx = sorted.FindVals([](int64_t x){return true;});
        FullyDistVec<int64_t, int64_t> val = idx;
        val.iota(idx.TotalLength(),1);
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



void RCMOrder(PSpMat_Bool & A, int64_t source)
{
 
    FullyDistVec<int64_t, int64_t> degrees ( A.getcommgrid());
    A.Reduce(degrees, Column, plus<int64_t>(), static_cast<int64_t>(0));
    
    int64_t nv = A.getnrow();
    FullyDistVec<int64_t, int64_t> order ( A.getcommgrid(),  nv , (int64_t) -1);
    FullyDistSpVec<int64_t, int64_t> fringe(A.getcommgrid(),  nv );
    order.SetElement(source, 0); // source has order = 1
    fringe.SetElement(source, 0);
    int64_t curOrder = 1;
    
    while(fringe.getnnz() > 0) // continue until the frontier is empty
    {
        
        fringe = EWiseApply<int64_t>(fringe, order,
                                    [](int64_t parent_order, int64_t ord){return ord;},
                                    [](int64_t parent_order, int64_t ord){return true;},
                                    false, (int64_t) -1);
        SpMV<SelectMinSR>(A, fringe, fringe, false);
        fringe = EWiseMult(fringe, order, true, (int64_t) -1);
        
        fringe.DebugPrint();
        FullyDistSpVec<int64_t, VertexType> fringeRow = EWiseApply<VertexType>(fringe, degrees,
                                           [](int64_t parent_order, int64_t degree){return VertexType(parent_order, degree);},
                                           [](int64_t parent_order, int64_t degree){return true;},
                                           false, (int64_t) -1);
        //fringeRow.ApplyInd([](VertexType vtx, int64_t idx){return VertexType(vtx.order, vtx.degree, idx);});
        
        FullyDistSpVec<int64_t, int64_t> sorted =  fringeRow.sort();
        //sorted.ApplyInd([](){});
        sorted.DebugPrint();
        FullyDistVec<int64_t, int64_t> idx = sorted.FindVals([](int64_t x){return true;});
        FullyDistVec<int64_t, int64_t> val = idx;
        val.iota(idx.TotalLength(),curOrder);
        curOrder += idx.TotalLength();
        FullyDistSpVec<int64_t, int64_t> levelOrder (fringe.TotalLength(), idx, val);
        order.Set(levelOrder);
        
    }
    
    order.DebugPrint();
    
}




void RCM(PSpMat_Bool & A)
{
    
    FullyDistVec<int64_t, int64_t> degrees ( A.getcommgrid());
    A.Reduce(degrees, Column, plus<int64_t>(), static_cast<int64_t>(0));
    
    int64_t cc = 0; // connected component
    
    int64_t nv = A.getnrow();
    int64_t prevLevel=-1, curLevel=0; // initialized just to make the first iteration going
    int64_t source = 5; // any starting vertex is fine. test if it really matters in practice.
    FullyDistVec<int64_t, int64_t> level ( A.getcommgrid(),  nv , (int64_t) -1); // level structure in the current BFS tree
    
    double tstart = MPI_Wtime();
    while(curLevel > prevLevel)
    {
        prevLevel = curLevel;
        FullyDistSpVec<int64_t, int64_t> fringe(A.getcommgrid(),  nv );
        level = (int64_t)-1; // reset level structure in every iteration
        level.SetElement(source, 1); // place source at level 1
        fringe.SetElement(source, source); // include source to the initial fringe 
        curLevel = 2;
        while(fringe.getnnz() > 0) // continue until the frontier is empty
        {
            fringe.setNumToInd(); // unncessary since we don't care about the parent
            SpMV<SelectMinSR>(A, fringe, fringe, false);
            fringe = EWiseMult(fringe, level, true, (int64_t) -1);
            // set value to the current level
            fringe=curLevel++;
            level.Set(fringe);
        }
        curLevel = curLevel-2;
        // last non-empty level
        fringe = level.Find(curLevel);
        fringe.setNumToInd();
        FullyDistSpVec<int64_t, pair<int64_t, int64_t>> fringe_degree =
                                            EWiseApply<pair<int64_t, int64_t>>(fringe, degrees,
                                            [](int64_t vtx, int64_t deg){return make_pair(deg, vtx);},
                                            [](int64_t vtx, int64_t deg){return true;},
                                            false, (int64_t) 0);
   
        //for(int i=0; i<fringe_degree.getnnz(); i++)
         //   cout << "(" << fringe_degree[i].first << ", " << fringe_degree[i].second << ")";
        
        pair<int64_t, int64_t> mindegree_vertex = fringe_degree.Reduce(minimum<pair<int64_t, int64_t> >(), make_pair(LLONG_MAX, (int64_t)-1));
        //cout << "\n ** (" << mindegree_vertex.first << ", " << mindegree_vertex.second << ")" << endl;
        if (curLevel > prevLevel)
            source = mindegree_vertex.second;
   
    }
    
    
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    if(myrank == 0)
    {
        cout << "vertex " << source+1 << " is a pseudo peripheral vertex" << endl;
        cout << "pseudo diameter = " << curLevel << endl;
        cout << "time taken: " <<  MPI_Wtime() - tstart << " seconds." << endl;
    }
    
    
    
    //RCMOrder(A, source);
    //level.DebugPrint();
}








