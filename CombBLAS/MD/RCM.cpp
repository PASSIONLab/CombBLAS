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



struct SelectMinSR
{
    typedef int64_t T_promote;
    static T_promote id(){ return -1; };
    static bool returnedSAID() { return false; }
    static MPI_Op mpi_op() { return MPI_MIN; };
    
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
        
        if(string(argv[1]) == string("input")) // input option
        {
            string filename(argv[2]);
            ifstream inf;
            inf.open(filename.c_str(), ios::in);
            string header;
            getline(inf,header);
            bool isSymmetric = header.find("symmetric");
            bool isUnweighted = header.find("pattern");
            inf.close();
            
            ABool = new PSpMat_Bool();
            ABool->ReadDistribute(filename, 0, isUnweighted);	// unweighted
            if(isSymmetric)
                Symmetricize(*ABool);
            SpParHelper::Print("Read input\n");
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
        
    }
    MPI_Finalize();
    return 0;
}


void RCM(PSpMat_Bool & A)
{
    
    FullyDistVec<int64_t, int64_t> degrees ( A.getcommgrid());
    A.Reduce(degrees, Column, plus<int64_t>(), static_cast<int64_t>(0));
    
    
    int64_t nvertices = A.getnrow();
    int64_t oldLevels=-1, newLevels=0; // initialized just to make the first iteration going
    int64_t source = 1;
    while(newLevels > oldLevels)
    {
        oldLevels = newLevels;
        FullyDistSpVec<int64_t, int64_t> fringe(A.getcommgrid(),  nvertices );
        FullyDistSpVec<int64_t, int64_t> next_fringe(A.getcommgrid(),  nvertices );
        FullyDistVec<int64_t, int64_t> visited ( A.getcommgrid(),  nvertices , (int64_t) -1);
        
        //SpMV<Select2ndMinSR<bool, int64_t>>(A, fringe, fringe, false);
        fringe.SetElement(source, source);
        newLevels = 0;
        while(1)
        {
            
            SpMV<SelectMinSR>(A, fringe, next_fringe, false);
            next_fringe = EWiseMult(next_fringe, visited, true, (int64_t) -1);
            visited.Set(next_fringe);
            if(next_fringe.getnnz() > 0)
            {
                fringe = next_fringe;
                newLevels++;
            }
            else
                break;
            
        }
        
        
        FullyDistSpVec<int64_t, pair<int64_t, int64_t>> fringe_degree = EWiseApply<pair<int64_t, int64_t>>(fringe, degrees,
                                                                                                           [](int64_t vtx, int64_t deg){return make_pair(deg, vtx);},
                                                                                                           [](int64_t vtx, int64_t deg){return true;},
                                                                                                           false, (int64_t) 0);
        pair<int64_t, int64_t> mindegree_vertex = fringe_degree.Reduce(minimum<pair<int64_t, int64_t> >(), make_pair(LLONG_MAX, (int64_t)-1));
        source = mindegree_vertex.first;
        
        
    }
}








