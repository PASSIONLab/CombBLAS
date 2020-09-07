#define DETERMINISTIC
#include "CombBLAS/CombBLAS.h"
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
using namespace combblas;



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
void MD(PSpMat_Int64 & A);


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
            cout << "Usage: ./md <rmat|er|input> <scale|filename>" << endl;
            cout << "Example: mpirun -np 4 ./md rmat 20" << endl;
            cout << "Example: mpirun -np 4 ./md er 20" << endl;
            cout << "Example: mpirun -np 4 ./md input a.mtx" << endl;
            
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
        PSpMat_Int64  A = *ABool;
        
        
        MD(A);
        
    }
    MPI_Finalize();
    return 0;
}



// Single source BFS
// Find all reachable vertices from surce via enodes
FullyDistSpVec<int64_t, int64_t> getReach(int64_t source, PSpMat_Int64 & A, FullyDistVec<int64_t, int64_t>& enodes)
{
    
    FullyDistSpVec<int64_t, int64_t> x(A.getcommgrid(), A.getncol());
    FullyDistSpVec<int64_t, int64_t> nx(A.getcommgrid(), A.getnrow());
    FullyDistVec<int64_t, int64_t> visited ( A.getcommgrid(), A.getnrow(), (int64_t) 0);
    x.SetElement(source, 1);
    visited.SetElement(source, 1);
    while(x.getnnz() > 0)
    {
        SpMV<SelectMinSR>(A, x, nx, false);
        nx.Select(visited, [](int64_t visit){return visit==0;});
        visited.Set(nx);
        nx.Select(enodes, [](int64_t ev){return ev!=0;}); // newly visited enodes
        x = nx;
    }
    
    FullyDistSpVec<int64_t, int64_t> reach(visited, [](int64_t visit){return visit!=0;});
    reach.Select(enodes, [](int64_t ev){return ev==0;});
    reach.DelElement(source); // remove source
    return reach;
}






/***************************************************************************
 * Multisource (guided) BFS from all vertices in "sources" with SpGEMM
 * Only Vertices in "enodes" are used in the traversal
 * Inputs: 
 *      sources: the sources of BFSs. Let nnz(sources) = k
 *      A: nxn adjacency matrix (n is the number of vertices)
 *      enodes: enodes[i]=1 if the ith vertex has already received an order. 
 *              let the number of enodes is r
 ****************************************************************************/
FullyDistSpVec<int64_t, int64_t> getReachesSPMM(FullyDistSpVec<int64_t, int64_t>& sources, PSpMat_Int64 & A, FullyDistVec<int64_t, int64_t>& enodes)
{
    
    // ---------------------------------------------------------------------------
    // create an nxk initial frontier from sources
    // ith column represent the current frontier of BFS tree rooted at the ith source vertex
    //----------------------------------------------------------------------------
    FullyDistVec<int64_t, int64_t> ri = sources.FindInds([](int64_t x){return true;});
    FullyDistVec<int64_t, int64_t> ci(A.getcommgrid());
    ci.iota(sources.getnnz(), 0);

    PSpMat_Int64  fringe(A.getnrow(), sources.getnnz(), ri, ci, (int64_t) 1, false);
    PSpMat_Int64  visited(A.getnrow(), sources.getnnz(), ri, ci, (int64_t) 1, false);
    typedef PlusTimesSRing<int64_t, int64_t> PTDD;
    
    
    // ---------------------------------------------------------------------------
    // create an nxn matrix E such that E(i,i) = 1 if i is an enode
    // Note: we can avoid creating this matrix from scratch
    //       by incrementally adding nonzero entries in this matrix
    //----------------------------------------------------------------------------
    FullyDistVec<int64_t, int64_t> ri1 = enodes.FindInds([](int64_t x){return x>0;});
    PSpMat_Int64 E(A.getnrow(), A.getnrow(), ri1, ri1, 1);
    
    
    
    while( fringe.getnnz() > 0 )
    {
        // get the next frontier
        fringe = PSpGEMM<SelectMinSR>(A, fringe);
        
        // remove previously visited vertices
        fringe = EWiseMult(fringe, visited, true);
        visited += fringe;
        
        // keep enodes in the frontier
        // this can be replaced by EWiseMult if we keep a matrix with repeated enodes (memory requirement can be high)
        fringe = PSpGEMM<PTDD>(E, fringe);
        
    }
    

    FullyDistVec<int64_t, int64_t> degrees(A.getcommgrid(), sources.getnnz(), 0);

    // create an nxn matrix NE such that E(i,i) = 1 if i is not an enode
    // Note: NE and E together create a diagonal matrix
    FullyDistVec<int64_t, int64_t> ri2 = enodes.FindInds([](int64_t x){return x==0;});
    PSpMat_Int64 NE(A.getnrow(), A.getnrow(), ri2, ri2, 1);
    
    // keep only visited non enodes
    visited = PSpGEMM<PTDD>(NE, visited);
    
    // count visited non enodes from each sources
    visited.Reduce(degrees, Column, plus<int64_t>(), static_cast<int64_t>(0));
    degrees.Apply([](int64_t val){return (val-1);}); // -1 to remove sources themselves
    
    // option 2, using maskedReduce
    /*
    FullyDistSpVec<int64_t, int64_t> nenodes(enodes,[](int64_t x){return x==0;});
    visited.MaskedReduce(degrees, nenodes, Column, plus<int64_t>(), static_cast<int64_t>(0));
    
    // or use negative mask
    //FullyDistSpVec<int64_t, int64_t> nenodes(enodes,[](int64_t x){return x>0;});
    //visited.MaskedReduce(degrees1, nenodes, Column, plus<int64_t>(), static_cast<int64_t>(0), true);
    degrees.Apply([](int64_t val){return (val-1);}); // -1 to remove sources themselves
     */

    return FullyDistSpVec<int64_t, int64_t>(sources.TotalLength(), ri, degrees);
}





// assume that source is an enode
FullyDistSpVec<int64_t, int64_t> getReachesSPMV(FullyDistSpVec<int64_t, int64_t>& sources, PSpMat_Int64 & A, FullyDistVec<int64_t, int64_t>& enodes)
{
    int nprocs = sources.getcommgrid()->GetSize();
    int myrank = sources.getcommgrid()->GetRank();
    
    FullyDistSpVec<int64_t, int64_t> degrees = sources;
    vector<int64_t> locvals = sources.GetLocalInd();
    int64_t j = 0;
    
    for(int i=0; i<nprocs; )
    {
        int64_t s = -1;
        if(myrank==i && j<sources.getlocnnz())
        {
            s = locvals[j++] + sources.LengthUntil();
        }
        MPI_Bcast(&s, 1, MPIType<int64_t>(), i, sources.getcommgrid()->GetWorld());
        if(s!=-1)
        {
            FullyDistSpVec<int64_t, int64_t> reach = getReach(s, A, enodes);
            degrees.SetElement(s, reach.getnnz());
        }
        else i++;
    }
    return degrees;
}



void MD(PSpMat_Int64 & A)
{
    FullyDistVec<int64_t, int64_t> degrees ( A.getcommgrid());
    FullyDistVec<int64_t, int64_t> enodes (A.getcommgrid(), A.getnrow(), (int64_t) 0);
    FullyDistVec<int64_t, int64_t> mdOrder (A.getcommgrid(), A.getnrow(), (int64_t) 0);
    A.Reduce(degrees, Column, plus<int64_t>(), static_cast<int64_t>(0));
    degrees.Apply([](int64_t x){return x-1;}); // magic
    
    FullyDistVec<int64_t, double> treach (A.getcommgrid(), A.getnrow(), (double) 0);
    FullyDistVec<int64_t, double> treaches (A.getcommgrid(), A.getnrow(), (double) 0);
    FullyDistVec<int64_t, int64_t> nreach (A.getcommgrid(), A.getnrow(), (int64_t) 0);
    
    int myrank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    double time_beg = MPI_Wtime();
    
    //degrees.DebugPrint();
    double time1=0, time2=0, time3=0;
    for(int64_t i=0; i<A.getnrow(); i++)
    {
        //degrees.DebugPrint();
        int64_t s = degrees.MinElement().first; // minimum degree vertex
        enodes.SetElement(s, i+1);
        mdOrder.SetElement(i, s+1);
        
        time1 = MPI_Wtime();
        FullyDistSpVec<int64_t, int64_t> reach = getReach(s, A, enodes);
        time2 += MPI_Wtime()-time1;
        
        
        time1 = MPI_Wtime();
        FullyDistSpVec<int64_t, int64_t> updatedDeg( A.getcommgrid());
        //updatedDeg = getReachesSPMV(reach, A, enodes);
        updatedDeg = getReachesSPMM(reach, A, enodes);
        
        
        time3 += MPI_Wtime()-time1;
        
        degrees.Set(updatedDeg);
        degrees.SetElement(s, A.getnrow()); // set degree to infinite
        //degrees.DebugPrint();
        
        
        int nnz = reach.getnnz();
        if(myrank==0)
        {
            if(i%20==0)
            {
                cout << i << " .................. " << nnz << " :: " << time2 << " + " << time3 << endl;
                time2 = 0; time3 = 0;
            }
            
        }
        
    }
    
    
    double time_end = MPI_Wtime();
    
    
    if(myrank==0)
        cout << " Total time: " << time_end - time_beg << endl;
    
    
    
    //mdOrder.DebugPrint();
    
    
    
}


