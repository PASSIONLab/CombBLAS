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
#ifdef THREADED
#ifndef _OPENMP
#define _OPENMP
#endif
#endif

#ifdef _OPENMP
#include <omp.h>
#endif


double cblas_alltoalltime;
double cblas_allgathertime;
double cblas_mergeconttime;
double cblas_transvectime;
double cblas_localspmvtime;
#ifdef _OPENMP
int cblas_splits = omp_get_max_threads();
#else
int cblas_splits = 1;
#endif

#define EDGEFACTOR 16  /// changed to 8
#define GREEDY 1
#define KARP_SIPSER 2
#define KS_GREEDY 3
using namespace std;


MTRand GlobalMT(123); // for reproducable result

struct VertexType
{
public:
    VertexType(){parent=-1; degree = 0; prob=0;};
    VertexType(int64_t pa){parent=pa; degree=0; prob=0;}; // this constructor is called when we assign vertextype=number. Called from ApplyInd function
    VertexType(int64_t pa, int64_t deg){parent=pa; degree = deg; prob=0;};
    VertexType(int64_t pa, int64_t deg, int64_t p){parent=pa; degree = deg; prob=p;};
    
    friend bool operator==(const VertexType & vtx1, const VertexType & vtx2 ){return vtx1.parent==vtx2.parent;};
    friend ostream& operator<<(ostream& os, const VertexType & vertex ){os << "(" << vertex.parent << "," << vertex.degree << "," << vertex.prob << ")"; return os;};
    //private:
    int64_t parent;
    int64_t degree;
    int64_t prob;
};


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


// This one is used for maximal matching
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
typedef SpParMat < int64_t, bool, SpDCCols<int32_t,bool> > PSpMat_s32p64;
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
        
        
        //FullyDistVec<int64_t, int64_t> degrees(A.getcommgrid(), 20, 0);
        //sources.setNumToInd();
        //FullyDistVec<int64_t, int64_t> ri(sources);
        
        /*
        FullyDistVec<int64_t, int64_t> idx(A.getcommgrid());
        idx.iota(10, 0);
        idx.SetElement(1,0);
        FullyDistVec<int64_t, int64_t> val(A.getcommgrid());
        val.iota(10, 0);    
        FullyDistSpVec<int64_t, int64_t> test(10, idx, val, true);
        test.DebugPrint();
         */
        MD(A);
        
        double tstart = MPI_Wtime();
        
    }
    MPI_Finalize();
    return 0;
}



// assume that source is an enode
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
    reach.DelElement(source); // think a better way to remove source
    return reach;
}



template <class IT, class NT>
class Dist
{
public:
    typedef SpDCCols < IT, NT > DCCols;
    typedef SpParMat < IT, NT, DCCols > PSpMat;
};


// assume that source is an enode
FullyDistSpVec<int64_t, int64_t> getReachesSPMM(FullyDistSpVec<int64_t, int64_t>& sources, PSpMat_Int64 & A, FullyDistVec<int64_t, int64_t>& enodes)
{
    FullyDistVec<int64_t, int64_t> degrees(A.getcommgrid(), sources.TotalLength(), 0);
    sources.setNumToInd();
    FullyDistVec<int64_t, int64_t> ri(sources);
    FullyDistVec<int64_t, int64_t> ci(A.getcommgrid());
    ci.iota(ri.TotalLength(), 0);
    typedef PlusTimesSRing<int64_t, int64_t> PTDD;
    
    //typedef PlusTimesSRing<bool, int> PTBOOLINT;
    
    PSpMat_Int64  fringe(A.getnrow(), ri.TotalLength(), ri, ci, (int64_t) 1, false);
    PSpMat_Int64  visited(A.getnrow(), ri.TotalLength(), ri, ci, (int64_t) 1, false);
    
    
    FullyDistVec<int64_t, int64_t> ri1 = enodes.FindInds([](int64_t val){return val!=0;});
    FullyDistVec<int64_t, int64_t> ci1(A.getcommgrid(), ri1.TotalLength(), 1);
    FullyDistVec<int64_t, int64_t> ri2(A.getcommgrid(), ri.TotalLength(), 1);
    PSpMat_Int64  C(A.getnrow(), 1, ri1, ci1, (int64_t) 1, false);
    PSpMat_Int64  R(1, ri.TotalLength(), ri2, ci, (int64_t) 1, false);
    
    PSpMat_Int64  menodes = PSpGEMM<PTDD>(C, R);
    
    
    while( fringe.getnnz() > 0 )
    {
        fringe = PSpGEMM<PTDD>(A, fringe);
        fringe = EWiseMult(fringe, visited, true);
        visited += fringe;
        fringe = EWiseMult(fringe, menodes, false);
        
        
        FullyDistVec<int64_t, int64_t> * ColSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid());
        fringe.Reduce(*ColSums, Column, plus<int64_t>(), static_cast<int64_t>(0));
        degrees += *ColSums;
        delete ColSums;
    }
    
    
    //return degrees;
}



// assume that source is an enode
FullyDistSpVec<int64_t, int64_t> getReachesSPMM1(FullyDistSpVec<int64_t, int64_t>& sources, PSpMat_Int64 & A, FullyDistVec<int64_t, int64_t>& enodes)
{
    sources.setNumToInd();
    FullyDistVec<int64_t, int64_t> ri(sources);
    FullyDistVec<int64_t, int64_t> ci(A.getcommgrid());
    ci.iota(ri.TotalLength(), 0);
    typedef PlusTimesSRing<int64_t, int64_t> PTDD;
    
    PSpMat_Int64  R(A.getncol(), ri.TotalLength(), ri, ci, (int64_t) 1, false);
    PSpMat_Int64  visited(A.getncol(), ri.TotalLength(), ri, ci, (int64_t) 1, false);
    
    while(R.getnnz() > 0)
    {
        PSpMat_Int64 L = Mult_AnXBn_DoubleBuff<PTDD, int64_t, SpDCCols<int64_t,int64_t>>(A, R, false, false);
        //EWiseApply (const SpParMat<IU,NU1,UDERA> & A, const SpParMat<IU,NU2,UDERB> & B, _BinaryOperation __binary_op, _BinaryPredicate do_op, bool allowANulls, bool allowBNulls, const NU1& ANullVal, const NU2& BNullVal, const bool allowIntersect, const bool useExtendedBinOp);
        
        /*
         L = EWiseApply<int64_t, SpDCCols<int64_t,int64_t>>(L, visited,
         [](int64_t x, int64_t y){ return x;},
         [](int64_t x, int64_t y){return true;},
         false, true, (int64_t) 0, (int64_t) 0, false, false);
         
         
         visited = EWiseApply<int64_t, SpDCCols<int64_t,int64_t>>(visited, L,
         [](int64_t x, int64_t y){ return (int64_t)1;},
         [](int64_t x, int64_t y){return true;},
         true, true, (int64_t) 0, (int64_t) 0, true, false);
         // only keep enodes ... change
         L = EWiseApply<int64_t, SpDCCols<int64_t,int64_t>>(L, visited,
         [](int64_t x, int64_t y){ return x;},
         [](int64_t x, int64_t y){return true;},
         false, true, (int64_t) 0, (int64_t) 0, false, false);
         */
    }
    /*
     for(int64_t i=0; i<sources.TotalLength(); i++)
     {
     int64_t s = sources[i];
     if(sources.WasFound())
     {
     FullyDistSpVec<int64_t, int64_t> reach = getReach(i, A, enodes);
     degrees.SetElement(i, reach.getnnz());
     }
     }
     */
    
    //return degrees;
}


// assume that source is an enode
FullyDistSpVec<int64_t, int64_t> getReachesSPMV(FullyDistSpVec<int64_t, int64_t>& sources, PSpMat_Int64 & A, FullyDistVec<int64_t, int64_t>& enodes)
{
    //sources.DebugPrint();
    int nprocs = sources.getcommgrid()->GetSize();
    int myrank = sources.getcommgrid()->GetRank();
    
    FullyDistSpVec<int64_t, int64_t> degrees = sources;
    //FullyDistSpVec<int64_t, int64_t> degrees(sources.getcommgrid(), sources.TotalLength()); // same memory used by two vector?? no problem here
    
    vector<int64_t> locvals = sources.GetLocalInd();
    int64_t j = 0;
    
    for(int i=0; i<nprocs; )
    {
        int64_t s = -1;
        if(myrank==i && j<sources.getlocnnz())
        {
            s = locvals[j++];
        }
        MPI_Bcast(&s, 1, MPIType<int64_t>(), i, sources.getcommgrid()->GetWorld());
        if(s!=-1)
        {
            FullyDistSpVec<int64_t, int64_t> reach = getReach(s, A, enodes);
            degrees.SetElement(s, reach.getnnz());
        }
        else i++;
    }
    //degrees.DebugPrint();
    return degrees;
}

/*
// assume that source is an enode
FullyDistSpVec<int64_t, int64_t> getReachesSPMV(FullyDistSpVec<int64_t, int64_t>& sources, PSpMat_Int64 & A, FullyDistVec<int64_t, int64_t>& enodes)
{
    
    FullyDistVec<int64_t, int64_t> degrees(sources.getcommgrid(), sources.getnnz(), (int64_t) 0);
    //FullyDistSpVec<int64_t, int64_t> degrees(sources); // same memory used by two vector?? no problem here
    for(int64_t i=0; i<sources.TotalLength(); i++)
    {
        int64_t s = sources[i];
        if(sources.WasFound())
        {
            FullyDistSpVec<int64_t, int64_t> reach = getReach(i, A, enodes);
            degrees.SetElement(i, reach.getnnz());
        }
    }
    
    return degrees;
}
*/

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
    
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    double time_beg = MPI_Wtime();
    
    for(int64_t i=0; i<A.getnrow(); i++)
    {
        //degrees.DebugPrint();
        int64_t s = degrees.MinElement().first; // minimum degree vertex
        enodes.SetElement(s, i+1);
        mdOrder.SetElement(i, s+1);
        
        double time1 = MPI_Wtime();
        FullyDistSpVec<int64_t, int64_t> reach = getReach(s, A, enodes);
        reach.DebugPrint();
        FullyDistSpVec<int64_t, int64_t> reachInd = reach.FindInds([](int64_t x){return true;});
        reachInd.DebugPrint();
        FullyDistSpVec<int64_t, int64_t> reachVal = reach.FindVals([](int64_t x){return true;});
        reachVal.DebugPrint();
        double time2 = MPI_Wtime();
        //treach.SetElement(i, time2 - time1);
        //nreach.SetElement(i, reach.getnnz());
        FullyDistSpVec<int64_t, int64_t> updatedDeg = getReachesSPMV(reach, A, enodes);
        //treaches.SetElement(i, MPI_Wtime() - time2);

        
        degrees.Set(updatedDeg);
        degrees.SetElement(s, A.getnrow()); // set degree to infinite
        
        if(myrank==0)
        {
            if(i%10==0) cout << i << " \n";
        }
    }
    
    
    double time_end = MPI_Wtime();
    
    
    if(myrank==0)
        cout << " Total time: " << time_end - time_beg << endl;
    
    //mdOrder.DebugPrint();
    //nreach.DebugPrint();
    
    
    ofstream outf_nreach;
    outf_nreach.open("nreach.txt", ios::out);
    nreach.SaveGathered(outf_nreach, 0);
    
    ofstream outf_treach;
    outf_treach.open("treach.txt", ios::out);
    treach.SaveGathered(outf_treach, 0);
    
    ofstream outf_treaches;
    outf_treaches.open("treaches.txt", ios::out);
    treaches.SaveGathered(outf_treaches, 0);
}




/**
 * Create a boolean matrix A
 * Input: ri: a sparse vector of row indices
 * Output: a boolean matrix A with m=size(ri) and n=ncol
 * Let ri[k] contain the kth nonzero in ri, then A[j,k]=1
 */

template <class IT, class NT, class DER, typename _UnaryOperation>
SpParMat<IT, bool, DER> PerMat (const FullyDistSpVec<IT,NT> & ri, _UnaryOperation __unop)
{
    
    IT procsPerRow = ri.commGrid->GetGridCols();	// the number of processor in a row of processor grid
    IT procsPerCol = ri.commGrid->GetGridRows();	// the number of processor in a column of processor grid
    
    
    IT global_nrow = ri.TotalLength();
    IT global_ncol = ri.getnnz();
    IT m_perprocrow = global_nrow / procsPerRow;
    IT n_perproccol = global_ncol / procsPerCol;
    
    
    // The indices for FullyDistVec are offset'd to 1/p pieces
    // The matrix indices are offset'd to 1/sqrt(p) pieces
    // Add the corresponding offset before sending the data
    
    vector< vector<IT> > rowid(procsPerRow); // rowid in the local matrix of each vector entry
    vector< vector<IT> > colid(procsPerRow); // colid in the local matrix of each vector entry
    
    IT locvec = ri.num.size();	// nnz in local vector
    IT roffset = ri.RowLenUntil(); // the number of vector elements in this processor row before the current processor
    for(typename vector<IT>::size_type i=0; i< (unsigned)locvec; ++i)
    {
        IT val = __unop(ri.num[i]);
        if(val>=0 && val<global_ncol)
        {
            IT rowrec = (n_perproccol!=0) ? std::min(val / n_perproccol, procsPerRow-1) : (procsPerRow-1);
            // ri's numerical values give the colids and its local indices give rowids
            rowid[rowrec].push_back( i + roffset);
            colid[rowrec].push_back(val - (rowrec * n_perproccol));
        }
    }
    
    
    
    int * sendcnt = new int[procsPerRow];
    int * recvcnt = new int[procsPerRow];
    for(IT i=0; i<procsPerRow; ++i)
    {
        sendcnt[i] = rowid[i].size();
    }
    
    MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, ri.commGrid->GetRowWorld()); // share the counts
    
    int * sdispls = new int[procsPerRow]();
    int * rdispls = new int[procsPerRow]();
    partial_sum(sendcnt, sendcnt+procsPerRow-1, sdispls+1);
    partial_sum(recvcnt, recvcnt+procsPerRow-1, rdispls+1);
    IT p_nnz = accumulate(recvcnt,recvcnt+procsPerRow, static_cast<IT>(0));
    
    
    IT * p_rows = new IT[p_nnz];
    IT * p_cols = new IT[p_nnz];
    IT * senddata = new IT[locvec];
    for(int i=0; i<procsPerRow; ++i)
    {
        copy(rowid[i].begin(), rowid[i].end(), senddata+sdispls[i]);
        vector<IT>().swap(rowid[i]);	// clear memory of rowid
    }
    MPI_Alltoallv(senddata, sendcnt, sdispls, MPIType<IT>(), p_rows, recvcnt, rdispls, MPIType<IT>(), ri.commGrid->GetRowWorld());
    
    for(int i=0; i<procsPerRow; ++i)
    {
        copy(colid[i].begin(), colid[i].end(), senddata+sdispls[i]);
        vector<IT>().swap(colid[i]);	// clear memory of colid
    }
    MPI_Alltoallv(senddata, sendcnt, sdispls, MPIType<IT>(), p_cols, recvcnt, rdispls, MPIType<IT>(), ri.commGrid->GetRowWorld());
    delete [] senddata;
    
    tuple<IT,IT,bool> * p_tuples = new tuple<IT,IT,bool>[p_nnz];
    for(IT i=0; i< p_nnz; ++i)
    {
        p_tuples[i] = make_tuple(p_rows[i], p_cols[i], 1);
    }
    DeleteAll(p_rows, p_cols);
    
    
    // Now create the local matrix
    IT local_nrow = ri.MyRowLength();
    int my_proccol = ri.commGrid->GetRankInProcRow();
    IT local_ncol = (my_proccol<(procsPerCol-1))? (n_perproccol) : (global_ncol - (n_perproccol*(procsPerCol-1)));
    
    // infer the concrete type SpMat<IT,IT>
    typedef typename create_trait<DER, IT, bool>::T_inferred DER_IT;
    DER_IT * PSeq = new DER_IT();
    PSeq->Create( p_nnz, local_nrow, local_ncol, p_tuples);		// deletion of tuples[] is handled by SpMat::Create
    
    SpParMat<IT,bool,DER_IT> P (PSeq, ri.commGrid);
    //PSpMat_Bool P (PSeq, ri.commGrid);
    return P;
}



