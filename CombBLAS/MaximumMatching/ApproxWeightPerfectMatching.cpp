
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


#include "BPMaximalMatching.h"
#include "BPMaximumMatching.h"

using namespace std;

// algorithmic options
bool prune, mvInvertMate, randMM, moreSplit;
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




template <typename PARMAT>
void Symmetricize(PARMAT & A)
{
    // boolean addition is practically a "logical or"
    // therefore this doesn't destruct any links
    PARMAT AT = A;
    AT.Transpose();
    A += AT;
}



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
        cout << "Usage (input matrix): ./awpm input <matrix> saveMatching\n\n";
        cout << " \n-------------- examples ----------\n";
        cout << "Example: mpirun -np 4 ./bpmm input cage12.mtx saveMatching\n" << endl;
    }
}



template <class IT, class NT>
vector<tuple<IT,IT,NT>> ExchangeData(vector<vector<tuple<IT,IT,NT>>> & tempTuples, MPI_Comm World)
{
    
    /* Create/allocate variables for vector assignment */
    MPI_Datatype MPI_tuple;
    MPI_Type_contiguous(sizeof(tuple<IT,IT,NT>), MPI_CHAR, &MPI_tuple);
    MPI_Type_commit(&MPI_tuple);
    
    int nprocs;
    MPI_Comm_size(World, &nprocs);
    
    int * sendcnt = new int[nprocs];
    int * recvcnt = new int[nprocs];
    int * sdispls = new int[nprocs]();
    int * rdispls = new int[nprocs]();
    
    // Set the newly found vector entries
    IT totsend = 0;
    for(IT i=0; i<nprocs; ++i)
    {
        sendcnt[i] = tempTuples[i].size();
        totsend += tempTuples[i].size();
    }
    
    MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, World);
    
    partial_sum(sendcnt, sendcnt+nprocs-1, sdispls+1);
    partial_sum(recvcnt, recvcnt+nprocs-1, rdispls+1);
    IT totrecv = accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));
    
    
    vector< tuple<IT,IT,NT> > sendTuples(totsend);
    for(int i=0; i<nprocs; ++i)
    {
        copy(tempTuples[i].begin(), tempTuples[i].end(), sendTuples.data()+sdispls[i]);
        vector< tuple<IT,IT,NT> >().swap(tempTuples[i]);	// clear memory
    }
    vector< tuple<IT,IT,NT> > recvTuples(totrecv);
    MPI_Alltoallv(sendTuples.data(), sendcnt, sdispls, MPI_tuple, recvTuples.data(), recvcnt, rdispls, MPI_tuple, World);
    DeleteAll(sendcnt, recvcnt, sdispls, rdispls); // free all memory
    MPI_Type_free(&MPI_tuple);
    return recvTuples;
    
}



template <class IT, class NT>
vector<tuple<IT,IT,IT,NT>> ExchangeData1(vector<vector<tuple<IT,IT,IT,NT>>> & tempTuples, MPI_Comm World)
{
    
    /* Create/allocate variables for vector assignment */
    MPI_Datatype MPI_tuple;
    MPI_Type_contiguous(sizeof(tuple<IT,IT,IT,NT>), MPI_CHAR, &MPI_tuple);
    MPI_Type_commit(&MPI_tuple);
    
    int nprocs;
    MPI_Comm_size(World, &nprocs);
    
    int * sendcnt = new int[nprocs];
    int * recvcnt = new int[nprocs];
    int * sdispls = new int[nprocs]();
    int * rdispls = new int[nprocs]();
    
    // Set the newly found vector entries
    IT totsend = 0;
    for(IT i=0; i<nprocs; ++i)
    {
        sendcnt[i] = tempTuples[i].size();
        totsend += tempTuples[i].size();
    }
    
    MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, World);
    
    partial_sum(sendcnt, sendcnt+nprocs-1, sdispls+1);
    partial_sum(recvcnt, recvcnt+nprocs-1, rdispls+1);
    IT totrecv = accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));
    
    vector< tuple<IT,IT,IT,NT> > sendTuples(totsend);
    for(int i=0; i<nprocs; ++i)
    {
        copy(tempTuples[i].begin(), tempTuples[i].end(), sendTuples.data()+sdispls[i]);
        vector< tuple<IT,IT,IT,NT> >().swap(tempTuples[i]);	// clear memory
    }
    vector< tuple<IT,IT,IT,NT> > recvTuples(totrecv);
    MPI_Alltoallv(sendTuples.data(), sendcnt, sdispls, MPI_tuple, recvTuples.data(), recvcnt, rdispls, MPI_tuple, World);
    DeleteAll(sendcnt, recvcnt, sdispls, rdispls); // free all memory
    MPI_Type_free(&MPI_tuple);
    return recvTuples;
    
}


template <class IT, class NT,class DER>
int OwnerProcs(SpParMat < IT, NT, DER > & A, IT grow, IT gcol, IT nrows, IT ncols)
{
    
    auto commGrid = A.getcommgrid();
    int procrows = commGrid->GetGridRows();
    int proccols = commGrid->GetGridCols();
    // remember that getnrow() and getncol() require collectives
    // Hence, we save them once and pass them to this function
    IT m_perproc = nrows / procrows;
    IT n_perproc = ncols / proccols;
    int pr, pc;
    if(m_perproc != 0)
    pr = std::min(static_cast<int>(grow / m_perproc), procrows-1);
    else	// all owned by the last processor row
    pr = procrows -1;
    if(n_perproc != 0)
    pc = std::min(static_cast<int>(gcol / n_perproc), proccols-1);
    else
    pc = proccols-1;
    if(grow > nrows)
    {
        cout << "grow > nrow: " << grow << " "<< nrows << endl;
        exit(1);
    }
    return commGrid->GetRank(pr, pc);
}



template <class IT>
vector<tuple<IT,IT>> MateBcast(vector<tuple<IT,IT>> sendTuples, MPI_Comm World)
{
    
    /* Create/allocate variables for vector assignment */
    MPI_Datatype MPI_tuple;
    MPI_Type_contiguous(sizeof(tuple<IT,IT>) , MPI_CHAR, &MPI_tuple);
    MPI_Type_commit(&MPI_tuple);
    
    
    int nprocs;
    MPI_Comm_size(World, &nprocs);
    
    int * recvcnt = new int[nprocs];
    int * rdispls = new int[nprocs]();
    int sendcnt  = sendTuples.size();
    
    
    MPI_Allgather(&sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, World);
    
    partial_sum(recvcnt, recvcnt+nprocs-1, rdispls+1);
    IT totrecv = accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));
    
    vector< tuple<IT,IT> > recvTuples(totrecv);
    
    
    MPI_Allgatherv(sendTuples.data(), sendcnt, MPI_tuple,
                   recvTuples.data(), recvcnt, rdispls,MPI_tuple,World );
    
    DeleteAll(recvcnt, rdispls); // free all memory
    MPI_Type_free(&MPI_tuple);
    return recvTuples;
    
}


// -----------------------------------------------------------
// replicate weights of mates
// Can be improved by removing AllReduce by All2All
// -----------------------------------------------------------

template <class IT, class NT,class DER>
void ReplicateMateWeights( SpParMat < IT, NT, DER > & A, vector<IT>& RepMateC2R, vector<NT>& RepMateWR2C, vector<NT>& RepMateWC2R, IT nrows, IT ncols)
{
    
    
    fill(RepMateWC2R.begin(), RepMateWC2R.end(), static_cast<NT>(0));
    fill(RepMateWR2C.begin(), RepMateWR2C.end(), static_cast<NT>(0));
    
    
    auto commGrid = A.getcommgrid();
    MPI_Comm World = commGrid->GetWorld();
    MPI_Comm ColWorld = commGrid->GetColWorld();
    MPI_Comm RowWorld = commGrid->GetRowWorld();
    int nprocs = commGrid->GetSize();
    int pr = commGrid->GetGridRows();
    int pc = commGrid->GetGridCols();
    int rowrank = commGrid->GetRankInProcRow();
    int colrank = commGrid->GetRankInProcCol();
    int diagneigh = commGrid->GetComplementRank();
    
    //Information about the matrix distribution
    //Assume that A is an nrow x ncol matrix
    //The local submatrix is an lnrow x lncol matrix
    IT m_perproc = nrows / pr;
    IT n_perproc = ncols / pc;
    DER* spSeq = A.seqptr(); // local submatrix
    IT lnrow = spSeq->getnrow();
    IT lncol = spSeq->getncol();
    IT localRowStart = colrank * m_perproc; // first row in this process
    IT localColStart = rowrank * n_perproc; // first col in this process
    
    
    for(auto colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit) // iterate over columns
    {
        IT lj = colit.colid(); // local numbering
        IT mj = RepMateC2R[lj]; // mate of j
        if(mj >= localRowStart && mj < (localRowStart+lnrow) )
        {
            
            for(auto nzit = spSeq->begnz(colit); nzit < spSeq->endnz(colit); ++nzit)
            {
                
                IT li = nzit.rowid();
                IT i = li + localRowStart;
                // TODO: use binary search to directly go to mj-th entry if more than 32 nonzero in this column
                if( i == mj)
                {
                    RepMateWC2R[lj] = nzit.value();
                    RepMateWR2C[mj-localRowStart] = nzit.value();
                }
            }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, RepMateWC2R.data(), RepMateWC2R.size(), MPIType<NT>(), MPI_SUM, ColWorld);
    MPI_Allreduce(MPI_IN_PLACE, RepMateWR2C.data(), RepMateWR2C.size(), MPIType<NT>(), MPI_SUM, RowWorld);
}



template <class NT>
NT MatchingWeight( vector<NT>& RepMateWC2R, MPI_Comm RowWorld)
{
    NT w = 0;
    for(int i=0; i<RepMateWC2R.size(); i++)
    {
        w += abs(RepMateWC2R[i]);
    }
    
    MPI_Allreduce(MPI_IN_PLACE, &w, 1, MPIType<NT>(), MPI_SUM, RowWorld);
    return w;
}

// update the distributed mate vectors from replicated mate vectors
template <class IT>
void UpdateMatching(FullyDistVec<IT, IT>& mateRow2Col, FullyDistVec<IT, IT>& mateCol2Row, vector<IT>& RepMateR2C, vector<IT>& RepMateC2R)
{
    
    auto commGrid = mateRow2Col.getcommgrid();
    MPI_Comm RowWorld = commGrid->GetRowWorld();
    int rowroot = commGrid->GetDiagOfProcRow();
    int pc = commGrid->GetGridCols();
    
    // mateRow2Col is easy
    IT localLenR2C = mateRow2Col.LocArrSize();
    IT* localR2C = mateRow2Col.GetLocArr();
    for(IT i=0, j = mateRow2Col.RowLenUntil(); i<localLenR2C; i++, j++)
    {
        localR2C[i] = RepMateR2C[j];
    }
    
    
    // mateCol2Row requires communication
    vector <int> sendcnts(pc);
    vector <int> dpls(pc);
    dpls[0] = 0;
    for(int i=1; i<pc; i++)
    {
        dpls[i] = mateCol2Row.RowLenUntil(i);
        sendcnts[i-1] = dpls[i] - dpls[i-1];
    }
    sendcnts[pc-1] = RepMateC2R.size() - dpls[pc-1];
    
    IT localLenC2R = mateCol2Row.LocArrSize();
    IT* localC2R = mateCol2Row.GetLocArr();
    MPI_Scatterv(RepMateC2R.data(),sendcnts.data(), dpls.data(), MPIType<IT>(), localC2R, localLenC2R, MPIType<IT>(),rowroot, RowWorld);
}

template <class IT, class NT, class DER>
void TwoThirdApprox(SpParMat < IT, NT, DER > & A, FullyDistVec<IT, IT>& mateRow2Col, FullyDistVec<IT, IT>& mateCol2Row)
{
    
    //A.PrintInfo();
    // Information about CommGrid and matrix layout
    // Assume that processes are laid in (pr x pc) process grid
    auto commGrid = A.getcommgrid();
    int myrank=commGrid->GetRank();
    MPI_Comm World = commGrid->GetWorld();
    MPI_Comm ColWorld = commGrid->GetColWorld();
    MPI_Comm RowWorld = commGrid->GetRowWorld();
    int nprocs = commGrid->GetSize();
    int pr = commGrid->GetGridRows();
    int pc = commGrid->GetGridCols();
    int rowrank = commGrid->GetRankInProcRow();
    int colrank = commGrid->GetRankInProcCol();
    int diagneigh = commGrid->GetComplementRank();
    
    //Information about the matrix distribution
    //Assume that A is an nrow x ncol matrix
    //The local submatrix is an lnrow x lncol matrix
    IT nrows = A.getnrow();
    IT ncols = A.getncol();
    IT m_perproc = nrows / pr;
    IT n_perproc = ncols / pc;
    DER* spSeq = A.seqptr(); // local submatrix
    IT lnrow = spSeq->getnrow();
    IT lncol = spSeq->getncol();
    IT localRowStart = colrank * m_perproc; // first row in this process
    IT localColStart = rowrank * n_perproc; // first col in this process
    
    
    
    //mateRow2Col.DebugPrint();
    //mateCol2Row.DebugPrint();
    
    // -----------------------------------------------------------
    // replicate mate vectors for mateCol2Row
    // Communication cost: same as the first communication of SpMV
    // -----------------------------------------------------------
    int xsize = (int)  mateCol2Row.LocArrSize();
    int trxsize = 0;
    MPI_Status status;
    MPI_Sendrecv(&xsize, 1, MPI_INT, diagneigh, TRX, &trxsize, 1, MPI_INT, diagneigh, TRX, World, &status);
    vector<IT> trxnums(trxsize);
    MPI_Sendrecv(mateCol2Row.GetLocArr(), xsize, MPIType<IT>(), diagneigh, TRX, trxnums.data(), trxsize, MPIType<IT>(), diagneigh, TRX, World, &status);
    
    
    vector<int> colsize(pc);
    colsize[colrank] = trxsize;
    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, colsize.data(), 1, MPI_INT, ColWorld);
    vector<int> dpls(pc,0);	// displacements (zero initialized pid)
    std::partial_sum(colsize.data(), colsize.data()+pc-1, dpls.data()+1);
    int accsize = std::accumulate(colsize.data(), colsize.data()+pc, 0);
    vector<IT> RepMateC2R(accsize);
    MPI_Allgatherv(trxnums.data(), trxsize, MPIType<IT>(), RepMateC2R.data(), colsize.data(), dpls.data(), MPIType<IT>(), ColWorld);
    // -----------------------------------------------------------
    
    
    //cout << endl;
    //for(int i=0; i<RepMateC2R.size(); i++ )
    //  cout << RepMateC2R[i] << " ";
    //cout << endl;
    
    // -----------------------------------------------------------
    // replicate mate vectors for mateRow2Col
    // Communication cost: same as the first communication of SpMV
    //                      (minus the cost of tranposing vector)
    // -----------------------------------------------------------
    
    
    xsize = (int)  mateRow2Col.LocArrSize();
    
    vector<int> rowsize(pr);
    rowsize[rowrank] = xsize;
    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, rowsize.data(), 1, MPI_INT, RowWorld);
    vector<int> rdpls(pr,0);	// displacements (zero initialized pid)
    std::partial_sum(rowsize.data(), rowsize.data()+pr-1, rdpls.data()+1);
    accsize = std::accumulate(rowsize.data(), rowsize.data()+pr, 0);
    vector<IT> RepMateR2C(accsize);
    MPI_Allgatherv(mateRow2Col.GetLocArr(), xsize, MPIType<IT>(), RepMateR2C.data(), rowsize.data(), rdpls.data(), MPIType<IT>(), RowWorld);
    // -----------------------------------------------------------
    
    
    // -----------------------------------------------------------
    // replicate weights of mates
    // -----------------------------------------------------------
    vector<NT> RepMateWR2C(lnrow);
    vector<NT> RepMateWC2R(lncol);
    ReplicateMateWeights(A, RepMateC2R, RepMateWR2C, RepMateWC2R, nrows, ncols);
    
    //cout << endl;
    //for(int i=0; i<RepMateR2C.size(); i++ )
    //cout << RepMateR2C[i] << " ";
    //cout << endl;
    
    
    
    int iterations = 0;
    NT weightCur = MatchingWeight(RepMateWC2R, RowWorld);
    NT weightPrev = weightCur - 9999999999;
    cout << "Iteration# " << iterations << " : current weight "<< weightCur << " prev: "<< weightPrev<< endl;
    while(weightCur > weightPrev && iterations++ < 5)
    {
        if(myrank==0) cout << "Iteration# " << iterations << " : matching weight "<< weightCur << endl;
        /*
         for(int ii=0; ii<RepMateC2R.size(); ii++)
         cout << "("<<RepMateC2R[ii] << "," << ii << ") ";
         cout << endl << "row: ";
         for(int ii=0; ii<RepMateR2C.size(); ii++)
         cout << "(" << ii << "," <<RepMateR2C[ii] << ") ";
         cout << endl;
         */
        
        MPI_Barrier(MPI_COMM_WORLD);
        cout << "Step1: " << endl;
        MPI_Barrier(MPI_COMM_WORLD);
        // C requests
        // each row is for a processor where C requests will be sent to
        vector<vector<tuple<IT,IT,NT>>> tempTuples (nprocs);
        MPI_Barrier(World);
        //cout << myrank << ") Step1: " << endl;
        for(auto colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit) // iterate over columns
        {
            
            IT lj = colit.colid(); // local numbering
            //if(myrank==0) cout << myrank << ") col: " << lj << " ********* "<< endl;
            IT j = lj + localColStart;
            IT mj = RepMateC2R[lj]; // mate of j
            //start nzit from mate colid;
            for(auto nzit = spSeq->begnz(colit); nzit < spSeq->endnz(colit); ++nzit)
            {
                IT li = nzit.rowid();
                IT i = li + localRowStart;
                IT mi = RepMateR2C[li];
                //if(myrank==0) cout << myrank << ") " << i << " " << mi << " "<< j << " " << mj << endl;
                // TODO: use binary search to directly start from RepMateC2R[colid]
                if( i > mj)
                {
                    double w = nzit.value()- RepMateWR2C[li] - RepMateWC2R[lj];
                    int owner = OwnerProcs(A, mj, mi, nrows, ncols); // think about the symmetry??
                    tempTuples[owner].push_back(make_tuple(mj, mi, w));
                    
                }
            }
        }
        
        //cout <<  myrank <<") Done Step1......: " << endl;
        //exchange C-request via All2All
        // there might be some empty mesages in all2all
        vector<tuple<IT,IT,NT>> recvTuples = ExchangeData(tempTuples, World);
        //tempTuples are cleared in ExchangeData function
        
        MPI_Barrier(MPI_COMM_WORLD);
        cout << "Step2: " << endl;
        MPI_Barrier(MPI_COMM_WORLD);
        
        vector<vector<tuple<IT,IT, IT, NT>>> tempTuples1 (nprocs);
        
        //if(myrank==0)
        //  cout << "dbg point 3 :: " << recvTuples.size() << endl;
        
        // at the owner of (mj,mi)
        //cout << "Step2: " << endl;
        for(int k=0; k<recvTuples.size(); ++k)
        {
            
            IT mj = get<0>(recvTuples[k]) ;
            IT mi = get<1>(recvTuples[k]) ;
            IT i = RepMateC2R[mi - localColStart];
            NT weight = get<2>(recvTuples[k]);
            
            DER temp = (*spSeq)(mj - localRowStart, mi - localColStart);
            // TODO: Add a function that returns the edge weight directly
            
            if(!temp.isZero()) // this entry exists
            {
                NT cw = weight + RepMateWR2C[mj - localRowStart]; //w+W[M'[j],M[i]];
                if (cw > 0)
                {
                    IT j = RepMateR2C[mj - localRowStart];
                    //if(myrank==0)
                    //cout << k << " mj=" << mj << " mi="<< mi << " i=" << i<< " j="<< j << endl;
                    //cout << i << " " << mi << " "<< j << " " << mj << endl;
                    int owner = OwnerProcs(A,  mj, j, nrows, ncols); // (mj,j)
                    if(owner > nprocs-1) cout << "error !!!\n";
                    tempTuples1[owner].push_back(make_tuple(mj, mi, i, cw)); // @@@@@ send i as well
                    //tempTuples[owner].push_back(make_tuple(mj, j, cw));
                }
            }
        }
        
        //vector< tuple<IT,IT,NT> >().swap(recvTuples);
        
        //exchange RC-requests via AllToAllv
        vector<tuple<IT,IT,IT,NT>> recvTuples1 = ExchangeData1(tempTuples1, World);
        
        MPI_Barrier(MPI_COMM_WORLD);
        cout << "Step3: " << endl;
        MPI_Barrier(MPI_COMM_WORLD);
        
        // at the owner of (mj,j)
        // Part 3
        //cout << "Step3: " << endl;
        vector<tuple<IT,IT,IT,NT>> bestTuplesPhase3 (lncol);
        for(int k=0; k<lncol; ++k)
        {
            bestTuplesPhase3[k] = make_tuple(-1,-1,-1,0); // fix this
        }
        
        for(int k=0; k<recvTuples1.size(); ++k)
        {
            IT mj = get<0>(recvTuples1[k]) ;
            IT mi = get<1>(recvTuples1[k]) ;
            IT i = get<2>(recvTuples1[k]) ;
            NT weight = get<3>(recvTuples1[k]);
            IT j = RepMateR2C[mj - localRowStart];
            IT lj = j - localColStart;
            
            // how can I get i from here ?? ***** // receive i as well
            
            // we can get rid of the first check if edge weights are non negative
            if( (get<0>(bestTuplesPhase3[lj]) == -1)  || (weight > get<3>(bestTuplesPhase3[lj])) )
            {
                bestTuplesPhase3[lj] = make_tuple(i,mi,mj,weight);
            }
        }
        
        
        for(int k=0; k<lncol; ++k)
        {
            if( get<0>(bestTuplesPhase3[k]) != -1)
            {
                //IT j = RepMateR2C[mj - localRowStart]; /// fix me
                
                IT i = get<0>(bestTuplesPhase3[k]) ;
                IT mi = get<1>(bestTuplesPhase3[k]) ;
                IT mj = get<2>(bestTuplesPhase3[k]) ;
                IT j = RepMateR2C[mj - localRowStart];
                NT weight = get<3>(bestTuplesPhase3[k]);
                int owner = OwnerProcs(A,  i, mi, nrows, ncols);
                tempTuples1[owner].push_back(make_tuple(i, j, mj, weight));
            }
        }
        
        //vector< tuple<IT,IT,IT, NT> >().swap(recvTuples1);
        recvTuples1 = ExchangeData1(tempTuples1, World);
        
        vector<tuple<IT,IT,IT,IT, NT>> bestTuplesPhase4 (lncol);
        // we could have used lnrow in both bestTuplesPhase3 and bestTuplesPhase4
        
        // Phase 4
        // at the owner of (i,mi)
        for(int k=0; k<lncol; ++k)
        {
            bestTuplesPhase4[k] = make_tuple(-1,-1,-1,-1,0);
        }
        
        for(int k=0; k<recvTuples1.size(); ++k)
        {
            IT i = get<0>(recvTuples1[k]) ;
            IT j = get<1>(recvTuples1[k]) ;
            IT mj = get<2>(recvTuples1[k]) ;
            IT mi = RepMateR2C[i-localRowStart];
            NT weight = get<3>(recvTuples1[k]);
            IT lmi = mi - localColStart;
            //IT lj = j - localColStart;
            
            // cout <<"****" << i << " " << mi << " "<< j << " " << mj << " " << get<0>(bestTuplesPhase4[lj]) << endl;
            // we can get rid of the first check if edge weights are non negative
            if( ((get<0>(bestTuplesPhase4[lmi]) == -1)  || (weight > get<4>(bestTuplesPhase4[lmi]))) && get<0>(bestTuplesPhase3[lmi])==-1 )
            {
                bestTuplesPhase4[lmi] = make_tuple(i,j,mi,mj,weight);
                //cout << "(("<< i << " " << mi << " "<< j << " " << mj << "))"<< endl;
            }
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        cout << "Step4: " << endl;
        MPI_Barrier(MPI_COMM_WORLD);
        
        vector<vector<tuple<IT,IT,IT, IT>>> winnerTuples (nprocs);
       
        
        for(int k=0; k<lncol; ++k)
        {
            if( get<0>(bestTuplesPhase4[k]) != -1)
            {
                //int owner = OwnerProcs(A,  get<0>(bestTuples[k]), get<1>(bestTuples[k]), nrows, ncols); // (i,mi)
                //tempTuples[owner].push_back(bestTuples[k]);
                IT i = get<0>(bestTuplesPhase4[k]) ;
                IT j = get<1>(bestTuplesPhase4[k]) ;
                IT mi = get<2>(bestTuplesPhase4[k]) ;
                IT mj = get<3>(bestTuplesPhase4[k]) ;
                
                
                int owner = OwnerProcs(A,  mj, j, nrows, ncols);
                winnerTuples[owner].push_back(make_tuple(i, j, mi, mj));
                
                /// be very careful here
                // passing the opposite of the matching to the owner of (i,mi)
                owner = OwnerProcs(A,  i, mi, nrows, ncols);
                winnerTuples[owner].push_back(make_tuple(mj, mi, j, i));
            }
        }
        
        
        //vector< tuple<IT,IT,IT, NT> >().swap(recvTuples1);
        MPI_Barrier(MPI_COMM_WORLD);
        cout << "Step5: " << endl;
        MPI_Barrier(MPI_COMM_WORLD);
        vector<tuple<IT,IT,IT,IT>> recvWinnerTuples = ExchangeData1(winnerTuples, World);
        
        // at the owner of (mj,j)
        vector<tuple<IT,IT>> rowBcastTuples; //(mi,mj)
        vector<tuple<IT,IT>> colBcastTuples; //(i,j)
        
        for(int k=0; k<recvWinnerTuples.size(); ++k)
        {
            IT i = get<0>(recvWinnerTuples[k]) ;
            IT j = get<1>(recvWinnerTuples[k]) ;
            IT mi = get<2>(recvWinnerTuples[k]) ;
            IT mj = get<3>(recvWinnerTuples[k]);
            //cout << "(("<< i << " " << mi << " "<< j << " " << mj << "))"<< endl;
            colBcastTuples.push_back(make_tuple(j,i));
            //rowBcastTuples.push_back(make_tuple(i,j));
            rowBcastTuples.push_back(make_tuple(mj,mi));
            //colBcastTuples.push_back(make_tuple(mi,mj));
        }
        MPI_Barrier(MPI_COMM_WORLD);
        cout << "Step6: " << endl;
        MPI_Barrier(MPI_COMM_WORLD);
        
        vector<tuple<IT,IT>> updatedR2C = MateBcast(rowBcastTuples, RowWorld);
        vector<tuple<IT,IT>> updatedC2R = MateBcast(colBcastTuples, ColWorld);
        
        MPI_Barrier(MPI_COMM_WORLD);
        cout << "Step7: " << endl;
        MPI_Barrier(MPI_COMM_WORLD);
        
        for(int k=0; k<updatedR2C.size(); k++)
        {
            IT row = get<0>(updatedR2C[k]);
            IT mate = get<1>(updatedR2C[k]);
            if( (row < localRowStart) || (row >= (localRowStart+lnrow)))
            {
                cout << "myrank: " << myrank << "row: " << row << "localRowStart: " << localRowStart << endl;
                exit(1);
            }
            RepMateR2C[row-localRowStart] = mate;
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        cout << "Step7.5: " << endl;
        MPI_Barrier(MPI_COMM_WORLD);
        
        for(int k=0; k<updatedC2R.size(); k++)
        {
            IT col = get<0>(updatedC2R[k]);
            IT mate = get<1>(updatedC2R[k]);
            RepMateC2R[col-localColStart] = mate;
        }
         
        
        MPI_Barrier(MPI_COMM_WORLD);
        cout << "Step8: " << endl;
        MPI_Barrier(MPI_COMM_WORLD);
        // update weights of matched edges
        // we can do better than this since we are doing sparse updates
        ReplicateMateWeights(A, RepMateC2R, RepMateWR2C, RepMateWC2R, nrows, ncols);
        
        MPI_Barrier(MPI_COMM_WORLD);
        cout << "Step9: " << endl;
        MPI_Barrier(MPI_COMM_WORLD);
        weightPrev = weightCur;
        weightCur = MatchingWeight(RepMateWC2R, RowWorld);
        
        MPI_Barrier(MPI_COMM_WORLD);
        cout << "Step10: " << endl;
        MPI_Barrier(MPI_COMM_WORLD);
        
        UpdateMatching(mateRow2Col, mateCol2Row, RepMateR2C, RepMateC2R);
        CheckMatching(mateRow2Col,mateCol2Row);
    }
    
    
    // update the distributed mate vectors from replicated mate vectors
    UpdateMatching(mateRow2Col, mateCol2Row, RepMateR2C, RepMateC2R);
    
    
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
    mvInvertMate = false;
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
            //GetOptions(argv+3, argc-3);
            
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
        
        
        SpParHelper::Print("**************************************************\n\n");
        
        // compute the maximum cardinality matching
        FullyDistVec<int64_t, int64_t> mateRow2Col ( A.getcommgrid(), A.getnrow(), (int64_t) -1);
        FullyDistVec<int64_t, int64_t> mateCol2Row ( A.getcommgrid(), A.getncol(), (int64_t) -1);
        
        // using best options for the maximum cardinality matching
        /*
         init = DMD; randMaximal = false; randMM = true; prune = true;
         MaximalMatching(A, AT, mateRow2Col, mateCol2Row, degCol, init, randMaximal);
         maximumMatching(A, mateRow2Col, mateCol2Row, prune, mvInvertMate, randMM);
         */
        
        init = DMD; randMaximal = false; randMM = false; prune = true;
        MaximalMatching(A, AT, mateRow2Col, mateCol2Row, degCol, init, randMaximal);
        CheckMatching(mateRow2Col,mateCol2Row);
        maximumMatching(A, mateRow2Col, mateCol2Row, prune, mvInvertMate, randMM);
        CheckMatching(mateRow2Col,mateCol2Row);
        TwoThirdApprox(*AWighted, mateRow2Col, mateCol2Row);
        CheckMatching(mateRow2Col,mateCol2Row);
        if(saveMatching && ofname!="")
        {
            mateRow2Col.ParallelWrite(ofname,false,false);
        }
        
        
    }
    MPI_Finalize();
    return 0;
}

