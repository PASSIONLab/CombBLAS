//
//  ApproxWeightPerfectMatching.h
//  
//
//  Created by Ariful Azad on 8/22/17.
//
//

#ifndef ApproxWeightPerfectMatching_h
#define ApproxWeightPerfectMatching_h

#include "CombBLAS/CombBLAS.h"
#include "BPMaximalMatching.h"
#include "BPMaximumMatching.h"
#include <parallel/algorithm>
#include <parallel/numeric>
#include <memory>
#include <limits>


using namespace std;

namespace combblas {


template <class IT>
struct AWPM_param
{
    int nprocs;
    int myrank;
    int pr;
    int pc;
    IT lncol;
    IT lnrow;
    IT localRowStart;
    IT localColStart;
    IT m_perproc;
    IT n_perproc;
    std::shared_ptr<CommGrid> commGrid;
};


template <class IT, class NT>
std::vector<std::tuple<IT,IT,NT>> ExchangeData(std::vector<std::vector<std::tuple<IT,IT,NT>>> & tempTuples, MPI_Comm World)
{
	
	/* Create/allocate variables for vector assignment */
	MPI_Datatype MPI_tuple;
	MPI_Type_contiguous(sizeof(std::tuple<IT,IT,NT>), MPI_CHAR, &MPI_tuple);
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
	
	std::partial_sum(sendcnt, sendcnt+nprocs-1, sdispls+1);
	std::partial_sum(recvcnt, recvcnt+nprocs-1, rdispls+1);
	IT totrecv = accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));
	
	
	std::vector< std::tuple<IT,IT,NT> > sendTuples(totsend);
	for(int i=0; i<nprocs; ++i)
	{
		copy(tempTuples[i].begin(), tempTuples[i].end(), sendTuples.data()+sdispls[i]);
		std::vector< std::tuple<IT,IT,NT> >().swap(tempTuples[i]);	// clear memory
	}
	std::vector< std::tuple<IT,IT,NT> > recvTuples(totrecv);
	MPI_Alltoallv(sendTuples.data(), sendcnt, sdispls, MPI_tuple, recvTuples.data(), recvcnt, rdispls, MPI_tuple, World);
	DeleteAll(sendcnt, recvcnt, sdispls, rdispls); // free all memory
	MPI_Type_free(&MPI_tuple);
	return recvTuples;
	
}



template <class IT, class NT>
std::vector<std::tuple<IT,IT,IT,NT>> ExchangeData1(std::vector<std::vector<std::tuple<IT,IT,IT,NT>>> & tempTuples, MPI_Comm World)
{
	
	/* Create/allocate variables for vector assignment */
	MPI_Datatype MPI_tuple;
	MPI_Type_contiguous(sizeof(std::tuple<IT,IT,IT,NT>), MPI_CHAR, &MPI_tuple);
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
	
	std::partial_sum(sendcnt, sendcnt+nprocs-1, sdispls+1);
	std::partial_sum(recvcnt, recvcnt+nprocs-1, rdispls+1);
	IT totrecv = std::accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));
	
	std::vector< std::tuple<IT,IT,IT,NT> > sendTuples(totsend);
	for(int i=0; i<nprocs; ++i)
	{
		copy(tempTuples[i].begin(), tempTuples[i].end(), sendTuples.data()+sdispls[i]);
		std::vector< std::tuple<IT,IT,IT,NT> >().swap(tempTuples[i]);	// clear memory
	}
	std::vector< std::tuple<IT,IT,IT,NT> > recvTuples(totrecv);
	MPI_Alltoallv(sendTuples.data(), sendcnt, sdispls, MPI_tuple, recvTuples.data(), recvcnt, rdispls, MPI_tuple, World);
	DeleteAll(sendcnt, recvcnt, sdispls, rdispls); // free all memory
	MPI_Type_free(&MPI_tuple);
	return recvTuples;
}




// remember that getnrow() and getncol() require collectives
// Hence, we save them once and pass them to this function
template <class IT, class NT,class DER>
int OwnerProcs(SpParMat < IT, NT, DER > & A, IT grow, IT gcol, IT nrows, IT ncols)
{
	auto commGrid = A.getcommgrid();
	int procrows = commGrid->GetGridRows();
	int proccols = commGrid->GetGridCols();
	
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
	return commGrid->GetRank(pr, pc);
}


/*
// Hence, we save them once and pass them to this function
template <class IT, class NT,class DER>
int OwnerProcs(SpParMat < IT, NT, DER > & A, IT grow, IT gcol, IT nrows, IT ncols)
{

    

    int pr1, pc1;
    if(m_perproc != 0)
        pr1 = std::min(static_cast<int>(grow / m_perproc), pr-1);
    else	// all owned by the last processor row
        pr1 = pr -1;
    if(n_perproc != 0)
        pc1 = std::min(static_cast<int>(gcol / n_perproc), pc-1);
    else
        pc1 = pc-1;
    return commGrid->GetRank(pr1, pc1);
}
 */


template <class IT>
std::vector<std::tuple<IT,IT>> MateBcast(std::vector<std::tuple<IT,IT>> sendTuples, MPI_Comm World)
{
	
	/* Create/allocate variables for vector assignment */
	MPI_Datatype MPI_tuple;
	MPI_Type_contiguous(sizeof(std::tuple<IT,IT>) , MPI_CHAR, &MPI_tuple);
	MPI_Type_commit(&MPI_tuple);
	
	
	int nprocs;
	MPI_Comm_size(World, &nprocs);
	
	int * recvcnt = new int[nprocs];
	int * rdispls = new int[nprocs]();
	int sendcnt  = sendTuples.size();
	
	
	MPI_Allgather(&sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, World);
	
	std::partial_sum(recvcnt, recvcnt+nprocs-1, rdispls+1);
	IT totrecv = std::accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));
	
	std::vector< std::tuple<IT,IT> > recvTuples(totrecv);
	
	
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

template <class IT, class NT>
void ReplicateMateWeights( const AWPM_param<IT>& param, Dcsc<IT, NT>*dcsc, const std::vector<IT>& colptr, std::vector<IT>& RepMateC2R, std::vector<NT>& RepMateWR2C, std::vector<NT>& RepMateWC2R)
{
	
	
	fill(RepMateWC2R.begin(), RepMateWC2R.end(), static_cast<NT>(0));
	fill(RepMateWR2C.begin(), RepMateWR2C.end(), static_cast<NT>(0));
	
	
#ifdef THREADED
#pragma omp parallel for
#endif
    for(int k=0; k<param.lncol; ++k)
    {
        
        IT lj = k; // local numbering
        IT mj = RepMateC2R[lj]; // mate of j
        
        if(mj >= param.localRowStart && mj < (param.localRowStart+param.lnrow) )
        {
             for(IT cp = colptr[k]; cp < colptr[k+1]; ++cp)
             {
                 IT li = dcsc->ir[cp];
                 IT i = li + param.localRowStart;
                 // TODO: use binary search to directly go to mj-th entry if more than 32 nonzero in this column
                 if( i == mj)
                 {
                     RepMateWC2R[lj] = dcsc->numx[cp];
                     RepMateWR2C[mj-param.localRowStart] = dcsc->numx[cp];
                     //break;
                 }
             }
        }
    }


    
        MPI_Comm ColWorld = param.commGrid->GetColWorld();
        MPI_Comm RowWorld = param.commGrid->GetRowWorld();
	
	MPI_Allreduce(MPI_IN_PLACE, RepMateWC2R.data(), RepMateWC2R.size(), MPIType<NT>(), MPI_SUM, ColWorld);
	MPI_Allreduce(MPI_IN_PLACE, RepMateWR2C.data(), RepMateWR2C.size(), MPIType<NT>(), MPI_SUM, RowWorld);
}




template <class IT, class NT,class DER>
NT Trace( SpParMat < IT, NT, DER > & A, IT& rettrnnz=0)
{
	
	IT nrows = A.getnrow();
	IT ncols = A.getncol();
	auto commGrid = A.getcommgrid();
	MPI_Comm World = commGrid->GetWorld();
	int myrank=commGrid->GetRank();
	int pr = commGrid->GetGridRows();
	int pc = commGrid->GetGridCols();
	
	
	//Information about the matrix distribution
	//Assume that A is an nrow x ncol matrix
	//The local submatrix is an lnrow x lncol matrix
	int rowrank = commGrid->GetRankInProcRow();
	int colrank = commGrid->GetRankInProcCol();
	IT m_perproc = nrows / pr;
	IT n_perproc = ncols / pc;
	DER* spSeq = A.seqptr(); // local submatrix
	IT localRowStart = colrank * m_perproc; // first row in this process
	IT localColStart = rowrank * n_perproc; // first col in this process
	
	
	IT trnnz = 0;
	NT trace = 0.0;
	for(auto colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit) // iterate over columns
	{
		IT lj = colit.colid(); // local numbering
		IT j = lj + localColStart;
		
		for(auto nzit = spSeq->begnz(colit); nzit < spSeq->endnz(colit); ++nzit)
		{
			
			IT li = nzit.rowid();
			IT i = li + localRowStart;
			if( i == j)
			{
				trnnz ++;
				trace += nzit.value();
				
			}
		}
		
	}
	MPI_Allreduce(MPI_IN_PLACE, &trnnz, 1, MPIType<IT>(), MPI_SUM, World);
	MPI_Allreduce(MPI_IN_PLACE, &trace, 1, MPIType<NT>(), MPI_SUM, World);
    rettrnnz = trnnz;
    /*
	if(myrank==0)
		cout <<"nrows: " << nrows << " Nnz in the diag: " << trnnz << " sum of diag: " << trace << endl;
     */
    return trace;
	
}


template <class NT>
NT MatchingWeight( std::vector<NT>& RepMateWC2R, MPI_Comm RowWorld, NT& minw)
{
	NT w = 0;
	minw = 99999999999999.0;
	for(int i=0; i<RepMateWC2R.size(); i++)
	{
		//w += fabs(RepMateWC2R[i]);
		//w += exp(RepMateWC2R[i]);
		//minw = min(minw, exp(RepMateWC2R[i]));
		
		w += RepMateWC2R[i];
		minw = std::min(minw, RepMateWC2R[i]);
	}
	
	MPI_Allreduce(MPI_IN_PLACE, &w, 1, MPIType<NT>(), MPI_SUM, RowWorld);
	MPI_Allreduce(MPI_IN_PLACE, &minw, 1, MPIType<NT>(), MPI_MIN, RowWorld);
	return w;
}





// update the distributed mate vectors from replicated mate vectors
template <class IT>
void UpdateMatching(FullyDistVec<IT, IT>& mateRow2Col, FullyDistVec<IT, IT>& mateCol2Row, std::vector<IT>& RepMateR2C, std::vector<IT>& RepMateC2R)
{
	
	auto commGrid = mateRow2Col.getcommgrid();
	MPI_Comm RowWorld = commGrid->GetRowWorld();
	int rowroot = commGrid->GetDiagOfProcRow();
	int pc = commGrid->GetGridCols();
	
	// mateRow2Col is easy
	IT localLenR2C = mateRow2Col.LocArrSize();
	//IT* localR2C = mateRow2Col.GetLocArr();
	for(IT i=0, j = mateRow2Col.RowLenUntil(); i<localLenR2C; i++, j++)
	{
		mateRow2Col.SetLocalElement(i, RepMateR2C[j]);
		//localR2C[i] = RepMateR2C[j];
	}
	
	
	// mateCol2Row requires communication
	std::vector <int> sendcnts(pc);
	std::vector <int> dpls(pc);
	dpls[0] = 0;
	for(int i=1; i<pc; i++)
	{
		dpls[i] = mateCol2Row.RowLenUntil(i);
		sendcnts[i-1] = dpls[i] - dpls[i-1];
	}
	sendcnts[pc-1] = RepMateC2R.size() - dpls[pc-1];
	
	IT localLenC2R = mateCol2Row.LocArrSize();
	IT* localC2R = (IT*) mateCol2Row.GetLocArr();
	MPI_Scatterv(RepMateC2R.data(),sendcnts.data(), dpls.data(), MPIType<IT>(), localC2R, localLenC2R, MPIType<IT>(),rowroot, RowWorld);
}



int ThreadBuffLenForBinning(int itemsize, int nbins)
{
    // 1MB shared cache (per 2 cores) in KNL
#ifndef L2_CACHE_SIZE
#define L2_CACHE_SIZE 256000
#endif
    int THREAD_BUF_LEN = 256;
    while(true)
    {
        int bufferMem = THREAD_BUF_LEN * nbins * itemsize ;
        if(bufferMem>L2_CACHE_SIZE ) THREAD_BUF_LEN/=2;
        else break;
    }
    THREAD_BUF_LEN = std::min(nbins+1,THREAD_BUF_LEN);

    return THREAD_BUF_LEN;
}



template <class IT, class NT>
std::vector< std::tuple<IT,IT,NT> > Phase1(const AWPM_param<IT>& param, Dcsc<IT, NT>* dcsc, const std::vector<IT>& colptr, const std::vector<IT>& RepMateR2C, const std::vector<IT>& RepMateC2R, const std::vector<NT>& RepMateWR2C, const std::vector<NT>& RepMateWC2R )
{
    
    
    
    double tstart = MPI_Wtime();
    
    
    MPI_Comm World = param.commGrid->GetWorld();
    
   
    
    //Step 1: Count the amount of data to be sent to different processors
     std::vector<int> sendcnt(param.nprocs,0); // number items to be sent to each processor

    
#ifdef THREADED
#pragma omp parallel
#endif
    {
        std::vector<int> tsendcnt(param.nprocs,0);
#ifdef THREADED
#pragma omp for
#endif
        for(int k=0; k<param.lncol; ++k)
        {
            IT mj = RepMateC2R[k]; // lj = k
            IT j = k + param.localColStart;
            
            for(IT cp = colptr[k]; cp < colptr[k+1]; ++cp)
            {
                IT li = dcsc->ir[cp];
                IT i = li + param.localRowStart;
                IT mi = RepMateR2C[li];
                if( i > mj) // TODO : stop when first come to this, may be use <
                {
                    
                    int rrank = param.m_perproc != 0 ? std::min(static_cast<int>(mj / param.m_perproc), param.pr-1) : (param.pr-1);
                    int crank = param.n_perproc != 0 ? std::min(static_cast<int>(mi / param.n_perproc), param.pc-1) : (param.pc-1);
                    int owner = param.commGrid->GetRank(rrank , crank);
                    tsendcnt[owner]++;
                }
            }
        }
        for(int i=0; i<param.nprocs; i++)
        {
            __sync_fetch_and_add(sendcnt.data()+i, tsendcnt[i]);
        }
    }
    
    
    
    
    IT totsend = std::accumulate(sendcnt.data(), sendcnt.data()+param.nprocs, static_cast<IT>(0));
    std::vector<int> sdispls (param.nprocs, 0);
    std::partial_sum(sendcnt.data(), sendcnt.data()+param.nprocs-1, sdispls.data()+1);
    
    std::vector< std::tuple<IT,IT,NT> > sendTuples(totsend);
    std::vector<int> transferCount(param.nprocs,0);
    int THREAD_BUF_LEN = ThreadBuffLenForBinning(24, param.nprocs);
    
    
    //Step 2: Compile data to be sent to different processors
#ifdef THREADED
#pragma omp parallel
#endif
    {
        std::vector<int> tsendcnt(param.nprocs,0);
        std::vector<std::tuple<IT,IT, NT>> tsendTuples (param.nprocs*THREAD_BUF_LEN);
#ifdef THREADED
#pragma omp for
#endif
        for(int k=0; k<param.lncol; ++k)
        {
            IT mj = RepMateC2R[k];
            IT lj = k;
            IT j = k + param.localColStart;
            
            for(IT cp = colptr[k]; cp < colptr[k+1]; ++cp)
            {
                IT li = dcsc->ir[cp];
                IT i = li + param.localRowStart;
                IT mi = RepMateR2C[li];
                if( i > mj) // TODO : stop when first come to this, may be use <
                {
                    double w = dcsc->numx[cp]- RepMateWR2C[li] - RepMateWC2R[lj];
                    int rrank = param.m_perproc != 0 ? std::min(static_cast<int>(mj / param.m_perproc), param.pr-1) : (param.pr-1);
                    int crank = param.n_perproc != 0 ? std::min(static_cast<int>(mi / param.n_perproc), param.pc-1) : (param.pc-1);
                    int owner = param.commGrid->GetRank(rrank , crank);
                    
                    if (tsendcnt[owner] < THREAD_BUF_LEN)
                    {
                        tsendTuples[THREAD_BUF_LEN * owner + tsendcnt[owner]] = std::make_tuple(mi, mj, w);
                        tsendcnt[owner]++;
                    }
                    else
                    {
                        int tt = __sync_fetch_and_add(transferCount.data()+owner, THREAD_BUF_LEN);
                        copy( tsendTuples.data()+THREAD_BUF_LEN * owner, tsendTuples.data()+THREAD_BUF_LEN * (owner+1) , sendTuples.data() + sdispls[owner]+ tt);
                        
                        tsendTuples[THREAD_BUF_LEN * owner] = std::make_tuple(mi, mj, w);
                        tsendcnt[owner] = 1;
                    }
                }
            }
        }
        for(int owner=0; owner < param.nprocs; owner++)
        {
            if (tsendcnt[owner] >0)
            {
                int tt = __sync_fetch_and_add(transferCount.data()+owner, tsendcnt[owner]);
                copy( tsendTuples.data()+THREAD_BUF_LEN * owner, tsendTuples.data()+THREAD_BUF_LEN * owner + tsendcnt[owner], sendTuples.data() + sdispls[owner]+ tt);
            }
        }
    }
    
    double t1Comp = MPI_Wtime() - tstart;
    tstart = MPI_Wtime();
    
    // Step 3: Communicate data
    
    std::vector<int> recvcnt (param.nprocs);
    std::vector<int> rdispls (param.nprocs, 0);
    
    MPI_Alltoall(sendcnt.data(), 1, MPI_INT, recvcnt.data(), 1, MPI_INT, World);
    std::partial_sum(recvcnt.data(), recvcnt.data()+param.nprocs-1, rdispls.data()+1);
    IT totrecv = std::accumulate(recvcnt.data(), recvcnt.data()+param.nprocs, static_cast<IT>(0));
    
    
    MPI_Datatype MPI_tuple;
    MPI_Type_contiguous(sizeof(std::tuple<IT,IT,NT>), MPI_CHAR, &MPI_tuple);
    MPI_Type_commit(&MPI_tuple);
    
    std::vector< std::tuple<IT,IT,NT> > recvTuples1(totrecv);
    MPI_Alltoallv(sendTuples.data(), sendcnt.data(), sdispls.data(), MPI_tuple, recvTuples1.data(), recvcnt.data(), rdispls.data(), MPI_tuple, World);
    MPI_Type_free(&MPI_tuple);
    double t1Comm = MPI_Wtime() - tstart;
    return recvTuples1;
}





template <class IT, class NT>
std::vector< std::tuple<IT,IT,IT,NT> > Phase2(const AWPM_param<IT>& param, std::vector<std::tuple<IT,IT,NT>>& recvTuples, Dcsc<IT, NT>* dcsc, const std::vector<IT>& colptr, const std::vector<IT>& RepMateR2C, const std::vector<IT>& RepMateC2R, const std::vector<NT>& RepMateWR2C, const std::vector<NT>& RepMateWC2R )
{
    
    MPI_Comm World = param.commGrid->GetWorld();
    
    double tstart = MPI_Wtime();
    
    // Step 1: Sort for effecient searching of indices
    __gnu_parallel::sort(recvTuples.begin(), recvTuples.end());
    std::vector<std::vector<std::tuple<IT,IT, IT, NT>>> tempTuples1 (param.nprocs);
    
    std::vector<int> sendcnt(param.nprocs,0); // number items to be sent to each processor

    //Step 2: Count the amount of data to be sent to different processors
    // Instead of binary search in each column, I am doing linear search
    // Linear search is faster here because, we need to search 40%-50% of nnz
    int nBins = 1;
#ifdef THREADED
#pragma omp parallel
    {
        nBins = omp_get_num_threads() * 4;
    }
#endif
    
#ifdef THREADED
#pragma omp parallel for
#endif
    for(int i=0; i<nBins; i++)
    {
        int perBin = recvTuples.size()/nBins;
        int startBinIndex = perBin * i;
        int endBinIndex = perBin * (i+1);
        if(i==nBins-1) endBinIndex  = recvTuples.size();
        
        
        std::vector<int> tsendcnt(param.nprocs,0);
        for(int k=startBinIndex; k<endBinIndex;)
        {
            
                IT mi = std::get<0>(recvTuples[k]);
                IT lcol = mi - param.localColStart;
                IT i = RepMateC2R[lcol];
                IT idx1 = k;
                IT idx2 = colptr[lcol];
                
                for(; std::get<0>(recvTuples[idx1]) == mi && idx2 < colptr[lcol+1];) //**
                {
                    
                    IT mj = std::get<1>(recvTuples[idx1]) ;
                    IT lrow = mj - param.localRowStart;
                    IT j = RepMateR2C[lrow];
                    IT lrowMat = dcsc->ir[idx2];
                    if(lrowMat ==  lrow)
                    {
                        NT weight = std::get<2>(recvTuples[idx1]);
                        NT cw = weight + RepMateWR2C[lrow]; //w+W[M'[j],M[i]];
                        if (cw > 0)
                        {
                            int rrank = (param.m_perproc != 0) ? std::min(static_cast<int>(mj / param.m_perproc), param.pr-1) : (param.pr-1);
                            int crank = (param.n_perproc != 0) ? std::min(static_cast<int>(j / param.n_perproc), param.pc-1) : (param.pc-1);
                            int owner = param.commGrid->GetRank(rrank , crank);
                            tsendcnt[owner]++;
                        }
                        
                        idx1++; idx2++;
                    }
                    else if(lrowMat >  lrow)
                        idx1 ++;
                    else
                        idx2 ++;
                }
                
                for(;std::get<0>(recvTuples[idx1]) == mi ; idx1++);
                k = idx1;
             
        }
        for(int i=0; i<param.nprocs; i++)
        {
            __sync_fetch_and_add(sendcnt.data()+i, tsendcnt[i]);
        }
    }

    

    
    IT totsend = std::accumulate(sendcnt.data(), sendcnt.data()+param.nprocs, static_cast<IT>(0));
    std::vector<int> sdispls (param.nprocs, 0);
    std::partial_sum(sendcnt.data(), sendcnt.data()+param.nprocs-1, sdispls.data()+1);
    
    std::vector< std::tuple<IT,IT,IT,NT> > sendTuples(totsend);
    std::vector<int> transferCount(param.nprocs,0);
    int THREAD_BUF_LEN = ThreadBuffLenForBinning(32, param.nprocs);
    
    
    //Step 3: Compile data to be sent to different processors
#ifdef THREADED
#pragma omp parallel for
#endif
    for(int i=0; i<nBins; i++)
    {
        int perBin = recvTuples.size()/nBins;
        int startBinIndex = perBin * i;
        int endBinIndex = perBin * (i+1);
        if(i==nBins-1) endBinIndex  = recvTuples.size();
        
        
        std::vector<int> tsendcnt(param.nprocs,0);
        std::vector<std::tuple<IT,IT, IT, NT>> tsendTuples (param.nprocs*THREAD_BUF_LEN);
        for(int k=startBinIndex; k<endBinIndex;)
        {
            IT mi = std::get<0>(recvTuples[k]);
            IT lcol = mi - param.localColStart;
            IT i = RepMateC2R[lcol];
            IT idx1 = k;
            IT idx2 = colptr[lcol];
            
            for(; std::get<0>(recvTuples[idx1]) == mi && idx2 < colptr[lcol+1];) //**
            {
                
                IT mj = std::get<1>(recvTuples[idx1]) ;
                IT lrow = mj - param.localRowStart;
                IT j = RepMateR2C[lrow];
                IT lrowMat = dcsc->ir[idx2];
                if(lrowMat ==  lrow)
                {
                    NT weight = std::get<2>(recvTuples[idx1]);
                    NT cw = weight + RepMateWR2C[lrow]; //w+W[M'[j],M[i]];
                    if (cw > 0)
                    {
                        int rrank = (param.m_perproc != 0) ? std::min(static_cast<int>(mj / param.m_perproc), param.pr-1) : (param.pr-1);
                        int crank = (param.n_perproc != 0) ? std::min(static_cast<int>(j / param.n_perproc), param.pc-1) : (param.pc-1);
                        int owner = param.commGrid->GetRank(rrank , crank);
                        
                        if (tsendcnt[owner] < THREAD_BUF_LEN)
                        {
                            tsendTuples[THREAD_BUF_LEN * owner + tsendcnt[owner]] = std::make_tuple(mj, mi, i, cw);
                            tsendcnt[owner]++;
                        }
                        else
                        {
                            int tt = __sync_fetch_and_add(transferCount.data()+owner, THREAD_BUF_LEN);
                            std::copy( tsendTuples.data()+THREAD_BUF_LEN * owner, tsendTuples.data()+THREAD_BUF_LEN * (owner+1) , sendTuples.data() + sdispls[owner]+ tt);
                            
                            tsendTuples[THREAD_BUF_LEN * owner] = std::make_tuple(mj, mi, i, cw);
                            tsendcnt[owner] = 1;
                        }
                        
                    }
                    
                    idx1++; idx2++;
                }
                else if(lrowMat >  lrow)
                    idx1 ++;
                else
                    idx2 ++;
            }
            
            for(;std::get<0>(recvTuples[idx1]) == mi ; idx1++);
            k = idx1;
        }
        
        for(int owner=0; owner < param.nprocs; owner++)
        {
            if (tsendcnt[owner] >0)
            {
                int tt = __sync_fetch_and_add(transferCount.data()+owner, tsendcnt[owner]);
                std::copy( tsendTuples.data()+THREAD_BUF_LEN * owner, tsendTuples.data()+THREAD_BUF_LEN * owner + tsendcnt[owner], sendTuples.data() + sdispls[owner]+ tt);
            }
        }
    }

    // Step 4: Communicate data
    
    double t2Comp = MPI_Wtime() - tstart;
    tstart = MPI_Wtime();
    
    std::vector<int> recvcnt (param.nprocs);
    std::vector<int> rdispls (param.nprocs, 0);
    
    MPI_Alltoall(sendcnt.data(), 1, MPI_INT, recvcnt.data(), 1, MPI_INT, World);
    std::partial_sum(recvcnt.data(), recvcnt.data()+param.nprocs-1, rdispls.data()+1);
    IT totrecv = std::accumulate(recvcnt.data(), recvcnt.data()+param.nprocs, static_cast<IT>(0));
    

    MPI_Datatype MPI_tuple;
    MPI_Type_contiguous(sizeof(std::tuple<IT,IT,IT,NT>), MPI_CHAR, &MPI_tuple);
    MPI_Type_commit(&MPI_tuple);
    
    std::vector< std::tuple<IT,IT,IT,NT> > recvTuples1(totrecv);
    MPI_Alltoallv(sendTuples.data(), sendcnt.data(), sdispls.data(), MPI_tuple, recvTuples1.data(), recvcnt.data(), rdispls.data(), MPI_tuple, World);
    MPI_Type_free(&MPI_tuple);
    double t2Comm = MPI_Wtime() - tstart;
    return recvTuples1;
}


// Old version of Phase 2
// Not multithreaded (uses binary search)
template <class IT, class NT>
std::vector<std::vector<std::tuple<IT,IT, IT, NT>>> Phase2_old(const AWPM_param<IT>& param, std::vector<std::tuple<IT,IT,NT>>& recvTuples, Dcsc<IT, NT>* dcsc, const std::vector<IT>& colptr, const std::vector<IT>& RepMateR2C, const std::vector<IT>& RepMateC2R, const std::vector<NT>& RepMateWR2C, const std::vector<NT>& RepMateWC2R )
{
    
    std::vector<std::vector<std::tuple<IT,IT, IT, NT>>> tempTuples1 (param.nprocs);
    for(int k=0; k<recvTuples.size(); ++k)
    {
        IT mi = std::get<0>(recvTuples[k]) ;
        IT mj = std::get<1>(recvTuples[k]) ;
        IT i = RepMateC2R[mi - param.localColStart];
        NT weight = std::get<2>(recvTuples[k]);
        
        if(colptr[mi- param.localColStart+1] > colptr[mi- param.localColStart] )
        {
            IT * ele = find(dcsc->ir+colptr[mi - param.localColStart], dcsc->ir+colptr[mi - param.localColStart+1], mj - param.localRowStart);
            
            // TODO: Add a function that returns the edge weight directly
            if (ele != dcsc->ir+colptr[mi - param.localColStart+1])
            {
                NT cw = weight + RepMateWR2C[mj - param.localRowStart]; //w+W[M'[j],M[i]];
                if (cw > 0)
                {
                    IT j = RepMateR2C[mj - param.localRowStart];
                    int rrank = (param.m_perproc != 0) ? std::min(static_cast<int>(mj / param.m_perproc), param.pr-1) : (param.pr-1);
                    int crank = (param.n_perproc != 0) ? std::min(static_cast<int>(j / param.n_perproc), param.pc-1) : (param.pc-1);
                    int owner = param.commGrid->GetRank(rrank , crank);
                    tempTuples1[owner].push_back(make_tuple(mj, mi, i, cw));
                }
            }
        }
    }
  
    return tempTuples1;
}

template <class IT, class NT, class DER>
void TwoThirdApprox(SpParMat < IT, NT, DER > & A, FullyDistVec<IT, IT>& mateRow2Col, FullyDistVec<IT, IT>& mateCol2Row)
{
	
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
    IT nnz = A.getnnz();
	IT m_perproc = nrows / pr;
	IT n_perproc = ncols / pc;
	DER* spSeq = A.seqptr(); // local submatrix
	Dcsc<IT, NT>* dcsc = spSeq->GetDCSC();
	IT lnrow = spSeq->getnrow();
	IT lncol = spSeq->getncol();
	IT localRowStart = colrank * m_perproc; // first row in this process
	IT localColStart = rowrank * n_perproc; // first col in this process
	
    AWPM_param<IT> param;
    param.nprocs = nprocs;
    param.pr = pr;
    param.pc = pc;
    param.lncol = lncol;
    param.lnrow = lnrow;
    param.m_perproc = m_perproc;
    param.n_perproc = n_perproc;
    param.localRowStart = localRowStart;
    param.localColStart = localColStart;
    param.myrank = myrank;
    param.commGrid = commGrid;
    
    //double t1CompAll = 0, t1CommAll = 0, t2CompAll = 0, t2CommAll = 0, t3CompAll = 0, t3CommAll = 0, t4CompAll = 0, t4CommAll = 0, t5CompAll = 0, t5CommAll = 0, tUpdateMateCompAll = 0, tUpdateWeightAll = 0;
    
    double tPhase1 = 0, tPhase2 = 0, tPhase3 = 0, tPhase4 = 0, tPhase5 = 0, tUpdate = 0;
    
	
	// -----------------------------------------------------------
	// replicate mate vectors for mateCol2Row
	// Communication cost: same as the first communication of SpMV
	// -----------------------------------------------------------
	int xsize = (int)  mateCol2Row.LocArrSize();
	int trxsize = 0;
	MPI_Status status;
	MPI_Sendrecv(&xsize, 1, MPI_INT, diagneigh, TRX, &trxsize, 1, MPI_INT, diagneigh, TRX, World, &status);
	std::vector<IT> trxnums(trxsize);
	MPI_Sendrecv(mateCol2Row.GetLocArr(), xsize, MPIType<IT>(), diagneigh, TRX, trxnums.data(), trxsize, MPIType<IT>(), diagneigh, TRX, World, &status);
	
	
	std::vector<int> colsize(pc);
	colsize[colrank] = trxsize;
	MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, colsize.data(), 1, MPI_INT, ColWorld);
	std::vector<int> dpls(pc,0);	// displacements (zero initialized pid)
	std::partial_sum(colsize.data(), colsize.data()+pc-1, dpls.data()+1);
	int accsize = std::accumulate(colsize.data(), colsize.data()+pc, 0);
	std::vector<IT> RepMateC2R(accsize);
	MPI_Allgatherv(trxnums.data(), trxsize, MPIType<IT>(), RepMateC2R.data(), colsize.data(), dpls.data(), MPIType<IT>(), ColWorld);
	// -----------------------------------------------------------
	
	
    // -----------------------------------------------------------
	// replicate mate vectors for mateRow2Col
	// Communication cost: same as the first communication of SpMV
	//                      (minus the cost of tranposing vector)
	// -----------------------------------------------------------
	
	
	xsize = (int)  mateRow2Col.LocArrSize();
	
	std::vector<int> rowsize(pr);
	rowsize[rowrank] = xsize;
	MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, rowsize.data(), 1, MPI_INT, RowWorld);
	std::vector<int> rdpls(pr,0);	// displacements (zero initialized pid)
	std::partial_sum(rowsize.data(), rowsize.data()+pr-1, rdpls.data()+1);
	accsize = std::accumulate(rowsize.data(), rowsize.data()+pr, 0);
	std::vector<IT> RepMateR2C(accsize);
	MPI_Allgatherv(mateRow2Col.GetLocArr(), xsize, MPIType<IT>(), RepMateR2C.data(), rowsize.data(), rdpls.data(), MPIType<IT>(), RowWorld);
	// -----------------------------------------------------------
	
	
	
	// Getting column pointers for all columns (for CSC-style access)
    std::vector<IT> colptr (lncol+1,-1);
	for(auto colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit) // iterate over all columns
	{
		IT lj = colit.colid(); // local numbering
        
		colptr[lj] = colit.colptr();
	}
	colptr[lncol] = spSeq->getnnz();
	for(IT k=lncol-1; k>=0; k--)
	{
		if(colptr[k] == -1)
        {
            colptr[k] = colptr[k+1];
        }
	}
    // TODO: will this fail empty local matrix where every entry of colptr will be zero
	
    
    // -----------------------------------------------------------
    // replicate weights of mates
    // -----------------------------------------------------------
    std::vector<NT> RepMateWR2C(lnrow);
    std::vector<NT> RepMateWC2R(lncol);
    ReplicateMateWeights(param, dcsc, colptr, RepMateC2R, RepMateWR2C, RepMateWC2R);
    
	
	int iterations = 0;
	NT minw;
	NT weightCur = MatchingWeight(RepMateWC2R, RowWorld, minw);
	NT weightPrev = weightCur - 999999999999;
	while(weightCur > weightPrev && iterations++ < 10)
	{
		
		
#ifdef DETAIL_STATS
		if(myrank==0) std::cout << "Iteration " << iterations << ". matching weight: sum = "<< weightCur << " min = " << minw << std::endl;
#endif
		// C requests
		// each row is for a processor where C requests will be sent to
        double tstart = MPI_Wtime();
        std::vector<std::tuple<IT,IT,NT>> recvTuples = Phase1(param, dcsc, colptr, RepMateR2C, RepMateC2R, RepMateWR2C, RepMateWC2R );
        tPhase1 += (MPI_Wtime() - tstart);
        tstart = MPI_Wtime();
        
        std::vector<std::tuple<IT,IT,IT,NT>> recvTuples1 = Phase2(param, recvTuples, dcsc, colptr, RepMateR2C, RepMateC2R, RepMateWR2C, RepMateWC2R );
        std::vector< std::tuple<IT,IT,NT> >().swap(recvTuples);
        tPhase2 += (MPI_Wtime() - tstart);
        tstart = MPI_Wtime();
        
        
		std::vector<std::tuple<IT,IT,IT,NT>> bestTuplesPhase3 (lncol);
#ifdef THREADED
#pragma omp parallel for
#endif
		for(int k=0; k<lncol; ++k)
		{
			bestTuplesPhase3[k] = std::make_tuple(-1,-1,-1,0); // fix this
		}
		
		for(int k=0; k<recvTuples1.size(); ++k)
		{
			IT mj = std::get<0>(recvTuples1[k]) ;
			IT mi = std::get<1>(recvTuples1[k]) ;
			IT i = std::get<2>(recvTuples1[k]) ;
			NT weight = std::get<3>(recvTuples1[k]);
			IT j = RepMateR2C[mj - localRowStart];
			IT lj = j - localColStart;
			
			// we can get rid of the first check if edge weights are non negative
			if( (std::get<0>(bestTuplesPhase3[lj]) == -1)  || (weight > std::get<3>(bestTuplesPhase3[lj])) )
			{
				bestTuplesPhase3[lj] = std::make_tuple(i,mi,mj,weight);
			}
		}
		
		std::vector<std::vector<std::tuple<IT,IT, IT, NT>>> tempTuples1 (nprocs);
		for(int k=0; k<lncol; ++k)
		{
			if( std::get<0>(bestTuplesPhase3[k]) != -1)
			{
				//IT j = RepMateR2C[mj - localRowStart]; /// fix me
				
				IT i = std::get<0>(bestTuplesPhase3[k]) ;
				IT mi = std::get<1>(bestTuplesPhase3[k]) ;
				IT mj = std::get<2>(bestTuplesPhase3[k]) ;
				IT j = RepMateR2C[mj - localRowStart];
				NT weight = std::get<3>(bestTuplesPhase3[k]);
				int owner = OwnerProcs(A,  i, mi, nrows, ncols);
                
				tempTuples1[owner].push_back(std::make_tuple(i, j, mj, weight));
			}
		}
		//vector< tuple<IT,IT,IT, NT> >().swap(recvTuples1);
		recvTuples1 = ExchangeData1(tempTuples1, World);
		
        tPhase3 += (MPI_Wtime() - tstart);
        tstart = MPI_Wtime();
		
		std::vector<std::tuple<IT,IT,IT,IT, NT>> bestTuplesPhase4 (lncol);
		// we could have used lnrow in both bestTuplesPhase3 and bestTuplesPhase4
		
		// Phase 4
		// at the owner of (i,mi)
#ifdef THREADED
#pragma omp parallel for
#endif
		for(int k=0; k<lncol; ++k)
		{
			bestTuplesPhase4[k] = std::make_tuple(-1,-1,-1,-1,0);
		}
		
		for(int k=0; k<recvTuples1.size(); ++k)
		{
			IT i = std::get<0>(recvTuples1[k]) ;
			IT j = std::get<1>(recvTuples1[k]) ;
			IT mj = std::get<2>(recvTuples1[k]) ;
			IT mi = RepMateR2C[i-localRowStart];
			NT weight = std::get<3>(recvTuples1[k]);
			IT lmi = mi - localColStart;
			//IT lj = j - localColStart;
			
			// cout <<"****" << i << " " << mi << " "<< j << " " << mj << " " << get<0>(bestTuplesPhase4[lj]) << endl;
			// we can get rid of the first check if edge weights are non negative
			if( ((std::get<0>(bestTuplesPhase4[lmi]) == -1)  || (weight > std::get<4>(bestTuplesPhase4[lmi]))) && std::get<0>(bestTuplesPhase3[lmi])==-1 )
			{
				bestTuplesPhase4[lmi] = std::make_tuple(i,j,mi,mj,weight);
				//cout << "(("<< i << " " << mi << " "<< j << " " << mj << "))"<< endl;
			}
		}
		
		
		std::vector<std::vector<std::tuple<IT,IT,IT, IT>>> winnerTuples (nprocs);
		
		
		for(int k=0; k<lncol; ++k)
		{
			if( std::get<0>(bestTuplesPhase4[k]) != -1)
			{
				//int owner = OwnerProcs(A,  get<0>(bestTuples[k]), get<1>(bestTuples[k]), nrows, ncols); // (i,mi)
				//tempTuples[owner].push_back(bestTuples[k]);
				IT i = std::get<0>(bestTuplesPhase4[k]) ;
				IT j = std::get<1>(bestTuplesPhase4[k]) ;
				IT mi = std::get<2>(bestTuplesPhase4[k]) ;
				IT mj = std::get<3>(bestTuplesPhase4[k]) ;
				
				
				int owner = OwnerProcs(A,  mj, j, nrows, ncols);
				winnerTuples[owner].push_back(std::make_tuple(i, j, mi, mj));
				
				/// be very careful here
				// passing the opposite of the matching to the owner of (i,mi)
				owner = OwnerProcs(A,  i, mi, nrows, ncols);
				winnerTuples[owner].push_back(std::make_tuple(mj, mi, j, i));
			}
		}
		std::vector<std::tuple<IT,IT,IT,IT>> recvWinnerTuples = ExchangeData1(winnerTuples, World);
        tPhase4 += (MPI_Wtime() - tstart);
        tstart = MPI_Wtime();
		
		// at the owner of (mj,j)
		std::vector<std::tuple<IT,IT>> rowBcastTuples(recvWinnerTuples.size()); //(mi,mj)
		std::vector<std::tuple<IT,IT>> colBcastTuples(recvWinnerTuples.size()); //(j,i)
#ifdef THREADED
#pragma omp parallel for
#endif
		for(int k=0; k<recvWinnerTuples.size(); ++k)
		{
			IT i = std::get<0>(recvWinnerTuples[k]) ;
			IT j = std::get<1>(recvWinnerTuples[k]) ;
			IT mi = std::get<2>(recvWinnerTuples[k]) ;
			IT mj = std::get<3>(recvWinnerTuples[k]);

			colBcastTuples[k] = std::make_tuple(j,i);
			rowBcastTuples[k] = std::make_tuple(mj,mi);
		}
		
		std::vector<std::tuple<IT,IT>> updatedR2C = MateBcast(rowBcastTuples, RowWorld);
		std::vector<std::tuple<IT,IT>> updatedC2R = MateBcast(colBcastTuples, ColWorld);
        
        tPhase5 += (MPI_Wtime() - tstart);
        tstart = MPI_Wtime();
		
		
#ifdef THREADED
#pragma omp parallel for
#endif
		for(int k=0; k<updatedR2C.size(); k++)
		{
			IT row = std::get<0>(updatedR2C[k]);
			IT mate = std::get<1>(updatedR2C[k]);
			RepMateR2C[row-localRowStart] = mate;
		}
		
#ifdef THREADED
#pragma omp parallel for
#endif
		for(int k=0; k<updatedC2R.size(); k++)
		{
			IT col = std::get<0>(updatedC2R[k]);
			IT mate = std::get<1>(updatedC2R[k]);
			RepMateC2R[col-localColStart] = mate;
		}
		
		// update weights of matched edges
		// we can do better than this since we are doing sparse updates
		ReplicateMateWeights(param, dcsc, colptr, RepMateC2R, RepMateWR2C, RepMateWC2R);
		weightPrev = weightCur;
		weightCur = MatchingWeight(RepMateWC2R, RowWorld, minw);
		
        tUpdate += (MPI_Wtime() - tstart);
		
		//UpdateMatching(mateRow2Col, mateCol2Row, RepMateR2C, RepMateC2R);
		//CheckMatching(mateRow2Col,mateCol2Row);
		
	}
	
#ifdef TIMING
    if(myrank==0)
    {
        std::cout << "------------- overal timing (HWPM) -------------" << std::endl;
        //std::cout  <<  t1CompAll << " " << t1CommAll << " " << t2CompAll << " " << t2CommAll << " " << t3CompAll << " " << t3CommAll << " " << t4CompAll << " " << t4CommAll << " " << t5CompAll << " " << t5CommAll << " " << tUpdateMateCompAll << " " << tUpdateWeightAll << std::endl;
        std::cout  <<"Phase1: "<<  tPhase1 << "\nPhase2: " << tPhase2 << "\nPhase3: " << tPhase3 << "\nPhase4: " << tPhase4 << "\nPhase5: " << tPhase5 << "\nUpdate: " << tUpdate << std::endl;
        std::cout << "-------------------------------------------------" << std::endl;
    }
#endif
	// update the distributed mate vectors from replicated mate vectors
	UpdateMatching(mateRow2Col, mateCol2Row, RepMateR2C, RepMateC2R);
	//weightCur = MatchingWeight(RepMateWC2R, RowWorld);
	
	
	
}
    
    
    template <class IT, class NT, class DER>
    void TransformWeight(SpParMat < IT, NT, DER > & A, bool applylog)
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
        
        if(applylog)
            A.Apply([](NT val){return log(val);});
        
    }
    
    template <class IT, class NT>
    void AWPM(SpParMat < IT, NT, SpDCCols<IT, NT> > & A1, FullyDistVec<IT, IT>& mateRow2Col, FullyDistVec<IT, IT>& mateCol2Row, bool optimizeProd=true, bool weightedCard=true)
    {
        SpParMat < IT, NT, SpDCCols<IT, NT> > A(A1); // creating a copy because it is being transformed
        
        if(optimizeProd)
            TransformWeight(A, true);
        else
            TransformWeight(A, false);
        SpParMat < IT, NT, SpCCols<IT, NT> > Acsc(A);
        SpParMat < IT, NT, SpDCCols<IT, bool> > Abool(A);
        SpParMat < IT, NT, SpCCols<IT, bool> > ABoolCSC(MPI_COMM_WORLD);
        if(weightedCard)
            ABoolCSC = A;
        
        FullyDistVec<IT, IT> degCol(A.getcommgrid());
        Abool.Reduce(degCol, Column, plus<IT>(), static_cast<IT>(0));
        double ts;
        
        // Compute the initial trace
        IT diagnnz;
        double origWeight = Trace(A, diagnnz);
        bool isOriginalPerfect = diagnnz==A.getnrow();
        
        //--------------------------------------------------------
        // Compute the maximal cardinality matching
        //--------------------------------------------------------
        if(weightedCard)
            WeightedGreedy(Acsc, mateRow2Col, mateCol2Row, degCol);
        else
            WeightedGreedy(ABoolCSC, mateRow2Col, mateCol2Row, degCol);
        
        double mclWeight = MatchingWeight( A, mateRow2Col, mateCol2Row);
        bool isPerfectMCL = CheckMatching(mateRow2Col,mateCol2Row);
        
        // if the original matrix has a perfect matching and better weight
        if(isOriginalPerfect && mclWeight<=origWeight)
        {
            
#ifdef DETAIL_STATS
            SpParHelper::Print("Maximal matching is not better that the natural ordering. Hence, keeping the natural ordering.\n");
#endif
            mateRow2Col.iota(A.getnrow(), 0);
            mateCol2Row.iota(A.getncol(), 0);
            mclWeight = origWeight;
            isPerfectMCL = true;
        }
        
        
        //--------------------------------------------------------
        // Compute the maximum cardinality matching
        //--------------------------------------------------------
        double tmcm = 0;
        double mcmWeight = mclWeight;
        bool isPerfectMCM = isPerfectMCL;
        if(!isPerfectMCL) // run MCM only if we don't have a perfect matching
        {
            ts = MPI_Wtime();
            if(weightedCard)
                maximumMatching(Acsc, mateRow2Col, mateCol2Row, true, false, true);
            else
                maximumMatching(ABoolCSC, mateRow2Col, mateCol2Row, true, false, false);
            
            
            tmcm = MPI_Wtime() - ts;
            mcmWeight =  MatchingWeight( A, mateRow2Col, mateCol2Row) ;
            isPerfectMCM = CheckMatching(mateRow2Col,mateCol2Row);
        }
        
        if(!isPerfectMCM)
            SpParHelper::Print("Warning: The Maximum Cardinality Matching did not return a perfect matching! Need to check the input matrix.\n");
        
        //--------------------------------------------------------
        // Increase the weight of the perfect matching
        //--------------------------------------------------------
        ts = MPI_Wtime();
        TwoThirdApprox(A, mateRow2Col, mateCol2Row);
        double tawpm = MPI_Wtime() - ts;
        
        double awpmWeight =  MatchingWeight( A, mateRow2Col, mateCol2Row) ;

        bool isPerfectAWPM = CheckMatching(mateRow2Col,mateCol2Row);
        if(!isPerfectAWPM)
            SpParHelper::Print("Warning: The HWPM code did not return a perfect matching! Need to check the input matrix.\n");
        
        if(isOriginalPerfect && awpmWeight<origWeight) // keep original
        {
#ifdef DETAIL_STATS
            SpParHelper::Print("AWPM is not better that the natural ordering. Hence, keeping the natural ordering.\n");
#endif
            mateRow2Col.iota(A.getnrow(), 0);
            mateCol2Row.iota(A.getncol(), 0);
            awpmWeight = origWeight;
        }
    }

}

#endif /* ApproxWeightPerfectMatching_h */
