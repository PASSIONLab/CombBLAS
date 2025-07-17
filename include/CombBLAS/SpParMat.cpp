/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.6 -------------------------------------------------*/
/* date: 6/15/2017 ---------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc  --------------------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2017, The Regents of the University of California
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */



#include "SpParMat.h"
#include "ParFriends.h"
#include "Operations.h"
#include "FileHeader.h"
extern "C" {
#include "mmio.h"
}
#include <sys/types.h>
#include <sys/stat.h>

#include <mpi.h>
#include <fstream>
#include <algorithm>
#include <set>
#include <stdexcept>

namespace combblas {

/**
  * If every processor has a distinct triples file such as {A_0, A_1, A_2,... A_p} for p processors
 **/
template <class IT, class NT, class DER>
SpParMat< IT,NT,DER >::SpParMat (std::ifstream & input, MPI_Comm & world)
{
	assert( (sizeof(IT) >= sizeof(typename DER::LocalIT)) );
	if(!input.is_open())
	{
		perror("Input file doesn't exist\n");
		exit(-1);
	}
	commGrid.reset(new CommGrid(world, 0, 0));
	input >> (*spSeq);
}

template <class IT, class NT, class DER>
SpParMat< IT,NT,DER >::SpParMat (DER * myseq, MPI_Comm & world): spSeq(myseq)
{
	assert( (sizeof(IT) >= sizeof(typename DER::LocalIT)) );
	commGrid.reset(new CommGrid(world, 0, 0));
}

template <class IT, class NT, class DER>
SpParMat< IT,NT,DER >::SpParMat (DER * myseq, std::shared_ptr<CommGrid> grid): spSeq(myseq)
{
	assert( (sizeof(IT) >= sizeof(typename DER::LocalIT)) );
	commGrid = grid;
}	

template <class IT, class NT, class DER>
SpParMat< IT,NT,DER >::SpParMat (std::shared_ptr<CommGrid> grid)
{
	assert( (sizeof(IT) >= sizeof(typename DER::LocalIT)) );
	spSeq = new DER();
	commGrid = grid;
}

//! Deprecated. Don't call the default constructor
template <class IT, class NT, class DER>
SpParMat< IT,NT,DER >::SpParMat ()
{
	SpParHelper::Print("COMBBLAS Warning: It is dangerous to create (matrix) objects without specifying the communicator, are you sure you want to create this object in MPI_COMM_WORLD?\n");
	assert( (sizeof(IT) >= sizeof(typename DER::LocalIT)) );
	spSeq = new DER();
	commGrid.reset(new CommGrid(MPI_COMM_WORLD, 0, 0));
}

/**
* If there is a single file read by the master process only, use this and then call ReadDistribute()
**/
template <class IT, class NT, class DER>
SpParMat< IT,NT,DER >::SpParMat (MPI_Comm world)
{
    
    assert( (sizeof(IT) >= sizeof(typename DER::LocalIT)) );
    spSeq = new DER();
    commGrid.reset(new CommGrid(world, 0, 0));
}

template <class IT, class NT, class DER>
SpParMat< IT,NT,DER >::~SpParMat ()
{
	if(spSeq != NULL) delete spSeq;
}

template <class IT, class NT, class DER>
void SpParMat< IT,NT,DER >::FreeMemory ()
{
	if(spSeq != NULL) delete spSeq;
	spSeq = NULL;
}


/**
 * Private function to guide Select2 communication and avoid code duplication due to loop ends
 * @param[int, out] klimits {per column k limit gets updated for the next iteration}
 * @param[out] converged {items to remove from actcolsmap at next iteration{
 **/
template <class IT, class NT, class DER>
template <typename VT, typename GIT>	// GIT: global index type of vector
void SpParMat<IT,NT,DER>::TopKGather(std::vector<NT> & all_medians, std::vector<IT> & nnz_per_col, int & thischunk, int & chunksize,
                                     const std::vector<NT> & activemedians, const std::vector<IT> & activennzperc, int itersuntil,
                                     std::vector< std::vector<NT> > & localmat, const std::vector<IT> & actcolsmap, std::vector<IT> & klimits,
                                     std::vector<IT> & toretain, std::vector<std::vector<std::pair<IT,NT>>> & tmppair, IT coffset, const FullyDistVec<GIT,VT> & rvec) const
{
    int rankincol = commGrid->GetRankInProcCol();
    int colneighs = commGrid->GetGridRows();
    int nprocs = commGrid->GetSize();
    std::vector<double> finalWeightedMedians(thischunk, 0.0);
    
    MPI_Gather(activemedians.data() + itersuntil*chunksize, thischunk, MPIType<NT>(), all_medians.data(), thischunk, MPIType<NT>(), 0, commGrid->GetColWorld());
    MPI_Gather(activennzperc.data() + itersuntil*chunksize, thischunk, MPIType<IT>(), nnz_per_col.data(), thischunk, MPIType<IT>(), 0, commGrid->GetColWorld());

    if(rankincol == 0)
    {
        std::vector<double> columnCounts(thischunk, 0.0);
        std::vector< std::pair<NT, double> > mediansNweights(colneighs);  // (median,weight) pairs    [to be reused at each iteration]
        
        for(int j = 0; j < thischunk; ++j)  // for each column
        {
            for(int k = 0; k<colneighs; ++k)
            {
                size_t fetchindex = k*thischunk+j;
                columnCounts[j] += static_cast<double>(nnz_per_col[fetchindex]);
            }
            for(int k = 0; k<colneighs; ++k)
            {
                size_t fetchindex = k*thischunk+j;
                mediansNweights[k] = std::make_pair(all_medians[fetchindex], static_cast<double>(nnz_per_col[fetchindex]) / columnCounts[j]);
            }
            sort(mediansNweights.begin(), mediansNweights.end());   // sort by median
            
            double sumofweights = 0;
            int k = 0;
            while( k<colneighs && sumofweights < 0.5)
            {
                sumofweights += mediansNweights[k++].second;
            }
            finalWeightedMedians[j] = mediansNweights[k-1].first;
        }
    }
    MPI_Bcast(finalWeightedMedians.data(), thischunk, MPIType<double>(), 0, commGrid->GetColWorld());
    
    std::vector<IT> larger(thischunk, 0);
    std::vector<IT> smaller(thischunk, 0);
    std::vector<IT> equal(thischunk, 0);

#ifdef THREADED
#pragma omp parallel for
#endif
    for(int j = 0; j < thischunk; ++j)  // for each active column
    {
        size_t fetchindex = actcolsmap[j+itersuntil*chunksize];        
        for(size_t k = 0; k < localmat[fetchindex].size(); ++k)
        {
            // count those above/below/equal to the median
            if(localmat[fetchindex][k] > finalWeightedMedians[j])
                larger[j]++;
            else if(localmat[fetchindex][k] < finalWeightedMedians[j])
                smaller[j]++;
            else
                equal[j]++;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, larger.data(), thischunk, MPIType<IT>(), MPI_SUM, commGrid->GetColWorld());
    MPI_Allreduce(MPI_IN_PLACE, smaller.data(), thischunk, MPIType<IT>(), MPI_SUM, commGrid->GetColWorld());
    MPI_Allreduce(MPI_IN_PLACE, equal.data(), thischunk, MPIType<IT>(), MPI_SUM, commGrid->GetColWorld());
    
    int numThreads = 1;	// default case
#ifdef THREADED
    omp_lock_t lock[nprocs];    // a lock per recipient
    for (int i=0; i<nprocs; i++)
        omp_init_lock(&(lock[i]));
#pragma omp parallel
    {
        numThreads = omp_get_num_threads();
    }
#endif
    
    std::vector < std::vector<IT> > perthread2retain(numThreads);
    
#ifdef THREADED
#pragma omp parallel for
#endif
    for(int j = 0; j < thischunk; ++j)  // for each active column
    {
#ifdef THREADED
        int myThread = omp_get_thread_num();
#else
        int myThread = 0;
#endif
        
        // both clmapindex and fetchindex are unique for a given j (hence not shared among threads)
        size_t clmapindex = j+itersuntil*chunksize;     // klimits is of the same length as actcolsmap
        size_t fetchindex = actcolsmap[clmapindex];     // localmat can only be dereferenced using the original indices.
        
        // these following if/else checks are the same (because klimits/large/equal vectors are mirrored) on every processor along ColWorld
        if(klimits[clmapindex] <= larger[j]) // the entries larger than Weighted-Median are plentiful, we can discard all the smaller/equal guys
        {
            std::vector<NT> survivors;
            for(size_t k = 0; k < localmat[fetchindex].size(); ++k)
            {
                if(localmat[fetchindex][k] > finalWeightedMedians[j])  // keep only the large guys (even equal guys go)
                    survivors.push_back(localmat[fetchindex][k]);
            }
            localmat[fetchindex].swap(survivors);
            perthread2retain[myThread].push_back(clmapindex);    // items to retain in actcolsmap
        }
        else if (klimits[clmapindex] > larger[j] + equal[j]) // the elements that are either larger or equal-to are surely keepers, no need to reprocess them
        {
            std::vector<NT> survivors;
            for(size_t k = 0; k < localmat[fetchindex].size(); ++k)
            {
                if(localmat[fetchindex][k] < finalWeightedMedians[j])  // keep only the small guys (even equal guys go)
                    survivors.push_back(localmat[fetchindex][k]);
            }
            localmat[fetchindex].swap(survivors);
            
            klimits[clmapindex] -= (larger[j] + equal[j]);   // update the k limit for this column only
            perthread2retain[myThread].push_back(clmapindex);    // items to retain in actcolsmap
        }
        else  	// larger[j] < klimits[clmapindex] &&  klimits[clmapindex] <= larger[j] + equal[j]
        {	// we will always have equal[j] > 0 because the weighted median is part of the dataset so it has to be equal to itself.
            std::vector<NT> survivors;
            for(size_t k = 0; k < localmat[fetchindex].size(); ++k)
            {
                if(localmat[fetchindex][k] >= finalWeightedMedians[j])  // keep the larger and equal to guys (might exceed k-limit but that's fine according to MCL)
                    survivors.push_back(localmat[fetchindex][k]);
            }
            localmat[fetchindex].swap(survivors);
            
            // We found it: the kth largest element in column (coffset + fetchindex) is finalWeightedMedians[j]
            // But everyone in the same processor column has the information, only one of them should send it
            IT n_perproc = getlocalcols() / colneighs;  // find a typical processor's share
            int assigned = std::max(static_cast<int>(fetchindex/n_perproc), colneighs-1);
            if( assigned == rankincol)
            {
                IT locid;
                int owner = rvec.Owner(coffset + fetchindex, locid);
                
            #ifdef THREADED
                omp_set_lock(&(lock[owner]));
            #endif
                tmppair[owner].emplace_back(std::make_pair(locid, finalWeightedMedians[j]));
            #ifdef THREADED
                omp_unset_lock(&(lock[owner]));
            #endif
            }
        } // end_else
    } // end_for
    // ------ concatenate toretain "indices" processed by threads ------
    std::vector<IT> tdisp(numThreads+1);
    tdisp[0] = 0;
    for(int i=0; i<numThreads; ++i)
    {
        tdisp[i+1] = tdisp[i] + perthread2retain[i].size();
    }
    toretain.resize(tdisp[numThreads]);
    
#pragma omp parallel for
    for(int i=0; i< numThreads; i++)
    {
        std::copy(perthread2retain[i].data() , perthread2retain[i].data()+ perthread2retain[i].size(), toretain.data() + tdisp[i]);
    }
    
#ifdef THREADED
    for (int i=0; i<nprocs; i++)    // destroy the locks
        omp_destroy_lock(&(lock[i]));
#endif
}


//! identify the k-th maximum element in each column of a matrix
//! if the number of nonzeros in a column is less than k, return the numeric_limits<NT>::min()
//! This is an efficient implementation of the Saukas/Song algorithm
//! http://www.ime.usp.br/~einar/select/INDEX.HTM
//! Preferred for large k values
template <class IT, class NT, class DER>
template <typename VT, typename GIT>	// GIT: global index type of vector
bool SpParMat<IT,NT,DER>::Kselect2(FullyDistVec<GIT,VT> & rvec, IT k_limit) const
{ 
    if(*rvec.commGrid != *commGrid)
    {
        SpParHelper::Print("Grids are not comparable, SpParMat::Kselect() fails!", commGrid->GetWorld());
        MPI_Abort(MPI_COMM_WORLD,GRIDMISMATCH);
    }
    
     
    int rankincol = commGrid->GetRankInProcCol();
    int rankinrow = commGrid->GetRankInProcRow();
    int rowneighs = commGrid->GetGridCols();	// get # of processors on the row
    int colneighs = commGrid->GetGridRows();
    int myrank = commGrid->GetRank();
    int nprocs = commGrid->GetSize();

    FullyDistVec<GIT,IT> colcnt(commGrid);
    Reduce(colcnt, Column, std::plus<IT>(), (IT) 0, [](NT i){ return (IT) 1;});

    // <begin> Gather vector along columns (Logic copied from DimApply)
    int xsize = (int) colcnt.LocArrSize();
    int trxsize = 0;
    int diagneigh = colcnt.commGrid->GetComplementRank();
    MPI_Status status;
    MPI_Sendrecv(&xsize, 1, MPI_INT, diagneigh, TRX, &trxsize, 1, MPI_INT, diagneigh, TRX, commGrid->GetWorld(), &status);
	
    IT * trxnums = new IT[trxsize];
    MPI_Sendrecv(const_cast<IT*>(SpHelper::p2a(colcnt.arr)), xsize, MPIType<IT>(), diagneigh, TRX, trxnums, trxsize, MPIType<IT>(), diagneigh, TRX, commGrid->GetWorld(), &status);

    int * colsize = new int[colneighs];
    colsize[rankincol] = trxsize;		
    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, colsize, 1, MPI_INT, commGrid->GetColWorld());	
    int * dpls = new int[colneighs]();	// displacements (zero initialized pid) 
    std::partial_sum(colsize, colsize+colneighs-1, dpls+1);
    int accsize = std::accumulate(colsize, colsize+colneighs, 0);
    std::vector<IT> percsum(accsize);	// per column sum of the number of entries 

    MPI_Allgatherv(trxnums, trxsize, MPIType<IT>(), percsum.data(), colsize, dpls, MPIType<VT>(), commGrid->GetColWorld());
    DeleteAll(trxnums,colsize, dpls);
    // <end> Gather vector along columns
    
    IT locm = getlocalcols();   // length (number of columns) assigned to this processor (and processor column)    
    std::vector< std::vector<NT> > localmat(locm);    // some sort of minimal local copy of matrix
   
#ifdef COMBBLAS_DEBUG
    if(accsize != locm) 	std::cout << "Gather vector along columns logic is wrong" << std::endl;
#endif
    
    for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)	// iterate over columns
    {
	if(percsum[colit.colid()] >= k_limit)	// don't make a copy of an inactive column
	{
		localmat[colit.colid()].reserve(colit.nnz());
        	for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit < spSeq->endnz(colit); ++nzit)
       	 	{
            		localmat[colit.colid()].push_back(nzit.value());
        	}
	}
    }
        
    int64_t activecols = std::count_if(percsum.begin(), percsum.end(), [k_limit](IT i){ return i >= k_limit;});
    int64_t activennz = std::accumulate(percsum.begin(), percsum.end(), (int64_t) 0);
    
    int64_t totactcols, totactnnzs;
    MPI_Allreduce(&activecols, &totactcols, 1, MPIType<int64_t>(), MPI_SUM, commGrid->GetRowWorld());
    if(myrank == 0)   std::cout << "Number of initial active columns are " << totactcols << std::endl;

    MPI_Allreduce(&activennz, &totactnnzs, 1, MPIType<int64_t>(), MPI_SUM, commGrid->GetRowWorld());
    if(myrank == 0)   std::cout << "Number of initial nonzeros are " << totactnnzs << std::endl;

#ifdef COMBBLAS_DEBUG
    IT glactcols = colcnt.Count([k_limit](IT i){ return i >= k_limit;});
    if(myrank == 0)   std::cout << "Number of initial active columns are " << glactcols << std::endl;
    if(glactcols != totactcols)  if(myrank == 0) std::cout << "Wrong number of active columns are computed" << std::endl;
#endif

    
    rvec = FullyDistVec<GIT,VT> ( rvec.getcommgrid(), getncol(), std::numeric_limits<NT>::min());	// set length of rvec correctly
    
#ifdef COMBBLAS_DEBUG
    PrintInfo();
    rvec.PrintInfo("rvector");
#endif
    
    if(totactcols == 0)
    {
        std::ostringstream ss;
        ss << "TopK: k_limit (" << k_limit <<")" << " >= maxNnzInColumn. Returning the result of Reduce(Column, minimum<NT>()) instead..." << std::endl;
        SpParHelper::Print(ss.str());
        return false;   // no prune needed
    }
   
    
    std::vector<IT> actcolsmap(activecols);  // the map that gives the original index of that active column (this map will shrink over iterations)
    for (IT i=0, j=0; i< locm; ++i) {
        if(percsum[i] >= k_limit)
            actcolsmap[j++] = i;
    }
    
    std::vector<NT> all_medians;
    std::vector<IT> nnz_per_col;
    std::vector<IT> klimits(activecols, k_limit); // is distributed management of this vector needed?
    int activecols_lowerbound = 10*colneighs;
    
    
    IT * locncols = new IT[rowneighs];
    locncols[rankinrow] = locm;
    MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IT>(),locncols, 1, MPIType<IT>(), commGrid->GetRowWorld());
    IT coffset = std::accumulate(locncols, locncols+rankinrow, static_cast<IT>(0));
    delete [] locncols;
    
    /* Create/allocate variables for vector assignment */
    MPI_Datatype MPI_pair;
    MPI_Type_contiguous(sizeof(std::pair<IT,NT>), MPI_CHAR, &MPI_pair);
    MPI_Type_commit(&MPI_pair);
    
    int * sendcnt = new int[nprocs];
    int * recvcnt = new int[nprocs];
    int * sdispls = new int[nprocs]();
    int * rdispls = new int[nprocs]();
    
    while(totactcols > 0)
    {
        int chunksize, iterations, lastchunk;
        if(activecols > activecols_lowerbound)
        {
            // two reasons for chunking:
            // (1) keep memory limited to activecols (<= n/sqrt(p))
            // (2) avoid overflow in sentcount
            chunksize = static_cast<int>(activecols/colneighs); // invariant chunksize >= 10 (by activecols_lowerbound)
            iterations = std::max(static_cast<int>(activecols/chunksize), 1);
            lastchunk = activecols - (iterations-1)*chunksize; // lastchunk >= chunksize by construction
        }
        else
        {
            chunksize = activecols;
            iterations = 1;
            lastchunk = activecols;
        }
        std::vector<NT> activemedians(activecols);   // one per "active" column
        std::vector<IT> activennzperc(activecols);
   
#ifdef THREADED
#pragma omp parallel for
#endif
        for(IT i=0; i< activecols; ++i) // recompute the medians and nnzperc
        {
            size_t orgindex = actcolsmap[i];	// assert: no two threads will share the same "orgindex"
            if(localmat[orgindex].empty())
            {
                activemedians[i] = (NT) 0;
                activennzperc[i] = 0;
            }
            else
            {
                // this actually *sorts* increasing but doesn't matter as long we solely care about the median as opposed to a general nth element
                auto entriesincol(localmat[orgindex]);   // create a temporary vector as nth_element modifies the vector
                std::nth_element(entriesincol.begin(), entriesincol.begin() + entriesincol.size()/2, entriesincol.end());
                activemedians[i] = entriesincol[entriesincol.size()/2];
                activennzperc[i] = entriesincol.size();
            }
        }
        
        percsum.resize(activecols, 0);
        MPI_Allreduce(activennzperc.data(), percsum.data(), activecols, MPIType<IT>(), MPI_SUM, commGrid->GetColWorld());
        activennz = std::accumulate(percsum.begin(), percsum.end(), (int64_t) 0);
        
#ifdef COMBBLAS_DEBUG
        MPI_Allreduce(&activennz, &totactnnzs, 1, MPIType<int64_t>(), MPI_SUM, commGrid->GetRowWorld());
        if(myrank == 0)   std::cout << "Number of active nonzeros are " << totactnnzs << std::endl;
#endif
        
        std::vector<IT> toretain;
        if(rankincol == 0)
        {
            all_medians.resize(lastchunk*colneighs);
            nnz_per_col.resize(lastchunk*colneighs);
        }
        std::vector< std::vector< std::pair<IT,NT> > > tmppair(nprocs);
        for(int i=0; i< iterations-1; ++i)  // this loop should not be parallelized if we want to keep storage small
        {
            TopKGather(all_medians, nnz_per_col, chunksize, chunksize, activemedians, activennzperc, i, localmat, actcolsmap, klimits, toretain, tmppair, coffset, rvec);
        }
        TopKGather(all_medians, nnz_per_col, lastchunk, chunksize, activemedians, activennzperc, iterations-1, localmat, actcolsmap, klimits, toretain, tmppair, coffset, rvec);
        
        /* Set the newly found vector entries */
        IT totsend = 0;
        for(IT i=0; i<nprocs; ++i)
        {
            sendcnt[i] = tmppair[i].size();
            totsend += tmppair[i].size();
        }
        
        MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, commGrid->GetWorld());
        
        std::partial_sum(sendcnt, sendcnt+nprocs-1, sdispls+1);
        std::partial_sum(recvcnt, recvcnt+nprocs-1, rdispls+1);
        IT totrecv = std::accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));
	assert((totsend < std::numeric_limits<int>::max()));	
	assert((totrecv < std::numeric_limits<int>::max()));
        
        std::pair<IT,NT> * sendpair = new std::pair<IT,NT>[totsend];
        for(int i=0; i<nprocs; ++i)
        {
            std::copy(tmppair[i].begin(), tmppair[i].end(), sendpair+sdispls[i]);
            std::vector< std::pair<IT,NT> >().swap(tmppair[i]);	// clear memory
        }
        std::vector< std::pair<IT,NT> > recvpair(totrecv);
        MPI_Alltoallv(sendpair, sendcnt, sdispls, MPI_pair, recvpair.data(), recvcnt, rdispls, MPI_pair, commGrid->GetWorld());
        delete [] sendpair;

        IT updated = 0;
        for(auto & update : recvpair )    // Now, write these to rvec
        {
            updated++;
            rvec.arr[update.first] =  update.second;
        }
#ifdef COMBBLAS_DEBUG
        MPI_Allreduce(MPI_IN_PLACE, &updated, 1, MPIType<IT>(), MPI_SUM, commGrid->GetWorld());
        if(myrank  == 0) std::cout << "Total vector entries updated " << updated << std::endl;
#endif

        /* End of setting up the newly found vector entries */
        
        std::vector<IT> newactivecols(toretain.size());
        std::vector<IT> newklimits(toretain.size());
        IT newindex = 0;
        for(auto & retind : toretain )
        {
            newactivecols[newindex] = actcolsmap[retind];
            newklimits[newindex++] = klimits[retind];
        }
        actcolsmap.swap(newactivecols);
        klimits.swap(newklimits);
        activecols = actcolsmap.size();
        
        MPI_Allreduce(&activecols, &totactcols, 1, MPIType<int64_t>(), MPI_SUM, commGrid->GetRowWorld());
#ifdef COMBBLAS_DEBUG
        if(myrank  == 0) std::cout << "Number of active columns are " << totactcols << std::endl;
#endif
    }
    DeleteAll(sendcnt, recvcnt, sdispls, rdispls);
    MPI_Type_free(&MPI_pair);
    
#ifdef COMBBLAS_DEBUG
    if(myrank == 0)   std::cout << "Exiting kselect2"<< std::endl;
#endif
    return true;    // prune needed
}



template <class IT, class NT, class DER>
void SpParMat< IT,NT,DER >::Dump(std::string filename) const
{
	MPI_Comm World = commGrid->GetWorld();
	int rank = commGrid->GetRank();
	int nprocs = commGrid->GetSize();
		
	MPI_File thefile;
    char * _fn = const_cast<char*>(filename.c_str());
	MPI_File_open(World, _fn, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &thefile);

	int rankinrow = commGrid->GetRankInProcRow();
	int rankincol = commGrid->GetRankInProcCol();
	int rowneighs = commGrid->GetGridCols();	// get # of processors on the row
	int colneighs = commGrid->GetGridRows();

	IT * colcnts = new IT[rowneighs];
	IT * rowcnts = new IT[colneighs];
	rowcnts[rankincol] = getlocalrows();
	colcnts[rankinrow] = getlocalcols();

	MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IT>(), colcnts, 1, MPIType<IT>(), commGrid->GetRowWorld());
	IT coloffset = std::accumulate(colcnts, colcnts+rankinrow, static_cast<IT>(0));

	MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IT>(), rowcnts, 1, MPIType<IT>(), commGrid->GetColWorld());	
	IT rowoffset = std::accumulate(rowcnts, rowcnts+rankincol, static_cast<IT>(0));
	DeleteAll(colcnts, rowcnts);

	IT * prelens = new IT[nprocs];
	prelens[rank] = 2*getlocalnnz();
	MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IT>(), prelens, 1, MPIType<IT>(), commGrid->GetWorld());
	IT lengthuntil = std::accumulate(prelens, prelens+rank, static_cast<IT>(0));

	// The disp displacement argument specifies the position 
	// (absolute offset in bytes from the beginning of the file) 
	MPI_Offset disp = lengthuntil * sizeof(uint32_t);
	char native[] = "native";
	MPI_File_set_view(thefile, disp, MPI_UNSIGNED, MPI_UNSIGNED, native, MPI_INFO_NULL);
	uint32_t * gen_edges = new uint32_t[prelens[rank]];
	
	IT k = 0;
	for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)	// iterate over columns
	{
		for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)
		{
			gen_edges[k++] = (uint32_t) (nzit.rowid() + rowoffset);
			gen_edges[k++] = (uint32_t) (colit.colid() +  coloffset);
		}
	}
	assert(k == prelens[rank]);
	MPI_File_write(thefile, gen_edges, prelens[rank], MPI_UNSIGNED, NULL);	
	MPI_File_close(&thefile);

	delete [] prelens;
	delete [] gen_edges;
}


template <class IT, class NT, class DER>
void SpParMat< IT,NT,DER >::ParallelBinaryWrite(std::string filename) const
{
    int myrank = commGrid->GetRank();
    int nprocs = commGrid->GetSize();
    IT totalm = getnrow();
    IT totaln = getncol();
    IT totnnz = getnnz();
        
    
    const int64_t headersize = 52; // 52 is the size of the header, 4 characters + 6*8 integer space
    int64_t elementsize = 2*sizeof(IT)+sizeof(NT);
    int64_t localentries =  getlocalnnz();
    int64_t localbytes = localentries*elementsize ;   // localsize in bytes
    if(myrank == 0)
        localbytes += headersize;
    
    int64_t bytesuntil = 0;
    MPI_Exscan( &localbytes, &bytesuntil, 1, MPIType<int64_t>(), MPI_SUM, commGrid->GetWorld());
    if(myrank == 0) bytesuntil = 0;    // because MPI_Exscan says the recvbuf in process 0 is undefined
    int64_t bytestotal;
    MPI_Allreduce(&localbytes, &bytestotal, 1, MPIType<int64_t>(), MPI_SUM, commGrid->GetWorld());
    
    size_t writtensofar = 0;
    char * localdata = new char[localbytes];
    if(myrank ==0)
    {
        char start[5] = "HKDT";
        uint64_t hdr[6];
        hdr[0] = 2;    // version: 2.0
        hdr[1] = sizeof(NT);   // object size
        hdr[2] = 0;    // format: binary
        hdr[3] = totalm;    // number of rows
        hdr[4] = totaln;    // number of columns
        hdr[5] = totnnz;    // number of nonzeros
        
        std::memmove(localdata, start, 4);
        std::memmove(localdata+4, hdr, sizeof(hdr));
        writtensofar = headersize;
    }
       
    IT roffset = 0;
    IT coffset = 0;
    GetPlaceInGlobalGrid(roffset, coffset);
    roffset += 1;    // increment by 1 (binary format is 1-based)
    coffset += 1;
       
    for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)    // iterate over nonempty subcolumns
    {
        for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)
        {
            IT glrowid = nzit.rowid() + roffset;
            IT glcolid = colit.colid() + coffset;
            NT glvalue = nzit.value();
            std::memmove(localdata+writtensofar, &glrowid, sizeof(IT));
            std::memmove(localdata+writtensofar+sizeof(IT), &glcolid, sizeof(IT));
            std::memmove(localdata+writtensofar+2*sizeof(IT), &glvalue, sizeof(NT));
            writtensofar += (2*sizeof(IT) + sizeof(NT));
        }
    }
#ifdef IODEBUG
    if(myrank == 0)
	    cout << "local move happened..., writing to file\n";
#endif


    MPI_File thefile;
    MPI_File_open(commGrid->GetWorld(), (char*) filename.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &thefile) ;
    MPI_File_set_view(thefile, bytesuntil, MPI_CHAR, MPI_CHAR, (char*)"native", MPI_INFO_NULL);
    
    int64_t batchSize = 256 * 1024 * 1024;   // 256 MB (per processor)
    size_t localfileptr = 0;
    int64_t remaining = localbytes;
    int64_t totalremaining = bytestotal;
       
    while(totalremaining > 0)
       {
       #ifdef IODEBUG
           if(myrank == 0)
               std::cout << "Remaining " << totalremaining << " bytes to write in aggregate" << std::endl;
       #endif
           MPI_Status status;
           int curBatch = std::min(batchSize, remaining);
           MPI_File_write_all(thefile, localdata+localfileptr, curBatch, MPI_CHAR, &status);
           int count;
           MPI_Get_count(&status, MPI_CHAR, &count); // known bug: https://github.com/pmodels/mpich/issues/2332
           assert( (curBatch == 0) || (count == curBatch) ); // count can return the previous/wrong value when 0 elements are written
           localfileptr += curBatch;
           remaining -= curBatch;
           MPI_Allreduce(&remaining, &totalremaining, 1, MPIType<int64_t>(), MPI_SUM, commGrid->GetWorld());
       }
       MPI_File_close(&thefile);
       
       delete [] localdata;
}

template <class IT, class NT, class DER>
SpParMat< IT,NT,DER >::SpParMat (const SpParMat< IT,NT,DER > & rhs)
{
	if(rhs.spSeq != NULL)	
		spSeq = new DER(*(rhs.spSeq));  	// Deep copy of local block

	commGrid =  rhs.commGrid;
}

template <class IT, class NT, class DER>
SpParMat< IT,NT,DER > & SpParMat< IT,NT,DER >::operator=(const SpParMat< IT,NT,DER > & rhs)
{
	if(this != &rhs)		
	{
		//! Check agains NULL is probably unneccessary, delete won't fail on NULL
		//! But useful in the presence of a user defined "operator delete" which fails to check NULL
		if(spSeq != NULL) delete spSeq;
		if(rhs.spSeq != NULL)	
			spSeq = new DER(*(rhs.spSeq));  // Deep copy of local block
	
		commGrid = rhs.commGrid;
	}
	return *this;
}

template <class IT, class NT, class DER>
SpParMat< IT,NT,DER > & SpParMat< IT,NT,DER >::operator+=(const SpParMat< IT,NT,DER > & rhs)
{
	if(this != &rhs)		
	{
		if(*commGrid == *rhs.commGrid)	
		{
			(*spSeq) += (*(rhs.spSeq));
		}
		else
		{
			std::cout << "Grids are not comparable for parallel addition (A+B)" << std::endl; 
		}
	}
	else
	{
		std::cout<< "Missing feature (A+A): Use multiply with 2 instead !"<<std::endl;	
	}
	return *this;	
}

template <class IT, class NT, class DER>
float SpParMat< IT,NT,DER >::LoadImbalance() const
{
	IT totnnz = getnnz();	// collective call
	IT maxnnz = 0;    
	IT localnnz = (IT) spSeq->getnnz();
	MPI_Allreduce( &localnnz, &maxnnz, 1, MPIType<IT>(), MPI_MAX, commGrid->GetWorld());
	if(totnnz == 0) return 1;
 	return static_cast<float>((commGrid->GetSize() * maxnnz)) / static_cast<float>(totnnz);  
}

template <class IT, class NT, class DER>
IT SpParMat< IT,NT,DER >::getnnz() const
{
	IT totalnnz = 0;    
	IT localnnz = spSeq->getnnz();
	MPI_Allreduce( &localnnz, &totalnnz, 1, MPIType<IT>(), MPI_SUM, commGrid->GetWorld());
 	return totalnnz;  
}

template <class IT, class NT, class DER>
IT SpParMat< IT,NT,DER >::getnrow() const
{
	IT totalrows = 0;
	IT localrows = spSeq->getnrow();    
	MPI_Allreduce( &localrows, &totalrows, 1, MPIType<IT>(), MPI_SUM, commGrid->GetColWorld());
 	return totalrows;  
}

template <class IT, class NT, class DER>
IT SpParMat< IT,NT,DER >::getncol() const
{
	IT totalcols = 0;
	IT localcols = spSeq->getncol();    
	MPI_Allreduce( &localcols, &totalcols, 1, MPIType<IT>(), MPI_SUM, commGrid->GetRowWorld());
 	return totalcols;  
}

template <class IT, class NT, class DER>
template <typename _BinaryOperation>	
void SpParMat<IT,NT,DER>::DimApply(Dim dim, const FullyDistVec<IT, NT>& x, _BinaryOperation __binary_op)
{

	if(!(*commGrid == *(x.commGrid))) 		
	{
		std::cout << "Grids are not comparable for SpParMat::DimApply" << std::endl; 
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
	}

	MPI_Comm World = x.commGrid->GetWorld();
	MPI_Comm ColWorld = x.commGrid->GetColWorld();
	MPI_Comm RowWorld = x.commGrid->GetRowWorld();
	switch(dim)
	{
		case Column:	// scale each column
		{
			int xsize = (int) x.LocArrSize();
			int trxsize = 0;
			int diagneigh = x.commGrid->GetComplementRank();
			MPI_Status status;
			MPI_Sendrecv(&xsize, 1, MPI_INT, diagneigh, TRX, &trxsize, 1, MPI_INT, diagneigh, TRX, World, &status);
	
			NT * trxnums = new NT[trxsize];
			MPI_Sendrecv(const_cast<NT*>(SpHelper::p2a(x.arr)), xsize, MPIType<NT>(), diagneigh, TRX, trxnums, trxsize, MPIType<NT>(), diagneigh, TRX, World, &status);

			int colneighs, colrank;
			MPI_Comm_size(ColWorld, &colneighs);
			MPI_Comm_rank(ColWorld, &colrank);
			int * colsize = new int[colneighs];
			colsize[colrank] = trxsize;
		
			MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, colsize, 1, MPI_INT, ColWorld);	
			int * dpls = new int[colneighs]();	// displacements (zero initialized pid) 
			std::partial_sum(colsize, colsize+colneighs-1, dpls+1);
			int accsize = std::accumulate(colsize, colsize+colneighs, 0);
			NT * scaler = new NT[accsize];

			MPI_Allgatherv(trxnums, trxsize, MPIType<NT>(), scaler, colsize, dpls, MPIType<NT>(), ColWorld);
			DeleteAll(trxnums,colsize, dpls);

			for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)	// iterate over columns
			{
				for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)
				{
					nzit.value() = __binary_op(nzit.value(), scaler[colit.colid()]);
				}
			}
			delete [] scaler;
			break;
		}
		case Row:
		{
			int xsize = (int) x.LocArrSize();
			int rowneighs, rowrank;
			MPI_Comm_size(RowWorld, &rowneighs);
			MPI_Comm_rank(RowWorld, &rowrank);
			int * rowsize = new int[rowneighs];
			rowsize[rowrank] = xsize;
			MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, rowsize, 1, MPI_INT, RowWorld);
			int * dpls = new int[rowneighs]();	// displacements (zero initialized pid) 
			std::partial_sum(rowsize, rowsize+rowneighs-1, dpls+1);
			int accsize = std::accumulate(rowsize, rowsize+rowneighs, 0);
			NT * scaler = new NT[accsize];

			MPI_Allgatherv(const_cast<NT*>(SpHelper::p2a(x.arr)), xsize, MPIType<NT>(), scaler, rowsize, dpls, MPIType<NT>(), RowWorld);
			DeleteAll(rowsize, dpls);

			for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)
			{
				for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)
				{
					nzit.value() = __binary_op(nzit.value(), scaler[nzit.rowid()]);
				}
			}
			delete [] scaler;			
			break;
		}
		default:
		{
			std::cout << "Unknown scaling dimension, returning..." << std::endl;
			break;
		}
	}
}

template <class IT, class NT, class DER>
template <typename _BinaryOperation, typename _UnaryOperation >	
FullyDistVec<IT,NT> SpParMat<IT,NT,DER>::Reduce(Dim dim, _BinaryOperation __binary_op, NT id, _UnaryOperation __unary_op) const
{
    IT length;
    switch(dim)
    {
        case Column:
        {
            length = getncol();
            break;
        }
        case Row:
        {
            length = getnrow();
            break;
        }
        default:
        {
            std::cout << "Unknown reduction dimension, returning empty vector" << std::endl;
            break;
        }
    }
	FullyDistVec<IT,NT> parvec(commGrid, length, id);
	Reduce(parvec, dim, __binary_op, id, __unary_op);			
	return parvec;
}

template <class IT, class NT, class DER>
template <typename _BinaryOperation>	
FullyDistVec<IT,NT> SpParMat<IT,NT,DER>::Reduce(Dim dim, _BinaryOperation __binary_op, NT id) const
{
    IT length;
    switch(dim)
    {
        case Column:
        {
            length = getncol();
            break;
        }
        case Row:
        {
            length = getnrow();
            break;
        }
        default:
        {
            std::cout << "Unknown reduction dimension, returning empty vector" << std::endl;
            break;
        }
    }
	FullyDistVec<IT,NT> parvec(commGrid, length, id);
	Reduce(parvec, dim, __binary_op, id, myidentity<NT>()); // myidentity<NT>() is a no-op function
	return parvec;
}


template <class IT, class NT, class DER>
template <typename VT, typename GIT, typename _BinaryOperation>	
void SpParMat<IT,NT,DER>::Reduce(FullyDistVec<GIT,VT> & rvec, Dim dim, _BinaryOperation __binary_op, VT id) const
{
	Reduce(rvec, dim, __binary_op, id, myidentity<NT>() );				
}


template <class IT, class NT, class DER>
template <typename VT, typename GIT, typename _BinaryOperation, typename _UnaryOperation>	// GIT: global index type of vector	
void SpParMat<IT,NT,DER>::Reduce(FullyDistVec<GIT,VT> & rvec, Dim dim, _BinaryOperation __binary_op, VT id, _UnaryOperation __unary_op) const
{
    Reduce(rvec, dim, __binary_op, id, __unary_op, MPIOp<_BinaryOperation, VT>::op() );
}


template <class IT, class NT, class DER>
template <typename VT, typename GIT, typename _BinaryOperation, typename _UnaryOperation>	// GIT: global index type of vector
void SpParMat<IT,NT,DER>::Reduce(FullyDistVec<GIT,VT> & rvec, Dim dim, _BinaryOperation __binary_op, VT id, _UnaryOperation __unary_op, MPI_Op mympiop) const
{
	if(*rvec.commGrid != *commGrid)
	{
		SpParHelper::Print("Grids are not comparable, SpParMat::Reduce() fails!", commGrid->GetWorld());
		MPI_Abort(MPI_COMM_WORLD,GRIDMISMATCH);
	}
	switch(dim)
	{
		case Column:	// pack along the columns, result is a vector of size n
		{
			// We can't use rvec's distribution (rows first, columns later) here 
			// because the ownership model of the vector has the order P(0,0), P(0,1),...
			// column reduction will first generate vector distribution in P(0,0), P(1,0),... order.
			
			IT n_thiscol = getlocalcols();   // length assigned to this processor column
			int colneighs = commGrid->GetGridRows();	// including oneself
            		int colrank = commGrid->GetRankInProcCol();

			GIT * loclens = new GIT[colneighs];
			GIT * lensums = new GIT[colneighs+1]();	// begin/end points of local lengths

            		GIT n_perproc = n_thiscol / colneighs;    // length on a typical processor
            		if(colrank == colneighs-1)
                		loclens[colrank] = n_thiscol - (n_perproc*colrank);
            		else
                		loclens[colrank] = n_perproc;

			MPI_Allgather(MPI_IN_PLACE, 0, MPIType<GIT>(), loclens, 1, MPIType<GIT>(), commGrid->GetColWorld());
			std::partial_sum(loclens, loclens+colneighs, lensums+1);	// loclens and lensums are different, but both would fit in 32-bits

			std::vector<VT> trarr;
			typename DER::SpColIter colit = spSeq->begcol();
			for(int i=0; i< colneighs; ++i)
			{
				VT * sendbuf = new VT[loclens[i]];
				std::fill(sendbuf, sendbuf+loclens[i], id);	// fill with identity
                
				for(; colit != spSeq->endcol() && colit.colid() < lensums[i+1]; ++colit)	// iterate over a portion of columns
				{
					for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)	// all nonzeros in this column
					{
						sendbuf[colit.colid()-lensums[i]] = __binary_op(static_cast<VT>(__unary_op(nzit.value())), sendbuf[colit.colid()-lensums[i]]);
					}
				}
                
				VT * recvbuf = NULL;
				if(colrank == i)
				{
					trarr.resize(loclens[i]);
					recvbuf = SpHelper::p2a(trarr);	
				}
				MPI_Reduce(sendbuf, recvbuf, loclens[i], MPIType<VT>(), mympiop, i, commGrid->GetColWorld()); // root  = i
				delete [] sendbuf;
			}
			DeleteAll(loclens, lensums);

			GIT reallen;	// Now we have to transpose the vector
			GIT trlen = trarr.size();
			int diagneigh = commGrid->GetComplementRank();
			MPI_Status status;
			MPI_Sendrecv(&trlen, 1, MPIType<IT>(), diagneigh, TRNNZ, &reallen, 1, MPIType<IT>(), diagneigh, TRNNZ, commGrid->GetWorld(), &status);
	
			rvec.arr.resize(reallen);
			MPI_Sendrecv(SpHelper::p2a(trarr), trlen, MPIType<VT>(), diagneigh, TRX, SpHelper::p2a(rvec.arr), reallen, MPIType<VT>(), diagneigh, TRX, commGrid->GetWorld(), &status);
			rvec.glen = getncol();	// ABAB: Put a sanity check here
			break;

		}
		case Row:	// pack along the rows, result is a vector of size m
		{
			rvec.glen = getnrow();
			int rowneighs = commGrid->GetGridCols();
			int rowrank = commGrid->GetRankInProcRow();
			GIT * loclens = new GIT[rowneighs];
			GIT * lensums = new GIT[rowneighs+1]();	// begin/end points of local lengths
			loclens[rowrank] = rvec.MyLocLength();
			MPI_Allgather(MPI_IN_PLACE, 0, MPIType<GIT>(), loclens, 1, MPIType<GIT>(), commGrid->GetRowWorld());
			std::partial_sum(loclens, loclens+rowneighs, lensums+1);
			try
			{
				rvec.arr.resize(loclens[rowrank], id);

				// keeping track of all nonzero iterators within columns at once is unscalable w.r.t. memory (due to sqrt(p) scaling)
				// thus we'll do batches of column as opposed to all columns at once. 5 million columns take 80MB (two pointers per column)
				#define MAXCOLUMNBATCH 5 * 1024 * 1024
				typename DER::SpColIter begfinger = spSeq->begcol();	// beginning finger to columns
				
				// Each processor on the same processor row should execute the SAME number of reduce calls
				int numreducecalls = (int) ceil(static_cast<float>(spSeq->getnzc()) / static_cast<float>(MAXCOLUMNBATCH));
				int maxreducecalls;
				MPI_Allreduce( &numreducecalls, &maxreducecalls, 1, MPI_INT, MPI_MAX, commGrid->GetRowWorld());
				
				for(int k=0; k< maxreducecalls; ++k)
				{
					std::vector<typename DER::SpColIter::NzIter> nziters;
					typename DER::SpColIter curfinger = begfinger; 
					for(; curfinger != spSeq->endcol() && nziters.size() < MAXCOLUMNBATCH ; ++curfinger)	
					{
						nziters.push_back(spSeq->begnz(curfinger));
					}
					for(int i=0; i< rowneighs; ++i)		// step by step to save memory
					{
						VT * sendbuf = new VT[loclens[i]];
						std::fill(sendbuf, sendbuf+loclens[i], id);	// fill with identity
		
						typename DER::SpColIter colit = begfinger;		
						IT colcnt = 0;	// "processed column" counter
						for(; colit != curfinger; ++colit, ++colcnt)	// iterate over this batch of columns until curfinger
						{
							typename DER::SpColIter::NzIter nzit = nziters[colcnt];
							for(; nzit != spSeq->endnz(colit) && nzit.rowid() < lensums[i+1]; ++nzit)	// a portion of nonzeros in this column
							{
								sendbuf[nzit.rowid()-lensums[i]] = __binary_op(static_cast<VT>(__unary_op(nzit.value())), sendbuf[nzit.rowid()-lensums[i]]);
							}
							nziters[colcnt] = nzit;	// set the new finger
						}

						VT * recvbuf = NULL;
						if(rowrank == i)
						{
							for(int j=0; j< loclens[i]; ++j)
							{
								sendbuf[j] = __binary_op(rvec.arr[j], sendbuf[j]);	// rvec.arr will be overriden with MPI_Reduce, save its contents
							}
							recvbuf = SpHelper::p2a(rvec.arr);	
						}
						MPI_Reduce(sendbuf, recvbuf, loclens[i], MPIType<VT>(), mympiop, i, commGrid->GetRowWorld()); // root = i
						delete [] sendbuf;
					}
					begfinger = curfinger;	// set the next begfilter
				}
				DeleteAll(loclens, lensums);	
			}
			catch (std::length_error& le) 
			{
	 			 std::cerr << "Length error: " << le.what() << std::endl;
  			}
			break;
		}
		default:
		{
			std::cout << "Unknown reduction dimension, returning empty vector" << std::endl;
			break;
		}
	}
}

#ifndef KSELECTLIMIT
#define KSELECTLIMIT 10000
#endif

//! Kselect wrapper for a select columns of the matrix
//! Indices of the input sparse vectors kth denote the queried columns of the matrix
//! Upon return, values of kth stores the kth entries of the queried columns
//! Returns true if Kselect algorithm is invoked for at least one column
//! Otherwise, returns false
template <class IT, class NT, class DER>
template <typename VT, typename GIT>
bool SpParMat<IT,NT,DER>::Kselect(FullyDistSpVec<GIT,VT> & kth, IT k_limit, int kselectVersion) const
{
#ifdef COMBBLAS_DEBUG
    FullyDistVec<GIT,VT> test1(kth.getcommgrid());
    FullyDistVec<GIT,VT> test2(kth.getcommgrid());
    Kselect1(test1, k_limit, myidentity<NT>());
    Kselect2(test2, k_limit);
    if(test1 == test2)
        SpParHelper::Print("Kselect1 and Kselect2 producing same results\n");
    else
    {
        SpParHelper::Print("WARNING: Kselect1 and Kselect2 producing DIFFERENT results\n");
        test1.PrintToFile("test1");
        test2.PrintToFile("test2");
    }
#endif

    if(kselectVersion==1 || k_limit < KSELECTLIMIT)
    {
        return Kselect1(kth, k_limit, myidentity<NT>());
    }
    else
    {
        FullyDistVec<GIT,VT> kthAll ( getcommgrid());
        bool ret = Kselect2(kthAll, k_limit);
        FullyDistSpVec<GIT,VT> temp = EWiseApply<VT>(kth, kthAll,
                                                     [](VT spval, VT dval){return dval;},
                                                     [](VT spval, VT dval){return true;},
                                                     false, NT());
        kth = temp;
        return ret;
    }
}


//! Returns true if Kselect algorithm is invoked for at least one column
//! Otherwise, returns false
//! if false, rvec contains either the minimum entry in each column or zero
template <class IT, class NT, class DER>
template <typename VT, typename GIT>
bool SpParMat<IT,NT,DER>::Kselect(FullyDistVec<GIT,VT> & rvec, IT k_limit, int kselectVersion) const
{
#ifdef COMBBLAS_DEBUG
    FullyDistVec<GIT,VT> test1(rvec.getcommgrid());
    FullyDistVec<GIT,VT> test2(rvec.getcommgrid());
    Kselect1(test1, k_limit, myidentity<NT>());
    Kselect2(test2, k_limit);
    if(test1 == test2)
        SpParHelper::Print("Kselect1 and Kselect2 producing same results\n");
    else
    {
        SpParHelper::Print("WARNING: Kselect1 and Kselect2 producing DIFFERENT results\n");
        //test1.PrintToFile("test1");
        //test2.PrintToFile("test2");
    }
#endif
    
    if(kselectVersion==1 || k_limit < KSELECTLIMIT)
        return Kselect1(rvec, k_limit, myidentity<NT>());
    else
        return Kselect2(rvec, k_limit);
   
}

/* identify the k-th maximum element in each column of a matrix
** if the number of nonzeros in a column is less than or equal to k, return minimum entry
** Caution: this is a preliminary implementation: needs 3*(n/sqrt(p))*k memory per processor
** this memory requirement is too high for larger k
 */
template <class IT, class NT, class DER>
template <typename VT, typename GIT, typename _UnaryOperation>	// GIT: global index type of vector
bool SpParMat<IT,NT,DER>::Kselect1(FullyDistVec<GIT,VT> & rvec, IT k, _UnaryOperation __unary_op) const
{
    if(*rvec.commGrid != *commGrid)
    {
        SpParHelper::Print("Grids are not comparable, SpParMat::Kselect() fails!", commGrid->GetWorld());
        MPI_Abort(MPI_COMM_WORLD,GRIDMISMATCH);
    }
   
    std::cerr << "Dense kslect is called!! " << std::endl;
    FullyDistVec<IT, IT> nnzPerColumn (getcommgrid());
    Reduce(nnzPerColumn, Column, std::plus<IT>(), (IT)0, [](NT val){return (IT)1;});
    IT maxnnzPerColumn = nnzPerColumn.Reduce(maximum<IT>(), (IT)0);
    if(k>maxnnzPerColumn)
    {
        SpParHelper::Print("Kselect: k is greater then maxNnzInColumn. Calling Reduce instead...\n");
        Reduce(rvec, Column, minimum<NT>(), static_cast<NT>(0));
        return false;
    }
    
    IT n_thiscol = getlocalcols();   // length (number of columns) assigned to this processor (and processor column)
    
    // check, memory should be min(n_thiscol*k, local nnz)
    // hence we will not overflow for very large k
    std::vector<VT> sendbuf(n_thiscol*k);
    std::vector<IT> send_coldisp(n_thiscol+1,0);
    std::vector<IT> local_coldisp(n_thiscol+1,0);
    
    
    //displacement of local columns
    //local_coldisp is the displacement of all nonzeros per column
    //send_coldisp is the displacement of k nonzeros per column
    IT nzc = 0;
    if(spSeq->getnnz()>0)
    {
        typename DER::SpColIter colit = spSeq->begcol();
        for(IT i=0; i<n_thiscol; ++i)
        {
            local_coldisp[i+1] = local_coldisp[i];
            send_coldisp[i+1] = send_coldisp[i];
            if((colit != spSeq->endcol()) && (i==colit.colid()))
            {
                local_coldisp[i+1] += colit.nnz();
                if(colit.nnz()>=k)
                    send_coldisp[i+1] += k;
                else
                    send_coldisp[i+1] += colit.nnz();
                colit++;
                nzc++;
            }
        }
    }
    assert(local_coldisp[n_thiscol] == spSeq->getnnz());
    
    // a copy of local part of the matrix
    // this can be avoided if we write our own local kselect function instead of using partial_sort
    std::vector<VT> localmat(spSeq->getnnz());


#ifdef THREADED
#pragma omp parallel for
#endif
    for(IT i=0; i<nzc; i++)
    //for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)	// iterate over columns
    {
        typename DER::SpColIter colit = spSeq->begcol() + i;
        IT colid = colit.colid();
        IT idx = local_coldisp[colid];
        for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit < spSeq->endnz(colit); ++nzit)
        {
            localmat[idx++] = static_cast<VT>(__unary_op(nzit.value()));
        }
        
        if(colit.nnz()<=k)
        {
            std::sort(localmat.begin()+local_coldisp[colid], localmat.begin()+local_coldisp[colid+1], std::greater<VT>());
            std::copy(localmat.begin()+local_coldisp[colid], localmat.begin()+local_coldisp[colid+1], sendbuf.begin()+send_coldisp[colid]);
        }
        else
        {
            std::partial_sort(localmat.begin()+local_coldisp[colid], localmat.begin()+local_coldisp[colid]+k, localmat.begin()+local_coldisp[colid+1], std::greater<VT>());
            std::copy(localmat.begin()+local_coldisp[colid], localmat.begin()+local_coldisp[colid]+k, sendbuf.begin()+send_coldisp[colid]);
        }
    }
    
    std::vector<VT>().swap(localmat);
    std::vector<IT>().swap(local_coldisp);

    std::vector<VT> recvbuf(n_thiscol*k);
    std::vector<VT> tempbuf(n_thiscol*k);
    std::vector<IT> recv_coldisp(n_thiscol+1);
    std::vector<IT> templen(n_thiscol);
    
    int colneighs = commGrid->GetGridRows();
    int colrank = commGrid->GetRankInProcCol();
    
    for(int p=2; p <= colneighs; p*=2)
    {
       
        if(colrank%p == p/2) // this processor is a sender in this round
        {
            int receiver = colrank - ceil(p/2);
            MPI_Send(send_coldisp.data(), n_thiscol+1, MPIType<IT>(), receiver, 0, commGrid->GetColWorld());
            MPI_Send(sendbuf.data(), send_coldisp[n_thiscol], MPIType<VT>(), receiver, 1, commGrid->GetColWorld());
            //break;
        }
        else if(colrank%p == 0) // this processor is a receiver in this round
        {
            int sender = colrank + ceil(p/2);
            if(sender < colneighs)
            {
                
                MPI_Recv(recv_coldisp.data(), n_thiscol+1, MPIType<IT>(), sender, 0, commGrid->GetColWorld(), MPI_STATUS_IGNORE);
                MPI_Recv(recvbuf.data(), recv_coldisp[n_thiscol], MPIType<VT>(), sender, 1, commGrid->GetColWorld(), MPI_STATUS_IGNORE);
                


#ifdef THREADED
#pragma omp parallel for
#endif
                for(IT i=0; i<n_thiscol; ++i)
                {
                    // partial merge until first k elements
                    IT j=send_coldisp[i], l=recv_coldisp[i];
                    //IT templen[i] = k*i;
                    IT offset = k*i;
                    IT lid = 0;
                    for(; j<send_coldisp[i+1] && l<recv_coldisp[i+1] && lid<k;)
                    {
                        if(sendbuf[j] > recvbuf[l])  // decision
                            tempbuf[offset+lid++] = sendbuf[j++];
                        else
                            tempbuf[offset+lid++] = recvbuf[l++];
                    }
                    while(j<send_coldisp[i+1] && lid<k) tempbuf[offset+lid++] = sendbuf[j++];
                    while(l<recv_coldisp[i+1] && lid<k) tempbuf[offset+lid++] = recvbuf[l++];
                    templen[i] = lid;
                }
                
                send_coldisp[0] = 0;
                for(IT i=0; i<n_thiscol; i++)
                {
                    send_coldisp[i+1] = send_coldisp[i] + templen[i];
                }
                
               
#ifdef THREADED
#pragma omp parallel for
#endif
                for(IT i=0; i<n_thiscol; i++) // direct copy
                {
                    IT offset = k*i;
                    std::copy(tempbuf.begin()+offset, tempbuf.begin()+offset+templen[i], sendbuf.begin() + send_coldisp[i]);
                }
            }
        }
    }
    MPI_Barrier(commGrid->GetWorld());
    std::vector<VT> kthItem(n_thiscol);

    int root = commGrid->GetDiagOfProcCol();
    if(root==0 && colrank==0) // rank 0
    {
#ifdef THREADED
#pragma omp parallel for
#endif
        for(IT i=0; i<n_thiscol; i++)
        {
            IT nitems = send_coldisp[i+1]-send_coldisp[i];
            if(nitems >= k)
                kthItem[i] = sendbuf[send_coldisp[i]+k-1];
            else
                kthItem[i] = std::numeric_limits<VT>::min(); // return minimum possible value if a column is empty or has less than k elements
        }
    }
    else if(root>0 && colrank==0) // send to the diagonl processor of this processor column
    {
#ifdef THREADED
#pragma omp parallel for
#endif
        for(IT i=0; i<n_thiscol; i++)
        {
            IT nitems = send_coldisp[i+1]-send_coldisp[i];
            if(nitems >= k)
                kthItem[i] = sendbuf[send_coldisp[i]+k-1];
            else
                kthItem[i] = std::numeric_limits<VT>::min(); // return minimum possible value if a column is empty or has less than k elements
        }
        MPI_Send(kthItem.data(), n_thiscol, MPIType<VT>(), root, 0, commGrid->GetColWorld());
    }
    else if(root>0 && colrank==root)
    {
        MPI_Recv(kthItem.data(), n_thiscol, MPIType<VT>(), 0, 0, commGrid->GetColWorld(), MPI_STATUS_IGNORE);
    }
    
    std::vector <int> sendcnts;
    std::vector <int> dpls;
    if(colrank==root)
    {
        int proccols = commGrid->GetGridCols();
        IT n_perproc = n_thiscol / proccols;
        sendcnts.resize(proccols);
        std::fill(sendcnts.data(), sendcnts.data()+proccols-1, n_perproc);
        sendcnts[proccols-1] = n_thiscol - (n_perproc * (proccols-1));
        dpls.resize(proccols,0);	// displacements (zero initialized pid)
        std::partial_sum(sendcnts.data(), sendcnts.data()+proccols-1, dpls.data()+1);
    }
    
    int rowroot = commGrid->GetDiagOfProcRow();
    int recvcnts = 0;
    // scatter received data size
    MPI_Scatter(sendcnts.data(),1, MPI_INT, & recvcnts, 1, MPI_INT, rowroot, commGrid->GetRowWorld());
    
    rvec.arr.resize(recvcnts);
    MPI_Scatterv(kthItem.data(),sendcnts.data(), dpls.data(), MPIType<VT>(), rvec.arr.data(), rvec.arr.size(), MPIType<VT>(),rowroot, commGrid->GetRowWorld());
    rvec.glen = getncol();
    return true;
}



template <class IT, class NT, class DER>
template <typename VT, typename GIT, typename _UnaryOperation>	// GIT: global index type of vector
bool SpParMat<IT,NT,DER>::Kselect1(FullyDistSpVec<GIT,VT> & rvec, IT k, _UnaryOperation __unary_op) const
{
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

    if(*rvec.commGrid != *commGrid)
    {
        SpParHelper::Print("Grids are not comparable, SpParMat::Kselect() fails!", commGrid->GetWorld());
        MPI_Abort(MPI_COMM_WORLD,GRIDMISMATCH);
    }
    
    /*
     FullyDistVec<IT, IT> nnzPerColumn (getcommgrid());
     Reduce(nnzPerColumn, Column, plus<IT>(), (IT)0, [](NT val){return (IT)1;});
     IT maxnnzPerColumn = nnzPerColumn.Reduce(maximum<IT>(), (IT)0);
     if(k>maxnnzPerColumn)
     {
     SpParHelper::Print("Kselect: k is greater then maxNnzInColumn. Calling Reduce instead...\n");
     Reduce(rvec, Column, minimum<NT>(), static_cast<NT>(0));
     return false;
     }
     */
   
 

    IT n_thiscol = getlocalcols();   // length (number of columns) assigned to this processor (and processor column)
    MPI_Comm World = rvec.commGrid->GetWorld();
    MPI_Comm ColWorld = rvec.commGrid->GetColWorld();
    MPI_Comm RowWorld = rvec.commGrid->GetRowWorld();
    int colneighs = commGrid->GetGridRows();
    int colrank = commGrid->GetRankInProcCol();
    int coldiagrank = commGrid->GetDiagOfProcCol();

    //double memk = 3 * (double)n_thiscol*k*sizeof(VT)/1000000000;
    //double maxmemk =0.0; // nnz in a process column
    //MPI_Allreduce(&memk, &maxmemk , 1, MPIType<double>(), MPI_MAX, MPI_COMM_WORLD);

    //int myrank;
    //MPI_Comm_rank( MPI_COMM_WORLD, &myrank ) ;
    //if(myrank==0)
//	    std::cerr << "Actual kselect memory: " << maxmemk << "GB " << " columns " << n_thiscol << " activecol: " << nActiveCols << " \n";
  //  MPI_Barrier(MPI_COMM_WORLD);

    //replicate sparse indices along processor column
    int accnz;
    int32_t trxlocnz;
    GIT lenuntil;
    int32_t *trxinds, *activeCols;
    VT *trxnums, *numacc=NULL;
    TransposeVector(World, rvec, trxlocnz, lenuntil, trxinds, trxnums, true);
    
    if(rvec.commGrid->GetGridRows() > 1)
    {
        //TODO: we only need to communicate indices
        AllGatherVector(ColWorld, trxlocnz, lenuntil, trxinds, trxnums, activeCols, numacc, accnz, true);  // trxindS/trxnums deallocated, indacc/numacc allocated, accnz set
    }
    else
    {
        accnz = trxlocnz;
        activeCols = trxinds;     //aliasing ptr
        // since indexisvalue is set true in TransposeVector(), trxnums is never allocated
        //numacc = trxnums;     //aliasing ptr
    }

    std::vector<bool> isactive(n_thiscol,false);
    for(int i=0; i<accnz ; i++)
    {
        isactive[activeCols[i]] = true;
    }
    IT nActiveCols = accnz;//count_if(isactive.begin(), isactive.end(), [](bool ac){return ac;});

	
    int64_t lannz = getlocalnnz();
    int64_t nnzColWorld=0; // nnz in a process column
    MPI_Allreduce(&lannz, &nnzColWorld, 1, MPIType<int64_t>(), MPI_SUM, ColWorld);
    int64_t maxPerProcMemory = std::min(nnzColWorld, (int64_t)nActiveCols*k) * sizeof(VT);

    // hence we will not overflow for very large k
    std::vector<IT> send_coldisp(n_thiscol+1,0);
    std::vector<IT> local_coldisp(n_thiscol+1,0);
    //vector<VT> sendbuf(nActiveCols*k);
    //VT * sendbuf = static_cast<VT *> (::operator new (n_thiscol*k*sizeof(VT)));
    //VT * sendbuf = static_cast<VT *> (::operator new (nActiveCols*k*sizeof(VT)));
    VT * sendbuf = static_cast<VT *> (::operator new (maxPerProcMemory));

    //displacement of local columns
    //local_coldisp is the displacement of all nonzeros per column
    //send_coldisp is the displacement of k nonzeros per column
    IT nzc = 0;
    if(spSeq->getnnz()>0)
    {
        typename DER::SpColIter colit = spSeq->begcol();
        for(IT i=0; i<n_thiscol; ++i)
        {
            local_coldisp[i+1] = local_coldisp[i];
            send_coldisp[i+1] = send_coldisp[i];
            if( (colit != spSeq->endcol()) && (i==colit.colid()) )
            {
                if(isactive[i])
                {
                    local_coldisp[i+1] += colit.nnz();
                    if(colit.nnz()>=k)
                        send_coldisp[i+1] += k;
                    else
                        send_coldisp[i+1] += colit.nnz();
                }
                colit++;
                nzc++;
            }
        }
    }
    
    // a copy of local part of the matrix
    // this can be avoided if we write our own local kselect function instead of using partial_sort
    //vector<VT> localmat(local_coldisp[n_thiscol]);
    VT * localmat = static_cast<VT *> (::operator new (local_coldisp[n_thiscol]*sizeof(VT)));
    
    
#ifdef THREADED
#pragma omp parallel for
#endif
    for(IT i=0; i<nzc; i++)
        //for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)	// iterate over columns
    {
        typename DER::SpColIter colit = spSeq->begcol() + i;
        IT colid = colit.colid();
        if(isactive[colid])
        {
            IT idx = local_coldisp[colid];
            for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit < spSeq->endnz(colit); ++nzit)
            {
                localmat[idx++] = static_cast<VT>(__unary_op(nzit.value()));
            }
            
            if(colit.nnz()<=k)
            {
                sort(localmat+local_coldisp[colid], localmat+local_coldisp[colid+1], std::greater<VT>());
                std::copy(localmat+local_coldisp[colid], localmat+local_coldisp[colid+1], sendbuf+send_coldisp[colid]);
            }
            else
            {
                partial_sort(localmat+local_coldisp[colid], localmat+local_coldisp[colid]+k, localmat+local_coldisp[colid+1], std::greater<VT>());
                std::copy(localmat+local_coldisp[colid], localmat+local_coldisp[colid]+k, sendbuf+send_coldisp[colid]);
            }
        }
    }
    
    //vector<VT>().swap(localmat);
    ::operator delete(localmat);
    std::vector<IT>().swap(local_coldisp);
    
    //VT * recvbuf = static_cast<VT *> (::operator new (n_thiscol*k*sizeof(VT)));
    //VT * tempbuf = static_cast<VT *> (::operator new (n_thiscol*k*sizeof(VT)));

    //VT * recvbuf = static_cast<VT *> (::operator new ( nActiveCols*k*sizeof(VT)));
    //VT * tempbuf = static_cast<VT *> (::operator new ( nActiveCols*k*sizeof(VT)));
    

    VT * recvbuf = static_cast<VT *> (::operator new (maxPerProcMemory));
    VT * tempbuf = static_cast<VT *> (::operator new (maxPerProcMemory));
    //vector<VT> recvbuf(n_thiscol*k);
    //vector<VT> tempbuf(n_thiscol*k);
    std::vector<IT> recv_coldisp(n_thiscol+1);
    std::vector<IT> temp_coldisp(n_thiscol+1);
    //std::vector<IT> templen(n_thiscol);
    
   // Put a barrier and then print sth 
    
    for(int p=2; p <= colneighs; p*=2)
    {
        
        if(colrank%p == p/2) // this processor is a sender in this round
        {
            int receiver = colrank - ceil(p/2);
            MPI_Send(send_coldisp.data(), n_thiscol+1, MPIType<IT>(), receiver, 0, commGrid->GetColWorld());
            MPI_Send(sendbuf, send_coldisp[n_thiscol], MPIType<VT>(), receiver, 1, commGrid->GetColWorld());
            //break;
        }
        else if(colrank%p == 0) // this processor is a receiver in this round
        {
            int sender = colrank + ceil(p/2);
            if(sender < colneighs)
            {
                
                MPI_Recv(recv_coldisp.data(), n_thiscol+1, MPIType<IT>(), sender, 0, commGrid->GetColWorld(), MPI_STATUS_IGNORE);
                MPI_Recv(recvbuf, recv_coldisp[n_thiscol], MPIType<VT>(), sender, 1, commGrid->GetColWorld(), MPI_STATUS_IGNORE);
                
		temp_coldisp[0] = 0;
                for(IT i=0; i<n_thiscol; ++i)
                {
		    IT sendlen = send_coldisp[i+1] - send_coldisp[i];
		    IT recvlen = recv_coldisp[i+1] - recv_coldisp[i];
                    IT templen = std::min((sendlen+recvlen), k);
		    temp_coldisp[i+1] = temp_coldisp[i] + templen;
                }
                
                
#ifdef THREADED
#pragma omp parallel for
#endif
                for(IT i=0; i<n_thiscol; ++i)
                {
                    // partial merge until first k elements
                    IT j=send_coldisp[i], l=recv_coldisp[i];
                    //IT templen[i] = k*i;
                    //IT offset = k*i;
		    IT offset = temp_coldisp[i];
                    IT lid = 0;
                    for(; j<send_coldisp[i+1] && l<recv_coldisp[i+1] && lid<k;)
                    {
                        if(sendbuf[j] > recvbuf[l])  // decision
                            tempbuf[offset+lid++] = sendbuf[j++];
                        else
                            tempbuf[offset+lid++] = recvbuf[l++];
                    }
                    while(j<send_coldisp[i+1] && lid<k) tempbuf[offset+lid++] = sendbuf[j++];
                    while(l<recv_coldisp[i+1] && lid<k) tempbuf[offset+lid++] = recvbuf[l++];
                    //templen[i] = lid;
                }
                
		std::copy(temp_coldisp.begin(), temp_coldisp.end(), send_coldisp.begin());
		std::copy(tempbuf, tempbuf+temp_coldisp[n_thiscol], sendbuf);
		
		/*
                send_coldisp[0] = 0;
                for(IT i=0; i<n_thiscol; i++)
                {
                    send_coldisp[i+1] = send_coldisp[i] + templen[i];
		    assert(send_coldisp[i+1] == temp_coldisp[i+1]);
                }
                
                
#ifdef THREADED
#pragma omp parallel for
#endif
                for(IT i=0; i<n_thiscol; i++) // direct copy
                {
                    //IT offset = k*i;
		    IT offset = temp_coldisp[i];
                    std::copy(tempbuf+offset, tempbuf+offset+templen[i], sendbuf + send_coldisp[i]);
                }

		*/
            }
        }
    }
    MPI_Barrier(commGrid->GetWorld());
    // Print sth here as well
    
    /*--------------------------------------------------------
     At this point, top k elements in every active column
     are gathered on the first processor row, P(0,:).
     
     Next step: At P(0,i) find the kth largest element in
     active columns belonging to P(0,i).
     If nnz in a column is less than k, keep the largest nonzero.
     If a column is empty, keep the lowest numeric value.
     --------------------------------------------------------*/
    
    std::vector<VT> kthItem(nActiveCols); // kth elements of local active columns
    if(colrank==0)
    {
#ifdef THREADED
#pragma omp parallel for
#endif
        for(IT i=0; i<nActiveCols; i++)
        {
            IT ai = activeCols[i]; // active column index
            IT nitems = send_coldisp[ai+1]-send_coldisp[ai];
            if(nitems >= k)
                kthItem[i] = sendbuf[send_coldisp[ai]+k-1];
            else if (nitems==0)
                kthItem[i] = std::numeric_limits<VT>::min(); // return minimum possible value if a column is empty
            else
                kthItem[i] = sendbuf[send_coldisp[ai+1]-1]; // returning the last entry if nnz in this column is less than k
            
        }
    }
    
    /*--------------------------------------------------------
     At this point, kth largest elements in every active column
     are gathered on the first processor row, P(0,:).
     
     Next step: Send the kth largest elements from P(0,i) to P(i,i)
     Nothing to do for P(0,0)
     --------------------------------------------------------*/
    if(coldiagrank>0 && colrank==0)
    {
        MPI_Send(kthItem.data(), nActiveCols, MPIType<VT>(), coldiagrank, 0, commGrid->GetColWorld());
    }
    else if(coldiagrank>0 && colrank==coldiagrank) // receive in the diagonal processor
    {
        MPI_Recv(kthItem.data(), nActiveCols, MPIType<VT>(), 0, 0, commGrid->GetColWorld(), MPI_STATUS_IGNORE);
    }
    
    // Put a barrier and print sth

    /*--------------------------------------------------------
     At this point, kth largest elements in every active column
     are gathered on the diagonal processors P(i,i).
     
     Next step: Scatter the kth largest elements from P(i,i)
     to all processors in the ith row, P(i,:).
     Each processor recevies exactly local nnz of rvec entries
     so that the received data can be directly put in rvec.
     --------------------------------------------------------*/
    int rowroot = commGrid->GetDiagOfProcRow();
    int proccols = commGrid->GetGridCols();
    std::vector <int> sendcnts(proccols,0);
    std::vector <int> dpls(proccols,0);
    int lsize = rvec.ind.size();
    // local sizes of the input vecotor will be sent from the doagonal processor
    MPI_Gather(&lsize,1, MPI_INT, sendcnts.data(), 1, MPI_INT, rowroot, RowWorld);
    std::partial_sum(sendcnts.data(), sendcnts.data()+proccols-1, dpls.data()+1);
    MPI_Scatterv(kthItem.data(),sendcnts.data(), dpls.data(), MPIType<VT>(), rvec.num.data(), rvec.num.size(), MPIType<VT>(),rowroot, RowWorld);

    delete [] activeCols;
    delete [] numacc;
    
    ::operator delete(sendbuf);
    ::operator delete(recvbuf);
    ::operator delete(tempbuf);
    //delete [] activeCols;
    //delete [] numacc;
    
    return true;
}

// only defined for symmetric matrix
template <class IT, class NT, class DER>
IT SpParMat<IT,NT,DER>::Bandwidth() const
{
    IT upperlBW = -1;
    IT lowerlBW = -1;
    IT m_perproc = getnrow() / commGrid->GetGridRows();
    IT n_perproc = getncol() / commGrid->GetGridCols();
    IT moffset = commGrid->GetRankInProcCol() * m_perproc;
    IT noffset = commGrid->GetRankInProcRow() * n_perproc;
    
    for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)	// iterate over columns
    {
        IT diagrow = colit.colid() + noffset;
        typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit);
        if(nzit != spSeq->endnz(colit)) // nonempty column
        {
            IT firstrow = nzit.rowid() + moffset;
            IT lastrow = (nzit+ colit.nnz()-1).rowid() + moffset;
           
            if(firstrow <= diagrow) // upper diagonal
            {
                IT dev = diagrow - firstrow;
                if(upperlBW < dev) upperlBW = dev;
            }
            if(lastrow >= diagrow) // lower diagonal
            {
                IT dev = lastrow - diagrow;
                if(lowerlBW < dev) lowerlBW = dev;
            }
        }
    }
    IT upperBW;
    //IT lowerBW;
    MPI_Allreduce( &upperlBW, &upperBW, 1, MPIType<IT>(), MPI_MAX, commGrid->GetWorld());
    //MPI_Allreduce( &lowerlBW, &lowerBW, 1, MPIType<IT>(), MPI_MAX, commGrid->GetWorld());
    
    //return (upperBW + lowerBW + 1);
    return (upperBW);
}



// only defined for symmetric matrix
template <class IT, class NT, class DER>
IT SpParMat<IT,NT,DER>::Profile() const
{
    int colrank = commGrid->GetRankInProcRow();
    IT cols = getncol();
    IT rows = getnrow();
    IT m_perproc = cols / commGrid->GetGridRows();
    IT n_perproc = rows / commGrid->GetGridCols();
    IT moffset = commGrid->GetRankInProcCol() * m_perproc;
    IT noffset = colrank * n_perproc;
  

    int pc = commGrid->GetGridCols();
    IT n_thisproc;
    if(colrank!=pc-1 ) n_thisproc = n_perproc;
    else n_thisproc =  cols - (pc-1)*n_perproc;
 
    
    std::vector<IT> firstRowInCol(n_thisproc,getnrow());
    std::vector<IT> lastRowInCol(n_thisproc,-1);
    
    for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)	// iterate over columns
    {
        IT diagrow = colit.colid() + noffset;
        typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit);
        if(nzit != spSeq->endnz(colit)) // nonempty column
        {
            IT firstrow = nzit.rowid() + moffset;
            IT lastrow = (nzit+ colit.nnz()-1).rowid() + moffset;
            if(firstrow <= diagrow) // upper diagonal
            {
                firstRowInCol[colit.colid()] = firstrow;
            }
            if(lastrow >= diagrow) // lower diagonal
            {
                lastRowInCol[colit.colid()] = lastrow;
            }
        }
    }
    
    std::vector<IT> firstRowInCol_global(n_thisproc,getnrow());
    //vector<IT> lastRowInCol_global(n_thisproc,-1);
    MPI_Allreduce( firstRowInCol.data(), firstRowInCol_global.data(), n_thisproc, MPIType<IT>(), MPI_MIN, commGrid->colWorld);
    //MPI_Allreduce( lastRowInCol.data(), lastRowInCol_global.data(), n_thisproc, MPIType<IT>(), MPI_MAX, commGrid->GetColWorld());
    
    IT profile = 0;
    for(IT i=0; i<n_thisproc; i++)
    {
        if(firstRowInCol_global[i]==rows) // empty column
            profile++;
        else
            profile += (i + noffset - firstRowInCol_global[i]);
    }
    
    IT profile_global = 0;
    MPI_Allreduce( &profile, &profile_global, 1, MPIType<IT>(), MPI_SUM, commGrid->rowWorld);
    
    return (profile_global);
}



template <class IT, class NT, class DER>
template <typename VT, typename GIT, typename _BinaryOperation>
void SpParMat<IT,NT,DER>::MaskedReduce(FullyDistVec<GIT,VT> & rvec, FullyDistSpVec<GIT,VT> & mask, Dim dim, _BinaryOperation __binary_op, VT id, bool exclude) const
{
    if (dim!=Column)
    {
        SpParHelper::Print("SpParMat::MaskedReduce() is only implemented for Colum\n");
        MPI_Abort(MPI_COMM_WORLD,GRIDMISMATCH);
    }
    MaskedReduce(rvec, mask, dim, __binary_op, id, myidentity<NT>(), exclude);
}

/**
 * Reduce along the column into a vector
 * @param[in] mask {A sparse vector indicating row indices included/excluded (based on exclude argument) in the reduction }
 * @param[in] __binary_op {the operation used for reduction; examples: max, min, plus, multiply, and, or. Its parameters and return type are all VT}
 * @param[in] id {scalar that is used as the identity for __binary_op; examples: zero, infinity}
 * @param[in] __unary_op {optional unary operation applied to nonzeros *before* the __binary_op; examples: 1/x, x^2}
 * @param[in] exclude {if true, masked row indices are included in the reduction}
 * @param[out] rvec {the return vector, specified as an output parameter to allow arbitrary return types via VT}
 **/
template <class IT, class NT, class DER>
template <typename VT, typename GIT, typename _BinaryOperation, typename _UnaryOperation>	// GIT: global index type of vector
void SpParMat<IT,NT,DER>::MaskedReduce(FullyDistVec<GIT,VT> & rvec, FullyDistSpVec<GIT,VT> & mask, Dim dim, _BinaryOperation __binary_op, VT id, _UnaryOperation __unary_op, bool exclude) const
{
    MPI_Comm World = commGrid->GetWorld();
    MPI_Comm ColWorld = commGrid->GetColWorld();
    MPI_Comm RowWorld = commGrid->GetRowWorld();

    if (dim!=Column)
    {
        SpParHelper::Print("SpParMat::MaskedReduce() is only implemented for Colum\n");
        MPI_Abort(MPI_COMM_WORLD,GRIDMISMATCH);
    }
    if(*rvec.commGrid != *commGrid)
    {
        SpParHelper::Print("Grids are not comparable, SpParMat::MaskedReduce() fails!", commGrid->GetWorld());
        MPI_Abort(MPI_COMM_WORLD,GRIDMISMATCH);
    }
    
    int rowneighs = commGrid->GetGridCols();
    int rowrank = commGrid->GetRankInProcRow();
    std::vector<int> rownz(rowneighs);
    int locnnzMask = static_cast<int> (mask.getlocnnz());
    rownz[rowrank] = locnnzMask;
    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, rownz.data(), 1, MPI_INT, RowWorld);
    std::vector<int> dpls(rowneighs+1,0);
    std::partial_sum(rownz.begin(), rownz.end(), dpls.data()+1);
    int accnz = std::accumulate(rownz.begin(), rownz.end(), 0);
    std::vector<GIT> sendInd(locnnzMask);
    auto rowlenuntil = mask.RowLenUntil();
	std::transform(mask.ind.begin(), mask.ind.end(), sendInd.begin(),
			   [rowlenuntil](const GIT& val) { return val + rowlenuntil; });
    
    std::vector<GIT> indMask(accnz);
    MPI_Allgatherv(sendInd.data(), rownz[rowrank], MPIType<GIT>(), indMask.data(), rownz.data(), dpls.data(), MPIType<GIT>(), RowWorld);
    
    
    // We can't use rvec's distribution (rows first, columns later) here
    IT n_thiscol = getlocalcols();   // length assigned to this processor column
    int colneighs = commGrid->GetGridRows();	// including oneself
    int colrank = commGrid->GetRankInProcCol();
    
    GIT * loclens = new GIT[colneighs];
    GIT * lensums = new GIT[colneighs+1]();	// begin/end points of local lengths
    
    GIT n_perproc = n_thiscol / colneighs;    // length on a typical processor
    if(colrank == colneighs-1)
        loclens[colrank] = n_thiscol - (n_perproc*colrank);
    else
        loclens[colrank] = n_perproc;
    
    MPI_Allgather(MPI_IN_PLACE, 0, MPIType<GIT>(), loclens, 1, MPIType<GIT>(), commGrid->GetColWorld());
    std::partial_sum(loclens, loclens+colneighs, lensums+1);	// loclens and lensums are different, but both would fit in 32-bits
    
    std::vector<VT> trarr;
    typename DER::SpColIter colit = spSeq->begcol();
    for(int i=0; i< colneighs; ++i)
    {
        VT * sendbuf = new VT[loclens[i]];
        std::fill(sendbuf, sendbuf+loclens[i], id);	// fill with identity
        
        for(; colit != spSeq->endcol() && colit.colid() < lensums[i+1]; ++colit)	// iterate over a portion of columns
        {
            int k=0;
            typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit);
            
            for(; nzit != spSeq->endnz(colit) && k < indMask.size(); )	// all nonzeros in this column
            {
                if(nzit.rowid() < indMask[k])
                {
                    if(exclude)
                    {
                        sendbuf[colit.colid()-lensums[i]] = __binary_op(static_cast<VT>(__unary_op(nzit.value())), sendbuf[colit.colid()-lensums[i]]);
                    }
                    ++nzit;
                }
                else if(nzit.rowid() > indMask[k]) ++k;
                else
                {
                    if(!exclude)
                    {
                        sendbuf[colit.colid()-lensums[i]] = __binary_op(static_cast<VT>(__unary_op(nzit.value())), sendbuf[colit.colid()-lensums[i]]);
                    }
                    ++k;
                    ++nzit;
                }
                
            }
            if(exclude)
            {
                while(nzit != spSeq->endnz(colit))
                {
                    sendbuf[colit.colid()-lensums[i]] = __binary_op(static_cast<VT>(__unary_op(nzit.value())), sendbuf[colit.colid()-lensums[i]]);
                    ++nzit;
                }
            }
        }
        
        VT * recvbuf = NULL;
        if(colrank == i)
        {
            trarr.resize(loclens[i]);
            recvbuf = SpHelper::p2a(trarr);
        }
        MPI_Reduce(sendbuf, recvbuf, loclens[i], MPIType<VT>(), MPIOp<_BinaryOperation, VT>::op(), i, commGrid->GetColWorld()); // root  = i
        delete [] sendbuf;
    }
    DeleteAll(loclens, lensums);
    
    GIT reallen;	// Now we have to transpose the vector
    GIT trlen = trarr.size();
    int diagneigh = commGrid->GetComplementRank();
    MPI_Status status;
    MPI_Sendrecv(&trlen, 1, MPIType<IT>(), diagneigh, TRNNZ, &reallen, 1, MPIType<IT>(), diagneigh, TRNNZ, commGrid->GetWorld(), &status);
    
    rvec.arr.resize(reallen);
    MPI_Sendrecv(SpHelper::p2a(trarr), trlen, MPIType<VT>(), diagneigh, TRX, SpHelper::p2a(rvec.arr), reallen, MPIType<VT>(), diagneigh, TRX, commGrid->GetWorld(), &status);
    rvec.glen = getncol();	// ABAB: Put a sanity check here
    
}




template <class IT, class NT, class DER>
template <typename NNT,typename NDER>
SpParMat<IT,NT,DER>::operator SpParMat<IT,NNT,NDER> () const
{
	NDER * convert = new NDER(*spSeq);
	return SpParMat<IT,NNT,NDER> (convert, commGrid);
}

//! Change index type as well
template <class IT, class NT, class DER>
template <typename NIT, typename NNT,typename NDER>
SpParMat<IT,NT,DER>::operator SpParMat<NIT,NNT,NDER> () const
{
	NDER * convert = new NDER(*spSeq);
	return SpParMat<NIT,NNT,NDER> (convert, commGrid);
}

/**
 * Create a submatrix of size m x (size(ci) * s) on a r x s processor grid
 * Essentially fetches the columns ci[0], ci[1],... ci[size(ci)] from every submatrix
 */
template <class IT, class NT, class DER>
SpParMat<IT,NT,DER> SpParMat<IT,NT,DER>::SubsRefCol (const std::vector<IT> & ci) const
{
	std::vector<IT> ri;
	DER * tempseq = new DER((*spSeq)(ri, ci)); 
	return SpParMat<IT,NT,DER> (tempseq, commGrid);	
} 

/** 
 * Generalized sparse matrix indexing (ri/ci are 0-based indexed)
 * Both the storage and the actual values in FullyDistVec should be IT
 * The index vectors are dense and FULLY distributed on all processors
 * We can use this function to apply a permutation like A(p,q) 
 * Sequential indexing subroutine (via multiplication) is general enough.
 */
template <class IT, class NT, class DER>
template <typename PTNTBOOL, typename PTBOOLNT>
SpParMat<IT,NT,DER> SpParMat<IT,NT,DER>::SubsRef_SR (const FullyDistVec<IT,IT> & ri, const FullyDistVec<IT,IT> & ci, bool inplace)
{
	typedef typename DER::LocalIT LIT;

	// infer the concrete type SpMat<LIT,LIT>
	typedef typename create_trait<DER, LIT, bool>::T_inferred DER_IT;

	if((*(ri.commGrid) != *(commGrid)) || (*(ci.commGrid) != *(commGrid)))
	{
		SpParHelper::Print("Grids are not comparable, SpRef fails !"); 
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
	}

	// Safety check
	IT locmax_ri = 0;
	IT locmax_ci = 0;
	if(!ri.arr.empty())
		locmax_ri = *std::max_element(ri.arr.begin(), ri.arr.end());
	if(!ci.arr.empty())
		locmax_ci = *std::max_element(ci.arr.begin(), ci.arr.end());

	IT totalm = getnrow();
	IT totaln = getncol();
	if(locmax_ri > totalm || locmax_ci > totaln)	
	{
		throw outofrangeexception();
	}

	// The indices for FullyDistVec are offset'd to 1/p pieces
	// The matrix indices are offset'd to 1/sqrt(p) pieces
	// Add the corresponding offset before sending the data 
	IT roffset = ri.RowLenUntil();
	IT rrowlen = ri.MyRowLength();
	IT coffset = ci.RowLenUntil();
	IT crowlen = ci.MyRowLength();

	// We create two boolean matrices P and Q
	// Dimensions:  P is size(ri) x m
	//		Q is n x size(ci) 
	// Range(ri) = {0,...,m-1}
	// Range(ci) = {0,...,n-1}

	IT rowneighs = commGrid->GetGridCols();	// number of neighbors along this processor row (including oneself)
	IT m_perproccol = totalm / rowneighs;
	IT n_perproccol = totaln / rowneighs;

	// Get the right local dimensions
	IT diagneigh = commGrid->GetComplementRank();
	IT mylocalrows = getlocalrows();
	IT mylocalcols = getlocalcols();
	IT trlocalrows;
	MPI_Status status;
	MPI_Sendrecv(&mylocalrows, 1, MPIType<IT>(), diagneigh, TRROWX, &trlocalrows, 1, MPIType<IT>(), diagneigh, TRROWX, commGrid->GetWorld(), &status);
	// we don't need trlocalcols because Q.Transpose() will take care of it

	std::vector< std::vector<IT> > rowid(rowneighs);	// reuse for P and Q 
	std::vector< std::vector<IT> > colid(rowneighs);

	// Step 1: Create P
	IT locvec = ri.arr.size();	// nnz in local vector
	for(typename std::vector<IT>::size_type i=0; i< (unsigned)locvec; ++i)
	{
		// numerical values (permutation indices) are 0-based
		// recipient alone progessor row
		IT rowrec = (m_perproccol!=0) ? std::min(ri.arr[i] / m_perproccol, rowneighs-1) : (rowneighs-1);	

		// ri's numerical values give the colids and its local indices give rowids
		rowid[rowrec].push_back( i + roffset);	
		colid[rowrec].push_back(ri.arr[i] - (rowrec * m_perproccol));
	}

	int * sendcnt = new int[rowneighs];	// reuse in Q as well
	int * recvcnt = new int[rowneighs];
	for(IT i=0; i<rowneighs; ++i)
		sendcnt[i] = rowid[i].size();

	MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, commGrid->GetRowWorld()); // share the counts
	int * sdispls = new int[rowneighs]();
	int * rdispls = new int[rowneighs]();
	std::partial_sum(sendcnt, sendcnt+rowneighs-1, sdispls+1);
	std::partial_sum(recvcnt, recvcnt+rowneighs-1, rdispls+1);
	IT p_nnz = std::accumulate(recvcnt,recvcnt+rowneighs, static_cast<IT>(0));	

	// create space for incoming data ... 
	IT * p_rows = new IT[p_nnz];
	IT * p_cols = new IT[p_nnz];
  	IT * senddata = new IT[locvec];	// re-used for both rows and columns
	for(int i=0; i<rowneighs; ++i)
	{
		std::copy(rowid[i].begin(), rowid[i].end(), senddata+sdispls[i]);
		std::vector<IT>().swap(rowid[i]);	// clear memory of rowid
	}
	MPI_Alltoallv(senddata, sendcnt, sdispls, MPIType<IT>(), p_rows, recvcnt, rdispls, MPIType<IT>(), commGrid->GetRowWorld());

	for(int i=0; i<rowneighs; ++i)
	{
		std::copy(colid[i].begin(), colid[i].end(), senddata+sdispls[i]);
		std::vector<IT>().swap(colid[i]);	// clear memory of colid
	}
	MPI_Alltoallv(senddata, sendcnt, sdispls, MPIType<IT>(), p_cols, recvcnt, rdispls, MPIType<IT>(), commGrid->GetRowWorld());
	delete [] senddata;

	std::tuple<LIT,LIT,bool> * p_tuples = new std::tuple<LIT,LIT,bool>[p_nnz]; 
	for(IT i=0; i< p_nnz; ++i)
	{
		p_tuples[i] = std::make_tuple(p_rows[i], p_cols[i], 1);	// here we can convert to local indices
	}
	DeleteAll(p_rows, p_cols);

	DER_IT * PSeq = new DER_IT(); 
	PSeq->Create( p_nnz, rrowlen, trlocalrows, p_tuples);		// deletion of tuples[] is handled by SpMat::Create

	SpParMat<IT,NT,DER> PA(commGrid);
	if(&ri == &ci)	// Symmetric permutation
	{
		DeleteAll(sendcnt, recvcnt, sdispls, rdispls);
		#ifdef SPREFDEBUG
		SpParHelper::Print("Symmetric permutation\n", commGrid->GetWorld());
		#endif
		SpParMat<IT,bool,DER_IT> P (PSeq, commGrid);
		if(inplace) 
		{
			#ifdef SPREFDEBUG	
			SpParHelper::Print("In place multiplication\n", commGrid->GetWorld());
			#endif
                *this = Mult_AnXBn_DoubleBuff<PTBOOLNT, NT, DER>(P, *this, false, true);	// clear the memory of *this
                //*this = Mult_AnXBn_Synch<PTBOOLNT, NT, DER>(P, *this, false, true);	// clear the memory of *this

			//ostringstream outb;
			//outb << "P_after_" << commGrid->myrank;
			//ofstream ofb(outb.str().c_str());
			//P.put(ofb);

			P.Transpose();	
                    *this = Mult_AnXBn_DoubleBuff<PTNTBOOL, NT, DER>(*this, P, true, true);	// clear the memory of both *this and P
                    //*this = Mult_AnXBn_Synch<PTNTBOOL, NT, DER>(*this, P, true, true);	// clear the memory of both *this and P
			return SpParMat<IT,NT,DER>(commGrid);	// dummy return to match signature
		}
		else
		{
            PA = Mult_AnXBn_DoubleBuff<PTBOOLNT, NT, DER>(P,*this);
			//PA = Mult_AnXBn_Synch<PTBOOLNT, NT, DER>(P,*this);
			P.Transpose();
            return Mult_AnXBn_DoubleBuff<PTNTBOOL, NT, DER>(PA, P);
			//return Mult_AnXBn_Synch<PTNTBOOL, NT, DER>(PA, P);
		}
	}
	else
	{
		// Intermediate step (to save memory): Form PA and store it in P
		// Distributed matrix generation (collective call)
		SpParMat<IT,bool,DER_IT> P (PSeq, commGrid);

		// Do parallel matrix-matrix multiply
            PA = Mult_AnXBn_DoubleBuff<PTBOOLNT, NT, DER>(P, *this);
            //PA = Mult_AnXBn_Synch<PTBOOLNT, NT, DER>(P, *this);
	}	// P is destructed here
#ifndef NDEBUG
	PA.PrintInfo();
#endif
	// Step 2: Create Q  (use the same row-wise communication and transpose at the end)
	// This temporary to-be-transposed Q is size(ci) x n 
	locvec = ci.arr.size();	// nnz in local vector (reset variable)
	for(typename std::vector<IT>::size_type i=0; i< (unsigned)locvec; ++i)
	{
		// numerical values (permutation indices) are 0-based
		IT rowrec = (n_perproccol!=0) ? std::min(ci.arr[i] / n_perproccol, rowneighs-1) : (rowneighs-1);	

		// ri's numerical values give the colids and its local indices give rowids
		rowid[rowrec].push_back( i + coffset);	
		colid[rowrec].push_back(ci.arr[i] - (rowrec * n_perproccol));
	}

	for(IT i=0; i<rowneighs; ++i)
		sendcnt[i] = rowid[i].size();	// update with new sizes

	MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, commGrid->GetRowWorld()); // share the counts
	std::fill(sdispls, sdispls+rowneighs, 0);	// reset
	std::fill(rdispls, rdispls+rowneighs, 0);
	std::partial_sum(sendcnt, sendcnt+rowneighs-1, sdispls+1);
	std::partial_sum(recvcnt, recvcnt+rowneighs-1, rdispls+1);
	IT q_nnz = std::accumulate(recvcnt,recvcnt+rowneighs, static_cast<IT>(0));	

	// create space for incoming data ... 
	IT * q_rows = new IT[q_nnz];
	IT * q_cols = new IT[q_nnz];
  	senddata = new IT[locvec];	
	for(int i=0; i<rowneighs; ++i)
	{
		std::copy(rowid[i].begin(), rowid[i].end(), senddata+sdispls[i]);
		std::vector<IT>().swap(rowid[i]);	// clear memory of rowid
	}
	MPI_Alltoallv(senddata, sendcnt, sdispls, MPIType<IT>(), q_rows, recvcnt, rdispls, MPIType<IT>(), commGrid->GetRowWorld());

	for(int i=0; i<rowneighs; ++i)
	{
		std::copy(colid[i].begin(), colid[i].end(), senddata+sdispls[i]);
		std::vector<IT>().swap(colid[i]);	// clear memory of colid
	}
	MPI_Alltoallv(senddata, sendcnt, sdispls, MPIType<IT>(), q_cols, recvcnt, rdispls, MPIType<IT>(), commGrid->GetRowWorld());
	DeleteAll(senddata, sendcnt, recvcnt, sdispls, rdispls);

	std::tuple<LIT,LIT,bool> * q_tuples = new std::tuple<LIT,LIT,bool>[q_nnz]; 	// here we can convert to local indices (2018 note by Aydin)
	for(IT i=0; i< q_nnz; ++i)
	{
		q_tuples[i] = std::make_tuple(q_rows[i], q_cols[i], 1);
	}
	DeleteAll(q_rows, q_cols);
	DER_IT * QSeq = new DER_IT(); 
	QSeq->Create( q_nnz, crowlen, mylocalcols, q_tuples);		// Creating Q' instead

	// Step 3: Form PAQ
	// Distributed matrix generation (collective call)
	SpParMat<IT,bool,DER_IT> Q (QSeq, commGrid);
	Q.Transpose();	
	if(inplace)
	{
               //*this = Mult_AnXBn_DoubleBuff<PTNTBOOL, NT, DER>(PA, Q, true, true);	// clear the memory of both PA and P
       		*this = Mult_AnXBn_Synch<PTNTBOOL, NT, DER>(PA, Q, true, true);	// clear the memory of both PA and P
		return SpParMat<IT,NT,DER>(commGrid);	// dummy return to match signature
	}
	else
	{
            //return Mult_AnXBn_DoubleBuff<PTNTBOOL, NT, DER>(PA, Q);
            return Mult_AnXBn_Synch<PTNTBOOL, NT, DER>(PA, Q);
	}
}



template<class IT,
		 class NT,
		 class DER>
template<typename PTNTBOOL,
		 typename PTBOOLNT>
SpParMat<IT, NT, DER>
SpParMat<IT, NT, DER>::SubsRef_SR (
	const FullyDistVec<IT, IT> &v,
	Dim dim,
	bool inplace
	)
{
	typedef typename DER::LocalIT LIT;
	typedef typename create_trait<DER, LIT, bool>::T_inferred DER_IT;

	if (*(v.commGrid) != *commGrid)
	{
		SpParHelper::Print("Grids are not comparable, SpRef fails!");
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
	}

	IT locmax = 0;
	if (!v.arr.empty())
		locmax = *std::max_element(v.arr.begin(), v.arr.end());

	IT	offset	   = v.RowLenUntil();
	IT	rowlen	   = v.MyRowLength();
	IT	totalm	   = getnrow();
	IT	totaln	   = getncol();
	IT	rowneighs  = commGrid->GetGridCols();
	IT	perproccol = -1;
	IT	dimy	   = -1;
	IT	diagneigh, tmp;

	switch(dim)
	{
	case Row:
		if (locmax > totalm)
			throw outofrangeexception();

		perproccol = totalm / rowneighs;

		diagneigh = commGrid->GetComplementRank();
		MPI_Status status;
		tmp = getlocalrows();
		MPI_Sendrecv(&tmp, 1, MPIType<IT>(), diagneigh, TRROWX,
					 &dimy, 1, MPIType<IT>(), diagneigh, TRROWX,
					 commGrid->GetWorld(), &status);

		break;
		

	case Column:
		if (locmax > totaln)
			throw outofrangeexception();

		perproccol = totaln / rowneighs;
		dimy	   = getlocalcols();

		break;


	default:
		break;
	}


	// find owner processes and fill in the vectors
	std::vector<std::vector<IT>> rowid(rowneighs);
	std::vector<std::vector<IT>> colid(rowneighs);
	IT locvec = v.arr.size();
	for(typename std::vector<IT>::size_type i = 0; i < (unsigned)locvec; ++i)
	{
		IT rowrec = (perproccol != 0)
			? std::min(v.arr[i] / perproccol, rowneighs - 1)
			: (rowneighs - 1);

		rowid[rowrec].push_back(i + offset);	
		colid[rowrec].push_back(v.arr[i] - (rowrec * perproccol));
	}

	
	// exchange data
	int *sendcnt = new int[rowneighs];
	int *recvcnt = new int[rowneighs];
	for (IT i = 0; i < rowneighs; ++i)
		sendcnt[i] = rowid[i].size();

	MPI_Alltoall(sendcnt, 1, MPI_INT,
				 recvcnt, 1, MPI_INT,
				 commGrid->GetRowWorld());

	int *sdispls = new int[rowneighs]();
	int *rdispls = new int[rowneighs]();
	std::partial_sum(sendcnt, sendcnt+rowneighs-1, sdispls+1);
	std::partial_sum(recvcnt, recvcnt+rowneighs-1, rdispls+1);
	IT v_nnz = std::accumulate(recvcnt, recvcnt+rowneighs, static_cast<IT>(0));

	IT	*v_rows	  = new IT[v_nnz];
	IT	*v_cols	  = new IT[v_nnz];
  	IT	*senddata = new IT[locvec];

	for(int i = 0; i < rowneighs; ++i)
	{
		std::copy(rowid[i].begin(), rowid[i].end(), senddata + sdispls[i]);
		std::vector<IT>().swap(rowid[i]);	// free memory
	}

	MPI_Alltoallv(senddata, sendcnt, sdispls, MPIType<IT>(),
				  v_rows, recvcnt, rdispls, MPIType<IT>(),
				  commGrid->GetRowWorld());

	for(int i = 0; i < rowneighs; ++i)
	{
		std::copy(colid[i].begin(), colid[i].end(), senddata + sdispls[i]);
		std::vector<IT>().swap(colid[i]);	// free memory
	}

	MPI_Alltoallv(senddata, sendcnt, sdispls, MPIType<IT>(),
				  v_cols, recvcnt, rdispls, MPIType<IT>(),
				  commGrid->GetRowWorld());

	delete [] senddata;


	// form tuples and create the permutation matrix
	std::tuple<LIT, LIT, bool> *v_tuples =
		new std::tuple<LIT, LIT, bool>[v_nnz];
	for(IT i = 0; i < v_nnz; ++i)
		v_tuples[i] = std::make_tuple(v_rows[i], v_cols[i], 1);

	DeleteAll(v_rows, v_cols);

	DER_IT *vseq = new DER_IT();
	vseq->Create(v_nnz, rowlen, dimy, v_tuples);
	SpParMat<IT, bool, DER_IT> V(vseq, commGrid); // permutation matrix


	// generate the final matrix
	switch(dim)
	{
	case Row:
		if (inplace)
		{
			*this = Mult_AnXBn_DoubleBuff<PTBOOLNT, NT, DER>(V, *this, true, true);
			return SpParMat<IT, NT, DER>(commGrid); // dummy
		}
		else
			return Mult_AnXBn_DoubleBuff<PTBOOLNT, NT, DER>(V, *this);

		break;


	case Column:
		V.Transpose();
		if (inplace)
		{
			*this = Mult_AnXBn_DoubleBuff<PTNTBOOL, NT, DER>(*this, V, true, true);
			return SpParMat<IT, NT, DER>(commGrid); // dummy
		}
		else
			return Mult_AnXBn_DoubleBuff<PTNTBOOL, NT, DER>(*this, V);

		
	default:
		break;
	}


	// should not reach at this point
	return SpParMat<IT, NT, DER>(commGrid); // dummy
}
								   


template <class IT, class NT, class DER>
void SpParMat<IT,NT,DER>::SpAsgn(const FullyDistVec<IT,IT> & ri, const FullyDistVec<IT,IT> & ci, SpParMat<IT,NT,DER> & B)
{
	typedef PlusTimesSRing<NT, NT> PTRing;
	
	if((*(ri.commGrid) != *(B.commGrid)) || (*(ci.commGrid) != *(B.commGrid)))
	{
		SpParHelper::Print("Grids are not comparable, SpAsgn fails !", commGrid->GetWorld());
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
	}
	IT total_m_A = getnrow();
	IT total_n_A = getncol();
	IT total_m_B = B.getnrow();
	IT total_n_B = B.getncol();
	
	if(total_m_B != ri.TotalLength())
	{
		SpParHelper::Print("First dimension of B does NOT match the length of ri, SpAsgn fails !", commGrid->GetWorld());
		MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
	}
	if(total_n_B != ci.TotalLength())
	{
		SpParHelper::Print("Second dimension of B does NOT match the length of ci, SpAsgn fails !", commGrid->GetWorld());
		MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
	}
	Prune(ri, ci);	// make a hole	
	
	// embed B to the size of A
	FullyDistVec<IT,IT> * rvec = new FullyDistVec<IT,IT>(ri.commGrid);
	rvec->iota(total_m_B, 0);	// sparse() expects a zero based index
	
	SpParMat<IT,NT,DER> R(total_m_A, total_m_B, ri, *rvec, 1);
	delete rvec;	// free memory
	SpParMat<IT,NT,DER> RB = Mult_AnXBn_DoubleBuff<PTRing, NT, DER>(R, B, true, false); // clear memory of R but not B
	
	FullyDistVec<IT,IT> * qvec = new FullyDistVec<IT,IT>(ri.commGrid);
	qvec->iota(total_n_B, 0);
	SpParMat<IT,NT,DER> Q(total_n_B, total_n_A, *qvec, ci, 1);
	delete qvec;	// free memory
	SpParMat<IT,NT,DER> RBQ = Mult_AnXBn_DoubleBuff<PTRing, NT, DER>(RB, Q, true, true); // clear memory of RB and Q
	*this += RBQ;	// extend-add
}

// this only prunes the submatrix A[ri,ci] in matlab notation
// or if the input is an adjacency matrix and ri=ci, it removes the connections
// between the subgraph induced by ri
// if you need to remove all contents of rows in ri and columns in ci:
// then call PruneFull
template <class IT, class NT, class DER>
void SpParMat<IT,NT,DER>::Prune(const FullyDistVec<IT,IT> & ri, const FullyDistVec<IT,IT> & ci)
{
	if((*(ri.commGrid) != *(commGrid)) || (*(ci.commGrid) != *(commGrid)))
	{
		SpParHelper::Print("Grids are not comparable, Prune fails!\n", commGrid->GetWorld());
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
	}

	// Safety check
	IT locmax_ri = 0;
	IT locmax_ci = 0;
	if(!ri.arr.empty())
		locmax_ri = *std::max_element(ri.arr.begin(), ri.arr.end());
	if(!ci.arr.empty())
		locmax_ci = *std::max_element(ci.arr.begin(), ci.arr.end());

	IT total_m = getnrow();
	IT total_n = getncol();
	if(locmax_ri > total_m || locmax_ci > total_n)	
	{
		throw outofrangeexception();
	}

        // infer the concrete types to replace the value with bool
	typedef typename DER::LocalIT LIT;
        typedef typename create_trait<DER, LIT, bool>::T_inferred DER_BOOL;
        typedef typename create_trait<DER, LIT, IT>::T_inferred DER_IT;

	// create and downcast to boolean because this type of constructor can not be booleand as FullyDist can not be boolean
	SpParMat<IT,bool,DER_BOOL> S = SpParMat<IT, IT, DER_IT> (total_m, total_m, ri, ri, 1);
	// clear memory of S but not *this
	SpParMat<IT,NT,DER> SA = Mult_AnXBn_DoubleBuff< BoolCopy2ndSRing<NT> , NT, DER>(S, *this, true, false);

	SpParMat<IT,bool,DER_BOOL> T = SpParMat<IT, IT, DER_IT> (total_n, total_n, ci, ci, 1);
	// clear memory of SA and T
	SpParMat<IT,NT,DER> SAT = Mult_AnXBn_DoubleBuff< BoolCopy1stSRing<NT> , NT, DER>(SA, T, true, true);


	// the type of the SAT matrix does not matter when calling set difference
	// because it just copies the non-excluded values from (*this) matrix, without touching values in SAT  
	SetDifference(SAT);	
}


// removes all nonzeros of rows in ri and columns in ci
// if A is an adjacency matrix and ri=ci,
// this prunes *all* (and not just the induced) connections of vertices in ri
// effectively rendering the vertices in ri *disconnected*
template <class IT, class NT, class DER>
void SpParMat<IT,NT,DER>::PruneFull(const FullyDistVec<IT,IT> & ri, const FullyDistVec<IT,IT> & ci)
{
	if((*(ri.commGrid) != *(commGrid)) || (*(ci.commGrid) != *(commGrid)))
	{
		SpParHelper::Print("Grids are not comparable, Prune fails!\n", commGrid->GetWorld());
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
	}

	// Safety check
	IT locmax_ri = 0;
	IT locmax_ci = 0;
	if(!ri.arr.empty())
		locmax_ri = *std::max_element(ri.arr.begin(), ri.arr.end());
	if(!ci.arr.empty())
		locmax_ci = *std::max_element(ci.arr.begin(), ci.arr.end());

	IT total_m = getnrow();
	IT total_n = getncol();
	if(locmax_ri > total_m || locmax_ci > total_n)	
	{
		throw outofrangeexception();
	}

        // infer the concrete types to replace the value with bool
	typedef typename DER::LocalIT LIT;
        typedef typename create_trait<DER, LIT, bool>::T_inferred DER_BOOL;
        typedef typename create_trait<DER, LIT, IT>::T_inferred DER_IT;

	// create and downcast to boolean because this type of constructor can not be booleand as FullyDist can not be boolean
	SpParMat<IT,bool,DER_BOOL> S = SpParMat<IT, IT, DER_IT> (total_m, total_m, ri, ri, 1);
	SpParMat<IT,NT,DER> SA = Mult_AnXBn_DoubleBuff< BoolCopy2ndSRing<NT> , NT, DER>(S, *this, true, false); // clear memory of S, but not *this

	SpParMat<IT,bool,DER_BOOL> T = SpParMat<IT, IT, DER_IT> (total_n, total_n, ci, ci, 1);
	SpParMat<IT,NT,DER> AT = Mult_AnXBn_DoubleBuff< BoolCopy1stSRing<NT> , NT, DER>(*this, T, false, true); // clear memory of T, but not *this

	// SA extracted rows of A in ri
	// AT extracted columns of A in ci

	SetDifference(SA);
	SetDifference(AT);	
}

//! Prune every column of a sparse matrix based on pvals
template <class IT, class NT, class DER>
template <typename _BinaryOperation>
SpParMat<IT,NT,DER> SpParMat<IT,NT,DER>::PruneColumn(const FullyDistVec<IT,NT> & pvals, _BinaryOperation __binary_op, bool inPlace)
{
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    //MPI_Barrier(MPI_COMM_WORLD);
    MPI_Comm World = pvals.commGrid->GetWorld();
    MPI_Barrier(World);
    if(getncol() != pvals.TotalLength())
    {
        std::ostringstream outs;
        outs << "Can not prune column-by-column, dimensions does not match"<< std::endl;
        outs << getncol() << " != " << pvals.TotalLength() << std::endl;
        SpParHelper::Print(outs.str());
        MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
    }
    if(! ( *(getcommgrid()) == *(pvals.getcommgrid())) )
    {
        std::cout << "Grids are not comparable for PurneColumn" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
    }
    
    MPI_Comm ColWorld = pvals.commGrid->GetColWorld();
    
    int xsize = (int) pvals.LocArrSize();
    int trxsize = 0;

    
    int diagneigh = pvals.commGrid->GetComplementRank();
    MPI_Status status;
    MPI_Sendrecv(&xsize, 1, MPI_INT, diagneigh, TRX, &trxsize, 1, MPI_INT, diagneigh, TRX, World, &status);


    NT * trxnums = new NT[trxsize];
    MPI_Sendrecv(const_cast<NT*>(SpHelper::p2a(pvals.arr)), xsize, MPIType<NT>(), diagneigh, TRX, trxnums, trxsize, MPIType<NT>(), diagneigh, TRX, World, &status);
    
    int colneighs, colrank;
    MPI_Comm_size(ColWorld, &colneighs);
    MPI_Comm_rank(ColWorld, &colrank);
    int * colsize = new int[colneighs];
    colsize[colrank] = trxsize;
    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, colsize, 1, MPI_INT, ColWorld);
    int * dpls = new int[colneighs]();	// displacements (zero initialized pid)
    std::partial_sum(colsize, colsize+colneighs-1, dpls+1);
    int accsize = std::accumulate(colsize, colsize+colneighs, 0);
    std::vector<NT> numacc(accsize);

#ifdef COMBBLAS_DEBUG
    std::ostringstream outs2; 
    outs2 << "PruneColumn displacements: ";
    for(int i=0; i< colneighs; ++i)
    {
	outs2 << dpls[i] << " ";
    }
    outs2 << std::endl;
    SpParHelper::Print(outs2.str());
    MPI_Barrier(World);
#endif
    
    
    MPI_Allgatherv(trxnums, trxsize, MPIType<NT>(), numacc.data(), colsize, dpls, MPIType<NT>(), ColWorld);
    delete [] trxnums;
    delete [] colsize;
    delete [] dpls;

    //sanity check
    if(accsize != getlocalcols()){
        fprintf(stderr, "[PruneColumn]\tmyrank:%d\taccsize:%d\tgetlocalcols():%d\n", myrank, accsize, getlocalcols());
    }
    assert(accsize == getlocalcols());
    if (inPlace)
    {
        spSeq->PruneColumn(numacc.data(), __binary_op, inPlace);
        return SpParMat<IT,NT,DER>(getcommgrid()); // return blank to match signature
    }
    else
    {
        return SpParMat<IT,NT,DER>(spSeq->PruneColumn(numacc.data(), __binary_op, inPlace), commGrid);
    }
}

template <class IT, class NT, class DER>
template <class IRRELEVANT_NT>
void SpParMat<IT,NT,DER>::PruneColumnByIndex(const FullyDistSpVec<IT,IRRELEVANT_NT>& ci)
{
    MPI_Comm World = ci.commGrid->GetWorld();
    MPI_Barrier(World);

    if (getncol() != ci.TotalLength())
    {
        std::ostringstream outs;
        outs << "Cannot prune column-by-column, dimensions do not match" << std::endl;
        outs << getncol() << " != " << ci.TotalLength() << std::endl;
        SpParHelper::Print(outs.str());
        MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
    }

    if (!(*(getcommgrid()) == *(ci.getcommgrid())))
    {
        std::cout << "Grids are not comparable for PruneColumnByIndex" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
    }

    MPI_Comm ColWorld = ci.commGrid->GetColWorld();
    int diagneigh = ci.commGrid->GetComplementRank();

    IT xlocnz = ci.getlocnnz();
    IT xrofst = ci.RowLenUntil();
    IT trxrofst;
    IT trxlocnz = 0;

    MPI_Sendrecv(&xrofst, 1, MPIType<IT>(), diagneigh, TROST, &trxrofst, 1, MPIType<IT>(), diagneigh, TROST, World, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&xlocnz, 1, MPIType<IT>(), diagneigh, TRNNZ, &trxlocnz, 1, MPIType<IT>(), diagneigh, TRNNZ, World, MPI_STATUS_IGNORE);

    std::vector<IT> trxinds(trxlocnz);

    MPI_Sendrecv(ci.ind.data(), xlocnz, MPIType<IT>(), diagneigh, TRI, trxinds.data(), trxlocnz, MPIType<IT>(), diagneigh, TRI, World, MPI_STATUS_IGNORE);

    std::transform(trxinds.data(), trxinds.data() + trxlocnz, trxinds.data(),
    	[trxrofst](IT val){return val + trxrofst;});

    int colneighs, colrank;
    MPI_Comm_size(ColWorld, &colneighs);
    MPI_Comm_rank(ColWorld, &colrank);

    std::vector<int> colnz(colneighs);
    colnz[colrank] = trxlocnz;
    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, colnz.data(), 1, MPI_INT, ColWorld);
    std::vector<int> dpls(colneighs, 0);
    std::partial_sum(colnz.begin(), colnz.end()-1, dpls.begin()+1);
    IT accnz = std::accumulate(colnz.begin(), colnz.end(), 0);

    std::vector<IT> indacc(accnz);
    MPI_Allgatherv(trxinds.data(), trxlocnz, MPIType<IT>(), indacc.data(), colnz.data(), dpls.data(), MPIType<IT>(), ColWorld);

    std::sort(indacc.begin(), indacc.end());

    spSeq->PruneColumnByIndex(indacc);
}


//! Prune columns of a sparse matrix selected by nonzero indices of pvals
//! Each selected column is pruned by corresponding values in pvals
template <class IT, class NT, class DER>
template <typename _BinaryOperation>
SpParMat<IT,NT,DER> SpParMat<IT,NT,DER>::PruneColumn(const FullyDistSpVec<IT,NT> & pvals, _BinaryOperation __binary_op, bool inPlace)
{
    //MPI_Barrier(MPI_COMM_WORLD);
    MPI_Comm World = pvals.commGrid->GetWorld();
    MPI_Barrier(World);
    if(getncol() != pvals.TotalLength())
    {
        std::ostringstream outs;
        outs << "Can not prune column-by-column, dimensions does not match"<< std::endl;
        outs << getncol() << " != " << pvals.TotalLength() << std::endl;
        SpParHelper::Print(outs.str());
        MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
    }
    if(! ( *(getcommgrid()) == *(pvals.getcommgrid())) )
    {
        std::cout << "Grids are not comparable for PurneColumn" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
    }
    
    MPI_Comm ColWorld = pvals.commGrid->GetColWorld();
    int diagneigh = pvals.commGrid->GetComplementRank();
    
    IT xlocnz = pvals.getlocnnz();
    IT roffst = pvals.RowLenUntil();
    IT roffset;
    IT trxlocnz = 0;
    
    MPI_Status status;
    MPI_Sendrecv(&roffst, 1, MPIType<IT>(), diagneigh, TROST, &roffset, 1, MPIType<IT>(), diagneigh, TROST, World, &status);
    MPI_Sendrecv(&xlocnz, 1, MPIType<IT>(), diagneigh, TRNNZ, &trxlocnz, 1, MPIType<IT>(), diagneigh, TRNNZ, World, &status);
    
    std::vector<IT> trxinds (trxlocnz);
    std::vector<NT> trxnums (trxlocnz);
    MPI_Sendrecv(pvals.ind.data(), xlocnz, MPIType<IT>(), diagneigh, TRI, trxinds.data(), trxlocnz, MPIType<IT>(), diagneigh, TRI, World, &status);
    MPI_Sendrecv(pvals.num.data(), xlocnz, MPIType<NT>(), diagneigh, TRX, trxnums.data(), trxlocnz, MPIType<NT>(), diagneigh, TRX, World, &status);
    std::transform(trxinds.data(), trxinds.data()+trxlocnz, trxinds.data(),
    	[roffset](IT val){return val + roffset;});
    
    int colneighs, colrank;
    MPI_Comm_size(ColWorld, &colneighs);
    MPI_Comm_rank(ColWorld, &colrank);
    int * colnz = new int[colneighs];
    colnz[colrank] = trxlocnz;
    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, colnz, 1, MPI_INT, ColWorld);
    int * dpls = new int[colneighs]();	// displacements (zero initialized pid)
    std::partial_sum(colnz, colnz+colneighs-1, dpls+1);
    IT accnz = std::accumulate(colnz, colnz+colneighs, 0);
 
    std::vector<IT> indacc(accnz);
    std::vector<NT> numacc(accnz);
    MPI_Allgatherv(trxinds.data(), trxlocnz, MPIType<IT>(), indacc.data(), colnz, dpls, MPIType<IT>(), ColWorld);
    MPI_Allgatherv(trxnums.data(), trxlocnz, MPIType<NT>(), numacc.data(), colnz, dpls, MPIType<NT>(), ColWorld);
    
    delete [] colnz;
    delete [] dpls;
    

    if (inPlace)
    {
        spSeq->PruneColumn(indacc.data(), numacc.data(), __binary_op, inPlace);
        return SpParMat<IT,NT,DER>(getcommgrid()); // return blank to match signature
    }
    else
    {
        return SpParMat<IT,NT,DER>(spSeq->PruneColumn(indacc.data(), numacc.data(), __binary_op, inPlace), commGrid);
    }
}



// In-place version where rhs type is the same (no need for type promotion)
template <class IT, class NT, class DER>
void SpParMat<IT,NT,DER>::EWiseMult (const SpParMat< IT,NT,DER >  & rhs, bool exclude)
{
	if(*commGrid == *rhs.commGrid)	
	{
		spSeq->EWiseMult(*(rhs.spSeq), exclude);		// Dimension compatibility check performed by sequential function
	}
	else
	{
		std::cout << "Grids are not comparable, EWiseMult() fails !" << std::endl; 
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
	}	
}


// Aydin (June 2021):
// This currently duplicates the work of EWiseMult with exclude = true
// However, this is the right way of implementing it because it allows set difference when 
// the types of two matrices do not have a valid multiplication operator defined
// set difference should not require such an operator so we will move all code 
// bases that use EWiseMult(..., exclude=true) to this one
template <class IT, class NT, class DER>
void SpParMat<IT,NT,DER>::SetDifference(const SpParMat<IT,NT,DER> & rhs)
{
	if(*commGrid == *rhs.commGrid)	
	{
		spSeq->SetDifference(*(rhs.spSeq));		// Dimension compatibility check performed by sequential function
	}
	else
	{
		std::cout << "Grids are not comparable, SetDifference() fails !" << std::endl; 
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
	}	
}


template <class IT, class NT, class DER>
void SpParMat<IT,NT,DER>::EWiseScale(const DenseParMat<IT, NT> & rhs)
{
	if(*commGrid == *rhs.commGrid)	
	{
		spSeq->EWiseScale(rhs.array, rhs.m, rhs.n);	// Dimension compatibility check performed by sequential function
	}
	else
	{
		std::cout << "Grids are not comparable, EWiseScale() fails !" << std::endl; 
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
	}
}

template <class IT, class NT, class DER>
template <typename _BinaryOperation>
void SpParMat<IT,NT,DER>::UpdateDense(DenseParMat<IT, NT> & rhs, _BinaryOperation __binary_op) const
{
	if(*commGrid == *rhs.commGrid)	
	{
		if(getlocalrows() == rhs.m  && getlocalcols() == rhs.n)
		{
			spSeq->UpdateDense(rhs.array, __binary_op);
		}
		else
		{
			std::cout << "Matrices have different dimensions, UpdateDense() fails !" << std::endl;
			MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
		}
	}
	else
	{
		std::cout << "Grids are not comparable, UpdateDense() fails !" << std::endl; 
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
	}
}

template <class IT, class NT, class DER>
void SpParMat<IT,NT,DER>::PrintInfo() const
{
	IT mm = getnrow(); 
	IT nn = getncol();
	IT nznz = getnnz();
	
	if (commGrid->myrank == 0)	
		std::cout << "As a whole: " << mm << " rows and "<< nn <<" columns and "<<  nznz << " nonzeros" << std::endl;
    
#ifdef DEBUG
	IT allprocs = commGrid->grrows * commGrid->grcols;
	for(IT i=0; i< allprocs; ++i)
	{
		if (commGrid->myrank == i)
		{
      std::cout << "Processor (" << commGrid->GetRankInProcRow() << "," << commGrid->GetRankInProcCol() << ")'s data: " << std::endl;
			spSeq->PrintInfo();
		}
		MPI_Barrier(commGrid->GetWorld());
	}
#endif
}

template <class IT, class NT, class DER>
bool SpParMat<IT,NT,DER>::operator== (const SpParMat<IT,NT,DER> & rhs) const
{
	int local = static_cast<int>((*spSeq) == (*(rhs.spSeq)));
	int whole = 1;
	MPI_Allreduce( &local, &whole, 1, MPI_INT, MPI_BAND, commGrid->GetWorld());
	return static_cast<bool>(whole);	
}


/**
 ** Private function that carries code common to different sparse() constructors
 ** Before this call, commGrid is already set
 **/
template <class IT, class NT, class DER>
template <typename _BinaryOperation, typename LIT>
void SpParMat< IT,NT,DER >::SparseCommon(std::vector< std::vector < std::tuple<LIT,LIT,NT> > > & data, LIT locsize, IT total_m, IT total_n, _BinaryOperation BinOp)
//void SpParMat< IT,NT,DER >::SparseCommon(std::vector< std::vector < std::tuple<typename DER::LocalIT,typename DER::LocalIT,NT> > > & data, typename DER::LocalIT locsize, IT total_m, IT total_n, _BinaryOperation BinOp)
{
    //typedef typename DER::LocalIT LIT;
	int nprocs = commGrid->GetSize();
	int * sendcnt = new int[nprocs];
	int * recvcnt = new int[nprocs];
	for(int i=0; i<nprocs; ++i)
		sendcnt[i] = data[i].size();	// sizes are all the same

	MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, commGrid->GetWorld()); // share the counts
	int * sdispls = new int[nprocs]();
	int * rdispls = new int[nprocs]();
	std::partial_sum(sendcnt, sendcnt+nprocs-1, sdispls+1);
	std::partial_sum(recvcnt, recvcnt+nprocs-1, rdispls+1);
	IT totrecv = std::accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));
	IT totsent = std::accumulate(sendcnt,sendcnt+nprocs, static_cast<IT>(0));	
	
	assert((totsent < std::numeric_limits<int>::max()));	
	assert((totrecv < std::numeric_limits<int>::max()));
	

#if 0 
	ofstream oput;
        commGrid->OpenDebugFile("Displacements", oput);
	copy(sdispls, sdispls+nprocs, ostream_iterator<int>(oput, " "));   oput << endl;
	copy(rdispls, rdispls+nprocs, ostream_iterator<int>(oput, " "));   oput << endl;
	oput.close();
	
	IT * gsizes;
	if(commGrid->GetRank() == 0) gsizes = new IT[nprocs];
    	MPI_Gather(&totrecv, 1, MPIType<IT>(), gsizes, 1, MPIType<IT>(), 0, commGrid->GetWorld());
	if(commGrid->GetRank() == 0) { std::copy(gsizes, gsizes+nprocs, std::ostream_iterator<IT>(std::cout, " "));   std::cout << std::endl; }
	MPI_Barrier(commGrid->GetWorld());
    	MPI_Gather(&totsent, 1, MPIType<IT>(), gsizes, 1, MPIType<IT>(), 0, commGrid->GetWorld());
	if(commGrid->GetRank() == 0) { copy(gsizes, gsizes+nprocs, ostream_iterator<IT>(cout, " "));   cout << endl; }
	MPI_Barrier(commGrid->GetWorld());
	if(commGrid->GetRank() == 0) delete [] gsizes;
#endif

  	std::tuple<LIT,LIT,NT> * senddata = new std::tuple<LIT,LIT,NT>[locsize];	// re-used for both rows and columns
	for(int i=0; i<nprocs; ++i)
	{
		std::copy(data[i].begin(), data[i].end(), senddata+sdispls[i]);
		data[i].clear();	// clear memory
		data[i].shrink_to_fit();
	}
	MPI_Datatype MPI_triple;
	MPI_Type_contiguous(sizeof(std::tuple<LIT,LIT,NT>), MPI_CHAR, &MPI_triple);
	MPI_Type_commit(&MPI_triple);

	std::tuple<LIT,LIT,NT> * recvdata = new std::tuple<LIT,LIT,NT>[totrecv];	
	MPI_Alltoallv(senddata, sendcnt, sdispls, MPI_triple, recvdata, recvcnt, rdispls, MPI_triple, commGrid->GetWorld());

	DeleteAll(senddata, sendcnt, recvcnt, sdispls, rdispls);
	MPI_Type_free(&MPI_triple);

	int r = commGrid->GetGridRows();
	int s = commGrid->GetGridCols();
	IT m_perproc = total_m / r;
	IT n_perproc = total_n / s;
	int myprocrow = commGrid->GetRankInProcCol();
	int myproccol = commGrid->GetRankInProcRow();
	IT locrows, loccols; 
	if(myprocrow != r-1)	locrows = m_perproc;
	else 	locrows = total_m - myprocrow * m_perproc;
	if(myproccol != s-1)	loccols = n_perproc;
	else	loccols = total_n - myproccol * n_perproc;
    
	SpTuples<LIT,NT> A(totrecv, locrows, loccols, recvdata);	// It is ~SpTuples's job to deallocate
	
    	// the previous constructor sorts based on columns-first (but that doesn't matter as long as they are sorted one way or another)
    	A.RemoveDuplicates(BinOp);
	
  	spSeq = new DER(A,false);        // Convert SpTuples to DER
}



template <class IT, class NT, class DER>
std::vector<std::vector<SpParMat<IT, NT, DER>>>
SpParMat<IT, NT, DER>::BlockSplit (int br, int bc)
{
	IT	g_nr   = this->getnrow();
	IT	g_nc   = this->getncol();
	
	if (br == 1 && bc == 1 || (br > g_nr || bc > g_nc))
		return std::vector<std::vector<SpParMat<IT, NT, DER>>>
			(1, std::vector<SpParMat<IT, NT, DER>>(1, *this));
	
	int np	 = commGrid->GetSize();
	int rank = commGrid->GetRank();
	
	std::vector<std::vector<SpParMat<IT, NT, DER>>>
		bmats(br,
			  std::vector<SpParMat<IT, NT, DER>>
			  (bc, SpParMat<IT, NT, DER>(commGrid)));
	std::vector<std::vector<std::vector<std::vector<std::tuple<IT, IT, NT>>>>>
		btuples(br,
				std::vector<std::vector<std::vector<std::tuple<IT, IT, NT>>>>
				(bc, std::vector<std::vector<std::tuple<IT, IT, NT>>>
				 (np, std::vector<std::tuple<IT, IT, NT>>())));

	assert(spSeq != NULL);
	
	SpTuples<IT, NT> tuples(*spSeq);	
	IT	g_rbeg = (g_nr/commGrid->GetGridRows()) * commGrid->GetRankInProcCol();
	IT	g_cbeg = (g_nc/commGrid->GetGridCols()) * commGrid->GetRankInProcRow();
	IT	br_sz  = g_nr / br;
	IT	br_r   = g_nr % br;
	IT	bc_sz  = g_nc / bc;
	IT	bc_r   = g_nc % bc;

	std::vector<IT> br_sizes(br, br_sz);
	std::vector<IT> bc_sizes(bc, bc_sz);
	for (IT i = 0; i < br_r; ++i)
		++br_sizes[i];
	for (IT i = 0; i < bc_r; ++i)
		++bc_sizes[i];

	auto get_block = [](IT x, IT sz, IT r, IT &bid, IT &idx)
		{
			if (x < (r*(sz+1)))
			{
				bid = x / (sz+1);
				idx = x % (sz+1);
			}
			else
			{
				bid = (x-r) / sz;
				idx = (x-r) % sz;
			}
		};
	

	// gather tuples
	for (int64_t i = 0; i < tuples.getnnz(); ++i)
	{
		IT g_ridx = g_rbeg + tuples.rowindex(i);
		IT g_cidx = g_cbeg + tuples.colindex(i);

		IT rbid, ridx, ridx_l, cbid, cidx, cidx_l;
		get_block(g_ridx, br_sz, br_r, rbid, ridx);
		get_block(g_cidx, bc_sz, bc_r, cbid, cidx);
		int owner = Owner(br_sizes[rbid], bc_sizes[cbid], ridx, cidx,
						  ridx_l, cidx_l);

		btuples[rbid][cbid][owner].push_back({ridx_l, cidx_l, tuples.numvalue(i)});		
	}

	
	// form matrices
	for (int i = 0; i < br; ++i)
	{
		for (int j = 0; j < bc; ++j)
		{
			IT locsize = 0;
			for (auto &el : btuples[i][j])
				locsize += el.size();

			auto &M = bmats[i][j];
			M.SparseCommon(btuples[i][j], locsize, br_sizes[i], bc_sizes[j],
						   maximum<NT>()); // there are no duplicates
		}
	}

	return bmats;
}



//! All vectors are zero-based indexed (as usual)
template <class IT, class NT, class DER>
SpParMat< IT,NT,DER >::SpParMat (IT total_m, IT total_n, const FullyDistVec<IT,IT> & distrows, 
				const FullyDistVec<IT,IT> & distcols, const FullyDistVec<IT,NT> & distvals, bool SumDuplicates)
{
	if((*(distrows.commGrid) != *(distcols.commGrid)) || (*(distcols.commGrid) != *(distvals.commGrid)))
	{
		SpParHelper::Print("Grids are not comparable, Sparse() fails!\n");  // commGrid is not initialized yet
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
	}
	if((distrows.TotalLength() != distcols.TotalLength()) || (distcols.TotalLength() != distvals.TotalLength()))
	{
		SpParHelper::Print("Vectors have different sizes, Sparse() fails!");
		MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
	}

	commGrid = distrows.commGrid;	
	int nprocs = commGrid->GetSize();
	std::vector< std::vector < std::tuple<IT,IT,NT> > > data(nprocs);

	IT locsize = distrows.LocArrSize();
	for(IT i=0; i<locsize; ++i)
	{
		IT lrow, lcol; 
		int owner = Owner(total_m, total_n, distrows.arr[i], distcols.arr[i], lrow, lcol);
		data[owner].push_back(std::make_tuple(lrow,lcol,distvals.arr[i]));	
	}
    if(SumDuplicates)
    {
        SparseCommon(data, locsize, total_m, total_n, std::plus<NT>());
    }
    else
    {
        SparseCommon(data, locsize, total_m, total_n, maximum<NT>());
    }
}



template <class IT, class NT, class DER>
SpParMat< IT,NT,DER >::SpParMat (IT total_m, IT total_n, const FullyDistVec<IT,IT> & distrows, 
				const FullyDistVec<IT,IT> & distcols, const NT & val, bool SumDuplicates)
{
	if((*(distrows.commGrid) != *(distcols.commGrid)) )
	{
		SpParHelper::Print("Grids are not comparable, Sparse() fails!\n");
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
	}
	if((distrows.TotalLength() != distcols.TotalLength()) )
	{
		SpParHelper::Print("Vectors have different sizes, Sparse() fails!\n");
		MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
	}
	commGrid = distrows.commGrid;
	int nprocs = commGrid->GetSize();
	std::vector< std::vector < std::tuple<IT,IT,NT> > > data(nprocs);

	IT locsize = distrows.LocArrSize();
	for(IT i=0; i<locsize; ++i)
	{
		IT lrow, lcol; 
		int owner = Owner(total_m, total_n, distrows.arr[i], distcols.arr[i], lrow, lcol);
		data[owner].push_back(std::make_tuple(lrow,lcol,val));	
	}
    if(SumDuplicates)
    {
        SparseCommon(data, locsize, total_m, total_n, std::plus<NT>());
    }
    else
    {
        SparseCommon(data, locsize, total_m, total_n, maximum<NT>());
    }
}

template <class IT, class NT, class DER>
template <class DELIT>
SpParMat< IT,NT,DER >::SpParMat (const DistEdgeList<DELIT> & DEL, bool removeloops)
{
	commGrid = DEL.commGrid;	
	typedef typename DER::LocalIT LIT;

	int nprocs = commGrid->GetSize();
	int gridrows = commGrid->GetGridRows();
	int gridcols = commGrid->GetGridCols();
	std::vector< std::vector<LIT> > data(nprocs);	// enties are pre-converted to local indices before getting pushed into "data"

	LIT m_perproc = DEL.getGlobalV() / gridrows;
	LIT n_perproc = DEL.getGlobalV() / gridcols;

	if(sizeof(LIT) < sizeof(DELIT))
	{
		std::ostringstream outs;
		outs << "Warning: Using smaller indices for the matrix than DistEdgeList\n";
		outs << "Local matrices are " << m_perproc << "-by-" << n_perproc << std::endl;
		SpParHelper::Print(outs.str(), commGrid->GetWorld());   // commgrid initialized
	}	
	
    LIT stages = MEM_EFFICIENT_STAGES;		// to lower memory consumption, form sparse matrix in stages
	
	// even if local indices (LIT) are 32-bits, we should work with 64-bits for global info
	int64_t perstage = DEL.nedges / stages;
	LIT totrecv = 0;
	std::vector<LIT> alledges;
    
	for(LIT s=0; s< stages; ++s)
	{
		int64_t n_befor = s*perstage;
		int64_t n_after= ((s==(stages-1))? DEL.nedges : ((s+1)*perstage));

		// clear the source vertex by setting it to -1
		int realedges = 0;	// these are "local" realedges

		if(DEL.pedges)	
		{
			for (int64_t i = n_befor; i < n_after; i++)
			{
				int64_t fr = get_v0_from_edge(&(DEL.pedges[i]));
				int64_t to = get_v1_from_edge(&(DEL.pedges[i]));

				if(fr >= 0 && to >= 0)	// otherwise skip
				{
                    IT lrow, lcol;
                    int owner = Owner(DEL.getGlobalV(), DEL.getGlobalV(), fr, to, lrow, lcol);
					data[owner].push_back(lrow);	// row_id
					data[owner].push_back(lcol);	// col_id
					++realedges;
				}
			}
		}
		else
		{
			for (int64_t i = n_befor; i < n_after; i++)
			{
				if(DEL.edges[2*i+0] >= 0 && DEL.edges[2*i+1] >= 0)	// otherwise skip
				{
                    IT lrow, lcol;
                    int owner = Owner(DEL.getGlobalV(), DEL.getGlobalV(), DEL.edges[2*i+0], DEL.edges[2*i+1], lrow, lcol);
					data[owner].push_back(lrow);
					data[owner].push_back(lcol);
					++realedges;
				}
			}
		}

  		LIT * sendbuf = new LIT[2*realedges];
		int * sendcnt = new int[nprocs];
		int * sdispls = new int[nprocs];
		for(int i=0; i<nprocs; ++i)
			sendcnt[i] = data[i].size();

		int * rdispls = new int[nprocs];
		int * recvcnt = new int[nprocs];
		MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT,commGrid->GetWorld()); // share the counts

		sdispls[0] = 0;
		rdispls[0] = 0;
		for(int i=0; i<nprocs-1; ++i)
		{
			sdispls[i+1] = sdispls[i] + sendcnt[i];
			rdispls[i+1] = rdispls[i] + recvcnt[i];
		}
		for(int i=0; i<nprocs; ++i)
			std::copy(data[i].begin(), data[i].end(), sendbuf+sdispls[i]);
		
		// clear memory
		for(int i=0; i<nprocs; ++i)
			std::vector<LIT>().swap(data[i]);

		// ABAB: Total number of edges received might not be LIT-addressible
		// However, each edge_id is LIT-addressible
		IT thisrecv = std::accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));	// thisrecv = 2*locedges
		LIT * recvbuf = new LIT[thisrecv];
		totrecv += thisrecv;
			
		MPI_Alltoallv(sendbuf, sendcnt, sdispls, MPIType<LIT>(), recvbuf, recvcnt, rdispls, MPIType<LIT>(), commGrid->GetWorld());
		DeleteAll(sendcnt, recvcnt, sdispls, rdispls,sendbuf);
    std::copy (recvbuf,recvbuf+thisrecv,std::back_inserter(alledges));	// copy to all edges
		delete [] recvbuf;
	}

	int myprocrow = commGrid->GetRankInProcCol();
	int myproccol = commGrid->GetRankInProcRow();
	LIT locrows, loccols; 
	if(myprocrow != gridrows-1)	locrows = m_perproc;
	else 	locrows = DEL.getGlobalV() - myprocrow * m_perproc;
	if(myproccol != gridcols-1)	loccols = n_perproc;
	else	loccols = DEL.getGlobalV() - myproccol * n_perproc;

  	SpTuples<LIT,NT> A(totrecv/2, locrows, loccols, alledges, removeloops);  	// alledges is empty upon return
  	spSeq = new DER(A,false);        // Convert SpTuples to DER
}

template <class IT, class NT, class DER>
IT SpParMat<IT,NT,DER>::RemoveLoops()
{
	MPI_Comm DiagWorld = commGrid->GetDiagWorld();
	IT totrem;
	IT removed = 0;
	if(DiagWorld != MPI_COMM_NULL) // Diagonal processors only
	{
		typedef typename DER::LocalIT LIT;
		SpTuples<LIT,NT> tuples(*spSeq);
		delete spSeq;
		removed  = tuples.RemoveLoops();
		spSeq = new DER(tuples, false);	// Convert to DER
	}
	MPI_Allreduce( &removed, & totrem, 1, MPIType<IT>(), MPI_SUM, commGrid->GetWorld());
	return totrem;
}		



template <class IT, class NT, class DER>
void SpParMat<IT,NT,DER>::AddLoops(NT loopval, bool replaceExisting)
{
	MPI_Comm DiagWorld = commGrid->GetDiagWorld();
	if(DiagWorld != MPI_COMM_NULL) // Diagonal processors only
	{
    		typedef typename DER::LocalIT LIT;
		SpTuples<LIT,NT> tuples(*spSeq);
		delete spSeq;
		tuples.AddLoops(loopval, replaceExisting);
        	tuples.SortColBased();
		spSeq = new DER(tuples, false);	// Convert to DER
	}
}


// Different values on the diagonal
template <class IT, class NT, class DER>
void SpParMat<IT,NT,DER>::AddLoops(FullyDistVec<IT,NT> loopvals, bool replaceExisting)
{
    
    
    if(*loopvals.commGrid != *commGrid)
    {
        SpParHelper::Print("Grids are not comparable, SpParMat::AddLoops() fails!\n", commGrid->GetWorld());
        MPI_Abort(MPI_COMM_WORLD,GRIDMISMATCH);
    }
    if (getncol()!= loopvals.TotalLength())
    {
        SpParHelper::Print("The number of entries in loopvals is not equal to the number of diagonal entries.\n");
        MPI_Abort(MPI_COMM_WORLD,DIMMISMATCH);
    }
    
    // Gather data on the diagonal processor
    IT locsize = loopvals.LocArrSize();
    int rowProcs = commGrid->GetGridCols();
    std::vector<int> recvcnt(rowProcs, 0);
    std::vector<int> rdpls(rowProcs, 0);
    MPI_Gather(&locsize, 1, MPI_INT, recvcnt.data(), 1, MPI_INT, commGrid->GetDiagOfProcRow(), commGrid->GetRowWorld());
    std::partial_sum(recvcnt.data(), recvcnt.data()+rowProcs-1, rdpls.data()+1);

    IT totrecv = rdpls[rowProcs-1] + recvcnt[rowProcs-1];
    assert((totrecv < std::numeric_limits<int>::max()));

    std::vector<NT> rowvals(totrecv);
	MPI_Gatherv(loopvals.arr.data(), locsize, MPIType<NT>(), rowvals.data(), recvcnt.data(), rdpls.data(),
                 MPIType<NT>(), commGrid->GetDiagOfProcRow(), commGrid->GetRowWorld());

   
    MPI_Comm DiagWorld = commGrid->GetDiagWorld();
    if(DiagWorld != MPI_COMM_NULL) // Diagonal processors only
    {
        typedef typename DER::LocalIT LIT;
        SpTuples<LIT,NT> tuples(*spSeq);
        delete spSeq;
        tuples.AddLoops(rowvals, replaceExisting);
        tuples.SortColBased();
        spSeq = new DER(tuples, false);	// Convert to DER
    }
}


//! Pre-allocates buffers for row communication
//! additionally (if GATHERVOPT is defined, incomplete as of March 2016):
//! - Splits the local column indices to sparse & dense pieces to avoid redundant AllGather (sparse pieces get p2p)
template <class IT, class NT, class DER>
template <typename LIT, typename OT>
void SpParMat<IT,NT,DER>::OptimizeForGraph500(OptBuf<LIT,OT> & optbuf)
{
	if(spSeq->getnsplit() > 0)
	{
		SpParHelper::Print("Can not declare preallocated buffers for multithreaded execution\n", commGrid->GetWorld());
		return;
    }

    typedef typename DER::LocalIT LocIT;    // ABAB: should match the type of LIT. Check?
    
    // Set up communication buffers, one for all
	LocIT mA = spSeq->getnrow();
    LocIT nA = spSeq->getncol();
    
	int p_c = commGrid->GetGridCols();
    int p_r = commGrid->GetGridRows();
    
    LocIT rwperproc = mA / p_c; // per processors in row-wise communication
    LocIT cwperproc = nA / p_r; // per processors in column-wise communication
    
#ifdef GATHERVOPT
    LocIT * colinds = seq->GetDCSC()->jc;   // local nonzero column id's
    LocIT locnzc = seq->getnzc();
    LocIT cci = 0;  // index to column id's array (cci: current column index)
    int * gsizes = NULL;
    IT * ents = NULL;
    IT * dpls = NULL;
    std::vector<LocIT> pack2send;
    
    FullyDistSpVec<IT,IT> dummyRHS ( commGrid, getncol()); // dummy RHS vector to estimate index start position
    IT recveclen;
    
    for(int pid = 1; pid <= p_r; pid++)
    {
        IT diagoffset;
        MPI_Status status;
        IT offset = dummyRHS.RowLenUntil(pid-1);
        int diagneigh = commGrid->GetComplementRank();
        MPI_Sendrecv(&offset, 1, MPIType<IT>(), diagneigh, TRTAGNZ, &diagoffset, 1, MPIType<IT>(), diagneigh, TRTAGNZ, commGrid->GetWorld(), &status);

        LocIT endind = (pid == p_r)? nA : static_cast<LocIT>(pid) * cwperproc;     // the last one might have a larger share (is this fitting to the vector boundaries?)
        while(cci < locnzc && colinds[cci] < endind)
        {
            pack2send.push_back(colinds[cci++]-diagoffset);
        }
        if(pid-1 == myrank) gsizes = new int[p_r];
        MPI_Gather(&mysize, 1, MPI_INT, gsizes, 1, MPI_INT, pid-1, commGrid->GetColWorld());
        if(pid-1 == myrank)
        {
            IT colcnt = std::accumulate(gsizes, gsizes+p_r, static_cast<IT>(0));
            recvbuf = new IT[colcnt];
            dpls = new IT[p_r]();     // displacements (zero initialized pid)
            std::partial_sum(gsizes, gsizes+p_r-1, dpls+1);
        }
        
        // int MPI_Gatherv (void* sbuf, int scount, MPI_Datatype stype, void* rbuf, int *rcount, int* displs, MPI_Datatype rtype, int root, MPI_Comm comm)
        MPI_Gatherv(SpHelper::p2a(pack2send), mysize, MPIType<LocIT>(), recvbuf, gsizes, dpls, MPIType<LocIT>(), pid-1, commGrid->GetColWorld());
        std::vector<LocIT>().swap(pack2send);
        
       if(pid-1 == myrank)
       {
           recveclen = dummyRHS.MyLocLength();
           std::vector< std::vector<LocIT> > service(recveclen);
           for(int i=0; i< p_r; ++i)
           {
               for(int j=0; j< gsizes[i]; ++j)
               {
                   IT colid2update = recvbuf[dpls[i]+j];
                   if(service[colid2update].size() < GATHERVNEIGHLIMIT)
                   {
                       service.push_back(i);
                   }
                   // else don't increase any further and mark it done after the iterations are complete
               }
           }
       }
    }
#endif

    
	std::vector<bool> isthere(mA, false); // perhaps the only appropriate use of this crippled data structure
	std::vector<int> maxlens(p_c,0);	// maximum data size to be sent to any neighbor along the processor row

	for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)
	{
		for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)
		{
			LocIT rowid = nzit.rowid();
			if(!isthere[rowid])
			{
				LocIT owner = std::min(nzit.rowid() / rwperproc, (LocIT) p_c-1);
				maxlens[owner]++;
				isthere[rowid] = true;
			}
		}
	}
	SpParHelper::Print("Optimization buffers set\n", commGrid->GetWorld());
	optbuf.Set(maxlens,mA);
}

template <class IT, class NT, class DER>
void SpParMat<IT,NT,DER>::ActivateThreading(int numsplits)
{
	spSeq->RowSplit(numsplits);
}


/**
 * Parallel routine that returns A*A on the semiring SR
 * Uses only MPI-1 features (relies on simple blocking broadcast)
 **/  
template <class IT, class NT, class DER>
template <typename SR>
void SpParMat<IT,NT,DER>::Square ()
{
	int stages, dummy; 	// last two parameters of productgrid are ignored for synchronous multiplication
	std::shared_ptr<CommGrid> Grid = ProductGrid(commGrid.get(), commGrid.get(), stages, dummy, dummy);		

	typedef typename DER::LocalIT LIT;
	
	LIT AA_m = spSeq->getnrow();
	LIT AA_n = spSeq->getncol();
	
	DER seqTrn = spSeq->TransposeConst();	// will be automatically discarded after going out of scope		

	MPI_Barrier(commGrid->GetWorld());

	LIT ** NRecvSizes = SpHelper::allocate2D<LIT>(DER::esscount, stages);
	LIT ** TRecvSizes = SpHelper::allocate2D<LIT>(DER::esscount, stages);
	
	SpParHelper::GetSetSizes( *spSeq, NRecvSizes, commGrid->GetRowWorld());
	SpParHelper::GetSetSizes( seqTrn, TRecvSizes, commGrid->GetColWorld());

	// Remotely fetched matrices are stored as pointers
	DER * NRecv; 
	DER * TRecv;
	std::vector< SpTuples<LIT,NT>  *> tomerge;

	int Nself = commGrid->GetRankInProcRow();
	int Tself = commGrid->GetRankInProcCol();	

	for(int i = 0; i < stages; ++i) 
    {
		std::vector<LIT> ess;	
		if(i == Nself)  NRecv = spSeq;	// shallow-copy 
		else
		{
			ess.resize(DER::esscount);
			for(int j=0; j< DER::esscount; ++j)
				ess[j] = NRecvSizes[j][i];		// essentials of the ith matrix in this row
			NRecv = new DER();				// first, create the object
		}

		SpParHelper::BCastMatrix(Grid->GetRowWorld(), *NRecv, ess, i);	// then, broadcast its elements	
		ess.clear();	
		
		if(i == Tself)  TRecv = &seqTrn;	// shallow-copy
		else
		{
			ess.resize(DER::esscount);		
			for(int j=0; j< DER::esscount; ++j)
				ess[j] = TRecvSizes[j][i];
			TRecv = new DER();
		}
		SpParHelper::BCastMatrix(Grid->GetColWorld(), *TRecv, ess, i);	

		SpTuples<LIT,NT> * AA_cont = MultiplyReturnTuples<SR, NT>(*NRecv, *TRecv, false, true);
		if(!AA_cont->isZero()) 
			tomerge.push_back(AA_cont);

		if(i != Nself)	delete NRecv;
		if(i != Tself)  delete TRecv;
	}

	SpHelper::deallocate2D(NRecvSizes, DER::esscount);
	SpHelper::deallocate2D(TRecvSizes, DER::esscount);
	
	delete spSeq;		
	spSeq = new DER(MergeAll<SR>(tomerge, AA_m, AA_n), false);	// First get the result in SpTuples, then convert to UDER
	for(unsigned int i=0; i<tomerge.size(); ++i)
		delete tomerge[i];
}


template <class IT, class NT, class DER>
void SpParMat<IT,NT,DER>::Transpose()
{
	if(commGrid->myproccol == commGrid->myprocrow)	// Diagonal
	{
		spSeq->Transpose();			
	}
	else
	{
		typedef typename DER::LocalIT LIT;
		SpTuples<LIT,NT> Atuples(*spSeq);
		LIT locnnz = Atuples.getnnz();
		LIT * rows = new LIT[locnnz];
		LIT * cols = new LIT[locnnz];
		NT * vals = new NT[locnnz];
		for(LIT i=0; i < locnnz; ++i)
		{
			rows[i] = Atuples.colindex(i);	// swap (i,j) here
			cols[i] = Atuples.rowindex(i);
			vals[i] = Atuples.numvalue(i);
		}
		LIT locm = getlocalcols();
		LIT locn = getlocalrows();
		delete spSeq;

		LIT remotem, remoten, remotennz;
		std::swap(locm,locn);
		int diagneigh = commGrid->GetComplementRank();

		MPI_Status status;
		MPI_Sendrecv(&locnnz, 1, MPIType<LIT>(), diagneigh, TRTAGNZ, &remotennz, 1, MPIType<LIT>(), diagneigh, TRTAGNZ, commGrid->GetWorld(), &status);
		MPI_Sendrecv(&locn, 1, MPIType<LIT>(), diagneigh, TRTAGM, &remotem, 1, MPIType<LIT>(), diagneigh, TRTAGM, commGrid->GetWorld(), &status);
		MPI_Sendrecv(&locm, 1, MPIType<LIT>(), diagneigh, TRTAGN, &remoten, 1, MPIType<LIT>(), diagneigh, TRTAGN, commGrid->GetWorld(), &status);

		LIT * rowsrecv = new LIT[remotennz];
		MPI_Sendrecv(rows, locnnz, MPIType<LIT>(), diagneigh, TRTAGROWS, rowsrecv, remotennz, MPIType<LIT>(), diagneigh, TRTAGROWS, commGrid->GetWorld(), &status);
		delete [] rows;

		LIT * colsrecv = new LIT[remotennz];
		MPI_Sendrecv(cols, locnnz, MPIType<LIT>(), diagneigh, TRTAGCOLS, colsrecv, remotennz, MPIType<LIT>(), diagneigh, TRTAGCOLS, commGrid->GetWorld(), &status);
		delete [] cols;

		NT * valsrecv = new NT[remotennz];
		MPI_Sendrecv(vals, locnnz, MPIType<NT>(), diagneigh, TRTAGVALS, valsrecv, remotennz, MPIType<NT>(), diagneigh, TRTAGVALS, commGrid->GetWorld(), &status);
		delete [] vals;

		std::tuple<LIT,LIT,NT> * arrtuples = new std::tuple<LIT,LIT,NT>[remotennz];
		for(LIT i=0; i< remotennz; ++i)
		{
			arrtuples[i] = std::make_tuple(rowsrecv[i], colsrecv[i], valsrecv[i]);
		}	
		DeleteAll(rowsrecv, colsrecv, valsrecv);
		ColLexiCompare<LIT,NT> collexicogcmp;
		sort(arrtuples , arrtuples+remotennz, collexicogcmp );	// sort w.r.t columns here

		spSeq = new DER();
		spSeq->Create( remotennz, remotem, remoten, arrtuples);		// the deletion of arrtuples[] is handled by SpMat::Create
	}	
}		


template <class IT, class NT, class DER>
template <class HANDLER>
void SpParMat< IT,NT,DER >::SaveGathered(std::string filename, HANDLER handler, bool transpose) const
{
	int proccols = commGrid->GetGridCols();
	int procrows = commGrid->GetGridRows();
	IT totalm = getnrow();
	IT totaln = getncol();
	IT totnnz = getnnz();
	int flinelen = 0;
	std::ofstream out;
	if(commGrid->GetRank() == 0)
	{
		std::string s;
		std::stringstream strm;
		strm << "%%MatrixMarket matrix coordinate real general" << std::endl;
		strm << totalm << " " << totaln << " " << totnnz << std::endl;
		s = strm.str();
		out.open(filename.c_str(),std::ios_base::trunc);
		flinelen = s.length();
		out.write(s.c_str(), flinelen);
		out.close();
	}
	int colrank = commGrid->GetRankInProcCol(); 
	int colneighs = commGrid->GetGridRows();
	IT * locnrows = new IT[colneighs];	// number of rows is calculated by a reduction among the processor column
	locnrows[colrank] = (IT) getlocalrows();
	MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IT>(),locnrows, 1, MPIType<IT>(), commGrid->GetColWorld());
	IT roffset = std::accumulate(locnrows, locnrows+colrank, 0);
	delete [] locnrows;	

	MPI_Datatype datatype;
	MPI_Type_contiguous(sizeof(std::pair<IT,NT>), MPI_CHAR, &datatype);
	MPI_Type_commit(&datatype);

	for(int i = 0; i < procrows; i++)	// for all processor row (in order)
	{
		if(commGrid->GetRankInProcCol() == i)	// only the ith processor row
		{ 
			IT localrows = spSeq->getnrow();    // same along the processor row
			std::vector< std::vector< std::pair<IT,NT> > > csr(localrows);
			if(commGrid->GetRankInProcRow() == 0)	// get the head of processor row 
			{
				IT localcols = spSeq->getncol();    // might be different on the last processor on this processor row
				MPI_Bcast(&localcols, 1, MPIType<IT>(), 0, commGrid->GetRowWorld());
				for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)	// iterate over nonempty subcolumns
				{
					for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)
					{
						csr[nzit.rowid()].push_back( std::make_pair(colit.colid(), nzit.value()) );
					}
				}
			}
			else	// get the rest of the processors
			{
				IT n_perproc;
				MPI_Bcast(&n_perproc, 1, MPIType<IT>(), 0, commGrid->GetRowWorld());
				IT noffset = commGrid->GetRankInProcRow() * n_perproc; 
				for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)	// iterate over nonempty subcolumns
				{
					for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)
					{
						csr[nzit.rowid()].push_back( std::make_pair(colit.colid() + noffset, nzit.value()) );
					}
				}
			}
			std::pair<IT,NT> * ents = NULL;
			int * gsizes = NULL, * dpls = NULL;
			if(commGrid->GetRankInProcRow() == 0)	// only the head of processor row 
			{
				out.open(filename.c_str(),std::ios_base::app);
				gsizes = new int[proccols];
				dpls = new int[proccols]();	// displacements (zero initialized pid) 
			}
			for(int j = 0; j < localrows; ++j)	
			{
				IT rowcnt = 0;
				sort(csr[j].begin(), csr[j].end());
				int mysize = csr[j].size();
				MPI_Gather(&mysize, 1, MPI_INT, gsizes, 1, MPI_INT, 0, commGrid->GetRowWorld());
				if(commGrid->GetRankInProcRow() == 0)	
				{
					rowcnt = std::accumulate(gsizes, gsizes+proccols, static_cast<IT>(0));
					std::partial_sum(gsizes, gsizes+proccols-1, dpls+1);
					ents = new std::pair<IT,NT>[rowcnt];	// nonzero entries in the j'th local row
				}

				// int MPI_Gatherv (void* sbuf, int scount, MPI_Datatype stype, 
				// 		    void* rbuf, int *rcount, int* displs, MPI_Datatype rtype, int root, MPI_Comm comm)	
				MPI_Gatherv(SpHelper::p2a(csr[j]), mysize, datatype, ents, gsizes, dpls, datatype, 0, commGrid->GetRowWorld());
				if(commGrid->GetRankInProcRow() == 0)	
				{
					for(int k=0; k< rowcnt; ++k)
					{
						//out << j + roffset + 1 << "\t" << ents[k].first + 1 <<"\t" << ents[k].second << endl;
						if (!transpose)
							// regular
							out << j + roffset + 1 << "\t" << ents[k].first + 1 << "\t";
						else
							// transpose row/column
							out << ents[k].first + 1 << "\t" << j + roffset + 1 << "\t";
						handler.save(out, ents[k].second, j + roffset, ents[k].first);
						out << std::endl;
					}
					delete [] ents;
				}
			}
			if(commGrid->GetRankInProcRow() == 0)
			{
				DeleteAll(gsizes, dpls);
				out.close();
			}
		} // end_if the ith processor row 
		MPI_Barrier(commGrid->GetWorld());		// signal the end of ith processor row iteration (so that all processors block)
	}
}


//! Private subroutine of ReadGeneralizedTuples
//! totallength is the length of the dictionary, which we don't know in this labeled tuples format apriori
template <class IT, class NT, class DER>
MPI_File SpParMat< IT,NT,DER >::TupleRead1stPassNExchange (const std::string & filename, TYPE2SEND * & senddata, IT & totsend, 
							FullyDistVec<IT,STRASARRAY> & distmapper, uint64_t & totallength)
{
    int myrank = commGrid->GetRank();
    int nprocs = commGrid->GetSize();     

    MPI_Offset fpos, end_fpos;    
    struct stat st;     // get file size
    if (stat(filename.c_str(), &st) == -1)
    {
        MPI_Abort(MPI_COMM_WORLD, NOFILE);
    }
    int64_t file_size = st.st_size;
    if(myrank == 0)    // the offset needs to be for this rank
    {
        std::cout << "File is " << file_size << " bytes" << std::endl;
    }
    fpos = myrank * file_size / nprocs;

    if(myrank != (nprocs-1)) end_fpos = (myrank + 1) * file_size / nprocs;
    else end_fpos = file_size;

    MPI_File mpi_fh;
    MPI_File_open (commGrid->commWorld, const_cast<char*>(filename.c_str()), MPI_MODE_RDONLY, MPI_INFO_NULL, &mpi_fh);

    typedef std::map<std::string, uint64_t> KEYMAP; // due to potential (but extremely unlikely) collusions in MurmurHash, make the key to the std:map the string itself
    std::vector< KEYMAP > allkeys(nprocs);	  // map keeps the outgoing data unique, we could have applied this to HipMer too

    std::vector<std::string> lines;
    bool finished = SpParHelper::FetchBatch(mpi_fh, fpos, end_fpos, true, lines, myrank);
    int64_t entriesread = lines.size();
    SpHelper::ProcessLinesWithStringKeys(allkeys, lines,nprocs);

    while(!finished)
    {
        finished = SpParHelper::FetchBatch(mpi_fh, fpos, end_fpos, false, lines, myrank);

        entriesread += lines.size();
    	SpHelper::ProcessLinesWithStringKeys(allkeys, lines,nprocs);
    }
    int64_t allentriesread;
    MPI_Reduce(&entriesread, &allentriesread, 1, MPIType<int64_t>(), MPI_SUM, 0, commGrid->commWorld);
#ifdef COMBBLAS_DEBUG
    if(myrank == 0)
        std::cout << "Initial reading finished. Total number of entries read across all processors is " << allentriesread << std::endl;
#endif

    int * sendcnt = new int[nprocs];
    int * recvcnt = new int[nprocs];
    for(int i=0; i<nprocs; ++i)
	sendcnt[i] = allkeys[i].size();

    MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, commGrid->GetWorld()); // share the counts
    int * sdispls = new int[nprocs]();
    int * rdispls = new int[nprocs]();
    std::partial_sum(sendcnt, sendcnt+nprocs-1, sdispls+1);
    std::partial_sum(recvcnt, recvcnt+nprocs-1, rdispls+1);
    totsend = std::accumulate(sendcnt,sendcnt+nprocs, static_cast<IT>(0));
    IT totrecv = std::accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));	

    assert((totsend < std::numeric_limits<int>::max()));	
    assert((totrecv < std::numeric_limits<int>::max()));

    // The following are declared in SpParMat.h
    // typedef std::array<char, MAXVERTNAME> STRASARRAY;
    // typedef std::pair< STRASARRAY, uint64_t> TYPE2SEND;
    senddata = new TYPE2SEND[totsend];	

    #pragma omp parallel for
    for(int i=0; i<nprocs; ++i)
    {
	size_t j = 0;
	for(auto pobj:allkeys[i])
	{
		// The naked C-style array type is not copyable or assignable, but pair will require it, hence used std::array
		std::array<char, MAXVERTNAME> vname;
		std::copy( pobj.first.begin(), pobj.first.end(), vname.begin() );  
		if(pobj.first.length() < MAXVERTNAME)  vname[pobj.first.length()] = '\0';	// null termination		

		senddata[sdispls[i]+j] = TYPE2SEND(vname, pobj.second);
		j++;
	}
    }
    allkeys.clear();  // allkeys is no longer needed after this point

    MPI_Datatype MPI_HASH;
    MPI_Type_contiguous(sizeof(TYPE2SEND), MPI_CHAR, &MPI_HASH);
    MPI_Type_commit(&MPI_HASH);

    TYPE2SEND * recvdata = new TYPE2SEND[totrecv];	
    MPI_Alltoallv(senddata, sendcnt, sdispls, MPI_HASH, recvdata, recvcnt, rdispls, MPI_HASH, commGrid->GetWorld());
    // do not delete send buffers yet as we will use them to recv back the data
    
    std::set< std::pair<uint64_t, std::string>  > uniqsorted;
    for(IT i=0; i< totrecv; ++i)
    {
	    auto locnull = std::find(recvdata[i].first.begin(), recvdata[i].first.end(), '\0'); // find the null character (or string::end)
	    std::string strtmp(recvdata[i].first.begin(), locnull); // range constructor 
	    
	    uniqsorted.insert(std::make_pair(recvdata[i].second, strtmp));
    }
    uint64_t uniqsize = uniqsorted.size();
    
#ifdef COMBBLAS_DEBUG
    if(myrank == 0)
	    std::cout << "out of " << totrecv << " vertices received, " << uniqsize << " were unique" << std::endl;
#endif
    uint64_t sizeuntil = 0;
    totallength = 0;
    MPI_Exscan( &uniqsize, &sizeuntil, 1, MPIType<uint64_t>(), MPI_SUM, commGrid->GetWorld() );
    MPI_Allreduce(&uniqsize, &totallength, 1,  MPIType<uint64_t>(), MPI_SUM, commGrid->GetWorld());
    if(myrank == 0) sizeuntil = 0;  // because MPI_Exscan says the recvbuf in process 0 is undefined

    distmapper =  FullyDistVec<IT,STRASARRAY>(commGrid, totallength,STRASARRAY{});	
	
    // invindex does not conform to FullyDistVec boundaries, otherwise its contents are essentially the same as distmapper    
    KEYMAP invindex;	// KEYMAP is map<string, uint64_t>. 
    uint64_t locindex = 0;
    std::vector< std::vector< IT > > locs_send(nprocs);
    std::vector< std::vector< std::string > > data_send(nprocs);
    int * map_scnt = new int[nprocs]();	// send counts for this map only (to no confuse with the other sendcnt)        
    for(auto itr = uniqsorted.begin(); itr != uniqsorted.end(); ++itr)
    {
	    uint64_t globalindex = sizeuntil + locindex;
	    invindex.insert(std::make_pair(itr->second, globalindex));
	    
	    IT newlocid;	
	    int owner = distmapper.Owner(globalindex, newlocid);

	    //if(myrank == 0)
	    //    std::cout << "invindex received " << itr->second << " with global index " << globalindex << " to be owned by " << owner << " with index " << newlocid << std::endl;

	    locs_send[owner].push_back(newlocid);
	    data_send[owner].push_back(itr->second);
	    map_scnt[owner]++;
	    locindex++;
    }
    uniqsorted.clear();	// clear memory


    /* BEGIN: Redistributing the permutation vector to fit the FullyDistVec semantics */
    SpParHelper::ReDistributeToVector(map_scnt, locs_send, data_send, distmapper.arr, commGrid->GetWorld());   // map_scnt is deleted here
    /* END: Redistributing the permutation vector to fit the FullyDistVec semantics */ 

    for(IT i=0; i< totrecv; ++i)
    {
	    auto locnull = std::find(recvdata[i].first.begin(), recvdata[i].first.end(), '\0');
	    std::string searchstr(recvdata[i].first.begin(), locnull); // range constructor 

	    auto resp = invindex.find(searchstr); // recvdata[i] is of type pair< STRASARRAY, uint64_t>
	    if (resp != invindex.end())
	    {
		recvdata[i].second = resp->second;	// now instead of random numbers, recvdata's second entry will be its new index
	    }
	    else
		std::cout << "Assertion failed at proc " << myrank << ": the absence of the entry in invindex is unexpected!!!" << std::endl;
    }
    MPI_Alltoallv(recvdata, recvcnt, rdispls, MPI_HASH, senddata, sendcnt, sdispls, MPI_HASH, commGrid->GetWorld());    
    DeleteAll(recvdata, sendcnt, recvcnt, sdispls, rdispls);
    MPI_Type_free(&MPI_HASH);

    // the following gets deleted here: allkeys
    return mpi_fh;
}



//! Handles all sorts of orderings as long as there are no duplicates
//! Does not take matrix market banner (only tuples)
//! Data can be load imbalanced and the vertex labels can be arbitrary strings
//! Replaces ReadDistribute for imbalanced arbitrary input in tuples format
template <class IT, class NT, class DER>
template <typename _BinaryOperation>
FullyDistVec<IT,std::array<char, MAXVERTNAME> > SpParMat< IT,NT,DER >::ReadGeneralizedTuples (const std::string & filename, _BinaryOperation BinOp)
{       
    int myrank = commGrid->GetRank();
    int nprocs = commGrid->GetSize();  
    TYPE2SEND * senddata;
    IT totsend;
    uint64_t totallength;
    FullyDistVec<IT,STRASARRAY> distmapper(commGrid); // choice of array<char, MAXVERTNAME> over string = array is required to be a contiguous container and an aggregate

    MPI_File mpi_fh = TupleRead1stPassNExchange(filename, senddata, totsend, distmapper, totallength);

    typedef std::map<std::string, uint64_t> KEYMAP;    
    KEYMAP ultimateperm;	// the ultimate permutation
    for(IT i=0; i< totsend; ++i)	
    {
	    auto locnull = std::find(senddata[i].first.begin(), senddata[i].first.end(), '\0');
	    
	    std::string searchstr(senddata[i].first.begin(), locnull);
	    auto ret = ultimateperm.emplace(std::make_pair(searchstr, senddata[i].second));
	    if(!ret.second)	// the second is the boolean that tells success
	    {
	        // remember, we only sent unique vertex ids in the first place so we are expecting unique values in return		
		std::cout << "the duplication in ultimateperm is unexpected!!!" << std::endl;	
	    }
    }
    delete [] senddata;

    // rename the data now, first reset file pointers    
    MPI_Offset fpos, end_fpos;
    struct stat st;     // get file size
    if (stat(filename.c_str(), &st) == -1)
    {
        MPI_Abort(MPI_COMM_WORLD, NOFILE);
    }
    int64_t file_size = st.st_size;

    fpos = myrank * file_size / nprocs;   

    if(myrank != (nprocs-1)) end_fpos = (myrank + 1) * file_size / nprocs;
    else end_fpos = file_size;

    typedef typename DER::LocalIT LIT;
    std::vector<LIT> rows;
    std::vector<LIT> cols;
    std::vector<NT> vals;

    std::vector<std::string> lines;
    bool finished = SpParHelper::FetchBatch(mpi_fh, fpos, end_fpos, true, lines, myrank);
    int64_t entriesread = lines.size();
   
    SpHelper::ProcessStrLinesNPermute(rows, cols, vals, lines, ultimateperm);

    while(!finished)
    {
        finished = SpParHelper::FetchBatch(mpi_fh, fpos, end_fpos, false, lines, myrank);
        entriesread += lines.size();
    	SpHelper::ProcessStrLinesNPermute(rows, cols, vals, lines, ultimateperm);
    }
    int64_t allentriesread;
    MPI_Reduce(&entriesread, &allentriesread, 1, MPIType<int64_t>(), MPI_SUM, 0, commGrid->commWorld);
#ifdef COMBBLAS_DEBUG
    if(myrank == 0)
        std::cout << "Second reading finished. Total number of entries read across all processors is " << allentriesread << std::endl;
#endif

    MPI_File_close(&mpi_fh);
    std::vector< std::vector < std::tuple<LIT,LIT,NT> > > data(nprocs);
    
    LIT locsize = rows.size();   // remember: locsize != entriesread (unless the matrix is unsymmetric)
    for(LIT i=0; i<locsize; ++i)
    {
        LIT lrow, lcol;
        int owner = Owner(totallength, totallength, rows[i], cols[i], lrow, lcol);
        data[owner].push_back(std::make_tuple(lrow,lcol,vals[i]));
    }
    std::vector<LIT>().swap(rows);
    std::vector<LIT>().swap(cols);
    std::vector<NT>().swap(vals);	

#ifdef COMBBLAS_DEBUG
    if(myrank == 0)
        std::cout << "Packing to recipients finished, about to send..." << std::endl;
#endif
    
    if(spSeq)   delete spSeq;
    SparseCommon(data, locsize, totallength, totallength, BinOp);
    // PrintInfo();
    // distmapper.ParallelWrite("distmapper.mtx", 1, CharArraySaveHandler());
    return distmapper; 
}



//! Handles all sorts of orderings, even duplicates (what happens to them is determined by BinOp)
//! Requires proper matrix market banner at the moment
//! Replaces ReadDistribute for properly load balanced input in matrix market format
template <class IT, class NT, class DER>
template <typename _BinaryOperation>
void SpParMat< IT,NT,DER >::ParallelReadMM (const std::string & filename, bool onebased, _BinaryOperation BinOp)
{
    int32_t type = -1;
    int32_t symmetric = 0;
    int64_t nrows, ncols, nonzeros;
    int64_t linesread = 0;
    
    FILE *f;
    int myrank = commGrid->GetRank();
    int nprocs = commGrid->GetSize();
    if(myrank == 0)
    {
        MM_typecode matcode;
        if ((f = fopen(filename.c_str(), "r")) == NULL)
        {
            printf("COMBBLAS: Matrix-market file %s can not be found\n", filename.c_str());
            MPI_Abort(MPI_COMM_WORLD, NOFILE);
        }
        if (mm_read_banner(f, &matcode) != 0)
        {
            printf("Could not process Matrix Market banner.\n");
            exit(1);
        }
        linesread++;
        
        if (mm_is_complex(matcode))
        {
            printf("Sorry, this application does not support complext types");
            printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        }
        else if(mm_is_real(matcode))
        {
            std::cout << "Matrix is Float" << std::endl;
            type = 0;
        }
        else if(mm_is_integer(matcode))
        {
            std::cout << "Matrix is Integer" << std::endl;
            type = 1;
        }
        else if(mm_is_pattern(matcode))
        {
            std::cout << "Matrix is Boolean" << std::endl;
            type = 2;
        }
        if(mm_is_symmetric(matcode) || mm_is_hermitian(matcode))
        {
            std::cout << "Matrix is symmetric" << std::endl;
            symmetric = 1;
        }
        int ret_code;
        if ((ret_code = mm_read_mtx_crd_size(f, &nrows, &ncols, &nonzeros, &linesread)) !=0)  // ABAB: mm_read_mtx_crd_size made 64-bit friendly
            exit(1);
    
        std::cout << "Total number of nonzeros expected across all processors is " << nonzeros << std::endl;

    }
    MPI_Bcast(&type, 1, MPI_INT, 0, commGrid->commWorld);
    MPI_Bcast(&symmetric, 1, MPI_INT, 0, commGrid->commWorld);
    MPI_Bcast(&nrows, 1, MPIType<int64_t>(), 0, commGrid->commWorld);
    MPI_Bcast(&ncols, 1, MPIType<int64_t>(), 0, commGrid->commWorld);
    MPI_Bcast(&nonzeros, 1, MPIType<int64_t>(), 0, commGrid->commWorld);

    // Use fseek again to go backwards two bytes and check that byte with fgetc
    struct stat st;     // get file size
    if (stat(filename.c_str(), &st) == -1)
    {
        MPI_Abort(MPI_COMM_WORLD, NOFILE);
    }
    int64_t file_size = st.st_size;
    MPI_Offset fpos, end_fpos, endofheader;
    if(commGrid->GetRank() == 0)    // the offset needs to be for this rank
    {
        std::cout << "File is " << file_size << " bytes" << std::endl;
	fpos = ftell(f);
	endofheader =  fpos;
    	MPI_Bcast(&endofheader, 1, MPIType<MPI_Offset>(), 0, commGrid->commWorld);
        fclose(f);
    }
    else
    {
    	MPI_Bcast(&endofheader, 1, MPIType<MPI_Offset>(), 0, commGrid->commWorld);  // receive the file loc at the end of header
	fpos = endofheader + myrank * (file_size-endofheader) / nprocs;
    }
    if(myrank != (nprocs-1)) end_fpos = endofheader + (myrank + 1) * (file_size-endofheader) / nprocs;
    else end_fpos = file_size;

    MPI_File mpi_fh;
    MPI_File_open (commGrid->commWorld, const_cast<char*>(filename.c_str()), MPI_MODE_RDONLY, MPI_INFO_NULL, &mpi_fh);

	 
    typedef typename DER::LocalIT LIT;
    std::vector<LIT> rows;
    std::vector<LIT> cols;
    std::vector<NT> vals;

    std::vector<std::string> lines;
    bool finished = SpParHelper::FetchBatch(mpi_fh, fpos, end_fpos, true, lines, myrank);
    int64_t entriesread = lines.size();
    SpHelper::ProcessLines(rows, cols, vals, lines, symmetric, type, onebased);
    MPI_Barrier(commGrid->commWorld);

    while(!finished)
    {
        finished = SpParHelper::FetchBatch(mpi_fh, fpos, end_fpos, false, lines, myrank);
        entriesread += lines.size();
        SpHelper::ProcessLines(rows, cols, vals, lines, symmetric, type, onebased);
    }
    int64_t allentriesread;
    MPI_Reduce(&entriesread, &allentriesread, 1, MPIType<int64_t>(), MPI_SUM, 0, commGrid->commWorld);
#ifdef COMBBLAS_DEBUG
    if(myrank == 0)
        std::cout << "Reading finished. Total number of entries read across all processors is " << allentriesread << std::endl;
#endif

    std::vector< std::vector < std::tuple<LIT,LIT,NT> > > data(nprocs);
    
    LIT locsize = rows.size();   // remember: locsize != entriesread (unless the matrix is unsymmetric)
    for(LIT i=0; i<locsize; ++i)
    {
        LIT lrow, lcol;
        int owner = Owner(nrows, ncols, rows[i], cols[i], lrow, lcol);
        data[owner].push_back(std::make_tuple(lrow,lcol,vals[i]));
    }
    std::vector<LIT>().swap(rows);
    std::vector<LIT>().swap(cols);
    std::vector<NT>().swap(vals);	

#ifdef COMBBLAS_DEBUG
    if(myrank == 0)
        std::cout << "Packing to recepients finished, about to send..." << std::endl;
#endif
    
    if(spSeq)   delete spSeq;
    SparseCommon(data, locsize, nrows, ncols, BinOp);
}


template <class IT, class NT, class DER>
template <class HANDLER>
void SpParMat< IT,NT,DER >::ParallelWriteMM(const std::string & filename, bool onebased, HANDLER handler)
{
    int myrank = commGrid->GetRank();
    int nprocs = commGrid->GetSize();
    IT totalm = getnrow();
    IT totaln = getncol();
    IT totnnz = getnnz();

    std::stringstream ss;
    if(myrank == 0)
    {
        ss << "%%MatrixMarket matrix coordinate real general" << std::endl;
        ss << totalm << " " << totaln << " " << totnnz << std::endl;
    }
    
    IT entries =  getlocalnnz();
    IT sizeuntil = 0;
    MPI_Exscan( &entries, &sizeuntil, 1, MPIType<IT>(), MPI_SUM, commGrid->GetWorld() );
    if(myrank == 0) sizeuntil = 0;    // because MPI_Exscan says the recvbuf in process 0 is undefined
    
    IT roffset = 0;
    IT coffset = 0;
    GetPlaceInGlobalGrid(roffset, coffset);
    if(onebased)
    {
        roffset += 1;    // increment by 1
        coffset += 1;
    }
    
    for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)    // iterate over nonempty subcolumns
    {
        for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)
        {
            IT glrowid = nzit.rowid() + roffset;
            IT glcolid = colit.colid() + coffset;
            ss << glrowid << '\t';
            ss << glcolid << '\t';
            handler.save(ss, nzit.value(), glrowid, glcolid);
            ss << '\n';
        }
    }
    std::string text = ss.str();

    int64_t * bytes = new int64_t[nprocs];
    bytes[myrank] = text.size();
    MPI_Allgather(MPI_IN_PLACE, 1, MPIType<int64_t>(), bytes, 1, MPIType<int64_t>(), commGrid->GetWorld());
    int64_t bytesuntil = std::accumulate(bytes, bytes+myrank, static_cast<int64_t>(0));
    int64_t bytestotal = std::accumulate(bytes, bytes+nprocs, static_cast<int64_t>(0));


    MPI_File thefile;
    MPI_File_open(commGrid->GetWorld(), (char*) filename.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &thefile) ;
    int mpi_err = MPI_File_set_view(thefile, bytesuntil, MPI_CHAR, MPI_CHAR, (char*)"external32", MPI_INFO_NULL);
    if (mpi_err == 51) {
        // external32 datarep is not supported, use native instead
        MPI_File_set_view(thefile, bytesuntil, MPI_CHAR, MPI_CHAR, (char*)"native", MPI_INFO_NULL);
    }
 
    int64_t batchSize = 256 * 1024 * 1024;
    size_t localfileptr = 0;
    int64_t remaining = bytes[myrank];
    int64_t totalremaining = bytestotal;
    
    while(totalremaining > 0)
    {
    #ifdef COMBBLAS_DEBUG
        if(myrank == 0)
            std::cout << "Remaining " << totalremaining << " bytes to write in aggregate" << std::endl;
    #endif
        MPI_Status status;
        int curBatch = std::min(batchSize, remaining);
        MPI_File_write_all(thefile, text.c_str()+localfileptr, curBatch, MPI_CHAR, &status);
        int count;
        MPI_Get_count(&status, MPI_CHAR, &count); // known bug: https://github.com/pmodels/mpich/issues/2332
        assert( (curBatch == 0) || (count == curBatch) ); // count can return the previous/wrong value when 0 elements are written
        localfileptr += curBatch;
        remaining -= curBatch;
        MPI_Allreduce(&remaining, &totalremaining, 1, MPIType<int64_t>(), MPI_SUM, commGrid->GetWorld());
    }
    MPI_File_close(&thefile);
    
    delete [] bytes;
}



//! Handles all sorts of orderings as long as there are no duplicates
//! May perform better when the data is already reverse column-sorted (i.e. in decreasing order)
//! if nonum is true, then numerics are not supplied and they are assumed to be all 1's
template <class IT, class NT, class DER>
template <class HANDLER>
void SpParMat< IT,NT,DER >::ReadDistribute (const std::string & filename, int master, bool nonum, HANDLER handler, bool transpose, bool pario)
{
#ifdef TAU_PROFILE
   	TAU_PROFILE_TIMER(rdtimer, "ReadDistribute", "void SpParMat::ReadDistribute (const string & , int, bool, HANDLER, bool)", TAU_DEFAULT);
   	TAU_PROFILE_START(rdtimer);
#endif

	std::ifstream infile;
	FILE * binfile = NULL;	// points to "past header" if the file is binary
	int seeklength = 0;
	HeaderInfo hfile;
	if(commGrid->GetRank() == master)	// 1 processor
	{
		hfile = ParseHeader(filename, binfile, seeklength);
	}
	MPI_Bcast(&seeklength, 1, MPI_INT, master, commGrid->commWorld);

	IT total_m, total_n, total_nnz;
	IT m_perproc = 0, n_perproc = 0;

	int colneighs = commGrid->GetGridRows();	// number of neighbors along this processor column (including oneself)
	int rowneighs = commGrid->GetGridCols();	// number of neighbors along this processor row (including oneself)

	IT buffpercolneigh = MEMORYINBYTES / (colneighs * (2 * sizeof(IT) + sizeof(NT)));
	IT buffperrowneigh = MEMORYINBYTES / (rowneighs * (2 * sizeof(IT) + sizeof(NT)));
	if(pario)
	{
		// since all colneighs will be reading the data at the same time
		// chances are they might all read the data that should go to one
		// in that case buffperrowneigh > colneighs * buffpercolneigh 
		// in order not to overflow
		buffpercolneigh /= colneighs; 
		if(seeklength == 0)
			SpParHelper::Print("COMBBLAS: Parallel I/O requested but binary header is corrupted\n", commGrid->GetWorld());
	}

	// make sure that buffperrowneigh >= buffpercolneigh to cover for this patological case:
	//   	-- all data received by a given column head (by vertical communication) are headed to a single processor along the row
	//   	-- then making sure buffperrowneigh >= buffpercolneigh guarantees that the horizontal buffer will never overflow
	buffperrowneigh = std::max(buffperrowneigh, buffpercolneigh);
	if(std::max(buffpercolneigh * colneighs, buffperrowneigh * rowneighs) > std::numeric_limits<int>::max())
	{  
		SpParHelper::Print("COMBBLAS: MPI doesn't support sending int64_t send/recv counts or displacements\n", commGrid->GetWorld());
	}
 
	int * cdispls = new int[colneighs];
	for (IT i=0; i<colneighs; ++i)  cdispls[i] = i*buffpercolneigh;
	int * rdispls = new int[rowneighs];
	for (IT i=0; i<rowneighs; ++i)  rdispls[i] = i*buffperrowneigh;		

	int *ccurptrs = NULL, *rcurptrs = NULL;	
	int recvcount = 0;
	IT * rows = NULL; 
	IT * cols = NULL;
	NT * vals = NULL;

	// Note: all other column heads that initiate the horizontal communication has the same "rankinrow" with the master
	int rankincol = commGrid->GetRankInProcCol(master);	// get master's rank in its processor column
	int rankinrow = commGrid->GetRankInProcRow(master);	
	std::vector< std::tuple<IT, IT, NT> > localtuples;

	if(commGrid->GetRank() == master)	// 1 processor
	{		
		if( !hfile.fileexists )
		{
			SpParHelper::Print( "COMBBLAS: Input file doesn't exist\n", commGrid->GetWorld());
			total_n = 0; total_m = 0;	
			BcastEssentials(commGrid->commWorld, total_m, total_n, total_nnz, master);
			return;
		}
		if (hfile.headerexists && hfile.format == 1) 
		{
			SpParHelper::Print("COMBBLAS: Ascii input with binary headers is not supported\n", commGrid->GetWorld());
			total_n = 0; total_m = 0;	
			BcastEssentials(commGrid->commWorld, total_m, total_n, total_nnz, master);
			return;
		}
		if ( !hfile.headerexists )	// no header - ascii file (at this point, file exists)
		{
			infile.open(filename.c_str());
			char comment[256];
			infile.getline(comment,256);
			while(comment[0] == '%')
			{
				infile.getline(comment,256);
			}
			std::stringstream ss;
			ss << std::string(comment);
			ss >> total_m >> total_n >> total_nnz;
			if(pario)
			{
				SpParHelper::Print("COMBBLAS: Trying to read binary headerless file in parallel, aborting\n", commGrid->GetWorld());
				total_n = 0; total_m = 0;	
				BcastEssentials(commGrid->commWorld, total_m, total_n, total_nnz, master);
				return;				
			}
		}
		else // hfile.headerexists && hfile.format == 0
		{
			total_m = hfile.m;
			total_n = hfile.n;
			total_nnz = hfile.nnz;
		}
		m_perproc = total_m / colneighs;
		n_perproc = total_n / rowneighs;
		BcastEssentials(commGrid->commWorld, total_m, total_n, total_nnz, master);
		AllocateSetBuffers(rows, cols, vals,  rcurptrs, ccurptrs, rowneighs, colneighs, buffpercolneigh);

		if(seeklength > 0 && pario)   // sqrt(p) processors also do parallel binary i/o
		{
			IT entriestoread =  total_nnz / colneighs;
			#ifdef IODEBUG
            std::ofstream oput;
			commGrid->OpenDebugFile("Read", oput);
			oput << "Total nnz: " << total_nnz << " entries to read: " << entriestoread << std::endl;
			oput.close();
			#endif
			ReadAllMine(binfile, rows, cols, vals, localtuples, rcurptrs, ccurptrs, rdispls, cdispls, m_perproc, n_perproc, 
				rowneighs, colneighs, buffperrowneigh, buffpercolneigh, entriestoread, handler, rankinrow, transpose);
		}
		else	// only this (master) is doing I/O (text or binary)
		{
			IT temprow, tempcol;
			NT tempval;	
			IT ntrow = 0, ntcol = 0; // not transposed row and column index
			char line[1024];
			bool nonumline = nonum;
			IT cnz = 0;
			for(; cnz < total_nnz; ++cnz)
			{	
				int colrec;
				size_t commonindex;
				std::stringstream linestream;
				if( (!hfile.headerexists) && (!infile.eof()))
				{
					// read one line at a time so that missing numerical values can be detected
					infile.getline(line, 1024);
					linestream << line;
					linestream >> temprow >> tempcol;
					if (!nonum)
					{
						// see if this line has a value
						linestream >> std::skipws;
						nonumline = linestream.eof();
					}
					--temprow;	// file is 1-based where C-arrays are 0-based
					--tempcol;
					ntrow = temprow;
					ntcol = tempcol;
				}
				else if(hfile.headerexists && (!feof(binfile)) ) 
				{
					handler.binaryfill(binfile, temprow , tempcol, tempval);
				}
				if (transpose)
				{
					IT swap = temprow;
					temprow = tempcol;
					tempcol = swap;
				}
				colrec = std::min(static_cast<int>(temprow / m_perproc), colneighs-1);	// precipient processor along the column
				commonindex = colrec * buffpercolneigh + ccurptrs[colrec];
					
				rows[ commonindex ] = temprow;
				cols[ commonindex ] = tempcol;
				if( (!hfile.headerexists) && (!infile.eof()))
				{
					vals[ commonindex ] = nonumline ? handler.getNoNum(ntrow, ntcol) : handler.read(linestream, ntrow, ntcol); //tempval;
				}
				else if(hfile.headerexists && (!feof(binfile)) ) 
				{
					vals[ commonindex ] = tempval;
				}
				++ (ccurptrs[colrec]);				
				if(ccurptrs[colrec] == buffpercolneigh || (cnz == (total_nnz-1)) )		// one buffer is full, or file is done !
				{
					MPI_Scatter(ccurptrs, 1, MPI_INT, &recvcount, 1, MPI_INT, rankincol, commGrid->colWorld); // first, send the receive counts

					// generate space for own recv data ... (use arrays because vector<bool> is cripled, if NT=bool)
					IT * temprows = new IT[recvcount];
					IT * tempcols = new IT[recvcount];
					NT * tempvals = new NT[recvcount];
					
					// then, send all buffers that to their recipients ...
					MPI_Scatterv(rows, ccurptrs, cdispls, MPIType<IT>(), temprows, recvcount,  MPIType<IT>(), rankincol, commGrid->colWorld);
					MPI_Scatterv(cols, ccurptrs, cdispls, MPIType<IT>(), tempcols, recvcount,  MPIType<IT>(), rankincol, commGrid->colWorld);
					MPI_Scatterv(vals, ccurptrs, cdispls, MPIType<NT>(), tempvals, recvcount,  MPIType<NT>(), rankincol, commGrid->colWorld);

					std::fill_n(ccurptrs, colneighs, 0);  				// finally, reset current pointers !
					DeleteAll(rows, cols, vals);
					
					HorizontalSend(rows, cols, vals,temprows, tempcols, tempvals, localtuples, rcurptrs, rdispls, 
							buffperrowneigh, rowneighs, recvcount, m_perproc, n_perproc, rankinrow);
					
					if( cnz != (total_nnz-1) )	// otherwise the loop will exit with noone to claim memory back
					{
						// reuse these buffers for the next vertical communication								
						rows = new IT [ buffpercolneigh * colneighs ];
						cols = new IT [ buffpercolneigh * colneighs ];
						vals = new NT [ buffpercolneigh * colneighs ];
					}
				} // end_if for "send buffer is full" case 
			} // end_for for "cnz < entriestoread" case
			assert (cnz == total_nnz);
			
			// Signal the end of file to other processors along the column
			std::fill_n(ccurptrs, colneighs, std::numeric_limits<int>::max());	
			MPI_Scatter(ccurptrs, 1, MPI_INT, &recvcount, 1, MPI_INT, rankincol, commGrid->colWorld);

			// And along the row ...
			std::fill_n(rcurptrs, rowneighs, std::numeric_limits<int>::max());				
			MPI_Scatter(rcurptrs, 1, MPI_INT, &recvcount, 1, MPI_INT, rankinrow, commGrid->rowWorld);
		}	// end of "else" (only one processor reads) block
	}	// end_if for "master processor" case
	else if( commGrid->OnSameProcCol(master) ) 	// (r-1) processors
	{
		BcastEssentials(commGrid->commWorld, total_m, total_n, total_nnz, master);
		m_perproc = total_m / colneighs;
		n_perproc = total_n / rowneighs;

		if(seeklength > 0 && pario)   // these processors also do parallel binary i/o
		{
			binfile = fopen(filename.c_str(), "rb");
			IT entrysize = handler.entrylength();
			int myrankincol = commGrid->GetRankInProcCol();
			IT perreader = total_nnz / colneighs;
			IT read_offset = entrysize * static_cast<IT>(myrankincol) * perreader + seeklength;
			IT entriestoread = perreader;
			if (myrankincol == colneighs-1) 
				entriestoread = total_nnz - static_cast<IT>(myrankincol) * perreader;
			fseek(binfile, read_offset, SEEK_SET);

			#ifdef IODEBUG
            std::ofstream oput;
			commGrid->OpenDebugFile("Read", oput);
			oput << "Total nnz: " << total_nnz << " OFFSET : " << read_offset << " entries to read: " << entriestoread << std::endl;
			oput.close();
			#endif
			
			AllocateSetBuffers(rows, cols, vals,  rcurptrs, ccurptrs, rowneighs, colneighs, buffpercolneigh);
			ReadAllMine(binfile, rows, cols, vals, localtuples, rcurptrs, ccurptrs, rdispls, cdispls, m_perproc, n_perproc, 
				rowneighs, colneighs, buffperrowneigh, buffpercolneigh, entriestoread, handler, rankinrow, transpose);
		}
		else // only master does the I/O
		{
			while(total_n > 0 || total_m > 0)	// otherwise input file does not exist !
			{
				// void MPI::Comm::Scatterv(const void* sendbuf, const int sendcounts[], const int displs[], const MPI::Datatype& sendtype,
				//				void* recvbuf, int recvcount, const MPI::Datatype & recvtype, int root) const
				// The outcome is as if the root executed n send operations,
				//	MPI_Send(sendbuf + displs[i] * extent(sendtype), sendcounts[i], sendtype, i, ...)
				// and each process executed a receive,
				// 	MPI_Recv(recvbuf, recvcount, recvtype, root, ...)
				// The send buffer is ignored for all nonroot processes.
				
				MPI_Scatter(ccurptrs, 1, MPI_INT, &recvcount, 1, MPI_INT, rankincol, commGrid->colWorld);                       // first receive the receive counts ...
				if( recvcount == std::numeric_limits<int>::max()) break;
				
				// create space for incoming data ... 
				IT * temprows = new IT[recvcount];
				IT * tempcols = new IT[recvcount];
				NT * tempvals = new NT[recvcount];
				
				// receive actual data ... (first 4 arguments are ignored in the receiver side)
				MPI_Scatterv(rows, ccurptrs, cdispls, MPIType<IT>(), temprows, recvcount,  MPIType<IT>(), rankincol, commGrid->colWorld);
				MPI_Scatterv(cols, ccurptrs, cdispls, MPIType<IT>(), tempcols, recvcount,  MPIType<IT>(), rankincol, commGrid->colWorld);
				MPI_Scatterv(vals, ccurptrs, cdispls, MPIType<NT>(), tempvals, recvcount,  MPIType<NT>(), rankincol, commGrid->colWorld);

				// now, send the data along the horizontal
				rcurptrs = new int[rowneighs];
				std::fill_n(rcurptrs, rowneighs, 0);	
				
				// HorizontalSend frees the memory of temp_xxx arrays and then creates and frees memory of all the six arrays itself
				HorizontalSend(rows, cols, vals,temprows, tempcols, tempvals, localtuples, rcurptrs, rdispls, 
					buffperrowneigh, rowneighs, recvcount, m_perproc, n_perproc, rankinrow);
			}
		}
		
		// Signal the end of file to other processors along the row
		std::fill_n(rcurptrs, rowneighs, std::numeric_limits<int>::max());				
		MPI_Scatter(rcurptrs, 1, MPI_INT, &recvcount, 1, MPI_INT, rankinrow, commGrid->rowWorld);
		delete [] rcurptrs;	
	}
	else		// r * (s-1) processors that only participate in the horizontal communication step
	{
		BcastEssentials(commGrid->commWorld, total_m, total_n, total_nnz, master);
        
		m_perproc = total_m / colneighs;
		n_perproc = total_n / rowneighs;
		while(total_n > 0 || total_m > 0)	// otherwise input file does not exist !
		{
			// receive the receive count
			MPI_Scatter(rcurptrs, 1, MPI_INT, &recvcount, 1, MPI_INT, rankinrow, commGrid->rowWorld);
			if( recvcount == std::numeric_limits<int>::max())
				break;

            #ifdef IODEBUG
            std::ofstream oput;
            commGrid->OpenDebugFile("Read", oput);
            oput << "Total nnz: " << total_nnz << " total_m : " << total_m << " recvcount: " << recvcount << std::endl;
            oput.close();
            #endif
            
			// create space for incoming data ... 
			IT * temprows = new IT[recvcount];
			IT * tempcols = new IT[recvcount];
			NT * tempvals = new NT[recvcount];

			MPI_Scatterv(rows, rcurptrs, rdispls, MPIType<IT>(), temprows, recvcount,  MPIType<IT>(), rankinrow, commGrid->rowWorld);
			MPI_Scatterv(cols, rcurptrs, rdispls, MPIType<IT>(), tempcols, recvcount,  MPIType<IT>(), rankinrow, commGrid->rowWorld);
			MPI_Scatterv(vals, rcurptrs, rdispls, MPIType<NT>(), tempvals, recvcount,  MPIType<NT>(), rankinrow, commGrid->rowWorld);

			// now push what is ours to tuples
			IT moffset = commGrid->myprocrow * m_perproc; 
			IT noffset = commGrid->myproccol * n_perproc;
			
			for(IT i=0; i< recvcount; ++i)
			{					
				localtuples.push_back( 	std::make_tuple(temprows[i]-moffset, tempcols[i]-noffset, tempvals[i]) );
			}
			DeleteAll(temprows,tempcols,tempvals);
		}
	}
	DeleteAll(cdispls, rdispls);
	std::tuple<IT,IT,NT> * arrtuples = new std::tuple<IT,IT,NT>[localtuples.size()];  // the vector will go out of scope, make it stick !
  std::copy(localtuples.begin(), localtuples.end(), arrtuples);

 	IT localm = (commGrid->myprocrow != (commGrid->grrows-1))? m_perproc: (total_m - (m_perproc * (commGrid->grrows-1)));
 	IT localn = (commGrid->myproccol != (commGrid->grcols-1))? n_perproc: (total_n - (n_perproc * (commGrid->grcols-1)));
	spSeq->Create( localtuples.size(), localm, localn, arrtuples);		// the deletion of arrtuples[] is handled by SpMat::Create

#ifdef TAU_PROFILE
   	TAU_PROFILE_STOP(rdtimer);
#endif
	return;
}

template <class IT, class NT, class DER>
void SpParMat<IT,NT,DER>::AllocateSetBuffers(IT * & rows, IT * & cols, NT * & vals,  int * & rcurptrs, int * & ccurptrs, int rowneighs, int colneighs, IT buffpercolneigh)
{
	// allocate buffers on the heap as stack space is usually limited
	rows = new IT [ buffpercolneigh * colneighs ];
	cols = new IT [ buffpercolneigh * colneighs ];
	vals = new NT [ buffpercolneigh * colneighs ];
	
	ccurptrs = new int[colneighs];
	rcurptrs = new int[rowneighs];
	std::fill_n(ccurptrs, colneighs, 0);	// fill with zero
	std::fill_n(rcurptrs, rowneighs, 0);	
}

template <class IT, class NT, class DER>
void SpParMat<IT,NT,DER>::BcastEssentials(MPI_Comm & world, IT & total_m, IT & total_n, IT & total_nnz, int master)
{
	MPI_Bcast(&total_m, 1, MPIType<IT>(), master, world);
	MPI_Bcast(&total_n, 1, MPIType<IT>(), master, world);
	MPI_Bcast(&total_nnz, 1, MPIType<IT>(), master, world);
}
	
/*
 * @post {rows, cols, vals are pre-allocated on the heap after this call} 
 * @post {ccurptrs are set to zero; so that if another call is made to this function without modifying ccurptrs, no data will be send from this procesor}
 */
template <class IT, class NT, class DER>
void SpParMat<IT,NT,DER>::VerticalSend(IT * & rows, IT * & cols, NT * & vals, std::vector< std::tuple<IT,IT,NT> > & localtuples, int * rcurptrs, int * ccurptrs, int * rdispls, int * cdispls, 
				  IT m_perproc, IT n_perproc, int rowneighs, int colneighs, IT buffperrowneigh, IT buffpercolneigh, int rankinrow)
{
	// first, send/recv the counts ...
	int * colrecvdispls = new int[colneighs];
	int * colrecvcounts = new int[colneighs];
	MPI_Alltoall(ccurptrs, 1, MPI_INT, colrecvcounts, 1, MPI_INT, commGrid->colWorld);      // share the request counts
	int totrecv = std::accumulate(colrecvcounts,colrecvcounts+colneighs,0);	
	colrecvdispls[0] = 0; 		// receive displacements are exact whereas send displacements have slack
	for(int i=0; i<colneighs-1; ++i)
		colrecvdispls[i+1] = colrecvdispls[i] + colrecvcounts[i];
	
	// generate space for own recv data ... (use arrays because vector<bool> is cripled, if NT=bool)
	IT * temprows = new IT[totrecv];
	IT * tempcols = new IT[totrecv];
	NT * tempvals = new NT[totrecv];
	
	// then, exchange all buffers that to their recipients ...
	MPI_Alltoallv(rows, ccurptrs, cdispls, MPIType<IT>(), temprows, colrecvcounts, colrecvdispls, MPIType<IT>(), commGrid->colWorld);
	MPI_Alltoallv(cols, ccurptrs, cdispls, MPIType<IT>(), tempcols, colrecvcounts, colrecvdispls, MPIType<IT>(), commGrid->colWorld);
	MPI_Alltoallv(vals, ccurptrs, cdispls, MPIType<NT>(), tempvals, colrecvcounts, colrecvdispls, MPIType<NT>(), commGrid->colWorld);

	// finally, reset current pointers !
	std::fill_n(ccurptrs, colneighs, 0);
	DeleteAll(colrecvdispls, colrecvcounts);
	DeleteAll(rows, cols, vals);
	
	// rcurptrs/rdispls are zero initialized scratch space
	HorizontalSend(rows, cols, vals,temprows, tempcols, tempvals, localtuples, rcurptrs, rdispls, buffperrowneigh, rowneighs, totrecv, m_perproc, n_perproc, rankinrow);
	
	// reuse these buffers for the next vertical communication								
	rows = new IT [ buffpercolneigh * colneighs ];
	cols = new IT [ buffpercolneigh * colneighs ];
	vals = new NT [ buffpercolneigh * colneighs ];
}


/**
 * Private subroutine of ReadDistribute. 
 * Executed by p_r processors on the first processor column. 
 * @pre {rows, cols, vals are pre-allocated on the heap before this call} 
 * @param[in] rankinrow {row head's rank in its processor row - determines the scatter person} 
 */
template <class IT, class NT, class DER>
template <class HANDLER>
void SpParMat<IT,NT,DER>::ReadAllMine(FILE * binfile, IT * & rows, IT * & cols, NT * & vals, std::vector< std::tuple<IT,IT,NT> > & localtuples, int * rcurptrs, int * ccurptrs, int * rdispls, int * cdispls, 
		IT m_perproc, IT n_perproc, int rowneighs, int colneighs, IT buffperrowneigh, IT buffpercolneigh, IT entriestoread, HANDLER handler, int rankinrow, bool transpose)
{
	assert(entriestoread != 0);
	IT cnz = 0;
	IT temprow, tempcol;
	NT tempval;
	int finishedglobal = 1;
	while(cnz < entriestoread && !feof(binfile))	// this loop will execute at least once
	{
		handler.binaryfill(binfile, temprow , tempcol, tempval);
        
		if (transpose)
		{
			IT swap = temprow;
			temprow = tempcol;
			tempcol = swap;
		}
		int colrec = std::min(static_cast<int>(temprow / m_perproc), colneighs-1);	// precipient processor along the column
		size_t commonindex = colrec * buffpercolneigh + ccurptrs[colrec];
		rows[ commonindex ] = temprow;
		cols[ commonindex ] = tempcol;
		vals[ commonindex ] = tempval;
		++ (ccurptrs[colrec]);	
		if(ccurptrs[colrec] == buffpercolneigh || (cnz == (entriestoread-1)) )		// one buffer is full, or this processor's share is done !
		{			
			#ifdef IODEBUG
            std::ofstream oput;
			commGrid->OpenDebugFile("Read", oput);
			oput << "To column neighbors: ";
            std::copy(ccurptrs, ccurptrs+colneighs, std::ostream_iterator<int>(oput, " ")); oput << std::endl;
			oput.close();
			#endif

			VerticalSend(rows, cols, vals, localtuples, rcurptrs, ccurptrs, rdispls, cdispls, m_perproc, n_perproc, 
					rowneighs, colneighs, buffperrowneigh, buffpercolneigh, rankinrow);

			if(cnz == (entriestoread-1))	// last execution of the outer loop
			{
				int finishedlocal = 1;	// I am done, but let me check others 
				MPI_Allreduce( &finishedlocal, &finishedglobal, 1, MPI_INT, MPI_BAND, commGrid->colWorld);
				while(!finishedglobal)
				{
					#ifdef IODEBUG
                    std::ofstream oput;
					commGrid->OpenDebugFile("Read", oput);
					oput << "To column neighbors: ";
                    std::copy(ccurptrs, ccurptrs+colneighs, std::ostream_iterator<int>(oput, " ")); oput << std::endl;
					oput.close();
					#endif

					// postcondition of VerticalSend: ccurptrs are set to zero
					// if another call is made to this function without modifying ccurptrs, no data will be send from this procesor
					VerticalSend(rows, cols, vals, localtuples, rcurptrs, ccurptrs, rdispls, cdispls, m_perproc, n_perproc, 
						rowneighs, colneighs, buffperrowneigh, buffpercolneigh, rankinrow);

					MPI_Allreduce( &finishedlocal, &finishedglobal, 1, MPI_INT, MPI_BAND, commGrid->colWorld);
				}
			}
			else // the other loop will continue executing
			{
				int finishedlocal = 0;
				MPI_Allreduce( &finishedlocal, &finishedglobal, 1, MPI_INT, MPI_BAND, commGrid->colWorld);
			}
		} // end_if for "send buffer is full" case 
		++cnz;
	}

	// signal the end to row neighbors
	std::fill_n(rcurptrs, rowneighs, std::numeric_limits<int>::max());				
	int recvcount;
	MPI_Scatter(rcurptrs, 1, MPI_INT, &recvcount, 1, MPI_INT, rankinrow, commGrid->rowWorld);
}


/**
 * Private subroutine of ReadDistribute
 * @param[in] rankinrow {Row head's rank in its processor row}
 * Initially temp_xxx arrays carry data received along the proc. column AND needs to be sent along the proc. row
 * After usage, function frees the memory of temp_xxx arrays and then creates and frees memory of all the six arrays itself
 */
template <class IT, class NT, class DER>
void SpParMat<IT,NT,DER>::HorizontalSend(IT * & rows, IT * & cols, NT * & vals, IT * & temprows, IT * & tempcols, NT * & tempvals, std::vector < std::tuple <IT,IT,NT> > & localtuples, 
					 int * rcurptrs, int * rdispls, IT buffperrowneigh, int rowneighs, int recvcount, IT m_perproc, IT n_perproc, int rankinrow)
{	
	rows = new IT [ buffperrowneigh * rowneighs ];
	cols = new IT [ buffperrowneigh * rowneighs ];
	vals = new NT [ buffperrowneigh * rowneighs ];
	
	// prepare to send the data along the horizontal
	for(int i=0; i< recvcount; ++i)
	{
		int rowrec = std::min(static_cast<int>(tempcols[i] / n_perproc), rowneighs-1);
		rows[ rowrec * buffperrowneigh + rcurptrs[rowrec] ] = temprows[i];
		cols[ rowrec * buffperrowneigh + rcurptrs[rowrec] ] = tempcols[i];
		vals[ rowrec * buffperrowneigh + rcurptrs[rowrec] ] = tempvals[i];
		++ (rcurptrs[rowrec]);	
	}

	#ifdef IODEBUG
    std::ofstream oput;
	commGrid->OpenDebugFile("Read", oput);
	oput << "To row neighbors: ";
    std::copy(rcurptrs, rcurptrs+rowneighs, std::ostream_iterator<int>(oput, " ")); oput << std::endl;
	oput << "Row displacements were: ";
    std::copy(rdispls, rdispls+rowneighs, std::ostream_iterator<int>(oput, " ")); oput << std::endl;
	oput.close();
	#endif

	MPI_Scatter(rcurptrs, 1, MPI_INT, &recvcount, 1, MPI_INT, rankinrow, commGrid->rowWorld); // Send the receive counts for horizontal communication	

	// the data is now stored in rows/cols/vals, can reset temporaries
	// sets size and capacity to new recvcount
	DeleteAll(temprows, tempcols, tempvals);
	temprows = new IT[recvcount];
	tempcols = new IT[recvcount];
	tempvals = new NT[recvcount];
	
	// then, send all buffers that to their recipients ...
	MPI_Scatterv(rows, rcurptrs, rdispls, MPIType<IT>(), temprows, recvcount,  MPIType<IT>(), rankinrow, commGrid->rowWorld);
	MPI_Scatterv(cols, rcurptrs, rdispls, MPIType<IT>(), tempcols, recvcount,  MPIType<IT>(), rankinrow, commGrid->rowWorld);
	MPI_Scatterv(vals, rcurptrs, rdispls, MPIType<NT>(), tempvals, recvcount,  MPIType<NT>(), rankinrow, commGrid->rowWorld);

	// now push what is ours to tuples
	IT moffset = commGrid->myprocrow * m_perproc; 
	IT noffset = commGrid->myproccol * n_perproc; 
	
	for(int i=0; i< recvcount; ++i)
	{					
		localtuples.push_back( 	std::make_tuple(temprows[i]-moffset, tempcols[i]-noffset, tempvals[i]) );
	}
	
	std::fill_n(rcurptrs, rowneighs, 0);
	DeleteAll(rows, cols, vals, temprows, tempcols, tempvals);		
}


//! The input parameters' identity (zero) elements as well as 
//! their communication grid is preserved while outputting
template <class IT, class NT, class DER>
void SpParMat<IT,NT,DER>::Find (FullyDistVec<IT,IT> & distrows, FullyDistVec<IT,IT> & distcols, FullyDistVec<IT,NT> & distvals) const
{
	if((*(distrows.commGrid) != *(distcols.commGrid)) || (*(distcols.commGrid) != *(distvals.commGrid)))
	{
		SpParHelper::Print("Grids are not comparable, Find() fails!", commGrid->GetWorld());
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
	}
	IT globallen = getnnz();
	SpTuples<IT,NT> Atuples(*spSeq);
	
	FullyDistVec<IT,IT> nrows ( distrows.commGrid, globallen, 0); 
	FullyDistVec<IT,IT> ncols ( distcols.commGrid, globallen, 0); 
	FullyDistVec<IT,NT> nvals ( distvals.commGrid, globallen, NT()); 
	
	IT prelen = Atuples.getnnz();
	//IT postlen = nrows.MyLocLength();

	int rank = commGrid->GetRank();
	int nprocs = commGrid->GetSize();
	IT * prelens = new IT[nprocs];
	prelens[rank] = prelen;
	MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IT>(), prelens, 1, MPIType<IT>(), commGrid->GetWorld());
	IT prelenuntil = std::accumulate(prelens, prelens+rank, static_cast<IT>(0));

	int * sendcnt = new int[nprocs]();	// zero initialize
	IT * rows = new IT[prelen];
	IT * cols = new IT[prelen];
	NT * vals = new NT[prelen];

	int rowrank = commGrid->GetRankInProcRow();
	int colrank = commGrid->GetRankInProcCol(); 
	int rowneighs = commGrid->GetGridCols();
	int colneighs = commGrid->GetGridRows();
	IT * locnrows = new IT[colneighs];	// number of rows is calculated by a reduction among the processor column
	IT * locncols = new IT[rowneighs];
	locnrows[colrank] = getlocalrows();
	locncols[rowrank] = getlocalcols();

	MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IT>(),locnrows, 1, MPIType<IT>(), commGrid->GetColWorld());
	MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IT>(),locncols, 1, MPIType<IT>(), commGrid->GetRowWorld());

	IT roffset = std::accumulate(locnrows, locnrows+colrank, static_cast<IT>(0));
	IT coffset = std::accumulate(locncols, locncols+rowrank, static_cast<IT>(0));
	
	DeleteAll(locnrows, locncols);
	for(int i=0; i< prelen; ++i)
	{
		IT locid;	// ignore local id, data will come in order
		int owner = nrows.Owner(prelenuntil+i, locid);
		sendcnt[owner]++;

		rows[i] = Atuples.rowindex(i) + roffset;	// need the global row index
		cols[i] = Atuples.colindex(i) + coffset;	// need the global col index
		vals[i] = Atuples.numvalue(i);
	}

	int * recvcnt = new int[nprocs];
	MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, commGrid->GetWorld());   // get the recv counts

	int * sdpls = new int[nprocs]();	// displacements (zero initialized pid) 
	int * rdpls = new int[nprocs](); 
	std::partial_sum(sendcnt, sendcnt+nprocs-1, sdpls+1);
	std::partial_sum(recvcnt, recvcnt+nprocs-1, rdpls+1);

	MPI_Alltoallv(rows, sendcnt, sdpls, MPIType<IT>(), SpHelper::p2a(nrows.arr), recvcnt, rdpls, MPIType<IT>(), commGrid->GetWorld());
	MPI_Alltoallv(cols, sendcnt, sdpls, MPIType<IT>(), SpHelper::p2a(ncols.arr), recvcnt, rdpls, MPIType<IT>(), commGrid->GetWorld());
	MPI_Alltoallv(vals, sendcnt, sdpls, MPIType<NT>(), SpHelper::p2a(nvals.arr), recvcnt, rdpls, MPIType<NT>(), commGrid->GetWorld());

	DeleteAll(sendcnt, recvcnt, sdpls, rdpls);
	DeleteAll(prelens, rows, cols, vals);
	distrows = nrows;
	distcols = ncols;
	distvals = nvals;
}

//! The input parameters' identity (zero) elements as well as 
//! their communication grid is preserved while outputting
template <class IT, class NT, class DER>
void SpParMat<IT,NT,DER>::Find (FullyDistVec<IT,IT> & distrows, FullyDistVec<IT,IT> & distcols) const
{
	if((*(distrows.commGrid) != *(distcols.commGrid)) )
	{
		SpParHelper::Print("Grids are not comparable, Find() fails!", commGrid->GetWorld());
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
	}
	IT globallen = getnnz();
	SpTuples<IT,NT> Atuples(*spSeq);
	
	FullyDistVec<IT,IT> nrows ( distrows.commGrid, globallen, 0); 
	FullyDistVec<IT,IT> ncols ( distcols.commGrid, globallen, 0); 
	
	IT prelen = Atuples.getnnz();

	int rank = commGrid->GetRank();
	int nprocs = commGrid->GetSize();
	IT * prelens = new IT[nprocs];
	prelens[rank] = prelen;
	MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IT>(), prelens, 1, MPIType<IT>(), commGrid->GetWorld());
	IT prelenuntil = std::accumulate(prelens, prelens+rank, static_cast<IT>(0));

	int * sendcnt = new int[nprocs]();	// zero initialize
	IT * rows = new IT[prelen];
	IT * cols = new IT[prelen];
	NT * vals = new NT[prelen];

	int rowrank = commGrid->GetRankInProcRow();
	int colrank = commGrid->GetRankInProcCol(); 
	int rowneighs = commGrid->GetGridCols();
	int colneighs = commGrid->GetGridRows();
	IT * locnrows = new IT[colneighs];	// number of rows is calculated by a reduction among the processor column
	IT * locncols = new IT[rowneighs];
	locnrows[colrank] = getlocalrows();
	locncols[rowrank] = getlocalcols();

	MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IT>(),locnrows, 1, MPIType<IT>(), commGrid->GetColWorld());
	MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IT>(),locncols, 1, MPIType<IT>(), commGrid->GetColWorld());
	IT roffset = std::accumulate(locnrows, locnrows+colrank, static_cast<IT>(0));
	IT coffset = std::accumulate(locncols, locncols+rowrank, static_cast<IT>(0));
	
	DeleteAll(locnrows, locncols);
	for(int i=0; i< prelen; ++i)
	{
		IT locid;	// ignore local id, data will come in order
		int owner = nrows.Owner(prelenuntil+i, locid);
		sendcnt[owner]++;

		rows[i] = Atuples.rowindex(i) + roffset;	// need the global row index
		cols[i] = Atuples.colindex(i) + coffset;	// need the global col index
	}

	int * recvcnt = new int[nprocs];
	MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, commGrid->GetWorld());   // get the recv counts

	int * sdpls = new int[nprocs]();	// displacements (zero initialized pid) 
	int * rdpls = new int[nprocs](); 
	std::partial_sum(sendcnt, sendcnt+nprocs-1, sdpls+1);
	std::partial_sum(recvcnt, recvcnt+nprocs-1, rdpls+1);

	MPI_Alltoallv(rows, sendcnt, sdpls, MPIType<IT>(), SpHelper::p2a(nrows.arr), recvcnt, rdpls, MPIType<IT>(), commGrid->GetWorld());
	MPI_Alltoallv(cols, sendcnt, sdpls, MPIType<IT>(), SpHelper::p2a(ncols.arr), recvcnt, rdpls, MPIType<IT>(), commGrid->GetWorld());

	DeleteAll(sendcnt, recvcnt, sdpls, rdpls);
	DeleteAll(prelens, rows, cols, vals);
	distrows = nrows;
	distcols = ncols;
}

template <class IT, class NT, class DER>
DER SpParMat<IT,NT,DER>::InducedSubgraphs2Procs(const FullyDistVec<IT,IT>& Assignments, std::vector<IT>& LocalIdxs) const
{
    int nprocs = commGrid->GetSize();
    int myrank = commGrid->GetRank();
    int nverts = getnrow();

    if (nverts != getncol()) {
        SpParHelper::Print("Number of rows and columns differ, not allowed for graphs!\n");
        MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
    }

    if (nverts != Assignments.TotalLength()) {
        SpParHelper::Print("Assignments vector length does not match number of vertices!\n");
        MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
    }

    IT maxproc = Assignments.Reduce(maximum<IT>(), static_cast<IT>(0));

    if (maxproc >= static_cast<IT>(nprocs)) {
        SpParHelper::Print("Assignments vector assigns to process not not in this group!\n");
        MPI_Abort(MPI_COMM_WORLD, INVALIDPARAMS);
    }

    MPI_Comm World = commGrid->GetWorld();
    MPI_Comm RowWorld = commGrid->GetRowWorld();
    MPI_Comm ColWorld = commGrid->GetColWorld();

    int rowneighs, rowrank;
    MPI_Comm_size(RowWorld, &rowneighs);
    MPI_Comm_rank(RowWorld, &rowrank);


    int mylocsize = Assignments.LocArrSize();
    std::vector<int> rowvecs_counts(rowneighs, 0);
    std::vector<int> rowvecs_displs(rowneighs, 0);

    rowvecs_counts[rowrank] = mylocsize;

    MPI_Allgather(MPI_IN_PLACE, 0, MPI_INT, rowvecs_counts.data(), 1, MPI_INT, RowWorld);

    /* TODO GRGR: make this resilent if displacements don't fit 32 bits */

    std::partial_sum(rowvecs_counts.begin(), rowvecs_counts.end()-1, rowvecs_displs.begin()+1);
    size_t rowvecs_size = std::accumulate(rowvecs_counts.begin(), rowvecs_counts.end(), static_cast<size_t>(0));

    std::vector<IT> rowvecs(rowvecs_size);

    MPI_Allgatherv(Assignments.GetLocArr(), mylocsize, MPIType<IT>(), rowvecs.data(), rowvecs_counts.data(), rowvecs_displs.data(), MPIType<IT>(), RowWorld);

    int complement_rank = commGrid->GetComplementRank();
    int complement_rowvecs_size = 0;

    MPI_Sendrecv(&rowvecs_size, 1, MPI_INT,
                 complement_rank, TRX,
                 &complement_rowvecs_size, 1, MPI_INT,
                 complement_rank, TRX,
                 World, MPI_STATUS_IGNORE);

    std::vector<IT> complement_rowvecs(complement_rowvecs_size);

    MPI_Sendrecv(rowvecs.data(), rowvecs_size, MPIType<IT>(),
                 complement_rank, TRX,
                 complement_rowvecs.data(), complement_rowvecs_size, MPIType<IT>(),
                 complement_rank, TRX,
                 World, MPI_STATUS_IGNORE);

    std::vector<std::vector<std::tuple<IT,IT,NT>>> svec(nprocs);

    std::vector<int> sendcounts(nprocs, 0);
    std::vector<int> recvcounts(nprocs, 0);
    std::vector<int> sdispls(nprocs, 0);
    std::vector<int> rdispls(nprocs, 0);

    int sbuflen = 0;

    IT row_offset, col_offset;
    GetPlaceInGlobalGrid(row_offset, col_offset);

    for (auto colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit) {
        IT destproc = complement_rowvecs[colit.colid()];
        if (destproc != -1)
            for (auto nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit) {
                if (destproc == rowvecs[nzit.rowid()]) {
                    svec[destproc].emplace_back(row_offset + nzit.rowid(), col_offset + colit.colid(), nzit.value());
                    sendcounts[destproc]++;
                    sbuflen++;
                }
            }
    }

    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, World);

    size_t rbuflen = std::accumulate(recvcounts.begin(), recvcounts.end(), static_cast<size_t>(0));

    std::partial_sum(sendcounts.begin(), sendcounts.end()-1, sdispls.begin()+1);
    std::partial_sum(recvcounts.begin(), recvcounts.end()-1, rdispls.begin()+1);

    std::tuple<IT,IT,NT> *sbuf = new std::tuple<IT,IT,NT>[sbuflen];
    std::tuple<IT,IT,NT> *rbuf = new std::tuple<IT,IT,NT>[rbuflen];

    for (int i = 0; i < nprocs; ++i)
        std::copy(svec[i].begin(), svec[i].end(), sbuf + sdispls[i]);


    MPI_Alltoallv(sbuf, sendcounts.data(), sdispls.data(), MPIType<std::tuple<IT,IT,NT>>(), rbuf, recvcounts.data(), rdispls.data(), MPIType<std::tuple<IT,IT,NT>>(), World);

    delete[] sbuf;

    LocalIdxs.clear();
    LocalIdxs.shrink_to_fit();

    std::unordered_map<IT, IT> locmap;

    IT new_id = 0;
    IT global_ids[2];

    for (int i = 0; i < rbuflen; ++i) {
        global_ids[0] = std::get<0>(rbuf[i]);
        global_ids[1] = std::get<1>(rbuf[i]);
        for (int j = 0; j < 2; ++j) {
            if (locmap.find(global_ids[j]) == locmap.end()) {
                locmap.insert(std::make_pair(global_ids[j], new_id++));
                LocalIdxs.push_back(global_ids[j]);
            }
        }
        std::get<0>(rbuf[i]) = locmap[global_ids[0]];
        std::get<1>(rbuf[i]) = locmap[global_ids[1]];
    }

    IT localdim = LocalIdxs.size();

    DER LocalMat;
    LocalMat.Create(rbuflen, localdim, localdim, rbuf);

    return LocalMat;
}

template <class IT, class NT, class DER>
std::ofstream& SpParMat<IT,NT,DER>::put(std::ofstream& outfile) const
{
	outfile << (*spSeq) << std::endl;
	return outfile;
}

template <class IU, class NU, class UDER>
std::ofstream& operator<<(std::ofstream& outfile, const SpParMat<IU, NU, UDER> & s)
{
	return s.put(outfile) ;	// use the right put() function

}

/**
  * @param[in] grow {global row index}
  * @param[in] gcol {global column index}
  * @param[out] lrow {row index local to the owner}
  * @param[out] lcol {col index local to the owner}
  * @returns {owner processor id}
 **/
template <class IT, class NT,class DER>
template <typename LIT>
int SpParMat<IT,NT,DER>::Owner(IT total_m, IT total_n, IT grow, IT gcol, LIT & lrow, LIT & lcol) const
{
	int procrows = commGrid->GetGridRows();
	int proccols = commGrid->GetGridCols();
	IT m_perproc = total_m / procrows;
	IT n_perproc = total_n / proccols;

	int own_procrow;	// owner's processor row
	if(m_perproc != 0)
	{
		own_procrow = std::min(static_cast<int>(grow / m_perproc), procrows-1);	// owner's processor row
	}
	else	// all owned by the last processor row
	{
		own_procrow = procrows -1;
	}
	int own_proccol;
	if(n_perproc != 0)
	{
		own_proccol = std::min(static_cast<int>(gcol / n_perproc), proccols-1);
	}
	else
	{
		own_proccol = proccols-1;
	}
	lrow = grow - (own_procrow * m_perproc);
	lcol = gcol - (own_proccol * n_perproc);
	return commGrid->GetRank(own_procrow, own_proccol);
}

/**
  * @param[out] rowOffset {Row offset imposed by process grid. Global row index = rowOffset + local row index.}
  * @param[out] colOffset {Column offset imposed by process grid. Global column index = colOffset + local column index.}
 **/
template <class IT, class NT,class DER>
void SpParMat<IT,NT,DER>::GetPlaceInGlobalGrid(IT& rowOffset, IT& colOffset) const
{
	IT total_rows = getnrow();
	IT total_cols = getncol();

	int procrows = commGrid->GetGridRows();
	int proccols = commGrid->GetGridCols();
	IT rows_perproc = total_rows / procrows;
	IT cols_perproc = total_cols / proccols;
	
	rowOffset = commGrid->GetRankInProcCol()*rows_perproc;
	colOffset = commGrid->GetRankInProcRow()*cols_perproc;
}
	
}
