/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.4 -------------------------------------------------*/
/* date: 1/17/2014 ---------------------------------------------*/
/* authors: Aydin Buluc (abuluc@lbl.gov), Adam Lugowski --------*/
/****************************************************************/
/*
 Copyright (c) 2010-2014, The Regents of the University of California
 
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

#ifndef _BFS_FRIENDS_H_
#define _BFS_FRIENDS_H_

#include "mpi.h"
#include <iostream>
#include "SpParMat.h"	
#include "SpParHelper.h"
#include "MPIType.h"
#include "Friends.h"
#include "OptBuf.h"
#include "ParFriends.h"
#include "BitMap.h"
#include "BitMapCarousel.h"
#include "BitMapFringe.h"

namespace combblas {

template <class IT, class NT, class DER>
class SpParMat;

/*************************************************************************************************/
/*********************** FRIEND FUNCTIONS FOR BFS ONLY (NO SEMIRINGS) RUNS  **********************/
/***************************** BOTH PARALLEL AND SEQUENTIAL FUNCTIONS ****************************/
/*************************************************************************************************/

/** 
 * Multithreaded SpMV with sparse vector and preset buffers
 * the assembly of outgoing buffers sendindbuf/sendnumbuf are done here
 */
template <typename IT, typename VT>
void dcsc_gespmv_threaded_setbuffers (const SpDCCols<IT, bool> & A, const int32_t * indx, const VT * numx, int32_t nnzx, 
				 int32_t * sendindbuf, VT * sendnumbuf, int * cnts, int * dspls, int p_c)
{
    Select2ndSRing<bool, VT, VT> BFSsring;
	if(A.getnnz() > 0 && nnzx > 0)
	{
		int splits = A.getnsplit();
		if(splits > 0)
		{
			std::vector< std::vector<int32_t> > indy(splits);
			std::vector< std::vector< VT > > numy(splits);
			int32_t nlocrows = static_cast<int32_t>(A.getnrow());
			int32_t perpiece = nlocrows / splits;
			
			#ifdef _OPENMP
			#pragma omp parallel for 
			#endif
			for(int i=0; i<splits; ++i)
			{
				if(i != splits-1)
					SpMXSpV_ForThreading<BFSsring>(*(A.GetDCSC(i)), perpiece, indx, numx, nnzx, indy[i], numy[i], i*perpiece);
				else
					SpMXSpV_ForThreading<BFSsring>(*(A.GetDCSC(i)), nlocrows - perpiece*i, indx, numx, nnzx, indy[i], numy[i], i*perpiece);
			}
			
			int32_t perproc = nlocrows / p_c;	
			int32_t last_rec = p_c-1;
			
			// keep recipients of last entries in each split (-1 for an empty split)
			// so that we can delete indy[] and numy[] contents as soon as they are processed		
			std::vector<int32_t> end_recs(splits);
			for(int i=0; i<splits; ++i)
			{
				if(indy[i].empty())
					end_recs[i] = -1;
				else
					end_recs[i] = std::min(indy[i].back() / perproc, last_rec);
			}
			
			int ** loc_rec_cnts = new int *[splits];	
			#ifdef _OPENMP	
			#pragma omp parallel for
			#endif	
			for(int i=0; i<splits; ++i)
			{
				loc_rec_cnts[i]  = new int[p_c](); // thread-local recipient data
				if(!indy[i].empty())	// guarantee that .begin() and .end() are not null
				{
					int32_t cur_rec = std::min( indy[i].front() / perproc, last_rec);
					int32_t lastdata = (cur_rec+1) * perproc;  // one past last entry that goes to this current recipient
					for(typename std::vector<int32_t>::iterator it = indy[i].begin(); it != indy[i].end(); ++it)
					{
						if( ( (*it) >= lastdata ) && cur_rec != last_rec)	
						{
							cur_rec = std::min( (*it) / perproc, last_rec);	
							lastdata = (cur_rec+1) * perproc;
						}
						++loc_rec_cnts[i][cur_rec];
					}
				}
			}
			#ifdef _OPENMP	
			#pragma omp parallel for 
			#endif
			for(int i=0; i<splits; ++i)
			{
				if(!indy[i].empty())	// guarantee that .begin() and .end() are not null
				{
					// FACT: Data is sorted, so if the recipient of begin is the same as the owner of end, 
					// then the whole data is sent to the same processor
					int32_t beg_rec = std::min( indy[i].front() / perproc, last_rec); 
					int32_t alreadysent = 0;	// already sent per recipient 
					for(int before = i-1; before >= 0; before--)
						 alreadysent += loc_rec_cnts[before][beg_rec];
						
					if(beg_rec == end_recs[i])	// fast case
					{
						std::transform(indy[i].begin(), indy[i].end(), indy[i].begin(),
							[perproc, beg_rec](int32_t val) { return val - perproc * beg_rec; }
							);
						std::copy(indy[i].begin(), indy[i].end(), sendindbuf + dspls[beg_rec] + alreadysent);
						std::copy(numy[i].begin(), numy[i].end(), sendnumbuf + dspls[beg_rec] + alreadysent);
					}
					else	// slow case
					{
						int32_t cur_rec = beg_rec;
						int32_t lastdata = (cur_rec+1) * perproc;  // one past last entry that goes to this current recipient
						for(typename std::vector<int32_t>::iterator it = indy[i].begin(); it != indy[i].end(); ++it)
						{
							if( ( (*it) >= lastdata ) && cur_rec != last_rec )
							{
								cur_rec = std::min( (*it) / perproc, last_rec);
								lastdata = (cur_rec+1) * perproc;

								// if this split switches to a new recipient after sending some data
								// then it's sure that no data has been sent to that recipient yet
						 		alreadysent = 0;
							}
							sendindbuf[ dspls[cur_rec] + alreadysent ] = (*it) - perproc*cur_rec;	// convert to receiver's local index
							sendnumbuf[ dspls[cur_rec] + (alreadysent++) ] = *(numy[i].begin() + (it-indy[i].begin()));
						}
					}
				}
			}
			// Deallocated rec counts serially once all threads complete
			for(int i=0; i< splits; ++i)	
			{
				for(int j=0; j< p_c; ++j)
					cnts[j] += loc_rec_cnts[i][j];
				delete [] loc_rec_cnts[i];
			}
			delete [] loc_rec_cnts;
		}
		else
		{
			std::cout << "Something is wrong, splits should be nonzero for multithreaded execution" << std::endl;
		}
	}
}

/**
  * Step 3 of the sparse SpMV algorithm, without the semiring (BFS only)
  * @param[in,out] optbuf {scratch space for all-to-all (fold) communication}
  * @param[in,out] indacc, numacc {index and values of the input vector, deleted upon exit}
  * @param[in,out] sendindbuf, sendnumbuf {index and values of the output vector, created}
 **/
template<typename VT, typename IT, typename UDER>
void LocalSpMV(const SpParMat<IT,bool,UDER> & A, int rowneighs, OptBuf<int32_t, VT > & optbuf, int32_t * & indacc, VT * & numacc, int * sendcnt, int accnz)
{

#ifdef TIMING
	double t0=MPI_Wtime();
#endif
	if(optbuf.totmax > 0)	// graph500 optimization enabled
	{ 
		if(A.spSeq->getnsplit() > 0)
		{
			// optbuf.{inds/nums/dspls} and sendcnt are all pre-allocated and only filled by dcsc_gespmv_threaded
            
        generic_gespmv_threaded_setbuffers< Select2ndSRing<bool, VT, VT> > (*(A.spSeq), indacc, numacc, (int32_t) accnz, optbuf.inds, optbuf.nums, sendcnt, optbuf.dspls, rowneighs);
		}
		else
		{
			// by-pass dcsc_gespmv call
			if(A.getlocalnnz() > 0 && accnz > 0)
			{
                // ABAB: ignoring optbuf.isthere here
                // \TODO: Remove .isthere from optbuf definition
				SpMXSpV< Select2ndSRing<bool, VT, VT> >(*((A.spSeq)->GetInternal()), (int32_t) A.getlocalrows(), indacc, numacc,
					accnz, optbuf.inds, optbuf.nums, sendcnt, optbuf.dspls, rowneighs);
			}
		}
		DeleteAll(indacc,numacc);
	}
	else
	{
		SpParHelper::Print("BFS only (no semiring) function only work with optimization buffers\n");
	}

#ifdef TIMING
	double t1=MPI_Wtime();
	cblas_localspmvtime += (t1-t0);
#endif
}


template <typename IU, typename VT>
void MergeContributions(FullyDistSpVec<IU,VT> & y, int * & recvcnt, int * & rdispls, int32_t * & recvindbuf, VT * & recvnumbuf, int rowneighs)
{
#ifdef TIMING
	double t0=MPI_Wtime();
#endif
	// free memory of y, in case it was aliased
	std::vector<IU>().swap(y.ind);
	std::vector<VT>().swap(y.num);
	
#ifndef HEAPMERGE
	IU ysize = y.MyLocLength();	// my local length is only O(n/p)
	bool * isthere = new bool[ysize];
	std::vector< std::pair<IU,VT> > ts_pairs;	
  std::fill_n(isthere, ysize, false);

	// We don't need to keep a "merger" because minimum will always come from the processor
	// with the smallest rank; so a linear sweep over the received buffer is enough	
	for(int i=0; i<rowneighs; ++i)
	{
		for(int j=0; j< recvcnt[i]; ++j) 
		{
			int32_t index = recvindbuf[rdispls[i] + j];
			if(!isthere[index])
				ts_pairs.push_back(std::make_pair(index, recvnumbuf[rdispls[i] + j]));
			
		}
	}
	DeleteAll(recvcnt, rdispls);
	DeleteAll(isthere, recvindbuf, recvnumbuf);
	__gnu_parallel::sort(ts_pairs.begin(), ts_pairs.end());
	int nnzy = ts_pairs.size();
	y.ind.resize(nnzy);
	y.num.resize(nnzy);
	for(int i=0; i< nnzy; ++i)
	{
		y.ind[i] = ts_pairs[i].first;
		y.num[i] = ts_pairs[i].second; 	
	}

#else
	// Alternative 2: Heap-merge
	int32_t hsize = 0;		
	int32_t inf = std::numeric_limits<int32_t>::min();
	int32_t sup = std::numeric_limits<int32_t>::max(); 
	KNHeap< int32_t, int32_t > sHeap(sup, inf); 
	int * processed = new int[rowneighs]();
	for(int32_t i=0; i<rowneighs; ++i)
	{
		if(recvcnt[i] > 0)
		{
			// key, proc_id
			sHeap.insert(recvindbuf[rdispls[i]], i);
			++hsize;
		}
	}	
	int32_t key, locv;
	if(hsize > 0)
	{
		sHeap.deleteMin(&key, &locv);
		y.ind.push_back( static_cast<IU>(key));
		y.num.push_back(recvnumbuf[rdispls[locv]]);	// nothing is processed yet
		
		if( (++(processed[locv])) < recvcnt[locv] )
			sHeap.insert(recvindbuf[rdispls[locv]+processed[locv]], locv);
		else
			--hsize;
	}

//	ofstream oput;
//	y.commGrid->OpenDebugFile("Merge", oput);
//	oput << "From displacements: "; copy(rdispls, rdispls+rowneighs, ostream_iterator<int>(oput, " ")); oput << endl;
//	oput << "From counts: "; copy(recvcnt, recvcnt+rowneighs, ostream_iterator<int>(oput, " ")); oput << endl;
	while(hsize > 0)
	{
		sHeap.deleteMin(&key, &locv);
		IU deref = rdispls[locv] + processed[locv];
		if(y.ind.back() != static_cast<IU>(key))	// y.ind is surely not empty
		{
			y.ind.push_back(static_cast<IU>(key));
			y.num.push_back(recvnumbuf[deref]);
		}
		
		if( (++(processed[locv])) < recvcnt[locv] )
			sHeap.insert(recvindbuf[rdispls[locv]+processed[locv]], locv);
		else
			--hsize;
	}
	DeleteAll(recvcnt, rdispls,processed);
	DeleteAll(recvindbuf, recvnumbuf);
#endif

#ifdef TIMING
	double t1=MPI_Wtime();
	cblas_mergeconttime += (t1-t0);
#endif
}	

/**
  * This is essentially a SpMV for BFS because it lacks the semiring.
  * It naturally justs selects columns of A (adjacencies of frontier) and 
  * merges with the minimum entry succeeding. SpParMat has to be boolean
  * input and output vectors are of type VT but their indices are IT
  */
template <typename VT, typename IT, typename UDER>
FullyDistSpVec<IT,VT>  SpMV (const SpParMat<IT,bool,UDER> & A, const FullyDistSpVec<IT,VT> & x, OptBuf<int32_t, VT > & optbuf)
{
	CheckSpMVCompliance(A,x);
	optbuf.MarkEmpty();
		
	MPI_Comm World = x.commGrid->GetWorld();
	MPI_Comm ColWorld = x.commGrid->GetColWorld();
	MPI_Comm RowWorld = x.commGrid->GetRowWorld();

	int accnz;
	int32_t trxlocnz;
	IT lenuntil;
	int32_t *trxinds, *indacc;
	VT *trxnums, *numacc;
    
	
#ifdef TIMING
	double t0=MPI_Wtime();
#endif
	TransposeVector(World, x, trxlocnz, lenuntil, trxinds, trxnums, true);			// trxinds (and potentially trxnums) is allocated
#ifdef TIMING
	double t1=MPI_Wtime();
	cblas_transvectime += (t1-t0);
#endif
	AllGatherVector(ColWorld, trxlocnz, lenuntil, trxinds, trxnums, indacc, numacc, accnz, true);	// trxinds (and potentially trxnums) is deallocated, indacc/numacc allocated
	
	FullyDistSpVec<IT, VT> y ( x.commGrid, A.getnrow());	// identity doesn't matter for sparse vectors
	int rowneighs; MPI_Comm_size(RowWorld,&rowneighs);
	int * sendcnt = new int[rowneighs]();	

	LocalSpMV(A, rowneighs, optbuf, indacc, numacc, sendcnt, accnz);	// indacc/numacc deallocated

	int * rdispls = new int[rowneighs];
	int * recvcnt = new int[rowneighs];
	MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, RowWorld);	// share the request counts
	
	// receive displacements are exact whereas send displacements have slack
	rdispls[0] = 0;
	for(int i=0; i<rowneighs-1; ++i)
	{
		rdispls[i+1] = rdispls[i] + recvcnt[i];
	}
	int totrecv = std::accumulate(recvcnt,recvcnt+rowneighs,0);	
	int32_t * recvindbuf = new int32_t[totrecv];
	VT * recvnumbuf = new VT[totrecv];
	
#ifdef TIMING
	double t2=MPI_Wtime();
#endif
	if(optbuf.totmax > 0 )	// graph500 optimization enabled
	{
        MPI_Alltoallv(optbuf.inds, sendcnt, optbuf.dspls, MPIType<int32_t>(), recvindbuf, recvcnt, rdispls, MPIType<int32_t>(), RowWorld);  
		MPI_Alltoallv(optbuf.nums, sendcnt, optbuf.dspls, MPIType<VT>(), recvnumbuf, recvcnt, rdispls, MPIType<VT>(), RowWorld);  
		delete [] sendcnt;
	}
	else
	{
		SpParHelper::Print("BFS only (no semiring) function only work with optimization buffers\n");
	}
#ifdef TIMING
	double t3=MPI_Wtime();
	cblas_alltoalltime += (t3-t2);
#endif

	MergeContributions(y,recvcnt, rdispls, recvindbuf, recvnumbuf, rowneighs);
	return y;	
}

template <typename VT, typename IT, typename UDER>
SpDCCols<int,bool>::SpColIter* CalcSubStarts(SpParMat<IT,bool,UDER> & A, FullyDistSpVec<IT,VT> & x, BitMapCarousel<IT,VT> &done) {
	std::shared_ptr<CommGrid> cg = A.getcommgrid();
	IT rowuntil = x.LengthUntil();
	MPI_Comm RowWorld = cg->GetRowWorld();
	MPI_Bcast(&rowuntil, 1, MPIType<IT>(), 0, RowWorld);
	int numcols = cg->GetGridCols();
	SpDCCols<int,bool>::SpColIter colit = A.seq().begcol();
#ifdef THREADED
    SpDCCols<int,bool>::SpColIter* starts = new SpDCCols<int,bool>::SpColIter[numcols*cblas_splits+1];
    for(int c=0; c<numcols; c++) {
		IT curr_sub_start = done.GetGlobalStartOfLocal(c) - rowuntil;
		IT next_sub_start = done.GetGlobalEndOfLocal(c) - rowuntil;
		IT sub_range = next_sub_start - curr_sub_start;
		IT per_thread = (sub_range + cblas_splits - 1) / cblas_splits;
		IT curr_thread_start = curr_sub_start;
		for (int t=0; t<cblas_splits; t++) {
			while(colit.colid() < curr_thread_start) {
				++colit;
			}
			starts[c*cblas_splits + t] = colit;
			curr_thread_start = std::min(curr_thread_start + per_thread, next_sub_start);
		}
    }
    starts[numcols*cblas_splits] = A.seq().endcol();
#else
    SpDCCols<int,bool>::SpColIter* starts = new SpDCCols<int,bool>::SpColIter[numcols+1];
    for(int c=0; c<numcols; c++) {
		IT next_start = done.GetGlobalStartOfLocal(c) - rowuntil;
		while(colit.colid() < next_start) {
			++colit;
		}
		starts[c] = colit;
    }
    starts[numcols] = A.seq().endcol();
#endif
	return starts;
}

template <typename VT, typename IT>
void UpdateParents(MPI_Comm & RowWorld, std::pair<IT,IT> *updates, int num_updates, FullyDistVec<IT,VT> &parents, int source, int dest, BitMapFringe<int64_t,int64_t> &bm_fringe) {
	int send_words = num_updates<<1, recv_words;
	MPI_Status status;
	MPI_Sendrecv(&send_words, 1, MPI_INT, dest, PUPSIZE,
					  &recv_words, 1, MPI_INT, source, PUPSIZE, RowWorld, &status);
	std::pair<IT,IT>* recv_buff = new std::pair<IT,IT>[recv_words>>1];
	MPI_Sendrecv(updates, send_words, MPIType<IT>(), dest, PUPDATA,
					  recv_buff, recv_words, MPIType<IT>(), source, PUPDATA, RowWorld, &status);
	
#ifdef THREADED
#pragma omp parallel for
#endif
	for (int i=0; i<recv_words>>1; i++) {
		parents.SetLocalElement(recv_buff[i].first, recv_buff[i].second);
	}
	
	bm_fringe.IncrementNumSet((recv_words>>1));
	delete[] recv_buff;
}


template <typename VT, typename IT, typename UDER>
void BottomUpStep(SpParMat<IT,bool,UDER> & A, FullyDistSpVec<IT,VT> & x, BitMapFringe<int64_t,int64_t> &bm_fringe, FullyDistVec<IT,VT> & parents, BitMapCarousel<IT,VT> &done, SpDCCols<int,bool>::SpColIter* starts)
{
	std::shared_ptr<CommGrid> cg = A.getcommgrid();
	MPI_Comm World = cg->GetWorld();
	MPI_Comm ColWorld = cg->GetColWorld();
	MPI_Comm RowWorld = cg->GetRowWorld();
	MPI_Status status;
	
	// get row and column offsets
	IT rowuntil = x.LengthUntil(), my_coluntil = x.LengthUntil(), coluntil;
	int diagneigh = cg->GetComplementRank();
	MPI_Sendrecv(&my_coluntil, 1, MPIType<IT>(), diagneigh, TROST, &coluntil, 1, MPIType<IT>(), diagneigh, TROST, World, &status);
	MPI_Bcast(&coluntil, 1, MPIType<IT>(), 0, ColWorld);
	MPI_Bcast(&rowuntil, 1, MPIType<IT>(), 0, RowWorld);
	
	BitMap* frontier = bm_fringe.TransposeGather();
	done.SaveOld();
	
#ifdef THREADED
	const int buff_size = 8192;
	std::pair<IT,IT>* local_update_heads[cblas_splits];
	for (int t=0; t<cblas_splits; t++)
		local_update_heads[t] = new std::pair<IT,IT>[buff_size];
#endif
	
	// do bottom up work
	int numcols = cg->GetGridCols();
	int mycol = cg->GetRankInProcRow();
	std::pair<IT,IT>* parent_updates = new std::pair<IT,IT>[done.SizeOfChunk()<<1]; // over-allocated
	
	for (int sub_step=0; sub_step<numcols; sub_step++) {
		int num_updates = 0;
		IT sub_start = done.GetGlobalStartOfLocal();
		int dest_slice = (mycol + sub_step) % numcols;
		int source_slice = (mycol - sub_step + numcols) % numcols;
#ifdef BOTTOMUPTIME
		double t1 = MPI_Wtime();
#endif
#ifdef THREADED
#pragma omp parallel
		{
			int id = omp_get_thread_num();
			int num_locals=0;
			SpDCCols<int,bool>::SpColIter::NzIter nzit, nzit_end;
			SpDCCols<int,bool>::SpColIter colit, colit_end;
			std::pair<IT,IT>* local_updates = local_update_heads[id];
			// vector<pair<IT,IT> > local_updates;
			colit_end = starts[dest_slice*cblas_splits + id + 1];
			for(colit = starts[dest_slice*cblas_splits + id]; colit != colit_end; ++colit) {
				int32_t local_row_ind = colit.colid();
				IT row = local_row_ind + rowuntil;
				if (!done.GetBit(row)) {
					nzit_end = A.seq().endnz(colit);
					for(nzit = A.seq().begnz(colit); nzit != nzit_end; ++nzit) {
						int32_t local_col_ind = nzit.rowid();
						IT col = local_col_ind + coluntil;
						if (frontier->get_bit(local_col_ind)) {
							// local_updates.push_back(make_pair(row-sub_start, col));
							if (num_locals == buff_size) {
								int copy_start = __sync_fetch_and_add(&num_updates, buff_size);
                std::copy(local_updates, local_updates + buff_size, parent_updates + copy_start);
								num_locals = 0;
							}
							local_updates[num_locals++] = std::make_pair(row-sub_start, col);
							done.SetBit(row);
							break;
						}
					}
				}
			}
			int copy_start = __sync_fetch_and_add(&num_updates, num_locals);
      std::copy(local_updates, local_updates + num_locals, parent_updates + copy_start);
		}
#else
		SpDCCols<int,bool>::SpColIter::NzIter nzit, nzit_end;
		SpDCCols<int,bool>::SpColIter colit, colit_end;
		colit_end = starts[dest_slice+1];
		for(colit = starts[dest_slice]; colit != colit_end; ++colit) 
		{
			int32_t local_row_ind = colit.colid();
			IT row = local_row_ind + rowuntil;
			if (!done.GetBit(row)) 
			{
				nzit_end = A.seq().endnz(colit);
				for(nzit = A.seq().begnz(colit); nzit != nzit_end; ++nzit) 
				{
					int32_t local_col_ind = nzit.rowid();
					IT col = local_col_ind + coluntil;
					if (frontier->get_bit(local_col_ind)) 
					{
						parent_updates[num_updates++] = std::make_pair(row-sub_start, col);
						done.SetBit(row);
						break;
					}
				} // end_for
			} // end_if
		} // end_for
#endif

#ifdef BOTTOMUPTIME
		double t2 = MPI_Wtime();
		bu_local += (t2-t1);
		t1 = MPI_Wtime();
#endif
		done.RotateAlongRow();

#ifdef BOTTOMUPTIME
		t2 = MPI_Wtime();
		bu_rotate += (t2-t1);
		t1 = MPI_Wtime();
#endif
		UpdateParents(RowWorld, parent_updates, num_updates, parents, source_slice, dest_slice, bm_fringe);
#ifdef BOTTOMUPTIME
		t2 = MPI_Wtime();
		bu_update += (t2-t1);
#endif
	}
	bm_fringe.LoadFromNext();
	done.UpdateFringe(bm_fringe);
#ifdef THREADED
	for (int t=0; t<cblas_splits; t++)
		delete[] local_update_heads[t];
#endif
	delete[] parent_updates;
}

}

#endif
