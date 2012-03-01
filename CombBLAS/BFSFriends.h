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
#include "SpImplNoSR.h"

using namespace std;

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
	if(A.getnnz() > 0 && nnzx > 0)
	{
		int splits = A.getnsplit();
		if(splits > 0)
		{
			vector< vector<int32_t> > indy(splits);
			vector< vector< VT > > numy(splits);
			int32_t nlocrows = static_cast<int32_t>(A.getnrow());
			int32_t perpiece = nlocrows / splits;
			
			#ifdef _OPENMP
			#pragma omp parallel for 
			#endif
			for(int i=0; i<splits; ++i)
			{
				if(i != splits-1)
					SpMXSpV_ForThreading(*(A.GetDCSC(i)), perpiece, indx, numx, nnzx, indy[i], numy[i], i*perpiece);
				else
					SpMXSpV_ForThreading(*(A.GetDCSC(i)), nlocrows - perpiece*i, indx, numx, nnzx, indy[i], numy[i], i*perpiece);
			}
			
			int32_t perproc = nlocrows / p_c;	
			int32_t last_rec = p_c-1;
			
			// keep recipients of last entries in each split (-1 for an empty split)
			// so that we can delete indy[] and numy[] contents as soon as they are processed		
			vector<int32_t> end_recs(splits);
			for(int i=0; i<splits; ++i)
			{
				if(indy[i].empty())
					end_recs[i] = -1;
				else
					end_recs[i] = min(indy[i].back() / perproc, last_rec);
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
					int32_t cur_rec = min( indy[i].front() / perproc, last_rec);
					int32_t lastdata = (cur_rec+1) * perproc;  // last entry that goes to this current recipient
					for(typename vector<int32_t>::iterator it = indy[i].begin(); it != indy[i].end(); ++it)
					{
						if(!((*it) < lastdata))
						{
							cur_rec = min( (*it) / perproc, last_rec);
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
					int32_t beg_rec = min( indy[i].front() / perproc, last_rec); 
					int32_t alreadysent = 0;	// already sent per recipient 
					for(int before = i-1; before >= 0; before--)
						 alreadysent += loc_rec_cnts[before][beg_rec];
						
					if(beg_rec == end_recs[i])	// fast case
					{
						transform(indy[i].begin(), indy[i].end(), indy[i].begin(), bind2nd(minus<int32_t>(), perproc*beg_rec));
						copy(indy[i].begin(), indy[i].end(), sendindbuf + dspls[beg_rec] + alreadysent);
						copy(numy[i].begin(), numy[i].end(), sendnumbuf + dspls[beg_rec] + alreadysent);
					}
					else	// slow case
					{
						int32_t cur_rec = beg_rec;
						int32_t lastdata = (cur_rec+1) * perproc;  // last entry that goes to this current recipient
						for(typename vector<int32_t>::iterator it = indy[i].begin(); it != indy[i].end(); ++it)
						{
							if(!((*it) < lastdata))
							{
								cur_rec = min( (*it) / perproc, last_rec);
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
			cout << "Something is wrong, splits should be nonzero for multithreaded execution" << endl;
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
void LocalSpMV(const SpParMat<IT,bool,UDER> & A, int rowneighs, OptBuf<int32_t, VT > & optbuf, int32_t * & indacc, VT * & numacc, 
			   int32_t * & sendindbuf, VT * & sendnumbuf, int * & sdispls, int * sendcnt, int accnz, bool indexisvalue)
{	
	if(optbuf.totmax > 0)	// graph500 optimization enabled
	{ 
		if(A.spSeq->getnsplit() > 0)
		{
			// optbuf.{inds/nums/dspls} and sendcnt are all pre-allocated and only filled by dcsc_gespmv_threaded
			dcsc_gespmv_threaded_setbuffers (*(A.spSeq), indacc, numacc, accnz, optbuf.inds, optbuf.nums, sendcnt, optbuf.dspls, rowneighs);	
		}
		else
		{
		//	call something else?
		//	dcsc_gespmv (*(A.spSeq), indacc, numacc, accnz, optbuf.inds, optbuf.nums, sendcnt, optbuf.dspls, rowneighs, indexisvalue);
		}
		DeleteAll(indacc,numacc);
	}
	else
	{
		SpParHelper::Print("BFS only (no semiring) function only work with optimization buffers\n");
	}
}


/**
  * This is essentially a SpMV for BFS because it lacks the semiring.
  * It naturally justs selects columns of A (adjacencies of frontier) and 
  * merges with the minimum entry succeeding. SpParMat has to be boolean
  * input and output vectors are of type VT but their indices are IT
  */
template <typename VT, typename IT, typename UDER>
FullyDistSpVec<IT,VT>  SpMV (const SpParMat<IT,bool,UDER> & A, const FullyDistSpVec<IT,VT> & x, bool indexisvalue, OptBuf<int32_t, VT > & optbuf)
{
	CheckSpMVCompliance(A,x);

	MPI::Intracomm World = x.commGrid->GetWorld();
	MPI::Intracomm ColWorld = x.commGrid->GetColWorld();
	MPI::Intracomm RowWorld = x.commGrid->GetRowWorld();

	int accnz;
	int32_t trxlocnz;
	IT lenuntil;
	int32_t *trxinds, *indacc;
	VT *trxnums, *numacc;

	TransposeVector(World, x, trxlocnz, lenuntil, trxinds, trxnums, indexisvalue);			// trxinds (and potentially trxnums) is allocated
	AllGatherVector(ColWorld, trxlocnz, lenuntil, trxinds, trxnums, indacc, numacc, accnz, indexisvalue);	// trxinds (and potentially trxnums) is deallocated, indacc/numacc allocated
	
	FullyDistSpVec<IT, VT> y ( x.commGrid, A.getnrow());	// identity doesn't matter for sparse vectors
	int rowneighs = RowWorld.Get_size();
	int * sendcnt = new int[rowneighs]();	
	int32_t * sendindbuf;	
	VT * sendnumbuf;
	int * sdispls;
	LocalSpMV(A, rowneighs, optbuf, indacc, numacc, sendindbuf, sendnumbuf, sdispls, sendcnt, accnz, indexisvalue);	// indacc/numacc deallocated, sendindbuf/sendnumbuf/sdispls allocated

	int * rdispls = new int[rowneighs];
	int * recvcnt = new int[rowneighs];
	RowWorld.Alltoall(sendcnt, 1, MPI::INT, recvcnt, 1, MPI::INT);	// share the request counts 
	
	// receive displacements are exact whereas send displacements have slack
	rdispls[0] = 0;
	for(int i=0; i<rowneighs-1; ++i)
	{
		rdispls[i+1] = rdispls[i] + recvcnt[i];
	}
	int totrecv = accumulate(recvcnt,recvcnt+rowneighs,0);	
	int32_t * recvindbuf = new int32_t[totrecv];
	VT * recvnumbuf = new VT[totrecv];
	
#ifdef TIMING
	World.Barrier();
	double t2=MPI::Wtime();
#endif
	if(optbuf.totmax > 0 )	// graph500 optimization enabled
	{
		RowWorld.Alltoallv(optbuf.inds, sendcnt, optbuf.dspls, MPIType<int32_t>(), recvindbuf, recvcnt, rdispls, MPIType<int32_t>());  
		RowWorld.Alltoallv(optbuf.nums, sendcnt, optbuf.dspls, MPIType<VT>(), recvnumbuf, recvcnt, rdispls, MPIType<VT>());  
		delete [] sendcnt;
	}
	else
	{
		SpParHelper::Print("BFS only (no semiring) function only work with optimization buffers\n");
	}
#ifdef TIMING
	World.Barrier();
	double t3=MPI::Wtime();
	cblas_alltoalltime += (t3-t2);
#endif

	//MergeContributions<SR>(y,recvcnt, rdispls, recvindbuf, recvnumbuf, rowneighs);
	return y;	
	
}

#endif
