#ifndef _PAR_FRIENDS_H_
#define _PAR_FRIENDS_H_

#include "mpi.h"
#include <iostream>
#include "SpParMat.h"	
#include "SpParHelper.h"
#include "MPIType.h"
#include "Friends.h"
#include "OptBuf.h"

using namespace std;

template <class IT, class NT, class DER>
class SpParMat;

/*************************************************************************************************/
/**************************** FRIEND FUNCTIONS FOR PARALLEL CLASSES ******************************/
/*************************************************************************************************/

template <typename MATRIXA, typename MATRIXB>
bool CheckSpGEMMCompliance(const MATRIXA & A, const MATRIXB & B)
{
	if(A.getncol() != B.getnrow())
	{
		ostringstream outs;
		outs << "Can not multiply, dimensions does not match"<< endl;
		outs << A.getncol() << " != " << B.getnrow() << endl;
		SpParHelper::Print(outs.str());
		MPI::COMM_WORLD.Abort(DIMMISMATCH);
		return false;
	}	
	return true;
}	


/**
 * Parallel C = A*B routine that uses a double buffered broadcasting scheme 
 * Most memory efficient version available. Total stages: 2*sqrt(p)
 * Memory requirement during first sqrt(p) stages: <= (3/2)*(nnz(A)+nnz(B))+(1/2)*nnz(C)
 * Memory requirement during second sqrt(p) stages: <= nnz(A)+nnz(B)+nnz(C)
 * Final memory requirement: nnz(C) if clearA and clearB are true 
 **/  
template <typename SR, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB> 
SpParMat<IU,typename promote_trait<NU1,NU2>::T_promote,typename promote_trait<UDERA,UDERB>::T_promote> Mult_AnXBn_DoubleBuff
		(SpParMat<IU,NU1,UDERA> & A, SpParMat<IU,NU2,UDERB> & B, bool clearA = false, bool clearB = false )

{
	typedef typename promote_trait<NU1,NU2>::T_promote N_promote;
	typedef typename promote_trait<UDERA,UDERB>::T_promote DER_promote;
	if(!CheckSpGEMMCompliance(A,B) )
	{
		return SpParMat< IU,N_promote,DER_promote >();
	}

	int stages, dummy; 	// last two parameters of ProductGrid are ignored for Synch multiplication
	shared_ptr<CommGrid> GridC = ProductGrid((A.commGrid).get(), (B.commGrid).get(), stages, dummy, dummy);		
	IU C_m = A.spSeq->getnrow();
	IU C_n = B.spSeq->getncol();

	UDERA * A1seq = new UDERA();
	UDERA * A2seq = new UDERA(); 
	UDERB * B1seq = new UDERB();
	UDERB * B2seq = new UDERB();
	(A.spSeq)->Split( *A1seq, *A2seq); 
	const_cast< UDERB* >(B.spSeq)->Transpose();
	(B.spSeq)->Split( *B1seq, *B2seq);
	GridC->GetWorld().Barrier();

	IU ** ARecvSizes = SpHelper::allocate2D<IU>(UDERA::esscount, stages);
	IU ** BRecvSizes = SpHelper::allocate2D<IU>(UDERB::esscount, stages);

	SpParHelper::GetSetSizes( *A1seq, ARecvSizes, (A.commGrid)->GetRowWorld());
	SpParHelper::GetSetSizes( *B1seq, BRecvSizes, (B.commGrid)->GetColWorld());

	// Remotely fetched matrices are stored as pointers
	UDERA * ARecv; 
	UDERB * BRecv;
	vector< SpTuples<IU,N_promote>  *> tomerge;

	int Aself = (A.commGrid)->GetRankInProcRow();
	int Bself = (B.commGrid)->GetRankInProcCol();	

	for(int i = 0; i < stages; ++i) 
	{
		vector<IU> ess;	
		if(i == Aself)
		{	
			ARecv = A1seq;	// shallow-copy 
		}
		else
		{
			ess.resize(UDERA::esscount);
			for(int j=0; j< UDERA::esscount; ++j)	
			{
				ess[j] = ARecvSizes[j][i];		// essentials of the ith matrix in this row	
			}
			ARecv = new UDERA();				// first, create the object
		}
		SpParHelper::BCastMatrix(GridC->GetRowWorld(), *ARecv, ess, i);	// then, receive its elements	
		ess.clear();	
		if(i == Bself)
		{
			BRecv = B1seq;	// shallow-copy
		}
		else
		{
			ess.resize(UDERB::esscount);		
			for(int j=0; j< UDERB::esscount; ++j)	
			{
				ess[j] = BRecvSizes[j][i];	
			}	
			BRecv = new UDERB();
		}
		SpParHelper::BCastMatrix(GridC->GetColWorld(), *BRecv, ess, i);	// then, receive its elements
		SpTuples<IU,N_promote> * C_cont = MultiplyReturnTuples<SR>
						(*ARecv, *BRecv, // parameters themselves
						false, true,	// transpose information (B is transposed)
						i != Aself, 	// 'delete A' condition
						i != Bself);	// 'delete B' condition

		if(!C_cont->isZero()) 
			tomerge.push_back(C_cont);
	}
	if(clearA) delete A1seq;
	if(clearB) delete B1seq;
	
	// Set the new dimensions
	SpParHelper::GetSetSizes( *A2seq, ARecvSizes, (A.commGrid)->GetRowWorld());
	SpParHelper::GetSetSizes( *B2seq, BRecvSizes, (B.commGrid)->GetColWorld());

	// Start the second round
	for(int i = 0; i < stages; ++i) 
	{
		vector<IU> ess;	
		if(i == Aself)
		{	
			ARecv = A2seq;	// shallow-copy 
		}
		else
		{
			ess.resize(UDERA::esscount);
			for(int j=0; j< UDERA::esscount; ++j)	
			{
				ess[j] = ARecvSizes[j][i];		// essentials of the ith matrix in this row	
			}
			ARecv = new UDERA();				// first, create the object
		}

		SpParHelper::BCastMatrix(GridC->GetRowWorld(), *ARecv, ess, i);	// then, receive its elements	
		ess.clear();	
		
		if(i == Bself)
		{
			BRecv = B2seq;	// shallow-copy
		}
		else
		{
			ess.resize(UDERB::esscount);		
			for(int j=0; j< UDERB::esscount; ++j)	
			{
				ess[j] = BRecvSizes[j][i];	
			}	
			BRecv = new UDERB();
		}
		SpParHelper::BCastMatrix(GridC->GetColWorld(), *BRecv, ess, i);	// then, receive its elements

		SpTuples<IU,N_promote> * C_cont = MultiplyReturnTuples<SR>
						(*ARecv, *BRecv, // parameters themselves
						false, true,	// transpose information (B is transposed)
						i != Aself, 	// 'delete A' condition
						i != Bself);	// 'delete B' condition

		if(!C_cont->isZero()) 
			tomerge.push_back(C_cont);
	}
	SpHelper::deallocate2D(ARecvSizes, UDERA::esscount);
	SpHelper::deallocate2D(BRecvSizes, UDERB::esscount);
	if(clearA) 
	{
		delete A2seq;
		delete A.spSeq;
		A.spSeq = NULL;
	}
	else
	{
		(A.spSeq)->Merge(*A1seq, *A2seq);
		delete A1seq;
		delete A2seq;
	}
	if(clearB) 
	{
		delete B2seq;
		delete B.spSeq;
		B.spSeq = NULL;	
	}
	else
	{
		(B.spSeq)->Merge(*B1seq, *B2seq);	
		delete B1seq;
		delete B2seq;
		const_cast< UDERB* >(B.spSeq)->Transpose();	// transpose back to original
	}
			
	DER_promote * C = new DER_promote(MergeAll<SR>(tomerge, C_m, C_n,true), false, NULL);	
	// First get the result in SpTuples, then convert to UDER
	return SpParMat<IU,N_promote,DER_promote> (C, GridC);		// return the result object
}


/**
 * Parallel A = B*C routine that uses only MPI-1 features
 * Relies on simple blocking broadcast
 **/  
template <typename SR, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB> 
SpParMat<IU,typename promote_trait<NU1,NU2>::T_promote,typename promote_trait<UDERA,UDERB>::T_promote> Mult_AnXBn_Synch 
		(SpParMat<IU,NU1,UDERA> & A, SpParMat<IU,NU2,UDERB> & B, bool clearA = false, bool clearB = false )

{
	typedef typename promote_trait<NU1,NU2>::T_promote N_promote;
	typedef typename promote_trait<UDERA,UDERB>::T_promote DER_promote;
	if(!CheckSpGEMMCompliance(A,B) )
	{
		return SpParMat< IU,N_promote,DER_promote >();
	}
	int stages, dummy; 	// last two parameters of ProductGrid are ignored for Synch multiplication
	shared_ptr<CommGrid> GridC = ProductGrid((A.commGrid).get(), (B.commGrid).get(), stages, dummy, dummy);		
	IU C_m = A.spSeq->getnrow();
	IU C_n = B.spSeq->getncol();
	
	const_cast< UDERB* >(B.spSeq)->Transpose();	
	GridC->GetWorld().Barrier();

	IU ** ARecvSizes = SpHelper::allocate2D<IU>(UDERA::esscount, stages);
	IU ** BRecvSizes = SpHelper::allocate2D<IU>(UDERB::esscount, stages);
	
	SpParHelper::GetSetSizes( *(A.spSeq), ARecvSizes, (A.commGrid)->GetRowWorld());
	SpParHelper::GetSetSizes( *(B.spSeq), BRecvSizes, (B.commGrid)->GetColWorld());

	// Remotely fetched matrices are stored as pointers
	UDERA * ARecv; 
	UDERB * BRecv;
	vector< SpTuples<IU,N_promote>  *> tomerge;

	int Aself = (A.commGrid)->GetRankInProcRow();
	int Bself = (B.commGrid)->GetRankInProcCol();	

	for(int i = 0; i < stages; ++i) 
	{
		vector<IU> ess;	
		if(i == Aself)
		{	
			ARecv = A.spSeq;	// shallow-copy 
		}
		else
		{
			ess.resize(UDERA::esscount);
			for(int j=0; j< UDERA::esscount; ++j)	
			{
				ess[j] = ARecvSizes[j][i];		// essentials of the ith matrix in this row	
			}
			ARecv = new UDERA();				// first, create the object
		}

		SpParHelper::BCastMatrix(GridC->GetRowWorld(), *ARecv, ess, i);	// then, receive its elements	
		ess.clear();	
		
		if(i == Bself)
		{
			BRecv = B.spSeq;	// shallow-copy
		}
		else
		{
			ess.resize(UDERB::esscount);		
			for(int j=0; j< UDERB::esscount; ++j)	
			{
				ess[j] = BRecvSizes[j][i];	
			}	
			BRecv = new UDERB();
		}
		SpParHelper::BCastMatrix(GridC->GetColWorld(), *BRecv, ess, i);	// then, receive its elements

		SpTuples<IU,N_promote> * C_cont = MultiplyReturnTuples<SR>
						(*ARecv, *BRecv, // parameters themselves
						false, true,	// transpose information (B is transposed)
						i != Aself, 	// 'delete A' condition
						i != Bself);	// 'delete B' condition

		if(!C_cont->isZero()) 
			tomerge.push_back(C_cont);

		#ifndef NDEBUG
		ostringstream outs;
		outs << i << "th SUMMA iteration"<< endl;
		SpParHelper::Print(outs.str());
		#endif
	}
	if(clearA && A.spSeq != NULL) 
	{	
		delete A.spSeq;
		A.spSeq = NULL;
	}	
	if(clearB && B.spSeq != NULL) 
	{
		delete B.spSeq;
		B.spSeq = NULL;
	}

	SpHelper::deallocate2D(ARecvSizes, UDERA::esscount);
	SpHelper::deallocate2D(BRecvSizes, UDERB::esscount);
			
	DER_promote * C = new DER_promote(MergeAll<SR>(tomerge, C_m, C_n,true), false, NULL);	
	// First get the result in SpTuples, then convert to UDER
	// the last parameter to MergeAll deletes tomerge arrays

	if(!clearB)
		const_cast< UDERB* >(B.spSeq)->Transpose();	// transpose back to original

	return SpParMat<IU,N_promote,DER_promote> (C, GridC);		// return the result object
}


template <typename MATRIX, typename VECTOR>
void CheckSpMVCompliance(const MATRIX & A, const VECTOR & x)
{
	if(A.getncol() != x.TotalLength())
	{
		ostringstream outs;
		outs << "Can not multiply, dimensions does not match"<< endl;
		outs << A.getncol() << " != " << x.TotalLength() << endl;
		SpParHelper::Print(outs.str());
		MPI::COMM_WORLD.Abort(DIMMISMATCH);
	}
	if(! ( *(A.getcommgrid()) == *(x.getcommgrid())) ) 		
	{
		cout << "Grids are not comparable for SpMV" << endl; 
		MPI::COMM_WORLD.Abort(GRIDMISMATCH);
	}
}			


template <typename SR, typename IU, typename NUM, typename UDER> 
FullyDistSpVec<IU,typename promote_trait<NUM,IU>::T_promote>  SpMV 
	(const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,IU> & x, bool indexisvalue, OptBuf<int32_t, typename promote_trait<NUM,IU>::T_promote > & optbuf);

template <typename SR, typename IU, typename NUM, typename UDER> 
FullyDistSpVec<IU,typename promote_trait<NUM,IU>::T_promote>  SpMV 
	(const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,IU> & x, bool indexisvalue)
{
	typedef typename promote_trait<NUM,IU>::T_promote T_promote;
	OptBuf<int32_t, T_promote > optbuf = OptBuf<int32_t, T_promote >();
	return SpMV<SR>(A, x, indexisvalue, optbuf);
}

/**
 * Step 1 of the sparse SpMV algorithm 
 * @param[in,out]   trxlocnz, lenuntil,trxinds,trxnums  { set or allocated }
 * @param[in] 	indexisvalue	
 **/
template<typename IU>
void TransposeVector(MPI::Intracomm & World, const FullyDistSpVec<IU,IU> & x, IU & trxlocnz, IU & lenuntil, int32_t * & trxinds, IU * & trxnums, bool indexisvalue)
{
	IU xlocnz = x.getlocnnz();
	IU roffst = x.RowLenUntil();
	IU luntil = x.LengthUntil();
	IU roffset;
	int diagneigh = x.commGrid->GetComplementRank();
	World.Sendrecv(&xlocnz, 1, MPIType<IU>(), diagneigh, TRNNZ, &trxlocnz, 1, MPIType<IU>(), diagneigh, TRNNZ);
	World.Sendrecv(&roffst, 1, MPIType<IU>(), diagneigh, TROST, &roffset, 1, MPIType<IU>(), diagneigh, TROST);
	World.Sendrecv(&luntil, 1, MPIType<IU>(), diagneigh, TRLUT, &lenuntil, 1, MPIType<IU>(), diagneigh, TRLUT);
	
	// ABAB: Important observation is that local indices (given by x.ind) is 32-bit addressible
	// Copy them to 32 bit integers and transfer that to save 50% of off-node bandwidth
	trxinds = new int32_t[trxlocnz];
	try
	{
		int32_t * temp_xind = new int32_t[xlocnz];
		for(int i=0; i< xlocnz; ++i)	temp_xind[i] = (int32_t) x.ind[i];
		World.Sendrecv(temp_xind, xlocnz, MPIType<int32_t>(), diagneigh, TRI, trxinds, trxlocnz, MPIType<int32_t>(), diagneigh, TRI);
		delete [] temp_xind;
		if(!indexisvalue)
		{
			trxnums = new IU[trxlocnz];
			World.Sendrecv(const_cast<IU*>(SpHelper::p2a(x.num)), xlocnz, MPIType<IU>(), diagneigh, TRX, trxnums, trxlocnz, MPIType<IU>(), diagneigh, TRX);
		}
		transform(trxinds, trxinds+trxlocnz, trxinds, bind2nd(plus<IU>(), roffset)); // fullydist indexing (p pieces) -> matrix indexing (sqrt(p) pieces)
	}
	catch(MPI::Exception e)
	{
		cerr << "Exception during Sendrecv file" << endl;
		cerr << e.Get_error_string() << endl;
	}
}	

/**
 * Step 2 of the sparse SpMV algorithm 
 * @param[in,out]   trxinds, trxnums { deallocated }
 * @param[in,out]   indacc, numacc { allocated }
 * @param[in,out]	accnz { set }
 * @param[in] 		trxlocnz, lenuntil, indexisvalue
 **/
template<typename IU>
void AllGatherVector(MPI::Intracomm & ColWorld, IU trxlocnz, IU lenuntil, int32_t * & trxinds, IU * & trxnums, int32_t * & indacc, IU * & numacc, int & accnz, bool indexisvalue)
{
	int colneighs = ColWorld.Get_size();
	int colrank = ColWorld.Get_rank();
	int * colnz = new int[colneighs];
	colnz[colrank] = static_cast<int>(trxlocnz);
	ColWorld.Allgather(MPI::IN_PLACE, 1, MPI::INT, colnz, 1, MPI::INT);
	int * dpls = new int[colneighs]();	// displacements (zero initialized pid) 
	std::partial_sum(colnz, colnz+colneighs-1, dpls+1);
	accnz = std::accumulate(colnz, colnz+colneighs, 0);
	indacc = new int32_t[accnz];
	numacc = new IU[accnz];
	
	// ABAB: Future issues here, colnz is of type int (MPI limitation)
	// What if the aggregate vector size along the processor row/column is not 32-bit addressible?
	// This will happen when n/sqrt(p) > 2^31
	// Currently we can solve a small problem (scale 32) with 4096 processor
	// For a medium problem (scale 35), we'll need 32K processors which gives sqrt(p) ~ 180
	// 2^35 / 180 ~ 2^29 / 3 which is not an issue !
	
#ifdef TIMING
	World.Barrier();
	double t0=MPI::Wtime();
#endif
	ColWorld.Allgatherv(trxinds, trxlocnz, MPIType<int32_t>(), indacc, colnz, dpls, MPIType<int32_t>());
#ifdef TIMING
	World.Barrier();
	double t1=MPI::Wtime();
	cblas_allgathertime += (t1-t0);
#endif
	
	delete [] trxinds;
	if(indexisvalue)
	{
		IU lenuntilcol;
		if(colrank == 0)  lenuntilcol = lenuntil;
		ColWorld.Bcast(&lenuntilcol, 1, MPIType<IU>(), 0);
		for(int i=0; i< accnz; ++i)	// fill numerical values from indices
		{
			numacc[i] = indacc[i] + lenuntilcol;
		}
	}
	else
	{
		ColWorld.Allgatherv(trxnums, trxlocnz, MPIType<IU>(), numacc, colnz, dpls, MPIType<IU>());
		delete [] trxnums;
	}	
	DeleteAll(colnz,dpls);
}	

/**
  * Step 3 of the sparse SpMV algorithm, with the semiring 
 **/
template<typename SR, typename T_promote, typename IU, typename MATRIX>
void LocalSpMV(MATRIX A, int rowneighs, OptBuf<int32_t, T_promote > & optbuf, int32_t * & indacc, IU * & numacc, int32_t * & sendindbuf, T_promote * & sendnumbuf, int * & sdispls, int * sendcnt, int accnz, bool indexisvalue)
{	
	if(optbuf.totmax > 0)	// graph500 optimization enabled
	{ 
		if(A.spSeq->getnsplit() > 0)
		{
			SpParHelper::Print("Preallocated buffers can not be used with multithreaded code yet\n");
			IU * tmpindbuf;
			IU * tmpindacc = new IU[accnz];
			for(int i=0; i< accnz; ++i) tmpindacc[i] = indacc[i];
			
			// sendindbuf/sendnumbuf/sdispls are all allocated and filled by dcsc_gespmv_threaded
			int totalsent = dcsc_gespmv_threaded<SR> (*(A.spSeq), tmpindacc, numacc, static_cast<IU>(accnz), tmpindbuf, sendnumbuf, sdispls, rowneighs);	
			
			delete [] tmpindacc;
			sendindbuf = new int32_t[totalsent];
			for(int i=0; i< totalsent; ++i)	sendindbuf[i] = tmpindbuf[i];
			delete [] tmpindbuf;
			
			for(int i=0; i<rowneighs-1; ++i)
				sendcnt[i] = sdispls[i+1] + sdispls[i];	
			sendcnt[rowneighs-1] = totalsent - sdispls[rowneighs-1];
		}
		else
		{
			dcsc_gespmv<SR> (*(A.spSeq), indacc, numacc, accnz, optbuf.inds, optbuf.nums, sendcnt, optbuf.dspls, rowneighs, indexisvalue);
		}
		DeleteAll(indacc,numacc);
	}
	else
	{
		if(A.spSeq->getnsplit() > 0)
		{
			IU * tmpindbuf;
			IU * tmpindacc = new IU[accnz];
			for(int i=0; i< accnz; ++i) tmpindacc[i] = indacc[i];
			delete [] indacc;
			
			// sendindbuf/sendnumbuf/sdispls are all allocated and filled by dcsc_gespmv_threaded
			int totalsent = dcsc_gespmv_threaded<SR> (*(A.spSeq), tmpindacc, numacc, static_cast<IU>(accnz), tmpindbuf, sendnumbuf, sdispls, rowneighs);	
			
			delete [] tmpindacc;
			sendindbuf = new int32_t[totalsent];
			for(int i=0; i< totalsent; ++i)	sendindbuf[i] = tmpindbuf[i];
			DeleteAll(tmpindbuf, numacc);
			for(int i=0; i<rowneighs-1; ++i)
				sendcnt[i] = sdispls[i+1] - sdispls[i];
			sendcnt[rowneighs-1] = totalsent - sdispls[rowneighs-1];
		}
		else
		{
			// serial SpMV with sparse vector
			vector< IU > indy;
			vector< T_promote >  numy;
			
			IU * tmpindacc = new IU[accnz];
			for(int i=0; i< accnz; ++i) tmpindacc[i] = indacc[i];
			delete [] indacc;
			
			dcsc_gespmv<SR>(*(A.spSeq), tmpindacc, numacc, static_cast<IU>(accnz), indy, numy);	// actual multiplication
			DeleteAll(tmpindacc, numacc);
			
			IU bufsize = indy.size();	// as compact as possible
			sendindbuf = new int32_t[bufsize];	
			sendnumbuf = new T_promote[bufsize];
			IU perproc = A.getlocalrows() / static_cast<IU>(rowneighs);	
			
			int k = 0;	// index to buffer
			for(int i=0; i<rowneighs; ++i)		
			{
				IU end_this = (i==rowneighs-1) ? A.getlocalrows(): (i+1)*perproc;
				while(k < bufsize && indy[k] < end_this) 
				{
					sendindbuf[k] = static_cast<int32_t>(indy[k] - i*perproc);
					sendnumbuf[k] = numy[k];
					++sendcnt[i];
					++k; 
				}
			}
			sdispls = new int[rowneighs]();	
			partial_sum(sendcnt, sendcnt+rowneighs-1, sdispls+1); 
		}
	}
}

/**
  * This is essentially a SpMV for BFS because it lacks the semiring.
  * It naturally justs selects columns of A (adjacencies of frontier) and 
  * merges with the minimum entry succeeding. 
 ** TODO: Refactor LocalSpMV !
template <typename IU, typename NUM, typename UDER>
FullyDistSpVec<IU,typename promote_trait<NUM,IU>::T_promote>  SpMV 
	(const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,IU> & x, bool indexisvalue, OptBuf<int32_t, typename promote_trait<NUM,IU>::T_promote > & optbuf)
{
	typedef typename promote_trait<NUM,IU>::T_promote T_promote;
	CheckSpMVCompliance(A,x);

	MPI::Intracomm World = x.commGrid->GetWorld();
	MPI::Intracomm ColWorld = x.commGrid->GetColWorld();
	MPI::Intracomm RowWorld = x.commGrid->GetRowWorld();

	int accnz;
	IU trxlocnz, lenuntil;
	int32_t *trxinds, *indacc;
	IU *trxnums, *numacc;
	TransposeVector(World, x, trxlocnz, lenuntil, trxinds, trxnums, indexisvalue);			// trxinds (and potentially trxnums) is allocated
	AllGatherVector(ColWorld, trxlocnz, lenuntil, trxinds, trxnums, indacc, numacc, accnz, indexisvalue);	// trxinds (and potentially trxnums) is deallocated, indacc/numacc allocated
	
	FullyDistSpVec<IU, T_promote> y ( x.commGrid, A.getnrow());	// identity doesn't matter for sparse vectors
	int rowneighs = RowWorld.Get_size();
	int * sendcnt = new int[rowneighs]();	
	int32_t * sendindbuf;	
	T_promote * sendnumbuf;
	int * sdispls;
	LocalSpMV(A, rowneighs, optbuf, indacc, numacc, sendindbuf, sendnumbuf, sdispls, sendcnt, accnz, indexisvalue);	// indacc/numacc deallocated, sendindbuf/sendnumbuf/sdispls allocated
}
 **/

//! The last parameter is a hint to the function 
//! If indexisvalues = true, then we do not need to transfer values for x
//! This happens for BFS iterations with boolean matrices and integer rhs vectors
template <typename SR, typename IU, typename NUM, typename UDER>
FullyDistSpVec<IU,typename promote_trait<NUM,IU>::T_promote>  SpMV 
	(const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,IU> & x, bool indexisvalue, OptBuf<int32_t, typename promote_trait<NUM,IU>::T_promote > & optbuf)
{
	typedef typename promote_trait<NUM,IU>::T_promote T_promote;
	CheckSpMVCompliance(A,x);

	MPI::Intracomm World = x.commGrid->GetWorld();
	MPI::Intracomm ColWorld = x.commGrid->GetColWorld();
	MPI::Intracomm RowWorld = x.commGrid->GetRowWorld();

	int accnz;
	IU trxlocnz, lenuntil;
	int32_t *trxinds, *indacc;
	IU *trxnums, *numacc;
	TransposeVector(World, x, trxlocnz, lenuntil, trxinds, trxnums, indexisvalue);
	AllGatherVector(ColWorld, trxlocnz, lenuntil, trxinds, trxnums, indacc, numacc, accnz, indexisvalue);
	
	FullyDistSpVec<IU, T_promote> y ( x.commGrid, A.getnrow());	// identity doesn't matter for sparse vectors
	int rowneighs = RowWorld.Get_size();
	int * sendcnt = new int[rowneighs]();	
	int32_t * sendindbuf;	
	T_promote * sendnumbuf;
	int * sdispls;
	LocalSpMV<SR>(A, rowneighs, optbuf, indacc, numacc, sendindbuf, sendnumbuf, sdispls, sendcnt, accnz, indexisvalue);	// indacc/numacc deallocated, sendindbuf/sendnumbuf/sdispls allocated
	
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
	T_promote * recvnumbuf = new T_promote[totrecv];

	#ifdef TIMING
	World.Barrier();
	double t2=MPI::Wtime();
	#endif
	if(optbuf.totmax > 0 && A.spSeq->getnsplit() == 0)	// graph500 optimization enabled
	{
		RowWorld.Alltoallv(optbuf.inds, sendcnt, optbuf.dspls, MPIType<int32_t>(), recvindbuf, recvcnt, rdispls, MPIType<int32_t>());  
		RowWorld.Alltoallv(optbuf.nums, sendcnt, optbuf.dspls, MPIType<T_promote>(), recvnumbuf, recvcnt, rdispls, MPIType<T_promote>());  // T_promote=NUM
		delete [] sendcnt;
	}
	else
	{
/*		ofstream oput;
		x.commGrid->OpenDebugFile("Send", oput);
		oput << "To displacements: "; copy(sdispls, sdispls+rowneighs, ostream_iterator<int>(oput, " ")); oput << endl;
		oput << "To counts: "; copy(sendcnt, sendcnt+rowneighs, ostream_iterator<int>(oput, " ")); oput << endl;
		for(int i=0; i< rowneighs; ++i)
		{
			oput << "To neighbor: " << i << endl; 
			copy(sendindbuf+sdispls[i], sendindbuf+sdispls[i]+sendcnt[i], ostream_iterator<IU>(oput, " ")); oput << endl;
			copy(sendnumbuf+sdispls[i], sendnumbuf+sdispls[i]+sendcnt[i], ostream_iterator<T_promote>(oput, " ")); oput << endl;
		}
		oput.close(); 
*/
		RowWorld.Alltoallv(sendindbuf, sendcnt, sdispls, MPIType<int32_t>(), recvindbuf, recvcnt, rdispls, MPIType<int32_t>());  
		RowWorld.Alltoallv(sendnumbuf, sendcnt, sdispls, MPIType<T_promote>(), recvnumbuf, recvcnt, rdispls, MPIType<T_promote>());  
		DeleteAll(sendindbuf, sendnumbuf);
		DeleteAll(sendcnt, sdispls);
	}
	#ifdef TIMING
	World.Barrier();
	double t3=MPI::Wtime();
	cblas_alltoalltime += (t3-t2);
	#endif

//	ofstream output;
//	A.commGrid->OpenDebugFile("Recv", output);
//	copy(recvindbuf, recvindbuf+totrecv, ostream_iterator<IU>(output," ")); output << endl;
//	output.close();

#ifndef HEAPMERGE
	// Alternative 1: SPA-like data structure
	DeleteAll(recvcnt, rdispls);
	IU ysize = y.MyLocLength();	// my local length is only O(n/p)
	T_promote * localy = new T_promote[ysize];
	bool * isthere = new bool[ysize];
	vector<IU> nzinds;	// nonzero indices		
	fill_n(isthere, ysize, false);
	
	for(int i=0; i< totrecv; ++i)
	{
		int32_t topush = recvindbuf[i];
		if(!isthere[topush])
		{
			localy[topush] = recvnumbuf[i];	// initial assignment
			nzinds.push_back(topush);
			isthere[topush] = true;
		} 
		else
		{
			localy[topush] = SR::add(localy[topush], recvnumbuf[i]);	
		}
	}
	DeleteAll(isthere, recvindbuf, recvnumbuf);
	sort(nzinds.begin(), nzinds.end());
	int nnzy = nzinds.size();
	y.ind.resize(nnzy);
	y.num.resize(nnzy);
	for(int i=0; i< nnzy; ++i)
	{
		y.ind[i] = nzinds[i];
		y.num[i] = localy[nzinds[i]]; 	
	}
	delete [] localy;

#else
	// Alternative 2: Heap-merge
	int32_t hsize = 0;		
	int32_t inf = numeric_limits<int32_t>::min();
	int32_t sup = numeric_limits<int32_t>::max(); 
        KNHeap< int32_t, int32_t > sHeap(sup, inf); 
	int * processed = new int[rowneighs]();
	for(int i=0; i<rowneighs; ++i)
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
	while(hsize > 0)
	{
		sHeap.deleteMin(&key, &locv);
		IU deref = rdispls[locv] + processed[locv];
		if(y.ind.back() == static_cast<IU>(key))	// y.ind is surely not empty
		{
			y.num.back() = SR::add(y.num.back(), recvnumbuf[deref]);
			// ABAB: Benchmark actually allows us to be non-deterministic in terms of parent selection
			// We can just skip this addition operator (if it's a max/min select)
		} 
		else
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

	return y;
}


template <typename SR, typename IU, typename NUM, typename NUV, typename UDER> 
FullyDistVec<IU,typename promote_trait<NUM,NUV>::T_promote>  SpMV 
	(const SpParMat<IU,NUM,UDER> & A, const FullyDistVec<IU,NUV> & x )
{
	typedef typename promote_trait<NUM,NUV>::T_promote T_promote;
	CheckSpMVCompliance(A, x);

	MPI::Intracomm World = x.commGrid->GetWorld();
	MPI::Intracomm ColWorld = x.commGrid->GetColWorld();
	MPI::Intracomm RowWorld = x.commGrid->GetRowWorld();

	int xsize = (int) x.LocArrSize();
	int trxsize = 0;

	int diagneigh = x.commGrid->GetComplementRank();
	World.Sendrecv(&xsize, 1, MPI::INT, diagneigh, TRX, &trxsize, 1, MPI::INT, diagneigh, TRX);
	
	NUV * trxnums = new NUV[trxsize];
	World.Sendrecv(const_cast<NUV*>(SpHelper::p2a(x.arr)), xsize, MPIType<NUV>(), diagneigh, TRX, trxnums, trxsize, MPIType<NUV>(), diagneigh, TRX);

	int colneighs = ColWorld.Get_size();
	int colrank = ColWorld.Get_rank();
	int * colsize = new int[colneighs];
	colsize[colrank] = trxsize;
	ColWorld.Allgather(MPI::IN_PLACE, 1, MPI::INT, colsize, 1, MPI::INT);
	int * dpls = new int[colneighs]();	// displacements (zero initialized pid) 
	std::partial_sum(colsize, colsize+colneighs-1, dpls+1);
	int accsize = std::accumulate(colsize, colsize+colneighs, 0);
	NUV * numacc = new NUV[accsize];

	ColWorld.Allgatherv(trxnums, trxsize, MPIType<NUV>(), numacc, colsize, dpls, MPIType<NUV>());
	delete [] trxnums;

	// serial SpMV with dense vector
	T_promote id = SR::id();
	IU ysize = A.getlocalrows();
	T_promote * localy = new T_promote[ysize];
	fill_n(localy, ysize, id);		
	dcsc_gespmv<SR>(*(A.spSeq), numacc, localy);	
	
	//ofstream oput;
	//A.commGrid->OpenDebugFile("localy", oput);
	//copy(localy, localy+ysize, ostream_iterator<T_promote>(oput, " ")); oput << endl;
	//oput.close();

	DeleteAll(numacc,colsize, dpls);

	// FullyDistVec<IT,NT>(shared_ptr<CommGrid> grid, IT globallen, NT initval, NT id)
	FullyDistVec<IU, T_promote> y ( x.commGrid, A.getnrow(), id);
	IU yintlen = y.MyRowLength();
	
	int rowneighs = RowWorld.Get_size();
	int rowrank = RowWorld.Get_rank();
	IU begptr, endptr;
	for(int i=0; i< rowneighs; ++i)
	{
		begptr = y.RowLenUntil(i);
		if(i == rowneighs-1)
		{
			endptr = ysize;
		}
		else
		{
			endptr = y.RowLenUntil(i+1);
		}
		// IntraComm::Reduce(sendbuf, recvbuf, count, type, op, root), recvbuf is irrelevant except root
                RowWorld.Reduce(localy+begptr, SpHelper::p2a(y.arr), endptr-begptr, MPIType<T_promote>(), SR::mpi_op(), i);
	}
	delete [] localy;
	return y;
}

	
template <typename SR, typename IU, typename NUM, typename NUV, typename UDER> 
FullyDistSpVec<IU,typename promote_trait<NUM,NUV>::T_promote>  SpMV 
	(const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,NUV> & x)
{
	typedef typename promote_trait<NUM,NUV>::T_promote T_promote;
	CheckSpMVCompliance(A, x);

	MPI::Intracomm World = x.commGrid->GetWorld();
	MPI::Intracomm ColWorld = x.commGrid->GetColWorld();
	MPI::Intracomm RowWorld = x.commGrid->GetRowWorld();

	int xlocnz = (int) x.getlocnnz();
	int trxlocnz = 0;
	int roffst = x.RowLenUntil();
	int offset;

	int diagneigh = x.commGrid->GetComplementRank();
	World.Sendrecv(&xlocnz, 1, MPI::INT, diagneigh, TRX, &trxlocnz, 1, MPI::INT, diagneigh, TRX);
	World.Sendrecv(&roffst, 1, MPI::INT, diagneigh, TROST, &offset, 1, MPI::INT, diagneigh, TROST);
	
	IU * trxinds = new IU[trxlocnz];
	NUV * trxnums = new NUV[trxlocnz];
	World.Sendrecv(const_cast<IU*>(SpHelper::p2a(x.ind)), xlocnz, MPIType<IU>(), diagneigh, TRX, trxinds, trxlocnz, MPIType<IU>(), diagneigh, TRX);
	World.Sendrecv(const_cast<NUV*>(SpHelper::p2a(x.num)), xlocnz, MPIType<NUV>(), diagneigh, TRX, trxnums, trxlocnz, MPIType<NUV>(), diagneigh, TRX);
	transform(trxinds, trxinds+trxlocnz, trxinds, bind2nd(plus<IU>(), offset)); // fullydist indexing (n pieces) -> matrix indexing (sqrt(p) pieces)

	int colneighs = ColWorld.Get_size();
	int colrank = ColWorld.Get_rank();
	int * colnz = new int[colneighs];
	colnz[colrank] = trxlocnz;
	ColWorld.Allgather(MPI::IN_PLACE, 1, MPI::INT, colnz, 1, MPI::INT);
	int * dpls = new int[colneighs]();	// displacements (zero initialized pid) 
	std::partial_sum(colnz, colnz+colneighs-1, dpls+1);
	int accnz = std::accumulate(colnz, colnz+colneighs, 0);
	IU * indacc = new IU[accnz];
	NUV * numacc = new NUV[accnz];

	// ABAB: Future issues here, colnz is of type int (MPI limitation)
	// What if the aggregate vector size along the processor row/column is not 32-bit addressible?
	ColWorld.Allgatherv(trxinds, trxlocnz, MPIType<IU>(), indacc, colnz, dpls, MPIType<IU>());
	ColWorld.Allgatherv(trxnums, trxlocnz, MPIType<NUV>(), numacc, colnz, dpls, MPIType<NUV>());
	DeleteAll(trxinds, trxnums);

	// serial SpMV with sparse vector
	vector< IU > indy;
	vector< T_promote >  numy;

	dcsc_gespmv<SR>(*(A.spSeq), indacc, numacc, static_cast<IU>(accnz), indy, numy);	// actual multiplication

	DeleteAll(indacc, numacc);
	DeleteAll(colnz, dpls);

	FullyDistSpVec<IU, T_promote> y ( x.commGrid, A.getnrow());	// identity doesn't matter for sparse vectors
	IU yintlen = y.MyRowLength();

	int rowneighs = RowWorld.Get_size();
	vector< vector<IU> > sendind(rowneighs);
	vector< vector<T_promote> > sendnum(rowneighs);
	typename vector<IU>::size_type outnz = indy.size();
	for(typename vector<IU>::size_type i=0; i< outnz; ++i)
	{
		IU locind;
		int rown = y.OwnerWithinRow(yintlen, indy[i], locind);
		sendind[rown].push_back(locind);
		sendnum[rown].push_back(numy[i]);
	}

	IU * sendindbuf = new IU[outnz];
	T_promote * sendnumbuf = new T_promote[outnz];
	int * sendcnt = new int[rowneighs];
	int * sdispls = new int[rowneighs];
	for(int i=0; i<rowneighs; ++i)
		sendcnt[i] = sendind[i].size();

	int * rdispls = new int[rowneighs];
	int * recvcnt = new int[rowneighs];
	RowWorld.Alltoall(sendcnt, 1, MPI::INT, recvcnt, 1, MPI::INT);	// share the request counts 

	sdispls[0] = 0;
	rdispls[0] = 0;
	for(int i=0; i<rowneighs-1; ++i)
	{
		sdispls[i+1] = sdispls[i] + sendcnt[i];
		rdispls[i+1] = rdispls[i] + recvcnt[i];
	}
	int totrecv = accumulate(recvcnt,recvcnt+rowneighs,0);
	IU * recvindbuf = new IU[totrecv];
	T_promote * recvnumbuf = new T_promote[totrecv];

	for(int i=0; i<rowneighs; ++i)
	{
		copy(sendind[i].begin(), sendind[i].end(), sendindbuf+sdispls[i]);
		vector<IU>().swap(sendind[i]);
	}
	for(int i=0; i<rowneighs; ++i)
	{
		copy(sendnum[i].begin(), sendnum[i].end(), sendnumbuf+sdispls[i]);
		vector<T_promote>().swap(sendnum[i]);
	}
		
	RowWorld.Alltoallv(sendindbuf, sendcnt, sdispls, MPIType<IU>(), recvindbuf, recvcnt, rdispls, MPIType<IU>());  
	RowWorld.Alltoallv(sendnumbuf, sendcnt, sdispls, MPIType<T_promote>(), recvnumbuf, recvcnt, rdispls, MPIType<T_promote>());  
	DeleteAll(sendindbuf, sendnumbuf);
	DeleteAll(sendcnt, recvcnt, sdispls, rdispls);
		
	// define a SPA-like data structure
	IU ysize = y.MyLocLength();
	T_promote * localy = new T_promote[ysize];
	bool * isthere = new bool[ysize];
	vector<IU> nzinds;	// nonzero indices		
	fill_n(isthere, ysize, false);
	
	for(int i=0; i< totrecv; ++i)
	{
		if(!isthere[recvindbuf[i]])
		{
			localy[recvindbuf[i]] = recvnumbuf[i];	// initial assignment
			nzinds.push_back(recvindbuf[i]);
			isthere[recvindbuf[i]] = true;
		} 
		else
		{
			localy[recvindbuf[i]] = SR::add(localy[recvindbuf[i]], recvnumbuf[i]);	
		}
	}
	DeleteAll(isthere, recvindbuf, recvnumbuf);
	sort(nzinds.begin(), nzinds.end());
	int nnzy = nzinds.size();
	y.ind.resize(nnzy);
	y.num.resize(nnzy);
	for(int i=0; i< nnzy; ++i)
	{
		y.ind[i] = nzinds[i];
		y.num[i] = localy[nzinds[i]]; 	
	}
	delete [] localy;
	return y;
}
	

template <typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB> 
SpParMat<IU,typename promote_trait<NU1,NU2>::T_promote,typename promote_trait<UDERA,UDERB>::T_promote> EWiseMult 
	(const SpParMat<IU,NU1,UDERA> & A, const SpParMat<IU,NU2,UDERB> & B , bool exclude)
{
	typedef typename promote_trait<NU1,NU2>::T_promote N_promote;
	typedef typename promote_trait<UDERA,UDERB>::T_promote DER_promote;

	if(*(A.commGrid) == *(B.commGrid))	
	{
		DER_promote * result = new DER_promote( EWiseMult(*(A.spSeq),*(B.spSeq),exclude) );
		return SpParMat<IU, N_promote, DER_promote> (result, A.commGrid);
	}
	else
	{
		cout << "Grids are not comparable elementwise multiplication" << endl; 
		MPI::COMM_WORLD.Abort(GRIDMISMATCH);
		return SpParMat< IU,N_promote,DER_promote >();
	}
}
	
template <typename RETT, typename RETDER, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB, typename _BinaryOperation> 
SpParMat<IU,RETT,RETDER> EWiseApply 
	(const SpParMat<IU,NU1,UDERA> & A, const SpParMat<IU,NU2,UDERB> & B, _BinaryOperation __binary_op, bool notB, const NU2& defaultBVal)
{
	//typedef typename promote_trait<NU1,NU2>::T_promote N_promote;
	//typedef typename promote_trait<UDERA,UDERB>::T_promote DER_promote;

	if(*(A.commGrid) == *(B.commGrid))	
	{
		RETDER * result = new RETDER( EWiseApply<RETT>(*(A.spSeq),*(B.spSeq), __binary_op, notB, defaultBVal) );
		return SpParMat<IU, RETT, RETDER> (result, A.commGrid);
	}
	else
	{
		cout << "Grids are not comparable elementwise apply" << endl; 
		MPI::COMM_WORLD.Abort(GRIDMISMATCH);
		return SpParMat< IU,RETT,RETDER >();
	}
}


/**
 * if exclude is true, then we prune all entries W[i] != zero from V 
 * if exclude is false, then we perform a proper elementwise multiplication
**/
template <typename IU, typename NU1, typename NU2>
SpParVec<IU,typename promote_trait<NU1,NU2>::T_promote> EWiseMult 
	(const SpParVec<IU,NU1> & V, const DenseParVec<IU,NU2> & W , bool exclude, NU2 zero)
{
	typedef typename promote_trait<NU1,NU2>::T_promote T_promote;

	if(*(V.commGrid) == *(W.commGrid))	
	{
		SpParVec< IU, T_promote> Product(V.commGrid);
		Product.length = V.length;
		if(Product.diagonal)
		{
			if(exclude)
			{
				IU size= V.ind.size();
				for(IU i=0; i<size; ++i)
				{
					if(W.arr.size() <= V.ind[i] || W.arr[V.ind[i]] == zero) 	// keep only those
					{
						Product.ind.push_back(V.ind[i]);
						Product.num.push_back(V.num[i]);
					}
				}		
			}	
			else
			{
				IU size= V.ind.size();
				for(IU i=0; i<size; ++i)
				{
					if(W.arr.size() > V.ind[i] && W.arr[V.ind[i]] != zero) 	// keep only those
					{
						Product.ind.push_back(V.ind[i]);
						Product.num.push_back(V.num[i] * W.arr[V.ind[i]]);
					}
				}
			}
		}
		return Product;
	}
	else
	{
		cout << "Grids are not comparable elementwise multiplication" << endl; 
		MPI::COMM_WORLD.Abort(GRIDMISMATCH);
		return SpParVec< IU,T_promote>();
	}
}

/**
 * if exclude is true, then we prune all entries W[i] != zero from V 
 * if exclude is false, then we perform a proper elementwise multiplication
**/
template <typename IU, typename NU1, typename NU2>
FullyDistSpVec<IU,typename promote_trait<NU1,NU2>::T_promote> EWiseMult 
	(const FullyDistSpVec<IU,NU1> & V, const FullyDistVec<IU,NU2> & W , bool exclude, NU2 zero)
{
	typedef typename promote_trait<NU1,NU2>::T_promote T_promote;

	if(*(V.commGrid) == *(W.commGrid))	
	{
		FullyDistSpVec< IU, T_promote> Product(V.commGrid);
		if(V.glen != W.glen)
		{
			cerr << "Vector dimensions don't match for EWiseMult\n";
			MPI::COMM_WORLD.Abort(DIMMISMATCH);
		}
		else
		{
			Product.glen = V.glen;
			IU size= V.getlocnnz();
			if(exclude)
			{
				#if defined(_OPENMP) && defined(CBLAS_EXPERIMENTAL)	// not faster than serial
				int actual_splits = cblas_splits * 1;	// 1 is the parallel slackness
				vector <IU> tlosizes (actual_splits, 0);
				vector < vector<IU> > tlinds(actual_splits);
				vector < vector<T_promote> > tlnums(actual_splits);
				IU tlsize = size / actual_splits;
				#pragma omp parallel for //schedule(dynamic, 1)
				for(IU t = 0; t < actual_splits; ++t)
				{
					IU tlbegin = t*tlsize;
					IU tlend = (t==actual_splits-1)? size : (t+1)*tlsize;
					for(IU i=tlbegin; i<tlend; ++i)
					{
						if(W.arr[V.ind[i]] == zero) 	// keep only those
						{
							tlinds[t].push_back(V.ind[i]);
							tlnums[t].push_back(V.num[i]);
							tlosizes[t]++;
						}
					}
				}
				vector<IU> prefix_sum(actual_splits+1,0);
				partial_sum(tlosizes.begin(), tlosizes.end(), prefix_sum.begin()+1); 
				Product.ind.resize(prefix_sum[actual_splits]);
				Product.num.resize(prefix_sum[actual_splits]);
			
				#pragma omp parallel for //schedule(dynamic, 1)
				for(IU t=0; t< actual_splits; ++t)
				{
					copy(tlinds[t].begin(), tlinds[t].end(), Product.ind.begin()+prefix_sum[t]);
					copy(tlnums[t].begin(), tlnums[t].end(), Product.num.begin()+prefix_sum[t]);
				}
				#else
				for(IU i=0; i<size; ++i)
				{
					if(W.arr[V.ind[i]] == zero)     // keep only those
					{
                        	       		Product.ind.push_back(V.ind[i]);
                                		Product.num.push_back(V.num[i]);
                                      	}	
				}
				#endif
			}
			else
			{
				for(IU i=0; i<size; ++i)
				{
					if(W.arr[V.ind[i]] != zero) 	// keep only those
					{
						Product.ind.push_back(V.ind[i]);
						Product.num.push_back(V.num[i] * W.arr[V.ind[i]]);
					}
				}
			}
		}
		return Product;
	}
	else
	{
		cout << "Grids are not comparable elementwise multiplication" << endl; 
		MPI::COMM_WORLD.Abort(GRIDMISMATCH);
		return FullyDistSpVec< IU,T_promote>();
	}
}


/**
 * This is a glorified EWiseMult that takes a general filter in the form of binary_op()
 * \attention {Since the signature is like EWiseApply(SparseVec V, DenseVec W,...) the binary function should be 
 * written keeping in mind that the first operand (x) is from the sparse vector V, and the second operand (y) is from the dense vector W}
 * @param[binary_op]  if ( y == -1 ) ? x: -1 
 *      \n              then we get the 'exclude = false' effect of EWiseMult
 * In this example, the function always returns -1 if x is -1 (regardless of the value of y), which makes sense because x is from the sparse vector "fringe". The four cases are: 
 *	A) if fringe[i] is nonexistent and parents[i] == -1, then pro = _binary_op(-1,-1) is executed which will return -1 and it will NOT exist in product. Correct.
 *	B) if fringe[i] is nonexistent and parents[i] == d for some d>=0, then pro = _binary_op(-1,d) returns -1 and will NOT exist in product. Correct.
 *	C) if fringe[i] = k for some k>=0 and parents[i] == -1, then pro = _binary_op(k,-1) is executed and returns k. Correct.
 *	D) if fringe[i] = k for some k>=0, and parents[i] == d for some d>=0, then pro = _binary_op(k,d) which returns -1 again. Correct.
**/
template <typename RET, typename IU, typename NU1, typename NU2, typename _BinaryOperation, typename _BinaryPredicate>
FullyDistSpVec<IU,RET> EWiseApply 
	(const FullyDistSpVec<IU,NU1> & V, const FullyDistVec<IU,NU2> & W , _BinaryOperation _binary_op, _BinaryPredicate _doOp, bool allowVNulls, NU1 Vzero)
{
	typedef RET T_promote; //typedef typename promote_trait<NU1,NU2>::T_promote T_promote;
	if(*(V.commGrid) == *(W.commGrid))	
	{
		FullyDistSpVec< IU, T_promote> Product(V.commGrid);
		FullyDistVec< IU, NU1> DV (V);
		if(V.glen != W.glen)
		{
			cerr << "Vector dimensions don't match for EWiseApply\n";
			MPI::COMM_WORLD.Abort(DIMMISMATCH);
		}
		else
		{
			Product.glen = V.glen;
			IU size= W.LocArrSize();
			IU spsize = V.getlocnnz();
			IU sp_iter = 0;
			if (allowVNulls)
			{
				// iterate over the dense vector
				for(IU i=0; i<size && sp_iter < spsize; ++i)
				{
					if(V.ind[sp_iter] == i)
					{
						if (_doOp(V.num[sp_iter], W.arr[i]))
						{
							Product.ind.push_back(i);
							Product.num.push_back(_binary_op(V.num[sp_iter], W.arr[i]));
						}
						sp_iter++;
					}
					else
					{
						if (_doOp(Vzero, W.arr[i]))
						{
							Product.ind.push_back(i);
							Product.num.push_back(_binary_op(Vzero, W.arr[i]));
						}
					}
				}
			}
			else
			{
				// iterate over the sparse vector
				for(sp_iter = 0; sp_iter < spsize; ++sp_iter)
				{
					if (_doOp(V.num[sp_iter], W.arr[V.ind[sp_iter]]))
					{
						Product.ind.push_back(V.ind[sp_iter]);
						Product.num.push_back(_binary_op(V.num[sp_iter], W.arr[V.ind[sp_iter]]));
					}
				}
			}
		}
		return Product;
	}
	else
	{
		cout << "Grids are not comparable for EWiseApply" << endl; 
		MPI::COMM_WORLD.Abort(GRIDMISMATCH);
		return FullyDistSpVec< IU,T_promote>();
	}
}

template <typename RET, typename IU, typename NU1, typename NU2, typename _BinaryOperation, typename _BinaryPredicate>
FullyDistSpVec<IU,RET> EWiseApply 
	(const FullyDistSpVec<IU,NU1> & V, const FullyDistSpVec<IU,NU2> & W , _BinaryOperation _binary_op, _BinaryPredicate _doOp, bool allowVNulls, bool allowWNulls, NU1 Vzero, NU2 Wzero)
{
	typedef RET T_promote; // typename promote_trait<NU1,NU2>::T_promote T_promote;
	if(*(V.commGrid) == *(W.commGrid))	
	{
		FullyDistSpVec< IU, T_promote> Product(V.commGrid);
		if(V.glen != W.glen)
		{
			cerr << "Vector dimensions don't match for EWiseApply\n";
			MPI::COMM_WORLD.Abort(DIMMISMATCH);
		}
		else
		{
			Product.glen = V.glen;
			typename vector< IU  >::const_iterator indV = V.ind.begin();
			typename vector< NU1 >::const_iterator numV = V.num.begin();
			typename vector< IU  >::const_iterator indW = W.ind.begin();
			typename vector< NU2 >::const_iterator numW = W.num.begin();
			
			while (indV < V.ind.end() && indW < W.ind.end())
			{
				if (*indV == *indW)
				{
					// overlap
					if (_doOp(*numV, *numW))
					{
						Product.ind.push_back(*indV);
						Product.num.push_back(_binary_op(*numV, *numW));
					}
					indV++; numV++;
					indW++; numW++;
				}
				else if (*indV < *indW)
				{
					// V has value but W does not
					if (allowWNulls)
					{
						if (_doOp(*numV, Wzero))
						{
							Product.ind.push_back(*indV);
							Product.num.push_back(_binary_op(*numV, Wzero));
						}
					}
					indV++; numV++;
				}
				else //(*indV > *indW)
				{
					// W has value but V does not
					if (allowVNulls)
					{
						if (_doOp(Vzero, *numW))
						{
							Product.ind.push_back(*indW);
							Product.num.push_back(_binary_op(Vzero, *numW));
						}
					}
					indW++; numW++;
				}
			}
			// clean up
			while (allowWNulls && indV < V.ind.end())
			{
				if (_doOp(*numV, Wzero))
				{
					Product.ind.push_back(*indV);
					Product.num.push_back(_binary_op(*numV, Wzero));
				}
				indV++; numV++;
			}
			while (allowVNulls && indW < W.ind.end())
			{
				if (_doOp(Vzero, *numW))
				{
					Product.ind.push_back(*indW);
					Product.num.push_back(_binary_op(Vzero, *numW));
				}
				indW++; numW++;
			}
		}
		return Product;
	}
	else
	{
		cout << "Grids are not comparable for EWiseApply" << endl; 
		MPI::COMM_WORLD.Abort(GRIDMISMATCH);
		return FullyDistSpVec< IU,T_promote>();
	}
}
#endif

