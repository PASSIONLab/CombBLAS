#ifndef _PAR_FRIENDS_H_
#define _PAR_FRIENDS_H_

#include "mpi.h"
#include "sys/time.h"
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
	IU ncolA = A.getncol();
	IU nrowB = B.getnrow();	

	if(ncolA != nrowB)
	{
		ostringstream outs;
		outs << "Can not multiply, dimensions does not match"<< endl;
		outs << ncolA << " != " << nrowB << endl;
		SpParHelper::Print(outs.str());
		MPI::COMM_WORLD.Abort(DIMMISMATCH);
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

	if(clearA) 
	{	
		delete A1seq;
	}	
	if(clearB) 
	{
		delete B1seq;
	}

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
	IU ncolA = A.getncol();
	IU nrowB = B.getnrow();	

	if(ncolA != nrowB)
	{
		ostringstream outs;
		outs << "Can not multiply, dimensions does not match"<< endl;
		outs << ncolA << " != " << nrowB << endl;
		SpParHelper::Print(outs.str());
		MPI::COMM_WORLD.Abort(DIMMISMATCH);
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

/**
 * Parallel A = B*C routine that uses one-sided MPI-2 features
 * General active target syncronization via MPI_Win_Post, MPI_Win_Start, MPI_Win_Complete, MPI_Win_Wait
 * Tested on my dual core Macbook with 1,4,9,16,25 MPI processes
 * No memory hog: splits the matrix into two along the column, prefetches the next half matrix while computing on the current one 
 **/  
template <typename SR, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB> 
SpParMat<IU,typename promote_trait<NU1,NU2>::T_promote,typename promote_trait<UDERA,UDERB>::T_promote> Mult_AnXBn_ActiveTarget 
		(const SpParMat<IU,NU1,UDERA> & A, const SpParMat<IU,NU2,UDERB> & B )

{
	typedef typename promote_trait<NU1,NU2>::T_promote N_promote;
	typedef typename promote_trait<UDERA,UDERB>::T_promote DER_promote;

	if(A.getncol() != B.getnrow())
	{
		cout<<"Can not multiply, dimensions does not match"<<endl;
		MPI::COMM_WORLD.Abort(DIMMISMATCH);
		return SpParMat< IU,N_promote,DER_promote >();
	}
	int stages, Aoffset, Boffset; 	// stages = inner dimension of matrix blocks
	shared_ptr<CommGrid> GridC = ProductGrid((A.commGrid).get(), (B.commGrid).get(), stages, Aoffset, Boffset);		

	IU C_m = A.spSeq->getnrow();
	IU C_n = B.spSeq->getncol();
		
	UDERA A1seq, A2seq;
	(A.spSeq)->Split( A1seq, A2seq); 
	
	// ABAB: It should be able to perform split/merge with the transpose option [single function call]
	const_cast< UDERB* >(B.spSeq)->Transpose();
	
	UDERB B1seq, B2seq;
	(B.spSeq)->Split( B1seq, B2seq);
	
	// Create row and column windows (collective operation, i.e. everybody exposes its window to others)
	vector<MPI::Win> rowwins1, rowwins2, colwins1, colwins2;
	SpParHelper::SetWindows((A.commGrid)->GetRowWorld(), A1seq, rowwins1);
	SpParHelper::SetWindows((A.commGrid)->GetRowWorld(), A2seq, rowwins2);
	SpParHelper::SetWindows((B.commGrid)->GetColWorld(), B1seq, colwins1);
	SpParHelper::SetWindows((B.commGrid)->GetColWorld(), B2seq, colwins2);

	SpParHelper::SetWinErrHandler(rowwins1);	// set the error handler to THROW_EXCEPTIONS
	SpParHelper::SetWinErrHandler(rowwins2);	
	SpParHelper::SetWinErrHandler(colwins1);	
	SpParHelper::SetWinErrHandler(colwins2);	

	// ABAB: We can optimize the call to create windows in the absence of passive synchronization
	// 	MPI_Info info; 
	// 	MPI_Info_create ( &info ); 
	// 	MPI_Info_set( info, "no_locks", "true" ); 
	// 	MPI_Win_create( . . ., info, . . . ); 
	// 	MPI_Info_free( &info ); 
	
	IU ** ARecvSizes1 = SpHelper::allocate2D<IU>(UDERA::esscount, stages);
	IU ** ARecvSizes2 = SpHelper::allocate2D<IU>(UDERA::esscount, stages);
	IU ** BRecvSizes1 = SpHelper::allocate2D<IU>(UDERB::esscount, stages);
	IU ** BRecvSizes2 = SpHelper::allocate2D<IU>(UDERB::esscount, stages);
		
	SpParHelper::GetSetSizes( A1seq, ARecvSizes1, (A.commGrid)->GetRowWorld());
	SpParHelper::GetSetSizes( A2seq, ARecvSizes2, (A.commGrid)->GetRowWorld());
	SpParHelper::GetSetSizes( B1seq, BRecvSizes1, (B.commGrid)->GetColWorld());
	SpParHelper::GetSetSizes( B2seq, BRecvSizes2, (B.commGrid)->GetColWorld());
	
	// Remotely fetched matrices are stored as pointers
	UDERA * ARecv1, * ARecv2; 
	UDERB * BRecv1, * BRecv2;
	vector< SpTuples<IU,N_promote>  *> tomerge;

	MPI::Group row_group = (A.commGrid)->GetRowWorld().Get_group();
	MPI::Group col_group = (B.commGrid)->GetColWorld().Get_group();

	int Aself = (A.commGrid)->GetRankInProcRow();
	int Bself = (B.commGrid)->GetRankInProcCol();	
	
	GridC->GetWorld().Barrier();	

	SpParHelper::Print("Writing to file\n");
	ofstream oput;
	GridC->OpenDebugFile("deb", oput);

	oput << "A1seq: " << A1seq.getnrow() << " " << A1seq.getncol() << " " << A1seq.getnnz() << endl;
	oput << "A2seq: " << A2seq.getnrow() << " " << A2seq.getncol() << " " << A2seq.getnnz() << endl;
	oput << "B1seq: " << B1seq.getnrow() << " " << B1seq.getncol() << " " << B1seq.getnnz() << endl;
	oput << "B2seq: " << B2seq.getnrow() << " " << B2seq.getncol() << " " << B2seq.getnnz() << endl;

	SpParHelper::Print("Wrote to file\n");
	GridC->GetWorld().Barrier();
	
	// Start exposure epochs to all windows
	try
	{
		SpParHelper::PostExposureEpoch(Aself, rowwins1, row_group);
		SpParHelper::PostExposureEpoch(Aself, rowwins2, row_group);
		SpParHelper::PostExposureEpoch(Bself, colwins1, col_group);
		SpParHelper::PostExposureEpoch(Bself, colwins2, col_group);
	}
    	catch(MPI::Exception e)
	{
		oput << "Exception while posting exposure epoch" << endl;
       		oput << e.Get_error_string() << endl;
     	}

	GridC->GetWorld().Barrier();
	SpParHelper::Print("Exposure epochs posted\n");	
	GridC->GetWorld().Barrier();

	
	int Aowner = (0+Aoffset) % stages;		
	int Bowner = (0+Boffset) % stages;
	try
	{	
		SpParHelper::AccessNFetch(ARecv1, Aowner, rowwins1, row_group, ARecvSizes1);
		SpParHelper::AccessNFetch(ARecv2, Aowner, rowwins2, row_group, ARecvSizes2);	// Start prefetching next half 

		for(int j=0; j< rowwins1.size(); ++j)	// wait for the first half to complete
			rowwins1[j].Complete();
		
		SpParHelper::AccessNFetch(BRecv1, Bowner, colwins1, col_group, BRecvSizes1);
		SpParHelper::AccessNFetch(BRecv2, Bowner, colwins2, col_group, BRecvSizes2);	// Start prefetching next half 
			
		for(int j=0; j< colwins1.size(); ++j)
			colwins1[j].Complete();
	}

    	catch(MPI::Exception e)
	{
		oput << "Exception while starting access epoch or the first fetch" << endl;
       		oput << e.Get_error_string() << endl;
     	}
	
	for(int i = 1; i < stages; ++i) 
	{
		SpParHelper::Print("Stage starting\n");
		SpTuples<IU,N_promote> * C_cont = MultiplyReturnTuples<SR>(*ARecv1, *BRecv1, false, true);

		SpParHelper::Print("Multiplied\n");

		if(!C_cont->isZero()) 
			tomerge.push_back(C_cont);

		SpParHelper::Print("Pushed back\n");

		
		GridC->GetWorld().Barrier();
		bool remoteA = false;
		bool remoteB = false;

		delete ARecv1;		// free the memory of the previous first half
		for(int j=0; j< rowwins2.size(); ++j)	// wait for the previous second half to complete
			rowwins2[j].Complete();
		SpParHelper::Print("Completed A\n");

		delete BRecv1;
		for(int j=0; j< colwins2.size(); ++j)	// wait for the previous second half to complete
			colwins2[j].Complete();
		
		SpParHelper::Print("Completed B\n");

		
		GridC->GetWorld().Barrier();
		Aowner = (i+Aoffset) % stages;		
		Bowner = (i+Boffset) % stages;
	

		// start fetching the current first half 
		SpParHelper::AccessNFetch(ARecv1, Aowner, rowwins1, row_group, ARecvSizes1);
		SpParHelper::AccessNFetch(BRecv1, Bowner, colwins1, col_group, BRecvSizes1);
	
		SpParHelper::Print("Fetched next\n");
		
		GridC->GetWorld().Barrier();
		// while multiplying the already completed previous second halfs
		C_cont = MultiplyReturnTuples<SR>(*ARecv2, *BRecv2, false, true);	
		if(!C_cont->isZero()) 
			tomerge.push_back(C_cont);

		SpParHelper::Print("Multiplied and pushed\n");
		GridC->GetWorld().Barrier();

		delete ARecv2;		// free memory of the previous second half
		delete BRecv2;

		for(int j=0; j< rowwins1.size(); ++j)	// wait for the current first half to complte
			rowwins1[j].Complete();
		for(int j=0; j< colwins1.size(); ++j)
			colwins1[j].Complete();
		
		SpParHelper::Print("Completed next\n");	
		GridC->GetWorld().Barrier();

		// start prefetching the current second half 
		SpParHelper::AccessNFetch(ARecv2, Aowner, rowwins2, row_group, ARecvSizes2);
		SpParHelper::AccessNFetch(BRecv2, Bowner, colwins2, col_group, BRecvSizes2);
	}
	//SpParHelper::Print("Stages finished\n");
	SpTuples<IU,N_promote> * C_cont = MultiplyReturnTuples<SR>(*ARecv1, *BRecv1, false, true);
	if(!C_cont->isZero()) 
		tomerge.push_back(C_cont);

	delete ARecv1;		// free the memory of the previous first half
	for(int j=0; j< rowwins2.size(); ++j)	// wait for the previous second half to complete
		rowwins2[j].Complete();
	delete BRecv1;
	for(int j=0; j< colwins2.size(); ++j)	// wait for the previous second half to complete
		colwins2[j].Complete();	

	C_cont = MultiplyReturnTuples<SR>(*ARecv2, *BRecv2, false, true);	
	if(!C_cont->isZero()) 
		tomerge.push_back(C_cont);
		
	delete ARecv2;
	delete BRecv2;

	SpHelper::deallocate2D(ARecvSizes1, UDERA::esscount);
	SpHelper::deallocate2D(ARecvSizes2, UDERA::esscount);
	SpHelper::deallocate2D(BRecvSizes1, UDERB::esscount);
	SpHelper::deallocate2D(BRecvSizes2, UDERB::esscount);
			
	DER_promote * C = new DER_promote(MergeAll<SR>(tomerge, C_m, C_n), false, NULL);	// First get the result in SpTuples, then convert to UDER
	for(int i=0; i<tomerge.size(); ++i)
	{
		delete tomerge[i];
	}

	// MPI_Win_Wait() works like a barrier as it waits for all origins to finish their remote memory operation on "this" window
	SpParHelper::WaitNFree(rowwins1);
	SpParHelper::WaitNFree(rowwins2);
	SpParHelper::WaitNFree(colwins1);
	SpParHelper::WaitNFree(colwins2);	
	
	(A.spSeq)->Merge(A1seq, A2seq);
	(B.spSeq)->Merge(B1seq, B2seq);	
	
	row_group.Free();
	col_group.Free();
	const_cast< UDERB* >(B.spSeq)->Transpose();	// transpose back to original
	return SpParMat<IU,N_promote,DER_promote> (C, GridC);		// return the result object
}


/**
 * Parallel A = B*C routine that uses one-sided MPI-2 features
 * Passive target syncronization via MPI_Win_Lock, MPI_Win_Unlock
 * No memory hog: splits the matrix into two along the column, prefetches the next half matrix while computing on the current one 
 **/  
template <typename SR, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB> 
SpParMat<IU,typename promote_trait<NU1,NU2>::T_promote,typename promote_trait<UDERA,UDERB>::T_promote> Mult_AnXBn_PassiveTarget 
		(const SpParMat<IU,NU1,UDERA> & A, const SpParMat<IU,NU2,UDERB> & B )

{
	typedef typename promote_trait<NU1,NU2>::T_promote N_promote;
	typedef typename promote_trait<UDERA,UDERB>::T_promote DER_promote;

	if(A.getncol() != B.getnrow())
	{
		cout<<"Can not multiply, dimensions does not match"<<endl;
		MPI::COMM_WORLD.Abort(DIMMISMATCH);
		return SpParMat< IU,N_promote,DER_promote >();
	}
	int stages, Aoffset, Boffset; 	// stages = inner dimension of matrix blocks
	shared_ptr<CommGrid> GridC = ProductGrid((A.commGrid).get(), (B.commGrid).get(), stages, Aoffset, Boffset);		

	IU C_m = A.spSeq->getnrow();
	IU C_n = B.spSeq->getncol();

	UDERA A1seq, A2seq;
	(A.spSeq)->Split( A1seq, A2seq); 
	
	// ABAB: It should be able to perform split/merge with the transpose option [single function call]
	const_cast< UDERB* >(B.spSeq)->Transpose();
	
	UDERB B1seq, B2seq;
	(B.spSeq)->Split( B1seq, B2seq);
	
	// Create row and column windows (collective operation, i.e. everybody exposes its window to others)
	vector<MPI::Win> rowwins1, rowwins2, colwins1, colwins2;
	SpParHelper::SetWindows((A.commGrid)->GetRowWorld(), A1seq, rowwins1);
	SpParHelper::SetWindows((A.commGrid)->GetRowWorld(), A2seq, rowwins2);
	SpParHelper::SetWindows((B.commGrid)->GetColWorld(), B1seq, colwins1);
	SpParHelper::SetWindows((B.commGrid)->GetColWorld(), B2seq, colwins2);

	IU ** ARecvSizes1 = SpHelper::allocate2D<IU>(UDERA::esscount, stages);
	IU ** ARecvSizes2 = SpHelper::allocate2D<IU>(UDERA::esscount, stages);
	IU ** BRecvSizes1 = SpHelper::allocate2D<IU>(UDERB::esscount, stages);
	IU ** BRecvSizes2 = SpHelper::allocate2D<IU>(UDERB::esscount, stages);
		
	SpParHelper::GetSetSizes( A1seq, ARecvSizes1, (A.commGrid)->GetRowWorld());
	SpParHelper::GetSetSizes( A2seq, ARecvSizes2, (A.commGrid)->GetRowWorld());
	SpParHelper::GetSetSizes( B1seq, BRecvSizes1, (B.commGrid)->GetColWorld());
	SpParHelper::GetSetSizes( B2seq, BRecvSizes2, (B.commGrid)->GetColWorld());

	// Remotely fetched matrices are stored as pointers
	UDERA * ARecv1, * ARecv2; 
	UDERB * BRecv1, * BRecv2;
	vector< SpTuples<IU,N_promote> *> tomerge;	// sorted triples to be merged

	MPI::Group row_group = (A.commGrid)->GetRowWorld().Get_group();
	MPI::Group col_group = (B.commGrid)->GetColWorld().Get_group();

	int Aself = (A.commGrid)->GetRankInProcRow();
	int Bself = (B.commGrid)->GetRankInProcCol();	
	
	int Aowner = (0+Aoffset) % stages;		
	int Bowner = (0+Boffset) % stages;
	
	SpParHelper::LockNFetch(ARecv1, Aowner, rowwins1, row_group, ARecvSizes1);
	SpParHelper::LockNFetch(ARecv2, Aowner, rowwins2, row_group, ARecvSizes2);	// Start prefetching next half 
	SpParHelper::LockNFetch(BRecv1, Bowner, colwins1, col_group, BRecvSizes1);
	SpParHelper::LockNFetch(BRecv2, Bowner, colwins2, col_group, BRecvSizes2);	// Start prefetching next half 
		
	// Finish the first halfs
	SpParHelper::UnlockWindows(Aowner, rowwins1);
	SpParHelper::UnlockWindows(Bowner, colwins1);

	for(int i = 1; i < stages; ++i) 
	{
		SpTuples<IU,N_promote> * C_cont = MultiplyReturnTuples<SR>(*ARecv1, *BRecv1, false, true);

		if(!C_cont->isZero()) 
			tomerge.push_back(C_cont);

		bool remoteA = false;
		bool remoteB = false;

		delete ARecv1;		// free the memory of the previous first half
		delete BRecv1;

		SpParHelper::UnlockWindows(Aowner, rowwins2);	// Finish the second half
		SpParHelper::UnlockWindows(Bowner, colwins2);	

		Aowner = (i+Aoffset) % stages;		
		Bowner = (i+Boffset) % stages;

		// start fetching the current first half 
		SpParHelper::LockNFetch(ARecv1, Aowner, rowwins1, row_group, ARecvSizes1);
		SpParHelper::LockNFetch(BRecv1, Bowner, colwins1, col_group, BRecvSizes1);
	
		// while multiplying the already completed previous second halfs
		C_cont = MultiplyReturnTuples<SR>(*ARecv2, *BRecv2, false, true);	
		if(!C_cont->isZero()) 
			tomerge.push_back(C_cont);

		delete ARecv2;		// free memory of the previous second half
		delete BRecv2;

		// wait for the current first half to complte
		SpParHelper::UnlockWindows(Aowner, rowwins1);
		SpParHelper::UnlockWindows(Bowner, colwins1);
		
		// start prefetching the current second half 
		SpParHelper::LockNFetch(ARecv2, Aowner, rowwins2, row_group, ARecvSizes2);
		SpParHelper::LockNFetch(BRecv2, Bowner, colwins2, col_group, BRecvSizes2);
	}

	SpTuples<IU,N_promote> * C_cont = MultiplyReturnTuples<SR>(*ARecv1, *BRecv1, false, true);
	if(!C_cont->isZero()) 
		tomerge.push_back(C_cont);

	delete ARecv1;		// free the memory of the previous first half
	delete BRecv1;
	
	SpParHelper::UnlockWindows(Aowner, rowwins2);
	SpParHelper::UnlockWindows(Bowner, colwins2);

	C_cont = MultiplyReturnTuples<SR>(*ARecv2, *BRecv2, false, true);	
	if(!C_cont->isZero()) 
		tomerge.push_back(C_cont);		
		
	delete ARecv2;
	delete BRecv2;

	SpHelper::deallocate2D(ARecvSizes1, UDERA::esscount);
	SpHelper::deallocate2D(ARecvSizes2, UDERA::esscount);
	SpHelper::deallocate2D(BRecvSizes1, UDERB::esscount);
	SpHelper::deallocate2D(BRecvSizes2, UDERB::esscount);
			
	DER_promote * C = new DER_promote(MergeAll<SR>(tomerge, C_m, C_n), false, NULL);	// First get the result in SpTuples, then convert to UDER
	for(int i=0; i<tomerge.size(); ++i)
	{
		delete tomerge[i];
	}
	
	SpParHelper::FreeWindows(rowwins1);
	SpParHelper::FreeWindows(rowwins2);
	SpParHelper::FreeWindows(colwins1);
	SpParHelper::FreeWindows(colwins2);	

	(A.spSeq)->Merge(A1seq, A2seq);
	(B.spSeq)->Merge(B1seq, B2seq);	

	row_group.Free();
	col_group.Free();	
	const_cast< UDERB* >(B.spSeq)->Transpose();	// transpose back to original
	return SpParMat<IU,N_promote,DER_promote> (C, GridC);		// return the result object
}

/**
 * Parallel A = B*C routine that uses one-sided MPI-2 features
 * Syncronization is through MPI_Win_Fence
 * Buggy as of September, 2009
 **/ 
template <typename SR, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB> 
SpParMat<IU,typename promote_trait<NU1,NU2>::T_promote,typename promote_trait<UDERA,UDERB>::T_promote> Mult_AnXBn_Fence
		(const SpParMat<IU,NU1,UDERA> & A, const SpParMat<IU,NU2,UDERB> & B )
{
	typedef typename promote_trait<NU1,NU2>::T_promote N_promote;
	typedef typename promote_trait<UDERA,UDERB>::T_promote DER_promote;
	
	if(A.getncol() != B.getnrow())
	{
		cout<<"Can not multiply, dimensions does not match"<<endl;
		MPI::COMM_WORLD.Abort(DIMMISMATCH);
		return SpParMat< IU,N_promote,DER_promote >();
	}

	int stages, Aoffset, Boffset; 	// stages = inner dimension of matrix blocks
	shared_ptr<CommGrid> GridC = ProductGrid((A.commGrid).get(), (B.commGrid).get(), stages, Aoffset, Boffset);		
			
	ofstream oput;
	GridC->OpenDebugFile("deb", oput);
	const_cast< UDERB* >(B.spSeq)->Transpose();
	
	// set row & col window handles
	vector<MPI::Win> rowwindows, colwindows;
	vector<MPI::Win> rowwinnext, colwinnext;
	SpParHelper::SetWindows((A.commGrid)->GetRowWorld(), *(A.spSeq), rowwindows);
	SpParHelper::SetWindows((B.commGrid)->GetColWorld(), *(B.spSeq), colwindows);
	SpParHelper::SetWindows((A.commGrid)->GetRowWorld(), *(A.spSeq), rowwinnext);
	SpParHelper::SetWindows((B.commGrid)->GetColWorld(), *(B.spSeq), colwinnext);
	
	IU ** ARecvSizes = SpHelper::allocate2D<IU>(UDERA::esscount, stages);
	IU ** BRecvSizes = SpHelper::allocate2D<IU>(UDERB::esscount, stages);
	
	SpParHelper::GetSetSizes( *(A.spSeq), ARecvSizes, (A.commGrid)->GetRowWorld());
	SpParHelper::GetSetSizes( *(B.spSeq), BRecvSizes, (B.commGrid)->GetColWorld());
	
	UDERA * ARecv, * ARecvNext; 
	UDERB * BRecv, * BRecvNext;
	vector< SpTuples<IU,N_promote>  *> tomerge;

	// Prefetch first
	for(int j=0; j< rowwindows.size(); ++j)
		MPI_Win_fence(MPI_MODE_NOPRECEDE, rowwindows[j]);
	for(int j=0; j< colwindows.size(); ++j)
		MPI_Win_fence(MPI_MODE_NOPRECEDE, colwindows[j]);

	for(int j=0; j< rowwinnext.size(); ++j)
		MPI_Win_fence(MPI_MODE_NOPRECEDE, rowwinnext[j]);
	for(int j=0; j< colwinnext.size(); ++j)
		MPI_Win_fence(MPI_MODE_NOPRECEDE, colwinnext[j]);


	int Aownind = (0+Aoffset) % stages;		
	int Bownind = (0+Boffset) % stages;
	if(Aownind == (A.commGrid)->GetRankInProcRow())
	{	
		ARecv = A.spSeq;	// shallow-copy 
	}
	else
	{
		vector<IU> ess1(UDERA::esscount);		// pack essentials to a vector
		for(int j=0; j< UDERA::esscount; ++j)	
		{
			ess1[j] = ARecvSizes[j][Aownind];	
		}
		ARecv = new UDERA();	// create the object first	

		oput << "For A (out), Fetching " << (void*)rowwindows[0] << endl;
		SpParHelper::FetchMatrix(*ARecv, ess1, rowwindows, Aownind);	// fetch its elements later
	}
	if(Bownind == (B.commGrid)->GetRankInProcCol())
	{
		BRecv = B.spSeq;	// shallow-copy
	}
	else
	{
		vector<IU> ess2(UDERB::esscount);		// pack essentials to a vector
		for(int j=0; j< UDERB::esscount; ++j)	
		{
			ess2[j] = BRecvSizes[j][Bownind];	
		}	
		BRecv = new UDERB();

		oput << "For B (out), Fetching " << (void*)colwindows[0] << endl;
		SpParHelper::FetchMatrix(*BRecv, ess2, colwindows, Bownind);	// No lock version, only get !
	}

	int Aownprev = Aownind;
	int Bownprev = Bownind;
	
	for(int i = 1; i < stages; ++i) 
	{
		Aownind = (i+Aoffset) % stages;		
		Bownind = (i+Boffset) % stages;

		if(i % 2 == 1)	// Fetch RecvNext via winnext, fence on Recv via windows
		{	
			if(Aownind == (A.commGrid)->GetRankInProcRow())
			{	
				ARecvNext = A.spSeq;	// shallow-copy 
			}
			else
			{
				vector<IU> ess1(UDERA::esscount);		// pack essentials to a vector
				for(int j=0; j< UDERA::esscount; ++j)	
				{
					ess1[j] = ARecvSizes[j][Aownind];	
				}
				ARecvNext = new UDERA();	// create the object first	

				oput << "For A, Fetching " << (void*) rowwinnext[0] << endl;
				SpParHelper::FetchMatrix(*ARecvNext, ess1, rowwinnext, Aownind);
			}
		
			if(Bownind == (B.commGrid)->GetRankInProcCol())
			{
				BRecvNext = B.spSeq;	// shallow-copy
			}
			else
			{
				vector<IU> ess2(UDERB::esscount);		// pack essentials to a vector
				for(int j=0; j< UDERB::esscount; ++j)	
				{
					ess2[j] = BRecvSizes[j][Bownind];	
				}		
				BRecvNext = new UDERB();

				oput << "For B, Fetching " << (void*)colwinnext[0] << endl;
				SpParHelper::FetchMatrix(*BRecvNext, ess2, colwinnext, Bownind);	// No lock version, only get !
			}
		
			oput << "Fencing " << (void*) rowwindows[0] << endl;
			oput << "Fencing " << (void*) colwindows[0] << endl;
		
			for(int j=0; j< rowwindows.size(); ++j)
				MPI_Win_fence(MPI_MODE_NOSTORE, rowwindows[j]);		// Synch using "other" windows
			for(int j=0; j< colwindows.size(); ++j)
				MPI_Win_fence(MPI_MODE_NOSTORE, colwindows[j]);
	
			SpTuples<IU,N_promote> * C_cont = MultiplyReturnTuples<SR>(*ARecv, *BRecv, false, true);
			if(!C_cont->isZero()) 
				tomerge.push_back(C_cont);

			if(Aownprev != (A.commGrid)->GetRankInProcRow()) delete ARecv;
			if(Bownprev != (B.commGrid)->GetRankInProcCol()) delete BRecv;

			Aownprev = Aownind;
			Bownprev = Bownind; 
		}	
		else	// fetch to Recv via windows, fence on RecvNext via winnext
		{	
			
			if(Aownind == (A.commGrid)->GetRankInProcRow())
			{	
				ARecv = A.spSeq;	// shallow-copy 
			}
			else
			{
				vector<IU> ess1(UDERA::esscount);		// pack essentials to a vector
				for(int j=0; j< UDERA::esscount; ++j)	
				{
					ess1[j] = ARecvSizes[j][Aownind];	
				}
				ARecv = new UDERA();	// create the object first	

				oput << "For A, Fetching " << (void*) rowwindows[0] << endl;
				SpParHelper::FetchMatrix(*ARecv, ess1, rowwindows, Aownind);
			}
		
			if(Bownind == (B.commGrid)->GetRankInProcCol())
			{
				BRecv = B.spSeq;	// shallow-copy
			}
			else
			{
				vector<IU> ess2(UDERB::esscount);		// pack essentials to a vector
				for(int j=0; j< UDERB::esscount; ++j)	
				{
					ess2[j] = BRecvSizes[j][Bownind];	
				}		
				BRecv = new UDERB();

				oput << "For B, Fetching " << (void*)colwindows[0] << endl;
				SpParHelper::FetchMatrix(*BRecv, ess2, colwindows, Bownind);	// No lock version, only get !
			}
		
			oput << "Fencing " << (void*) rowwinnext[0] << endl;
			oput << "Fencing " << (void*) rowwinnext[0] << endl;
		
			for(int j=0; j< rowwinnext.size(); ++j)
				MPI_Win_fence(MPI_MODE_NOSTORE, rowwinnext[j]);		// Synch using "other" windows
			for(int j=0; j< colwinnext.size(); ++j)
				MPI_Win_fence(MPI_MODE_NOSTORE, colwinnext[j]);

			SpTuples<IU,N_promote> * C_cont = MultiplyReturnTuples<SR>(*ARecvNext, *BRecvNext, false, true);
			if(!C_cont->isZero()) 
				tomerge.push_back(C_cont);


			if(Aownprev != (A.commGrid)->GetRankInProcRow()) delete ARecvNext;
			if(Bownprev != (B.commGrid)->GetRankInProcCol()) delete BRecvNext;

			Aownprev = Aownind;
			Bownprev = Bownind; 
		}

	}

	if(stages % 2 == 1)	// fence on Recv via windows
	{
		oput << "Fencing " << (void*) rowwindows[0] << endl;
		oput << "Fencing " << (void*) colwindows[0] << endl;

		for(int j=0; j< rowwindows.size(); ++j)
			MPI_Win_fence(MPI_MODE_NOSUCCEED, rowwindows[j]);		// Synch using "prev" windows
		for(int j=0; j< colwindows.size(); ++j)
			MPI_Win_fence(MPI_MODE_NOSUCCEED, colwindows[j]);

		SpTuples<IU,N_promote> * C_cont = MultiplyReturnTuples<SR>(*ARecv, *BRecv, false, true);
		if(!C_cont->isZero()) 
			tomerge.push_back(C_cont);

		if(Aownprev != (A.commGrid)->GetRankInProcRow()) delete ARecv;
		if(Bownprev != (B.commGrid)->GetRankInProcRow()) delete BRecv;
	}
	else		// fence on RecvNext via winnext
	{
		oput << "Fencing " << (void*) rowwinnext[0] << endl;
		oput << "Fencing " << (void*) colwinnext[0] << endl;

		for(int j=0; j< rowwinnext.size(); ++j)
			MPI_Win_fence(MPI_MODE_NOSUCCEED, rowwinnext[j]);		// Synch using "prev" windows
		for(int j=0; j< colwinnext.size(); ++j)
			MPI_Win_fence(MPI_MODE_NOSUCCEED, colwinnext[j]);

		SpTuples<IU,N_promote> * C_cont = MultiplyReturnTuples<SR>(*ARecvNext, *BRecvNext, false, true);
		if(!C_cont->isZero()) 
			tomerge.push_back(C_cont);

		if(Aownprev != (A.commGrid)->GetRankInProcRow()) delete ARecvNext;
		if(Bownprev != (B.commGrid)->GetRankInProcRow()) delete BRecvNext;
	}
	for(int i=0; i< rowwindows.size(); ++i)
	{
		rowwindows[i].Free();
		rowwinnext[i].Free();
	}
	for(int i=0; i< colwindows.size(); ++i)
	{
		colwindows[i].Free();
		colwinnext[i].Free();
	}
	GridC->GetWorld().Barrier();

	IU C_m = A.spSeq->getnrow();
	IU C_n = B.spSeq->getncol();
	DER_promote * C = new DER_promote(MergeAll<SR>(tomerge, C_m, C_n), false, NULL);	// First get the result in SpTuples, then convert to UDER
	for(int i=0; i<tomerge.size(); ++i)
	{
		delete tomerge[i];
	}
	SpHelper::deallocate2D(ARecvSizes, UDERA::esscount);
	SpHelper::deallocate2D(BRecvSizes, UDERB::esscount);
	
	const_cast< UDERB* >(B.spSeq)->Transpose();	// transpose back to original	
	return SpParMat<IU,N_promote,DER_promote> (C, GridC);			// return the result object
}


// Randomly permutes an already existing vector
// Preserves the data distribution (doesn't rebalance)
template <typename IU>
void RandPerm(SpParVec<IU,IU> & V)
{
	SpParHelper::Print("COMBBLAS: This version of RandPerm(SpParVec &) is obsolete, please use DenseParVec::RandPerm()\n");
	MPI::Intracomm DiagWorld = V.commGrid->GetDiagWorld();

	if(DiagWorld != MPI::COMM_NULL) // Diagonal processors only
	{
		pair<double,IU> * vecpair = new pair<double,IU>[V.getlocnnz()];

		int nproc = DiagWorld.Get_size();
		int diagrank = DiagWorld.Get_rank();

		long * dist = new long[nproc];
		dist[diagrank] = (long) V.getlocnnz();
		DiagWorld.Allgather(MPI::IN_PLACE, 0, MPIType<long>(), dist, 1, MPIType<long>());

  		MTRand M;	// generate random numbers with Mersenne Twister
		for(int i=0; i<V.getlocnnz(); ++i)
		{
			vecpair[i].first = M.rand();
			vecpair[i].second = V.num[i];
		}

		// less< pair<T1,T2> > works correctly (sorts wrt first elements)	
    		vpsort::parallel_sort (vecpair, vecpair + V.getlocnnz(),  dist, DiagWorld);

		vector< IU > nind(V.getlocnnz());
		vector< IU > nnum(V.getlocnnz());
		for(int i=0; i<V.getlocnnz(); ++i)
		{
			nind[i] = i;
			nnum[i] = vecpair[i].second;
		}
		delete [] vecpair;
		delete [] dist;

		V.ind.swap(nind);
		V.num.swap(nnum);
	}
}
		
template <typename SR, typename IU, typename NUM, typename NUV, typename UDER> 
DenseParVec<IU,typename promote_trait<NUM,NUV>::T_promote>  SpMV 
	(const SpParMat<IU,NUM,UDER> & A, const DenseParVec<IU,NUV> & x )
{
	typedef typename promote_trait<NUM,NUV>::T_promote T_promote;

	IU ncolA = A.getncol();
	if(ncolA != x.getTotalLength())
	{
		ostringstream outs;
		outs << "Can not multiply, dimensions does not match"<< endl;
		outs << ncolA << " != " << x.getTotalLength() << endl;
		SpParHelper::Print(outs.str());
		MPI::COMM_WORLD.Abort(DIMMISMATCH);
	}
	if(!(*A.commGrid == *x.commGrid)) 		
	{
		cout << "Grids are not comparable for SpMV" << endl; 
		MPI::COMM_WORLD.Abort(GRIDMISMATCH);
	}
	MPI::Intracomm DiagWorld = x.commGrid->GetDiagWorld();
	MPI::Intracomm ColWorld = x.commGrid->GetColWorld();
	MPI::Intracomm RowWorld = x.commGrid->GetRowWorld();
	int diaginrow = x.commGrid->GetDiagOfProcRow();
        int diagincol = x.commGrid->GetDiagOfProcCol();

	T_promote id = (T_promote) 0;	// do we need a better identity?
	DenseParVec<IU, T_promote> y ( x.commGrid, id);	
	IU ysize = A.getlocalrows();
	if(x.diagonal)
	{
		IU size = x.arr.size();
		ColWorld.Bcast(&size, 1, MPIType<IU>(), diagincol);
		ColWorld.Bcast(const_cast<NUV*>(&x.arr[0]), size, MPIType<NUV>(), diagincol); 

		T_promote * localy = new T_promote[ysize];
		fill_n(localy, ysize, id);		
		dcsc_gespmv<SR>(*(A.spSeq), &x.arr[0], localy);	

		// IntraComm::Reduce(sendbuf, recvbuf, count, type, op, root)
                RowWorld.Reduce(MPI::IN_PLACE, localy, ysize, MPIType<T_promote>(), SR::mpi_op(), diaginrow);
		y.arr.resize(ysize);
		copy(localy, localy+ysize, y.arr.begin());
		delete [] localy;
	}
	else
	{
		IU size;
		ColWorld.Bcast(&size, 1, MPIType<IU>(), diagincol);

		NUV * localx = new NUV[size];
		ColWorld.Bcast(localx, size, MPIType<NUV>(), diagincol); 
	
		T_promote * localy = new T_promote[ysize];		
		fill_n(localy, ysize, id);		

		dcsc_gespmv<SR>(*(A.spSeq), localx, localy);
		delete [] localx;

                RowWorld.Reduce(localy, NULL, ysize, MPIType<T_promote>(), SR::mpi_op(), diaginrow);	
		delete [] localy;
	}
	return y;
}
	

template <typename SR, typename IU, typename NUM, typename NUV, typename UDER> 
SpParVec<IU,typename promote_trait<NUM,NUV>::T_promote>  SpMV 
	(const SpParMat<IU,NUM,UDER> & A, const SpParVec<IU,NUV> & x )
{
	typedef typename promote_trait<NUM,NUV>::T_promote T_promote;

	IU ncolA = A.getncol();
	if(ncolA != x.getTotalLength())
	{
		ostringstream outs;
		outs << "Can not multiply, dimensions does not match"<< endl;
		outs << ncolA << " != " << x.getTotalLength() << endl;
		SpParHelper::Print(outs.str());
		MPI::COMM_WORLD.Abort(DIMMISMATCH);
	}
	if(!(*A.commGrid == *x.commGrid)) 		
	{
		cout << "Grids are not comparable for SpMV" << endl; 
		MPI::COMM_WORLD.Abort(GRIDMISMATCH);
	}

	MPI::Intracomm DiagWorld = x.commGrid->GetDiagWorld();
	MPI::Intracomm ColWorld = x.commGrid->GetColWorld();
	MPI::Intracomm RowWorld = x.commGrid->GetRowWorld();
	int diaginrow = x.commGrid->GetDiagOfProcRow();
        int diagincol = x.commGrid->GetDiagOfProcCol();

	SpParVec<IU, T_promote> y ( x.commGrid);	// identity doesn't matter for sparse vectors
	IU ysize = A.getlocalrows();
	if(x.diagonal)
	{
		IU nnzx = x.getlocnnz();
		ColWorld.Bcast(&nnzx, 1, MPIType<IU>(), diagincol);
		ColWorld.Bcast(const_cast<IU*>(&x.ind[0]), nnzx, MPIType<IU>(), diagincol); 
		ColWorld.Bcast(const_cast<NUV*>(&x.num[0]), nnzx, MPIType<NUV>(), diagincol); 

		// define a SPA-like data structure
		T_promote * localy = new T_promote[ysize];
		bool * isthere = new bool[ysize];
		vector<IU> nzinds;	// nonzero indices		
		fill_n(isthere, ysize, false);

		// serial SpMV with sparse vector
		vector< IU > indy;
		vector< T_promote >  numy;
		dcsc_gespmv<SR>(*(A.spSeq), &x.ind[0], &x.num[0], nnzx, indy, numy);	

		int proccols = x.commGrid->GetGridCols();
		int * gsizes = new int[proccols];	// # of processor columns = number of processors in the RowWorld
		int mysize = indy.size();
		RowWorld.Gather(&mysize, 1, MPI::INT, gsizes, 1, MPI::INT, diaginrow);
		int maxnnz = std::accumulate(gsizes, gsizes+proccols, 0);
		int * dpls = new int[proccols]();	// displacements (zero initialized pid) 
		std::partial_sum(gsizes, gsizes+proccols-1, dpls+1);
		
		IU * indbuf = new IU[maxnnz];	
		T_promote * numbuf = new T_promote[maxnnz];

		// IntraComm::GatherV(sendbuf, int sentcnt, sendtype, recvbuf, int * recvcnts, int * displs, recvtype, root)
                RowWorld.Gatherv(&(indy[0]), mysize, MPIType<IU>(), indbuf, gsizes, dpls, MPIType<IU>(), diaginrow);
                RowWorld.Gatherv(&(numy[0]), mysize, MPIType<T_promote>(), numbuf, gsizes, dpls, MPIType<T_promote>(), diaginrow);

		for(int i=0; i< maxnnz; ++i)
		{
			if(!isthere[indbuf[i]])
			{
				localy[indbuf[i]] = numbuf[i];	// initial assignment
				nzinds.push_back(indbuf[i]);
				isthere[indbuf[i]] = true;
			} 
			else
			{
				localy[indbuf[i]] = SR::add(localy[indbuf[i]], numbuf[i]);	
			}
		}
		DeleteAll(gsizes, dpls, indbuf, numbuf,isthere);
		sort(nzinds.begin(), nzinds.end());
		
		int nnzy = nzinds.size();
		y.ind.resize(nnzy);
		y.num.resize(nnzy);
		for(int i=0; i< nnzy; ++i)
		{
			y.ind[i] = nzinds[i];
			y.num[i] = localy[nzinds[i]]; 	
		}
		y.length = ysize;
		delete [] localy;
	}
	else
	{
		IU nnzx;
		ColWorld.Bcast(&nnzx, 1, MPIType<IU>(), diagincol);

		IU * xinds = new IU[nnzx];
		NUV * xnums = new NUV[nnzx];
		ColWorld.Bcast(xinds, nnzx, MPIType<IU>(), diagincol); 
		ColWorld.Bcast(xnums, nnzx, MPIType<NUV>(), diagincol); 

		// serial SpMV with sparse vector
		vector< IU > indy;
		vector< T_promote >  numy;
		dcsc_gespmv<SR>(*(A.spSeq), xinds, xnums, nnzx, indy, numy);	

		int mysize = indy.size();
		RowWorld.Gather(&mysize, 1, MPI::INT, NULL, 1, MPI::INT, diaginrow);

		// IntraComm::GatherV(sendbuf, int sentcnt, sendtype, recvbuf, int * recvcnts, int * displs, recvtype, root)
                RowWorld.Gatherv(&(indy[0]), mysize, MPIType<IU>(), NULL, NULL, NULL, MPIType<IU>(), diaginrow);
                RowWorld.Gatherv(&(numy[0]), mysize, MPIType<T_promote>(), NULL, NULL, NULL, MPIType<T_promote>(), diaginrow);

		delete [] xinds;
		delete [] xnums;
	}
	return y;
}

template <typename SR, typename IU, typename NUM, typename UDER> 
FullyDistSpVec<IU,typename promote_trait<NUM,IU>::T_promote>  SpMV 
	(const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,IU> & x, bool indexisvalue, OptBuf<IU, typename promote_trait<NUM,IU>::T_promote > & optbuf);

template <typename SR, typename IU, typename NUM, typename UDER> 
FullyDistSpVec<IU,typename promote_trait<NUM,IU>::T_promote>  SpMV 
	(const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,IU> & x, bool indexisvalue)
{
	typedef typename promote_trait<NUM,IU>::T_promote T_promote;
	OptBuf<IU, T_promote > optbuf = OptBuf<IU, T_promote >();
	return SpMV<SR>(A, x, indexisvalue, optbuf);
}

//! The last parameter is a hint to the function 
//! If indexisvalues = true, then we do not need to transfer values for x
//! This happens for BFS iterations with boolean matrices and integer rhs vectors
template <typename SR, typename IU, typename NUM, typename UDER> 
FullyDistSpVec<IU,typename promote_trait<NUM,IU>::T_promote>  SpMV 
	(const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,IU> & x, bool indexisvalue, OptBuf<IU, typename promote_trait<NUM,IU>::T_promote > & optbuf)
{
	typedef typename promote_trait<NUM,IU>::T_promote T_promote;

	IU ncolA = A.getncol();
	if(ncolA != x.TotalLength())
	{
		ostringstream outs;
		outs << "Can not multiply, dimensions does not match"<< endl;
		outs << ncolA << " != " << x.TotalLength() << endl;
		SpParHelper::Print(outs.str());
		MPI::COMM_WORLD.Abort(DIMMISMATCH);
	}
	if(!(*A.commGrid == *x.commGrid)) 		
	{
		cout << "Grids are not comparable for SpMV" << endl; 
		MPI::COMM_WORLD.Abort(GRIDMISMATCH);
	}

	MPI::Intracomm World = x.commGrid->GetWorld();
	MPI::Intracomm ColWorld = x.commGrid->GetColWorld();
	MPI::Intracomm RowWorld = x.commGrid->GetRowWorld();

	IU xlocnz = x.getlocnnz();
	IU roffst = x.RowLenUntil();
	IU luntil = x.LengthUntil();
	IU trxlocnz, roffset, lenuntil;

	int diagneigh = x.commGrid->GetComplementRank();
	World.Sendrecv(&xlocnz, 1, MPIType<IU>(), diagneigh, TRNNZ, &trxlocnz, 1, MPIType<IU>(), diagneigh, TRNNZ);
	World.Sendrecv(&roffst, 1, MPIType<IU>(), diagneigh, TROST, &roffset, 1, MPIType<IU>(), diagneigh, TROST);
	World.Sendrecv(&luntil, 1, MPIType<IU>(), diagneigh, TRLUT, &lenuntil, 1, MPIType<IU>(), diagneigh, TRLUT);
	
	// ABAB: Important observation is that local indices (given by x.ind) is 32-bit addressible
	// Copy them to 32 bit integers and transfer that to save 50% of off-node bandwidth
	IU * trxinds = new IU[trxlocnz];
	IU * trxnums;
	World.Sendrecv(const_cast<IU*>(&x.ind[0]), xlocnz, MPIType<IU>(), diagneigh, TRI, trxinds, trxlocnz, MPIType<IU>(), diagneigh, TRI);
	if(!indexisvalue)
	{
		trxnums = new IU[trxlocnz];
		World.Sendrecv(const_cast<IU*>(&x.num[0]), xlocnz, MPIType<IU>(), diagneigh, TRX, trxnums, trxlocnz, MPIType<IU>(), diagneigh, TRX);
	}
	transform(trxinds, trxinds+trxlocnz, trxinds, bind2nd(plus<IU>(), roffset)); // fullydist indexing (p pieces) -> matrix indexing (sqrt(p) pieces)

	int colneighs = ColWorld.Get_size();
	int colrank = ColWorld.Get_rank();
	int * colnz = new int[colneighs];
	colnz[colrank] = static_cast<int>(trxlocnz);
	ColWorld.Allgather(MPI::IN_PLACE, 1, MPI::INT, colnz, 1, MPI::INT);
	int * dpls = new int[colneighs]();	// displacements (zero initialized pid) 
	std::partial_sum(colnz, colnz+colneighs-1, dpls+1);
	int accnz = std::accumulate(colnz, colnz+colneighs, 0);
	IU * indacc = new IU[accnz];
	IU * numacc = new IU[accnz];

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
	ColWorld.Allgatherv(trxinds, trxlocnz, MPIType<IU>(), indacc, colnz, dpls, MPIType<IU>());
	#ifdef TIMING
	World.Barrier();
	double t1=MPI::Wtime();
	cblas_allgathertime += (t1-t0);
	#endif

	delete [] trxinds;
	if(indexisvalue)
	{
		IU lenuntilcol;
		if(colrank == 0)
		{
			lenuntilcol = lenuntil;
		}
		ColWorld.Bcast(&lenuntilcol, 1, MPIType<IU>(), 0);
		transform(indacc, indacc+accnz, numacc, bind2nd(plus<IU>(), lenuntilcol));	// fill numerical values from indices
	}
	else
	{
		ColWorld.Allgatherv(trxnums, trxlocnz, MPIType<IU>(), numacc, colnz, dpls, MPIType<IU>());
		delete [] trxnums;
	}	

	DeleteAll(colnz,dpls);
	int rowneighs = RowWorld.Get_size();
	int * sendcnt = new int[rowneighs]();	
	FullyDistSpVec<IU, T_promote> y ( x.commGrid, A.getnrow());	// identity doesn't matter for sparse vectors

	IU * sendindbuf;
	T_promote * sendnumbuf;
	int * sdispls;
	if(optbuf.totmax > 0)	// graph500 optimization enabled
	{ 
		if(A.spSeq->getnsplit() > 0)
		{
			SpParHelper::Print("Preallocated buffers can not be used with multithreaded code yet\n");
			// sendindbuf/sendnumbuf/sdispls are all allocated and filled by dcsc_gespmv_threaded
			int totalsent = dcsc_gespmv_threaded<SR> (*(A.spSeq), indacc, numacc, static_cast<IU>(accnz), sendindbuf, sendnumbuf, sdispls, rowneighs);	
			for(int i=0; i<rowneighs-1; ++i)
				sendcnt[i] = sdispls[i+1] + sdispls[i];	
			sendcnt[rowneighs-1] = totalsent - sdispls[rowneighs-1];
		}
		else
		{
			dcsc_gespmv<SR> (*(A.spSeq), indacc, numacc, static_cast<IU>(accnz), optbuf.inds, optbuf.nums, sendcnt, optbuf.dspls, rowneighs);
		}
		DeleteAll(indacc,numacc);
	}
	else
	{
		if(A.spSeq->getnsplit() > 0)
		{
			// sendindbuf/sendnumbuf/sdispls are all allocated and filled by dcsc_gespmv_threaded
			int totalsent = dcsc_gespmv_threaded<SR> (*(A.spSeq), indacc, numacc, static_cast<IU>(accnz), sendindbuf, sendnumbuf, sdispls, rowneighs);	
			DeleteAll(indacc, numacc);
			for(int i=0; i<rowneighs-1; ++i)
				sendcnt[i] = sdispls[i+1] - sdispls[i];
			sendcnt[rowneighs-1] = totalsent - sdispls[rowneighs-1];
		}
		else
		{
			// serial SpMV with sparse vector
			vector< IU > indy;
			vector< T_promote >  numy;
			dcsc_gespmv<SR>(*(A.spSeq), indacc, numacc, static_cast<IU>(accnz), indy, numy);	// actual multiplication
			DeleteAll(indacc, numacc);

			IU bufsize = indy.size();	// as compact as possible
			sendindbuf = new IU[bufsize];	
			sendnumbuf = new T_promote[bufsize];
			IU perproc = A.getlocalrows() / static_cast<IU>(rowneighs);	

			int k = 0;	// index to buffer
			for(int i=0; i<rowneighs; ++i)		
			{
				IU end_this = (i==rowneighs-1) ? A.getlocalrows(): (i+1)*perproc;
				while(k < bufsize && indy[k] < end_this) 
				{
					sendindbuf[k] = indy[k] - i*perproc;
					sendnumbuf[k] = numy[k];
					++sendcnt[i];
					++k; 
				}
			}
			sdispls = new int[rowneighs]();	
			partial_sum(sendcnt, sendcnt+rowneighs-1, sdispls+1); 
		}
	}

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
	IU * recvindbuf = new IU[totrecv];
	T_promote * recvnumbuf = new T_promote[totrecv];

	#ifdef TIMING
	World.Barrier();
	double t2=MPI::Wtime();
	#endif
	if(optbuf.totmax > 0 && A.spSeq->getnsplit() == 0)	// graph500 optimization enabled
	{
		RowWorld.Alltoallv(optbuf.inds, sendcnt, optbuf.dspls, MPIType<IU>(), recvindbuf, recvcnt, rdispls, MPIType<IU>());  
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
		RowWorld.Alltoallv(sendindbuf, sendcnt, sdispls, MPIType<IU>(), recvindbuf, recvcnt, rdispls, MPIType<IU>());  
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
		IU topush = recvindbuf[i];
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
	IU hsize = 0;		
	IU inf = numeric_limits<IU>::min();
	IU sup = numeric_limits<IU>::max(); 
        KNHeap< IU, IU > sHeap(sup, inf); 
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
	IU key, locv;
	if(hsize > 0)
	{
		sHeap.deleteMin(&key, &locv);
		y.ind.push_back(key);
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
		if(y.ind.back() == key)	// y.ind is surely not empty
		{
			y.num.back() = SR::add(y.num.back(), recvnumbuf[deref]);
			// ABAB: Benchmark actually allows us to be non-deterministic in terms of parent selection
			// We can just skip this addition operator (if it's a max/min select)
		} 
		else
		{
			y.ind.push_back(key);
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

	IU ncolA = A.getncol();
	if(ncolA != x.TotalLength())
	{
		ostringstream outs;
		outs << "Can not multiply, dimensions does not match"<< endl;
		outs << ncolA << " != " << x.TotalLength() << endl;
		SpParHelper::Print(outs.str());
		MPI::COMM_WORLD.Abort(DIMMISMATCH);
	}
	if(!(*A.commGrid == *x.commGrid)) 		
	{
		cout << "Grids are not comparable for SpMV" << endl; 
		MPI::COMM_WORLD.Abort(GRIDMISMATCH);
	}
	MPI::Intracomm World = x.commGrid->GetWorld();
	MPI::Intracomm ColWorld = x.commGrid->GetColWorld();
	MPI::Intracomm RowWorld = x.commGrid->GetRowWorld();

	int xsize = (int) x.LocArrSize();
	int trxsize = 0;

	int diagneigh = x.commGrid->GetComplementRank();
	World.Sendrecv(&xsize, 1, MPI::INT, diagneigh, TRX, &trxsize, 1, MPI::INT, diagneigh, TRX);
	
	NUV * trxnums = new NUV[trxsize];
	World.Sendrecv(const_cast<NUV*>(&x.arr[0]), xsize, MPIType<NUV>(), diagneigh, TRX, trxnums, trxsize, MPIType<NUV>(), diagneigh, TRX);

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
	FullyDistVec<IU, T_promote> y ( x.commGrid, A.getnrow(), id, id);
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
                RowWorld.Reduce(localy+begptr, &(y.arr[0]), endptr-begptr, MPIType<T_promote>(), SR::mpi_op(), i);
	}
	delete [] localy;
	return y;
}


	
template <typename SR, typename IU, typename NUM, typename NUV, typename UDER> 
FullyDistSpVec<IU,typename promote_trait<NUM,NUV>::T_promote>  SpMV 
	(const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,NUV> & x)
{
	typedef typename promote_trait<NUM,NUV>::T_promote T_promote;
	IU ncolA = A.getncol();
	if(ncolA != x.TotalLength())
	{
		ostringstream outs;
		outs << "Can not multiply, dimensions does not match"<< endl;
		outs << ncolA << " != " << x.TotalLength() << endl;
		SpParHelper::Print(outs.str());
		MPI::COMM_WORLD.Abort(DIMMISMATCH);
	}
	if(!(*A.commGrid == *x.commGrid)) 		
	{
		cout << "Grids are not comparable for SpMV" << endl; 
		MPI::COMM_WORLD.Abort(GRIDMISMATCH);
	}

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
	World.Sendrecv(const_cast<IU*>(&x.ind[0]), xlocnz, MPIType<IU>(), diagneigh, TRX, trxinds, trxlocnz, MPIType<IU>(), diagneigh, TRX);
	World.Sendrecv(const_cast<NUV*>(&x.num[0]), xlocnz, MPIType<NUV>(), diagneigh, TRX, trxnums, trxlocnz, MPIType<NUV>(), diagneigh, TRX);
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

/////////////////////
// Apply
// based on SpMV
template <typename _BinaryOperation, typename IU, typename NUM, typename NUV, typename UDER> 
void ColWiseApply (const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,NUV> & x, _BinaryOperation __binary_op)
{
	if(!(*A.commGrid == *x.commGrid)) 		
	{
		cout << "Grids are not comparable for ColWiseApply" << endl; 
		MPI::COMM_WORLD.Abort(GRIDMISMATCH);
	}

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
	World.Sendrecv(const_cast<IU*>(&x.ind[0]), xlocnz, MPIType<IU>(), diagneigh, TRX, trxinds, trxlocnz, MPIType<IU>(), diagneigh, TRX);
	World.Sendrecv(const_cast<NUV*>(&x.num[0]), xlocnz, MPIType<NUV>(), diagneigh, TRX, trxnums, trxlocnz, MPIType<NUV>(), diagneigh, TRX);
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

	//dcsc_gespmv<SR>(*(A.spSeq), indacc, numacc, static_cast<IU>(accnz), indy, numy);	// actual multiplication
	dcsc_colwise_apply(*(A.spSeq), indacc, numacc, static_cast<IU>(accnz), __binary_op);	// actual operation

	DeleteAll(indacc, numacc);
	DeleteAll(colnz, dpls);
}

/////////////////////
	

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

template <typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB, typename _BinaryOperation> 
SpParMat<IU,typename promote_trait<NU1,NU2>::T_promote,typename promote_trait<UDERA,UDERB>::T_promote> EWiseApply 
	(const SpParMat<IU,NU1,UDERA> & A, const SpParMat<IU,NU2,UDERB> & B, _BinaryOperation __binary_op, bool notB, const NU2& defaultBVal)
{
	typedef typename promote_trait<NU1,NU2>::T_promote N_promote;
	typedef typename promote_trait<UDERA,UDERB>::T_promote DER_promote;

	if(*(A.commGrid) == *(B.commGrid))	
	{
		DER_promote * result = new DER_promote( EWiseApply(*(A.spSeq),*(B.spSeq), __binary_op, notB, defaultBVal) );
		return SpParMat<IU, N_promote, DER_promote> (result, A.commGrid);
	}
	else
	{
		cout << "Grids are not comparable elementwise apply" << endl; 
		MPI::COMM_WORLD.Abort(GRIDMISMATCH);
		return SpParMat< IU,N_promote,DER_promote >();
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



#endif

