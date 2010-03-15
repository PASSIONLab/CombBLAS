#ifndef _PAR_FRIENDS_H_
#define _PAR_FRIENDS_H_

#include "mpi.h"
#include "sys/time.h"
#include <iostream>
#include "SpParMat.h"	
#include "SpParHelper.h"
#include "MPIType.h"

using namespace std;

template <class IT, class NT, class DER>
class SpParMat;

#define INVFREQ 3

/*************************************************************************************************/
/**************************** FRIEND FUNCTIONS FOR PARALLEL CLASSES ******************************/
/*************************************************************************************************/

/**
 * Parallel A = B*C routine that uses one-sided MPI-2 features
 * General active target syncronization via MPI_Win_Post, MPI_Win_Start, MPI_Win_Complete, MPI_Win_Wait
 * Tested on my dual core Macbook with 1,4,9,16,25 MPI processes
 * No memory hog: splits the matrix into two along the column, prefetches the next half matrix while computing on the current one 
 **/  
template <typename SR, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB> 
SpParMat<IU,typename promote_trait<NU1,NU2>::T_promote,typename promote_trait<UDERA,UDERB>::T_promote> Mult_AnXBn_Synch 
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

	int stages, dummy; 	// last two parameters of ProductGrid are ignored for Synch multiplication
	shared_ptr<CommGrid> GridC = ProductGrid((A.commGrid).get(), (B.commGrid).get(), stages, dummy, dummy);		
	IU C_m = A.spSeq->getnrow();
	IU C_n = B.spSeq->getncol();
	
	ofstream oput;
	GridC->OpenDebugFile("deb", oput);
	double t1 = MPI::Wtime();	
	
	const_cast< UDERB* >(B.spSeq)->Transpose();	
	GridC->GetWorld().Barrier();
	
	double t2 = MPI::Wtime();
	oput << "Transpose:\t" << (t2-t1) << endl;

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

		t1 = MPI::Wtime();
		SpParHelper::BCastMatrix(GridC->GetRowWorld(), *ARecv, ess, i);	// then, receive its elements	
		ess.clear();	
		t2 = MPI::Wtime();
		oput << "Broadcast A:\t" << (t2-t1) << endl;
		
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
		t1 = MPI::Wtime();
		SpParHelper::BCastMatrix(GridC->GetColWorld(), *BRecv, ess, i);	// then, receive its elements
		t2 = MPI::Wtime();
		oput << "Broadcast B:\t" << (t2-t1) << endl;			

		SpTuples<IU,N_promote> * C_cont = MultiplyReturnTuples<SR>(*ARecv, *BRecv, false, true);
		if(!C_cont->isZero()) 
			tomerge.push_back(C_cont);

		t1 = MPI::Wtime();
		oput << "Multiply:\t" << (t1-t2) << endl;

		if(i != Aself)	
		{
			delete ARecv;		
		}
		if(i != Bself)	
		{
			delete BRecv;
		}
	}

	SpHelper::deallocate2D(ARecvSizes, UDERA::esscount);
	SpHelper::deallocate2D(BRecvSizes, UDERB::esscount);
			
	DER_promote * C = new DER_promote(*(MergeAll<SR>(tomerge, C_m, C_n)), false, NULL);	// First get the result in SpTuples, then convert to UDER
	for(int i=0; i<tomerge.size(); ++i)
	{
		delete tomerge[i];
	}
	t2 = MPI::Wtime();
	oput << "Deallocate and merge:\t" << (t2-t1) << endl;

	const_cast< UDERB* >(B.spSeq)->Transpose();	// transpose back to original
	
	t1 = MPI::Wtime();
	oput << "Retranspose:\t" << (t1-t2) << endl;
	
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
			
	DER_promote * C = new DER_promote(*(MergeAll<SR>(tomerge, C_m, C_n)), false, NULL);	// First get the result in SpTuples, then convert to UDER
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

	ofstream oput;
	GridC->OpenDebugFile("deb", oput);
	double t1 = MPI::Wtime();	
			
	UDERA A1seq, A2seq;
	(A.spSeq)->Split( A1seq, A2seq); 
	
	// ABAB: It should be able to perform split/merge with the transpose option [single function call]
	const_cast< UDERB* >(B.spSeq)->Transpose();
	
	UDERB B1seq, B2seq;
	(B.spSeq)->Split( B1seq, B2seq);

	double t2 = MPI::Wtime();
	oput << "Transpose and Split:\t" << (t2-t1) << endl;
	
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

	t1 = MPI::Wtime();
	oput << "Window creation and metadata exchange:\t" << (t1-t2) << endl;
	
	// Remotely fetched matrices are stored as pointers
	UDERA * ARecv1, * ARecv2; 
	UDERB * BRecv1, * BRecv2;
	vector< SpTuples<IU,N_promote> *> highmerge;	// higher level of unmerged triples
	vector< SpTuples<IU,N_promote> *> lowmerge;	// lower level of unmerged triples

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

	t2 = MPI::Wtime();
	oput << "First half, fetch only:\t" << (t2-t1) << endl;
	
	for(int i = 1; i < stages; ++i) 
	{
		SpTuples<IU,N_promote> * C_cont = MultiplyReturnTuples<SR>(*ARecv1, *BRecv1, false, true);

		if(!C_cont->isZero()) 
			lowmerge.push_back(C_cont);

		if(i % INVFREQ == 0)
		{
			SpTuples<IU,N_promote> * InterC = MergeAll<SR>(lowmerge, C_m, C_n);	
			for(int i=0; i< lowmerge.size(); ++i)
			{
				delete lowmerge[i];
			}
			lowmerge.clear();
			highmerge.push_back(InterC);
		}


		t1 = MPI::Wtime();
		oput << "Multiply and potential merge:\t" << (t1-t2) << endl;

		bool remoteA = false;
		bool remoteB = false;

		delete ARecv1;		// free the memory of the previous first half
		delete BRecv1;

		SpParHelper::UnlockWindows(Aowner, rowwins2);	// Finish the second half
		SpParHelper::UnlockWindows(Bowner, colwins2);	

		t2 = MPI::Wtime();
		oput << "Unmatched communication:\t" << (t2-t1) << endl;

		Aowner = (i+Aoffset) % stages;		
		Bowner = (i+Boffset) % stages;

		// start fetching the current first half 
		SpParHelper::LockNFetch(ARecv1, Aowner, rowwins1, row_group, ARecvSizes1);
		SpParHelper::LockNFetch(BRecv1, Bowner, colwins1, col_group, BRecvSizes1);
	
		// while multiplying the already completed previous second halfs
		C_cont = MultiplyReturnTuples<SR>(*ARecv2, *BRecv2, false, true);	
		if(!C_cont->isZero()) 
			lowmerge.push_back(C_cont);

		t1 = MPI::Wtime();
		oput << "Multiply:\t" << (t1-t2) << endl;

		delete ARecv2;		// free memory of the previous second half
		delete BRecv2;

		// wait for the current first half to complte
		SpParHelper::UnlockWindows(Aowner, rowwins1);
		SpParHelper::UnlockWindows(Bowner, colwins1);
		
		t2 = MPI::Wtime();
		oput << "Unmatched communication:\t" << (t2-t1) << endl;
		
		// start prefetching the current second half 
		SpParHelper::LockNFetch(ARecv2, Aowner, rowwins2, row_group, ARecvSizes2);
		SpParHelper::LockNFetch(BRecv2, Bowner, colwins2, col_group, BRecvSizes2);
	}

	SpTuples<IU,N_promote> * C_cont = MultiplyReturnTuples<SR>(*ARecv1, *BRecv1, false, true);
	if(!C_cont->isZero()) 
		lowmerge.push_back(C_cont);

	t1 = MPI::Wtime();
	oput << "Multiply:\t" << (t1-t2) << endl;

	delete ARecv1;		// free the memory of the previous first half
	delete BRecv1;
	
	SpParHelper::UnlockWindows(Aowner, rowwins2);
	SpParHelper::UnlockWindows(Bowner, colwins2);

	t2 = MPI::Wtime();
	oput << "Unmatched communication:\t" << (t2-t1) << endl;

	C_cont = MultiplyReturnTuples<SR>(*ARecv2, *BRecv2, false, true);	
	if(!C_cont->isZero()) 
		lowmerge.push_back(C_cont);		
		
	{
		SpTuples<IU,N_promote> * InterC = MergeAll<SR>(lowmerge, C_m, C_n);
		for(int i=0; i< lowmerge.size(); ++i)
		{
			delete lowmerge[i];
		}
		lowmerge.clear();
		highmerge.push_back(InterC);
	}

	t1 = MPI::Wtime();
	oput << "Multiply and last intermediate merge:\t" << (t1-t2) << endl;

	delete ARecv2;
	delete BRecv2;

	SpHelper::deallocate2D(ARecvSizes1, UDERA::esscount);
	SpHelper::deallocate2D(ARecvSizes2, UDERA::esscount);
	SpHelper::deallocate2D(BRecvSizes1, UDERB::esscount);
	SpHelper::deallocate2D(BRecvSizes2, UDERB::esscount);
			
	DER_promote * C = new DER_promote(*(MergeAll<SR>(highmerge, C_m, C_n)), false, NULL);	// First get the result in SpTuples, then convert to UDER
	for(int i=0; i<highmerge.size(); ++i)
	{
		delete highmerge[i];
	}
	
	t2 = MPI::Wtime();
	oput << "Deallocation and merge:\t" << (t2-t1) << endl;

	SpParHelper::FreeWindows(rowwins1);
	SpParHelper::FreeWindows(rowwins2);
	SpParHelper::FreeWindows(colwins1);
	SpParHelper::FreeWindows(colwins2);	

	t1 = MPI::Wtime();
	oput << "Free windows:\t" << (t1-t2) << endl;
	
	(A.spSeq)->Merge(A1seq, A2seq);
	(B.spSeq)->Merge(B1seq, B2seq);	
	
	const_cast< UDERB* >(B.spSeq)->Transpose();	// transpose back to original

	t2 = MPI::Wtime();
	oput << "Retranspose and remerge:\t" << (t2-t1)	<< endl;

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

	DER_promote * C = new DER_promote(*(MergeAll<SR>(tomerge, C_m, C_n)), false, NULL);	// First get the result in SpTuples, then convert to UDER
	for(int i=0; i<tomerge.size(); ++i)
	{
		delete tomerge[i];
	}
	
	SpHelper::deallocate2D(ARecvSizes, UDERA::esscount);
	SpHelper::deallocate2D(BRecvSizes, UDERB::esscount);

	
	const_cast< UDERB* >(B.spSeq)->Transpose();	// transpose back to original
	
	return SpParMat<IU,N_promote,DER_promote> (C, GridC);			// return the result object
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
#endif

