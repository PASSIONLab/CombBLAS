#ifndef _PAR_FRIENDS_H_
#define _PAR_FRIENDS_H_

#include <iostream>
#include "SpParMat.h"	
#include "SpParHelper.h"
#include "MPIType.h"

using namespace std;

template <class IT, class NT, class DER>
class SpParMat;




/*************************************************************************************************/
/**************************** FRIEND FUNCTIONS FOR PARALLEL CLASSES ******************************/
/*************************************************************************************************/

template <typename SR, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB> 
SpParMat<IU,typename promote_trait<NU1,NU2>::T_promote,typename promote_trait<UDERA,UDERB>::T_promote> Mult_AnXBn 
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

	ofstream oput;
	GridC->OpenDebugFile("deb", oput);

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
	C->PrintInfo();
	
	SpHelper::deallocate2D(ARecvSizes, UDERA::esscount);
	SpHelper::deallocate2D(BRecvSizes, UDERB::esscount);

	
	const_cast< UDERB* >(B.spSeq)->Transpose();	// transpose back to original
	
	return SpParMat<IU,N_promote,DER_promote> (C, GridC);			// return the result object
}


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
		
	UDERA A1seq, A2seq;
	(A.spSeq)->Split( A1seq, A2seq); 

	const_cast< UDERB* >(B.spSeq)->Transpose();
	UDERB B1seq, B2seq;
	(B.spSeq)->Split( B1seq, B2seq); 
	
	// Create row and column windows (collective operation, i.e. everybody exposes its window to others)
	vector<MPI::Win> rowwins1, rowwins2, colwins1, colwins2;
	SpParHelper::SetWindows((A.commGrid)->GetRowWorld(), A1seq, rowwins1);
	SpParHelper::SetWindows((A.commGrid)->GetRowWorld(), A2seq, rowwins2);
	SpParHelper::SetWindows((B.commGrid)->GetColWorld(), B1seq, colwins1);
	SpParHelper::SetWindows((B.commGrid)->GetColWorld(), B2seq, colwins2);

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
	
	UDERA * ARecv1, * ARecv2; 
	UDERB * BRecv1, * BRecv2;
	vector< SpTuples<IU,N_promote>  *> tomerge;

	MPI::Group row_group = (A.commGrid)->GetRowWorld().Get_group();
	MPI::Group col_group = (B.commGrid)->GetColWorld().Get_group();

	int Aself = (A.commGrid)->GetRankInProcRow();
	int Bself = (B.commGrid)->GetRankInProcCol();	

	// Start exposure epochs to first set of windows
	SpParHelper::PostExposureEpoch(Aself, rowwins1, row_group);
	SpParHelper::PostExposureEpoch(Bself, colwins1, col_group);

	int Aowner = (0+Aoffset) % stages;		
	int Bowner = (0+Boffset) % stages;

	if(Aowner == Aself)
	{	
		ARecv1 = A1seq;		// shallow-copy 
	}
	else
	{
		SpParHelper::AccessNFetch(ARecv1, Aowner, rowwins1, row_group, ARecvSizes1);

		for(int j=0; j< rowwins1.size(); ++j)
			rowwins1[j].Complete();

		SpParHelper::AccessNFetch(ARecv2, Aowner, rowwins2, row_group, ARecvSizes2);	// Start prefetching next half 


	}
	if(Bowner == Bself)
	{
		BRecv1 = B1seq;		// shallow-copy
	}
	else
	{
		SpParHelper::StartAccessEpoch(Bownind, colwins1, col_group);

		vector<IU> ess(UDERB::esscount);		// pack essentials to a vector
		for(int j=0; j< UDERB::esscount; ++j)	
			ess[j] = BRecvSizes1[j][Bownind];	
	
		BRecv1 = new UDERB();
		SpParHelper::FetchMatrix(*BRecv1, ess, colwins1, Bownind);	// No lock version, only get()
	}

	for(int i = 1; i < stages; ++i) 
	{
		Aowner = (i+Aoffset) % stages;		
		Bowner = (i+Boffset) % stages;
		
		


	// End the exposure epochs for the arrays of the local matrices A and B
	// The Wait() call matches calls to Complete() issued by ** EACH OF THE ORIGIN PROCESSES ** that were granted access to the window during this epoch.
	for(int j=0; j< rowwindows.size(); ++j)
		rowwindows[j].Wait();
	for(int j=0; j< colwindows.size(); ++j)
		colwindows[j].Wait();
			
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

