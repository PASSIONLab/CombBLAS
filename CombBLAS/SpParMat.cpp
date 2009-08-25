/****************************************************************/
/* Sequential and Parallel Sparse Matrix Multiplication Library */
/* version 2.3 -------------------------------------------------*/
/* date: 01/18/2009 --------------------------------------------*/
/* author: Aydin Buluc (aydin@cs.ucsb.edu) ---------------------*/
/****************************************************************/

#include "SpParMat.h"
#include "ParFriends.h"

// If every processor has a distinct triples file such as {A_0, A_1, A_2,... A_p} for p processors
template <class IT, class NT, class DER>
SpParMat< IT,NT,DER >::SpParMat (ifstream & input, MPI::Intracomm & world)
{
	if(!input.is_open())
	{
		perror("Input file doesn't exist\n");
		exit(-1);
	}
	commGrid.reset(new CommGrid(world, 0, 0));
	input >> (*spSeq);
}

template <class IT, class NT, class DER>
SpParMat< IT,NT,DER >::SpParMat (DER * myseq, MPI::Intracomm & world): spSeq(myseq)
{
	commGrid.reset(new CommGrid(world, 0, 0));
}

template <class IT, class NT, class DER>
SpParMat< IT,NT,DER >::SpParMat (DER * myseq, shared_ptr<CommGrid> grid): spSeq(myseq)
{
	commGrid.reset(new CommGrid(*grid)); 
}	

// If there is a single file read by the master process only, use this and then call ReadDistribute()
template <class IT, class NT, class DER>
SpParMat< IT,NT,DER >::SpParMat ()
{
	spSeq = new DER();
	commGrid.reset(new CommGrid(MPI::COMM_WORLD, 0, 0));
}

template <class IT, class NT, class DER>
SpParMat< IT,NT,DER >::~SpParMat ()
{
	if(spSeq != NULL) delete spSeq;
}


template <class IT, class NT, class DER>
SpParMat< IT,NT,DER >::SpParMat (const SpParMat< IT,NT,DER > & rhs)
{
	if(rhs.spSeq != NULL)	
		spSeq = new DER(*(rhs.spSeq));  	// Deep copy of local block

	commGrid.reset(new CommGrid(*(rhs.commGrid)));		
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
	
		commGrid.reset(new CommGrid(*(rhs.commGrid)));		
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
			cout << "Grids are not comparable for parallel addition (A+B)" << endl; 
		}
	}
	else
	{
		cout<< "Missing feature (A+A): Use multiply with 2 instead !"<<endl;	
	}
	return *this;	
}

template <class IT, class NT, class DER>
IT SpParMat< IT,NT,DER >::getnnz() const
{
	IT totalnnz = 0;    
	IT localnnz = spSeq->getnnz();
	(commGrid->GetWorld()).Allreduce( &localnnz, &totalnnz, 1, MPIType<IT>(), MPI::SUM);
 	return totalnnz;  
}

template <class IT, class NT, class DER>
IT SpParMat< IT,NT,DER >::getnrow() const
{
	IT totalrows = 0;
	IT localrows = spSeq->getnrow();    
	(commGrid->GetColWorld()).Allreduce( &localrows, &totalrows, 1, MPIType<IT>(), MPI::SUM);
 	return totalrows;  
}

template <class IT, class NT, class DER>
IT SpParMat< IT,NT,DER >::getncol() const
{
	IT totalcols = 0;
	IT localcols = spSeq->getncol();    
	(commGrid->GetRowWorld()).Allreduce( &localcols, &totalcols, 1, MPIType<IT>(), MPI::SUM);
 	return totalcols;  
}

template <class IT, class NT, class DER>
template <typename NNT,typename NDER>	 
SpParMat<IT,NT,DER>::operator SpParMat<IT,NNT,NDER> () const
{
	NDER * convert = new NDER(*spSeq);
	return SpParMat<IT,NNT,NDER> (convert, commGrid);
}

/** 
 * Create a submatrix of size m x (size(ncols) * s) on a r x s processor grid
 * Essentially fetches the columns ci[0], ci[1],... ci[size(ci)] from every submatrix
 */
template <class IT, class NT, class DER>
SpParMat<IT,NT,DER> SpParMat<IT,NT,DER>::SubsRefCol (const vector<IT> & ci) const
{
	vector<IT> ri;
	DER * tempseq = new DER((*spSeq)(ri, ci)); 
	return SpParMat<IT,NT,DER> (tempseq, commGrid);	
} 

template <class IT, class NT, class DER>
SpParMat<IT,NT,DER> SpParMat<IT,NT,DER>::operator() (const vector<IT> & ri, const vector<IT> & ci) const
{
	int colneighs = commGrid->GetGridRows();	// number of neighbors along this processor column (including oneself)
	int rowneighs = commGrid->GetGridCols();	// number of neighbors along this processor row (including oneself)

	IT totalm = getnrow();
	IT totaln = getncol();
	IT m_perproc = totalm / colneighs;
	IT n_perproc = totaln / rowneighs;

	vector<IT> locri, locci;
	pair<IT,IT> rowboun, colboun; 
	if( commGrid->myprocrow !=  colneighs)	// not the last processor on the processor column
		rowboun = make_pair(m_perproc * commGrid->myprocrow, m_perproc * (commGrid->myprocrow + 1) );
	else
		rowboun = make_pair(m_perproc * commGrid->myprocrow, totalm);

	if( commGrid->myproccol !=  rowneighs)	// not the last processor on the processor row
		colboun = make_pair(n_perproc * commGrid->myproccol, n_perproc * (commGrid->myproccol + 1) );
	else
		colboun = make_pair(n_perproc * commGrid->myproccol, totaln);

	for(int i=0; i<ri.size(); ++i)
	{
		if( ri[i] >= rowboun.first && ri[i] < rowboun.second )	
			locri.push_back(ri[i]-rowboun.first);
	}
	for(int i=0; i<ci.size(); ++i)
	{
		if( ci[i] >= colboun.first && ci[i] < colboun.second )	
			locci.push_back(ci[i]-colboun.first);
	}
	DER * tempseq = new DER((*spSeq)(locri, locci)); 
	return SpParMat<IT,NT,DER> (tempseq, commGrid);	
} 


template <typename IU, typename NU1, typename NU2, typename UDER1, typename UDER2> 
SpParMat<IU,typename promote_trait<NU1,NU2>::T_promote,typename promote_trait<UDER1,UDER2>::T_promote> EWiseMult 
	(const SpParMat<IU,NU1,UDER1> & A, const SpParMat<IU,NU2,UDER2> & B , bool exclude)
{
	typedef typename promote_trait<NU1,NU2>::T_promote N_promote;
	typedef typename promote_trait<UDER1,UDER2>::T_promote DER_promote;

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

template <typename SR, typename IU, typename NU1, typename NU2, typename UDER1, typename UDER2> 
SpParMat<IU,typename promote_trait<NU1,NU2>::T_promote,typename promote_trait<UDER1,UDER2>::T_promote> Mult_AnXBn 
		(const SpParMat<IU,NU1,UDER1> & A, const SpParMat<IU,NU2,UDER2> & B )
{
	typedef typename promote_trait<NU1,NU2>::T_promote N_promote;
	typedef typename promote_trait<UDER1,UDER2>::T_promote DER_promote;
	
	if(A.getncol() != B.getnrow())
	{
		cout<<"Can not multiply, dimensions does not match"<<endl;
		MPI::COMM_WORLD.Abort(DIMMISMATCH);
		return SpParMat< IU,N_promote,DER_promote >();
	}

	int stages, Aoffset, Boffset; 	// stages = inner dimension of matrix blocks
	shared_ptr<CommGrid> GridC = ProductGrid((A.commGrid).get(), (B.commGrid).get(), stages, Aoffset, Boffset);		
		
	const_cast< UDER2* >(B.spSeq)->Transpose();
	
	// set row & col window handles
	vector<MPI::Win> rowwindows, colwindows;
	vector<MPI::Win> rowwinnext, colwinnext;
	SpParHelper::SetWindows((A.commGrid)->GetRowWorld(), *(A.spSeq), rowwindows);
	SpParHelper::SetWindows((B.commGrid)->GetColWorld(), *(B.spSeq), colwindows);
	SpParHelper::SetWindows((A.commGrid)->GetRowWorld(), *(A.spSeq), rowwinnext);
	SpParHelper::SetWindows((B.commGrid)->GetColWorld(), *(B.spSeq), colwinnext);
	
	IU ** ARecvSizes = SpHelper::allocate2D<IU>(UDER1::esscount, stages);
	IU ** BRecvSizes = SpHelper::allocate2D<IU>(UDER2::esscount, stages);
	
	SpParHelper::GetSetSizes( *(A.spSeq), ARecvSizes, (A.commGrid)->GetRowWorld());
	SpParHelper::GetSetSizes( *(B.spSeq), BRecvSizes, (B.commGrid)->GetColWorld());
	
	UDER1 * ARecv, * ARecvNext; 
	UDER2 * BRecv, * BRecvNext;
	vector< SpTuples<IU,N_promote>  *> tomerge;

	ofstream oput;
	GridC->OpenDebugFile("deb", oput);

	// Prefetch first
	for(int j=0; j< rowwindows.size(); ++j)
		MPI_Win_fence(MPI_MODE_NOSTORE, rowwindows[j]);
	for(int j=0; j< colwindows.size(); ++j)
		MPI_Win_fence(MPI_MODE_NOSTORE, colwindows[j]);

	// Prefetch first
	for(int j=0; j< rowwinnext.size(); ++j)
		MPI_Win_fence(MPI_MODE_NOSTORE, rowwinnext[j]);
	for(int j=0; j< colwinnext.size(); ++j)
		MPI_Win_fence(MPI_MODE_NOSTORE, colwinnext[j]);


	int Aownind = (0+Aoffset) % stages;		
	int Bownind = (0+Boffset) % stages;
	if(Aownind == (A.commGrid)->GetRankInProcRow())
	{	
		ARecv = A.spSeq;	// shallow-copy 
	}
	else
	{
		vector<IU> ess1(UDER1::esscount);		// pack essentials to a vector
		for(int j=0; j< UDER1::esscount; ++j)	
		{
			ess1[j] = ARecvSizes[j][Aownind];	
		}
		ARecv = new UDER1();	// create the object first	

		oput << "For A (out), Fetching " << (void*)rowwindows[0] << endl;
		SpParHelper::FetchMatrix(*ARecv, ess1, rowwindows, Aownind);	// fetch its elements later
	}
	if(Bownind == (B.commGrid)->GetRankInProcCol())
	{
		BRecv = B.spSeq;	// shallow-copy
	}
	else
	{
		vector<IU> ess2(UDER2::esscount);		// pack essentials to a vector
		for(int j=0; j< UDER2::esscount; ++j)	
		{
			ess2[j] = BRecvSizes[j][Bownind];	
		}	
		BRecv = new UDER2();

		oput << "For B (out), Fetching " << (void*)colwindows[0] << endl;
		SpParHelper::FetchMatrix(*BRecv, ess2, colwindows, Bownind);	// No lock version, only get !
	}

	int Aownprev = Aownind;
	int Bownprev = Bownind;
	
	for(int i = 1; i < stages; ++i) 	// Robust generalization to non-square grids will require block-cyclic distibution	
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
				vector<IU> ess1(UDER1::esscount);		// pack essentials to a vector
				for(int j=0; j< UDER1::esscount; ++j)	
				{
					ess1[j] = ARecvSizes[j][Aownind];	
				}
				ARecvNext = new UDER1();	// create the object first	

				oput << "For A, Fetching " << (void*) rowwinnext[0] << endl;
				SpParHelper::FetchMatrix(*ARecvNext, ess1, rowwinnext, Aownind);
			}
		
			if(Bownind == (B.commGrid)->GetRankInProcCol())
			{
				BRecvNext = B.spSeq;	// shallow-copy
			}
			else
			{
				vector<IU> ess2(UDER2::esscount);		// pack essentials to a vector
				for(int j=0; j< UDER2::esscount; ++j)	
				{
					ess2[j] = BRecvSizes[j][Bownind];	
				}		
				BRecvNext = new UDER2();

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
				vector<IU> ess1(UDER1::esscount);		// pack essentials to a vector
				for(int j=0; j< UDER1::esscount; ++j)	
				{
					ess1[j] = ARecvSizes[j][Aownind];	
				}
				ARecv = new UDER1();	// create the object first	

				oput << "For A, Fetching " << (void*) rowwindows[0] << endl;
				SpParHelper::FetchMatrix(*ARecv, ess1, rowwindows, Aownind);
			}
		
			if(Bownind == (B.commGrid)->GetRankInProcCol())
			{
				BRecv = B.spSeq;	// shallow-copy
			}
			else
			{
				vector<IU> ess2(UDER2::esscount);		// pack essentials to a vector
				for(int j=0; j< UDER2::esscount; ++j)	
				{
					ess2[j] = BRecvSizes[j][Bownind];	
				}		
				BRecv = new UDER2();

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
			MPI_Win_fence(MPI_MODE_NOSTORE, rowwindows[j]);		// Synch using "prev" windows
		for(int j=0; j< colwindows.size(); ++j)
			MPI_Win_fence(MPI_MODE_NOSTORE, colwindows[j]);

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
			MPI_Win_fence(MPI_MODE_NOSTORE, rowwinnext[j]);		// Synch using "prev" windows
		for(int j=0; j< colwinnext.size(); ++j)
			MPI_Win_fence(MPI_MODE_NOSTORE, colwinnext[j]);

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
	
	SpHelper::deallocate2D(ARecvSizes, UDER1::esscount);
	SpHelper::deallocate2D(BRecvSizes, UDER2::esscount);

	
	const_cast< UDER2* >(B.spSeq)->Transpose();	// transpose back to original
	
	return SpParMat<IU,N_promote,DER_promote> (C, GridC);			// return the result object
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
		cout << "Grids are not comparable, EWiseMult() fails !" << endl; 
		MPI::COMM_WORLD.Abort(DIMMISMATCH);
	}	
}

template <class IT, class NT, class DER>
void SpParMat<IT,NT,DER>::EWiseScale(DenseParMat<IT, NT> & rhs)
{
	if(*commGrid == *rhs.commGrid)	
	{
		spSeq->EWiseScale(rhs.array, rhs.m, rhs.n);	// Dimension compatibility check performed by sequential function
	}
	else
	{
		cout << "Grids are not comparable, EWiseScale() fails !" << endl; 
			MPI::COMM_WORLD.Abort(DIMMISMATCH);
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
			cout << "Matrices have different dimensions, UpdateDense() fails !" << endl;
			MPI::COMM_WORLD.Abort(DIMMISMATCH);
		}
	}
	else
	{
		cout << "Grids are not comparable, UpdateDense() fails !" << endl; 
		MPI::COMM_WORLD.Abort(GRIDMISMATCH);
	}
}

template <class IT, class NT, class DER>
void SpParMat<IT,NT,DER>::PrintInfo() const
{
	IT mm = getnrow(); 
	IT nn = getncol();
	IT nznz = getnnz();
	
	if (commGrid->myrank == 0)	
		cout << "As a whole: " << mm << " rows and "<< nn <<" columns and "<<  nznz << " nonzeros" << endl; 

	if ((commGrid->grrows * commGrid->grcols) ==  1)
		spSeq->PrintInfo();
}

template <class IT, class NT, class DER>
bool SpParMat<IT,NT,DER>::operator== (const SpParMat<IT,NT,DER> & rhs) const
{
	int local = static_cast<int>((*spSeq) == (*(rhs.spSeq)));
	int whole = 1;
	commGrid->GetWorld().Allreduce( &local, &whole, 1, MPI::INT, MPI::BAND);
	return static_cast<bool>(whole);	
}


template <class IT, class NT, class DER>
void SpParMat<IT,NT,DER>::Transpose()
{
	#define TRTAGNZ 121
	#define TRTAGM 122
	#define TRTAGN 123
	#define TRTAGROWS 124
	#define TRTAGCOLS 125
	#define TRTAGVALS 126

	ofstream oput;
	commGrid->OpenDebugFile("transpose", oput);

	if(commGrid->myproccol == commGrid->myprocrow)	// Diagonal
	{
		spSeq->Transpose();			
	}
	else
	{
		SpTuples<IT,NT> Atuples(*spSeq);
		IT locnnz = Atuples.getnnz();
		IT * rows = new IT[locnnz];
		IT * cols = new IT[locnnz];
		NT * vals = new NT[locnnz];
		for(IT i=0; i < locnnz; ++i)
		{
			rows[i] = Atuples.colindex(i);	// swap (i,j) here
			cols[i] = Atuples.rowindex(i);
			vals[i] = Atuples.numvalue(i);
		}

		IT locm = getlocalcols();
		IT locn = getlocalrows();
		delete spSeq;

		IT remotem, remoten, remotennz;
		swap(locm,locn);
		int diagneigh = commGrid->GetComplementRank();

		commGrid->GetWorld().Sendrecv(&locnnz, 1, MPIType<IT>(), diagneigh, TRTAGNZ, &remotennz, 1, MPIType<IT>(), diagneigh, TRTAGNZ);
		commGrid->GetWorld().Sendrecv(&locn, 1, MPIType<IT>(), diagneigh, TRTAGM, &remotem, 1, MPIType<IT>(), diagneigh, TRTAGM);
		commGrid->GetWorld().Sendrecv(&locm, 1, MPIType<IT>(), diagneigh, TRTAGN, &remoten, 1, MPIType<IT>(), diagneigh, TRTAGN);
		oput << "Locs: " << locm << " " << locn << " " << locnnz << endl;
		oput << "Rems: " << remotem << " " << remoten << " " << remotennz << endl;
		oput << "Diagonal:" << diagneigh << endl;

		
		IT * rowsrecv = new IT[remotennz];
		commGrid->GetWorld().Sendrecv(rows, locnnz, MPIType<IT>(), diagneigh, TRTAGROWS, rowsrecv, remotennz, MPIType<IT>(), diagneigh, TRTAGROWS);
		delete [] rows;

		IT * colsrecv = new IT[remotennz];
		commGrid->GetWorld().Sendrecv(cols, locnnz, MPIType<IT>(), diagneigh, TRTAGCOLS, colsrecv, remotennz, MPIType<IT>(), diagneigh, TRTAGCOLS);
		delete [] cols;

		NT * valsrecv = new NT[remotennz];
		commGrid->GetWorld().Sendrecv(vals, locnnz, MPIType<NT>(), diagneigh, TRTAGVALS, valsrecv, remotennz, MPIType<NT>(), diagneigh, TRTAGVALS);
		delete [] vals;

		tuple<IT,IT,NT> * arrtuples = new tuple<IT,IT,NT>[remotennz];
		for(IT i=0; i< remotennz; ++i)
		{
			arrtuples[i] = make_tuple(rowsrecv[i], colsrecv[i], valsrecv[i]);
		}	
		DeleteAll(rowsrecv, colsrecv, valsrecv);
		ColLexiCompare<IT,NT> collexicogcmp;
		sort(arrtuples , arrtuples+remotennz, collexicogcmp );	// sort w.r.t columns here

		spSeq = new DER();
		spSeq->Create( remotennz, remotem, remoten, arrtuples);		// the deletion of arrtuples[] is handled by SpMat::Create
	}	
}		

//! Handles all sorts of orderings as long as there are no duplicates
//! May perform better when the data is already reverse column-sorted (i.e. in decreasing order)
template <class IT, class NT, class DER>
ifstream& SpParMat< IT,NT,DER >::ReadDistribute (ifstream& infile, int master)
{
	IT total_m, total_n, total_nnz;
	IT m_perproc, n_perproc;

	int colneighs = commGrid->GetGridRows();	// number of neighbors along this processor column (including oneself)
	int rowneighs = commGrid->GetGridCols();	// number of neighbors along this processor row (including oneself)

	IT buffpercolneigh = MEMORYINBYTES / (colneighs * (2 * sizeof(IT) + sizeof(NT)));
	IT buffperrowneigh = MEMORYINBYTES / (rowneighs * (2 * sizeof(IT) + sizeof(NT)));

	// make sure that buffperrowneigh >= buffpercolneigh to cover for this patological case:
	//   	-- all data received by a given column head (by vertical communication) are headed to a single processor along the row
	//   	-- then making sure buffperrowneigh >= buffpercolneigh guarantees that the horizontal buffer will never overflow
	buffperrowneigh = std::max(buffperrowneigh, buffpercolneigh);

	IT * cdispls = new IT[colneighs];
	for (int i=0; i<colneighs; ++i)
		cdispls[i] = i*buffpercolneigh;

	IT * rdispls = new IT[colneighs];
	for (int i=0; i<rowneighs; ++i)
		rdispls[i] = i*buffperrowneigh;		

	IT *ccurptrs, *rcurptrs;
	IT recvcount;

	IT * rows; 
	IT * cols;
	NT * vals;

	// Note: all other column heads that initiate the horizontal communication has the same "rankinrow" with the master
	int rankincol = commGrid->GetRankInProcCol(master);	// get master's rank in its processor column
	int rankinrow = commGrid->GetRankInProcRow(master);	
  	
	vector< tuple<IT, IT, NT> > localtuples;

	if(commGrid->GetRank() == master)	// 1 processor
	{		
		// allocate buffers on the heap as stack space is usually limited
		rows = new IT [ buffpercolneigh * colneighs ];
		cols = new IT [ buffpercolneigh * colneighs ];
		vals = new NT [ buffpercolneigh * colneighs ];

		ccurptrs = new IT[colneighs];
		rcurptrs = new IT[rowneighs];
		fill_n(ccurptrs, colneighs, (IT) zero);	// fill with zero
		fill_n(rcurptrs, rowneighs, (IT) zero);	
		
		if (infile.is_open())
		{
			infile >> total_m >> total_n >> total_nnz;
			m_perproc = total_m / colneighs;
			n_perproc = total_n / rowneighs;
	
			(commGrid->commWorld).Bcast(&total_m, 1, MPIType<IT>(), master);
			(commGrid->commWorld).Bcast(&total_n, 1, MPIType<IT>(), master);
			
			IT temprow, tempcol;
			NT tempval;
			IT cnz = 0;
			while ( (!infile.eof()) && cnz < total_nnz)
			{
				infile >> temprow >> tempcol >> tempval;
				--temprow;	// file is 1-based where C-arrays are 0-based
				--tempcol;

				int colrec = std::min(temprow / m_perproc, colneighs-1);	// precipient processor along the column
				rows[ colrec * buffpercolneigh + ccurptrs[colrec] ] = temprow;
				cols[ colrec * buffpercolneigh + ccurptrs[colrec] ] = tempcol;
				vals[ colrec * buffpercolneigh + ccurptrs[colrec] ] = tempval;
				++ (ccurptrs[colrec]);				

				if(ccurptrs[colrec] == buffpercolneigh || (cnz == (total_nnz-1)) )		// one buffer is full, or file is done !
				{
					// first, send the receive counts ...
					(commGrid->colWorld).Scatter(ccurptrs, 1, MPIType<IT>(), &recvcount, 1, MPIType<IT>(), rankincol);

					// generate space for own recv data ... (use arrays because vector<bool> is cripled, if NT=bool)
					IT * temprows = new IT[recvcount];
					IT * tempcols = new IT[recvcount];
					NT * tempvals = new NT[recvcount];
					
					// then, send all buffers that to their recipients ...
					(commGrid->colWorld).Scatterv(rows, ccurptrs, cdispls, MPIType<IT>(), temprows, recvcount,  MPIType<IT>(), rankincol); 
					(commGrid->colWorld).Scatterv(cols, ccurptrs, cdispls, MPIType<IT>(), tempcols, recvcount,  MPIType<IT>(), rankincol); 
					(commGrid->colWorld).Scatterv(vals, ccurptrs, cdispls, MPIType<NT>(), tempvals, recvcount,  MPIType<NT>(), rankincol); 

					// finally, reset current pointers !
					fill_n(ccurptrs, colneighs, (IT) zero);
					DeleteAll(rows, cols, vals);
			
					/* Begin horizontal distribution */

					rows = new IT [ buffperrowneigh * rowneighs ];
					cols = new IT [ buffperrowneigh * rowneighs ];
					vals = new NT [ buffperrowneigh * rowneighs ];
			
					// prepare to send the data along the horizontal
					for(IT i=zero; i< recvcount; ++i)
					{
						int rowrec = std::min(tempcols[i] / n_perproc, rowneighs-1);
						rows[ rowrec * buffperrowneigh + rcurptrs[rowrec] ] = temprows[i];
						cols[ rowrec * buffperrowneigh + rcurptrs[rowrec] ] = tempcols[i];
						vals[ rowrec * buffperrowneigh + rcurptrs[rowrec] ] = tempvals[i];
						++ (rcurptrs[rowrec]);	
					}
				
					// Send the receive counts for horizontal communication ...
					(commGrid->rowWorld).Scatter(rcurptrs, 1, MPIType<IT>(), &recvcount, 1, MPIType<IT>(), rankinrow);

					// the data is now stored in rows/cols/vals, can reset temporaries
					// sets size and capacity to new recvcount
					DeleteAll(temprows, tempcols, tempvals);
					temprows = new IT[recvcount];
					tempcols = new IT[recvcount];
					tempvals = new NT[recvcount];

					// then, send all buffers that to their recipients ...
					(commGrid->rowWorld).Scatterv(rows, rcurptrs, rdispls, MPIType<IT>(), temprows, recvcount,  MPIType<IT>(), rankinrow); 
					(commGrid->rowWorld).Scatterv(cols, rcurptrs, rdispls, MPIType<IT>(), tempcols, recvcount,  MPIType<IT>(), rankinrow); 
					(commGrid->rowWorld).Scatterv(vals, rcurptrs, rdispls, MPIType<NT>(), tempvals, recvcount,  MPIType<NT>(), rankinrow); 

					// now push what is ours to tuples
					IT moffset = commGrid->myprocrow * m_perproc; 
					IT noffset = commGrid->myproccol * n_perproc; 
					for(IT i=zero; i< recvcount; ++i)
					{					
						localtuples.push_back( 	make_tuple(temprows[i]-moffset, tempcols[i]-noffset, tempvals[i]) );
					}
					
					fill_n(rcurptrs, rowneighs, (IT) zero);
					DeleteAll(rows, cols, vals, temprows, tempcols, tempvals);		
					
					// reuse these buffers for the next vertical communication								
					rows = new IT [ buffpercolneigh * colneighs ];
					cols = new IT [ buffpercolneigh * colneighs ];
					vals = new NT [ buffpercolneigh * colneighs ];
				}
				++ cnz;
			}
			assert (cnz == total_nnz);
			
			// Signal the end of file to other processors along the column
			fill_n(ccurptrs, colneighs, numeric_limits<IT>::max());	
			(commGrid->colWorld).Scatter(ccurptrs, 1, MPIType<IT>(), &recvcount, 1, MPIType<IT>(), rankincol);

			// And along the row ...
			fill_n(rcurptrs, rowneighs, numeric_limits<IT>::max());				
			(commGrid->rowWorld).Scatter(rcurptrs, 1, MPIType<IT>(), &recvcount, 1, MPIType<IT>(), rankinrow);
			
		}
		DeleteAll(rows,cols,vals, ccurptrs, rcurptrs);

	}
	else if( commGrid->OnSameProcCol(master) ) 	// (r-1) processors
	{
		(commGrid->commWorld).Bcast(&total_m, 1, MPIType<IT>(), master);
		(commGrid->commWorld).Bcast(&total_n, 1, MPIType<IT>(), master);
		m_perproc = total_m / colneighs;
		n_perproc = total_n / rowneighs;

		// void MPI::Comm::Scatterv(const void* sendbuf, const int sendcounts[], const int displs[], const MPI::Datatype& sendtype,
		//				void* recvbuf, int recvcount, const MPI::Datatype & recvtype, int root) const
		// The outcome is as if the root executed n send operations,
    		//	MPI_Send(sendbuf + displs[i] * extent(sendtype), sendcounts[i], sendtype, i, ...)
		// and each process executed a receive,
   		// 	MPI_Recv(recvbuf, recvcount, recvtype, root, ...)
		// The send buffer is ignored for all nonroot processes.

		while(true)
		{
			// first receive the receive counts ...
			(commGrid->colWorld).Scatter(ccurptrs, 1, MPIType<IT>(), &recvcount, 1, MPIType<IT>(), rankincol);

			if( recvcount == numeric_limits<IT>::max())
				break;
	
			// create space for incoming data ... 
			IT * temprows = new IT[recvcount];
			IT * tempcols = new IT[recvcount];
			NT * tempvals = new NT[recvcount];

			// receive actual data ... (first 4 arguments are ignored in the receiver side)
			(commGrid->colWorld).Scatterv(rows, ccurptrs, cdispls, MPIType<IT>(), temprows, recvcount,  MPIType<IT>(), rankincol); 
			(commGrid->colWorld).Scatterv(cols, ccurptrs, cdispls, MPIType<IT>(), tempcols, recvcount,  MPIType<IT>(), rankincol); 
			(commGrid->colWorld).Scatterv(vals, ccurptrs, cdispls, MPIType<NT>(), tempvals, recvcount,  MPIType<NT>(), rankincol); 

			// now, send the data along the horizontal
			rcurptrs = new IT[rowneighs];
			fill_n(rcurptrs, rowneighs, (IT) zero);	
		
			rows = new IT [ buffperrowneigh * rowneighs ];
			cols = new IT [ buffperrowneigh * rowneighs ];
			vals = new NT [ buffperrowneigh * rowneighs ];

			// prepare to send the data along the horizontal
			for(IT i=zero; i< recvcount; ++i)
			{
				IT rowrec = std::min(tempcols[i] / n_perproc, rowneighs-1);
				rows[ rowrec * buffperrowneigh + rcurptrs[rowrec] ] = temprows[i];
				cols[ rowrec * buffperrowneigh + rcurptrs[rowrec] ] = tempcols[i];
				vals[ rowrec * buffperrowneigh + rcurptrs[rowrec] ] = tempvals[i];
				++ (rcurptrs[rowrec]);	
			}
				
			// Send the receive counts for horizontal communication ...
			(commGrid->rowWorld).Scatter(rcurptrs, 1, MPIType<IT>(), &recvcount, 1, MPIType<IT>(), rankinrow);
			
			// the data is now stored in rows/cols/vals, can reset temporaries
			// sets size and capacity to new recvcount
			DeleteAll(temprows, tempcols, tempvals);
			temprows = new IT[recvcount];
			tempcols = new IT[recvcount];
			tempvals = new NT[recvcount];
			
			// then, send all buffers that to their recipients ...
			(commGrid->rowWorld).Scatterv(rows, rcurptrs, rdispls, MPIType<IT>(), temprows, recvcount,  MPIType<IT>(), rankinrow); 
			(commGrid->rowWorld).Scatterv(cols, rcurptrs, rdispls, MPIType<IT>(), tempcols, recvcount,  MPIType<IT>(), rankinrow); 
			(commGrid->rowWorld).Scatterv(vals, rcurptrs, rdispls, MPIType<NT>(), tempvals, recvcount,  MPIType<NT>(), rankinrow); 

			// now push what is ours to tuples
			IT moffset = commGrid->myprocrow * m_perproc; 
			IT noffset = commGrid->myproccol * n_perproc;
			for(IT i=zero; i< recvcount; ++i)
			{					
				localtuples.push_back( 	make_tuple(temprows[i]-moffset, tempcols[i]-noffset, tempvals[i]) );
			}
					
			fill_n(rcurptrs, rowneighs, (IT) zero);
			DeleteAll(rows, cols, vals,temprows, tempcols, tempvals);	
		}
		// Signal the end of file to other processors along the row
		fill_n(rcurptrs, rowneighs, numeric_limits<IT>::max());				
		(commGrid->rowWorld).Scatter(rcurptrs, 1, MPIType<IT>(), &recvcount, 1, MPIType<IT>(), rankinrow);
		delete [] rcurptrs;	
	}
	else		// r * (s-1) processors that only participate in the horizontal communication step
	{
		(commGrid->commWorld).Bcast(&total_m, 1, MPIType<IT>(), master);
		(commGrid->commWorld).Bcast(&total_n, 1, MPIType<IT>(), master);
		m_perproc = total_m / colneighs;
		n_perproc = total_n / rowneighs;
		
		while (true)
		{
			// receive the receive count
			(commGrid->rowWorld).Scatter(rcurptrs, 1, MPIType<IT>(), &recvcount, 1, MPIType<IT>(), rankinrow);
			if( recvcount == numeric_limits<IT>::max())
				break;
		
			// create space for incoming data ... 
			IT * temprows = new IT[recvcount];
			IT * tempcols = new IT[recvcount];
			NT * tempvals = new NT[recvcount];

			(commGrid->rowWorld).Scatterv(rows, rcurptrs, rdispls, MPIType<IT>(), &temprows[0], recvcount,  MPIType<IT>(), rankinrow); 
			(commGrid->rowWorld).Scatterv(cols, rcurptrs, rdispls, MPIType<IT>(), &tempcols[0], recvcount,  MPIType<IT>(), rankinrow); 
			(commGrid->rowWorld).Scatterv(vals, rcurptrs, rdispls, MPIType<NT>(), &tempvals[0], recvcount,  MPIType<NT>(), rankinrow);

			// now push what is ours to tuples
			IT moffset = commGrid->myprocrow * m_perproc; 
			IT noffset = commGrid->myproccol * n_perproc;
			for(IT i=zero; i< recvcount; ++i)
			{					
				localtuples.push_back( 	make_tuple(temprows[i]-moffset, tempcols[i]-noffset, tempvals[i]) );
			}
			DeleteAll(temprows,tempcols,tempvals);
		}
	
	}
	
	DeleteAll(cdispls, rdispls);
	tuple<IT,IT,NT> * arrtuples = new tuple<IT,IT,NT>[localtuples.size()];	// the vector will go out of scope, make it stick !
	copy(localtuples.begin(), localtuples.end(), arrtuples);

 	IT localm = (commGrid->myprocrow != (commGrid->grrows-1))? m_perproc: (total_m - (m_perproc * (commGrid->grrows-1)));
 	IT localn = (commGrid->myproccol != (commGrid->grcols-1))? n_perproc: (total_n - (n_perproc * (commGrid->grcols-1)));
	
	spSeq->Create( localtuples.size(), localm, localn, arrtuples);		// the deletion of arrtuples[] is handled by SpMat::Create

	return infile;
}



template <class IT, class NT, class DER>
ofstream& SpParMat<IT,NT,DER>::put(ofstream& outfile) const
{
	outfile << (*spSeq) << endl;
}

template <class IU, class NU, class UDER>
ofstream& operator<<(ofstream& outfile, const SpParMat<IU, NU, UDER> & s)
{
	return s.put(outfile) ;	// use the right put() function

}
