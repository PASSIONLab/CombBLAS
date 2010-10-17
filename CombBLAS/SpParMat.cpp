/****************************************************************/
/* Sequential and Parallel Sparse Matrix Multiplication Library */
/* version 2.3 -------------------------------------------------*/
/* date: 01/18/2009 --------------------------------------------*/
/* author: Aydin Buluc (aydin@cs.ucsb.edu) ---------------------*/
/****************************************************************/

#include "SpParMat.h"
#include "ParFriends.h"
#include "Operations.h"
#include <fstream>
using namespace std;

/**
  * If every processor has a distinct triples file such as {A_0, A_1, A_2,... A_p} for p processors
 **/
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

/**
  * If there is a single file read by the master process only, use this and then call ReadDistribute()
  * Since this is the default constructor, you don't need to explicitly call it, just a declaration will call it
 **/
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
void SpParMat<IT,NT,DER>::DimScale(const DenseParVec<IT,NT> & v, Dim dim)
{
	switch(dim)
	{
		case Column:	// scale each "Column", using a row vector
		{
			// Diagonal processor broadcast data so that everyone gets the scaling vector 
			NT * scaler = NULL;
			int root = commGrid->GetDiagOfProcCol();
			if(v.diagonal)
			{	
				scaler = const_cast<NT*>(&v.arr[0]);	
			}
			else
			{	
				scaler = new NT[getlocalcols()];	
			}
			(commGrid->GetColWorld()).Bcast(scaler, getlocalcols(), MPIType<NT>(), root);	

			for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)	// iterate over columns
			{
				for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)
				{
					nzit.value() *=  scaler[colit.colid()];
				}
			}
			if(!v.diagonal)	delete [] scaler;
			break;
		}
		case Row:
		{
			NT * scaler = NULL;
			int root = commGrid->GetDiagOfProcRow();
			if(v.diagonal)
			{	
				scaler = const_cast<NT*>(&v.arr[0]);	
			}
			else
			{	
				scaler = new NT[getlocalrows()];	
			}
			(commGrid->GetRowWorld()).Bcast(scaler, getlocalrows(), MPIType<NT>(), root);	

			for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)
			{
				for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)
				{
					nzit.value() *= scaler[nzit.rowid()];
				}
			}
			if(!v.diagonal)	delete [] scaler;			
			break;
		}
		default:
		{
			cout << "Unknown scaling dimension, returning..." << endl;
			break;
		}
	}
}

/**
  * Reduce along the column/row into a vector
  * @param[in] __binary_op {the operation used for reduction; examples: max, min, plus, multiply, and, or}
  * @param[in] id {scalar that is used as the identity for __binary_op; examples: zero, infinity}
  * @param[in] __unary_op {optional unary operation applied to nonzeros *before* the __binary_op; examples: 1/x, x^2}
 **/ 
template <class IT, class NT, class DER>
template <typename _BinaryOperation, typename _UnaryOperation>	
DenseParVec<IT,NT> SpParMat<IT,NT,DER>::Reduce(Dim dim, _BinaryOperation __binary_op, NT id, _UnaryOperation __unary_op) const
{
	DenseParVec<IT,NT> parvec(commGrid, id);

	switch(dim)
	{
		case Row:	// pack along the columns, result is a "Row" vector of size n
		{
			NT * sendbuf = new NT[getlocalcols()];
			fill(sendbuf, sendbuf+getlocalcols(), id);	// fill with identity

			for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)	// iterate over columns
			{
				for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)
				{
					sendbuf[colit.colid()] = __binary_op(__unary_op(nzit.value()), sendbuf[colit.colid()]);
				}
			}
			NT * recvbuf = NULL;
			int root = commGrid->GetDiagOfProcCol();
			if(parvec.diagonal)
			{
				parvec.arr.resize(getlocalcols());
				recvbuf = &parvec.arr[0];	
			}
			(commGrid->GetColWorld()).Reduce(sendbuf, recvbuf, getlocalcols(), MPIType<NT>(), MPIOp<_BinaryOperation, NT>::op(), root);
			delete [] sendbuf;
			break;
		}
		case Column:	// pack along the rows, result is a "Column" vector of size m
		{
			NT * sendbuf = new NT[getlocalrows()];
			fill(sendbuf, sendbuf+getlocalcols(), id);	// fill with identity
			
			for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)	// iterate over columns
			{
				for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)
				{
					sendbuf[nzit.rowid()] = __binary_op(__unary_op(nzit.value()), sendbuf[nzit.rowid()]);
				}
			}
			NT * recvbuf = NULL;
			int root = commGrid->GetDiagOfProcRow();
			if(parvec.diagonal)
			{
				parvec.arr.resize(getlocalrows());
				recvbuf = &parvec.arr[0];	
			}
			(commGrid->GetRowWorld()).Reduce(sendbuf, recvbuf, getlocalrows(), MPIType<NT>(), MPIOp<_BinaryOperation, NT>::op(), root);
			delete [] sendbuf;
			break;
		}
		default:
		{
			cout << "Unknown reduction dimension, returning empty vector" << endl;
			break;
		}
	}
	return parvec;
}

template <class IT, class NT, class DER>
template <typename NNT,typename NDER>	 
SpParMat<IT,NT,DER>::operator SpParMat<IT,NNT,NDER> () const
{
	NDER * convert = new NDER(*spSeq);
	return SpParMat<IT,NNT,NDER> (convert, commGrid);
}

/** 
 * Create a submatrix of size m x (size(ci) * s) on a r x s processor grid
 * Essentially fetches the columns ci[0], ci[1],... ci[size(ci)] from every submatrix
 */
template <class IT, class NT, class DER>
SpParMat<IT,NT,DER> SpParMat<IT,NT,DER>::SubsRefCol (const vector<IT> & ci) const
{
	vector<IT> ri;
	DER * tempseq = new DER((*spSeq)(ri, ci)); 
	return SpParMat<IT,NT,DER> (tempseq, commGrid);	
} 

/** 
 * Generalized sparse matrix indexing
 * Both the storage and the actual values in SpParVec should be IT
 * The index vectors are distributed on diagonal processors
 * We can use this function to apply a permutation like A(p,q) 
 * Sequential indexing subroutine (via multiplication) is general enough.
 */
template <class IT, class NT, class DER>
SpParMat<IT,NT,DER> SpParMat<IT,NT,DER>::operator() (const SpParVec<IT,IT> & ri, const SpParVec<IT,IT> & ci) const
{
	// We create two boolean matrices P and Q
	// Dimensions:  P is size(ri) x m
	//		Q is n x size(ci) 

	int colneighs = commGrid->GetGridRows();	// number of neighbors along this processor column (including oneself)
	int rowneighs = commGrid->GetGridCols();	// number of neighbors along this processor row (including oneself)
	IT totalm = getnrow();	// collective call
	IT totaln = getncol();
	IT m_perproc = totalm / rowneighs;
	IT n_perproc = totaln / colneighs;
	IT p_nnz, q_nnz, rilen, cilen; 
	IT * pcnts;
	IT * qcnts;
	
	int diaginrow = commGrid->GetDiagOfProcRow();
	int diagincol = commGrid->GetDiagOfProcCol();

	// infer the concrete type SpMat<IT,IT>
	typedef typename create_trait<DER, IT, bool>::T_inferred DER_IT;
	DER_IT * PSeq;
	DER_IT * QSeq;

	if(ri.diagonal)		// only the diagonal processors hold vectors
	{
		// broadcast the size 
		rilen = ri.length;
		cilen = ci.length;
		(commGrid->rowWorld).Bcast(&rilen, 1, MPIType<IT>(), diaginrow);
		(commGrid->colWorld).Bcast(&cilen, 1, MPIType<IT>(), diagincol);

		vector< vector<IT> > rowdata_rowid(rowneighs);
		vector< vector<IT> > rowdata_colid(rowneighs);
	
		vector< vector<IT> > coldata_rowid(colneighs);
		vector< vector<IT> > coldata_colid(colneighs);

		IT locvecr = ri.ind.size();	// nnz in local vector
		for(IT i=0; i < locvecr; ++i)
		{	
			// make 1-based indices 0-based
			int rowrec = std::min((ri.num[i]-1) / m_perproc, rowneighs-1);	// precipient processor along the column

			// ri's numerical values give the colids and its indices give rowids
			// thus, the rowid's are already offset'd where colid's are not
			rowdata_rowid[rowrec].push_back(ri.ind[i]);
			rowdata_colid[rowrec].push_back(ri.num[i] - 1 - (rowrec * m_perproc));
		}
		pcnts = new IT[rowneighs];
		for(IT i=0; i<rowneighs; ++i)
			pcnts[i] = rowdata_rowid[i].size();

		// the second parameter, sendcount, is the number of elements sent to *each* processor
		(commGrid->rowWorld).Scatter(pcnts, 1, MPIType<IT>(), &p_nnz, 1, MPIType<IT>(), diaginrow);

		for(IT i=0; i<rowneighs; ++i)
		{
			if(i != diaginrow)	// destination is not me	
			{
				(commGrid->rowWorld).Send(&(rowdata_rowid[i][0]), pcnts[i], MPIType<IT>(), i, RFROWIDS); 
				(commGrid->rowWorld).Send(&(rowdata_colid[i][0]), pcnts[i], MPIType<IT>(), i, RFCOLIDS); 
			}
		}

		IT locvecc = ci.ind.size();	// nnz in local vector
		for(IT i=0; i < locvecc; ++i)
		{	
			// make 1-based indices 0-based
			int colrec = std::min((ci.num[i]-1) / n_perproc, colneighs-1);	// precipient processor along the column

			// ci's numerical values give the rowids and its indices give colids
			// thus, the colid's are already offset'd where rowid's are not
			coldata_rowid[colrec].push_back(ci.num[i] - 1 - (colrec * n_perproc));
			coldata_colid[colrec].push_back(ci.ind[i]);
		}

		qcnts = new IT[colneighs];
		for(IT i=0; i<colneighs; ++i)
			qcnts[i] = coldata_rowid[i].size();

		(commGrid->colWorld).Scatter(qcnts, 1, MPIType<IT>(), &q_nnz, 1, MPIType<IT>(), diagincol);

		for(IT i=0; i<colneighs; ++i)
		{
			if(i != diagincol)	// destination is not me	
			{
				(commGrid->colWorld).Send(&(coldata_rowid[i][0]), qcnts[i], MPIType<IT>(), i, RFROWIDS); 
				(commGrid->colWorld).Send(&(coldata_colid[i][0]), qcnts[i], MPIType<IT>(), i, RFCOLIDS); 
			}
		}
		DeleteAll(pcnts, qcnts);

		tuple<IT,IT,bool> * p_tuples = new tuple<IT,IT,bool>[p_nnz]; 
		for(int i=0; i< p_nnz; ++i)
		{
			p_tuples[i] = make_tuple(rowdata_rowid[diaginrow][i], rowdata_colid[diaginrow][i], 1);
		}

		tuple<IT,IT,bool> * q_tuples = new tuple<IT,IT,bool>[q_nnz]; 
		for(int i=0; i< q_nnz; ++i)
		{
			q_tuples[i] = make_tuple(coldata_rowid[diagincol][i], coldata_colid[diagincol][i], 1);
		}

		PSeq = new DER_IT(); 
		PSeq->Create( p_nnz, rilen, getlocalrows(), p_tuples);		// deletion of tuples[] is handled by SpMat::Create

		QSeq = new DER_IT();  
		QSeq->Create( q_nnz, getlocalcols(), cilen, q_tuples);		// deletion of tuples[] is handled by SpMat::Create
	}
	else	// all others receive data from the diagonal
	{
		(commGrid->rowWorld).Bcast(&rilen, 1, MPIType<IT>(), diaginrow);
		(commGrid->colWorld).Bcast(&cilen, 1, MPIType<IT>(), diagincol);

		// receive the receive counts ...
		(commGrid->rowWorld).Scatter(pcnts, 1, MPIType<IT>(), &p_nnz, 1, MPIType<IT>(), diaginrow);
		(commGrid->colWorld).Scatter(qcnts, 1, MPIType<IT>(), &q_nnz, 1, MPIType<IT>(), diagincol);
		
		// create space for incoming data ... 
		IT * p_rows = new IT[p_nnz];
		IT * p_cols = new IT[p_nnz];
		IT * q_rows = new IT[q_nnz];
		IT * q_cols = new IT[q_nnz];
		
		// receive actual data ... 
		(commGrid->rowWorld).Recv(p_rows, p_nnz, MPIType<IT>(), diaginrow, RFROWIDS);	
		(commGrid->rowWorld).Recv(p_cols, p_nnz, MPIType<IT>(), diaginrow, RFCOLIDS);	
	
		(commGrid->colWorld).Recv(q_rows, q_nnz, MPIType<IT>(), diagincol, RFROWIDS);	
		(commGrid->colWorld).Recv(q_cols, q_nnz, MPIType<IT>(), diagincol, RFCOLIDS);	

		tuple<IT,IT,bool> * p_tuples = new tuple<IT,IT,bool>[p_nnz]; 
		for(int i=0; i< p_nnz; ++i)
		{
			p_tuples[i] = make_tuple(p_rows[i], p_cols[i], 1);
		}

		tuple<IT,IT,bool> * q_tuples = new tuple<IT,IT,bool>[q_nnz]; 
		for(int i=0; i< q_nnz; ++i)
		{
			q_tuples[i] = make_tuple(q_rows[i], q_cols[i], 1);
		}
		DeleteAll(p_rows, p_cols, q_rows, q_cols);

		PSeq = new DER_IT(); 
		PSeq->Create( p_nnz, rilen, getlocalrows(), p_tuples);		// deletion of tuples[] is handled by SpMat::Create

		QSeq = new DER_IT();  
		QSeq->Create( q_nnz, getlocalcols(), cilen, q_tuples);		// deletion of tuples[] is handled by SpMat::Create
	}
	
	// Distributed matrix generation (collective call)
	SpParMat<IT,bool,DER_IT> P (PSeq, commGrid);
	SpParMat<IT,bool,DER_IT> Q (QSeq, commGrid);

	// Do parallel matrix-matrix multiply
	typedef PlusTimesSRing<bool, NT> PTBOOLNT;
	typedef PlusTimesSRing<NT, bool> PTNTBOOL;

        return Mult_AnXBn_Synch<PTNTBOOL>(Mult_AnXBn_Synch<PTBOOLNT>(P, *this), Q);
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
void SpParMat<IT,NT,DER>::EWiseScale(const DenseParMat<IT, NT> & rhs)
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

#ifdef DEBUG
	if ((commGrid->grrows * commGrid->grcols) ==  1)
		spSeq->PrintInfo();
#endif
}

template <class IT, class NT, class DER>
bool SpParMat<IT,NT,DER>::operator== (const SpParMat<IT,NT,DER> & rhs) const
{
	int local = static_cast<int>((*spSeq) == (*(rhs.spSeq)));
	int whole = 1;
	commGrid->GetWorld().Allreduce( &local, &whole, 1, MPI::INT, MPI::BAND);
	return static_cast<bool>(whole);	
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
	shared_ptr<CommGrid> Grid = ProductGrid(commGrid.get(), commGrid.get(), stages, dummy, dummy);		
	
	IT AA_m = spSeq->getnrow();
	IT AA_n = spSeq->getncol();
	
	DER seqTrn = spSeq->TransposeConst();	// will be automatically discarded after going out of scope		

	Grid->GetWorld().Barrier();

	IT ** NRecvSizes = SpHelper::allocate2D<IT>(DER::esscount, stages);
	IT ** TRecvSizes = SpHelper::allocate2D<IT>(DER::esscount, stages);
	
	SpParHelper::GetSetSizes( *spSeq, NRecvSizes, commGrid->GetRowWorld());
	SpParHelper::GetSetSizes( seqTrn, TRecvSizes, commGrid->GetColWorld());

	// Remotely fetched matrices are stored as pointers
	DER * NRecv; 
	DER * TRecv;
	vector< SpTuples<IT,NT>  *> tomerge;

	int Nself = commGrid->GetRankInProcRow();
	int Tself = commGrid->GetRankInProcCol();	

	for(int i = 0; i < stages; ++i) 
	{
		vector<IT> ess;	
		if(i == Nself)
		{	
			NRecv = spSeq;	// shallow-copy 
		}
		else
		{
			ess.resize(DER::esscount);
			for(int j=0; j< DER::esscount; ++j)	
			{
				ess[j] = NRecvSizes[j][i];		// essentials of the ith matrix in this row	
			}
			NRecv = new DER();				// first, create the object
		}

		SpParHelper::BCastMatrix(Grid->GetRowWorld(), *NRecv, ess, i);	// then, broadcast its elements	
		ess.clear();	
		
		if(i == Tself)
		{
			TRecv = &seqTrn;	// shallow-copy
		}
		else
		{
			ess.resize(DER::esscount);		
			for(int j=0; j< DER::esscount; ++j)	
			{
				ess[j] = TRecvSizes[j][i];	
			}	
			TRecv = new DER();
		}
		SpParHelper::BCastMatrix(Grid->GetColWorld(), *TRecv, ess, i);	

		SpTuples<IT,NT> * AA_cont = MultiplyReturnTuples<SR>(*NRecv, *TRecv, false, true);
		if(!AA_cont->isZero()) 
			tomerge.push_back(AA_cont);

		if(i != Nself)	
		{
			delete NRecv;		
		}
		if(i != Tself)	
		{
			delete TRecv;
		}
	}

	SpHelper::deallocate2D(NRecvSizes, DER::esscount);
	SpHelper::deallocate2D(TRecvSizes, DER::esscount);
	
	delete spSeq;		
	spSeq = new DER(MergeAll<SR>(tomerge, AA_m, AA_n), false, NULL);	// First get the result in SpTuples, then convert to UDER
	for(int i=0; i<tomerge.size(); ++i)
	{
		delete tomerge[i];
	}
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

// Prints in the following format suitable for I/O with PaToH
// 1st line: <index_base(0 or 1)> <|V|> <|N|> <pins>
// For row-wise sparse matrix partitioning (our case): 1 <num_rows> <num_cols> <nnz>
// For the rest of the file, (i+1)th line is a list of vertices that are members of the ith net
// Treats the matrix as binary (for now)
template <class IT, class NT, class DER>
void SpParMat< IT,NT,DER >::PrintForPatoh(string filename) const
{
	int proccols = commGrid->GetGridCols();
	int procrows = commGrid->GetGridRows();
	IT * gsizes;

	// collective calls
	IT totalm = getnrow();
	IT totaln = getncol();
	IT totnnz = getnnz();
	int flinelen = 0;
	if(commGrid->GetRank() == 0)
	{
		std::string s;
		std::stringstream strm;
		strm << 0 << " " << totalm << " " << totaln << " " << totnnz << endl;
		s = strm.str();
		std::ofstream out(filename.c_str(),std::ios_base::app);
		flinelen = s.length();
		out.write(s.c_str(), flinelen);
		out.close();
	}

	IT nzc = 0;	// nonempty column counts per processor column
	for(int i = 0; i < proccols; i++)	// for all processor columns (in order)
	{
		if(commGrid->GetRankInProcRow() == i)	// only the ith processor column
		{ 
			if(commGrid->GetRankInProcCol() == 0)	// get the head of column
			{
				std::ofstream out(filename.c_str(),std::ios_base::app);
				std::ostream_iterator<IT> dest(out, " ");

				gsizes = new IT[procrows];
				IT localrows = spSeq->getnrow();    
				(commGrid->GetColWorld()).Bcast(&localrows, 1, MPIType<IT>(), 0); 
				IT netid = 0;
				for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)	// iterate over nonempty subcolumns
				{
					IT mysize;
					IT * vertices;
					while(netid <= colit.colid())
					{
						if(netid < colit.colid())	// empty subcolumns
						{
							mysize = 0;
						}
						else
						{
							mysize = colit.nnz();
							vertices = new IT[mysize];
							IT j = 0;
							for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)
							{
								vertices[j] = nzit.rowid();
								++j;
							}
						}
						(commGrid->GetColWorld()).Gather(&mysize, 1, MPIType<IT>(), gsizes, 1, MPIType<IT>(), 0);
						IT colcnt = std::accumulate(gsizes, gsizes+procrows, 0);
						IT * ents = new IT[colcnt];	// nonzero entries in the netid'th column
						IT * dpls = new IT[procrows]();	// displacements (zero initialized pid) 
						std::partial_sum(gsizes, gsizes+procrows-1, dpls+1);

						// int MPI_Gatherv (void* sbuf, int scount, MPI_Datatype stype, 
						// 		    void* rbuf, int *rcount, int* displs, MPI_Datatype rtype, int root, MPI_Comm comm)	
						(commGrid->GetColWorld()).Gatherv(vertices, mysize, MPIType<IT>(), ents, gsizes, dpls, MPIType<IT>(), 0);
						if(colcnt != 0)
						{
							std::copy(ents, ents+colcnt, dest);	
							out << endl;
							++nzc;
						}
						delete [] ents; delete [] dpls;
						
						if(netid == colit.colid())	delete [] vertices;
						++netid;
					} 
				}
				while(netid < spSeq->getncol())
				{
					IT mysize = 0; 	
					(commGrid->GetColWorld()).Gather(&mysize, 1, MPIType<IT>(), gsizes, 1, MPIType<IT>(), 0);
					IT colcnt = std::accumulate(gsizes, gsizes+procrows, 0);
					IT * ents = new IT[colcnt];	// nonzero entries in the netid'th column
					IT * dpls = new IT[procrows]();	// displacements (zero initialized pid) 
					std::partial_sum(gsizes, gsizes+procrows-1, dpls+1);

					(commGrid->GetColWorld()).Gatherv(NULL, mysize, MPIType<IT>(), ents, gsizes, dpls, MPIType<IT>(), 0);
					if(colcnt != 0)
					{
						std::copy(ents, ents+colcnt, dest);	
						out << endl;
						++nzc;
					}
					delete [] ents; delete [] dpls;
					++netid;
				} 
				delete [] gsizes;
				out.close();
			}
			else	// get the rest of the processors
			{
				IT m_perproc;
				(commGrid->GetColWorld()).Bcast(&m_perproc, 1, MPIType<IT>(), 0); 
				IT moffset = commGrid->GetRankInProcCol() * m_perproc; 
				IT netid = 0;
				for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)	// iterate over columns
				{
					IT mysize;
					IT * vertices;	
					while(netid <= colit.colid())
					{
						if(netid < colit.colid())	// empty subcolumns
						{
							mysize = 0;
						}
						else
						{
							mysize = colit.nnz();
							vertices = new IT[mysize];
							IT j = 0;
							for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)
							{
								vertices[j] = nzit.rowid() + moffset; 
								++j;
							}
						}
						(commGrid->GetColWorld()).Gather(&mysize, 1, MPIType<IT>(), gsizes, 1, MPIType<IT>(), 0);
					
						// rbuf, rcount, displs, rtype are only significant at the root
						(commGrid->GetColWorld()).Gatherv(vertices, mysize, MPIType<IT>(), NULL, NULL, NULL, MPIType<IT>(), 0);
						
						if(netid == colit.colid())	delete [] vertices;
						++netid;
					}
				}
				while(netid < spSeq->getncol())
				{
					IT mysize = 0; 	
					(commGrid->GetColWorld()).Gather(&mysize, 1, MPIType<IT>(), gsizes, 1, MPIType<IT>(), 0);
					(commGrid->GetColWorld()).Gatherv(NULL, mysize, MPIType<IT>(), NULL, NULL, NULL, MPIType<IT>(), 0);
					++netid;
				} 
			}
		} // end_if the ith processor column

		commGrid->GetWorld().Barrier();		// signal the end of ith processor column iteration (so that all processors block)
		if((i == proccols-1) && (commGrid->GetRankInProcCol() == 0))	// if that was the last iteration and we are the column heads
		{
			IT totalnzc = 0;
			(commGrid->GetRowWorld()).Reduce(&nzc, &totalnzc, 1, MPIType<IT>(), MPI::SUM, 0);

			if(commGrid->GetRank() == 0)	// I am the master, hence I'll change the first line
			{
				// Don't just open with std::ios_base::app here
				// The std::ios::app flag causes fstream to call seekp(0, ios::end); at the start of every operator<<() inserter
				std::ofstream out(filename.c_str(),std::ios_base::in | std::ios_base::out | std::ios_base::binary);
				out.seekp(0, std::ios_base::beg);
				
				string s;
				std::stringstream strm;
				strm << 0 << " " << totalm << " " << totalnzc << " " << totnnz;
				s = strm.str();
				s.resize(flinelen-1, ' ');	// in case totalnzc has fewer digits than totaln

				out.write(s.c_str(), flinelen-1);
				out.close();
			}
		}
	} // end_for all processor columns
}

//! Handles all sorts of orderings as long as there are no duplicates
//! May perform better when the data is already reverse column-sorted (i.e. in decreasing order)
//! if nonum is true, then numerics are not supplied and they are assumed to be all 1's
template <class IT, class NT, class DER>
ifstream& SpParMat< IT,NT,DER >::ReadDistribute (ifstream& infile, int master, bool nonum)
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
			char comment[256];
			infile.getline(comment,256);
			while(comment[0] == '%')
			{
				infile.getline(comment,256);
			}
			stringstream ss;
			ss << string(comment);
			ss >> total_m >> total_n >> total_nnz;
			m_perproc = total_m / colneighs;
			n_perproc = total_n / rowneighs;

			(commGrid->commWorld).Bcast(&total_m, 1, MPIType<IT>(), master);
			(commGrid->commWorld).Bcast(&total_n, 1, MPIType<IT>(), master);

			IT temprow, tempcol;
			NT tempval;
			IT cnz = 0;
			while ( (!infile.eof()) && cnz < total_nnz)
			{
				infile >> temprow >> tempcol;
				if(nonum)	tempval = static_cast<NT>(1);
				else 		infile >> tempval;

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
		else	// input file does not exist !
		{
			total_n = 0; total_m = 0;	
			(commGrid->commWorld).Bcast(&total_m, 1, MPIType<IT>(), master);
			(commGrid->commWorld).Bcast(&total_n, 1, MPIType<IT>(), master);								
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

		while(total_n > 0 || total_m > 0)	// otherwise input file does not exist !
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
		
		while(total_n > 0 || total_m > 0)	// otherwise input file does not exist !
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
	tuple<IT,IT,NT> * arrtuples = new tuple<IT,IT,NT>[localtuples.size()];  // the vector will go out of scope, make it stick !
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
