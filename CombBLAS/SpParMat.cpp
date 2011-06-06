/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.1 -------------------------------------------------*/
/* date: 12/25/2010 --------------------------------------------*/
/* authors: Aydin Buluc (abuluc@lbl.gov), Adam Lugowski --------*/
/****************************************************************/

#include "SpParMat.h"
#include "ParFriends.h"
#include "Operations.h"

#include <fstream>
#include <algorithm>
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
void SpParMat< IT,NT,DER >::Dump(string filename) const
{
	MPI::Intracomm World = commGrid->GetWorld();
    	int rank = World.Get_rank();
    	int nprocs = World.Get_size();
	try 
	{
    		MPI::File thefile = MPI::File::Open(World, filename.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI::INFO_NULL);    

		int rankinrow = commGrid->GetRankInProcRow();
		int rankincol = commGrid->GetRankInProcCol();
		int rowneighs = commGrid->GetGridCols();	// get # of processors on the row
		int colneighs = commGrid->GetGridRows();

		IT * colcnts = new IT[rowneighs];
		IT * rowcnts = new IT[colneighs];
		rowcnts[rankincol] = getlocalrows();
		colcnts[rankinrow] = getlocalcols();

		commGrid->GetRowWorld().Allgather(MPI::IN_PLACE, 0, MPIType<IT>(), colcnts, 1, MPIType<IT>());
		IT coloffset = accumulate(colcnts, colcnts+rankinrow, 0);
	
		commGrid->GetColWorld().Allgather(MPI::IN_PLACE, 0, MPIType<IT>(), rowcnts, 1, MPIType<IT>());
		IT rowoffset = accumulate(rowcnts, rowcnts+rankincol, 0);
		DeleteAll(colcnts, rowcnts);

		IT * prelens = new IT[nprocs];
		prelens[rank] = 2*getlocalnnz();
		commGrid->GetWorld().Allgather(MPI::IN_PLACE, 0, MPIType<IT>(), prelens, 1, MPIType<IT>());
		IT lengthuntil = accumulate(prelens, prelens+rank, static_cast<IT>(0));

		// The disp displacement argument specifies the position 
		// (absolute offset in bytes from the beginning of the file) 
		MPI::Offset disp = lengthuntil * sizeof(uint32_t);
		cout << "Displacement: " << disp << endl;
    		thefile.Set_view(disp, MPI::UNSIGNED, MPI::UNSIGNED, "native", MPI::INFO_NULL);
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
	
		thefile.Write(gen_edges, prelens[rank], MPI::UNSIGNED);
		thefile.Close();

		delete [] prelens;
		delete [] gen_edges;
	} 
    	catch(MPI::Exception e)
	{
		cerr << "Exception while dumping file" << endl;
       		cerr << e.Get_error_string() << endl;
	}
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
float SpParMat< IT,NT,DER >::LoadImbalance() const
{
	IT totnnz = getnnz();	// collective call
	IT maxnnz = 0;    
	IT localnnz = spSeq->getnnz();
	(commGrid->GetWorld()).Allreduce( &localnnz, &maxnnz, 1, MPIType<IT>(), MPI::MAX);
	if(totnnz == 0) return 1;
 	return static_cast<float>(((commGrid->GetWorld()).Get_size() * maxnnz)) / static_cast<float>(totnnz);  
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
template <class IT, class NT, class DER>
template <typename _BinaryOperation>	
void SpParMat<IT,NT,DER>::DimApply(Dim dim, const FullyDistVec<IT, NT>& x, _BinaryOperation __binary_op)
{

	if(!(*commGrid == *(x.commGrid))) 		
	{
		cout << "Grids are not comparable for SpParMat::DimApply" << endl; 
		MPI::COMM_WORLD.Abort(GRIDMISMATCH);
	}

	MPI::Intracomm World = x.commGrid->GetWorld();
	MPI::Intracomm ColWorld = x.commGrid->GetColWorld();
	MPI::Intracomm RowWorld = x.commGrid->GetRowWorld();
	switch(dim)
	{
		case Column:	// scale each column
		{
			int xsize = (int) x.LocArrSize();
			int trxsize = 0;
			int diagneigh = x.commGrid->GetComplementRank();
			World.Sendrecv(&xsize, 1, MPI::INT, diagneigh, TRX, &trxsize, 1, MPI::INT, diagneigh, TRX);
	
			NT * trxnums = new NT[trxsize];
			World.Sendrecv(const_cast<NT*>(&x.arr[0]), xsize, MPIType<NT>(), diagneigh, TRX, trxnums, trxsize, MPIType<NT>(), diagneigh, TRX);

			int colneighs = ColWorld.Get_size();
			int colrank = ColWorld.Get_rank();
			int * colsize = new int[colneighs];
			colsize[colrank] = trxsize;
			ColWorld.Allgather(MPI::IN_PLACE, 1, MPI::INT, colsize, 1, MPI::INT);
			int * dpls = new int[colneighs]();	// displacements (zero initialized pid) 
			std::partial_sum(colsize, colsize+colneighs-1, dpls+1);
			int accsize = std::accumulate(colsize, colsize+colneighs, 0);
			NT * scaler = new NT[accsize];

			ColWorld.Allgatherv(trxnums, trxsize, MPIType<NT>(), scaler, colsize, dpls, MPIType<NT>());
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
			int rowneighs = RowWorld.Get_size();
			int rowrank = RowWorld.Get_rank();
			int * rowsize = new int[rowneighs];
			rowsize[rowrank] = xsize;
			RowWorld.Allgather(MPI::IN_PLACE, 1, MPI::INT, rowsize, 1, MPI::INT);
			int * dpls = new int[rowneighs]();	// displacements (zero initialized pid) 
			std::partial_sum(rowsize, rowsize+rowneighs-1, dpls+1);
			int accsize = std::accumulate(rowsize, rowsize+rowneighs, 0);
			NT * scaler = new NT[accsize];

			RowWorld.Allgatherv(const_cast<NT*>(&x.arr[0]), xsize, MPIType<NT>(), scaler, rowsize, dpls, MPIType<NT>());
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
			cout << "Unknown scaling dimension, returning..." << endl;
			break;
		}
	}
}

template <class IT, class NT, class DER>
template <typename _BinaryOperation, typename _UnaryOperation >	
DenseParVec<IT,NT> SpParMat<IT,NT,DER>::Reduce(Dim dim, _BinaryOperation __binary_op, NT id, _UnaryOperation __unary_op) const
{
	DenseParVec<IT,NT> parvec(commGrid, id);
	Reduce(parvec, dim, __binary_op, id, __unary_op);			
	return parvec;
}

// default template arguments don't work with function templates
template <class IT, class NT, class DER>
template <typename _BinaryOperation>	
DenseParVec<IT,NT> SpParMat<IT,NT,DER>::Reduce(Dim dim, _BinaryOperation __binary_op, NT id) const
{
	DenseParVec<IT,NT> parvec(commGrid, id);
	Reduce(parvec, dim, __binary_op, id, myidentity<NT>() );			
	return parvec;
}

template <class IT, class NT, class DER>
template <typename VT, typename _BinaryOperation>	
void SpParMat<IT,NT,DER>::Reduce(DenseParVec<IT,VT> & rvec, Dim dim, _BinaryOperation __binary_op, VT id) const
{
	Reduce(rvec, dim, __binary_op, id, myidentity<NT>() );			
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
	if(rvec.zero != id)
	{
		ostringstream outs;
		outs << "SpParMat::Reduce(): Return vector's zero is different than set id"  << endl;
		outs << "Setting rvec.zero to id (" << id << ") instead" << endl;
		SpParHelper::Print(outs.str());
		rvec.zero = id;
	}
	if(*rvec.commGrid != *commGrid)
	{
		SpParHelper::Print("Grids are not comparable, SpParMat::Reduce() fails !"); 
		MPI::COMM_WORLD.Abort(GRIDMISMATCH);
	}
	switch(dim)
	{
		case Column:	// pack along the columns, result is a vector of size n
		{
			// We can't use rvec's distribution (rows first, columns later) here
        		IT n_thiscol = getlocalcols();   // length assigned to this processor column
			int colneighs = commGrid->GetGridRows();	// including oneself
        		int colrank = commGrid->GetRankInProcCol();

			IT * loclens = new IT[colneighs];
			IT * lensums = new IT[colneighs+1]();	// begin/end points of local lengths

        		IT n_perproc = n_thiscol / colneighs;    // length on a typical processor
        		if(colrank == colneighs-1)
                		loclens[colrank] = n_thiscol - (n_perproc*colrank);
        		else
                		loclens[colrank] = n_perproc;

			commGrid->GetColWorld().Allgather(MPI::IN_PLACE, 0, MPIType<IT>(), loclens, 1, MPIType<IT>());
			partial_sum(loclens, loclens+colneighs, lensums+1);	// loclens and lensums are different, but both would fit in 32-bits

			vector<VT> trarr;
			typename DER::SpColIter colit = spSeq->begcol();
			for(int i=0; i< colneighs; ++i)
			{
				VT * sendbuf = new VT[loclens[i]];
				fill(sendbuf, sendbuf+loclens[i], id);	// fill with identity

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
					recvbuf = &trarr[0];	
				}
				(commGrid->GetColWorld()).Reduce(sendbuf, recvbuf, loclens[i], MPIType<VT>(), MPIOp<_BinaryOperation, VT>::op(), i);	// root  = i
				delete [] sendbuf;
			}
			DeleteAll(loclens, lensums);

			IT reallen;	// Now we have to transpose the vector
			IT trlen = trarr.size();
			int diagneigh = commGrid->GetComplementRank();
			(commGrid->GetWorld()).Sendrecv(&trlen, 1, MPIType<IT>(), diagneigh, TRNNZ, &reallen, 1, MPIType<IT>(), diagneigh, TRNNZ);
	
			rvec.arr.resize(reallen);
			(commGrid->GetWorld()).Sendrecv(&trarr[0], trlen, MPIType<VT>(), diagneigh, TRX, &rvec.arr[0], reallen, MPIType<VT>(), diagneigh, TRX);
			rvec.glen = getncol();	// ABAB: Put a sanity check here
			break;

		}
		case Row:	// pack along the rows, result is a vector of size m
		{
			rvec.glen = getnrow();
			int rowneighs = commGrid->GetGridCols();
			int rowrank = commGrid->GetRankInProcRow();
			IT * loclens = new IT[rowneighs];
			IT * lensums = new IT[rowneighs+1]();	// begin/end points of local lengths
			loclens[rowrank] = rvec.MyLocLength();
			commGrid->GetRowWorld().Allgather(MPI::IN_PLACE, 0, MPIType<IT>(), loclens, 1, MPIType<IT>());
			partial_sum(loclens, loclens+rowneighs, lensums+1);

			vector<typename DER::SpColIter::NzIter> nziters;	// keep track of nonzero iterators within columns
			for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)	
			{
				nziters.push_back(spSeq->begnz(colit));
			}

			for(int i=0; i< rowneighs; ++i)		// step by step to save memory
			{
				VT * sendbuf = new VT[loclens[i]];
				fill(sendbuf, sendbuf+loclens[i], id);	// fill with identity
		
				typename DER::SpColIter colit = spSeq->begcol();		
				IT colcnt = 0;	// column counter
				for(; colit != spSeq->endcol(); ++colit, ++colcnt)	// iterate over all columns
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
					rvec.arr.resize(loclens[i]);
					recvbuf = &rvec.arr[0];	
				}
				(commGrid->GetRowWorld()).Reduce(sendbuf, recvbuf, loclens[i], MPIType<VT>(), MPIOp<_BinaryOperation, VT>::op(), i);	// root = i
				delete [] sendbuf;
			}
			DeleteAll(loclens, lensums);	
			break;
		}
		default:
		{
			cout << "Unknown reduction dimension, returning empty vector" << endl;
			break;
		}
	}
}

/**
  * Reduce along the column/row into a vector
  * @param[in] __binary_op {the operation used for reduction; examples: max, min, plus, multiply, and, or. Its parameters and return type are all VT}
  * @param[in] id {scalar that is used as the identity for __binary_op; examples: zero, infinity}
  * @param[in] __unary_op {optional unary operation applied to nonzeros *before* the __binary_op; examples: 1/x, x^2}
  * @param[out] rvec {the return vector, specified as an output parameter to allow arbitrary return types via VT}
 **/ 
template <class IT, class NT, class DER>
template <typename VT, typename _BinaryOperation, typename _UnaryOperation>	
void SpParMat<IT,NT,DER>::Reduce(DenseParVec<IT,VT> & rvec, Dim dim, _BinaryOperation __binary_op, VT id, _UnaryOperation __unary_op) const
{
	if(rvec.zero != id)
	{
		ostringstream outs;
		outs << "SpParMat::Reduce(): Return vector's zero is different than set id"  << endl;
		outs << "Setting rvec.zero to id (" << id << ") instead" << endl;
		SpParHelper::Print(outs.str());
		rvec.zero = id;
	}
	if(*rvec.commGrid != *commGrid)
	{
		SpParHelper::Print("Grids are not comparable, SpParMat::Reduce() fails !"); 
		MPI::COMM_WORLD.Abort(GRIDMISMATCH);
	}
	switch(dim)
	{
		case Column:	// pack along the columns, result is a vector of size n
		{
			VT * sendbuf = new VT[getlocalcols()];
			fill(sendbuf, sendbuf+getlocalcols(), id);	// fill with identity

			for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)	// iterate over columns
			{
				for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)
				{
					sendbuf[colit.colid()] = __binary_op(static_cast<VT>(__unary_op(nzit.value())), sendbuf[colit.colid()]);
				}
			}
			VT * recvbuf = NULL;
			int root = commGrid->GetDiagOfProcCol();
			if(rvec.diagonal)
			{
				rvec.arr.resize(getlocalcols());
				recvbuf = &rvec.arr[0];	
			}
			(commGrid->GetColWorld()).Reduce(sendbuf, recvbuf, getlocalcols(), MPIType<VT>(), MPIOp<_BinaryOperation, VT>::op(), root);
			delete [] sendbuf;
			break;
		}
		case Row:	// pack along the rows, result is a vector of size m
		{
			VT * sendbuf = new VT[getlocalrows()];
			fill(sendbuf, sendbuf+getlocalrows(), id);	// fill with identity
			
			for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)	// iterate over columns
			{
				for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)
				{
					sendbuf[nzit.rowid()] = __binary_op(static_cast<VT>(__unary_op(nzit.value())), sendbuf[nzit.rowid()]);
				}
			}
			VT * recvbuf = NULL;
			int root = commGrid->GetDiagOfProcRow();
			if(rvec.diagonal)
			{
				rvec.arr.resize(getlocalrows());
				recvbuf = &(rvec.arr[0]);	
			}
			(commGrid->GetRowWorld()).Reduce(sendbuf, recvbuf, getlocalrows(), MPIType<VT>(), MPIOp<_BinaryOperation, VT>::op(), root);
			delete [] sendbuf;
			break;
		}
		default:
		{
			cout << "Unknown reduction dimension, returning empty vector" << endl;
			break;
		}
	}
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
SpParMat<IT,NT,DER> SpParMat<IT,NT,DER>::SubsRefCol (const vector<IT> & ci) const
{
	vector<IT> ri;
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
SpParMat<IT,NT,DER> SpParMat<IT,NT,DER>::operator() (const FullyDistVec<IT,IT> & ri, const FullyDistVec<IT,IT> & ci, bool inplace)
{
	// infer the concrete type SpMat<IT,IT>
	typedef typename create_trait<DER, IT, bool>::T_inferred DER_IT;
	typedef PlusTimesSRing<NT, bool> PTNTBOOL;
	typedef PlusTimesSRing<bool, NT> PTBOOLNT;

	if((*(ri.commGrid) != *(commGrid)) || (*(ci.commGrid) != *(commGrid)))
	{
		SpParHelper::Print("Grids are not comparable, SpRef fails !"); 
		MPI::COMM_WORLD.Abort(GRIDMISMATCH);
	}

	// Safety check
	IT locmax_ri = 0;
	IT locmax_ci = 0;
	if(!ri.arr.empty())
		locmax_ri = *max_element(ri.arr.begin(), ri.arr.end());
	if(!ci.arr.empty())
		locmax_ci = *max_element(ci.arr.begin(), ci.arr.end());

	IT total_m = getnrow();
	IT total_n = getncol();
	if(locmax_ri > total_m || locmax_ci > total_n)	
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
	IT totalm = getnrow();	// collective call
	IT totaln = getncol();
	IT m_perproccol = totalm / rowneighs;
	IT n_perproccol = totaln / rowneighs;

	// Get the right local dimensions
	IT diagneigh = commGrid->GetComplementRank();
	IT mylocalrows = getlocalrows();
	IT mylocalcols = getlocalcols();
	IT trlocalrows;
	commGrid->GetWorld().Sendrecv(&mylocalrows, 1, MPIType<IT>(), diagneigh, TRROWX, &trlocalrows, 1, MPIType<IT>(), diagneigh, TRROWX);
	// we don't need trlocalcols because Q.Transpose() will take care of it

	vector< vector<IT> > rowid(rowneighs);	// reuse for P and Q 
	vector< vector<IT> > colid(rowneighs);

	// Step 1: Create P
	IT locvec = ri.arr.size();	// nnz in local vector
	for(typename vector<IT>::size_type i=0; i< locvec; ++i)
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

	commGrid->GetRowWorld().Alltoall(sendcnt, 1, MPI::INT, recvcnt, 1, MPI::INT); // share the counts 
	int * sdispls = new int[rowneighs]();
	int * rdispls = new int[rowneighs]();
	partial_sum(sendcnt, sendcnt+rowneighs-1, sdispls+1);
	partial_sum(recvcnt, recvcnt+rowneighs-1, rdispls+1);
	IT p_nnz = accumulate(recvcnt,recvcnt+rowneighs, static_cast<IT>(0));	

	// create space for incoming data ... 
	IT * p_rows = new IT[p_nnz];
	IT * p_cols = new IT[p_nnz];
  	IT * senddata = new IT[locvec];	// re-used for both rows and columns
	for(int i=0; i<rowneighs; ++i)
	{
		copy(rowid[i].begin(), rowid[i].end(), senddata+sdispls[i]);
		vector<IT>().swap(rowid[i]);	// clear memory of rowid
	}
	commGrid->GetRowWorld().Alltoallv(senddata, sendcnt, sdispls, MPIType<IT>(), p_rows, recvcnt, rdispls, MPIType<IT>()); 

	for(int i=0; i<rowneighs; ++i)
	{
		copy(colid[i].begin(), colid[i].end(), senddata+sdispls[i]);
		vector<IT>().swap(colid[i]);	// clear memory of colid
	}
	commGrid->GetRowWorld().Alltoallv(senddata, sendcnt, sdispls, MPIType<IT>(), p_cols, recvcnt, rdispls, MPIType<IT>()); 
	delete [] senddata;

	tuple<IT,IT,bool> * p_tuples = new tuple<IT,IT,bool>[p_nnz]; 
	for(IT i=0; i< p_nnz; ++i)
	{
		p_tuples[i] = make_tuple(p_rows[i], p_cols[i], 1);
	}
	DeleteAll(p_rows, p_cols);

	DER_IT * PSeq = new DER_IT(); 
	PSeq->Create( p_nnz, rrowlen, trlocalrows, p_tuples);		// deletion of tuples[] is handled by SpMat::Create

	SpParMat<IT,NT,DER> PA;
	if(&ri == &ci)	// Symmetric permutation
	{
		DeleteAll(sendcnt, recvcnt, sdispls, rdispls);
		#ifdef DEBUG
		SpParHelper::Print("Symmetric permutation\n");
		#endif
		SpParMat<IT,bool,DER_IT> P (PSeq, commGrid);
		if(inplace) 
		{
			#ifdef DEBUG	
			SpParHelper::Print("In place multiplication\n");
			#endif
        		*this = Mult_AnXBn_DoubleBuff<PTBOOLNT>(P, *this, false, true);	// clear the memory of *this

			//ostringstream outb;
			//outb << "P_after_" << commGrid->myrank;
			//ofstream ofb(outb.str().c_str());
			//P.put(ofb);

			P.Transpose();	
       	 		*this = Mult_AnXBn_DoubleBuff<PTNTBOOL>(*this, P, true, true);	// clear the memory of both *this and P
			return SpParMat<IT,NT,DER>();	// dummy return to match signature
		}
		else
		{
			PA = Mult_AnXBn_DoubleBuff<PTBOOLNT>(P,*this);
			P.Transpose();
			return Mult_AnXBn_DoubleBuff<PTNTBOOL>(PA, P);
		}
	}
	else
	{
		// Intermediate step (to save memory): Form PA and store it in P
		// Distributed matrix generation (collective call)
		SpParMat<IT,bool,DER_IT> P (PSeq, commGrid);

		// Do parallel matrix-matrix multiply
        	PA = Mult_AnXBn_DoubleBuff<PTBOOLNT>(P, *this);
	}	// P is destructed here
#ifndef NDEBUG
	PA.PrintInfo();
#endif
	// Step 2: Create Q  (use the same row-wise communication and transpose at the end)
	// This temporary to-be-transposed Q is size(ci) x n 
	locvec = ci.arr.size();	// nnz in local vector (reset variable)
	for(typename vector<IT>::size_type i=0; i< locvec; ++i)
	{
		// numerical values (permutation indices) are 0-based
		IT rowrec = (n_perproccol!=0) ? std::min(ci.arr[i] / n_perproccol, rowneighs-1) : (rowneighs-1);	

		// ri's numerical values give the colids and its local indices give rowids
		rowid[rowrec].push_back( i + coffset);	
		colid[rowrec].push_back(ci.arr[i] - (rowrec * n_perproccol));
	}

	for(IT i=0; i<rowneighs; ++i)
		sendcnt[i] = rowid[i].size();	// update with new sizes

	commGrid->GetRowWorld().Alltoall(sendcnt, 1, MPI::INT, recvcnt, 1, MPI::INT); // share the counts 
	fill(sdispls, sdispls+rowneighs, 0);	// reset
	fill(rdispls, rdispls+rowneighs, 0);
	partial_sum(sendcnt, sendcnt+rowneighs-1, sdispls+1);
	partial_sum(recvcnt, recvcnt+rowneighs-1, rdispls+1);
	IT q_nnz = accumulate(recvcnt,recvcnt+rowneighs, static_cast<IT>(0));	

	// create space for incoming data ... 
	IT * q_rows = new IT[q_nnz];
	IT * q_cols = new IT[q_nnz];
  	senddata = new IT[locvec];	
	for(int i=0; i<rowneighs; ++i)
	{
		copy(rowid[i].begin(), rowid[i].end(), senddata+sdispls[i]);
		vector<IT>().swap(rowid[i]);	// clear memory of rowid
	}
	commGrid->GetRowWorld().Alltoallv(senddata, sendcnt, sdispls, MPIType<IT>(), q_rows, recvcnt, rdispls, MPIType<IT>()); 

	for(int i=0; i<rowneighs; ++i)
	{
		copy(colid[i].begin(), colid[i].end(), senddata+sdispls[i]);
		vector<IT>().swap(colid[i]);	// clear memory of colid
	}
	commGrid->GetRowWorld().Alltoallv(senddata, sendcnt, sdispls, MPIType<IT>(), q_cols, recvcnt, rdispls, MPIType<IT>()); 
	DeleteAll(senddata, sendcnt, recvcnt, sdispls, rdispls);

	tuple<IT,IT,bool> * q_tuples = new tuple<IT,IT,bool>[q_nnz]; 
	for(IT i=0; i< q_nnz; ++i)
	{
		q_tuples[i] = make_tuple(q_rows[i], q_cols[i], 1);
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
       		*this = Mult_AnXBn_DoubleBuff<PTNTBOOL>(PA, Q, true, true);	// clear the memory of both PA and P
		return SpParMat<IT,NT,DER>();	// dummy return to match signature
	}
	else
	{
        	return Mult_AnXBn_DoubleBuff<PTNTBOOL>(PA, Q);
	}
}

/** 
 * Generalized sparse matrix indexing (ri/ci are 0-based indexed)
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

	IT colneighs = commGrid->GetGridRows();	// number of neighbors along this processor column (including oneself)
	IT rowneighs = commGrid->GetGridCols();	// number of neighbors along this processor row (including oneself)
	IT totalm = getnrow();	// collective call
	IT totaln = getncol();
	IT m_perproc = totalm / rowneighs;	// these are CORRECT, as P's column dimension is m
	IT n_perproc = totaln / colneighs;	// and Q's row dimension is n
	IT p_nnz, q_nnz, rilen, cilen; 
	IT * pcnts;
	IT * qcnts;
	
	IT diaginrow = commGrid->GetDiagOfProcRow();
	IT diagincol = commGrid->GetDiagOfProcCol();

	// infer the concrete type SpMat<IT,IT>
	typedef typename create_trait<DER, IT, NT>::T_inferred DER_IT;
	DER_IT * PSeq;
	DER_IT * QSeq;

	IT diagneigh = commGrid->GetComplementRank();
	IT mylocalrows = getlocalrows();
	IT mylocalcols = getlocalcols();
	IT trlocalrows, trlocalcols;
	commGrid->GetWorld().Sendrecv(&mylocalrows, 1, MPIType<IT>(), diagneigh, TRROWX, &trlocalrows, 1, MPIType<IT>(), diagneigh, TRROWX);
	commGrid->GetWorld().Sendrecv(&mylocalcols, 1, MPIType<IT>(), diagneigh, TRCOLX, &trlocalcols, 1, MPIType<IT>(), diagneigh, TRCOLX);

	if(ri.diagonal)		// only the diagonal processors hold vectors
	{
		// broadcast the size 
		rilen = ri.ind.size();
		cilen = ci.ind.size();
		(commGrid->rowWorld).Bcast(&rilen, 1, MPIType<IT>(), diaginrow);
		(commGrid->colWorld).Bcast(&cilen, 1, MPIType<IT>(), diagincol);

		vector< vector<IT> > rowdata_rowid(rowneighs);
		vector< vector<IT> > rowdata_colid(rowneighs);
		vector< vector<IT> > coldata_rowid(colneighs);
		vector< vector<IT> > coldata_colid(colneighs);

		IT locvecr = ri.ind.size();	// nnz in local vector
		for(IT i=0; i < locvecr; ++i)
		{	
			// numerical values (permutation indices) are 0-based
			IT rowrec = std::min(ri.num[i] / m_perproc, rowneighs-1);	// recipient processor along the column

			// ri's numerical values give the colids and its indices give rowids
			// thus, the rowid's are already offset'd where colid's are not
			rowdata_rowid[rowrec].push_back(ri.ind[i]);
			rowdata_colid[rowrec].push_back(ri.num[i] - (rowrec * m_perproc));
		}
		pcnts = new IT[rowneighs];
		for(IT i=0; i<rowneighs; ++i)
			pcnts[i] = rowdata_rowid[i].size();

		// Now, do it for ci
		IT locvecc = ci.ind.size();	
		for(IT i=0; i < locvecc; ++i)
		{	
			// numerical values (permutation indices) are 0-based
			IT colrec = std::min(ci.num[i] / n_perproc, colneighs-1);	// recipient processor along the column

			// ci's numerical values give the rowids and its indices give colids
			// thus, the colid's are already offset'd where rowid's are not
			coldata_rowid[colrec].push_back(ci.num[i] - (colrec * n_perproc));
			coldata_colid[colrec].push_back(ci.ind[i]);
		}

		qcnts = new IT[colneighs];
		for(IT i=0; i<colneighs; ++i)
			qcnts[i] = coldata_rowid[i].size();

		// the second parameter, sendcount, is the number of elements sent to *each* processor
		(commGrid->rowWorld).Scatter(pcnts, 1, MPIType<IT>(), &p_nnz, 1, MPIType<IT>(), diaginrow);
		(commGrid->colWorld).Scatter(qcnts, 1, MPIType<IT>(), &q_nnz, 1, MPIType<IT>(), diagincol);

		for(IT i=0; i<rowneighs; ++i)
		{
			if(i != diaginrow)	// destination is not me	
			{
				(commGrid->rowWorld).Send(&(rowdata_rowid[i][0]), pcnts[i], MPIType<IT>(), i, RFROWIDS); 
				(commGrid->rowWorld).Send(&(rowdata_colid[i][0]), pcnts[i], MPIType<IT>(), i, RFCOLIDS); 
			}
		}

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
		for(IT i=0; i< p_nnz; ++i)
		{
			p_tuples[i] = make_tuple(rowdata_rowid[diaginrow][i], rowdata_colid[diaginrow][i], 1);
		}

		tuple<IT,IT,bool> * q_tuples = new tuple<IT,IT,bool>[q_nnz]; 
		for(IT i=0; i< q_nnz; ++i)
		{
			q_tuples[i] = make_tuple(coldata_rowid[diagincol][i], coldata_colid[diagincol][i], 1);
		}

		PSeq = new DER_IT(); 
		PSeq->Create( p_nnz, rilen, trlocalrows, p_tuples);		// deletion of tuples[] is handled by SpMat::Create

		QSeq = new DER_IT();  
		QSeq->Create( q_nnz, trlocalcols, cilen, q_tuples);		// deletion of tuples[] is handled by SpMat::Create

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
		for(IT i=0; i< p_nnz; ++i)
		{
			p_tuples[i] = make_tuple(p_rows[i], p_cols[i], 1);
		}

		tuple<IT,IT,bool> * q_tuples = new tuple<IT,IT,bool>[q_nnz]; 
		for(IT i=0; i< q_nnz; ++i)
		{
			q_tuples[i] = make_tuple(q_rows[i], q_cols[i], 1);
		}
		DeleteAll(p_rows, p_cols, q_rows, q_cols);

		PSeq = new DER_IT(); 
		PSeq->Create( p_nnz, rilen, trlocalrows, p_tuples);		// deletion of tuples[] is handled by SpMat::Create

		QSeq = new DER_IT();  
		QSeq->Create( q_nnz, trlocalcols, cilen, q_tuples);		// deletion of tuples[] is handled by SpMat::Create
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
		MPI::COMM_WORLD.Abort(GRIDMISMATCH);
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
			MPI::COMM_WORLD.Abort(GRIDMISMATCH);
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
	IT allprocs = commGrid->grrows * commGrid->grcols;
	for(IT i=0; i< allprocs; ++i)
	{
		if (commGrid->myrank == i)
		{
			cout << "Processor (" << commGrid->GetRankInProcRow() << "," << commGrid->GetRankInProcCol() << ")'s data: " << endl;
			spSeq->PrintInfo();
		}
		commGrid->GetWorld().Barrier();
	}
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


template <class IT, class NT, class DER>
SpParMat< IT,NT,DER >::SpParMat (IT total_m, IT total_n, const FullyDistVec<IT,IT> & distrows, 
				const FullyDistVec<IT,IT> & distcols, const FullyDistVec<IT,NT> & distvals)
{
	if((*(distrows.commGrid) != *(distcols.commGrid)) || (*(distcols.commGrid) != *(distvals.commGrid)))
	{
		SpParHelper::Print("Grids are not comparable, Sparse() fails !"); 
		MPI::COMM_WORLD.Abort(GRIDMISMATCH);
	}
	if((distrows.TotalLength() != distcols.TotalLength()) || (distcols.TotalLength() != distvals.TotalLength()))
	{
		SpParHelper::Print("Vectors have different sizes, Sparse() fails !");
		MPI::COMM_WORLD.Abort(DIMMISMATCH);
	}

	commGrid.reset(new CommGrid(*(distrows.commGrid)));		
	int nprocs = commGrid->GetSize();
	vector< vector < tuple<IT,IT,NT> > > data(nprocs);

	IT locsize = distrows.LocArrSize();
	for(IT i=0; i<locsize; ++i)
	{
		IT lrow, lcol; 
		int owner = Owner(total_m, total_n, distrows.arr[i], distcols.arr[i], lrow, lcol);
		data[owner].push_back(make_tuple(lrow,lcol,distvals.arr[i]));	
	}

	int * sendcnt = new int[nprocs];
	int * recvcnt = new int[nprocs];
	for(int i=0; i<nprocs; ++i)
		sendcnt[i] = data[i].size();	// sizes are all the same

	commGrid->GetWorld().Alltoall(sendcnt, 1, MPI::INT, recvcnt, 1, MPI::INT); // share the counts 
	int * sdispls = new int[nprocs]();
	int * rdispls = new int[nprocs]();
	partial_sum(sendcnt, sendcnt+nprocs-1, sdispls+1);
	partial_sum(recvcnt, recvcnt+nprocs-1, rdispls+1);

  	tuple<IT,IT,NT> * senddata = new tuple<IT,IT,NT>[locsize];	// re-used for both rows and columns
	for(int i=0; i<nprocs; ++i)
	{
		copy(data[i].begin(), data[i].end(), senddata+sdispls[i]);
		vector< tuple<IT,IT,NT> >().swap(data[i]);	// clear memory
	}
	MPI::Datatype MPI_triple = MPI::CHAR.Create_contiguous(sizeof(tuple<IT,IT,NT>));
	MPI_triple.Commit();

	IT totrecv = accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));	
	tuple<IT,IT,NT> * recvdata = new tuple<IT,IT,NT>[totrecv];	
	commGrid->GetWorld().Alltoallv(senddata, sendcnt, sdispls, MPI_triple, recvdata, recvcnt, rdispls, MPI_triple); 
	DeleteAll(senddata, sendcnt, recvcnt, sdispls, rdispls);

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

	SpTuples<IT,NT> A(totrecv, locrows, loccols, recvdata);	// It is ~SpTuples's job to deallocate
  	spSeq = new DER(A,false);        // Convert SpTuples to DER
}

template <class IT, class NT, class DER>
SpParMat< IT,NT,DER >::SpParMat (IT total_m, IT total_n, const FullyDistVec<IT,IT> & distrows, 
				const FullyDistVec<IT,IT> & distcols, const NT & val)
{
	if((*(distrows.commGrid) != *(distcols.commGrid)) )
	{
		SpParHelper::Print("Grids are not comparable, Sparse() fails !"); 
		MPI::COMM_WORLD.Abort(GRIDMISMATCH);
	}
	if((distrows.TotalLength() != distcols.TotalLength()) )
	{
		SpParHelper::Print("Vectors have different sizes, Sparse() fails !");
		MPI::COMM_WORLD.Abort(DIMMISMATCH);
	}

	commGrid.reset(new CommGrid(*(distrows.commGrid)));		
	int nprocs = commGrid->GetSize();
	vector< vector < tuple<IT,IT,NT> > > data(nprocs);

	IT locsize = distrows.LocArrSize();
	for(IT i=0; i<locsize; ++i)
	{
		IT lrow, lcol; 
		int owner = Owner(total_m, total_n, distrows.arr[i], distcols.arr[i], lrow, lcol);
		data[owner].push_back(make_tuple(lrow,lcol,val));	
	}

	int * sendcnt = new int[nprocs];
	int * recvcnt = new int[nprocs];
	for(int i=0; i<nprocs; ++i)
		sendcnt[i] = data[i].size();	// sizes are all the same

	commGrid->GetWorld().Alltoall(sendcnt, 1, MPI::INT, recvcnt, 1, MPI::INT); // share the counts 
	int * sdispls = new int[nprocs]();
	int * rdispls = new int[nprocs]();
	partial_sum(sendcnt, sendcnt+nprocs-1, sdispls+1);
	partial_sum(recvcnt, recvcnt+nprocs-1, rdispls+1);

  	tuple<IT,IT,NT> * senddata = new tuple<IT,IT,NT>[locsize];	// re-used for both rows and columns
	for(int i=0; i<nprocs; ++i)
	{
		copy(data[i].begin(), data[i].end(), senddata+sdispls[i]);
		vector< tuple<IT,IT,NT> >().swap(data[i]);	// clear memory
	}
	MPI::Datatype MPI_triple = MPI::CHAR.Create_contiguous(sizeof(tuple<IT,IT,NT>));
	MPI_triple.Commit();

	IT totrecv = accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));	
	tuple<IT,IT,NT> * recvdata = new tuple<IT,IT,NT>[totrecv];	
	commGrid->GetWorld().Alltoallv(senddata, sendcnt, sdispls, MPI_triple, recvdata, recvcnt, rdispls, MPI_triple); 
	DeleteAll(senddata, sendcnt, recvcnt, sdispls, rdispls);

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

	SpTuples<IT,NT> A(totrecv, locrows, loccols, recvdata);	// It is ~SpTuples's job to deallocate
  	spSeq = new DER(A,false);        // Convert SpTuples to DER
}

template <class IT, class NT, class DER>
template <class DELIT>
SpParMat< IT,NT,DER >::SpParMat (const DistEdgeList<DELIT> & DEL, bool removeloops)
{
	commGrid.reset(new CommGrid(*(DEL.commGrid)));		
	//int rank = commGrid->GetRank();
	int nprocs = commGrid->GetSize();
	int r = commGrid->GetGridRows();
	int s = commGrid->GetGridCols();
	vector< vector<IT> > data(nprocs);

	IT m_perproc = DEL.getGlobalV() / r;
	IT n_perproc = DEL.getGlobalV() / s;

	if(sizeof(IT) < sizeof(DELIT))
	{
		ostringstream outs;
		outs << "Warning: Using smaller indices for the matrix than DistEdgeList\n";
		outs << "Local matrices are " << m_perproc << "-by-" << n_perproc << endl;
		SpParHelper::Print(outs.str());
	}	
	
	// to lower memory consumption, form sparse matrix in stages
	IT stages = MEM_EFFICIENT_STAGES;	
	
	// even if local indices (IT) are 32-bits, we should work with 64-bits for global info
	int64_t perstage = DEL.nedges / stages;
	IT totrecv = 0;
	vector<IT> alledges;

	int maxr = r-1;
	int maxs = s-1;	
	for(IT s=0; s< stages; ++s)
	{
		int64_t n_befor = s*perstage;
		int64_t n_after= ((s==(stages-1))? DEL.nedges : ((s+1)*perstage));

		// clear the source vertex by setting it to -1
		int realedges = 0;	// these are "local" realedges

		if(DEL.pedges)	
		{
			for (IT i = n_befor; i < n_after; i++)
			{
				int64_t fr = get_v0_from_edge(&(DEL.pedges[i]));
				int64_t to = get_v1_from_edge(&(DEL.pedges[i]));

				if(fr >= 0 && to >= 0)	// otherwise skip
				{
					int rowowner = min(static_cast<int>(fr / m_perproc), maxr);
					int colowner = min(static_cast<int>(to / n_perproc), maxs); 
					int owner = commGrid->GetRank(rowowner, colowner);
					data[owner].push_back(fr - (rowowner * m_perproc));	// row_id
					data[owner].push_back(to - (colowner * n_perproc));	// col_id
					++realedges;
				}
			}
		}
		else
		{
			for (IT i = n_befor; i < n_after; i++)
			{
				if(DEL.edges[2*i+0] >= 0 && DEL.edges[2*i+1] >= 0)	// otherwise skip
				{
					int rowowner = min(static_cast<int>(DEL.edges[2*i+0] / m_perproc), maxr);
					int colowner = min(static_cast<int>(DEL.edges[2*i+1] / n_perproc), maxs); 
					int owner = commGrid->GetRank(rowowner, colowner);
					data[owner].push_back(DEL.edges[2*i+0]- (rowowner * m_perproc));	// row_id
					data[owner].push_back(DEL.edges[2*i+1]- (colowner * n_perproc));	// col_id
					++realedges;
				}
			}
		}

  		IT * sendbuf = new IT[2*realedges];
		int * sendcnt = new int[nprocs];
		int * sdispls = new int[nprocs];
		for(int i=0; i<nprocs; ++i)
			sendcnt[i] = data[i].size();

		int * rdispls = new int[nprocs];
		int * recvcnt = new int[nprocs];
		commGrid->GetWorld().Alltoall(sendcnt, 1, MPI::INT, recvcnt, 1, MPI::INT); // share the counts 

		sdispls[0] = 0;
		rdispls[0] = 0;
		for(int i=0; i<nprocs-1; ++i)
		{
			sdispls[i+1] = sdispls[i] + sendcnt[i];
			rdispls[i+1] = rdispls[i] + recvcnt[i];
		}
		for(int i=0; i<nprocs; ++i)
			copy(data[i].begin(), data[i].end(), sendbuf+sdispls[i]);
		
		// clear memory
		for(int i=0; i<nprocs; ++i)
			vector<IT>().swap(data[i]);

		IT thisrecv = accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));	// thisrecv = 2*locedges
		IT * recvbuf = new IT[thisrecv];
		totrecv += thisrecv;
			
		commGrid->GetWorld().Alltoallv(sendbuf, sendcnt, sdispls, MPIType<IT>(), recvbuf, recvcnt, rdispls, MPIType<IT>()); 
		DeleteAll(sendcnt, recvcnt, sdispls, rdispls,sendbuf);
		copy (recvbuf,recvbuf+thisrecv,back_inserter(alledges));	// copy to all edges
		delete [] recvbuf;
	}

	int myprocrow = commGrid->GetRankInProcCol();
	int myproccol = commGrid->GetRankInProcRow();
	IT locrows, loccols; 
	if(myprocrow != r-1)	locrows = m_perproc;
	else 	locrows = DEL.getGlobalV() - myprocrow * m_perproc;
	if(myproccol != s-1)	loccols = n_perproc;
	else	loccols = DEL.getGlobalV() - myproccol * n_perproc;

  	SpTuples<IT,NT> A(totrecv/2, locrows, loccols, alledges, removeloops);  	// alledges is empty upon return
  	spSeq = new DER(A,false);        // Convert SpTuples to DER
}

template <class IT, class NT, class DER>
IT SpParMat<IT,NT,DER>::RemoveLoops()
{
	MPI::Intracomm DiagWorld = commGrid->GetDiagWorld();
	IT totrem;
	IT removed = 0;
	if(DiagWorld != MPI::COMM_NULL) // Diagonal processors only
	{
		SpTuples<IT,NT> tuples(*spSeq);
		delete spSeq;
		removed  = tuples.RemoveLoops();
		spSeq = new DER(tuples, false, NULL);	// Convert to DER
	}
	commGrid->GetWorld().Allreduce( &removed, & totrem, 1, MPIType<IT>(), MPI::SUM); 
	return totrem;
}		


template <class IT, class NT, class DER>
template <typename OT>
void SpParMat<IT,NT,DER>::OptimizeForGraph500(OptBuf<IT,OT> & optbuf)
{
	if(spSeq->getnsplit() > 0)
	{
		SpParHelper::Print("Can not declare preallocated buffers for multithreaded execution");
		return;
	}

	// Set up communication buffers, one for all
	IT mA = spSeq->getnrow();
	IT p_c = commGrid->GetGridCols();
	vector<bool> isthere(mA, false); // perhaps the only appropriate use of this crippled data structure
	IT perproc = mA / p_c;
	vector<int> maxlens(p_c,0);	// maximum data size to be sent to any neighbor along the processor row

	for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)
	{
		for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)
		{
			IT rowid = nzit.rowid();
			if(!isthere[rowid])
			{
				IT owner = min(nzit.rowid() / perproc, p_c-1); 			
				maxlens[owner]++;
				isthere[rowid] = true;
			}
		}
	}
	SpParHelper::Print("Optimization buffers set\n");
	optbuf.Set(maxlens);
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
	IT m_perproc = 0, n_perproc = 0;

	int colneighs = commGrid->GetGridRows();	// number of neighbors along this processor column (including oneself)
	int rowneighs = commGrid->GetGridCols();	// number of neighbors along this processor row (including oneself)

	IT buffpercolneigh = MEMORYINBYTES / (colneighs * (2 * sizeof(IT) + sizeof(NT)));
	IT buffperrowneigh = MEMORYINBYTES / (rowneighs * (2 * sizeof(IT) + sizeof(NT)));

	// make sure that buffperrowneigh >= buffpercolneigh to cover for this patological case:
	//   	-- all data received by a given column head (by vertical communication) are headed to a single processor along the row
	//   	-- then making sure buffperrowneigh >= buffpercolneigh guarantees that the horizontal buffer will never overflow
	buffperrowneigh = std::max(buffperrowneigh, buffpercolneigh);

	if(std::max(buffpercolneigh * colneighs, buffperrowneigh * rowneighs) > numeric_limits<int>::max())
	{  
		cout << "COMBBLAS: MPI doesn't support sending int64_t send/recv counts or displacements" << endl;
	}
 
	int * cdispls = new int[colneighs];
	for (IT i=0; i<colneighs; ++i)
		cdispls[i] = i*buffpercolneigh;

	int * rdispls = new int[rowneighs];
	for (IT i=0; i<rowneighs; ++i)
		rdispls[i] = i*buffperrowneigh;		

	int *ccurptrs = NULL, *rcurptrs = NULL;	// MPI doesn't support sending int64_t send/recv counts
	int recvcount = 0;

	IT * rows = NULL; 
	IT * cols = NULL;
	NT * vals = NULL;

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

		ccurptrs = new int[colneighs];
		rcurptrs = new int[rowneighs];
		fill_n(ccurptrs, colneighs, 0);	// fill with zero
		fill_n(rcurptrs, rowneighs, 0);	
		
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
			double loadval;
			IT cnz = 0;
			char line[1024];
			while ( (!infile.eof()) && cnz < total_nnz)
			{
				/*
				infile >> temprow >> tempcol;
				if(nonum)
					tempval = static_cast<NT>(1);
				else
				{
					//infile >> tempval;
					infile >> loadval;
					tempval = static_cast<NT>(loadval);
				}*/
				
				// read one line at a time so that missing numerical values can be detected
				infile.getline(line, 1024);
				stringstream linestream(line);
				linestream >> temprow >> tempcol;
				if(nonum)
					tempval = static_cast<NT>(1);
				else
				{
					linestream >> skipws;
					if (linestream.eof())
					{
						// there isn't a value
						tempval = static_cast<NT>(1);
					}
					else
					{
						linestream >> loadval;
						tempval = static_cast<NT>(loadval);
					}
				}

				--temprow;	// file is 1-based where C-arrays are 0-based
				--tempcol;

				int colrec = std::min(static_cast<int>(temprow / m_perproc), colneighs-1);	// precipient processor along the column
				rows[ colrec * buffpercolneigh + ccurptrs[colrec] ] = temprow;
				cols[ colrec * buffpercolneigh + ccurptrs[colrec] ] = tempcol;
				vals[ colrec * buffpercolneigh + ccurptrs[colrec] ] = tempval;
				++ (ccurptrs[colrec]);				

				if(ccurptrs[colrec] == buffpercolneigh || (cnz == (total_nnz-1)) )		// one buffer is full, or file is done !
				{
					// first, send the receive counts ...
					(commGrid->colWorld).Scatter(ccurptrs, 1, MPI::INT, &recvcount, 1, MPI::INT, rankincol);

					// generate space for own recv data ... (use arrays because vector<bool> is cripled, if NT=bool)
					IT * temprows = new IT[recvcount];
					IT * tempcols = new IT[recvcount];
					NT * tempvals = new NT[recvcount];
					
					// then, send all buffers that to their recipients ...
					(commGrid->colWorld).Scatterv(rows, ccurptrs, cdispls, MPIType<IT>(), temprows, recvcount,  MPIType<IT>(), rankincol); 
					(commGrid->colWorld).Scatterv(cols, ccurptrs, cdispls, MPIType<IT>(), tempcols, recvcount,  MPIType<IT>(), rankincol); 
					(commGrid->colWorld).Scatterv(vals, ccurptrs, cdispls, MPIType<NT>(), tempvals, recvcount,  MPIType<NT>(), rankincol); 

					// finally, reset current pointers !
					fill_n(ccurptrs, colneighs, 0);
					DeleteAll(rows, cols, vals);
			
					/* Begin horizontal distribution */

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
				
					// Send the receive counts for horizontal communication ...
					(commGrid->rowWorld).Scatter(rcurptrs, 1, MPI::INT, &recvcount, 1, MPI::INT, rankinrow);

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
					for(int i=0; i< recvcount; ++i)
					{					
						localtuples.push_back( 	make_tuple(temprows[i]-moffset, tempcols[i]-noffset, tempvals[i]) );
					}
					
					fill_n(rcurptrs, rowneighs, 0);
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
			fill_n(ccurptrs, colneighs, numeric_limits<int>::max());	
			(commGrid->colWorld).Scatter(ccurptrs, 1, MPI::INT, &recvcount, 1, MPI::INT, rankincol);

			// And along the row ...
			fill_n(rcurptrs, rowneighs, numeric_limits<int>::max());				
			(commGrid->rowWorld).Scatter(rcurptrs, 1, MPI::INT, &recvcount, 1, MPI::INT, rankinrow);
			
		}
		else	// input file does not exist !
		{
			cout << "COMBBLAS: Input file doesn't exist" << endl;
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
			(commGrid->colWorld).Scatter(ccurptrs, 1, MPI::INT, &recvcount, 1, MPI::INT, rankincol);

			if( recvcount == numeric_limits<int>::max())
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
			rcurptrs = new int[rowneighs];
			fill_n(rcurptrs, rowneighs, 0);	
		
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
				
			// Send the receive counts for horizontal communication ...
			(commGrid->rowWorld).Scatter(rcurptrs, 1, MPI::INT, &recvcount, 1, MPI::INT, rankinrow);
			
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
			for(int i=0; i< recvcount; ++i)
			{					
				localtuples.push_back( 	make_tuple(temprows[i]-moffset, tempcols[i]-noffset, tempvals[i]) );
			}
					
			fill_n(rcurptrs, rowneighs,0);
			DeleteAll(rows, cols, vals,temprows, tempcols, tempvals);	
		}
		// Signal the end of file to other processors along the row
		fill_n(rcurptrs, rowneighs, numeric_limits<int>::max());				
		(commGrid->rowWorld).Scatter(rcurptrs, 1, MPI::INT, &recvcount, 1, MPI::INT, rankinrow);
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
			(commGrid->rowWorld).Scatter(rcurptrs, 1, MPI::INT, &recvcount, 1, MPI::INT, rankinrow);
			if( recvcount == numeric_limits<int>::max())
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
			for(IT i=0; i< recvcount; ++i)
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


//! The input parameters' identity (zero) elements as well as 
//! their communication grid is preserved while outputting
template <class IT, class NT, class DER>
void SpParMat<IT,NT,DER>::Find (FullyDistVec<IT,IT> & distrows, FullyDistVec<IT,IT> & distcols, FullyDistVec<IT,NT> & distvals) const
{
	if((*(distrows.commGrid) != *(distcols.commGrid)) || (*(distcols.commGrid) != *(distvals.commGrid)))
	{
		SpParHelper::Print("Grids are not comparable, Find() fails !"); 
		MPI::COMM_WORLD.Abort(GRIDMISMATCH);
	}
	IT globallen = getnnz();
	SpTuples<IT,NT> Atuples(*spSeq);
	
	FullyDistVec<IT,IT> nrows ( distrows.commGrid, globallen, 0, distrows.zero); 
	FullyDistVec<IT,IT> ncols ( distcols.commGrid, globallen, 0, distcols.zero); 
	FullyDistVec<IT,NT> nvals ( distvals.commGrid, globallen, 0, distvals.zero); 
	
	IT prelen = Atuples.getnnz();
	//IT postlen = nrows.MyLocLength();

	int rank = commGrid->GetRank();
	int nprocs = commGrid->GetSize();
	IT * prelens = new IT[nprocs];
	prelens[rank] = prelen;
	commGrid->GetWorld().Allgather(MPI::IN_PLACE, 0, MPIType<IT>(), prelens, 1, MPIType<IT>());
	IT prelenuntil = accumulate(prelens, prelens+rank, 0);

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

	commGrid->GetColWorld().Allgather(MPI::IN_PLACE, 0, MPIType<IT>(),locnrows, 1, MPIType<IT>());
	commGrid->GetRowWorld().Allgather(MPI::IN_PLACE, 0, MPIType<IT>(),locncols, 1, MPIType<IT>());
	IT roffset = accumulate(locnrows, locnrows+colrank, 0);
	IT coffset = accumulate(locncols, locncols+rowrank, 0);
	
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
	commGrid->GetWorld().Alltoall(sendcnt, 1, MPI::INT, recvcnt, 1, MPI::INT);	// get the recv counts

	int * sdpls = new int[nprocs]();	// displacements (zero initialized pid) 
	int * rdpls = new int[nprocs](); 
	partial_sum(sendcnt, sendcnt+nprocs-1, sdpls+1);
	partial_sum(recvcnt, recvcnt+nprocs-1, rdpls+1);

	commGrid->GetWorld().Alltoallv(rows, sendcnt, sdpls, MPIType<IT>(), &(nrows.arr[0]), recvcnt, rdpls, MPIType<IT>()); 
	commGrid->GetWorld().Alltoallv(cols, sendcnt, sdpls, MPIType<IT>(), &(ncols.arr[0]), recvcnt, rdpls, MPIType<IT>()); 
	commGrid->GetWorld().Alltoallv(vals, sendcnt, sdpls, MPIType<NT>(), &(nvals.arr[0]), recvcnt, rdpls, MPIType<NT>()); 
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
		SpParHelper::Print("Grids are not comparable, Find() fails !"); 
		MPI::COMM_WORLD.Abort(GRIDMISMATCH);
	}
	IT globallen = getnnz();
	SpTuples<IT,NT> Atuples(*spSeq);
	
	FullyDistVec<IT,IT> nrows ( distrows.commGrid, globallen, 0, distrows.zero); 
	FullyDistVec<IT,IT> ncols ( distcols.commGrid, globallen, 0, distcols.zero); 
	
	IT prelen = Atuples.getnnz();
	//IT postlen = nrows.MyLocLength();

	int rank = commGrid->GetRank();
	int nprocs = commGrid->GetSize();
	IT * prelens = new IT[nprocs];
	prelens[rank] = prelen;
	commGrid->GetWorld().Allgather(MPI::IN_PLACE, 0, MPIType<IT>(), prelens, 1, MPIType<IT>());
	IT prelenuntil = accumulate(prelens, prelens+rank, 0);

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

	commGrid->GetColWorld().Allgather(MPI::IN_PLACE, 0, MPIType<IT>(),locnrows, 1, MPIType<IT>());
	commGrid->GetRowWorld().Allgather(MPI::IN_PLACE, 0, MPIType<IT>(),locncols, 1, MPIType<IT>());
	IT roffset = accumulate(locnrows, locnrows+colrank, 0);
	IT coffset = accumulate(locncols, locncols+rowrank, 0);
	
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
	commGrid->GetWorld().Alltoall(sendcnt, 1, MPI::INT, recvcnt, 1, MPI::INT);	// get the recv counts

	int * sdpls = new int[nprocs]();	// displacements (zero initialized pid) 
	int * rdpls = new int[nprocs](); 
	partial_sum(sendcnt, sendcnt+nprocs-1, sdpls+1);
	partial_sum(recvcnt, recvcnt+nprocs-1, rdpls+1);

	commGrid->GetWorld().Alltoallv(rows, sendcnt, sdpls, MPIType<IT>(), &(nrows.arr[0]), recvcnt, rdpls, MPIType<IT>()); 
	commGrid->GetWorld().Alltoallv(cols, sendcnt, sdpls, MPIType<IT>(), &(ncols.arr[0]), recvcnt, rdpls, MPIType<IT>()); 
	DeleteAll(sendcnt, recvcnt, sdpls, rdpls);
	DeleteAll(prelens, rows, cols, vals);
	distrows = nrows;
	distcols = ncols;
}

template <class IT, class NT, class DER>
ofstream& SpParMat<IT,NT,DER>::put(ofstream& outfile) const
{
	outfile << (*spSeq) << endl;
	return outfile;
}

template <class IU, class NU, class UDER>
ofstream& operator<<(ofstream& outfile, const SpParMat<IU, NU, UDER> & s)
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
int SpParMat<IT,NT,DER>::Owner(IT total_m, IT total_n, IT grow, IT gcol, IT & lrow, IT & lcol) const
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
