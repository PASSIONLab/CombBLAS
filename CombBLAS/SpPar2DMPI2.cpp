/****************************************************************/
/* Sequential and Parallel Sparse Matrix Multiplication Library */
/* version 2.3 --------------------------------------------------/
/* date: 01/18/2009 ---------------------------------------------/
/* author: Aydin Buluc (aydin@cs.ucsb.edu) ----------------------/
/****************************************************************/

#include "SparseOneSidedMPI.h"

template <class T>
SparseOneSidedMPI<T>::SparseOneSidedMPI (ifstream & input, MPI_Comm world)
{
	if(!input.is_open())
	{
		perror("Input file doesn't exist\n");
		exit(-1);
	}

	commGrid.reset(new CommGrid(world, 0, 0));

	ITYPE m,n,nnz;
	input >> m >> n >> nnz;

	SparseTriplets<T> * s = new SparseTriplets<T>(nnz,m,n);
	if(commGrid->myrank == 0)
		cout<<"Reading Triplets"<<endl;
	input >> (*s);

	s->SortColBased();
	if(commGrid->myrank == 0)
		cout<<"Converting to SparseDColumn"<<endl;

	shared_ptr< SparseDColumn<T> > d(new SparseDColumn<T>(*s, false, NULL));		// Smart pointer
	delete s;

	spSeq = d;
}

template <class T>
SparseOneSidedMPI<T>::SparseOneSidedMPI (const SparseOneSidedMPI<T> & rhs)
{
	commGrid.reset(new CommGrid(*(rhs.commGrid)));
	spSeq.reset(new SparseDColumn<T>(*(rhs.spSeq)));	// Deep copy of local block
}


template <class T>
SparseOneSidedMPI<T> & SparseOneSidedMPI<T>::operator=(const SparseOneSidedMPI<T> & rhs)
{
	if(this != &rhs)		
	{
		// Note: smart pointer will automatically call destructors if the existing object is not NULL
		spSeq.reset(new SparseDColumn<T>(*(rhs.spSeq)));	// Deep copy of local block
		commGrid.reset(new CommGrid(*(rhs.commGrid)));
	}
	return *this;
}

template <class T>
SparseOneSidedMPI<T> & SparseOneSidedMPI<T>::operator+=(const SparseOneSidedMPI<T> & rhs)
{
	if(this != &rhs)		
	{
		if(commGrid == rhs.commGrid)	
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
		cout<< "Missing feauture (A+A): Use multiply with 2 instead !"<<endl;	
	}
	return *this;
	
}

template <class T>
ITYPE SparseOneSidedMPI<T>::getnnz() const
{
	ITYPE totalnnz = 0;    
	ITYPE localnnz = spSeq->getnzmax();
	MPI_Allreduce( &localnnz, &totalnnz, 1, DataTypeToMPI<ITYPE>(), MPI_SUM, commGrid->commWorld );
 	return totalnnz;  
}

template <class T>
ITYPE SparseOneSidedMPI<T>::getrows() const
{
	ITYPE totalrows = 0;
	ITYPE localrows = spSeq->getrows();    
	MPI_Allreduce( &localrows, &totalrows, 1, DataTypeToMPI<ITYPE>(), MPI_SUM, commGrid->colWorld );
 	return totalrows;  
}

template <class T>
ITYPE SparseOneSidedMPI<T>::getcols() const
{
	ITYPE totalcols = 0;
	ITYPE localcols = spSeq->getcols();    
	MPI_Allreduce( &localcols, &totalcols, 1, DataTypeToMPI<ITYPE>(), MPI_SUM, commGrid->rowWorld );
 	return totalcols;  
}

/** 
 * Create a submatrix of size m x (size(ncols) * s) on a r x s processor grid
 * Essentially fetches the columns ci[0], ci[1],... ci[size(ci)] from every submatrix
 */
template <class T>
SpParMatrix<T> * SparseOneSidedMPI<T>::SubsRefCol (const vector<ITYPE> & ci) const
{
	vector<ITYPE> ri;
	
 	shared_ptr< SparseDColumn<T> > ARef (new SparseDColumn<T> (spSeq->SubsRefCol(ci)));	

	return new SparseOneSidedMPI<T> (ARef, commGrid->commWorld);
}


template <class T>
const SparseOneSidedMPI<T> operator* (const SparseOneSidedMPI<T> & A, SparseOneSidedMPI<T> & B )
{
	if((A.spSeq)->getcols() != (B.spSeq)->getrows())
	{
		cout<<"Can not multiply, dimensions does not match"<<endl;
		return SparseOneSidedMPI<T>(MPI_COMM_WORLD);
	}

	int stages;
	int Aoffset, Boffset;
	CommGrid GridC = GridConformance(*(A.commGrid), *(B.commGrid), stages, Aoffset, Boffset);	// stages = inner dimension of matrix blocks
	
	double t1=MPI_Wtime();

	// SpProduct is the output matrix (stored as a smart pointer)
	ITYPE zero = static_cast<ITYPE>(0);
	shared_ptr< SparseDColumn<T> > SpProduct(new SparseDColumn<T>(zero, (A.spSeq)->getrows(), (B.spSeq)->getcols(), zero)); 	
	
	// Attention: *(B.spSeq) is practically destroyed after Transpose is called	
	SparseDColumn<T> Btrans = (B.spSeq)->Transpose();
	Btrans.TransposeInPlace();	// calls SparseMatrix's Transpose in place which is swap(m,n);
	
	// set row & col window handles
	SpWins rowwindows, colwindows;
	SparseOneSidedMPI<T>::SetWindows((A.commGrid)->rowWorld, *(A.spSeq), rowwindows);
	SparseOneSidedMPI<T>::SetWindows((B.commGrid)->colWorld,  Btrans, colwindows);

	SpSizes ARecvSizes(stages);
	SpSizes BRecvSizes(stages);
	SparseOneSidedMPI<T>::GetSetSizes((A.commGrid)->mycol, *(A.spSeq), ARecvSizes, (A.commGrid)->rowWorld);
	SparseOneSidedMPI<T>::GetSetSizes((B.commGrid)->myrow, Btrans, BRecvSizes, (B.commGrid)->colWorld);
	
	double t2 = MPI_Wtime();
	if(GridC.myrank == 0)
		fprintf(stdout, "setup (matrix transposition and memory registration) took %.6lf seconds\n", t2-t1);
	
	SparseDColumn<T> * ARecv;
	SparseDColumn<T> * BRecv; 

	for(int i = 0; i < stages; i++) //!< Robust generalization to non-square grids require block-cyclic distibution	
	{
		int Aownind = (i+Aoffset) % (A.commGrid)->grcol;		
		int Bownind = (i+Boffset) % (B.commGrid)->grrow;

		if(Aownind == GridC.mycol)
		{
			ARecv = (A.spSeq).get();	// shallow-copy
		}
		else
		{
			GridC.GetA(ARecv, Aownind, rowwindows, ARecvSizes);
		}
		if(Bownind == GridC.myrow)
		{
			BRecv = &Btrans;	// shallow-copy
		}
		else
		{
			GridC.GetB(BRecv, Bownind, colwindows, BRecvSizes);	
		}
	
		GridC.UnlockWindows(Aownind, Bownind, rowwindows, colwindows);	// unlock the windows

		SpProduct->MultiplyAdd(*ARecv, *BRecv, false, true);
		
		if(Aownind != GridC.mycol) delete ARecv;
		if(Bownind != GridC.myrow) delete BRecv; 
	} 

	MPI_Barrier(GridC.commWorld);
	MPI_Win_free(&rowwindows.maswin);
	MPI_Win_free(&rowwindows.jcwin);
	MPI_Win_free(&rowwindows.irwin);
	MPI_Win_free(&rowwindows.numwin);
	MPI_Win_free(&colwindows.maswin);
	MPI_Win_free(&colwindows.jcwin);
	MPI_Win_free(&colwindows.irwin);
	MPI_Win_free(&colwindows.numwin);
	
	(B.spSeq).reset(new SparseDColumn<T>(Btrans.Transpose()));	// Btrans does no longer point to a valid chunk of data	
	(B.spSeq)->TransposeInPlace();
	
	return SparseOneSidedMPI<T>(SpProduct, GridC.commWorld);
}

template <class T>
void SparseOneSidedMPI<T>::SetWindows(MPI_Comm & comm1d, SparseDColumn<T> & Matrix, SpWins & wins) 
{
	size_t sit = sizeof(ITYPE);

	// int MPI_Win_create(void *base, MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, MPI_Win *win);
	// The displacement unit argument is provided to facilitate address arithmetic in RMA operations
	// Collective operation, everybody exposes its own array to everyone else in the communicator
	MPI_Win_create(Matrix.GetMAS(), (Matrix.GetJCSize()+1) * sit, sit, MPI_INFO_NULL, comm1d, &(wins.maswin));
	MPI_Win_create(Matrix.GetJC(), Matrix.GetJCSize() * sit, sit, MPI_INFO_NULL, comm1d, &(wins.jcwin));
	MPI_Win_create(Matrix.GetIR(), Matrix.GetSize() * sit, sit, MPI_INFO_NULL, comm1d, &(wins.irwin));
	MPI_Win_create(Matrix.GetNUM(), Matrix.GetSize() * sizeof(T), sizeof(T), MPI_INFO_NULL, comm1d, &(wins.numwin));
}


/**
 * @param[in] index of this processor within its row/col, can be {0,...r/s-1}
 */
template <class T>
void SparseOneSidedMPI<T>::GetSetSizes(ITYPE index, SparseDColumn<T> & Matrix, SpSizes & sizes, MPI_Comm & comm1d)
{
	sizes.nrows[index] = Matrix.getrows();
	sizes.ncols[index] = Matrix.getcols();
	sizes.nzcs[index] = Matrix.GetJCSize();
	sizes.nnzs[index] = Matrix.GetSize();

	MPI_Allgather(MPI_IN_PLACE, 1, DataTypeToMPI<ITYPE>(), sizes.nrows, 1, DataTypeToMPI<ITYPE>(), comm1d);
	MPI_Allgather(MPI_IN_PLACE, 1, DataTypeToMPI<ITYPE>(), sizes.ncols, 1, DataTypeToMPI<ITYPE>(), comm1d);
	MPI_Allgather(MPI_IN_PLACE, 1, DataTypeToMPI<ITYPE>(), sizes.nzcs, 1, DataTypeToMPI<ITYPE>(), comm1d);
	MPI_Allgather(MPI_IN_PLACE, 1, DataTypeToMPI<ITYPE>(), sizes.nnzs, 1, DataTypeToMPI<ITYPE>(), comm1d);
}

template <class T>
ofstream& SparseOneSidedMPI<T>::put(ofstream& outfile) const
{
	SparseTriplets<T> triplets(*spSeq);
	outfile << triplets << endl;
}

template <typename U>
ofstream& operator<<(ofstream& outfile, const SparseOneSidedMPI<U> & s)
{
	return s.put(outfile) ;	// use the right put() function

}
