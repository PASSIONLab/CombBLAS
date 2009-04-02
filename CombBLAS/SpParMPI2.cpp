/****************************************************************/
/* Sequential and Parallel Sparse Matrix Multiplication Library */
/* version 2.3 --------------------------------------------------/
/* date: 01/18/2009 ---------------------------------------------/
/* author: Aydin Buluc (aydin@cs.ucsb.edu) ----------------------/
/****************************************************************/

#include "SpParMPI2.h"

template <class T>
SpParMPI2<T>::SpParMPI2 (ifstream & input, MPI::IntraComm & world)
{
	if(!input.is_open())
	{
		perror("Input file doesn't exist\n");
		exit(-1);
	}

	commGrid.reset(new CommGrid(world, 0, 0));

	ITYPE m,n,nnz;
	input >> m >> n >> nnz;

	SpTuples<T> * s = new SpTuples<T>(nnz,m,n);
	if(commGrid->myrank == 0)
		cout<<"Reading to SpTuples"<<endl;
	input >> (*s);

	s->SortColBased();
	if(commGrid->myrank == 0)
		cout<<"Converting to SpDCCols"<<endl;

	spSeq = new SpDCCols<T>(*s, false, NULL);	
	delete s;
}

template <class T>
SpParMPI2<T>::SpParMPI2 (const SpParMPI2<T> & rhs)
{
	if(rhs.spSeq != NULL)	
		spSeq = new SpMat<IT,NT,DER>(*(rhs.spSeq));  // Deep copy of local block

	if(rhs.commGrid != NULL)	
		commGrid = new CommGrid(*(rhs.commGrid));  // Deep copy of communication grid
}

template <class T>
SpParMPI2<T> & SpParMPI2<T>::operator=(const SpParMPI2<T> & rhs)
{
	if(this != &rhs)		
	{
		//! Check agains NULL is probably unneccessary, delete won't fail on NULL
		//! But useful in the presence of a user defined "operator delete" which fails to check NULL
		if(spSeq != NULL) delete spSeq;
		if(rhs.spSeq != NULL)	
			spSeq = new SpMat<IT,NT,DER>(*(rhs.spSeq));  // Deep copy of local block
		
		if(commGrid != NULL) delete commGrid;
		if(rhs.commGrid != NULL)	
			commGrid = new CommGrid(*(rhs.commGrid));  // Deep copy of communication grid
	}
	return *this;
}

template <class T>
SpParMPI2<T> & SpParMPI2<T>::operator+=(const SpParMPI2<T> & rhs)
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
ITYPE SpParMPI2<T>::getnnz() const
{
	ITYPE totalnnz = 0;    
	ITYPE localnnz = spSeq->getnzmax();
	MPI_Allreduce( &localnnz, &totalnnz, 1, DataTypeToMPI<ITYPE>(), MPI_SUM, commGrid->commWorld );
 	return totalnnz;  
}

template <class T>
ITYPE SpParMPI2<T>::getrows() const
{
	ITYPE totalrows = 0;
	ITYPE localrows = spSeq->getrows();    
	MPI_Allreduce( &localrows, &totalrows, 1, DataTypeToMPI<ITYPE>(), MPI_SUM, commGrid->colWorld );
 	return totalrows;  
}

template <class T>
ITYPE SpParMPI2<T>::getcols() const
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
SpParMatrix<T> * SpParMPI2<T>::SubsRefCol (const vector<ITYPE> & ci) const
{
	vector<ITYPE> ri;
	
 	shared_ptr< SpDCCols<T> > ARef (new SpDCCols<T> (spSeq->SubsRefCol(ci)));	

	return new SpParMPI2<T> (ARef, commGrid->commWorld);
}


template <class T>
const SpParMPI2<T> operator* (const SpParMPI2<T> & A, SpParMPI2<T> & B )
{
	if((A.spSeq)->getcols() != (B.spSeq)->getrows())
	{
		cout<<"Can not multiply, dimensions does not match"<<endl;
		return SpParMPI2<T>(MPI_COMM_WORLD);
	}

	int stages;
	int Aoffset, Boffset;
	CommGrid GridC = GridConformance(*(A.commGrid), *(B.commGrid), stages, Aoffset, Boffset);	// stages = inner dimension of matrix blocks
	
	double t1=MPI_Wtime();

		
	// Attention: *(B.spSeq) is practically destroyed after Transpose is called	
	SpDCCols<T> Btrans = (B.spSeq)->Transpose();
	Btrans.TransposeInPlace();	// calls SparseMatrix's Transpose in place which is swap(m,n);
	
	// set row & col window handles
	SpWins rowwindows, colwindows;
	SpParMPI2<T>::SetWindows((A.commGrid)->rowWorld, *(A.spSeq), rowwindows);
	SpParMPI2<T>::SetWindows((B.commGrid)->colWorld,  Btrans, colwindows);

	IT ** ARecvSizes = allocate2D(DER::esscount, stages);
	IT ** BRecvSizes = allocate2D(DER::esscount, stages);
  
	SpParMPI2<T>::GetSetSizes((A.commGrid)->mycol, *(A.spSeq), ARecvSizes, (A.commGrid)->rowWorld);
	SpParMPI2<T>::GetSetSizes((B.commGrid)->myrow, Btrans, BRecvSizes, (B.commGrid)->colWorld);
	
	double t2 = MPI_Wtime();
	if(GridC.myrank == 0)
		fprintf(stdout, "setup (matrix transposition and memory registration) took %.6lf seconds\n", t2-t1);
	
	SpDCCols<T> * ARecv;
	SpDCCols<T> * BRecv; 

	SpMat<IT,NT,DER> C;   // Create an empty object for the product	


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
	
	(B.spSeq).reset(new SpDCCols<T>(Btrans.Transpose()));	// Btrans does no longer point to a valid chunk of data	
	(B.spSeq)->TransposeInPlace();
	
	return SpParMPI2<T>(SpProduct, GridC.commWorld);
}

template <class T>
void SpParMPI2<T>::SetWindows(MPI_Comm & comm1d, SpDCCols<T> & Matrix, SpWins & wins) 
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
 * @param[in] index Index of this processor within its row/col, can be {0,...r/s-1}
 * @param[in] sizes 2D array where 
 *  	sizes[i] is an array of size r/s representing the ith essential component of all local blocks within that row/col
 *	sizes[i][j] is the size of the ith essential component of the jth local block within this row/col
 */
template <class IT, class NT, class DER>
void SpParMPI2<IT,NT,DER>::GetSetSizes(IT index, SpMat<IT,NT,DER> & Matrix, IT ** & sizes, MPI::IntraComm & comm1d)
{
	vector<IT> essentials = Matrix.GetEssentials();
	for(IT i=0; i< essentials.size(); ++i)
	{
		sizes[i][index] = essentials[i]; 
		comm1d.Allgather(MPI::IN_PLACE, 1, DataTypeToMPI<IT>(), sizes[i], 1, DataTypeToMPI<IT>());
	}
}

template <class IT, class NT, class DER>
ofstream& SpParMPI2<IT,NT,DER>::put(ofstream& outfile) const
{
	SpTuples<T> triplets(*spSeq);
	outfile << triplets << endl;
}

template <class UIT, class UNT, class UDER>
ofstream& operator<<(ofstream& outfile, const SpParMPI2<UIT, UNT, UDER> & s)
{
	return s.put(outfile) ;	// use the right put() function

}
