/****************************************************************/
/* Sequential and Parallel Sparse Matrix Multiplication Library */
/* version 2.3 --------------------------------------------------/
/* date: 01/18/2009 ---------------------------------------------/
/* author: Aydin Buluc (aydin@cs.ucsb.edu) ----------------------/
/****************************************************************/

#include "SpParMPI2.h"

template <class IT, class NT, class DER>
SpParMPI2< IT,NT,DER >::SpParMPI2 (ifstream & input, MPI::Intracomm & world)
{
	if(!input.is_open())
	{
		perror("Input file doesn't exist\n");
		exit(-1);
	}
	commGrid = new CommGrid(world, 0, 0);
	input >> (*spSeq);
}

template <class IT, class NT, class DER>
SpParMPI2< IT,NT,DER >::SpParMPI2 (SpMat<IT,NT,DER> * myseq, MPI::Intracomm & world): spSeq(myseq)
{
	commGrid = new CommGrid(world, 0, 0);
}

template <class IT, class NT, class DER>
SpParMPI2< IT,NT,DER >::SpParMPI2 (): spSeq(NULL)
{
	commGrid = new CommGrid(MPI::COMM_WORLD, 0, 0);
}


template <class IT, class NT, class DER>
SpParMPI2< IT,NT,DER >::~SpParMPI2 ()
{
	if(spSeq != NULL) delete spSeq;
	if(commGrid != NULL) delete commGrid;
}


template <class IT, class NT, class DER>
SpParMPI2< IT,NT,DER >::SpParMPI2 (const SpParMPI2< IT,NT,DER > & rhs)
{
	if(rhs.spSeq != NULL)	
		spSeq = new SpMat< IT,NT,DER >(*(rhs.spSeq));  	// Deep copy of local block

	if(rhs.commGrid != NULL)	
		commGrid = new CommGrid(*(rhs.commGrid));  	// Deep copy of communication grid
}

template <class IT, class NT, class DER>
SpParMPI2< IT,NT,DER > & SpParMPI2< IT,NT,DER >::operator=(const SpParMPI2< IT,NT,DER > & rhs)
{
	if(this != &rhs)		
	{
		//! Check agains NULL is probably unneccessary, delete won't fail on NULL
		//! But useful in the presence of a user defined "operator delete" which fails to check NULL
		if(spSeq != NULL) delete spSeq;
		if(rhs.spSeq != NULL)	
			spSeq = new SpMat< IT,NT,DER >(*(rhs.spSeq));  // Deep copy of local block
		
		if(commGrid != NULL) delete commGrid;
		if(rhs.commGrid != NULL)	
			commGrid = new CommGrid(*(rhs.commGrid));  // Deep copy of communication grid
	}
	return *this;
}

template <class IT, class NT, class DER>
SpParMPI2< IT,NT,DER > & SpParMPI2< IT,NT,DER >::operator+=(const SpParMPI2< IT,NT,DER > & rhs)
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
		cout<< "Missing feature (A+A): Use multiply with 2 instead !"<<endl;	
	}
	return *this;	
}

template <class IT, class NT, class DER>
IT SpParMPI2< IT,NT,DER >::getnnz() const
{
	IT totalnnz = 0;    
	IT localnnz = spSeq->getnnz();
	(commGrid->commWorld).Allreduce( &localnnz, &totalnnz, 1, MPIType<IT>(), MPI::SUM);
 	return totalnnz;  
}

template <class IT, class NT, class DER>
IT SpParMPI2< IT,NT,DER >::getnrow() const
{
	IT totalrows = 0;
	IT localrows = spSeq->getnrow();    
	(commGrid->colWorld).Allreduce( &localrows, &totalrows, 1, MPIType<IT>(), MPI::SUM);
 	return totalrows;  
}

template <class IT, class NT, class DER>
IT SpParMPI2< IT,NT,DER >::getncol() const
{
	IT totalcols = 0;
	IT localcols = spSeq->getcols();    
	(commGrid->rowWorld).Allreduce( &localcols, &totalcols, 1, MPIType<IT>(), MPI::SUM);
 	return totalcols;  
}


/** 
 * Create a submatrix of size m x (size(ncols) * s) on a r x s processor grid
 * Essentially fetches the columns ci[0], ci[1],... ci[size(ci)] from every submatrix
 *
template <class T>
SpParMatrix<T> * SpParMPI2<T>::SubsRefCol (const vector<ITYPE> & ci) const
{
	vector<ITYPE> ri;
	
 	shared_ptr< SpDCCols<T> > ARef (new SpDCCols<T> (spSeq->SubsRefCol(ci)));	

	return new SpParMPI2<T> (ARef, commGrid->commWorld);
} */

template <typename SR, typename IU, typename NU, typename UDER> 
SpParMPI2<IU,NU,UDER> Mult_AnXBn (const SpParMPI2<IU,NU,UDER> & A, const SpParMPI2<IU,NU,UDER> & B )
{
	double t1 = MPI::Wtime();

	if(A.getncol() != B.getnrow())
	{
		cout<<"Can not multiply, dimensions does not match"<<endl;
		MPI::COMM_WORLD.Abort();
		return SpParMPI2< IU,NU,UDER >();
	}

	int stages, Aoffset, Boffset; 	// stages = inner dimension of matrix blocks
	CommGrid GridC = ProductGrid(*(A.commGrid), *(B.commGrid), stages, Aoffset, Boffset);		
		
	const_cast< SpParMPI2<IU,NU,UDER>* >(B.spSeq)->Transpose();
	
	// set row & col window handles
	vector<MPI::Win> rowwindows, colwindows;
	SpParHelper::SetWindows((A.commGrid)->rowWorld, *(A.spSeq), rowwindows);
	SpParHelper::SetWindows((B.commGrid)->colWorld, *(B.spSeq), colwindows);

	IU ** ARecvSizes = allocate2D(UDER::esscount, stages);
	IU ** BRecvSizes = allocate2D(UDER::esscount, stages);
 
	SpParHelper::GetSetSizes((A.commGrid)->mycol, *(A.spSeq), ARecvSizes, (A.commGrid)->rowWorld);
	SpParHelper::GetSetSizes((B.commGrid)->myrow, *(B.spSeq), BRecvSizes, (B.commGrid)->colWorld);
	
	double t2 = MPI::Wtime();
	if(GridC.myrank == 0)
		fprintf(stdout, "setup (matrix transposition and memory registration) took %.6lf seconds\n", t2-t1);
	
	SpMat<IU,NU,UDER> * ARecv;
	SpMat<IU,NU,UDER> * BRecv; 

	SpMat<IU,NU,UDER> * C = new SpMat<IU,NU,UDER>();   // Create an empty object for the product	

	for(int i = 0; i < stages; ++i) 	// Robust generalization to non-square grids will require block-cyclic distibution	
	{
		int Aownind = (i+Aoffset) % (A.commGrid)->grcol;		
		int Bownind = (i+Boffset) % (B.commGrid)->grrow;

		if(Aownind == GridC.mycol)
		{
			ARecv = A.spSeq;	// shallow-copy
		}
		else
		{
			// pack essentials to a vector
			vector<IU> ess(UDER::esscount);
			for(int j=0; j< UDER::esscount; ++j)	
			{
				ess[j] = ARecvSizes[j][Aownind];	
			}		
			SpParHelper::FetchMatrix(*ARecv, ess, rowwindows, Aownind);
		}
		if(Bownind == GridC.myrow)
		{
			BRecv = B.spSeq;	// shallow-copy
		}
		else
		{
			// pack essentials to a vector
			vector<IU> ess(UDER::esscount);
			for(int j=0; j< UDER::esscount; ++j)	
			{
				ess[j] = BRecvSizes[j][Bownind];	
			}	
			SpParHelper::FetchMatrix(*BRecv, ess, colwindows, Bownind);	
		}
	
		SpParHelper::UnlockWindows(Aownind, rowwindows);	// unlock windows for A
		SpParHelper::UnlockWindows(Bownind, colwindows);	// unlock windows for B	

		C->template SpGEMM < SR > ( *ARecv, *BRecv, false, true);
		
		if(Aownind != GridC.mycol) delete ARecv;
		if(Bownind != GridC.myrow) delete BRecv; 
	} 

	GridC.commWorld.Barrier();

	for(int i=0; i< rowwindows.size(); ++i)
	{
		rowwindows[i].Free();
	}
	for(int i=0; i< colwindows.size(); ++i)
	{
		colwindows[i].Free();
	}

	const_cast< SpParMPI2<IU,NU,UDER>* >(B.spSeq)->Transpose();	// transpose back to original
	
	return SpParMPI2<IU,NU,UDER> (C, GridC);			// return the result object
}


template <class IT, class NT, class DER>
ofstream& SpParMPI2<IT,NT,DER>::put(ofstream& outfile) const
{
	outfile << (*spSeq) << endl;
}

template <class IU, class NU, class UDER>
ofstream& operator<<(ofstream& outfile, const SpParMPI2<IU, NU, UDER> & s)
{
	return s.put(outfile) ;	// use the right put() function

}
