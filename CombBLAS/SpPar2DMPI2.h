/****************************************************************/
/* Sequential and Parallel Sparse Matrix Multiplication Library */
/* version 2.3 --------------------------------------------------/
/* date: 01/18/2009 ---------------------------------------------/
/* author: Aydin Buluc (aydin@cs.ucsb.edu) ----------------------/
/****************************************************************/

#ifndef _SPARSE_ONESIDED_MPI_H
#define _SPARSE_ONESIDED_MPI_H

#include <iostream>
#include <fstream>
#include <cmath>
#include <mpi.h>
#include "SparseMatrix.h"
#include "SpParMatrix.h"
#include "SparseTriplets.h"
#include "SparseDColumn.h"
#include "CommGrid.h"
#include "SpWins.h"
#include "SpSizes.h"
#include "DataTypeConvert.h"
#include <boost/shared_ptr.hpp>

using namespace std;
using namespace boost;


/**
  * This class implements an asynchronous 2D algorithm, in the sense that there is no notion of stages.
  * \n The process that completes its submatrix update requests subsequent matrices from their owners without waiting to sychronize with other processors
  * \n This partially remedies the severe load balancing problem in sparse matrices. 
  * \n The class uses MPI-2 to achieve one-sided asynchronous communication
  * \n The algorithm treats each submatrix as a single block
  * \n Local Data Structure used is SparseDColumn which is composed of DCSC only 
  */
template <class T>
class SparseOneSidedMPI: public SpParMatrix<T>
{
public:
	// Constructors
	SparseOneSidedMPI (MPI_Comm world) 
	{
		commGrid.reset(new CommGrid(world, 0, 0)); 	
	};
	SparseOneSidedMPI (ifstream & input, MPI_Comm world);
	SparseOneSidedMPI (shared_ptr< SparseDColumn<T> > myseq, MPI_Comm world):spSeq(myseq) 
	{	
		commGrid.reset(new CommGrid(world, 0, 0));
	};
	SparseOneSidedMPI (shared_ptr< SparseDColumn<T> > myseq,shared_ptr< CommGrid > grid): spSeq(myseq), commGrid(grid) {};

	SparseOneSidedMPI (const SparseOneSidedMPI<T> & rhs);			// copy constructor
	SparseOneSidedMPI<T> & operator=(const SparseOneSidedMPI<T> & rhs);	// assignment operator
	SparseOneSidedMPI<T> & operator+=(const SparseOneSidedMPI<T> & rhs);

	//! No exclicit destructor needed as smart pointers take care of calling the appropriate destructors 

	//! operator* should alter the second matrix B during the computation, due to memory limitations.
	//! It transposes the second matrix before the multiplications start, therefore it has no memory to store the old version
	//! After the multiplications are done, B is correctly restored back.
	template <typename U> 
	friend const SparseOneSidedMPI<U> operator* (const SparseOneSidedMPI<U> & A, SparseOneSidedMPI<U> & B );

	virtual ITYPE getrows() const;
	virtual ITYPE getcols() const;
	virtual ITYPE getnnz() const;	
	virtual shared_ptr<CommGrid> getcommgrid() const { return commGrid; }
	virtual SpParMatrix<T> * SubsRefCol (const vector<ITYPE> & ci) const;
	virtual ofstream& put(ofstream& outfile) const;

	virtual ITYPE getlocalrows() const { return spSeq->getrows(); }
	virtual ITYPE getlocalcols() const { return spSeq->getcols();} 
	virtual ITYPE getlocalnnz() const { return spSeq->getnzmax(); }

	virtual void operator+= (const SpParMatrix<T> & A) { cout << "Just dummy for now" << endl; };


private:
	static void SetWindows(MPI_Comm & comm1d, SparseDColumn<T> & Matrix, SpWins & wins);
	static void GetSetSizes(ITYPE index, SparseDColumn<T> & Matrix, SpSizes & sizes, MPI_Comm & comm1d);

	shared_ptr< CommGrid > commGrid; 
	shared_ptr< SparseDColumn<T> > spSeq;
	
	template <typename U>
	friend ofstream& operator<< (ofstream& outfile, const SparseOneSidedMPI<U> & s);	
};

#include "SparseOneSidedMPI.cpp"
#endif
