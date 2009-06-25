/****************************************************************/
/* Sequential and Parallel Sparse Matrix Multiplication Library */
/* version 2.3 --------------------------------------------------/
/* date: 01/18/2009 ---------------------------------------------/
/* author: Aydin Buluc (aydin@cs.ucsb.edu) ----------------------/
/****************************************************************/

#ifndef _SP_PAR_MPI2_H_
#define _SP_PAR_MPI2_H_

#include <iostream>
#include <fstream>
#include <cmath>
#include <mpi.h>
#include <tr1/memory>	// for shared_ptr

#include "SpMat.h"
#include "SpTuples.h"
#include "SpDCCols.h"
#include "CommGrid.h"
#include "DataTypeConvert.h"
#include "LocArr.h"

using namespace std;

/**
  * This class implements an asynchronous 2D algorithm, in the sense that there is no notion of stages.
  * \n The process that completes its submatrix update, requests subsequent matrices from their owners w/out waiting to sychronize with other processors
  * \n This partially remedies the severe load balancing problem in sparse matrices. 
  * \n The class uses MPI-2 to achieve one-sided asynchronous communication
  * \n The algorithm treats each submatrix as a single block
  * \n Local data structure can be any SpMat that has a constructor with array sizes and getarrs() member 
  */
template <class IT, class NT, class DER>
class SpParMPI2
{
public:
	// Constructors
	SpParMPI2 () {};
	SpParMPI2 (ifstream & input, MPI::IntraComm & world);
	SpParMPI2 (SpMat<IT,NT,DER> * myseq, MPI::IntraComm & world);	// ABAB: Provide !
	SpParMPI2 (SpMat<IT,NT,DER> * myseq, CommGrid * grid);		// ABAB: Provide !

	SpParMPI2 (const SpParMPI2< IT,NT,DER > & rhs);				// copy constructor
	SpParMPI2< IT,NT,DER > & operator=(const SpParMPI2< IT,NT,DER > & rhs);	// assignment operator
	SpParMPI2< IT,NT,DER > & operator+=(const SpParMPI2< IT,NT,DER > & rhs);
	~SpParMPI2 ();							// ABAB: Provide !

	//! operator* should alter the second matrix B during the computation, due to memory limitations.
	//! It transposes the second matrix before the multiplications start, therefore it has no memory to store the old version
	//! After the multiplications are done, B is correctly restored back.
	template <typename U> 
	friend const SpParMPI2<U> operator* (const SpParMPI2<U> & A, const SpParMPI2<U> & B );		// ABAB: Provide (should accept a Semiring as well)

	IT getnrows() const;
	IT getncols() const;
	IT getnnz() const;	
	
	shared_ptr<CommGrid> getcommgrid() const { return commGrid; }
	SpParMatrix<IT,NT,DER> * SubsRefCol (const vector<ITYPE> & ci) const;	// ABAB: change with a real indexing as in SpMat !
	ofstream& put(ofstream& outfile) const;

	IT getlocalrows() const { return spSeq->getrows(); }
	IT getlocalcols() const { return spSeq->getcols();} 
	IT getlocalnnz() const { return spSeq->getnzmax(); }

private:
	// Static functions that do not need access to "this" pointer
	static void GetSetSizes(ITYPE index, SparseDColumn<T> & Matrix, SpSizes & sizes, MPI_Comm & comm1d);

	const static IT zero = static_cast<IT>(0);
	CommGrid * commGrid; 
	SpMat<IT, NT, DER> * spSeq;
	
	template <typename U>
	friend ofstream& operator<< (ofstream& outfile, const SpParMPI2<U> & s);	
};

#include "SpParMPI2.cpp"
#endif
