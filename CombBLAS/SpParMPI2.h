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
#include "SpMat.h"
#include "SpParMat.h"
#include "SpTuples.h"
#include "SpDCCols.h"
#include "CommGrid.h"
#include "SpWins.h"
#include "SpSizes.h"
#include "DataTypeConvert.h"

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
	SpParMPI2 (SpMat<IT,NT,DER> * myseq, MPI::IntraComm & world);
	SpParMPI2 (SpMat<IT,NT,DER> * myseq, CommGrid * grid);

	SpParMPI2 (const SpParMPI2< IT,NT,DER > & rhs);			// copy constructor
	SpParMPI2< IT,NT,DER > & operator=(const SpParMPI2< IT,NT,DER > & rhs);	// assignment operator
	SpParMPI2< IT,NT,DER > & operator+=(const SpParMPI2< IT,NT,DER > & rhs);
	~SpParMPI2 ();

	//! operator* should alter the second matrix B during the computation, due to memory limitations.
	//! It transposes the second matrix before the multiplications start, therefore it has no memory to store the old version
	//! After the multiplications are done, B is correctly restored back.
	template <typename U> 
	friend const SpParMPI2<U> operator* (const SpParMPI2<U> & A, const SpParMPI2<U> & B );

	IT getnrows() const;
	IT getncols() const;
	IT getnnz() const;	
	
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

	const static IT zero = static_cast<IT>(0);
	CommGrid * commGrid; 
	SpMat<IT, NT, DER> * spSeq;
	
	template <typename U>
	friend ofstream& operator<< (ofstream& outfile, const SpParMPI2<U> & s);	
};

#include "SpParMPI2.cpp"
#endif
