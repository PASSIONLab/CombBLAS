/****************************************************************/
/* Sequential and Parallel Sparse Matrix Multiplication Library */
/* version 2.3 --------------------------------------------------/
/* date: 01/18/2009 ---------------------------------------------/
/* author: Aydin Buluc (aydin@cs.ucsb.edu) ----------------------/
/****************************************************************/

#ifndef _SP_PAR_MPI2_H
#define _SP_PAR_MPI2_H

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
#include <boost/shared_ptr.hpp>

using namespace std;
using namespace boost;


/**
  * This class implements an asynchronous 2D algorithm, in the sense that there is no notion of stages.
  * \n The process that completes its submatrix update requests subsequent matrices from their owners without waiting to sychronize with other processors
  * \n This partially remedies the severe load balancing problem in sparse matrices. 
  * \n The class uses MPI-2 to achieve one-sided asynchronous communication
  * \n The algorithm treats each submatrix as a single block
  * \n Local data structure can be any SpMat that has a constructor with array sizes and getarrs() member 
  */
template <class IT, class NT, class DER>
class SpParMPI2: public SpParMat<IT, NT>
{
public:
	// Constructors
	SpParMPI2 (MPI_Comm world) 
	{
		commGrid.reset(new CommGrid(world, 0, 0)); 	
	};
	SpParMPI2 (ifstream & input, MPI_Comm world);
	SpParMPI2 (shared_ptr< SparseDColumn<T> > myseq, MPI_Comm world):spSeq(myseq) 
	{	
		commGrid.reset(new CommGrid(world, 0, 0));
	};
	SpParMPI2 (shared_ptr< SparseDColumn<T> > myseq,shared_ptr< CommGrid > grid): spSeq(myseq), commGrid(grid) {};

	SpParMPI2 (const SpParMPI2<T> & rhs);			// copy constructor
	SpParMPI2<T> & operator=(const SpParMPI2<T> & rhs);	// assignment operator
	SpParMPI2<T> & operator+=(const SpParMPI2<T> & rhs);

	//! No exclicit destructor needed as smart pointers take care of calling the appropriate destructors 

	//! operator* should alter the second matrix B during the computation, due to memory limitations.
	//! It transposes the second matrix before the multiplications start, therefore it has no memory to store the old version
	//! After the multiplications are done, B is correctly restored back.
	template <typename U> 
	friend const SpParMPI2<U> operator* (const SpParMPI2<U> & A, SpParMPI2<U> & B );

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

	const static IT zero = static_cast<IT>(0);
	CommGrid * commGrid; 
	SpMat<IT, NT, DER> * spSeq;
	
	template <typename U>
	friend ofstream& operator<< (ofstream& outfile, const SpParMPI2<U> & s);	
};

#include "SpParMPI2.cpp"
#endif
