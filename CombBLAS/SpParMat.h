/****************************************************************/
/* Sequential and Parallel Sparse Matrix Multiplication Library */
/* version 2.3 --------------------------------------------------/
/* date: 01/18/2009 ---------------------------------------------/
/* author: Aydin Buluc (aydin@cs.ucsb.edu) ----------------------/
/****************************************************************/

#ifndef _SP_PAR_MATRIX_H
#define _SP_PAR_MATRIX_H


#include <iostream>
#include <vector>
#include <utility>
#include <mpi.h>
#include "SpDefines.h"

using namespace std;

template <class T> 
class SparseOneSidedMPI;


// The Abstract Base class for all derived parallel sparse matrix classes
template <class T>
class SpParMatrix
{
public:
	// Constructors
	SpParMatrix () {}

	// Virtual destructor
	virtual ~SpParMatrix(){};

	virtual ofstream& put(ofstream& outfile) const { return outfile; };
	virtual void operator+= (const SpParMatrix<T> & A) = 0;


	virtual SpParMatrix<T> * SubsRefCol (const vector<ITYPE> & ci) const = 0;
	virtual shared_ptr<CommGrid> getcommgrid() const = 0;
	virtual ITYPE getrows() const = 0 ;
	virtual ITYPE getcols() const = 0 ;
	virtual ITYPE getnnz() const = 0 ;

	virtual ITYPE getlocalrows() const = 0 ;
	virtual ITYPE getlocalcols() const = 0 ;
	virtual ITYPE getlocalnnz() const = 0 ;

protected:
	template <typename U>
	friend ofstream& operator<< (ofstream& outfile, const SpParMatrix<U> & s);

	//template <typename U> 
	//friend const SpParMatrix<U> operator* (const SpParMatrix<U> & A, SpParMatrix<U> & B );
};

template <typename U>
ofstream& operator<<(ofstream& outfile, const SpParMatrix<U> & s)
{
	return s.put(outfile) ;	// use the right put() function
}



#endif

