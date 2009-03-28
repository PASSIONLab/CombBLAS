/****************************************************************/
/* Sequential and Parallel Sparse Matrix Multiplication Library */
/* version 2.3 --------------------------------------------------/
/* date: 01/18/2009 ---------------------------------------------/
/* Works well with pre-pinned memory and memory pools	---------/
/* author: Aydin Buluc (aydin@cs.ucsb.edu) ----------------------/
/****************************************************************/

#ifndef _DCSC_H
#define _DCSC_H

#include <cstdlib>
#include <vector>
#include <limits>
#include "SpDefs.h"
#include "StackEntry.h"
#include "MemoryPool.h"

using namespace std;

template <class IT, NT>
class Dcsc
{
public:
	Dcsc ();
	Dcsc (IT nnz, IT nzcol);
	Dcsc (IT nnz, IT nzcol, MemoryPool * pool);	//!< Placement constructor

	Dcsc (IT nnz, const vector<IT> & indices, bool isRow); 	//!< Create a logical matrix from (row/column) indices vector
	Dcsc (StackEntry<T, pair<IT,IT> > * multstack, IT mdim, IT ndim, IT nnz);

	Dcsc (const Dcsc<IT,NT> & rhs);				// copy constructor
	Dcsc<IT,NT> & operator=(const Dcsc<IT,NT> & rhs);	// assignment operator
	Dcsc<IT,NT> & operator+=(const Dcsc<IT,NT> & rhs);	// add and assign operator
	~Dcsc();

	template <typename NNT>
	Dcsc<IT,NNT> ConvertNumericType();

	IT AuxIndex(IT colind, bool found, IT * aux, IT csize);
	void Split(Dcsc<IT,NT> * & A, Dcsc<IT,NT> * & B, IT cut); 	
	void Merge(const Dcsc<IT,NT> * Adcsc, const Dcsc<IT,NT> * B, IT cut);		

	IT Dcsc<IT,NT>::ConstructAux(IT ndim, IT * & aux);
	void Resize(IT nzcnew, IT nznew);
	Dcsc<IT,NT> & AddAndAssign (StackEntry<NT, pair<IT,IT> > * multstack, IT mdim, IT ndim, IT nnz);
	
	IT * cp;		//!<  The master array, size nzc+1 (keeps column pointers)
	IT * jc ;		//!<  col indices, size nzc
	IT * ir ;		//!<  row indices, size nz
	T * numx;		//!<  generic values, size nz

	IT nz;
	IT nzc;			//!<  number of columns with at least one non-zero in them
private:
	void getindices (StackEntry<NT, pair<IT,IT> > * multstack, IT & rindex, IT & cindex, IT & j, IT nnz);

	//! Special memory management functions, both respect the memory pool
	void * mallocarray (size_t size);
	void deletearray(void * array, size_t size);

	MemoryPool * pool;
	const static IT zero = static_cast<IT>(0);
};

#include "dcsc.cpp"	
#endif


