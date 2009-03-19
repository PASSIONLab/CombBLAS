/****************************************************************/
/* Sequential and Parallel Sparse Matrix Multiplication Library */
/* version 2.3 --------------------------------------------------/
/* date: 01/18/2009 ---------------------------------------------/
/* design detail: AUX array is not created by the constructor, 	*/
/* instead it is generated on demand only for:			*/	
/*		- Col Indexing					*/
/*		- Algorithm 2					*/
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

	void DeleteAux();
	void ConstructAux(IT ndim);
	void Resize(IT nzcnew, IT nznew);
	Dcsc<IT,NT> & AddAndAssign (StackEntry<NT, pair<IT,IT> > * multstack, IT mdim, IT ndim, IT nnz);
	
	IT * aux;		//!<  AUX array, keeps pointers to MAS, size: colchunks+1 
	IT * mas;		//!<  The master array, size nzc+1 (keeps column pointers)
	IT * jc ;		//!<  col indices, size nzc
	IT * ir ;		//!<  row indices, size nz
	T * numx;		//!<  generic values, size nz

	IT nz;
	IT nzc;			//!<  number of columns with at least one non-zero in them
	float cf;		//!<  Compression factor, size: (n+1)/nzc
	IT colchunks;
private:
	void getindices (StackEntry<NT, pair<IT,IT> > * multstack, IT & rindex, IT & cindex, IT & j, IT nnz);

	//! Special memory management functions, both respect the memory pool
	void * mallocarray (size_t size);
	void deletearray(void * array, size_t size);

	MemoryPool * pool;
};





#include "dcsc.cpp"	
#endif


