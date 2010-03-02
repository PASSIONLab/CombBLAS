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
#include <cassert>
#include "SpDefs.h"
#include "SpHelper.h"
#include "StackEntry.h"
#include "MemoryPool.h"
#include "promote.h"

using namespace std;

template <class IT, class NT>
class Dcsc
{
public:
	Dcsc ();
	Dcsc (IT nnz, IT nzcol);
	Dcsc (IT nnz, IT nzcol, MemoryPool * pool);		//!< Placement constructor

	Dcsc (IT nnz, const vector<IT> & indices, bool isRow); 	//!< Create a logical matrix from (row/column) indices vector
	Dcsc (StackEntry<NT, pair<IT,IT> > * multstack, IT mdim, IT ndim, IT nnz);

	Dcsc (const Dcsc<IT,NT> & rhs);				// copy constructor
	Dcsc<IT,NT> & operator=(const Dcsc<IT,NT> & rhs);	// assignment operator
	Dcsc<IT,NT> & operator+=(const Dcsc<IT,NT> & rhs);	// add and assign operator
	~Dcsc();
	
	bool operator==(const Dcsc<IT,NT> & rhs);	
	template <typename NNT> operator Dcsc<IT,NNT>() const;	//<! numeric type conversion
	
	void EWiseMult(const Dcsc<IT,NT> & rhs, bool exclude); 
	void EWiseScale(NT ** scaler);				//<! scale elements of "this" with the elements dense rhs matrix
	
	template <typename IU, typename NU1, typename NU2>
	friend Dcsc<IU, typename promote_trait<NU1,NU2>::T_promote> EWiseMult(const Dcsc<IU,NU1> & A, const Dcsc<IU,NU2> & B, bool exclude);

	template <typename _UnaryOperation>
	void Apply(_UnaryOperation __unary_op)
	{	
		transform(numx, numx+nz, numx, __unary_op);
	}

	template <typename _UnaryOperation>
	void Prune(_UnaryOperation __unary_op);

	IT AuxIndex(IT colind, bool & found, IT * aux, IT csize) const;
	void Split(Dcsc<IT,NT> * & A, Dcsc<IT,NT> * & B, IT cut); 	
	void Merge(const Dcsc<IT,NT> * Adcsc, const Dcsc<IT,NT> * B, IT cut);		

	IT ConstructAux(IT ndim, IT * & aux) const;
	void Resize(IT nzcnew, IT nznew);
	void FillColInds(const vector<IT> & colnums, vector< pair<IT,IT> > & colinds, IT * aux, IT csize) const;

	Dcsc<IT,NT> & AddAndAssign (StackEntry<NT, pair<IT,IT> > * multstack, IT mdim, IT ndim, IT nnz);

	template <typename _BinaryOperation>
	void UpdateDense(NT ** array, _BinaryOperation __binary_op) const;	// update dense 2D array's entries with __binary_op using elements of "this"

	IT * cp;		//!<  The master array, size nzc+1 (keeps column pointers)
	IT * jc ;		//!<  col indices, size nzc
	IT * ir ;		//!<  row indices, size nz
	NT * numx;		//!<  generic values, size nz

	IT nz;
	IT nzc;			//!<  number of columns with at least one non-zero in them
private:
	void getindices (StackEntry<NT, pair<IT,IT> > * multstack, IT & rindex, IT & cindex, IT & j, IT nnz);

	//! Special memory management functions, both respect the memory pool
	void * mallocarray (size_t size) const;
	void deletearray(void * array, size_t size) const;

	MemoryPool * pool;
	const static IT zero;
};

#include "dcsc.cpp"	
#endif


