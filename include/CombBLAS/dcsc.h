/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.6 -------------------------------------------------*/
/* date: 6/15/2017 ---------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc  --------------------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2017, The Regents of the University of California

 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */


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

namespace combblas {

template <class IT, class NT>
class Dcsc
{
public:
    typedef NT value_type;
    typedef IT index_type;
	Dcsc ();
	Dcsc (IT nnz, IT nzcol);

	Dcsc (IT nnz, const std::vector<IT> & indices, bool isRow); 	//!< Create a logical matrix from (row/column) indices vector
	Dcsc (StackEntry<NT, std::pair<IT,IT> > * multstack, IT mdim, IT ndim, IT nnz);
    Dcsc (IT * colptrs, IT * rowinds, NT * vals, IT ncols, IT nonzeros);           // CSC -> DCSC constructor (except that CSC's internals are passed)

	Dcsc (const Dcsc<IT,NT> & rhs);				// copy constructor
	Dcsc<IT,NT> & operator=(const Dcsc<IT,NT> & rhs);	// assignment operator
	Dcsc<IT,NT> & operator+=(const Dcsc<IT,NT> & rhs);	// add and assign operator
	~Dcsc();
	
	bool operator==(const Dcsc<IT,NT> & rhs);	
	template <typename NNT> operator Dcsc<IT,NNT>() const;	//<! numeric type conversion
	template <typename NIT, typename NNT> operator Dcsc<NIT,NNT>() const;	//<! index+numeric type conversion
	
	void EWiseMult(const Dcsc<IT,NT> & rhs, bool exclude); 
	void SetDifference(const Dcsc<IT,NT> & rhs);		//<! Aydin (June 2021): generalize this to any rhs NT type; as it isn't used anyway 
	void EWiseScale(NT ** scaler);				//<! scale elements of "this" with the elements dense rhs matrix
	
	template <typename IU, typename NU1, typename NU2>
	friend Dcsc<IU, typename promote_trait<NU1,NU2>::T_promote> EWiseMult(const Dcsc<IU,NU1> & A, const Dcsc<IU,NU2> * B, bool exclude);	// Note that the second parameter is a POINTER

	template <typename IU, typename NU1, typename NU2>
	friend Dcsc<IU, typename promote_trait<NU1,NU2>::T_promote> SetDifference(const Dcsc<IU,NU1> & A, const Dcsc<IU,NU2> * B);	// Note that the second parameter is a POINTER

	template <typename _UnaryOperation>
	void Apply(_UnaryOperation __unary_op)
	{	
		//transform(numx, numx+nz, numx, __unary_op);
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(IT i=0; i < nz; ++i)
            numx[i] = __unary_op(numx[i]);

	}

	template <typename _UnaryOperation, typename GlobalIT>
	Dcsc<IT,NT>* PruneI(_UnaryOperation __unary_op, bool inPlace, GlobalIT rowOffset, GlobalIT colOffset);
	template <typename _UnaryOperation>
	Dcsc<IT,NT>* Prune(_UnaryOperation __unary_op, bool inPlace);
    template <typename _BinaryOperation>
    Dcsc<IT,NT>* PruneColumn(NT* pvals, _BinaryOperation __binary_op, bool inPlace);
    template <typename _BinaryOperation>
    Dcsc<IT,NT>* PruneColumn(IT* pinds, NT* pvals, _BinaryOperation __binary_op, bool inPlace);

    void PruneColumnByIndex(const std::vector<IT>& ci);

	IT AuxIndex(const IT colind, bool & found, IT * aux, IT csize) const;
	
	void RowSplit(int numsplits);
    void ColSplit(std::vector< Dcsc<IT,NT>* > & parts, std::vector<IT> & cuts);
    void ColConcatenate(std::vector< Dcsc<IT,NT>* > & parts, std::vector<IT> & offsets);

	void Split(Dcsc<IT,NT> * & A, Dcsc<IT,NT> * & B, IT cut); 	//! \todo{special case of ColSplit, to be deprecated...}
	void Merge(const Dcsc<IT,NT> * Adcsc, const Dcsc<IT,NT> * B, IT cut);	 //! \todo{special case of ColConcatenate, to be deprecated...}

	IT ConstructAux(IT ndim, IT * & aux) const;
	void Resize(IT nzcnew, IT nznew);

	template<class VT>	
	void FillColInds(const VT * colnums, IT nind, std::vector< std::pair<IT,IT> > & colinds, IT * aux, IT csize) const;

	Dcsc<IT,NT> & AddAndAssign (StackEntry<NT, std::pair<IT,IT> > * multstack, IT mdim, IT ndim, IT nnz);

	template <typename _BinaryOperation>
	void UpdateDense(NT ** array, _BinaryOperation __binary_op) const;	// update dense 2D array's entries with __binary_op using elements of "this"
    
    //! wrap object around pre-allocated arrays (possibly RDMA registered)
    Dcsc (IT * _cp, IT * _jc, IT * _ir, NT * _numx, IT _nz, IT _nzc, bool _memowned = true)
    : cp(_cp), jc(_jc), ir(_ir), numx(_numx), nz(_nz), nzc(_nzc), memowned(_memowned) {};

	IT * cp;		//!<  The master array, size nzc+1 (keeps column pointers)
	IT * jc ;		//!<  col indices, size nzc
	IT * ir ;		//!<  row indices, size nz
	NT * numx;		//!<  generic values, size nz
	
	IT nz;
	IT nzc;			//!<  number of columns with at least one non-zero in them
    bool memowned;

private:
	void getindices (StackEntry<NT, std::pair<IT,IT> > * multstack, IT & rindex, IT & cindex, IT & j, IT nnz);
};

}

#include "dcsc.cpp"	

#endif
