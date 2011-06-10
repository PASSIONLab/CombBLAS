/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.2 -------------------------------------------------*/
/* date: 10/06/2011 --------------------------------------------*/
/* authors: Aydin Buluc (abuluc@lbl.gov), Adam Lugowski --------*/
/****************************************************************/
/*
Copyright (c) 2011, Aydin Buluc

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

#ifndef _FULLY_DIST_SP_VEC_H_
#define _FULLY_DIST_SP_VEC_H_

#include <iostream>
#include <vector>
#include <utility>
#ifdef NOTR1
	#include <boost/tr1/memory.hpp>
	#include <boost/tr1/unordered_map.hpp>
#else
	#include <tr1/memory>
	#include <tr1/unordered_map>
#endif
#include "CommGrid.h"
#include "promote.h"
#include "SpParMat.h"
#include "FullyDist.h"
#include "Exception.h"
#include "OptBuf.h"
using namespace std;
using namespace std::tr1;

template <class IT, class NT, class DER>
class SpParMat;

template <class IT>
class DistEdgeList;

template <class IU, class NU>
class FullyDistVec;

template <class IU, class NU>
class SparseVectorLocalIterator;

/** 
  * A sparse vector of length n (with nnz <= n of them being nonzeros) is distributed to 
  * "all the processors" in a way that "respects ordering" of the nonzero indices
  * Example: x = [5,1,6,2,9] for nnz(x)=5 and length(x)=12 
  *	we use 4 processors P_00, P_01, P_10, P_11
  * 	Then P_00 owns [1,2] (in the range [0,...,2]), P_01 ow`ns [5] (in the range [3,...,5]), and so on.
  * In the case of A(v,w) type sparse matrix indexing, this doesn't matter because n = nnz
  * 	After all, A(v,w) will have dimensions length(v) x length (w) 
  * 	v and w will be of numerical type (NT) "int" and their indices (IT) will be consecutive integers 
  * It is possibly that nonzero counts are distributed unevenly
  * Example: x=[1,2,3,4,5] and length(x) = 20, then P_00 would own all the nonzeros and the rest will hold empry vectors
  * Just like in SpParMat case, indices are local to processors (they belong to range [0,...,length-1] on each processor)
  * \warning Always create vectors with the right length, setting elements won't increase its length (similar to operator[] on std::vector)
 **/
template <class IT, class NT>
class FullyDistSpVec: public FullyDist<IT,NT,typename disable_if< is_boolean<NT>::value, NT >::type>
{
public:
	FullyDistSpVec ( );
	FullyDistSpVec ( IT glen );
	FullyDistSpVec ( shared_ptr<CommGrid> grid);
	FullyDistSpVec ( shared_ptr<CommGrid> grid, IT glen);

	FullyDistSpVec (const FullyDistVec<IT,NT> & rhs);					// Conversion copy-constructor

	//! like operator=, but instead of making a deep copy it just steals the contents. 
	//! Useful for places where the "victim" will be distroyed immediately after the call.
	void stealFrom(FullyDistSpVec<IT,NT> & victim); 
	FullyDistSpVec<IT,NT> &  operator=(const FullyDistSpVec< IT,NT > & rhs);
	FullyDistSpVec<IT,NT> &  operator=(const FullyDistVec< IT,NT > & rhs);	// convert from dense
	FullyDistSpVec<IT,NT> & operator+=(const FullyDistSpVec<IT,NT> & rhs);
	FullyDistSpVec<IT,NT> & operator-=(const FullyDistSpVec<IT,NT> & rhs);
	ifstream& ReadDistribute (ifstream& infile, int master);	

	template <typename NNT> operator FullyDistSpVec< IT,NNT > () const	//!< Type conversion operator
	{
		FullyDistSpVec<IT,NNT> CVT(commGrid);
		CVT.ind = vector<IT>(ind.begin(), ind.end());
		CVT.num = vector<NNT>(num.begin(), num.end());
		CVT.glen = glen;
		CVT.NOT_FOUND = NOT_FOUND;
		return CVT;
	}

	bool operator==(const FullyDistSpVec<IT,NT> & rhs) const
	{
		FullyDistVec<IT,NT> v =  *this;
		FullyDistVec<IT,NT> w =  rhs;
		return (v == w);
	}

	void PrintInfo(string vecname) const;
	void iota(IT globalsize, NT first);
	FullyDistVec<IT,NT> operator() (const FullyDistVec<IT,IT> & ri) const;	//!< SpRef (expects ri to be 0-based)
	void SetElement (IT indx, NT numx);	// element-wise assignment
	void DelElement (IT indx); // element-wise deletion
	NT operator[](IT indx) const;

	NT GetZero() const			{ return zero; }
	void SetZero(const NT& z)	{ zero = z; }

	// sort the vector itself
	// return the permutation vector (0-based)
	FullyDistSpVec<IT, IT> sort();	

	IT getlocnnz() const 
	{
		return ind.size();
	}
	IT getnnz() const
	{
		IT totnnz = 0;
		IT locnnz = ind.size();
		(commGrid->GetWorld()).Allreduce( &locnnz, & totnnz, 1, MPIType<IT>(), MPI::SUM); 
		return totnnz;
	}
	using FullyDist<IT,NT,typename disable_if< is_boolean<NT>::value, NT >::type>::LengthUntil;
	using FullyDist<IT,NT,typename disable_if< is_boolean<NT>::value, NT >::type>::MyLocLength;
	using FullyDist<IT,NT,typename disable_if< is_boolean<NT>::value, NT >::type>::MyRowLength;
	using FullyDist<IT,NT,typename disable_if< is_boolean<NT>::value, NT >::type>::TotalLength;
	using FullyDist<IT,NT,typename disable_if< is_boolean<NT>::value, NT >::type>::Owner;
	using FullyDist<IT,NT,typename disable_if< is_boolean<NT>::value, NT >::type>::RowLenUntil;

	void setNumToInd()
	{
		IT offset = LengthUntil();
		IT spsize = ind.size();
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for(IT i=0; i< spsize; ++i)
			num[i] = ind[i] + offset;
	}

	template <typename _Predicate>
	IT Count(_Predicate pred) const;	//!< Return the number of elements for which pred is true

	template <typename _UnaryOperation>
	void Apply(_UnaryOperation __unary_op)
	{
		transform(num.begin(), num.end(), num.begin(), __unary_op);
	}

	template <typename _BinaryOperation>
	NT Reduce(_BinaryOperation __binary_op, NT init);
	
	template <typename _BinaryOperation, typename _UnaryOperation>
	NT Reduce(_BinaryOperation __binary_op, NT default_val, _UnaryOperation __unary_op);

	void DebugPrint();
	shared_ptr<CommGrid> getCommGrid() { return commGrid; }
	NT NOT_FOUND; 

protected:
	using FullyDist<IT,NT,typename disable_if< is_boolean<NT>::value, NT >::type>::glen; 
	using FullyDist<IT,NT,typename disable_if< is_boolean<NT>::value, NT >::type>::commGrid; 

private:
	vector< IT > ind;	// ind.size() give the number of nonzeros
	vector< NT > num;
	NT zero;

	template <class IU, class NU>
	friend class FullyDistSpVec;

	template <class IU, class NU>
	friend class FullyDistVec;
	
	template <class IU, class NU, class UDER>
	friend class SpParMat;
	
	template <class IU, class NU>
	friend class SparseVectorLocalIterator;

	template <typename SR, typename IU, typename NUM, typename NUV, typename UDER> 
	friend FullyDistSpVec<IU,typename promote_trait<NUM,NUV>::T_promote> 
	SpMV (const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,NUV> & x );

	template <typename SR, typename IU, typename NUM, typename UDER> 
	friend FullyDistSpVec<IU,typename promote_trait<NUM,IU>::T_promote>  
	SpMV (const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,IU> & x, bool indexisvalue);

	template <typename SR, typename IU, typename NUM, typename UDER> 
	friend FullyDistSpVec<IU,typename promote_trait<NUM,IU>::T_promote>  
	SpMV (const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,IU> & x, bool indexisvalue, OptBuf<IU, typename promote_trait<NUM,IU>::T_promote > & optbuf);
	
	template <typename _BinaryOperation, typename IU, typename NUM, typename NUV, typename UDER> 
	friend void
	ColWiseApply (const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,NUV> & x, _BinaryOperation __binary_op);

	template <typename IU, typename NU1, typename NU2>
	friend FullyDistSpVec<IU,typename promote_trait<NU1,NU2>::T_promote> 
	EWiseMult (const FullyDistSpVec<IU,NU1> & V, const FullyDistVec<IU,NU2> & W , bool exclude, NU2 zero);

	template <typename IU>
	friend void RandPerm(FullyDistSpVec<IU,IU> & V); 	// called on an existing object, randomly permutes it
	
	template <typename IU>
	friend void RenameVertices(DistEdgeList<IU> & DEL);
};

#include "FullyDistSpVec.cpp"
#endif

