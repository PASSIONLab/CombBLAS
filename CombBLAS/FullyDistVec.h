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

#ifndef _FULLY_DIST_VEC_H_
#define _FULLY_DIST_VEC_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <iterator>
#ifdef NOTR1
	#include <boost/tr1/memory.hpp>
#else
	#include <tr1/memory>
#endif
#include "CommGrid.h"
#include "FullyDist.h"
#include "Exception.h"
using namespace std;
using namespace std::tr1;

template <class IT, class NT>
class FullyDistSpVec;

template <class IT, class NT, class DER>
class SpParMat;

template <class IT>
class DistEdgeList;

template <class IU, class NU>
class DenseVectorLocalIterator;

// ABAB: As opposed to SpParMat, IT here is used to encode global size and global indices;
// therefore it can not be 32-bits, in general.
template <class IT, class NT>
class FullyDistVec: public FullyDist<IT,NT, typename disable_if< is_boolean<NT>::value, NT >::type >
{
public:
	FullyDistVec ( );
	FullyDistVec ( NT id) ;
	FullyDistVec ( IT globallen, NT initval, NT id); 
	FullyDistVec ( shared_ptr<CommGrid> grid, NT id);
	FullyDistVec ( shared_ptr<CommGrid> grid, IT globallen, NT initval, NT id);
	FullyDistVec ( const FullyDistSpVec<IT, NT> & rhs ); // Sparse -> Dense conversion constructor

	template <class ITRHS, class NTRHS>
	FullyDistVec ( const FullyDistVec<ITRHS, NTRHS>& rhs ); // type converter constructor

	ifstream& ReadDistribute (ifstream& infile, int master);
	template <class ITRHS, class NTRHS>
	FullyDistVec<IT,NT> & operator=(const FullyDistVec< ITRHS,NTRHS > & rhs);	// assignment with type conversion
	FullyDistVec<IT,NT> & operator=(const FullyDistVec<IT,NT> & rhs);	//!< Actual assignment operator
	FullyDistVec<IT,NT> & operator=(const FullyDistSpVec<IT,NT> & rhs);		//!< FullyDistSpVec->FullyDistVec conversion operator
	FullyDistVec<IT,NT> & operator=(const DenseParVec<IT,NT> & rhs);		//!< DenseParVec->FullyDistVec conversion operator
	FullyDistVec<IT,NT> operator() (const FullyDistVec<IT,IT> & ri) const;	//<! subsref
	
	//! like operator=, but instead of making a deep copy it just steals the contents. 
	//! Useful for places where the "victim" will be distroyed immediately after the call.
	FullyDistVec<IT,NT> & stealFrom(FullyDistVec<IT,NT> & victim); 
	FullyDistVec<IT,NT> & operator+=(const FullyDistSpVec<IT,NT> & rhs);		
	FullyDistVec<IT,NT> & operator+=(const FullyDistVec<IT,NT> & rhs);
	FullyDistVec<IT,NT> & operator-=(const FullyDistSpVec<IT,NT> & rhs);		
	FullyDistVec<IT,NT> & operator-=(const FullyDistVec<IT,NT> & rhs);
	bool operator==(const FullyDistVec<IT,NT> & rhs) const;

	void SetElement (IT indx, NT numx);	// element-wise assignment
	NT   GetElement (IT indx) const;	// element-wise fetch
	NT operator[](IT indx) const		// more c++ like API
	{
		return GetElement(indx);
	}
	NT GetZero() const			{ return zero; }
	void SetZero(const NT& z)	{ zero = z; }
	
	void iota(IT globalsize, NT first);
	void RandPerm();	// randomly permute the vector
	FullyDistVec<IT,IT> sort();	// sort and return the permutation

	using FullyDist<IT,NT,typename disable_if< is_boolean<NT>::value, NT >::type>::LengthUntil;
	using FullyDist<IT,NT,typename disable_if< is_boolean<NT>::value, NT >::type>::TotalLength;
	using FullyDist<IT,NT,typename disable_if< is_boolean<NT>::value, NT >::type>::Owner;
	using FullyDist<IT,NT,typename disable_if< is_boolean<NT>::value, NT >::type>::MyLocLength;
	IT LocArrSize() const { return arr.size(); }	// = MyLocLength() once arr is resized
	
	template <typename _Predicate>
	FullyDistSpVec<IT,NT> Find(_Predicate pred) const;	//!< Return the elements for which pred is true

	template <typename _Predicate>
	FullyDistVec<IT,IT> FindInds(_Predicate pred) const;	//!< Return the indices where pred is true

	template <typename _Predicate>
	IT Count(_Predicate pred) const;	//!< Return the number of elements for which pred is true

	template <typename _UnaryOperation>
	void Apply(_UnaryOperation __unary_op)
	{	
		transform(arr.begin(), arr.end(), arr.begin(), __unary_op);
	}	

	template <typename _UnaryOperation>
	void Apply(_UnaryOperation __unary_op, const FullyDistSpVec<IT,NT>& mask);

	template <typename _BinaryOperation>
	void EWiseApply(const FullyDistVec<IT,NT> & other, _BinaryOperation __binary_op);
	template <typename _BinaryOperation>
	void EWiseApply(const FullyDistSpVec<IT,NT> & other, _BinaryOperation __binary_op, bool applyNulls, NT nullValue);
	
	void PrintToFile(string prefix)
	{
		ofstream output;
		commGrid->OpenDebugFile(prefix, output);
		copy(arr.begin(), arr.end(), ostream_iterator<NT> (output, " "));
		output << endl;
		output.close();
	}

	void PrintInfo(string vectorname) const;
	void DebugPrint();
	shared_ptr<CommGrid> getCommGrid() { return commGrid; }
	
	template <typename _BinaryOperation>
	NT Reduce(_BinaryOperation __binary_op, NT identity);	//! Reduce can be used to implement max_element, for instance

	template <typename _BinaryOperation, typename _UnaryOperation>
	NT Reduce(_BinaryOperation __binary_op, NT default_val, _UnaryOperation __unary_op);

	using FullyDist<IT,NT,typename disable_if< is_boolean<NT>::value, NT >::type>::glen; 
	using FullyDist<IT,NT,typename disable_if< is_boolean<NT>::value, NT >::type>::commGrid; 

private:
	vector< NT > arr;
	NT zero;	//!< the element for non-existings scalars (0.0 for a vector on Reals, +infinity for a vector on the tropical semiring) 

	template <typename _BinaryOperation>	
	void EWise(const FullyDistVec<IT,NT> & rhs,  _BinaryOperation __binary_op);

	template <class IU, class NU>
	friend class DenseParMat;

	template <class IU, class NU, class UDER>
	friend class SpParMat;

	template <class IU, class NU>
	friend class FullyDistVec;

	template <class IU, class NU>
	friend class FullyDistSpVec;
	
	template <class IU, class NU>
	friend class DenseVectorLocalIterator;

	template <typename SR, typename IU, typename NUM, typename NUV, typename UDER> 
	friend FullyDistVec<IU,typename promote_trait<NUM,NUV>::T_promote> 
	SpMV (const SpParMat<IU,NUM,UDER> & A, const FullyDistVec<IU,NUV> & x );

	template <typename IU, typename NU1, typename NU2>
	friend FullyDistSpVec<IU,typename promote_trait<NU1,NU2>::T_promote> 
	EWiseMult (const FullyDistSpVec<IU,NU1> & V, const FullyDistVec<IU,NU2> & W , bool exclude, NU2 zero);

	template <typename IU>
	friend void RenameVertices(DistEdgeList<IU> & DEL);
};

#include "FullyDistVec.cpp"
#endif


