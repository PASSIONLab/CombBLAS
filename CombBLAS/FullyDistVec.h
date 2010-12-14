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

template <class IT, class NT>
class FullyDistVec: public FullyDist<IT,NT>
{
public:
	FullyDistVec ( );
	FullyDistVec ( NT id) ;
	FullyDistVec ( IT globallen, NT initval, NT id); 
	FullyDistVec ( shared_ptr<CommGrid> grid, NT id);
	FullyDistVec ( shared_ptr<CommGrid> grid, IT globallen, NT initval, NT id);
	
	ifstream& ReadDistribute (ifstream& infile, int master);
	FullyDistVec<IT,NT> & operator=(const FullyDistSpVec<IT,NT> & rhs);		//!< FullyDistSpVec->FullyDistVec conversion operator
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
	
	void iota(IT globalsize, NT first);
	void RandPerm();	// randomly permute the vector

	using FullyDist<IT,NT>::LengthUntil;
	using FullyDist<IT,NT>::TotalLength;
	using FullyDist<IT,NT>::Owner;
	using FullyDist<IT,NT>::MyLocLength;
	IT LocArrSize() const { return arr.size(); }	// = MyLocLength() once arr is resize
	
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

protected:
	using FullyDist<IT,NT>::glen; 
	using FullyDist<IT,NT>::commGrid; 

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


