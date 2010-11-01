#ifndef _DENSE_PAR_VEC_H_
#define _DENSE_PAR_VEC_H_

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

using namespace std;
using namespace std::tr1;

template <class IT, class NT>
class SpParVec;

template <class IT, class NT, class DER>
class SpParMat;

template <class IT, class NT>
class DenseParVec
{
public:
	DenseParVec ( );
	DenseParVec ( shared_ptr<CommGrid> grid, NT id);
	ifstream& ReadDistribute (ifstream& infile, int master);
	DenseParVec<IT,NT> &  operator=(const SpParVec<IT,NT> & rhs);		//!< SpParVec->DenseParVec conversion operator
	DenseParVec<IT,NT> & operator+=(const DenseParVec<IT,NT> & rhs);
	DenseParVec<IT,NT> & operator-=(const DenseParVec<IT,NT> & rhs);
	bool operator==(const DenseParVec<IT,NT> & rhs) const;

	template <typename _Predicate>
	SpParVec<IT,NT> Find(_Predicate pred) const;	//!< Return the elements for which pred is true

	template <typename _Predicate>
	SpParVec<IT,IT> FindInds(_Predicate pred) const;	//!< Return the indices where pred is true

	template <typename _Predicate>
	IT Count(_Predicate pred) const;	//!< Return the number of elements for which pred is true

	template <typename _UnaryOperation>
	void Apply(_UnaryOperation __unary_op)
	{	
		transform(arr.begin(), arr.end(), arr.begin(), __unary_op);
	}	
	
	void PrintToFile(string prefix)
	{
		ofstream output;
		commGrid->OpenDebugFile(prefix, output);
		copy(arr.begin(), arr.end(), ostream_iterator<NT> (output, " "));
		output << endl;
		output.close();
	}
	
	template <typename _BinaryOperation>
	NT Reduce(_BinaryOperation __binary_op, NT identity);	//! Reduce can be used to implement max_element, for instance
			
private:
	shared_ptr<CommGrid> commGrid;
	vector< NT > arr;
	bool diagonal;
	NT zero;	//!< the element for non-existings scalars (0.0 for a vector on Reals, +infinity for a vector on the tropical semiring) 

	template <typename _BinaryOperation>	
	void EWise(const DenseParVec<IT,NT> & rhs,  _BinaryOperation __binary_op);

	template <class IU, class NU>
	friend class DenseParMat;

	template <class IU, class NU, class UDER>
	friend class SpParMat;

	template <typename SR, typename IU, typename NUM, typename NUV, typename UDER> 
	friend DenseParVec<IU,typename promote_trait<NUM,NUV>::T_promote> 
	SpMV (const SpParMat<IU,NUM,UDER> & A, const DenseParVec<IU,NUV> & x );
};

#include "DenseParVec.cpp"
#endif


