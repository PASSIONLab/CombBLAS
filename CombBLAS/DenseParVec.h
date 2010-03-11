#ifndef _DENSE_PAR_VEC_H_
#define _DENSE_PAR_VEC_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <utility>

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

template <class IT, class NT>
class DenseParVec
{
public:
	DenseParVec ( );
	DenseParVec ( shared_ptr<CommGrid> grid, NT id);
	bool operator== (const DenseParVec<IT,NT> & rhs) const;
	ifstream& ReadDistribute (ifstream& infile, int master);
	DenseParVec<IT,NT> &  operator=(const SpParVec<IT,NT> & rhs);		//!< SpParVec->DenseParVec conversion operator
	DenseParVec<IT,NT> & operator+=(const DenseParVec<IT,NT> & rhs);
	DenseParVec<IT,NT> & operator-=(const DenseParVec<IT,NT> & rhs);

	template <typename _UnaryOperation>
	void Apply(_UnaryOperation __unary_op)
	{	
		transform(arr.begin(), arr.end(), arr.begin(), __unary_op);
	}	
	
	void PrintToFile(string prefix, ofstream & output)
	{
		commGrid->OpenDebugFile(prefix, output);
		copy(arr.begin(), arr.end(), ostream_iterator<NT> (output, "\n"));
		output.close();
	}
	
	template <typename _BinaryOperation>
	NT Reduce(_BinaryOperation __binary_op, NT identity);
			
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
};

#include "DenseParVec.cpp"
#endif


