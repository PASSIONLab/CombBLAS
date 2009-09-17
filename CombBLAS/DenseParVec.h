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
class DenseParVec
{
public:
	DenseParVec ( shared_ptr<CommGrid> grid, NT id);
	bool operator== (const DenseParVec<IT,NT> & rhs) const;
	ifstream& ReadDistribute (ifstream& infile, int master);
	DenseParVec< IT,NT > &  operator=(const SpParVec< IT,NT > & rhs);		// SpParVec->DenseParVec conversion operator
	
private:
	shared_ptr<CommGrid> commGrid;
	vector< NT > arr;
	bool diagonal;
	NT identity;	// the element for non-existings scalars (0.0 for a vector on Reals, +infinity for a vector on the tropical semiring) 

	template <class IU, class NU>
	friend class DenseParMat;
};

#include "DenseParVec.cpp"
#endif


