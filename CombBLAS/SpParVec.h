#ifndef _SP_PAR_VEC_H_
#define _SP_PAR_VEC_H_

#include <iostream>
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
class SpParVec
{
public:
	SpParVec ( shared_ptr<CommGrid> grid);
	SpParVec<IT,NT> & operator+=(const SpParVec<IT,NT> & rhs);
	
private:
	shared_ptr<CommGrid> commGrid;
	vector< pair<IT, NT> > arr;
	bool diagonal;
};

#include "SpParVec.cpp"
#endif

