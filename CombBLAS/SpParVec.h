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
	ifstream& ReadDistribute (ifstream& infile, int master);	
	
private:
	shared_ptr<CommGrid> commGrid;
	vector< pair<IT, NT> > arr;	// arr.length() give the number of nonzeros
	IT length;			// actual length of the vector (including zeros)
	bool diagonal;
	const static IT zero = static_cast<IT>(0);

	template <typename IU, class NU>
	friend class DenseParVec;
	
};

#include "SpParVec.cpp"
#endif

