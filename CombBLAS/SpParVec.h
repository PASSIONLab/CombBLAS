#ifndef _SP_PAR_VEC_H_
#define _SP_PAR_VEC_H_

#include <iostream>
#include <vector>
#include <utility>
#ifndef _SP_PAR_VEC_H
#define _SP_PAR_VEC_H

#include <tr1/memory>
#include "CommGrid.h"

using namespace std;
using namespace std::tr1;


template <class IT, class NT>
class SpParVec
{
public:
	SpParVec ( shared_ptr< CommGrid > grid);
	
	SpParVec<T> & operator+=(const SpParVec<T> & rhs);
	
	//SpParVec<T> & operator+=(const MMmul< SpParMatrix<T>, SpParVec<T> > & matmul);	

private:
	shared_ptr< CommGrid > commGrid;
	vector< pair<IT, NT> > arr;
	bool diagonal;
};

#include "SpParVec.cpp"
#endif

