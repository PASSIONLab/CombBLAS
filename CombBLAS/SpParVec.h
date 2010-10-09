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


/** 
  * A sparse vector of length n (with nnz <= n of them being nonzeros) is distributed to diagonal processors 
  * This is done without any reference to ordering of the nonzero indices
  * Example: x = [5,1,6,2,9] and we use 4 processors P_00, P_01, P_10, P_11
  * Then P_00 owns [5,1,6] and P_11 owns [2,9]
  * ABAB: As of 08/2010, this class is used only to generate random permutations  
  **/
  
template <class IT, class NT>
class SpParVec
{
public:
	SpParVec ( shared_ptr<CommGrid> grid);
	SpParVec<IT,NT> & operator+=(const SpParVec<IT,NT> & rhs);
	ifstream& ReadDistribute (ifstream& infile, int master);	
	
	IT getnnz() const
	{
		IT totnnz = 0;
		IT locnnz = arr.length();
		(commGrid->GetDiagWorld()).Allreduce( &locnnz, & totnnz, 1, MPIType<IT>(), MPI::SUM); 
	}

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

