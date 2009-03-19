#ifndef _SP_PAR_VEC_H_
#define _SP_PAR_VEC_H_

#include <iostream>
#include <vector>
#include <utility>
#include "CommGrid.h"
#include <boost/shared_ptr.hpp>

using namespace std;
using namespace boost;


template <class T>
class SpParVec
{
public:
	SpParVec ( shared_ptr< CommGrid > grid): commGrid(grid) 
	{
		int rowrank, colrank;
		MPI_Comm_rank(grid->rowWorld, &rowrank);
		MPI_Comm_rank(grid->colWorld, &colrank);
		if(rowrank == colrank)
			diagonal = true;
		else
			diagonal = false;	
	};
	
	
	SpParVec<T> & operator+=(const SpParVec<T> & rhs)
	{
		if(diagonal)
		{
			vector< pair<ITYPE, T> > narr;
			ITYPE lsize = arr.size();
			ITYPE rsize = rhs.arr.size();
			narr.reserve(lsize+rsize);

			ITYPE i =0, j=0, k=0;
			while(i < lsize && j < rsize)
			{
				if(arr[i].first > rhs.arr[j].first)	
					narr[k++] = rhs.arr[j++];
				else if(arr[i].first < rhs.arr[j].first)
					narr[k++] = arr[i++];
				else
					narr[k++] = pair<ITYPE, T>(arr[i].first, arr[i++].second + rhs.arr[j++].second);
			}
			narr = arr;
		}	
	};	
	
	//SpParVec<T> & operator+=(const MMmul< SpParMatrix<T>, SpParVec<T> > & matmul);	

private:
	shared_ptr< CommGrid > commGrid;
	vector< pair<ITYPE, T> > arr;
	bool diagonal;
};

#endif

