#include "SpParVec.h"

template <class IT, class NT>
SpParVec<IT, NT>::SpParVec ( shared_ptr<CommGrid> grid): commGrid(grid) 
{
	if(commGrid->GetRankInProcRow() == commGrid->GetRankInProcCol())
		diagonal = true;
	else
		diagonal = false;	
};
	

template <class IT, class NT>
SpParVec<IT,NT> & SpParVec<IT, NT>::operator+=(const SpParVec<IT,NT> & rhs)
{
	if(this != &rhs)		
	{	
		if(diagonal)	// Only the diagonal processors hold values
		{
			vector< pair<IT, NT> > narr;
			IT lsize = arr.size();
			IT rsize = rhs.arr.size();
			narr.reserve(lsize+rsize);

			IT i=0, j=0;
			while(i < lsize && j < rsize)
			{
				// assignment won't change the size of vector, push_back is necessary
				if(arr[i].first > rhs.arr[j].first)
				{	
					narr.push_back( rhs.arr[j++] );
				}
				else if(arr[i].first < rhs.arr[j].first)
				{
					narr.push_back( arr[i++] );
				}
				else
				{
					narr.push_back( make_pair(arr[i].first, arr[i++].second + rhs.arr[j++].second) );
				}
			}
			arr.swap(narr);		// arr will contain the elements of narr with capacity shrunk-to-fit size
		} 	
	}	
	return *this;
};	

