#include "DenseParVec.h"
#include "SpParVec.h"

template <class IT, class NT>
DenseParVec<IT, NT>::DenseParVec ( shared_ptr<CommGrid> grid, NT id): commGrid(grid), identity(id)
{
	if(commGrid->GetRankInProcRow() == commGrid->GetRankInProcCol())
		diagonal = true;
	else
		diagonal = false;	
};

template <class IT, class NT>
DenseParVec< IT,NT > &  DenseParVec<IT,NT>::operator=(const SpParVec< IT,NT > & rhs)		// SpParVec->DenseParVec conversion operator
{
	arr.resize(rhs.length);
	std::fill(arr.begin(), arr.end(), identity);	
	typename vector< pair<IT,NT> >::const_iterator it; 

	for(it = rhs.arr.begin(); it!= rhs.arr.end(); ++it) 
	{
		arr[it->first] = it->second;
	}
}

template <class IT, class NT>
bool DenseParVec<IT,NT>::operator== (const DenseParVec<IT,NT> & rhs) const
{
	ErrorTolerantEqual<NT> epsilonequal;
	//for(int i=0; i<arr.size(); ++i)
	//{
	//	if(std::abs(arr[i] - rhs.arr[i]) > EPSILON)
	//		cout << i << ": " << arr[i] << " != " << rhs.arr[i] << endl;
	//}

	int local = static_cast<int>(std::equal(arr.begin(), arr.end(), rhs.arr.begin(), epsilonequal));
	int whole = 1;
	commGrid->GetWorld().Allreduce( &local, &whole, 1, MPI::INT, MPI::BAND);
	return static_cast<bool>(whole);	
}

template <class IT, class NT>
ifstream& DenseParVec<IT,NT>::ReadDistribute (ifstream& infile, int master)
{
	SpParVec<IT,NT> tmpSpVec(commGrid);
	tmpSpVec.ReadDistribute(infile, master);

	*this = tmpSpVec;
	return infile;
}

