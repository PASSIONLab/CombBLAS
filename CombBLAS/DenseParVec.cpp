#include "DenseParVec.h"
#include "SpParVec.h"
#include "Operations.h"

template <class IT, class NT>
DenseParVec<IT, NT>::DenseParVec ()
{
	zero = static_cast<NT>(0);
	commGrid.reset(new CommGrid(MPI::COMM_WORLD, 0, 0));

	if(commGrid->GetRankInProcRow() == commGrid->GetRankInProcCol())
		diagonal = true;
	else
		diagonal = false;	
}

template <class IT, class NT>
DenseParVec<IT, NT>::DenseParVec ( shared_ptr<CommGrid> grid, NT id): commGrid(grid), zero(id)
{
	if(commGrid->GetRankInProcRow() == commGrid->GetRankInProcCol())
		diagonal = true;
	else
		diagonal = false;	
};

template <class IT, class NT>
template <typename _BinaryOperation>
NT DenseParVec<IT,NT>::Reduce(_BinaryOperation __binary_op, NT identity)
{
	// std::accumulate returns identity for empty sequences
	NT localsum = std::accumulate( arr.begin(), arr.end(), identity, __binary_op);

	NT totalsum = identity;
	(commGrid->GetWorld()).Allreduce( &localsum, &totalsum, 1, MPIType<NT>(), MPIOp<_BinaryOperation, NT>::op());
	return totalsum;
}

template <class IT, class NT>
DenseParVec< IT,NT > &  DenseParVec<IT,NT>::operator=(const SpParVec< IT,NT > & rhs)		// SpParVec->DenseParVec conversion operator
{
	arr.resize(rhs.length);
	std::fill(arr.begin(), arr.end(), zero);	

	IT spvecsize = rhs.ind.size();
	for(IT i=0; i< spvecsize; ++i)
	{
		arr[rhs.ind[i]] = rhs.num[i];
	}
}

/**
  * Perform __binary_op(*this, v2) for every element in rhs, *this, 
  * which are of the same size. and write the result back to *this
  */ 
template <class IT, class NT>
template <typename _BinaryOperation>	
void DenseParVec<IT,NT>::EWise(const DenseParVec<IT,NT> & rhs,  _BinaryOperation __binary_op)
{
	if(zero == rhs.zero)
	{
		transform ( arr.begin(), arr.end(), rhs.arr.begin(), arr.begin(), __binary_op );
	}
	else
	{
		cout << "DenseParVec objects have different identity (zero) elements..." << endl;
		cout << "Operation didn't happen !" << endl;
	}
};


template <class IT, class NT>
DenseParVec<IT,NT> & DenseParVec<IT, NT>::operator+=(const DenseParVec<IT,NT> & rhs)
{
	if(this != &rhs)		
	{	
		if(!(*commGrid == *rhs.commGrid)) 		
		{
			cout << "Grids are not comparable elementwise addition" << endl; 
			MPI::COMM_WORLD.Abort(GRIDMISMATCH);
		}
		else if(diagonal)	// Only the diagonal processors hold values
		{
			EWise(rhs, std::plus<NT>());
		} 	
	}	
	return *this;
};

template <class IT, class NT>
DenseParVec<IT,NT> & DenseParVec<IT, NT>::operator-=(const DenseParVec<IT,NT> & rhs)
{
	if(this != &rhs)		
	{	
		if(!(*commGrid == *rhs.commGrid)) 		
		{
			cout << "Grids are not comparable elementwise addition" << endl; 
			MPI::COMM_WORLD.Abort(GRIDMISMATCH);
		}
		else if(diagonal)	// Only the diagonal processors hold values
		{
			EWise(rhs, std::minus<NT>());
		} 	
	}	
	return *this;
};		


template <class IT, class NT>
ifstream& DenseParVec<IT,NT>::ReadDistribute (ifstream& infile, int master)
{
	SpParVec<IT,NT> tmpSpVec(commGrid);
	tmpSpVec.ReadDistribute(infile, master);

	*this = tmpSpVec;
	return infile;
}

