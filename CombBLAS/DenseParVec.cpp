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
DenseParVec<IT, NT>::DenseParVec (IT locallength, NT id): zero(id)
{
	commGrid.reset(new CommGrid(MPI::COMM_WORLD, 0, 0));

	if(commGrid->GetRankInProcRow() == commGrid->GetRankInProcCol())
		diagonal = true;
	else
		diagonal = false;
		
	if (diagonal)
		arr.resize(locallength, zero);
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
DenseParVec<IT, NT>::DenseParVec ( shared_ptr<CommGrid> grid, IT locallength, NT id): commGrid(grid), zero(id)
{
	if(commGrid->GetRankInProcRow() == commGrid->GetRankInProcCol())
		diagonal = true;
	else
		diagonal = false;	

	if (diagonal)
		arr.resize(locallength, zero);
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
	
	return *this;
}

template <class IT, class NT>
DenseParVec< IT,NT > &  DenseParVec<IT,NT>::operator=(const DenseParVec< IT,NT > & rhs)		// SpParVec->DenseParVec conversion operator
{
	commGrid = rhs.commGrid;
	arr = rhs.arr;
	diagonal = rhs.diagonal;
	zero = rhs.zero;
	return *this;
}

template <class IT, class NT>
DenseParVec< IT,NT > &  DenseParVec<IT,NT>::stealFrom(DenseParVec<IT,NT> & victim)		// SpParVec->DenseParVec conversion operator
{
	commGrid = victim.commGrid;
	arr.swap(victim.arr);
	diagonal = victim.diagonal;
	zero = victim.zero;
}

template <class IT, class NT>
DenseParVec< IT,NT > &  DenseParVec<IT,NT>::operator+=(const SpParVec< IT,NT > & rhs)		
{
	IT spvecsize = rhs.ind.size();
	for(IT i=0; i< spvecsize; ++i)
	{
		arr[rhs.ind[i]] += rhs.num[i];
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
bool DenseParVec<IT,NT>::operator==(const DenseParVec<IT,NT> & rhs) const
{
	ErrorTolerantEqual<NT> epsilonequal;
	int local = 1;
	if(diagonal)
	{
		local = (int) std::equal(arr.begin(), arr.end(), rhs.arr.begin(), epsilonequal );
#ifdef DEBUG
		vector<NT> diff(arr.size());
		transform(arr.begin(), arr.end(), rhs.arr.begin(), diff.begin(), minus<NT>());
		typename vector<NT>::iterator maxitr;
		maxitr = max_element(diff.begin(), diff.end()); 			
		cout << maxitr-diff.begin() << ": " << *maxitr << " where lhs: " << *(arr.begin()+(maxitr-diff.begin())) 
						<< " and rhs: " << *(rhs.arr.begin()+(maxitr-diff.begin())) << endl; 
#endif
	}
	int whole = 1;
	commGrid->GetWorld().Allreduce( &local, &whole, 1, MPI::INT, MPI::BAND);
	return static_cast<bool>(whole);	
}

template <class IT, class NT>
template <typename _Predicate>
IT DenseParVec<IT,NT>::Count(_Predicate pred) const
{
	IT local = 0;
	if(diagonal)
	{
		IT local = count_if( arr.begin(), arr.end(), pred );
	}
	IT whole = 0;
	commGrid->GetWorld().Allreduce( &local, &whole, 1, MPIType<IT>(), MPI::SUM);
	return whole;	
}

template <class IT, class NT>
template <typename _Predicate>
SpParVec<IT,IT> DenseParVec<IT,NT>::FindInds(_Predicate pred) const
{
	SpParVec<IT,IT> found(commGrid);
	if(diagonal)
	{
		IT size = arr.size();
		for(IT i=0; i<size; ++i)
		{
			if(pred(arr[i]))
			{
				found.ind.push_back(i);
				found.num.push_back(i);
			}
		}
		found.length = size;
	}
	
	return found;
}


//! Requires no communication because SpParVec (the return object)
//! is distributed based on length, not nonzero counts
template <class IT, class NT>
template <typename _Predicate>
SpParVec<IT,NT> DenseParVec<IT,NT>::Find(_Predicate pred) const
{
	SpParVec<IT,NT> found(commGrid);
	if(diagonal)
	{
		IT size = arr.size();
		for(IT i=0; i<size; ++i)
		{
			if(pred(arr[i]))
			{
				found.ind.push_back(i);
				found.num.push_back(arr[i]);
			}
		}
		found.length = size;
	}
	return found;	
}

template <class IT, class NT>
ifstream& DenseParVec<IT,NT>::ReadDistribute (ifstream& infile, int master)
{
	SpParVec<IT,NT> tmpSpVec(commGrid);
	tmpSpVec.ReadDistribute(infile, master);

	*this = tmpSpVec;
	return infile;
}

template <class IT, class NT>
void DenseParVec<IT,NT>::SetElement (IT indx, NT numx)
{
	MPI::Intracomm DiagWorld = commGrid->GetDiagWorld();
	if(DiagWorld != MPI::COMM_NULL) // Diagonal processors only
	{
		int dgrank = DiagWorld.Get_rank();
		int nprocs = DiagWorld.Get_size();
		IT n_perproc = getTotalLength() / nprocs;
		IT offset = dgrank * n_perproc;
		
		if (n_perproc == 0) {
			cout << "DenseParVec::SetElement can't be called on an empty vector." << endl;
			return;
		}
		
		IT owner = (indx-1) / n_perproc;	
		IT rec = (owner < nprocs-1) ? owner : nprocs-1;	// find its owner 

		//cout << "rank " << dgrank << ". nprocs " << nprocs << ".  n_perproc " << n_perproc;
		//cout << ".  offset " << offset << ".  owner " << owner << ".   size " << arr.size();
		//cout << ".  localind " << (indx-1-offset) << endl;		

		if(rec == dgrank) // this process is the owner
		{
			IT locindx = indx-1-offset;
			
			if (locindx > arr.size()-1)
			{
				cout << "DenseParVec::SetElement cannot expand array" << endl;
			}
			else if (locindx < 0)
			{
				cout << "DenseParVec::SetElement local index < 0" << endl;
			}
			else
			{
				arr[locindx] = numx;
			}
		}
	}
}

template <class IT, class NT>
NT DenseParVec<IT,NT>::GetElement (IT indx)
{
	NT ret;
	
	int owner = 0;
	MPI::Intracomm DiagWorld = commGrid->GetDiagWorld();
	if(DiagWorld != MPI::COMM_NULL) // Diagonal processors only
	{
		int dgrank = DiagWorld.Get_rank();
		int nprocs = DiagWorld.Get_size();
		IT n_perproc = getTotalLength() / nprocs;
		IT offset = dgrank * n_perproc;
		
		if (n_perproc == 0) {
			cout << "DenseParVec::GetElement can't be called on an empty vector." << endl;
			return ret;
		}
		
		owner = (indx-1) / n_perproc;	
		IT rec = (owner < nprocs-1) ? owner : nprocs-1;	// find its owner 
		
		if(rec == dgrank) // this process is the owner
		{
			IT locindx = indx-1-offset;

			if (locindx > arr.size()-1)
			{
				cout << "DenseParVec::GetElement cannot expand array" << endl;
			}
			else if (locindx < 0)
			{
				cout << "DenseParVec::GetElement local index < 0" << endl;
			}
			else
			{
				ret = arr[locindx];
			}

		}
	}
	
	int worldowner = (commGrid->GetGridCols()+1)*owner;
	
	(commGrid->GetWorld()).Bcast(&worldowner, 1, MPIType<int>(), 0);
	(commGrid->GetWorld()).Bcast(&ret, 1, MPIType<NT>(), worldowner);
	return ret;
}

template <class IT, class NT>
void DenseParVec<IT,NT>::DebugPrint()
{
	MPI::Intracomm DiagWorld = commGrid->GetDiagWorld();
	if(DiagWorld != MPI::COMM_NULL) // Diagonal processors only
	{
		int dgrank = DiagWorld.Get_rank();
		int nprocs = DiagWorld.Get_size();

		int64_t* all_nnzs = new int64_t[nprocs];
		
		all_nnzs[dgrank] = arr.size();
		DiagWorld.Allgather(MPI::IN_PLACE, 0, MPI::DATATYPE_NULL, all_nnzs, 1, MPIType<int64_t>());
		int64_t offset = 0;
		
		for (int i = 0; i < nprocs; i++)
		{
			if (i == dgrank) {
				cout << "stored on proc " << dgrank << ":" << endl;
				
				for (int j = 0; j < arr.size(); j++)
				{
					cout << "[" << (j+offset+1) << "] = " << arr[j] << endl;
				}
			}
			offset += all_nnzs[i];
			DiagWorld.Barrier();
		}
		DiagWorld.Barrier();
		if (dgrank == 0)
			cout << "total size: " << offset << endl;
		DiagWorld.Barrier();
	}
}

