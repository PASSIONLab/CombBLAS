#include <numeric>
#include "DenseParMat.h"
#include "MPIType.h"
#include "Operations.h"

using namespace std;

template <class IT, class NT>
template <typename _BinaryOperation>
DenseParVec< IT,NT > DenseParMat<IT,NT>::Reduce(Dim dim, _BinaryOperation __binary_op, NT identity) const
{
	DenseParVec<IT,NT> parvec(commGrid, identity);

	switch(dim)
	{
		case Row:	// pack (collapse) rows 
		{
			NT * sendbuf = new NT[n];
			for(int j=0; j < n; ++j)
			{
				sendbuf[j] = identity;
				for(int i=0; i < m; ++i)
				{
					sendbuf[j] = __binary_op(array[i][j], sendbuf[j]); 
				}
			}
			NT * recvbuf = NULL;
			int root = commGrid->GetDiagOfProcCol();
			if(parvec.diagonal)
			{
				parvec.arr.resize(n);
				recvbuf = &parvec.arr[0];	
			}
			(commGrid->GetColWorld()).Reduce(sendbuf, recvbuf, n, MPIType<NT>(), MPIOp<_BinaryOperation, NT>::op(), root);
			delete sendbuf;
			break;
		}
		case Column:	// pack (collapse) columns
		{
			NT * sendbuf = new NT[m];
			for(int i=0; i < m; ++i)
			{
				sendbuf[i] = std::accumulate( array[i], array[i]+n, identity, __binary_op);
			}
			NT * recvbuf = NULL;
			int root = commGrid->GetDiagOfProcRow();
			if(parvec.diagonal)
			{
				parvec.arr.resize(m);
				recvbuf = &parvec.arr[0];	
			}
			(commGrid->GetRowWorld()).Reduce(sendbuf, recvbuf, m, MPIType<NT>(), MPIOp<_BinaryOperation, NT>::op(), root);
			delete sendbuf;
			break;
		}
		default:
		{
			cout << "Unknown reduction dimension, returning empty vector" << endl;
			break;
		}
	}
	return parvec;
}

template <class IT, class NT>
template <typename DER>
DenseParMat< IT,NT > & DenseParMat<IT,NT>::operator+=(const SpParMat< IT,NT,DER > & rhs)		// add a sparse matrix
{
	if(*commGrid == *rhs.commGrid)	
	{
		(rhs.spSeq)->UpdateDense(array, plus<double>());
	}
	else
	{
		cout << "Grids are not comparable elementwise addition" << endl; 
		MPI::COMM_WORLD.Abort(GRIDMISMATCH);
	}
	return *this;	
}


template <class IT, class NT>
DenseParMat< IT,NT > &  DenseParMat<IT,NT>::operator=(const DenseParMat< IT,NT > & rhs)		// assignment operator
{
	if(this != &rhs)		
	{
		if(array != NULL) 
			SpHelper::deallocate2D(array, m);

		m = rhs.m;
		n = rhs.n;
		if(rhs.array != NULL)	
		{
			array = SpHelper::allocate2D<NT>(m, n);
			for(int i=0; i< m; ++i)
				copy(array[i], array[i]+n, rhs.array[i]);
		}
		commGrid.reset(new CommGrid(*(rhs.commGrid)));		
	}
	return *this;
}


