/****************************************************************/
/* Sequential and Parallel Sparse Matrix Multiplication Library */
/* version 2.3 --------------------------------------------------/
/* date: 01/18/2009 ---------------------------------------------/
/* author: Aydin Buluc (aydin@cs.ucsb.edu) ----------------------/
/****************************************************************/

#ifndef _DENSE_PAR_MAT_H_
#define _DENSE_PAR_MAT_H_

#include <iostream>
#include <fstream>
#include <cmath>
#include <mpi.h>
#include <vector>
#ifdef NOTR1
	#include <boost/tr1/memory.hpp>
	#include <boost/tr1/tuple.hpp>
#else
	#include <tr1/memory>	// for shared_ptr
	#include <tr1/tuple>
#endif
#include "CommGrid.h"
#include "MPIType.h"
#include "Deleter.h"
#include "SpHelper.h"
#include "SpParMat.h"
#include "DenseParVec.h"

template <class IU, class NU, class DER>
class SpParMat;

using namespace std;
using namespace std::tr1;

template <class IT, class NT>
class DenseParMat
{
public:
	// Constructors
	DenseParMat (): array(NULL), m(0), n(0)
	{
		commGrid.reset(new CommGrid(MPI::COMM_WORLD, 0, 0));		
	}
	DenseParMat (NT ** seqarr, shared_ptr<CommGrid> grid, IT rows, IT cols): array(seqarr), m(rows), n(cols)
	{
		commGrid.reset(new CommGrid(*grid)); 
	}

	DenseParMat (const DenseParMat< IT,NT > & rhs): m(rhs.m), n(rhs.n)			// copy constructor
	{
		if(rhs.array != NULL)	
		{
			array = SpHelper::allocate2D<NT>(m, n);
			for(int i=0; i< m; ++i)
				copy(array[i], array[i]+n, rhs.array[i]);
		}
		commGrid.reset(new CommGrid(*(rhs.commGrid)));		
	}
	DenseParMat< IT,NT > & operator=(const DenseParMat< IT,NT > & rhs)		// assignment operator
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

	template <typename DER>
	DenseParMat< IT,NT > & operator+=(const SpParMat< IT,NT,DER > & rhs)		// add a sparse matrix
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
	~DenseParMat ()
	{
		if(array != NULL) 
			SpHelper::deallocate2D(array, m);
	}					

	DenseParVec< IT,NT > SumRows();
	DenseParVec< IT,NT > SumCols();

	shared_ptr<CommGrid> getcommgrid () { return commGrid; }	

private:
	const static IT zero = static_cast<IT>(0);
	shared_ptr<CommGrid> commGrid; 
	NT ** array;
	IT m, n;

	template <class IU, class NU, class DER>
	friend class SpParMat; 
};

#endif

