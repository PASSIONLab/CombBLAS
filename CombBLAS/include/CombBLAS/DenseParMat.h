/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.6 -------------------------------------------------*/
/* date: 6/15/2017 ---------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc  --------------------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2017, The Regents of the University of California
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */

#ifndef _DENSE_PAR_MAT_H_
#define _DENSE_PAR_MAT_H_

#include <iostream>
#include <fstream>
#include <cmath>
#include <mpi.h>
#include <vector>

#include "CombBLAS.h"
#include "Operations.h"
#include "MPIOp.h"
#include "FullyDistVec.h"

namespace combblas {

template <class IU, class NU, class DER>
class SpParMat;

template <class IT, class NT>
class DenseParMat
{
public:
	// Constructors
	DenseParMat (): array(NULL), m(0), n(0)
	{
		commGrid.reset(new CommGrid(MPI_COMM_WORLD, 0, 0));		
	}
	DenseParMat (NT value, std::shared_ptr<CommGrid> grid, IT rows, IT cols): m(rows), n(cols)
	{
		array = SpHelper::allocate2D<double>(rows, cols);
		for(int i=0; i< rows; ++i)
		{
			std::fill_n(array[i], cols, value);		// fill array[i][0] ... array[i][cols] with "value"
		}
		commGrid.reset(new CommGrid(*grid)); 
	}
	DenseParMat (NT ** seqarr, std::shared_ptr<CommGrid> grid, IT rows, IT cols): array(seqarr), m(rows), n(cols)
	{
		commGrid.reset(new CommGrid(*grid)); 
	}

	DenseParMat (const DenseParMat< IT,NT > & rhs): m(rhs.m), n(rhs.n)		// copy constructor
	{
		if(rhs.array != NULL)	
		{
			array = SpHelper::allocate2D<NT>(m, n);
			for(int i=0; i< m; ++i)
			{
        std::copy(array[i], array[i]+n, rhs.array[i]);
			}
		}
		commGrid.reset(new CommGrid(*(rhs.commGrid)));		
	}
	
	DenseParMat< IT,NT > &  operator=(const DenseParMat< IT,NT > & rhs);	

	template <typename DER>
	DenseParMat< IT,NT > & operator+=(const SpParMat< IT,NT,DER > & rhs);		// add a sparse matrix
	
	template <typename _BinaryOperation>
	FullyDistVec< IT,NT > Reduce(Dim dim, _BinaryOperation __binary_op, NT identity) const;

	~DenseParMat ()
	{
		if(array != NULL) 
			SpHelper::deallocate2D(array, m);
	}					

	std::shared_ptr<CommGrid> getcommgrid () { return commGrid; }
    
   IT grows() const
    {
        IT glrows;
        MPI_Allreduce(&m, &glrows, 1, MPIType<IT>(), MPI_SUM, commGrid->GetColWorld());
        return glrows;
    }
    
    IT gcols() const
    {
        IT glcols;
        MPI_Allreduce(&n, &glcols, 1, MPIType<IT>(), MPI_SUM, commGrid->GetRowWorld());
        return glcols;
    }

private:
	std::shared_ptr<CommGrid> commGrid; 
	NT ** array;
	IT m, n;	// Local row and columns

	template <class IU, class NU, class DER>
	friend class SpParMat; 
};

}

#include "DenseParMat.cpp"

#endif

