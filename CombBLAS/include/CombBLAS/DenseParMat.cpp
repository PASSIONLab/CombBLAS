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

#include <numeric>
#include "DenseParMat.h"
#include "MPIType.h"
#include "Operations.h"

namespace combblas {

template <class IT, class NT>
template <typename _BinaryOperation>
FullyDistVec< IT,NT > DenseParMat<IT,NT>::Reduce(Dim dim, _BinaryOperation __binary_op, NT identity) const
{

	switch(dim)
	{
		case Column:	// pack along the columns, result is a vector of size (global) n
		{
			// we can use parvec's grid as long as the process grid is square (currently a CombBLAS requirement)
			
			int colneighs = commGrid->GetGridRows();	// including oneself
            		int colrank = commGrid->GetRankInProcCol();

			IT * loclens = new IT[colneighs];
			IT * lensums = new IT[colneighs+1]();	// begin/end points of local lengths

            		IT n_perproc = n / colneighs;    // length on a typical processor
            		if(colrank == colneighs-1)
                		loclens[colrank] = n - (n_perproc*colrank);
            		else
                		loclens[colrank] = n_perproc;

			MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IT>(), loclens, 1, MPIType<IT>(), commGrid->GetColWorld());
			std::partial_sum(loclens, loclens+colneighs, lensums+1);	// loclens and lensums are different, but both would fit in 32-bits

			std::vector<NT> trarr(loclens[colrank]);
			NT * sendbuf = new NT[n];
			for(int j=0; j < n; ++j)
			{
				sendbuf[j] = identity;
				for(int i=0; i < m; ++i)
				{
					sendbuf[j] = __binary_op(array[i][j], sendbuf[j]); 
				}
			}
             		
			// The MPI_REDUCE_SCATTER routine is functionally equivalent to:
            		// an MPI_REDUCE collective operation with count equal to the sum of loclens[i]
           		// followed by MPI_SCATTERV with sendcounts equal to loclens as well
            		MPI_Reduce_scatter(sendbuf, trarr.data(), loclens, MPIType<NT>(), MPIOp<_BinaryOperation, NT>::op(), commGrid->GetColWorld());
			
			DeleteAll(sendbuf, loclens, lensums);

			IT reallen;	// Now we have to transpose the vector
			IT trlen = trarr.size();
			int diagneigh = commGrid->GetComplementRank();
			MPI_Status status;
			MPI_Sendrecv(&trlen, 1, MPIType<IT>(), diagneigh, TRNNZ, &reallen, 1, MPIType<IT>(), diagneigh, TRNNZ, commGrid->GetWorld(), &status);
			IT glncols = gcols();
            		FullyDistVec<IT,NT> parvec(commGrid, glncols, identity);			

			assert((parvec.arr.size() ==  reallen));
			MPI_Sendrecv(trarr.data(), trlen, MPIType<NT>(), diagneigh, TRX, parvec.arr.data(), reallen, MPIType<NT>(), diagneigh, TRX, commGrid->GetWorld(), &status);

			
            		return parvec;
			break;
		}
		case Row:	// pack along the rows, result is a vector of size m
		{
			IT glnrows = grows();
            		FullyDistVec<IT,NT> parvec(commGrid, glnrows, identity);

			NT * sendbuf = new NT[m];
			for(int i=0; i < m; ++i)
			{
				sendbuf[i] = std::accumulate( array[i], array[i]+n, identity, __binary_op);
			}
			NT * recvbuf = parvec.arr.data();
            
            
            		int rowneighs = commGrid->GetGridCols();
           		int rowrank = commGrid->GetRankInProcRow();
            		IT * recvcounts = new IT[rowneighs];
            		recvcounts[rowrank] = parvec.MyLocLength();  // local vector lengths are the ultimate receive counts
            		MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IT>(), recvcounts, 1, MPIType<IT>(), commGrid->GetRowWorld());
            
            		// The MPI_REDUCE_SCATTER routine is functionally equivalent to:
            		// an MPI_REDUCE collective operation with count equal to the sum of recvcounts[i]
           		// followed by MPI_SCATTERV with sendcounts equal to recvcounts.
            		MPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, MPIType<NT>(), MPIOp<_BinaryOperation, NT>::op(), commGrid->GetRowWorld());
			delete [] sendbuf;
            		delete [] recvcounts;
           		return parvec;
			break;
		}
		default:
		{
			std::cout << "Unknown reduction dimension, returning empty vector" << std::endl;
            		return FullyDistVec<IT,NT>(commGrid);
			break;
		}
	}
}

template <class IT, class NT>
template <typename DER>
DenseParMat< IT,NT > & DenseParMat<IT,NT>::operator+=(const SpParMat< IT,NT,DER > & rhs)		// add a sparse matrix
{
	if(*commGrid == *rhs.commGrid)	
	{
		(rhs.spSeq)->UpdateDense(array, std::plus<double>());
	}
	else
	{
		std::cout << "Grids are not comparable elementwise addition" << std::endl; 
		MPI_Abort(MPI_COMM_WORLD,GRIDMISMATCH);
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
				std::copy(array[i], array[i]+n, rhs.array[i]);
		}
		commGrid.reset(new CommGrid(*(rhs.commGrid)));		
	}
	return *this;
}

}
