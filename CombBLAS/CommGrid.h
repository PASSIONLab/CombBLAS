/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.2 -------------------------------------------------*/
/* date: 10/06/2011 --------------------------------------------*/
/* authors: Aydin Buluc (abuluc@lbl.gov), Adam Lugowski --------*/
/****************************************************************/
/*
Copyright (c) 2011, Aydin Buluc

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

#ifndef _COMM_GRID_H_
#define _COMM_GRID_H_

#include <iostream>
#include <cmath>
#include <cassert>
#include <mpi.h>
#include <sstream>
#include <string>
#include <fstream>
#include <stdint.h>

#ifdef NOTR1
	#include <boost/tr1/memory.hpp>
#else
	#include <tr1/memory>
#endif
#include "MPIType.h"

using namespace std;
using namespace std::tr1;

class CommGrid
{
public:
	CommGrid(MPI::Intracomm & world, int nrowproc, int ncolproc);

	~CommGrid()
	{
		commWorld.Free();
		rowWorld.Free();
		colWorld.Free();
		if(diagWorld != MPI::COMM_NULL) diagWorld.Free();
	}
	CommGrid (const CommGrid & rhs): grrows(rhs.grrows), grcols(rhs.grcols),
			myprocrow(rhs.myprocrow), myproccol(rhs.myproccol), myrank(rhs.myrank) // copy constructor
	{
		commWorld = rhs.commWorld.Dup();
		rowWorld = rhs.rowWorld.Dup();
		colWorld = rhs.colWorld.Dup();

		// don't use the shortcut ternary ? operator, C++ syntax fails as
		// mpich implements MPI::COMM_NULL of different type than MPI::IntraComm
		if(rhs.diagWorld == MPI::COMM_NULL)
			diagWorld = MPI::COMM_NULL;
		else
			diagWorld = rhs.diagWorld.Dup();
	}
	
	CommGrid & operator=(const CommGrid & rhs)	// assignment operator
	{
		if(this != &rhs)		
		{
			commWorld.Free();
			rowWorld.Free();
			colWorld.Free();

			grrows = rhs.grrows;
			grcols = rhs.grcols;
			myrank = rhs.myrank;
			myprocrow = rhs.myprocrow;
			myproccol = rhs.myproccol;

			commWorld = rhs.commWorld.Dup();
			rowWorld = rhs.rowWorld.Dup();
			colWorld = rhs.colWorld.Dup();
			
			if(rhs.diagWorld == MPI::COMM_NULL)
				diagWorld = MPI::COMM_NULL;
			else
				diagWorld = rhs.diagWorld.Dup();
		}
		return *this;
	}
	void CreateDiagWorld();
	
	bool operator== (const CommGrid & rhs) const;
	bool operator!= (const CommGrid & rhs) const
	{
		return (! (*this == rhs));
	}
	bool OnSameProcCol( int rhsrank );
	bool OnSameProcRow( int rhsrank );

	int GetRank(int rowrank, int colrank) { return rowrank * grcols + colrank; }	
	int GetRank(int diagrank) { return diagrank * grcols + diagrank; }
	int GetRank() { return myrank; }
	int GetRankInProcRow() { return myproccol; }
	int GetRankInProcCol() { return myprocrow; }

	int GetRankInProcRow(int wholerank);
	int GetRankInProcCol(int wholerank);

	int GetDiagOfProcRow();
	int GetDiagOfProcCol();

	int GetComplementRank()	// For P(i,j), get rank of P(j,i)
	{
		return ((grcols * myproccol) + myprocrow);
	}
	
	MPI::Intracomm & GetWorld() { return commWorld; }
	MPI::Intracomm & GetRowWorld() { return rowWorld; }
	MPI::Intracomm & GetColWorld() { return colWorld; }
	MPI::Intracomm & GetDiagWorld() { return diagWorld; }
	MPI::Intracomm GetWorld() const { return commWorld; }
	MPI::Intracomm GetRowWorld() const { return rowWorld; }
	MPI::Intracomm GetColWorld() const { return colWorld; }
	MPI::Intracomm GetDiagWorld() const { return diagWorld; }

	int GetGridRows() { return grrows; }
	int GetGridCols() { return grcols; }
	int GetSize() { return grrows * grcols; }

	void OpenDebugFile(string prefix, ofstream & output) const; 

	friend shared_ptr<CommGrid> ProductGrid(CommGrid * gridA, CommGrid * gridB, int & innerdim, int & Aoffset, int & Boffset);
private:
	// A "normal" MPI-1 communicator is an intracommunicator; MPI::COMM_WORLD is also an MPI::Intracomm object
	MPI::Intracomm commWorld, rowWorld, colWorld, diagWorld;

	// Processor grid is (grrow X grcol)
	int grrows, grcols;
	int myprocrow;
	int myproccol;
	int myrank;
	
	template <class IT, class NT, class DER>
	friend class SpParMat;
};

#endif
