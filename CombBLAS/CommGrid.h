/****************************************************************/
/* Sequential and Parallel Sparse Matrix Multiplication Library */
/* version 2.3 --------------------------------------------------/
/* date: 01/18/2009 ---------------------------------------------/
/* author: Aydin Buluc (aydin@cs.ucsb.edu) ----------------------/
/****************************************************************/


#ifndef _COMM_GRID_H_
#define _COMM_GRID_H_

#include <iostream>
#include <cmath>
#include <cassert>
#include <mpi.h>
#include <sstream>
#include <string>
#include <fstream>

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
	}
	CommGrid (const CommGrid & rhs): grrows(rhs.grrows), grcols(rhs.grcols),
			myrank(rhs.myrank), myprocrow(rhs.myprocrow), myproccol(rhs.myproccol) // copy constructor
	{
		commWorld = rhs.commWorld.Dup();
		rowWorld = rhs.rowWorld.Dup();
		colWorld = rhs.colWorld.Dup();
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
		}
		return *this;
	}


	bool operator== (const CommGrid & rhs) const;
	bool OnSameProcCol( int rhsrank );
	bool OnSameProcRow( int rhsrank );
	
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
	MPI::Intracomm GetWorld() const { return commWorld; }
	MPI::Intracomm GetRowWorld() const { return rowWorld; }
	MPI::Intracomm GetColWorld() const { return colWorld; }
	
	int GetGridRows() { return grrows; }
	int GetGridCols() { return grcols; }

	void OpenDebugFile(string prefix, ofstream & output); 

	friend shared_ptr<CommGrid> ProductGrid(CommGrid * gridA, CommGrid * gridB, int & innerdim, int & Aoffset, int & Boffset);
private:
	// A "normal" MPI-1 communicator is an intracommunicator; MPI::COMM_WORLD is also an MPI::Intracomm object
	MPI::Intracomm commWorld, rowWorld, colWorld;

	// Processor grid is (grrow X grcol)
	int grrows, grcols;
	int myproccol;
	int myprocrow;
	int myrank;
	
	template <class IT, class NT, class DER>
	friend class SpParMat;
};


#endif
