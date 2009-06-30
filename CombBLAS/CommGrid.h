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
#include <cmath>
#include <mpi.h>
#include <sstream>
#include <string>
#include <fstream>
#include "MPIType.h"

using namespace std;

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

	bool operator== (const CommGrid & rhs) const;
	bool OnSameProcCol( int rhsrank );
	bool OnSameProcRow( int rhsrank );
	
	int GetRank() { return myrank; }
	int GetRankInProcRow() { return myproccol; }
	int GetRankInProcCol() { return myprocrow; }

	int GetRankInProcRow(int wholerank);
	int GetRankInProcCol(int wholerank);
	
	MPI::Intracomm & GetWorld() { return commWorld; }
	MPI::Intracomm & GetRowWorld() { return rowWorld; }
	MPI::Intracomm & GetColWorld() { return colWorld; }
	MPI::Intracomm GetWorld() const { return commWorld; }
	MPI::Intracomm GetRowWorld() const { return rowWorld; }
	MPI::Intracomm GetColWorld() const { return colWorld; }
	
	int GetGridRows() { return grrows; }
	int GetGridCols() { return grcols; }

	void OpenDebugFile(string prefix, ofstream & output); 

	friend CommGrid * ProductGrid(CommGrid * gridA, CommGrid * gridB, int & innerdim, int & Aoffset, int & Boffset);
private:
	// A "normal" MPI-1 communicator is an intracommunicator; MPI::COMM_WORLD is also an MPI::Intracomm object
	MPI::Intracomm commWorld, rowWorld, colWorld;

	// Processor grid is (grrow X grcol)
	int grrows, grcols;
	int myproccol;
	int myprocrow;
	int myrank;
	
	template <class IT, class NT, class DER>
	friend class SpParMPI2;
};


#endif
