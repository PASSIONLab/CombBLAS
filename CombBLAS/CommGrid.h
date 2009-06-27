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
#include "DataTypeToMPI.h"

class CommGrid
{
public:
	CommGrid(MPI::IntraComm & world, int nrowproc, int ncolproc);

	~CommGrid()
	{
		commWorld.Free();
		rowWorld.Free();
		colWorld.Free();
	}

	bool operator== (const CommGrid & rhs) const;
	
	int GetRank() { return myrank; }
	int GetRowRank() { return colWorld.Get_rank(); }
	int GetColRank() { return rowWorld.Get_rank(); }

	void OpenDebugFile(string prefix, ofstream & output); 

	friend CommGrid ProductGrid(CommGrid & gridA, CommGrid & gridB, int & innerdim, int & Aoffset, int & Boffset);
private:
	// A "normal" MPI-1 communicator is an intracommunicator; MPI::COMM_WORLD is also an MPI::Intracomm object
	MPI::IntraComm commWorld, rowWorld, colWorld;

	// Processor grid is (grrow X grcol)
	int grrow, grcol;
	int mycol;
	int myrow;
	int myrank;
	
	template <class IT, class NT, class DER>
	friend class SpParMPI2;
};


#endif
