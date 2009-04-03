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
#include <mpi.h>
#include "SpWins.h"
#include "SpSizes.h"
#include "SpMat.h"
#include "SpDCCols.h"
#include "DataTypeToMPI.h"

class CommGrid

public:
	CommGrid(MPI::IntraComm & world, int nrowproc, int ncolproc): grrow(nrowproc), grcol(ncolproc)
	{
		commWorld = world.Dup();
		myrank = commWorld.Get_rank();
		int nproc = commWorld.Get_size();

		if(grrow == 0 && grcol == 0)
		{
			grrow = (int)std::sqrt((float)nproc);
			grcol = grrow;
		}
		assert((nproc == (grrow*grcol)));

		mycol =  (int) myrank % grcol;
		myrow =  (int) myrank / grcol;
		
		/** 
		  * Create row and column communicators (must be collectively called)
		  * C syntax: int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm)
		  * C++ syntax: MPI::Intercomm MPI::Intercomm::Split(int color, int key) consts  
		  * Semantics: Processes with the same color are in the same new communicator 
		  */
		rowworld = commWorld.Split(myrow, myrank);
		colworld = commWorld.Split(mycol, myrank);
	}
	~CommGrid()
	{
		commWorld.Free();
		rowWorld.Free();
		colWorld.Free();
	}
	bool operator == (const CommGrid & rhs) const
	{
		result = MPI::Comm::Compare(commWorld, rhs.commWorld);
    		if ((result != MPI::IDENT) && (result != MPI::CONGRUENT))
		{
			//! A call to MPI::Comm::Compare after MPI::Comm::Dup returns MPI_CONGRUENT
			//! MPI::CONGRUENT means the communicators have the same group members, in the same order
    			return false;
		}
		return ( (grrow == rhs.grrow) && (grcol == rhs.grcol) && (myrow == rhs.myrow) && (mycol == rhs.mycol));
	}	

	void UnlockWindows(int Aownind, int Bownind, SpWins & rwin, SpWins & cwin) const;
	void OpenDebugFile(string prefix, ofstream & output); 

	template <typename U>
	void GetA(SparseDColumn<U>* & ARecv, int Aownind, SpWins & rwin, SpSizes & ASizes );

	template <typename U>
	void GetB(SparseDColumn<U>* & BRecv, int Bownind, SpWins & cwin, SpSizes & BSizes );

	friend CommGrid ProductGrid(CommGrid & gridA, CommGrid & gridB, int & innerdim, int & Aoffset, int & Boffset);
private:
	//! A "normal" MPI-1 communicator is an intracommunicator; MPI::COMM_WORLD is also an MPI::Intracomm object
	MPI::IntraComm commWorld, rowWorld, colWorld;
	int grrow, grcol;
	int mycol;
	int myrow;
	int myrank;
};

void CommGrid::OpenDebugFile(string prefix, ofstream & output) 
{
	stringstream ss;
	string rank;
	ss << myrank;
	ss >> rank;
	string ofilename = prefix;
	ofilename += rank;
	
	output.open(ofilename.c_str(), ios_base::app );
}

void CommGrid::UnlockWindows(int Aownind, int Bownind, SpWins & rwin, SpWins & cwin) const
{
	if(Aownind != mycol)	
	{
		MPI_Win_unlock(Aownind, rwin.maswin);
		MPI_Win_unlock(Aownind, rwin.jcwin);
		MPI_Win_unlock(Aownind, rwin.irwin);
		MPI_Win_unlock(Aownind, rwin.numwin);
	}
	if(Bownind != myrow)
	{
		MPI_Win_unlock(Bownind, cwin.maswin);
		MPI_Win_unlock(Bownind, cwin.jcwin);
		MPI_Win_unlock(Bownind, cwin.irwin);
		MPI_Win_unlock(Bownind, cwin.numwin);
	}
}

template <typename U>
void CommGrid::GetA(SparseDColumn<U>* & ARecv, int Aownind, SpWins & rwin, SpSizes & ASizes )
{
	// allocate memory for arrays 
	ARecv = new SparseDColumn<U>(ASizes.nnzs[Aownind], ASizes.nrows[Aownind], ASizes.ncols[Aownind], ASizes.nzcs[Aownind]);	

	ITYPE nzcrecv = ARecv->GetJCSize();
	ITYPE nnzrecv = ARecv->GetSize();

	//int MPI_Get(void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp,
        //    		int target_count, MPI_Datatype target_datatype, MPI_Win win)
	MPI_Win_lock(MPI_LOCK_SHARED, Aownind, 0, rwin.maswin);
	MPI_Get(ARecv->GetMAS(), (nzcrecv+1), DataTypeToMPI<ITYPE>(), Aownind, 0, (nzcrecv+1), DataTypeToMPI<ITYPE>(), rwin.maswin);
		
	MPI_Win_lock(MPI_LOCK_SHARED, Aownind, 0, rwin.jcwin);
	MPI_Get(ARecv->GetJC(), nzcrecv, DataTypeToMPI<ITYPE>(), Aownind, 0, nzcrecv, DataTypeToMPI<ITYPE>(), rwin.jcwin);
		
	MPI_Win_lock(MPI_LOCK_SHARED, Aownind, 0, rwin.irwin);
	MPI_Get(ARecv->GetIR(), nnzrecv, DataTypeToMPI<ITYPE>(), Aownind, 0, nnzrecv, DataTypeToMPI<ITYPE>(), rwin.irwin);

	MPI_Win_lock(MPI_LOCK_SHARED, Aownind, 0, rwin.numwin);
	MPI_Get(ARecv->GetNUM(), nnzrecv, DataTypeToMPI<U>(), Aownind, 0, nnzrecv, DataTypeToMPI<U>(), rwin.numwin);
}


template <typename U>
void CommGrid::GetB(SparseDColumn<U>* & BRecv, int Bownind, SpWins & cwin, SpSizes & BSizes )
{	
	// allocate memory for arrays 
	BRecv = new SparseDColumn<U>(BSizes.nnzs[Bownind], BSizes.nrows[Bownind], BSizes.ncols[Bownind], BSizes.nzcs[Bownind]);	

	ITYPE nzcrecv = BRecv->GetJCSize();
	ITYPE nnzrecv = BRecv->GetSize();

	MPI_Win_lock(MPI_LOCK_SHARED, Bownind, 0, cwin.maswin);
	MPI_Get(BRecv->GetMAS(), (nzcrecv+1), DataTypeToMPI<ITYPE>(), Bownind, 0, (nzcrecv+1), DataTypeToMPI<ITYPE>(), cwin.maswin);
		
	MPI_Win_lock(MPI_LOCK_SHARED, Bownind, 0, cwin.jcwin);
	MPI_Get(BRecv->GetJC(), nzcrecv, DataTypeToMPI<ITYPE>(), Bownind, 0, nzcrecv, DataTypeToMPI<ITYPE>(), cwin.jcwin);
		
	MPI_Win_lock(MPI_LOCK_SHARED, Bownind, 0, cwin.irwin);
	MPI_Get(BRecv->GetIR(), nnzrecv, DataTypeToMPI<ITYPE>(), Bownind, 0, nnzrecv, DataTypeToMPI<ITYPE>(), cwin.irwin);
		
	MPI_Win_lock(MPI_LOCK_SHARED, Bownind, 0, cwin.numwin);
	MPI_Get(BRecv->GetNUM(), nnzrecv, DataTypeToMPI<U>(), Bownind, 0, nnzrecv, DataTypeToMPI<U>(), cwin.numwin);
}


CommGrid ProductGrid(CommGrid & gridA, CommGrid & gridB, int & innerdim, int & Aoffset, int & Boffset)
{
	if(gridA.grcol != gridB.grrow)
	{
		cout << "Grids don't confirm for multiplication" << endl;
		MPI::COMM_WORLD.Abort();
	}
	innerdim = gridA.grcol;

	Aoffset = (gridA.myrow + gridA.mycol) % gridA.grcol;	// get sequences that avoids contention
	Boffset = (gridB.myrow + gridB.mycol) % gridB.grrow;

	return CommGrid(MPI_COMM_WORLD, gridA.grrow, gridB.grcol); 
}


#endif
