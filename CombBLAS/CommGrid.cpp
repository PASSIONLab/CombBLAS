#include "CommGrid.h"
#include "SpDefs.h"

CommGrid::CommGrid(MPI::Intracomm & world, int nrowproc, int ncolproc): grrows(nrowproc), grcols(ncolproc)
{
	commWorld = world.Dup();
	myrank = commWorld.Get_rank();
	int nproc = commWorld.Get_size();

	if(grrows == 0 && grcols == 0)
	{
		grrows = (int)std::sqrt((float)nproc);
		grcols = grrows;
	}
	assert((nproc == (grrows*grcols)));

	myproccol =  (int) (myrank % grcols);
	myprocrow =  (int) (myrank / grcols);
		
	/** 
	  * Create row and column communicators (must be collectively called)
	  * C syntax: int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm)
	  * C++ syntax: MPI::Intercomm MPI::Intercomm::Split(int color, int key) consts  
	  * Semantics: Processes with the same color are in the same new communicator 
	  */
	rowWorld = commWorld.Split(myprocrow, myrank);
	colWorld = commWorld.Split(myproccol, myrank);

	assert( ((rowWorld.Get_rank()) == myproccol) );
	assert( ((colWorld.Get_rank()) == myprocrow) );
}



bool CommGrid::OnSameProcCol( int rhsrank)
{
	return ( myproccol == ((int) (rhsrank % grcols)) );
} 

bool CommGrid::OnSameProcRow( int rhsrank)
{
	return ( myprocrow == ((int) (rhsrank / grcols)) );
} 

//! Return rank in the column world
int CommGrid::GetRankInProcCol( int wholerank)
{
	return ((int) (wholerank / grcols));
} 

//! Return rank in the row world
int CommGrid::GetRankInProcRow( int wholerank)
{
	return ((int) (wholerank % grcols));
}

//! Get the rank of the diagonal processor in that particular row 
//! In the ith processor row, the diagonal processor is the ith processor within that row 
int CommGrid::GetDiagOfProcRow( )
{
	return myprocrow;
}

//! Get the rank of the diagonal processor in that particular col 
//! In the ith processor col, the diagonal processor is the ith processor within that col
int CommGrid::GetDiagOfProcCol( )
{
	return myproccol;
}


bool CommGrid::operator== (const CommGrid & rhs) const
{
	int result = MPI::Comm::Compare(commWorld, rhs.commWorld);
	if ((result != MPI::IDENT) && (result != MPI::CONGRUENT))
	{
		// A call to MPI::Comm::Compare after MPI::Comm::Dup returns MPI_CONGRUENT
		// MPI::CONGRUENT means the communicators have the same group members, in the same order
    		return false;
	}
	return ( (grrows == rhs.grrows) && (grcols == rhs.grcols) && (myprocrow == rhs.myprocrow) && (myproccol == rhs.myproccol));
}	


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


shared_ptr<CommGrid> ProductGrid(CommGrid * gridA, CommGrid * gridB, int & innerdim, int & Aoffset, int & Boffset)
{
	if(gridA->grcols != gridB->grrows)
	{
		cout << "Grids don't confirm for multiplication" << endl;
		MPI::COMM_WORLD.Abort(GRIDMISMATCH);
	}
	innerdim = gridA->grcols;

	Aoffset = (gridA->myprocrow + gridA->myproccol) % gridA->grcols;	// get sequences that avoids contention
	Boffset = (gridB->myprocrow + gridB->myproccol) % gridB->grrows;

		
	return shared_ptr<CommGrid>( new CommGrid(MPI::COMM_WORLD, gridA->grrows, gridB->grcols) );
}

