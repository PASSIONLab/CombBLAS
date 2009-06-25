#include "CommGrid.h"

CommGrid::CommGrid(MPI::IntraComm & world, int nrowproc, int ncolproc): grrow(nrowproc), grcol(ncolproc)
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

bool CommGrid::operator== (const CommGrid & rhs) const
{
	result = MPI::Comm::Compare(commWorld, rhs.commWorld);
	if ((result != MPI::IDENT) && (result != MPI::CONGRUENT))
	{
		// A call to MPI::Comm::Compare after MPI::Comm::Dup returns MPI_CONGRUENT
		// MPI::CONGRUENT means the communicators have the same group members, in the same order
    		return false;
	}
	return ( (grrow == rhs.grrow) && (grcol == rhs.grcol) && (myrow == rhs.myrow) && (mycol == rhs.mycol));
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


