#include <mpi.h>
#include <sys/time.h> 
#include <iostream>
#include <iomanip>
#include <functional>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include "MPIType.h"

#ifdef BUPC
extern "C" void * VoidRMat(unsigned scale, unsigned EDGEFACTOR, double initiator[4], int layer_grid, int rankinlayer);
#endif



void * VoidRMat(unsigned scale, unsigned EDGEFACTOR, double initiator[4], int layer_grid, int rankinlayer)
{
	MPI::COMM_WORLD.Barrier();
	
	uint64_t val0 = 0;
	if(MPI::COMM_WORLD.Get_rank() )
		val0 = 1;
	MPI::COMM_WORLD.Bcast(&val0, 1, MPIType<uint64_t>(),0);

	stringstream ss;
        string rank;
        ss << MPI::COMM_WORLD.Get_rank();
        ss >> rank;
        string ofilename = "debug";
        ofilename += rank;
	ofstream output;
       	output.open(ofilename.c_str(), ios_base::app );
	output << MPI::COMM_WORLD.Get_rank() << " bcasted" << endl;

	int nprocs = MPI::COMM_WORLD.Get_size();
	uint64_t * sendcnt = new uint64_t[nprocs];
	for(int i=0; i<nprocs; ++i)
		sendcnt[i] = (uint64_t) i;

	uint64_t * recvcnt = new uint64_t[nprocs];
	MPI::COMM_WORLD.Alltoall(sendcnt, 1, MPIType<uint64_t>(), recvcnt, 1, MPIType<uint64_t>()); // share the counts 
	output << MPI::COMM_WORLD.Get_rank() << " alltoall'd" << endl;
	output.flush();
	output.close();

	// MPI::Intercomm MPI::Intercomm::Split(int color, int key) consts  
        // Semantics: Processes with the same color are in the same new communicator
	MPI::Intracomm layerWorld = MPI::COMM_WORLD.Split(layer_grid, rankinlayer);

	if(layer_grid == 0)
	{		
		cerr << MPI::COMM_WORLD.Get_rank() << " maps to " << layerWorld.Get_rank() << endl;

		int nprocs = layerWorld.Get_size();
		int * sendcnt = new int[nprocs];
		for(int i=0; i<nprocs; ++i)
			sendcnt[i] = i;

		int * recvcnt = new int[nprocs];
		layerWorld.Alltoall(sendcnt, 1, MPI::INT, recvcnt, 1, MPI::INT); // share the counts 
       		output.open(ofilename.c_str(), ios_base::app );
		output << "Second alltoall" << endl;
		output.flush();
		output.close();
	}

	MPI::COMM_WORLD.Barrier();
	return NULL;
}
