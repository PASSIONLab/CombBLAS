#include <mpi.h>

#include <iostream>
#include "DiGraph.h"

DiGraph::DiGraph()
{
}

int DiGraph::nedges()
{
	return g.getnnz();
}

int DiGraph::nverts()
{
	return g.getncol();
}
	
void DiGraph::load(const char* filename)
{
	ifstream input(filename);
	g.ReadDistribute(input, 0);
	input.close();
}

void DiGraph::SpMV_SelMax(const SpVectList& v)
{
	cout << "SpMV on SelectMax semiring with vector of size " << v.length() << endl;
}


void init_pyCombBLAS_MPI()
{
	cout << "calling MPI::Init" << endl;
	MPI::Init();
	int nprocs = MPI::COMM_WORLD.Get_size();
	int myrank = MPI::COMM_WORLD.Get_rank();
	MPI::COMM_WORLD.Barrier();
	
	int sum = 0;
	int one = 1;
	MPI_Reduce(&one, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD); 

	cout << "I am proc " << myrank << " out of " << nprocs << ". Hear me roar!" << endl;
	if (myrank == 0) {
		cout << "We all reduced our ones to get " << sum;
		if (sum == nprocs)
			cout << ". Success! MPI works." << endl;
		else
			cout << ". SHOULD GET #PROCS! MPI is broken!" << endl;
	}
	
}

void finalize() {
	cout << "calling MPI::Finalize" << endl;
	MPI::Finalize();
}
