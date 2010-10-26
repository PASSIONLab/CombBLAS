#include <mpi.h>

#include <iostream>
#include "DiGraph.h"

DiGraph::DiGraph()
{
}

int DiGraph::nedges()
{
	return 0;//g.getnnz();
}

int DiGraph::nverts()
{
	return 0;//g.getncol();
}
	
void DiGraph::load(const char* filename)
{
	//ifstream input(filename);
	//g.ReadDistribute(input, 0);
}

void DiGraph::init()
{
	printf("calling MPI::Init\n");
	MPI::Init();
}

void DiGraph::finalize()
{
	printf("calling MPI::Finalize\n");
	MPI::Finalize();
}

