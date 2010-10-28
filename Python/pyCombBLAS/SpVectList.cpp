#include <mpi.h>

#include <iostream>
#include "SpVectList.h"

SpVectList::SpVectList()
{
}

int SpVectList::length() const
{
	return v.getnnz();
}
	
void SpVectList::load(const char* filename)
{
	ifstream input(filename);
	v.ReadDistribute(input, 0);
	input.close();
}


