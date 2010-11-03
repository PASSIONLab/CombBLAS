#include <mpi.h>

#include <iostream>
#include "pyDenseParVec.h"

pyDenseParVec::pyDenseParVec()
{
}

int pyDenseParVec::length() const
{
	return -1; //v.getnnz();
}
	
void pyDenseParVec::load(const char* filename)
{
	ifstream input(filename);
	v.ReadDistribute(input, 0);
	input.close();
}


