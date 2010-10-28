#include <mpi.h>

#include <iostream>
#include "VectList.h"

VectList::VectList()
{
}

int VectList::length() const
{
	return -1; //v.getnnz();
}
	
void VectList::load(const char* filename)
{
	ifstream input(filename);
	v.ReadDistribute(input, 0);
	input.close();
}


