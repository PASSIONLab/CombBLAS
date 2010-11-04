#include <mpi.h>

#include <iostream>
#include "pyDenseParVec.h"

pyDenseParVec::pyDenseParVec()
{
}

pyDenseParVec::pyDenseParVec(const pySpParMat& commSource, int64_t zero)
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


const pyDenseParVec& pyDenseParVec::add(const pyDenseParVec& other) {
	v.operator+=(other.v);
	return *this;
}

const pyDenseParVec& pyDenseParVec::add(const pySpParVec& other) {
	v.operator+=(other.v);
	return *this;
}
