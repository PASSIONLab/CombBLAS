#include <mpi.h>

#include <iostream>
#include "pyDenseParVec.h"

pyDenseParVec::pyDenseParVec()
{
}

pyDenseParVec::pyDenseParVec(int64_t size, int64_t id): v(size, id)
{
}

//pyDenseParVec::pyDenseParVec(const pySpParMat& commSource, int64_t zero)
//{
//}

int pyDenseParVec::length() const
{
	return v.getTotalLength();
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
