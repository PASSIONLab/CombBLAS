#include <mpi.h>

#include <iostream>
#include "pySpParVec.h"

pySpParVec::pySpParVec()
{
}

int pySpParVec::length() const
{
	return v.getnnz();
}

const pySpParVec& pySpParVec::add(const pySpParVec& other)
{
	v.operator+=(other.v);

	return *this;
}

const pySpParVec& pySpParVec::subtract(const pySpParVec& other)
{
	return *this;
}

const pySpParVec& pySpParVec::invert() // "~";  almost equal to logical_not
{
	return *this;
}

const pySpParVec& pySpParVec::abs()
{
	return *this;
}

bool pySpParVec::anyNonzeros() const
{
	return false;
}

bool pySpParVec::allNonzeros() const
{
	return false;
}

int64_t pySpParVec::intersectSize(const pySpParVec& other)
{
	return 0;
}

	
void pySpParVec::load(const char* filename)
{
	ifstream input(filename);
	v.ReadDistribute(input, 0);
	input.close();
}


pySpParVec* pySpParVec::zeros(int64_t howmany)
{
	pySpParVec* ret = new pySpParVec();
	return ret;
}

pySpParVec* pySpParVec::range(int64_t howmany, int64_t start)
{
	pySpParVec* ret = new pySpParVec();
	ret->v.iota(howmany, start);
	return ret;
}

