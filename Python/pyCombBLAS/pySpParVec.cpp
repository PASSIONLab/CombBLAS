#include <mpi.h>

#include <iostream>
#include <math.h>

#include "pySpParVec.h"

using namespace std;

pySpParVec::pySpParVec()
{
}


//pySpParVec::pySpParVec(const pySpParMat& commSource): v(commSource.A.commGrid);
//{
//}

//pySpParVec::pySpParVec(SpParVec<int64_t, int64_t> & in_v): v(in_v)
//{
//}


int64_t pySpParVec::getnnz() const
{
	return v.getnnz();
}

void pySpParVec::add(const pySpParVec& other)
{
	v.operator+=(other.v);

	//return *this;
}

void pySpParVec::SetElement(int64_t index, int64_t numx)	// element-wise assignment
{
	v.SetElement(index, numx);
}


//const pySpParVec& pySpParVec::subtract(const pySpParVec& other)
//{
//	return *this;
//}

pySpParVec* pySpParVec::copy()
{
	pySpParVec* ret = new pySpParVec();
	ret->v = v;
	return ret;
}


void pySpParVec::invert() // "~";  almost equal to logical_not
{
	v.Apply(invert64);
}


void pySpParVec::abs()
{
	v.Apply(abs64);
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
	cout << "intersectSize missing CombBLAS piece" << endl;
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




