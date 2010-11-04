#include <mpi.h>

#include <iostream>
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

const pySpParVec& pySpParVec::add(const pySpParVec& other)
{
	v.operator+=(other.v);

	return *this;
}

void pySpParVec::SetElement(int64_t index, int64_t numx)	// element-wise assignment
{
	v.SetElement(index, numx);
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

pySpParVec* EWiseMult(const pySpParVec& a, const pySpParVec& b, bool exclude)
{
	pySpParVec* ret = new pySpParVec();
	//ret->v = ::EWiseMult(a.v, b.v, exclude);
	return ret;
}

pySpParVec* EWiseMult(const pySpParVec& a, const pyDenseParVec& b, bool exclude, int64_t zero)
{
	pySpParVec* ret = new pySpParVec();
	cout << "running EWiseMult" << endl;
	ret->v = EWiseMult(a.v, b.v, exclude, (int64_t)0);
	cout << "finished running EWiseMult" << endl;
	return ret;
}










