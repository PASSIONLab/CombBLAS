#include <mpi.h>

#include <iostream>
#include "pyDenseParVec.h"

pyDenseParVec::pyDenseParVec()
{
}

pyDenseParVec::pyDenseParVec(int64_t size, int64_t id)
{
	MPI::Intracomm comm = v.getCommGrid()->GetDiagWorld();
	
	int64_t locsize = 0;
	
	if (comm != MPI::COMM_NULL)
	{
		int nprocs = comm.Get_size();
		locsize = (int64_t)ceil(double(size)/double(nprocs));
	}

	DenseParVec<int64_t, int64_t> temp(locsize, id);
	v.stealFrom(temp);
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


void pyDenseParVec::add(const pyDenseParVec& other) {
	v.operator+=(other.v);
	//return *this;
}

void pyDenseParVec::add(const pySpParVec& other) {
	v.operator+=(other.v);
	//return *this;
}

pyDenseParVec& pyDenseParVec::operator+=(const pyDenseParVec & rhs)
{
	v.operator+=(rhs.v);
	return *this;
}

pyDenseParVec& pyDenseParVec::operator-=(const pyDenseParVec & rhs)
{
	v.operator-=(rhs.v);
	return *this;
}

//pyDenseParVec& pyDenseParVec::operator=(const pyDenseParVec & rhs)
//{
//	v.operator=(rhs.v);
//	return *this;
//}

pyDenseParVec* pyDenseParVec::copy()
{
	pyDenseParVec* ret = new pyDenseParVec();
	ret->v = v;
	return ret;
}


pySpParVec* pyDenseParVec::FindInds_GreaterThan(int64_t value)
{
	pySpParVec* ret = new pySpParVec();
	ret->v = v.FindInds(bind2nd(greater<int64_t>(), value));
	return ret;
}


void pyDenseParVec::invert() // "~";  almost equal to logical_not
{
	v.Apply(invert64);
}


void pyDenseParVec::abs()
{
	v.Apply(abs64);
}

void pyDenseParVec::negate()
{
	v.Apply(negate64);
}

int64_t pyDenseParVec::getnnz() const
{
	return v.Count(nonzero64);
}

int64_t pyDenseParVec::getnz() const
{
	return v.Count(zero64);
}

void pyDenseParVec::SetElement (int64_t indx, int64_t numx)	// element-wise assignment
{
	v.SetElement(indx, numx);
}

int64_t pyDenseParVec::GetElement (int64_t indx)	// element-wise fetch
{
	return v.GetElement(indx);
}

void pyDenseParVec::printall()
{
	v.DebugPrint();
}
