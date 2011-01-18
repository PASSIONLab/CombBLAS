#include <mpi.h>

#include <iostream>
#include "pyDenseParVec.h"

pyDenseParVec::pyDenseParVec()
{
}

pyDenseParVec::pyDenseParVec(int64_t size, int64_t id): v(size, id, 0)
{
}

pyDenseParVec::pyDenseParVec(int64_t size, int64_t init, int64_t zero): v(size, init, zero)
{
}


pySpParVec* pyDenseParVec::sparse() const
{
	pySpParVec* ret = new pySpParVec(0);
	ret->v = v.Find(bind2nd(not_equal_to<int64_t>(), (int64_t)0));
	return ret;
}

pySpParVec* pyDenseParVec::sparse(int64_t zero) const
{
	pySpParVec* ret = new pySpParVec(0);
	
	ret->v = v.Find(bind2nd(not_equal_to<int64_t>(), zero));
	return ret;
}

int64_t pyDenseParVec::len() const
{
	return v.TotalLength();
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

pyDenseParVec& pyDenseParVec::operator+=(const pySpParVec & rhs)
{
	v.operator+=(rhs.v);
	return *this;
}

pyDenseParVec& pyDenseParVec::operator-=(const pySpParVec & rhs)
{
	v.operator-=(rhs.v);
	return *this;
}

//pyDenseParVec& pyDenseParVec::operator=(const pyDenseParVec & rhs)
//{
//	v.operator=(rhs.v);
//	return *this;
//}

pyDenseParVec* pyDenseParVec::operator+(const pyDenseParVec & rhs)
{
	pyDenseParVec* ret = this->copy();
	*(ret) += rhs;	
	return ret;
}

pyDenseParVec* pyDenseParVec::operator-(const pyDenseParVec & rhs)
{
	pyDenseParVec* ret = this->copy();
	*(ret) -= rhs;	
	return ret;
}

pyDenseParVec* pyDenseParVec::operator+(const pySpParVec & rhs)
{
	pyDenseParVec* ret = this->copy();
	*(ret) += rhs;	
	return ret;
}

pyDenseParVec* pyDenseParVec::operator-(const pySpParVec & rhs)
{
	pyDenseParVec* ret = this->copy();
	*(ret) -= rhs;	
	return ret;
}



pyDenseParVec* pyDenseParVec::copy()
{
	pyDenseParVec* ret = new pyDenseParVec();
	ret->v = v;
	return ret;
}

/////////////////////////

int64_t pyDenseParVec::Count(op::UnaryFunction* op)
{
	return v.Count(*op);
}

pySpParVec* pyDenseParVec::Find(op::UnaryFunction* op)
{
	pySpParVec* ret = new pySpParVec();
	ret->v = v.Find(*op);
	return ret;
}

pyDenseParVec* pyDenseParVec::FindInds(op::UnaryFunction* op)
{
	pyDenseParVec* ret = new pyDenseParVec();
	ret->v = v.FindInds(*op);
	return ret;
}

void pyDenseParVec::Apply(op::UnaryFunction* op)
{
	v.Apply(*op);
}

void pyDenseParVec::ApplyMasked(op::UnaryFunction* op, const pySpParVec& mask)
{
	v.Apply(*op, mask.v);
}

	
pyDenseParVec* pyDenseParVec::SubsRef(const pyDenseParVec& ri)
{
	pyDenseParVec* ret = new pyDenseParVec();
	ret->v = v(ri.v);
	return ret;
}

/*
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
*/

int64_t pyDenseParVec::getnnz() const
{
	return v.Count(bind2nd(not_equal_to<int64_t>(), (int64_t)0));
}

int64_t pyDenseParVec::getnz() const
{
	return v.Count(bind2nd(equal_to<int64_t>(), (int64_t)0));
}

void pyDenseParVec::SetElement (int64_t indx, int64_t numx)	// element-wise assignment
{
	v.SetElement(indx, numx);
}

int64_t pyDenseParVec::GetElement (int64_t indx)	// element-wise fetch
{
	return v.GetElement(indx);
}

void pyDenseParVec::RandPerm()
{
	v.RandPerm();
}

void pyDenseParVec::printall()
{
	v.DebugPrint();
}

pyDenseParVec* pyDenseParVec::range(int64_t howmany, int64_t start)
{
	pyDenseParVec* ret = new pyDenseParVec();
	ret->v.iota(howmany, start);
	return ret;
}
