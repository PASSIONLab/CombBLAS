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

int64_t pyDenseParVec::__len__() const
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

pyDenseParVec& pyDenseParVec::operator*=(const pyDenseParVec & rhs)
{
	v.EWiseApply(rhs.v, multiplies<int64_t>());
	return *this;
}

pyDenseParVec& pyDenseParVec::operator*=(const pySpParVec & rhs)
{
	v.EWiseApply(rhs.v, multiplies<int64_t>(), true, 0);
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

pyDenseParVec* pyDenseParVec::operator*(const pyDenseParVec & rhs)
{
	pyDenseParVec* ret = this->copy();
	*(ret) *= rhs;	
	return ret;
}

pyDenseParVec* pyDenseParVec::operator*(const pySpParVec & rhs)
{
	pyDenseParVec* ret = this->copy();
	*(ret) *= rhs;	
	return ret;
}


bool pyDenseParVec::operator==(const pyDenseParVec & rhs)
{
	return v.operator==(rhs.v);
}

bool pyDenseParVec::operator!=(const pyDenseParVec & rhs)
{
	return !(v.operator==(rhs.v));
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
pySpParVec* pyDenseParVec::__getitem__(op::UnaryFunction* op)
{
	return Find(op);
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

void pyDenseParVec::EWiseApply(const pyDenseParVec& other, op::BinaryFunction *f)
{
	v.EWiseApply(other.v, *f);
}
	
pyDenseParVec* pyDenseParVec::SubsRef(const pyDenseParVec& ri)
{
	pyDenseParVec* ret = new pyDenseParVec();
	ret->v = v(ri.v);
	return ret;
}

int64_t pyDenseParVec::getnee() const
{
	return __len__();
}

int64_t pyDenseParVec::getnnz() const
{
	return v.Count(bind2nd(not_equal_to<int64_t>(), (int64_t)0));
}

int64_t pyDenseParVec::getnz() const
{
	return v.Count(bind2nd(equal_to<int64_t>(), (int64_t)0));
}

bool pyDenseParVec::any() const
{
	return getnnz() > 0;
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

pyDenseParVec* pyDenseParVec::abs()
{
	pyDenseParVec* ret = copy();
	op::UnaryFunction* a = op::abs();
	ret->Apply(a);
	delete a;
	return ret;
}

pyDenseParVec& pyDenseParVec::operator+=(int64_t value)
{
	v.Apply(bind2nd(plus<int64_t>(), value));
}

pyDenseParVec* pyDenseParVec::operator+(int64_t value)
{
	pyDenseParVec* ret = this->copy();
	*(ret) += value;
	return ret;
}

pyDenseParVec& pyDenseParVec::operator-=(int64_t value)
{
	v.Apply(bind2nd(minus<int64_t>(), value));
}

pyDenseParVec* pyDenseParVec::operator-(int64_t value)
{
	pyDenseParVec* ret = this->copy();
	*(ret) -= value;
	return ret;
}

pyDenseParVec* pyDenseParVec::__and__(const pyDenseParVec& other)
{
	pyDenseParVec* ret = copy();
	op::BinaryFunction* a = op::logical_and();
	ret->EWiseApply(other, a);
	delete a;
	return ret;
}

int64_t pyDenseParVec::__getitem__(int64_t key)
{
	return GetElement(key);
}

pyDenseParVec* pyDenseParVec::__getitem__(const pyDenseParVec& key)
{
	return SubsRef(key);
}

void pyDenseParVec::__setitem__(int64_t key, int64_t value)
{
	SetElement(key, value);
}

void pyDenseParVec::__setitem__(const pySpParVec& key, const pySpParVec& value)
{
	v.Apply(set<int64_t>(0), key.v);
	v += value.v;
}

void pyDenseParVec::__setitem__(const pySpParVec& key, int64_t value)
{
	v.Apply(set<int64_t>(value), key.v);
}


pyDenseParVec* pyDenseParVec::range(int64_t howmany, int64_t start)
{
	pyDenseParVec* ret = new pyDenseParVec();
	ret->v.iota(howmany, start);
	return ret;
}
