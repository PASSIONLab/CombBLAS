#include <mpi.h>

#include <iostream>
#include "pyDenseParVec.h"

pyDenseParVec::pyDenseParVec()
{
}

pyDenseParVec::pyDenseParVec(int64_t size, double id): v(size, id, 0)
{
}

pyDenseParVec::pyDenseParVec(int64_t size, double init, double zero): v(size, init, zero)
{
}


pySpParVec* pyDenseParVec::sparse() const
{
	pySpParVec* ret = new pySpParVec(0);
	ret->v = v.Find(bind2nd(not_equal_to<doubleint>(), doubleint(0)));
	return ret;
}

pySpParVec* pyDenseParVec::sparse(double zero) const
{
	pySpParVec* ret = new pySpParVec(0);
	
	ret->v = v.Find(bind2nd(not_equal_to<doubleint>(), doubleint(zero)));
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
	v.EWiseApply(rhs.v, multiplies<doubleint>());
	return *this;
}

pyDenseParVec& pyDenseParVec::operator*=(const pySpParVec & rhs)
{
	v.EWiseApply(rhs.v, multiplies<doubleint>(), true, 0);
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


pyDenseParVec* pyDenseParVec::operator==(const pyDenseParVec & rhs)
{
	//return v.operator==(rhs.v);
	pyDenseParVec* ret = copy();
	op::BinaryFunction* eq = op::equal_to();
	ret->EWiseApply(rhs, eq);
	delete eq;
	return ret;
}

pyDenseParVec* pyDenseParVec::operator!=(const pyDenseParVec & rhs)
{
	//return !(v.operator==(rhs.v));
	pyDenseParVec* ret = copy();
	op::BinaryFunction* neq = op::not_equal_to();
	ret->EWiseApply(rhs, neq);
	delete neq;
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

double pyDenseParVec::Reduce(op::BinaryFunction* f)
{
	if (!f->associative && root())
		cout << "Attempting to Reduce with a non-associative function! Results will be undefined" << endl;

	f->getMPIOp();
	doubleint ret = v.Reduce(*f, 0);
	f->releaseMPIOp();
	return ret;
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
	
	FullyDistVec<INDEXTYPE, INDEXTYPE> fi_ret = v.FindInds(*op);
	ret->v = fi_ret;
	//ret->v = v.FindInds(*op);
	//cout << "FindInds commented out " << endl;
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

void pyDenseParVec::EWiseApply(const pySpParVec& other, op::BinaryFunction *f, bool doNulls, double nullValue)
{
	v.EWiseApply(other.v, *f, doNulls, nullValue);
}
	
pyDenseParVec* pyDenseParVec::SubsRef(const pyDenseParVec& ri)
{
	pyDenseParVec* ret = new pyDenseParVec();
	FullyDistVec<INDEXTYPE, INDEXTYPE> indexv = ri.v;
	ret->v = v(indexv);
	//cout << "SubsRef commented out " << endl;
	return ret;
}

int64_t pyDenseParVec::getnee() const
{
	return __len__();
}

int64_t pyDenseParVec::getnnz() const
{
	return v.Count(bind2nd(not_equal_to<doubleint>(), (double)0));
}

int64_t pyDenseParVec::getnz() const
{
	return v.Count(bind2nd(equal_to<doubleint>(), (double)0));
}

bool pyDenseParVec::any() const
{
	return getnnz() > 0;
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

pyDenseParVec& pyDenseParVec::operator+=(double value)
{
	v.Apply(bind2nd(plus<doubleint>(), doubleint(value)));
}

pyDenseParVec* pyDenseParVec::operator+(double value)
{
	pyDenseParVec* ret = this->copy();
	*(ret) += value;
	return ret;
}

pyDenseParVec& pyDenseParVec::operator-=(double value)
{
	v.Apply(bind2nd(minus<doubleint>(), doubleint(value)));
}

pyDenseParVec* pyDenseParVec::operator-(double value)
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

double pyDenseParVec::__getitem__(int64_t key)
{
	return v.GetElement(key);
}

double pyDenseParVec::__getitem__(double  key)
{
	return v.GetElement(static_cast<int64_t>(key));
}

pyDenseParVec* pyDenseParVec::__getitem__(const pyDenseParVec& key)
{
	return SubsRef(key);
}

void pyDenseParVec::__setitem__(int64_t key, double value)
{
	v.SetElement(key, value);
}

void pyDenseParVec::__setitem__(double  key, double value)
{
	v.SetElement(static_cast<int64_t>(key), value);
}

void pyDenseParVec::__setitem__(const pySpParVec& key, const pySpParVec& value)
{
	v.Apply(set<doubleint>(doubleint(0)), key.v);
	v += value.v;
}

void pyDenseParVec::__setitem__(const pySpParVec& key, double value)
{
	v.Apply(set<doubleint>(value), key.v);
}


pyDenseParVec* pyDenseParVec::range(int64_t howmany, int64_t start)
{
	pyDenseParVec* ret = new pyDenseParVec();
	ret->v.iota(howmany, start);
	return ret;
}

void testfunc()
{
	doubleint one(1), zero(0);
	FullyDistVec<doubleint, doubleint> a(10, one, zero);
	FullyDistVec<doubleint, doubleint> b(10, one, zero);
	
	FullyDistVec<doubleint, doubleint> x = a(b);
}
