#if 0

#include <mpi.h>

#include <iostream>
#include "pyDenseParVecObj1.h"

pyDenseParVecObj1::pyDenseParVecObj1()
{
}

pyDenseParVecObj1::pyDenseParVecObj1(VectType other): v(other)
{
}

pyDenseParVecObj1::pyDenseParVecObj1(int64_t size, double id): v(size, id, 0)
{
}

pyDenseParVecObj1::pyDenseParVecObj1(int64_t size, double init, double zero): v(size, init, zero)
{
}


pySpParVecObj1 pyDenseParVecObj1::sparse() const
{
	return pySpParVecObj1(v.Find(bind2nd(not_equal_to<doubleint>(), doubleint(0))));
}

pySpParVecObj1 pyDenseParVecObj1::sparse(double zero) const
{
	return pySpParVecObj1(v.Find(bind2nd(not_equal_to<doubleint>(), doubleint(zero))));
}

int64_t pyDenseParVecObj1::len() const
{
	return v.TotalLength();
}

int64_t pyDenseParVecObj1::__len__() const
{
	return v.TotalLength();
}
	
void pyDenseParVecObj1::load(const char* filename)
{
	ifstream input(filename);
	v.ReadDistribute(input, 0);
	input.close();
}

/*
void pyDenseParVecObj1::add(const pyDenseParVecObj1& other) {
	v.operator+=(other.v);
	//return *this;
}

void pyDenseParVecObj1::add(const pySpParVec& other) {
	v.operator+=(other.v);
	//return *this;
}

pyDenseParVecObj1& pyDenseParVecObj1::operator+=(const pyDenseParVecObj1 & rhs)
{
	v.operator+=(rhs.v);
	return *this;
}

pyDenseParVecObj1& pyDenseParVecObj1::operator-=(const pyDenseParVecObj1 & rhs)
{
	v.operator-=(rhs.v);
	return *this;
}

pyDenseParVecObj1& pyDenseParVecObj1::operator+=(const pySpParVec & rhs)
{
	v.operator+=(rhs.v);
	return *this;
}

pyDenseParVecObj1& pyDenseParVecObj1::operator-=(const pySpParVec & rhs)
{
	v.operator-=(rhs.v);
	return *this;
}

pyDenseParVecObj1& pyDenseParVecObj1::operator*=(const pyDenseParVecObj1 & rhs)
{
	v.EWiseApply(rhs.v, multiplies<doubleint>());
	return *this;
}

pyDenseParVecObj1& pyDenseParVecObj1::operator*=(const pySpParVec & rhs)
{
	v.EWiseApply(rhs.v, multiplies<doubleint>(), true, 0);
	return *this;
}
*/
//pyDenseParVecObj1& pyDenseParVecObj1::operator=(const pyDenseParVecObj1 & rhs)
//{
//	v.operator=(rhs.v);
//	return *this;
//}
/*
pyDenseParVecObj1 pyDenseParVecObj1::operator+(const pyDenseParVecObj1 & rhs)
{
	pyDenseParVecObj1 ret = this->copy();
	ret += rhs;	
	return ret;
}

pyDenseParVecObj1 pyDenseParVecObj1::operator-(const pyDenseParVecObj1 & rhs)
{
	pyDenseParVecObj1 ret = this->copy();
	ret -= rhs;	
	return ret;
}

pyDenseParVecObj1 pyDenseParVecObj1::operator+(const pySpParVec & rhs)
{
	pyDenseParVecObj1 ret = this->copy();
	ret += rhs;	
	return ret;
}

pyDenseParVecObj1 pyDenseParVecObj1::operator-(const pySpParVec & rhs)
{
	pyDenseParVecObj1 ret = this->copy();
	ret -= rhs;	
	return ret;
}

pyDenseParVecObj1 pyDenseParVecObj1::operator*(const pyDenseParVecObj1 & rhs)
{
	pyDenseParVecObj1 ret = this->copy();
	ret *= rhs;	
	return ret;
}

pyDenseParVecObj1 pyDenseParVecObj1::operator*(const pySpParVec & rhs)
{
	pyDenseParVecObj1 ret = this->copy();
	ret *= rhs;	
	return ret;
}


pyDenseParVecObj1 pyDenseParVecObj1::operator==(const pyDenseParVecObj1 & rhs)
{
	//return v.operator==(rhs.v);
	pyDenseParVecObj1 ret = copy();
	ret.EWiseApply(rhs, &op::equal_to());
	return ret;
}

pyDenseParVecObj1 pyDenseParVecObj1::operator!=(const pyDenseParVecObj1 & rhs)
{
	//return !(v.operator==(rhs.v));
	pyDenseParVecObj1 ret = copy();
	ret.EWiseApply(rhs, &op::not_equal_to());
	return ret;
}*/

pyDenseParVecObj1 pyDenseParVecObj1::copy()
{
	pyDenseParVecObj1 ret;
	ret.v = v;
	return ret;
}

/////////////////////////

int64_t pyDenseParVecObj1::Count(op::UnaryPredicateObj* op)
{
	return v.Count(*op);
}

double pyDenseParVecObj1::Reduce(op::BinaryFunctionObj* bf, op::UnaryFunctionObj* uf)
{
	if (!bf->associative && root())
		cout << "Attempting to Reduce with a non-associative function! Results will be undefined" << endl;

	doubleint ret;
	
	bf->getMPIOp();
	if (uf == NULL)
		ret = v.Reduce(*bf, doubleint::nan(), ::identity<doubleint>());
	else
		ret = v.Reduce(*bf, doubleint::nan(), *uf);
	bf->releaseMPIOp();
	return ret;
}

pySpParVecObj1 pyDenseParVecObj1::Find(op::UnaryPredicateObj* op)
{
	return pySpParVecObj1(v.Find(*op));
}
pySpParVecObj1 pyDenseParVecObj1::__getitem__(op::UnaryPredicateObj* op)
{
	return Find(op);
}

pyDenseParVecObj1 pyDenseParVecObj1::FindInds(op::UnaryPredicateObj* op)
{
	pyDenseParVecObj1 ret;
	
	FullyDistVec<INDEXTYPE, INDEXTYPE> fi_ret = v.FindInds(*op);
	ret.v = fi_ret;
	return ret;
}

void pyDenseParVecObj1::Apply(op::UnaryFunctionObj* op)
{
	v.Apply(*op);
}

void pyDenseParVecObj1::ApplyMasked(op::UnaryFunctionObj* op, const pySpParVec& mask)
{
	v.Apply(*op, mask.v);
}

void pyDenseParVecObj1::EWiseApply(const pyDenseParVecObj1& other, op::BinaryFunctionObj *f)
{
	v.EWiseApply(other.v, *f);
}

void pyDenseParVecObj1::EWiseApply(const pySpParVecObj1& other, op::BinaryFunctionObj *f, bool doNulls, double nullValue)
{
	v.EWiseApply(other.v, *f, doNulls, nullValue);
}
	
pyDenseParVecObj1 pyDenseParVecObj1::SubsRef(const pyDenseParVecObj1& ri)
{
	FullyDistVec<INDEXTYPE, INDEXTYPE> indexv = ri.v;
	return pyDenseParVecObj1(v(indexv));
}

int64_t pyDenseParVecObj1::getnee() const
{
	return __len__();
}

int64_t pyDenseParVecObj1::getnnz() const
{
	return v.Count(bind2nd(not_equal_to<doubleint>(), (double)0));
}

int64_t pyDenseParVecObj1::getnz() const
{
	return v.Count(bind2nd(equal_to<doubleint>(), (double)0));
}

bool pyDenseParVecObj1::any() const
{
	return getnnz() > 0;
}

void pyDenseParVecObj1::RandPerm()
{
	v.RandPerm();
}

pyDenseParVecObj1 pyDenseParVecObj1::Sort()
{
	pyDenseParVecObj1 ret(1, 0, 0);
	ret.v = v.sort();
	return ret; // Sort is in-place. The return value is the permutation used.
}

pyDenseParVecObj1 pyDenseParVecObj1::TopK(int64_t k)
{
	FullyDistVec<INDEXTYPE, INDEXTYPE> sel(k);
	sel.iota(k, 0);

	pyDenseParVecObj1 sorted = copy();
	op::UnaryFunctionObj negate = op::negate();
	sorted.Apply(&negate); // the negation is so that the sort direction is reversed
//	sorted.printall();
	FullyDistVec<INDEXTYPE, INDEXTYPE> perm = sorted.v.sort();
//	sorted.printall();
//	perm.DebugPrint();
	sorted.Apply(&negate);

	// return dense	
	return pyDenseParVecObj1(sorted.v(sel));
}

void pyDenseParVecObj1::printall()
{
	v.DebugPrint();
}

/*
pyDenseParVecObj1 pyDenseParVecObj1::abs()
{
	pyDenseParVecObj1 ret = copy();
	ret.Apply(&op::abs());
	return ret;
}

pyDenseParVecObj1& pyDenseParVecObj1::operator+=(double value)
{
	v.Apply(bind2nd(plus<doubleint>(), doubleint(value)));
	return *this;
}

pyDenseParVecObj1 pyDenseParVecObj1::operator+(double value)
{
	pyDenseParVecObj1 ret = this->copy();
	ret += value;
	return ret;
}

pyDenseParVecObj1& pyDenseParVecObj1::operator-=(double value)
{
	v.Apply(bind2nd(minus<doubleint>(), doubleint(value)));
	return *this;
}

pyDenseParVecObj1 pyDenseParVecObj1::operator-(double value)
{
	pyDenseParVecObj1 ret = this->copy();
	ret -= value;
	return ret;
}

pyDenseParVecObj1 pyDenseParVecObj1::__and__(const pyDenseParVecObj1& other)
{
	pyDenseParVecObj1 ret = copy();
	ret.EWiseApply(other, &op::logical_and());
	return ret;
}*/

double pyDenseParVecObj1::__getitem__(int64_t key)
{
	return v.GetElement(key);
}

double pyDenseParVecObj1::__getitem__(double  key)
{
	return v.GetElement(static_cast<int64_t>(key));
}

pyDenseParVecObj1 pyDenseParVecObj1::__getitem__(const pyDenseParVecObj1& key)
{
	return SubsRef(key);
}

void pyDenseParVecObj1::__setitem__(int64_t key, double value)
{
	v.SetElement(key, value);
}

void pyDenseParVecObj1::__setitem__(double  key, double value)
{
	v.SetElement(static_cast<int64_t>(key), value);
}

void pyDenseParVecObj1::__setitem__(const pySpParVec& key, const pySpParVecObj1& value)
{
	v.Apply(::set<doubleint>(doubleint(0)), key.v);
	v += value.v;
}

void pyDenseParVecObj1::__setitem__(const pySpParVec& key, double value)
{
	v.Apply(::set<doubleint>(value), key.v);
}

/*
pyDenseParVecObj1 pyDenseParVecObj1::range(int64_t howmany, int64_t start)
{
	pyDenseParVecObj1 ret;
	ret.v.iota(howmany, start-1);
	return ret;
}
*/
#endif
