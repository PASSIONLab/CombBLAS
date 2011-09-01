#include <mpi.h>

#include <iostream>
#include "pyDenseParVecObj.h"

pyDenseParVecObj::pyDenseParVecObj()
{
}

pyDenseParVecObj::pyDenseParVecObj(VectType other): v(other)
{
}

pyDenseParVecObj::pyDenseParVecObj(int64_t size, double id): v(size, id, 0)
{
}

pyDenseParVecObj::pyDenseParVecObj(int64_t size, double init, double zero): v(size, init, zero)
{
}


pySpParVec pyDenseParVecObj::sparse() const
{
	return pySpParVec(v.Find(bind2nd(not_equal_to<doubleint>(), doubleint(0))));
}

pySpParVec pyDenseParVecObj::sparse(double zero) const
{
	return pySpParVec(v.Find(bind2nd(not_equal_to<doubleint>(), doubleint(zero))));
}

int64_t pyDenseParVecObj::len() const
{
	return v.TotalLength();
}

int64_t pyDenseParVecObj::__len__() const
{
	return v.TotalLength();
}
	
void pyDenseParVecObj::load(const char* filename)
{
	ifstream input(filename);
	v.ReadDistribute(input, 0);
	input.close();
}


void pyDenseParVecObj::add(const pyDenseParVecObj& other) {
	v.operator+=(other.v);
	//return *this;
}

void pyDenseParVecObj::add(const pySpParVec& other) {
	v.operator+=(other.v);
	//return *this;
}

pyDenseParVecObj& pyDenseParVecObj::operator+=(const pyDenseParVecObj & rhs)
{
	v.operator+=(rhs.v);
	return *this;
}

pyDenseParVecObj& pyDenseParVecObj::operator-=(const pyDenseParVecObj & rhs)
{
	v.operator-=(rhs.v);
	return *this;
}

pyDenseParVecObj& pyDenseParVecObj::operator+=(const pySpParVec & rhs)
{
	v.operator+=(rhs.v);
	return *this;
}

pyDenseParVecObj& pyDenseParVecObj::operator-=(const pySpParVec & rhs)
{
	v.operator-=(rhs.v);
	return *this;
}

pyDenseParVecObj& pyDenseParVecObj::operator*=(const pyDenseParVecObj & rhs)
{
	v.EWiseApply(rhs.v, multiplies<doubleint>());
	return *this;
}

pyDenseParVecObj& pyDenseParVecObj::operator*=(const pySpParVec & rhs)
{
	v.EWiseApply(rhs.v, multiplies<doubleint>(), true, 0);
	return *this;
}

//pyDenseParVecObj& pyDenseParVecObj::operator=(const pyDenseParVecObj & rhs)
//{
//	v.operator=(rhs.v);
//	return *this;
//}

pyDenseParVecObj pyDenseParVecObj::operator+(const pyDenseParVecObj & rhs)
{
	pyDenseParVecObj ret = this->copy();
	ret += rhs;	
	return ret;
}

pyDenseParVecObj pyDenseParVecObj::operator-(const pyDenseParVecObj & rhs)
{
	pyDenseParVecObj ret = this->copy();
	ret -= rhs;	
	return ret;
}

pyDenseParVecObj pyDenseParVecObj::operator+(const pySpParVec & rhs)
{
	pyDenseParVecObj ret = this->copy();
	ret += rhs;	
	return ret;
}

pyDenseParVecObj pyDenseParVecObj::operator-(const pySpParVec & rhs)
{
	pyDenseParVecObj ret = this->copy();
	ret -= rhs;	
	return ret;
}

pyDenseParVecObj pyDenseParVecObj::operator*(const pyDenseParVecObj & rhs)
{
	pyDenseParVecObj ret = this->copy();
	ret *= rhs;	
	return ret;
}

pyDenseParVecObj pyDenseParVecObj::operator*(const pySpParVec & rhs)
{
	pyDenseParVecObj ret = this->copy();
	ret *= rhs;	
	return ret;
}


pyDenseParVecObj pyDenseParVecObj::operator==(const pyDenseParVecObj & rhs)
{
	//return v.operator==(rhs.v);
	pyDenseParVecObj ret = copy();
	ret.EWiseApply(rhs, &op::equal_to());
	return ret;
}

pyDenseParVecObj pyDenseParVecObj::operator!=(const pyDenseParVecObj & rhs)
{
	//return !(v.operator==(rhs.v));
	pyDenseParVecObj ret = copy();
	ret.EWiseApply(rhs, &op::not_equal_to());
	return ret;
}

pyDenseParVecObj pyDenseParVecObj::copy()
{
	pyDenseParVecObj ret;
	ret.v = v;
	return ret;
}

/////////////////////////

int64_t pyDenseParVecObj::Count(op::UnaryFunction* op)
{
	return v.Count(*op);
}

double pyDenseParVecObj::Reduce(op::BinaryFunction* bf, op::UnaryFunction* uf)
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

pySpParVec pyDenseParVecObj::Find(op::UnaryFunction* op)
{
	return pySpParVec(v.Find(*op));
}
pySpParVec pyDenseParVecObj::__getitem__(op::UnaryFunction* op)
{
	return Find(op);
}

pyDenseParVecObj pyDenseParVecObj::FindInds(op::UnaryFunction* op)
{
	pyDenseParVecObj ret;
	
	FullyDistVec<INDEXTYPE, INDEXTYPE> fi_ret = v.FindInds(*op);
	ret.v = fi_ret;
	return ret;
}

void pyDenseParVecObj::Apply(op::UnaryFunction* op)
{
	v.Apply(*op);
}

void pyDenseParVecObj::ApplyMasked(op::UnaryFunction* op, const pySpParVec& mask)
{
	v.Apply(*op, mask.v);
}

void pyDenseParVecObj::EWiseApply(const pyDenseParVecObj& other, op::BinaryFunction *f)
{
	v.EWiseApply(other.v, *f);
}

void pyDenseParVecObj::EWiseApply(const pySpParVec& other, op::BinaryFunction *f, bool doNulls, double nullValue)
{
	v.EWiseApply(other.v, *f, doNulls, nullValue);
}
	
pyDenseParVecObj pyDenseParVecObj::SubsRef(const pyDenseParVecObj& ri)
{
	FullyDistVec<INDEXTYPE, INDEXTYPE> indexv = ri.v;
	return pyDenseParVecObj(v(indexv));
}

int64_t pyDenseParVecObj::getnee() const
{
	return __len__();
}

int64_t pyDenseParVecObj::getnnz() const
{
	return v.Count(bind2nd(not_equal_to<doubleint>(), (double)0));
}

int64_t pyDenseParVecObj::getnz() const
{
	return v.Count(bind2nd(equal_to<doubleint>(), (double)0));
}

bool pyDenseParVecObj::any() const
{
	return getnnz() > 0;
}

void pyDenseParVecObj::RandPerm()
{
	v.RandPerm();
}

pyDenseParVecObj pyDenseParVecObj::Sort()
{
	pyDenseParVecObj ret(1, 0, 0);
	ret.v = v.sort();
	return ret; // Sort is in-place. The return value is the permutation used.
}

pyDenseParVecObj pyDenseParVecObj::TopK(int64_t k)
{
	FullyDistVec<INDEXTYPE, INDEXTYPE> sel(k);
	sel.iota(k, 0);

	pyDenseParVecObj sorted = copy();
	op::UnaryFunction negate = op::negate();
	sorted.Apply(&negate); // the negation is so that the sort direction is reversed
//	sorted.printall();
	FullyDistVec<INDEXTYPE, INDEXTYPE> perm = sorted.v.sort();
//	sorted.printall();
//	perm.DebugPrint();
	sorted.Apply(&negate);

	// return dense	
	return pyDenseParVecObj(sorted.v(sel));
}

void pyDenseParVecObj::printall()
{
	v.DebugPrint();
}

pyDenseParVecObj pyDenseParVecObj::abs()
{
	pyDenseParVecObj ret = copy();
	ret.Apply(&op::abs());
	return ret;
}

pyDenseParVecObj& pyDenseParVecObj::operator+=(double value)
{
	v.Apply(bind2nd(plus<doubleint>(), doubleint(value)));
	return *this;
}

pyDenseParVecObj pyDenseParVecObj::operator+(double value)
{
	pyDenseParVecObj ret = this->copy();
	ret += value;
	return ret;
}

pyDenseParVecObj& pyDenseParVecObj::operator-=(double value)
{
	v.Apply(bind2nd(minus<doubleint>(), doubleint(value)));
	return *this;
}

pyDenseParVecObj pyDenseParVecObj::operator-(double value)
{
	pyDenseParVecObj ret = this->copy();
	ret -= value;
	return ret;
}

pyDenseParVecObj pyDenseParVecObj::__and__(const pyDenseParVecObj& other)
{
	pyDenseParVecObj ret = copy();
	ret.EWiseApply(other, &op::logical_and());
	return ret;
}

double pyDenseParVecObj::__getitem__(int64_t key)
{
	return v.GetElement(key);
}

double pyDenseParVecObj::__getitem__(double  key)
{
	return v.GetElement(static_cast<int64_t>(key));
}

pyDenseParVecObj pyDenseParVecObj::__getitem__(const pyDenseParVecObj& key)
{
	return SubsRef(key);
}

void pyDenseParVecObj::__setitem__(int64_t key, double value)
{
	v.SetElement(key, value);
}

void pyDenseParVecObj::__setitem__(double  key, double value)
{
	v.SetElement(static_cast<int64_t>(key), value);
}

void pyDenseParVecObj::__setitem__(const pySpParVec& key, const pySpParVec& value)
{
	v.Apply(::set<doubleint>(doubleint(0)), key.v);
	v += value.v;
}

void pyDenseParVecObj::__setitem__(const pySpParVec& key, double value)
{
	v.Apply(::set<doubleint>(value), key.v);
}


pyDenseParVecObj pyDenseParVecObj::range(int64_t howmany, int64_t start)
{
	pyDenseParVecObj ret;
	ret.v.iota(howmany, start-1);
	return ret;
}

