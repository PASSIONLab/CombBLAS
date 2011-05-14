#include <mpi.h>

#include <iostream>
#include "pyObjDenseParVec.h"

pyObjDenseParVec::pyObjDenseParVec()
{
}

pyObjDenseParVec::pyObjDenseParVec(VectType other): v(other)
{
}

pyObjDenseParVec::pyObjDenseParVec(int64_t size, PyObject* id): v(size, id, 0)
{
}

pyObjDenseParVec::pyObjDenseParVec(int64_t size, PyObject* init, PyObject* zero): v(size, init, zero)
{
}

/*
pySpParVec pyObjDenseParVec::sparse() const
{
	return pySpParVec(v.Find(bind2nd(not_equal_to<doubleint>(), doubleint(0))));
}

pySpParVec pyObjDenseParVec::sparse(PyObject* zero) const
{
	return pySpParVec(v.Find(bind2nd(not_equal_to<doubleint>(), zero)));
}*/

int64_t pyObjDenseParVec::len() const
{
	return v.TotalLength();
}

int64_t pyObjDenseParVec::__len__() const
{
	return v.TotalLength();
}

/*
void pyObjDenseParVec::load(const char* filename)
{
	ifstream input(filename);
	v.ReadDistribute(input, 0);
	input.close();
}*/

/*
void pyObjDenseParVec::add(const pyObjDenseParVec& other) {
	v.operator+=(other.v);
	//return *this;
}*/

/*
void pyObjDenseParVec::add(const pySpParVec& other) {
	v.operator+=(other.v);
	//return *this;
}

pyObjDenseParVec& pyObjDenseParVec::operator+=(const pyObjDenseParVec & rhs)
{
	v.operator+=(rhs.v);
	return *this;
}

pyObjDenseParVec& pyObjDenseParVec::operator-=(const pyObjDenseParVec & rhs)
{
	v.operator-=(rhs.v);
	return *this;
}

pyObjDenseParVec& pyObjDenseParVec::operator+=(const pySpParVec & rhs)
{
	v.operator+=(rhs.v);
	return *this;
}

pyObjDenseParVec& pyObjDenseParVec::operator-=(const pySpParVec & rhs)
{
	v.operator-=(rhs.v);
	return *this;
}

pyObjDenseParVec& pyObjDenseParVec::operator*=(const pyObjDenseParVec & rhs)
{
	v.EWiseApply(rhs.v, multiplies<doubleint>());
	return *this;
}

pyObjDenseParVec& pyObjDenseParVec::operator*=(const pySpParVec & rhs)
{
	v.EWiseApply(rhs.v, multiplies<doubleint>(), true, 0);
	return *this;
}

//pyObjDenseParVec& pyObjDenseParVec::operator=(const pyObjDenseParVec & rhs)
//{
//	v.operator=(rhs.v);
//	return *this;
//}

pyObjDenseParVec pyObjDenseParVec::operator+(const pyObjDenseParVec & rhs)
{
	pyObjDenseParVec ret = this->copy();
	ret += rhs;	
	return ret;
}

pyObjDenseParVec pyObjDenseParVec::operator-(const pyObjDenseParVec & rhs)
{
	pyObjDenseParVec ret = this->copy();
	ret -= rhs;	
	return ret;
}

pyObjDenseParVec pyObjDenseParVec::operator+(const pySpParVec & rhs)
{
	pyObjDenseParVec ret = this->copy();
	ret += rhs;	
	return ret;
}

pyObjDenseParVec pyObjDenseParVec::operator-(const pySpParVec & rhs)
{
	pyObjDenseParVec ret = this->copy();
	ret -= rhs;	
	return ret;
}

pyObjDenseParVec pyObjDenseParVec::operator*(const pyObjDenseParVec & rhs)
{
	pyObjDenseParVec ret = this->copy();
	ret *= rhs;	
	return ret;
}

pyObjDenseParVec pyObjDenseParVec::operator*(const pySpParVec & rhs)
{
	pyObjDenseParVec ret = this->copy();
	ret *= rhs;	
	return ret;
}


pyObjDenseParVec pyObjDenseParVec::operator==(const pyObjDenseParVec & rhs)
{
	//return v.operator==(rhs.v);
	pyObjDenseParVec ret = copy();
	ret.EWiseApply(rhs, &op::equal_to());
	return ret;
}

pyObjDenseParVec pyObjDenseParVec::operator!=(const pyObjDenseParVec & rhs)
{
	//return !(v.operator==(rhs.v));
	pyObjDenseParVec ret = copy();
	ret.EWiseApply(rhs, &op::not_equal_to());
	return ret;
}
*/

pyObjDenseParVec pyObjDenseParVec::copy()
{
	pyObjDenseParVec ret;
	ret.v = v;
	return ret;
}

/////////////////////////
/*
int64_t pyObjDenseParVec::Count(op::UnaryFunction* op)
{
	return v.Count(*op);
}

double pyObjDenseParVec::Reduce(op::BinaryFunction* bf, op::UnaryFunction* uf)
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

pySpParVec pyObjDenseParVec::Find(op::UnaryFunction* op)
{
	return pySpParVec(v.Find(*op));
}
pySpParVec pyObjDenseParVec::__getitem__(op::UnaryFunction* op)
{
	return Find(op);
}

pyDenseParVec pyObjDenseParVec::FindInds(op::UnaryFunction* op)
{
	pyDenseParVec ret;
	
	FullyDistVec<INDEXTYPE, INDEXTYPE> fi_ret = v.FindInds(*op);
	ret.v = fi_ret;
	return ret;
}*/

void pyObjDenseParVec::Apply(op::ObjUnaryFunction* op)
{
	v.Apply(*op);
}
/*
void pyObjDenseParVec::ApplyMasked(op::UnaryFunction* op, const pySpParVec& mask)
{
	v.Apply(*op, mask.v);
}*/
/*
void pyObjDenseParVec::EWiseApply(const pyObjDenseParVec& other, op::BinaryFunction *f)
{
	v.EWiseApply(other.v, *f);
}*/
/*
void pyObjDenseParVec::EWiseApply(const pySpParVec& other, op::BinaryFunction *f, bool doNulls, double nullValue)
{
	v.EWiseApply(other.v, *f, doNulls, nullValue);
}
	
pyObjDenseParVec pyObjDenseParVec::SubsRef(const pyObjDenseParVec& ri)
{
	FullyDistVec<INDEXTYPE, INDEXTYPE> indexv = ri.v;
	return pyObjDenseParVec(v(indexv));
}
*/
int64_t pyObjDenseParVec::getnee() const
{
	return __len__();
}
/*
int64_t pyObjDenseParVec::getnnz() const
{
	return v.Count(bind2nd(not_equal_to<doubleint>(), (double)0));
}

int64_t pyObjDenseParVec::getnz() const
{
	return v.Count(bind2nd(equal_to<doubleint>(), (double)0));
}

bool pyObjDenseParVec::any() const
{
	return getnnz() > 0;
}
*/
/*
void pyObjDenseParVec::RandPerm()
{
	v.RandPerm();
}*/
/*
pyObjDenseParVec pyObjDenseParVec::Sort()
{
	pyObjDenseParVec ret(1, 0, 0);
	ret.v = v.sort();
	return ret; // Sort is in-place. The return value is the permutation used.
}

pyObjDenseParVec pyObjDenseParVec::TopK(int64_t k)
{
	FullyDistVec<INDEXTYPE, INDEXTYPE> sel(k);
	sel.iota(k, 0);

	pyObjDenseParVec sorted = copy();
	op::UnaryFunction negate = op::negate();
	sorted.Apply(&negate); // the negation is so that the sort direction is reversed
//	sorted.printall();
	FullyDistVec<INDEXTYPE, INDEXTYPE> perm = sorted.v.sort();
//	sorted.printall();
//	perm.DebugPrint();
	sorted.Apply(&negate);

	// return dense	
	return pyObjDenseParVec(sorted.v(sel));
}
*/

void pyObjDenseParVec::printall()
{
	v.DebugPrint();
}
/*
pyObjDenseParVec pyObjDenseParVec::abs()
{
	pyObjDenseParVec ret = copy();
	ret.Apply(&op::abs());
	return ret;
}

pyObjDenseParVec& pyObjDenseParVec::operator+=(double value)
{
	v.Apply(bind2nd(plus<doubleint>(), doubleint(value)));
	return *this;
}

pyObjDenseParVec pyObjDenseParVec::operator+(double value)
{
	pyObjDenseParVec ret = this->copy();
	ret += value;
	return ret;
}

pyObjDenseParVec& pyObjDenseParVec::operator-=(double value)
{
	v.Apply(bind2nd(minus<doubleint>(), doubleint(value)));
	return *this;
}

pyObjDenseParVec pyObjDenseParVec::operator-(double value)
{
	pyObjDenseParVec ret = this->copy();
	ret -= value;
	return ret;
}

pyObjDenseParVec pyObjDenseParVec::__and__(const pyObjDenseParVec& other)
{
	pyObjDenseParVec ret = copy();
	ret.EWiseApply(other, &op::logical_and());
	return ret;
}
*/
PyObject* pyObjDenseParVec::__getitem__(int64_t key)
{
	return v.GetElement(key);
}

PyObject* pyObjDenseParVec::__getitem__(double  key)
{
	return v.GetElement(static_cast<int64_t>(key));
}
/*
pyObjDenseParVec pyObjDenseParVec::__getitem__(const pyObjDenseParVec& key)
{
	return SubsRef(key);
}*/

void pyObjDenseParVec::__setitem__(int64_t key, PyObject* value)
{
	v.SetElement(key, value);
}

void pyObjDenseParVec::__setitem__(double  key, PyObject* value)
{
	v.SetElement(static_cast<int64_t>(key), value);
}
/*
void pyObjDenseParVec::__setitem__(const pySpParVec& key, const pySpParVec& value)
{
	v.Apply(::set<doubleint>(doubleint(0)), key.v);
	v += value.v;
}

void pyObjDenseParVec::__setitem__(const pySpParVec& key, double value)
{
	v.Apply(::set<doubleint>(value), key.v);
}


pyObjDenseParVec pyObjDenseParVec::range(int64_t howmany, int64_t start)
{
	pyObjDenseParVec ret;
	ret.v.iota(howmany, start-1);
	return ret;
}
*/
