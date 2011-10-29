#include <mpi.h>

#include <iostream>
#include "pyDenseParVec.h"

pyDenseParVec::pyDenseParVec()
{
}

pyDenseParVec::pyDenseParVec(VectType other): v(other)
{
}

pyDenseParVec::pyDenseParVec(int64_t size, double id): v(size, id)
{
}

pyDenseParVec::pyDenseParVec(int64_t size, double init, double zero): v(size, init)
{
}


pySpParVec pyDenseParVec::sparse() const
{
	return pySpParVec(v.Find(bind2nd(not_equal_to<doubleint>(), doubleint(0))));
}

pySpParVec pyDenseParVec::sparse(double zero) const
{
	return pySpParVec(v.Find(bind2nd(not_equal_to<doubleint>(), doubleint(zero))));
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
	op::BinaryFunction m = op::multiplies();
	EWiseApply(rhs, &m);
	return *this;
}

pyDenseParVec& pyDenseParVec::operator*=(const pySpParVec & rhs)
{
	op::BinaryFunction m = op::multiplies();
	EWiseApply(rhs, &m, true, doubleint(0.0));
	return *this;
}

//pyDenseParVec& pyDenseParVec::operator=(const pyDenseParVec & rhs)
//{
//	v.operator=(rhs.v);
//	return *this;
//}

pyDenseParVec pyDenseParVec::operator+(const pyDenseParVec & rhs)
{
	pyDenseParVec ret = this->copy();
	ret += rhs;	
	return ret;
}

pyDenseParVec pyDenseParVec::operator-(const pyDenseParVec & rhs)
{
	pyDenseParVec ret = this->copy();
	ret -= rhs;	
	return ret;
}

pyDenseParVec pyDenseParVec::operator+(const pySpParVec & rhs)
{
	pyDenseParVec ret = this->copy();
	ret += rhs;	
	return ret;
}

pyDenseParVec pyDenseParVec::operator-(const pySpParVec & rhs)
{
	pyDenseParVec ret = this->copy();
	ret -= rhs;	
	return ret;
}

pyDenseParVec pyDenseParVec::operator*(const pyDenseParVec & rhs)
{
	pyDenseParVec ret = this->copy();
	ret *= rhs;	
	return ret;
}

pyDenseParVec pyDenseParVec::operator*(const pySpParVec & rhs)
{
	pyDenseParVec ret = this->copy();
	ret *= rhs;	
	return ret;
}


pyDenseParVec pyDenseParVec::operator==(const pyDenseParVec & rhs)
{
	//return v.operator==(rhs.v);
	pyDenseParVec ret = copy();
	op::BinaryFunction eq = op::equal_to();
	ret.EWiseApply(rhs, &eq);
	return ret;
}

pyDenseParVec pyDenseParVec::operator!=(const pyDenseParVec & rhs)
{
	//return !(v.operator==(rhs.v));
	pyDenseParVec ret = copy();
	op::BinaryFunction ne = op::not_equal_to();
	ret.EWiseApply(rhs, &ne);
	return ret;
}

pyDenseParVec pyDenseParVec::copy()
{
	pyDenseParVec ret;
	ret.v = v;
	return ret;
}

/////////////////////////

int64_t pyDenseParVec::Count(op::UnaryFunction* op)
{
	return v.Count(*op);
}

int64_t pyDenseParVec::Count(op::UnaryFunctionObj* op)
{
	return v.Count(*op);
}

double pyDenseParVec::Reduce(op::BinaryFunction* bf, op::UnaryFunction* uf)
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

double pyDenseParVec::Reduce(op::BinaryFunctionObj* bf, op::UnaryFunctionObj* uf, double init)
{
	if (!bf->associative && root())
		cout << "Attempting to Reduce with a non-associative function! Results will be undefined" << endl;

	doubleint ret;
	
	bf->getMPIOp();
	if (uf == NULL)
		ret = v.Reduce(*bf, doubleint(init), ::identity<doubleint>());
	else
		ret = v.Reduce(*bf, doubleint(init), *uf);
	bf->releaseMPIOp();
	return ret;
}

pySpParVec pyDenseParVec::Find(op::UnaryFunction* op)
{
	return pySpParVec(v.Find(*op));
}

pySpParVec pyDenseParVec::Find(op::UnaryFunctionObj* op)
{
	return pySpParVec(v.Find(*op));
}

pySpParVec pyDenseParVec::__getitem__(op::UnaryFunction* op)
{
	return Find(op);
}

pyDenseParVec pyDenseParVec::FindInds(op::UnaryFunction* op)
{
	pyDenseParVec ret;
	
	FullyDistVec<INDEXTYPE, INDEXTYPE> fi_ret = v.FindInds(*op);
	ret.v = fi_ret;
	return ret;
}

pyDenseParVec pyDenseParVec::FindInds(op::UnaryFunctionObj* op)
{
	pyDenseParVec ret;
	
	FullyDistVec<INDEXTYPE, INDEXTYPE> fi_ret = v.FindInds(*op);
	ret.v = fi_ret;
	return ret;
}

void pyDenseParVec::Apply(op::UnaryFunction* op)
{
	v.Apply(*op);
}

void pyDenseParVec::Apply(op::UnaryFunctionObj* op)
{
	v.Apply(*op);
}

void pyDenseParVec::ApplyMasked(op::UnaryFunction* op, const pySpParVec& mask)
{
	v.Apply(*op, mask.v);
}

void pyDenseParVec::ApplyMasked(op::UnaryFunctionObj* op, const pySpParVec& mask)
{
	v.Apply(*op, mask.v);
}

template <typename T1, typename T2>
bool retTrue(const T1& x, const T2& y)
{
	return true;
}

void pyDenseParVec::EWiseApply(const pyDenseParVec& other, op::BinaryFunction *f)
{
	v.EWiseApply(other.v, *f, retTrue<doubleint, doubleint>);
}

void pyDenseParVec::EWiseApply(const pySpParVec& other, op::BinaryFunction *f, bool doNulls, double nullValue)
{
	v.EWiseApply(other.v, *f, retTrue<doubleint, doubleint>, doNulls, doubleint(nullValue));
}

void pyDenseParVec::EWiseApply(const pyDenseParVecObj1& other, op::BinaryFunctionObj *f, op::BinaryPredicateObj *doOp)
{
	if (doOp != NULL)
		v.EWiseApply(other.v, *f, *doOp);
	else
		v.EWiseApply(other.v, *f, retTrue<doubleint, Obj1>);
}

void pyDenseParVec::EWiseApply(const pyDenseParVecObj2& other, op::BinaryFunctionObj *f, op::BinaryPredicateObj *doOp)
{
	if (doOp != NULL)
		v.EWiseApply(other.v, *f, *doOp);
	else
		v.EWiseApply(other.v, *f, retTrue<doubleint, Obj2>);
}

void pyDenseParVec::EWiseApply(const pyDenseParVec&     other, op::BinaryFunctionObj *f, op::BinaryPredicateObj *doOp)
{
	if (doOp != NULL)
		v.EWiseApply(other.v, *f, *doOp);
	else
		v.EWiseApply(other.v, *f, retTrue<doubleint, doubleint>);
}

void pyDenseParVec::EWiseApply(const pySpParVecObj1& other, op::BinaryFunctionObj *f, op::BinaryPredicateObj *doOp, bool doNulls, Obj1 *nullValue)
{
	if (doOp != NULL)
		v.EWiseApply(other.v, *f, *doOp, doNulls, *nullValue);
	else
		v.EWiseApply(other.v, *f, retTrue<doubleint, Obj1>, doNulls, *nullValue);
}

void pyDenseParVec::EWiseApply(const pySpParVecObj2& other, op::BinaryFunctionObj *f, op::BinaryPredicateObj *doOp, bool doNulls, Obj2 *nullValue)
{
	if (doOp != NULL)
		v.EWiseApply(other.v, *f, *doOp, doNulls, *nullValue);
	else
		v.EWiseApply(other.v, *f, retTrue<doubleint, Obj2>, doNulls, *nullValue);
}

void pyDenseParVec::EWiseApply(const pySpParVec&     other, op::BinaryFunctionObj *f, op::BinaryPredicateObj *doOp, bool doNulls, double nullValue)
{
	if (doOp != NULL)
		v.EWiseApply(other.v, *f, *doOp, doNulls, doubleint(nullValue));
	else
		v.EWiseApply(other.v, *f, retTrue<doubleint, doubleint>, doNulls, doubleint(nullValue));
}

// predicates:
void pyDenseParVec::EWiseApply(const pyDenseParVecObj1& other, op::BinaryPredicateObj *f, op::BinaryPredicateObj *doOp)
{
	if (doOp != NULL)
		v.EWiseApply(other.v, *f, *doOp);
	else
		v.EWiseApply(other.v, *f, retTrue<doubleint, Obj1>);
}

void pyDenseParVec::EWiseApply(const pyDenseParVecObj2& other, op::BinaryPredicateObj *f, op::BinaryPredicateObj *doOp)
{
	if (doOp != NULL)
		v.EWiseApply(other.v, *f, *doOp);
	else
		v.EWiseApply(other.v, *f, retTrue<doubleint, Obj2>);
}

void pyDenseParVec::EWiseApply(const pyDenseParVec&     other, op::BinaryPredicateObj *f, op::BinaryPredicateObj *doOp)
{
	if (doOp != NULL)
		v.EWiseApply(other.v, *f, *doOp);
	else
		v.EWiseApply(other.v, *f, retTrue<doubleint, doubleint>);
}

void pyDenseParVec::EWiseApply(const pySpParVecObj1& other, op::BinaryPredicateObj *f, op::BinaryPredicateObj *doOp, bool doNulls, Obj1 *nullValue)
{
	if (doOp != NULL)
		v.EWiseApply(other.v, *f, *doOp, doNulls, *nullValue);
	else
		v.EWiseApply(other.v, *f, retTrue<doubleint, Obj1>, doNulls, *nullValue);
}

void pyDenseParVec::EWiseApply(const pySpParVecObj2& other, op::BinaryPredicateObj *f, op::BinaryPredicateObj *doOp, bool doNulls, Obj2 *nullValue)
{
	if (doOp != NULL)
		v.EWiseApply(other.v, *f, *doOp, doNulls, *nullValue);
	else
		v.EWiseApply(other.v, *f, retTrue<doubleint, Obj2>, doNulls, *nullValue);
}

void pyDenseParVec::EWiseApply(const pySpParVec&     other, op::BinaryPredicateObj *f, op::BinaryPredicateObj *doOp, bool doNulls, double nullValue)
{
	if (doOp != NULL)
		v.EWiseApply(other.v, *f, *doOp, doNulls, doubleint(nullValue));
	else
		v.EWiseApply(other.v, *f, retTrue<doubleint, doubleint>, doNulls, doubleint(nullValue));
}



pyDenseParVec pyDenseParVec::SubsRef(const pyDenseParVec& ri)
{
	FullyDistVec<INDEXTYPE, INDEXTYPE> indexv = ri.v;
	return pyDenseParVec(v(indexv));
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

pyDenseParVec pyDenseParVec::Sort()
{
	pyDenseParVec ret(1, 0, 0);
	ret.v = v.sort();
	return ret; // Sort is in-place. The return value is the permutation used.
}

pyDenseParVec pyDenseParVec::TopK(int64_t k)
{
	FullyDistVec<INDEXTYPE, INDEXTYPE> sel(k);
	sel.iota(k, 0);

	pyDenseParVec sorted = copy();
	op::UnaryFunction negate = op::negate();
	sorted.Apply(&negate); // the negation is so that the sort direction is reversed
//	sorted.printall();
	FullyDistVec<INDEXTYPE, INDEXTYPE> perm = sorted.v.sort();
//	sorted.printall();
//	perm.DebugPrint();
	sorted.Apply(&negate);

	// return dense	
	return pyDenseParVec(sorted.v(sel));
}

void pyDenseParVec::printall()
{
	v.DebugPrint();
}

pyDenseParVec pyDenseParVec::abs()
{
	pyDenseParVec ret = copy();
	op::UnaryFunction abs = op::abs();
	ret.Apply(&abs);
	return ret;
}

pyDenseParVec& pyDenseParVec::operator+=(double value)
{
	v.Apply(bind2nd(plus<doubleint>(), doubleint(value)));
	return *this;
}

pyDenseParVec pyDenseParVec::operator+(double value)
{
	pyDenseParVec ret = this->copy();
	ret += value;
	return ret;
}

pyDenseParVec& pyDenseParVec::operator-=(double value)
{
	v.Apply(bind2nd(minus<doubleint>(), doubleint(value)));
	return *this;
}

pyDenseParVec pyDenseParVec::operator-(double value)
{
	pyDenseParVec ret = this->copy();
	ret -= value;
	return ret;
}

pyDenseParVec pyDenseParVec::__and__(const pyDenseParVec& other)
{
	pyDenseParVec ret = copy();
	op::BinaryFunction la = op::logical_and();
	ret.EWiseApply(other, &la);
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

pyDenseParVec pyDenseParVec::__getitem__(const pyDenseParVec& key)
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
	v.Apply(::set<doubleint>(doubleint(0)), key.v);
	v += value.v;
}

void pyDenseParVec::__setitem__(const pySpParVec& key, double value)
{
	v.Apply(::set<doubleint>(value), key.v);
}


pyDenseParVec pyDenseParVec::range(int64_t howmany, int64_t start)
{
	pyDenseParVec ret;
	ret.v.iota(howmany, start-1);
	return ret;
}

