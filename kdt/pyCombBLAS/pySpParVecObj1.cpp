//#include <mpi.h>

#include <iostream>
#include <math.h>

#include "pySpParVecObj1.h"

using namespace std;

pySpParVecObj1::pySpParVecObj1()
{
}

pySpParVecObj1::pySpParVecObj1(int64_t size): v(size)
{
}

pySpParVecObj1::pySpParVecObj1(VectType other): v(other)
{
}

/*
pyDenseParVec pySpParVecObj1::dense() const
{
	pyDenseParVec ret(v.TotalLength(), 0, v.GetZero());
	ret.v += v;
	return ret;
}*/

int64_t pySpParVecObj1::getnee() const
{
	return v.TotalLength();
}

/*int64_t pySpParVecObj1::getnnz() const
{
	return v.Count(bind2nd(not_equal_to<doubleint>(), doubleint(0)));
}*/

int64_t pySpParVecObj1::__len__() const
{
	return v.TotalLength();
}

int64_t pySpParVecObj1::len() const
{
	return v.TotalLength();
}

/*
pySpParVecObj1 pySpParVecObj1::operator+(const pySpParVecObj1& other)
{
	pySpParVecObj1 ret = copy();
	ret.operator+=(other);
	return ret;
}

pySpParVecObj1 pySpParVecObj1::operator-(const pySpParVecObj1& other)
{
	pySpParVecObj1 ret = copy();
	ret.operator-=(other);
	return ret;
}*/

/*
pySpParVecObj1 pySpParVecObj1::operator+(const pyDenseParVec& other)
{
	pySpParVecObj1 ret = copy();
	ret.operator+=(other);
	return ret;
}

pySpParVecObj1 pySpParVecObj1::operator-(const pyDenseParVec& other)
{
	pySpParVecObj1 ret = copy();
	ret.operator-=(other);
	return ret;
}*/

/*
pySpParVecObj1& pySpParVecObj1::operator+=(const pySpParVecObj1& other)
{
	v.operator+=(other.v);

	return *this;
}

pySpParVecObj1& pySpParVecObj1::operator-=(const pySpParVecObj1& other)
{
	v -= other.v;
	return *this;
}*/

/*
pySpParVecObj1& pySpParVecObj1::operator+=(const pyDenseParVec& other)
{
	pyDenseParVec tmpd = dense();
	tmpd.v += other.v;
	pySpParVecObj1 tmps = tmpd.sparse();
	this->v.stealFrom(tmps.v);
	return *this;
}

pySpParVecObj1& pySpParVecObj1::operator-=(const pyDenseParVec& other)
{
	pyDenseParVec tmpd = dense();
	tmpd.v -= other.v;
	pySpParVecObj1 tmps = tmpd.sparse();
	this->v.stealFrom(tmps.v);

	return *this;
}*/

pySpParVecObj1 pySpParVecObj1::copy()
{
	pySpParVecObj1 ret(0);
	ret.v = v;
	return ret;
}

bool pySpParVecObj1::any() const
{
	return getnee() != 0;
}

bool pySpParVecObj1::all() const
{
	return getnee() == v.TotalLength();
}

int64_t pySpParVecObj1::intersectSize(const pySpParVecObj1& other)
{
	cout << "intersectSize missing CombBLAS piece" << endl;
	return 0;
}

	
void pySpParVecObj1::load(const char* filename)
{
	ifstream input(filename);
	v.ReadDistribute(input, 0);
	input.close();
}

void pySpParVecObj1::printall()
{
	v.DebugPrint();
}


/////////////////////////

int64_t pySpParVecObj1::Count(op::UnaryPredicateObj* op)
{
	return v.Count(*op);
}

/*
pySpParVecObj1 pySpParVecObj1::Find(op::UnaryFunctionObj* op)
{
	pySpParVecObj1 ret;
	ret->v = v.Find(*op);
	return ret;
}

pyDenseParVec pySpParVecObj1::FindInds(op::UnaryFunctionObj* op)
{
	pyDenseParVec ret = new pyDenseParVec();
	ret->v = v.FindInds(*op);
	return ret;
}
*/
void pySpParVecObj1::Apply(op::UnaryFunctionObj* op)
{
	v.Apply(*op);
}
/*
void pySpParVecObj1::ApplyMasked(op::UnaryFunctionObj* op, const pySpParVecObj1& mask)
{
	v.Apply(*op, mask.v);
}
*/

/*
pySpParVecObj1 pySpParVecObj1::SubsRef(const pySpParVecObj1& ri)
{
	return pySpParVecObj1(v(ri.v));
}*/

/*
pyDenseParVec pySpParVecObj1::SubsRef(const pyDenseParVec& ri)
{
	return pyDenseParVec(v(ri.v));
}*/


Obj1 pySpParVecObj1::Reduce(op::BinaryFunctionObj* bf, op::UnaryFunctionObj* uf)
{
	if (!bf->associative && root())
		cout << "Attempting to Reduce with a non-associative function! Results will be undefined" << endl;

	Obj1 ret;
	
	bf->getMPIOp();
	if (uf == NULL)
		ret = v.Reduce(*bf, Obj1(), ::identity<Obj1>());
	else
		ret = v.Reduce(*bf, Obj1(), *uf);
	bf->releaseMPIOp();
	return ret;
}

/*
pySpParVecObj1 pySpParVecObj1::Sort()
{
	pySpParVecObj1 ret(0);
	ret.v = v.sort();
	return ret; // Sort is in-place. The return value is the permutation used.
}*/

/*
pyDenseParVec pySpParVecObj1::TopK(int64_t k)
{
	// FullyDistVec::FullyDistVec(IT glen, NT initval, NT id) 
	FullyDistVec<INDEXTYPE,INDEXTYPE> sel(k, 0, 0);
	
	//void FullyDistVec::iota(IT globalsize, NT first)
	sel.iota(k, v.TotalLength() - k);
	
	FullyDistSpVec<INDEXTYPE,Obj1> sorted(v);
	FullyDistSpVec<INDEXTYPE,INDEXTYPE> perm = sorted.sort();
	
	// FullyDistVec FullyDistSpVec::operator(FullyDistVec & v)
	FullyDistVec<INDEXTYPE,INDEXTYPE> topkind = perm(sel);
	FullyDistVec<INDEXTYPE,Obj1> topkele = v(topkind);
	//return make_pair(topkind, topkele);

	return pyDenseParVec(topkele);
}*/

void pySpParVecObj1::setNumToInd()
{
	v.setNumToInd();
}

/*
void pySpParVecObj1::__delitem__(const pyDenseParVec& key)
{
	//v = EWiseMult(v, key.v, 1, doubleint(0));
	cout << "UNSUPORTED OP: pySpParVecObj1Obj::__delitem__(dense vector)" << endl;
}*/

void pySpParVecObj1::__delitem__(int64_t key)
{
	v.DelElement(key);
}

Obj1 pySpParVecObj1::__getitem__(int64_t key)
{
	Obj1 val = v[key];
	
	if (val == v.NOT_FOUND)
	{
		//cout << "Element " << key << " not found." << endl;
		return v.GetZero();
	}
	
	return val;
}

/*Obj1 pySpParVecObj1::__getitem__(double key)
{
	return __getitem__(static_cast<int64_t>(key));
}*/

/*
pySpParVecObj1 pySpParVecObj1::__getitem__(const pySpParVecObj1& key)
{
	return SubsRef(key);
}*/

/*
pyDenseParVec pySpParVecObj1::__getitem__(const pyDenseParVec& key)
{
	return SubsRef(key);
}*/

void pySpParVecObj1::__setitem__(int64_t key, const Obj1* value)
{
	v.SetElement(key, *value);
}

/*void pySpParVecObj1::__setitem__(double  key, double value)
{
	__setitem__(static_cast<int64_t>(key), value);
}*/

/*
void pySpParVecObj1::__setitem__(const pyDenseParVec& key, const pyDenseParVec& value)
{
	if (__len__() != key.__len__())
	{
		cout << "Vector and Key different lengths" << endl;
		// throw
	}
	EWiseMult_inplacefirst(*this, key, 1, 0);
	*this += value;
}*/

void pySpParVecObj1::__setitem__(const char* key, const Obj1* value)
{
	if (strcmp(key, "existent") == 0)
	{
		v.Apply(::set<Obj1>(*value));
	}
	else
	{
		// throw
	}
}

char* pySpParVecObj1::__repr__()
{
	printall();
	return " ";
}

pySpParVecObj1 EWiseApply(const pySpParVecObj1& a, const pySpParVecObj1& b, op::BinaryFunctionObj* op, bool allowANulls, bool allowBNulls)
{
	return pySpParVecObj1(EWiseApply(a.v, b.v, *op, allowANulls, allowBNulls));
}

/*
pySpParVecObj1 pySpParVecObj1::zeros(int64_t howmany)
{
	return pySpParVecObj1(howmany);
}

pySpParVecObj1 pySpParVecObj1::range(int64_t howmany, int64_t start)
{
	pySpParVecObj1 ret(howmany);
	ret.v.iota(howmany, start-1);
	return ret;
}
*/
