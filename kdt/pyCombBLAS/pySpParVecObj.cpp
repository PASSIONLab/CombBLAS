//#include <mpi.h>

#include <iostream>
#include <math.h>

#include "pySpParVecObj.h"

using namespace std;

pySpParVecObj::pySpParVecObj()
{
}

pySpParVecObj::pySpParVecObj(int64_t size): v(size)
{
}

pySpParVecObj::pySpParVecObj(VectType other): v(other)
{
}

/*
pyDenseParVec pySpParVecObj::dense() const
{
	pyDenseParVec ret(v.TotalLength(), 0, v.GetZero());
	ret.v += v;
	return ret;
}*/

int64_t pySpParVecObj::getnee() const
{
	return v.TotalLength();
}

/*int64_t pySpParVecObj::getnnz() const
{
	return v.Count(bind2nd(not_equal_to<doubleint>(), doubleint(0)));
}*/

int64_t pySpParVecObj::__len__() const
{
	return v.TotalLength();
}

int64_t pySpParVecObj::len() const
{
	return v.TotalLength();
}

/*
pySpParVecObj pySpParVecObj::operator+(const pySpParVecObj& other)
{
	pySpParVecObj ret = copy();
	ret.operator+=(other);
	return ret;
}

pySpParVecObj pySpParVecObj::operator-(const pySpParVecObj& other)
{
	pySpParVecObj ret = copy();
	ret.operator-=(other);
	return ret;
}*/

/*
pySpParVecObj pySpParVecObj::operator+(const pyDenseParVec& other)
{
	pySpParVecObj ret = copy();
	ret.operator+=(other);
	return ret;
}

pySpParVecObj pySpParVecObj::operator-(const pyDenseParVec& other)
{
	pySpParVecObj ret = copy();
	ret.operator-=(other);
	return ret;
}*/

/*
pySpParVecObj& pySpParVecObj::operator+=(const pySpParVecObj& other)
{
	v.operator+=(other.v);

	return *this;
}

pySpParVecObj& pySpParVecObj::operator-=(const pySpParVecObj& other)
{
	v -= other.v;
	return *this;
}*/

/*
pySpParVecObj& pySpParVecObj::operator+=(const pyDenseParVec& other)
{
	pyDenseParVec tmpd = dense();
	tmpd.v += other.v;
	pySpParVecObj tmps = tmpd.sparse();
	this->v.stealFrom(tmps.v);
	return *this;
}

pySpParVecObj& pySpParVecObj::operator-=(const pyDenseParVec& other)
{
	pyDenseParVec tmpd = dense();
	tmpd.v -= other.v;
	pySpParVecObj tmps = tmpd.sparse();
	this->v.stealFrom(tmps.v);

	return *this;
}*/

pySpParVecObj pySpParVecObj::copy()
{
	pySpParVecObj ret(0);
	ret.v = v;
	return ret;
}

bool pySpParVecObj::any() const
{
	return getnee() != 0;
}

bool pySpParVecObj::all() const
{
	return getnee() == v.TotalLength();
}

int64_t pySpParVecObj::intersectSize(const pySpParVecObj& other)
{
	cout << "intersectSize missing CombBLAS piece" << endl;
	return 0;
}

	
void pySpParVecObj::load(const char* filename)
{
	ifstream input(filename);
	v.ReadDistribute(input, 0);
	input.close();
}

void pySpParVecObj::printall()
{
	v.DebugPrint();
}


/////////////////////////

int64_t pySpParVecObj::Count(op::UnaryFunctionObj* op)
{
	return v.Count(*op);
}

/*
pySpParVecObj pySpParVecObj::Find(op::UnaryFunctionObj* op)
{
	pySpParVecObj ret;
	ret->v = v.Find(*op);
	return ret;
}

pyDenseParVec pySpParVecObj::FindInds(op::UnaryFunctionObj* op)
{
	pyDenseParVec ret = new pyDenseParVec();
	ret->v = v.FindInds(*op);
	return ret;
}
*/
void pySpParVecObj::Apply(op::UnaryFunctionObj* op)
{
	v.Apply(*op);
}
/*
void pySpParVecObj::ApplyMasked(op::UnaryFunctionObj* op, const pySpParVecObj& mask)
{
	v.Apply(*op, mask.v);
}
*/

/*
pySpParVecObj pySpParVecObj::SubsRef(const pySpParVecObj& ri)
{
	return pySpParVecObj(v(ri.v));
}*/

/*
pyDenseParVec pySpParVecObj::SubsRef(const pyDenseParVec& ri)
{
	return pyDenseParVec(v(ri.v));
}*/

/*
VERTEXTYPE pySpParVecObj::Reduce(op::BinaryFunction* bf, op::UnaryFunctionObj* uf)
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
}*/


pySpParVecObj pySpParVecObj::Sort()
{
	pySpParVecObj ret(0);
	ret.v = v.sort();
	return ret; // Sort is in-place. The return value is the permutation used.
}

/*
pyDenseParVec pySpParVecObj::TopK(int64_t k)
{
	// FullyDistVec::FullyDistVec(IT glen, NT initval, NT id) 
	FullyDistVec<INDEXTYPE,INDEXTYPE> sel(k, 0, 0);
	
	//void FullyDistVec::iota(IT globalsize, NT first)
	sel.iota(k, v.TotalLength() - k);
	
	FullyDistSpVec<INDEXTYPE,VERTEXTYPE> sorted(v);
	FullyDistSpVec<INDEXTYPE,INDEXTYPE> perm = sorted.sort();
	
	// FullyDistVec FullyDistSpVec::operator(FullyDistVec & v)
	FullyDistVec<INDEXTYPE,INDEXTYPE> topkind = perm(sel);
	FullyDistVec<INDEXTYPE,VERTEXTYPE> topkele = v(topkind);
	//return make_pair(topkind, topkele);

	return pyDenseParVec(topkele);
}*/

void pySpParVecObj::setNumToInd()
{
	v.setNumToInd();
}

pySpParVecObj pySpParVecObj::abs()
{
	pySpParVecObj ret = copy();
	//ret.Apply(&op::abs());
	cout << "abs unsupported" << endl;
	return ret;
}

/*
void pySpParVecObj::__delitem__(const pyDenseParVec& key)
{
	//v = EWiseMult(v, key.v, 1, doubleint(0));
	cout << "UNSUPORTED OP: pySpParVecObjObj::__delitem__(dense vector)" << endl;
}*/

void pySpParVecObj::__delitem__(int64_t key)
{
	v.DelElement(key);
}

VERTEXTYPE pySpParVecObj::__getitem__(int64_t key)
{
	VERTEXTYPE val = v[key];
	
	if (val == v.NOT_FOUND)
	{
		//cout << "Element " << key << " not found." << endl;
		return v.GetZero();
	}
	
	return val;
}

/*VERTEXTYPE pySpParVecObj::__getitem__(double key)
{
	return __getitem__(static_cast<int64_t>(key));
}*/

/*
pySpParVecObj pySpParVecObj::__getitem__(const pySpParVecObj& key)
{
	return SubsRef(key);
}*/

/*
pyDenseParVec pySpParVecObj::__getitem__(const pyDenseParVec& key)
{
	return SubsRef(key);
}*/

void pySpParVecObj::__setitem__(int64_t key, const VERTEXTYPE* value)
{
	v.SetElement(key, *value);
}

/*void pySpParVecObj::__setitem__(double  key, double value)
{
	__setitem__(static_cast<int64_t>(key), value);
}*/

/*
void pySpParVecObj::__setitem__(const pyDenseParVec& key, const pyDenseParVec& value)
{
	if (__len__() != key.__len__())
	{
		cout << "Vector and Key different lengths" << endl;
		// throw
	}
	EWiseMult_inplacefirst(*this, key, 1, 0);
	*this += value;
}*/

void pySpParVecObj::__setitem__(const char* key, const VERTEXTYPE* value)
{
	if (strcmp(key, "existent") == 0)
	{
		v.Apply(::set<VERTEXTYPE>(*value));
	}
	else
	{
		// throw
	}
}

char* pySpParVecObj::__repr__()
{
	printall();
	return " ";
}

/*
pySpParVecObj pySpParVecObj::zeros(int64_t howmany)
{
	return pySpParVecObj(howmany);
}

pySpParVecObj pySpParVecObj::range(int64_t howmany, int64_t start)
{
	pySpParVecObj ret(howmany);
	ret.v.iota(howmany, start-1);
	return ret;
}
*/
