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

pyDenseParVecObj1 pySpParVecObj1::dense() const
{
	pyDenseParVecObj1 ret(v.TotalLength(), Obj1());
	ret.v.EWiseApply(v, use2nd<Obj1>(), false, Obj1());
	return ret;
}

int64_t pySpParVecObj1::getnee() const
{
	return v.getnnz();
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

class Obj1ReadSaveHandler
{
public:
	Obj1 getNoNum(pySpParVecObj1::INDEXTYPE row, pySpParVecObj1::INDEXTYPE col) { return Obj1(); }

	template <typename c, typename t>
	Obj1 read(std::basic_istream<c,t>& is, pySpParVecObj1::INDEXTYPE index)
	{
		Obj1 ret;
		ret.loadCpp(is, index, 0);
		return ret;
	}

	template <typename c, typename t>
	void save(std::basic_ostream<c,t>& os, const Obj1& v, pySpParVecObj1::INDEXTYPE index)
	{
		v.saveCpp(os);
	}
};

void pySpParVecObj1::load(const char* filename)
{
	ifstream input(filename);
	v.ReadDistribute(input, 0, Obj1ReadSaveHandler());
	input.close();
}

void pySpParVecObj1::save(const char* filename)
{
	ofstream output(filename);
	v.SaveGathered(output, 0, Obj1ReadSaveHandler());
	output.close();
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

void pySpParVecObj1::ApplyInd(op::BinaryFunctionObj* op)
{
	v.ApplyInd(*op);
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

pyDenseParVecObj1 pySpParVecObj1::SubsRef(const pyDenseParVec& ri)
{
	return pyDenseParVecObj1(v(ri.v));
}


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

pySpParVec pySpParVecObj1::Sort()
{
	pySpParVec ret(0);
	ret.v = v.sort();
	return ret; // Sort is in-place. The return value is the permutation used.
}

pyDenseParVecObj1 pySpParVecObj1::TopK(int64_t k)
{
	//// FullyDistVec::FullyDistVec(IT glen, NT initval, NT id) 
	//FullyDistVec<INDEXTYPE,INDEXTYPE> sel(k, 0, 0);
	
	////void FullyDistVec::iota(IT globalsize, NT first)
	//sel.iota(k, v.TotalLength() - k);
	
	FullyDistSpVec<INDEXTYPE,Obj1> sorted(v);
	FullyDistSpVec<INDEXTYPE,INDEXTYPE> perm = sorted.sort();

	FullyDistVec<INDEXTYPE,INDEXTYPE> sel(k, 0);
	sel.iota(k, perm.getnnz() - k);
	
	// FullyDistVec FullyDistSpVec::operator(FullyDistVec & v)
	FullyDistVec<INDEXTYPE,INDEXTYPE> topkind = perm(sel);
	cout << "perm: " << endl;
	perm.DebugPrint();
	cout << "sel: " << endl;
	sel.DebugPrint();
	cout << "topkind: " << endl;
	topkind.DebugPrint();
	
	FullyDistVec<INDEXTYPE,Obj1> topkele = v(topkind);
	//return make_pair(topkind, topkele);

	return pyDenseParVecObj1(topkele);
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
	
	if (!v.WasFound())
	{
		//cout << "Element " << key << " not found." << endl;
	}
	
	return val;
}

Obj1 pySpParVecObj1::__getitem__(double key)
{
	return __getitem__(static_cast<int64_t>(key));
}

/*
pySpParVecObj1 pySpParVecObj1::__getitem__(const pySpParVec& key)
{
	return SubsRef(key);
}*/

pyDenseParVecObj1 pySpParVecObj1::__getitem__(const pyDenseParVec& key)
{
	return SubsRef(key);
}

void pySpParVecObj1::__setitem__(int64_t key, const Obj1* value)
{
	v.SetElement(key, *value);
}

void pySpParVecObj1::__setitem__(double  key, const Obj1* value)
{
	__setitem__(static_cast<int64_t>(key), value);
}

/*
void pySpParVecObj1::__setitem__(const pyDenseParVec& key, const pyDenseParVecObj1& value)
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
	static char empty[] = {'\0'};
	printall();
	return empty;
}

template <typename T1, typename T2>
bool retTrue(const T1& x, const T2& y)
{
	return true;
}

pySpParVecObj1 EWiseApply(const pySpParVecObj1& a, const pySpParVecObj1& b, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, Obj1 ANull, Obj1 BNull)
{
	if (doOp != NULL)
		return pySpParVecObj1(EWiseApply<Obj1>(a.v, b.v, *op, *doOp, allowANulls, allowBNulls, ANull, BNull));
	else
		return pySpParVecObj1(EWiseApply<Obj1>(a.v, b.v, *op, retTrue<Obj1, Obj1>, allowANulls, allowBNulls, ANull, BNull));
}

pySpParVecObj1 EWiseApply(const pySpParVecObj1& a, const pySpParVecObj2& b, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, Obj1 ANull, Obj2 BNull)
{
	if (doOp != NULL)
		return pySpParVecObj1(EWiseApply<Obj1>(a.v, b.v, *op, *doOp, allowANulls, allowBNulls, ANull, BNull));
	else
		return pySpParVecObj1(EWiseApply<Obj1>(a.v, b.v, *op, retTrue<Obj1, Obj2>, allowANulls, allowBNulls, ANull, BNull));
}

pySpParVecObj1 EWiseApply(const pySpParVecObj1& a, const pySpParVec&     b, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, Obj1 ANull, double BNull)
{
	if (doOp != NULL)
		return pySpParVecObj1(EWiseApply<Obj1>(a.v, b.v, *op, *doOp, allowANulls, allowBNulls, ANull, doubleint(BNull)));
	else
		return pySpParVecObj1(EWiseApply<Obj1>(a.v, b.v, *op, retTrue<Obj1, doubleint>, allowANulls, allowBNulls, ANull, doubleint(BNull)));
}

pySpParVec EWiseApply(const pySpParVecObj1& a, const pySpParVecObj1& b, op::BinaryPredicateObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, Obj1 ANull, Obj1 BNull)
{
	if (doOp != NULL)
		return pySpParVec(EWiseApply<doubleint>(a.v, b.v, *op, *doOp, allowANulls, allowBNulls, ANull, BNull));
	else
		return pySpParVec(EWiseApply<doubleint>(a.v, b.v, *op, retTrue<Obj1, Obj1>, allowANulls, allowBNulls, ANull, BNull));
}

pySpParVec EWiseApply(const pySpParVecObj1& a, const pySpParVecObj2& b, op::BinaryPredicateObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, Obj1 ANull, Obj2 BNull)
{
	if (doOp != NULL)
		return pySpParVec(EWiseApply<doubleint>(a.v, b.v, *op, *doOp, allowANulls, allowBNulls, ANull, BNull));
	else
		return pySpParVec(EWiseApply<doubleint>(a.v, b.v, *op, retTrue<Obj1, Obj2>, allowANulls, allowBNulls, ANull, BNull));
}

pySpParVec EWiseApply(const pySpParVecObj1& a, const pySpParVec&     b, op::BinaryPredicateObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, Obj1 ANull, double BNull)
{
	if (doOp != NULL)
		return pySpParVec(EWiseApply<doubleint>(a.v, b.v, *op, *doOp, allowANulls, allowBNulls, ANull, doubleint(BNull)));
	else
		return pySpParVec(EWiseApply<doubleint>(a.v, b.v, *op, retTrue<Obj1, doubleint>, allowANulls, allowBNulls, ANull, doubleint(BNull)));
}


/////////// with Dense

pySpParVecObj1 EWiseApply(const pySpParVecObj1& a, const pyDenseParVec& b, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, Obj1 ANull)
{
	if (doOp != NULL)
		return pySpParVecObj1(EWiseApply<Obj1>(a.v, b.v, *op, *doOp, allowANulls, ANull));
	else
		return pySpParVecObj1(EWiseApply<Obj1>(a.v, b.v, *op, retTrue<Obj1, doubleint>, allowANulls, ANull));
}

pySpParVecObj1 EWiseApply(const pySpParVecObj1& a, const pyDenseParVecObj1& b, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, Obj1 ANull)
{
	if (doOp != NULL)
		return pySpParVecObj1(EWiseApply<Obj1>(a.v, b.v, *op, *doOp, allowANulls, ANull));
	else
		return pySpParVecObj1(EWiseApply<Obj1>(a.v, b.v, *op, retTrue<Obj1, Obj1>, allowANulls, ANull));
}

pySpParVecObj1 EWiseApply(const pySpParVecObj1& a, const pyDenseParVecObj2& b, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, Obj1 ANull)
{
	if (doOp != NULL)
		return pySpParVecObj1(EWiseApply<Obj1>(a.v, b.v, *op, *doOp, allowANulls, ANull));
	else
		return pySpParVecObj1(EWiseApply<Obj1>(a.v, b.v, *op, retTrue<Obj1, Obj2>, allowANulls, ANull));
}

pySpParVec EWiseApply(const pySpParVecObj1& a, const pyDenseParVec& b, op::BinaryPredicateObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, Obj1 ANull)
{
	if (doOp != NULL)
		return pySpParVec(EWiseApply<doubleint>(a.v, b.v, *op, *doOp, allowANulls, ANull));
	else
		return pySpParVec(EWiseApply<doubleint>(a.v, b.v, *op, retTrue<Obj1, doubleint>, allowANulls, ANull));
}

pySpParVec EWiseApply(const pySpParVecObj1& a, const pyDenseParVecObj1& b, op::BinaryPredicateObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, Obj1 ANull)
{
	if (doOp != NULL)
		return pySpParVec(EWiseApply<doubleint>(a.v, b.v, *op, *doOp, allowANulls, ANull));
	else
		return pySpParVec(EWiseApply<doubleint>(a.v, b.v, *op, retTrue<Obj1, Obj1>, allowANulls, ANull));
}

pySpParVec EWiseApply(const pySpParVecObj1& a, const pyDenseParVecObj2& b, op::BinaryPredicateObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, Obj1 ANull)
{
	if (doOp != NULL)
		return pySpParVec(EWiseApply<doubleint>(a.v, b.v, *op, *doOp, allowANulls, ANull));
	else
		return pySpParVec(EWiseApply<doubleint>(a.v, b.v, *op, retTrue<Obj1, Obj2>, allowANulls, ANull));
}

/*
pySpParVecObj1 pySpParVecObj1::zeros(int64_t howmany)
{
	return pySpParVecObj1(howmany);
}*/

/*
pySpParVecObj1 pySpParVecObj1::range(int64_t howmany, int64_t start)
{
	pySpParVecObj1 ret(howmany);
	ret.v.iota(howmany, start-1);
	return ret;
}
*/
