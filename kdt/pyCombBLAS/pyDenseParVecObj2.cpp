#include <mpi.h>

#include <iostream>
#include "pyDenseParVecObj2.h"

pyDenseParVecObj2::pyDenseParVecObj2()
{
}

pyDenseParVecObj2::pyDenseParVecObj2(VectType other): v(other)
{
}

pyDenseParVecObj2::pyDenseParVecObj2(int64_t size, Obj2 id): v(size, id, Obj2())
{
}

pySpParVecObj2 pyDenseParVecObj2::sparse(op::UnaryPredicateObj* keep) const
{
	if (keep != NULL)
		return pySpParVecObj2(v.Find(*keep));
	else
		return pySpParVecObj2(v.Find(bind2nd(not_equal_to<Obj2>(), Obj2())));
}

int64_t pyDenseParVecObj2::len() const
{
	return v.TotalLength();
}

int64_t pyDenseParVecObj2::__len__() const
{
	return v.TotalLength();
}
	
void pyDenseParVecObj2::load(const char* filename)
{
	ifstream input(filename);
	v.ReadDistribute(input, 0);
	input.close();
}

/*
pyDenseParVecObj2 pyDenseParVecObj2::operator==(const pyDenseParVecObj2 & rhs)
{
	//return v.operator==(rhs.v);
	pyDenseParVecObj2 ret = copy();
	ret.EWiseApply(rhs, &op::equal_to());
	return ret;
}

pyDenseParVecObj2 pyDenseParVecObj2::operator!=(const pyDenseParVecObj2 & rhs)
{
	//return !(v.operator==(rhs.v));
	pyDenseParVecObj2 ret = copy();
	ret.EWiseApply(rhs, &op::not_equal_to());
	return ret;
}*/

pyDenseParVecObj2 pyDenseParVecObj2::copy()
{
	pyDenseParVecObj2 ret;
	ret.v = v;
	return ret;
}

/////////////////////////

int64_t pyDenseParVecObj2::Count(op::UnaryPredicateObj* op)
{
	return v.Count(*op);
}

Obj2 pyDenseParVecObj2::Reduce(op::BinaryFunctionObj* bf, op::UnaryFunctionObj* uf)
{
	if (!bf->associative && root())
		cout << "Attempting to Reduce with a non-associative function! Results will be undefined" << endl;

	Obj2 ret;
	
	bf->getMPIOp();
	if (uf == NULL)
		ret = v.Reduce(*bf, Obj2(), ::identity<Obj2>());
	else
		ret = v.Reduce(*bf, Obj2(), *uf);
	bf->releaseMPIOp();
	return ret;
}

pySpParVecObj2 pyDenseParVecObj2::Find(op::UnaryPredicateObj* op)
{
	return pySpParVecObj2(v.Find(*op));
}
pySpParVecObj2 pyDenseParVecObj2::__getitem__(op::UnaryPredicateObj* op)
{
	return Find(op);
}

pyDenseParVecObj2 pyDenseParVecObj2::FindInds(op::UnaryPredicateObj* op)
{
	pyDenseParVecObj2 ret;
	
	FullyDistVec<INDEXTYPE, INDEXTYPE> fi_ret = v.FindInds(*op);
	ret.v = fi_ret;
	return ret;
}

void pyDenseParVecObj2::Apply(op::UnaryFunctionObj* op)
{
	v.Apply(*op);
}

void pyDenseParVecObj2::ApplyMasked(op::UnaryFunctionObj* op, const pySpParVec& mask)
{
	v.Apply(*op, mask.v);
}

void pyDenseParVecObj2::EWiseApply(const pyDenseParVecObj2& other, op::BinaryFunctionObj *f)
{
	v.EWiseApply(other.v, *f);
}

void pyDenseParVecObj2::EWiseApply(const pyDenseParVecObj1& other, op::BinaryFunctionObj *f)
{
	v.EWiseApply(other.v, *f);
}

void pyDenseParVecObj2::EWiseApply(const pySpParVecObj2& other, op::BinaryFunctionObj *f, bool doNulls, Obj2 nullValue)
{
	v.EWiseApply(other.v, *f, doNulls, nullValue);
}

void pyDenseParVecObj2::EWiseApply(const pySpParVecObj1& other, op::BinaryFunctionObj *f, bool doNulls, Obj1 nullValue)
{
	v.EWiseApply(other.v, *f, doNulls, nullValue);
}
	
pyDenseParVecObj2 pyDenseParVecObj2::SubsRef(const pyDenseParVec& ri)
{
	FullyDistVec<INDEXTYPE, INDEXTYPE> indexv = ri.v;
	return pyDenseParVecObj2(v(indexv));
}

int64_t pyDenseParVecObj2::getnee() const
{
	return __len__();
}
/*
int64_t pyDenseParVecObj2::getnnz() const
{
	return v.Count(bind2nd(not_equal_to<doubleint>(), (double)0));
}

int64_t pyDenseParVecObj2::getnz() const
{
	return v.Count(bind2nd(equal_to<doubleint>(), (double)0));
}*/
/*
bool pyDenseParVecObj2::any() const
{
	return getnnz() > 0;
}*/

void pyDenseParVecObj2::RandPerm()
{
	v.RandPerm();
}

pyDenseParVec pyDenseParVecObj2::Sort()
{
	pyDenseParVec ret(1, 0);
	ret.v = v.sort();
	return ret; // Sort is in-place. The return value is the permutation used.
}

pyDenseParVecObj2 pyDenseParVecObj2::TopK(int64_t k)
{
	FullyDistVec<INDEXTYPE, INDEXTYPE> sel(k);
	sel.iota(k, 0);

	pyDenseParVecObj2 sorted = copy();
	//op::UnaryFunctionObj negate = op::negate();
	//sorted.Apply(&negate); // the negation is so that the sort direction is reversed
//	sorted.printall();
	FullyDistVec<INDEXTYPE, INDEXTYPE> perm = sorted.v.sort();
//	sorted.printall();
//	perm.DebugPrint();
	//sorted.Apply(&negate);

	// return dense	
	return pyDenseParVecObj2(sorted.v(sel));
}

void pyDenseParVecObj2::printall()
{
	v.DebugPrint();
}

char* pyDenseParVecObj2::__repr__()
{
	static char empty[] = {'\0'};
	printall();
	return empty;
}

Obj2 pyDenseParVecObj2::__getitem__(int64_t key)
{
	return v.GetElement(key);
}

pyDenseParVecObj2 pyDenseParVecObj2::__getitem__(const pyDenseParVec& key)
{
	return SubsRef(key);
}

void pyDenseParVecObj2::__setitem__(int64_t key, Obj2 * value)
{
	v.SetElement(key, *value);
}

/*
void pyDenseParVecObj2::__setitem__(const pySpParVec& key, const pySpParVecObj2& value)
{
	//v.Apply(::set<Obj2>(Obj2()), key.v);
	//v += value.v;
}*/

void pyDenseParVecObj2::__setitem__(const pySpParVec& key, Obj2 * value)
{
	v.Apply(::set<Obj2>(*value), key.v);
}

/*
pyDenseParVecObj2 pyDenseParVecObj2::range(int64_t howmany, int64_t start)
{
	pyDenseParVecObj2 ret;
	ret.v.iota(howmany, start-1);
	return ret;
}
*/
