#include <mpi.h>

#include <iostream>
#include "pyDenseParVecObj2.h"

pyDenseParVecObj2::pyDenseParVecObj2(): v(commGrid)
{
}

pyDenseParVecObj2::pyDenseParVecObj2(VectType other): v(other)
{
}

pyDenseParVecObj2::pyDenseParVecObj2(int64_t size, Obj2 id): v(commGrid, size, id)
{
}


bool returnTrue(const Obj2& o)
{
	return true;
}

pySpParVecObj2 pyDenseParVecObj2::sparse(op::UnaryPredicateObj* keep) const
{
	if (keep != NULL)
		return pySpParVecObj2(v.Find(*keep));
	else
		return pySpParVecObj2(v.Find(returnTrue));
}

int64_t pyDenseParVecObj2::len() const
{
	return v.TotalLength();
}

int64_t pyDenseParVecObj2::__len__() const
{
	return v.TotalLength();
}
	
class Obj2ReadSaveHandlerD
{
public:
	Obj2 getNoNum(pySpParVecObj2::INDEXTYPE row, pySpParVecObj2::INDEXTYPE col) { return Obj2(); }

	template <typename c, typename t>
	Obj2 read(std::basic_istream<c,t>& is, pySpParVecObj2::INDEXTYPE index)
	{
		Obj2 ret;
		ret.loadCpp(is, index, 0);
		return ret;
	}

	template <typename c, typename t>
	void save(std::basic_ostream<c,t>& os, const Obj2& v, pySpParVecObj2::INDEXTYPE index)
	{
		v.saveCpp(os);
	}
};

void pyDenseParVecObj2::load(const char* filename)
{
	ifstream input(filename);
	v.ReadDistribute(input, 0, Obj2ReadSaveHandlerD());
	input.close();
}

void pyDenseParVecObj2::save(const char* filename)
{
	ofstream output(filename);
	v.SaveGathered(output, 0, Obj2ReadSaveHandlerD());
	output.close();
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

Obj2 pyDenseParVecObj2::Reduce(op::BinaryFunctionObj* bf, op::UnaryFunctionObj* uf, Obj2 *init)
{
	if (!bf->associative && root())
		cout << "Attempting to Reduce with a non-associative function! Results will be undefined" << endl;

	Obj2 ret;
	
	bf->getMPIOp();
	if (uf == NULL)
		ret = v.Reduce(*bf, *init, ::identity<Obj2>());
	else
		ret = v.Reduce(*bf, *init, *uf);
	bf->releaseMPIOp();
	return ret;
}

double pyDenseParVecObj2::Reduce(op::BinaryFunctionObj* bf, op::UnaryFunctionObj* uf, double init)
{
	if (!bf->associative && root())
		cout << "Attempting to Reduce with a non-associative function! Results will be undefined" << endl;

	double ret;
	
	bf->getMPIOp();
	if (uf == NULL)
	{
		ret = 0;
		cout << "unary operation must be specified when changing types!" << endl;
	}
	else
		ret = v.Reduce(*bf, doubleint(init), *uf);
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

pyDenseParVec pyDenseParVecObj2::FindInds(op::UnaryPredicateObj* op)
{
	pyDenseParVec ret(0, 0);
	
	FullyDistVec<INDEXTYPE, INDEXTYPE> fi_ret = v.FindInds(*op);
	ret.v = fi_ret;
	return ret;
}

void pyDenseParVecObj2::Apply(op::UnaryFunctionObj* op)
{
	v.Apply(*op);
}

void pyDenseParVecObj2::ApplyInd(op::BinaryFunctionObj* op)
{
	v.ApplyInd(*op);
}

void pyDenseParVecObj2::ApplyMasked(op::UnaryFunctionObj* op, const pySpParVec& mask)
{
	v.Apply(*op, mask.v);
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
	//v.Apply(::myset<Obj2>(Obj2()), key.v);
	//v += value.v;
}*/

void pyDenseParVecObj2::__setitem__(const pySpParVec& key, Obj2 * value)
{
	v.Apply(::myset<Obj2>(*value), key.v);
}

/*
pyDenseParVecObj2 pyDenseParVecObj2::range(int64_t howmany, int64_t start)
{
	pyDenseParVecObj2 ret;
	ret.v.iota(howmany, start-1);
	return ret;
}
*/

#define VECCLASS pyDenseParVecObj2
#define DENSE_VEC
#define OBJ_VEC

#include "pyCommonVecFuncs.cpp"
