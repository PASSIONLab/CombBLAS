#include <mpi.h>

#include <iostream>
#include "pyDenseParVec.h"

pyDenseParVec::pyDenseParVec(): v(commGrid)
{
}

pyDenseParVec::pyDenseParVec(VectType other): v(other)
{
}

pyDenseParVec::pyDenseParVec(int64_t size, double id): v(commGrid, size, id)
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

void pyDenseParVec::save(const char* filename)
{
	ofstream output(filename);
	v.SaveGathered(output, 0);
	output.close();
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

int64_t pyDenseParVec::Count(op::UnaryPredicateObj* op)
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
		ret = v.Reduce(*bf, doubleint(), ::identity<doubleint>());
	else
		ret = v.Reduce(*bf, doubleint(), *uf);
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

pySpParVec pyDenseParVec::Find(op::UnaryPredicateObj* op)
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

pyDenseParVec pyDenseParVec::FindInds(op::UnaryPredicateObj* op)
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

void pyDenseParVec::ApplyInd(op::BinaryFunctionObj* op)
{
	v.ApplyInd(*op);
}

void pyDenseParVec::ApplyMasked(op::UnaryFunction* op, const pySpParVec& mask)
{
	v.Apply(*op, mask.v);
}

void pyDenseParVec::ApplyMasked(op::UnaryFunctionObj* op, const pySpParVec& mask)
{
	v.Apply(*op, mask.v);
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

void pyDenseParVec::RandPerm()
{
	v.RandPerm();
}

pyDenseParVec pyDenseParVec::Sort()
{
	pyDenseParVec ret(1, 0);
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

void pyDenseParVec::SelectCandidates(double nvert, bool deterministic)
{
	v.SelectCandidates(nvert);
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
	v.Apply(::myset<doubleint>(doubleint(0)), key.v);
	v += value.v;
}

void pyDenseParVec::__setitem__(const pySpParVec& key, double value)
{
	v.Apply(::myset<doubleint>(value), key.v);
}


pyDenseParVec pyDenseParVec::range(int64_t howmany, int64_t start)
{
	pyDenseParVec ret;
	ret.v.iota(howmany, start-1);
	return ret;
}

#define VECCLASS pyDenseParVec
#define DENSE_VEC
//#define OBJ_VEC

#include "pyCommonVecFuncs.cpp"
