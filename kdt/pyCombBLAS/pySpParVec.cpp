//#include <mpi.h>

#include <iostream>
#include <math.h>

#include "pySpParVec.h"

using namespace std;

pySpParVec::pySpParVec()
{
}

pySpParVec::pySpParVec(int64_t size): v(size)
{
}

pySpParVec::pySpParVec(VectType other): v(other)
{
}

pyDenseParVec pySpParVec::dense() const
{
	pyDenseParVec ret(v.TotalLength(), 0, v.GetZero());
	ret.v += v;
	return ret;
}

int64_t pySpParVec::getnee() const
{
	return v.getnnz();
}

int64_t pySpParVec::getnnz() const
{
	return v.Count(bind2nd(not_equal_to<doubleint>(), doubleint(0)));
}

int64_t pySpParVec::__len__() const
{
	return v.TotalLength();
}

int64_t pySpParVec::len() const
{
	return v.TotalLength();
}

pySpParVec pySpParVec::operator+(const pySpParVec& other)
{
	pySpParVec ret = copy();
	ret.operator+=(other);
	return ret;
}

pySpParVec pySpParVec::operator-(const pySpParVec& other)
{
	pySpParVec ret = copy();
	ret.operator-=(other);
	return ret;
}

pySpParVec pySpParVec::operator+(const pyDenseParVec& other)
{
	pySpParVec ret = copy();
	ret.operator+=(other);
	return ret;
}

pySpParVec pySpParVec::operator-(const pyDenseParVec& other)
{
	pySpParVec ret = copy();
	ret.operator-=(other);
	return ret;
}

pySpParVec& pySpParVec::operator+=(const pySpParVec& other)
{
	v.operator+=(other.v);

	return *this;
}

pySpParVec& pySpParVec::operator-=(const pySpParVec& other)
{
	v -= other.v;
	return *this;
}

pySpParVec& pySpParVec::operator+=(const pyDenseParVec& other)
{
	pyDenseParVec tmpd = dense();
	tmpd.v += other.v;
	pySpParVec tmps = tmpd.sparse();
	this->v.stealFrom(tmps.v);
	return *this;
}

pySpParVec& pySpParVec::operator-=(const pyDenseParVec& other)
{
	pyDenseParVec tmpd = dense();
	tmpd.v -= other.v;
	pySpParVec tmps = tmpd.sparse();
	this->v.stealFrom(tmps.v);

	return *this;
}

pySpParVec pySpParVec::copy()
{
	pySpParVec ret(0);
	ret.v = v;
	return ret;
}

bool pySpParVec::any() const
{
	return getnnz() != 0;
}

bool pySpParVec::all() const
{
	return getnnz() == v.TotalLength();
}

int64_t pySpParVec::intersectSize(const pySpParVec& other)
{
	cout << "intersectSize missing CombBLAS piece" << endl;
	return 0;
}

	
void pySpParVec::load(const char* filename)
{
	ifstream input(filename);
	v.ReadDistribute(input, 0);
	input.close();
}

void pySpParVec::printall()
{
	v.DebugPrint();
}


/////////////////////////

int64_t pySpParVec::Count(op::UnaryFunction* op)
{
	return v.Count(*op);
}

/*
pySpParVec pySpParVec::Find(op::UnaryFunction* op)
{
	pySpParVec ret;
	ret->v = v.Find(*op);
	return ret;
}

pyDenseParVec pySpParVec::FindInds(op::UnaryFunction* op)
{
	pyDenseParVec ret = new pyDenseParVec();
	ret->v = v.FindInds(*op);
	return ret;
}
*/
void pySpParVec::Apply(op::UnaryFunction* op)
{
	v.Apply(*op);
}
/*
void pySpParVec::ApplyMasked(op::UnaryFunction* op, const pySpParVec& mask)
{
	v.Apply(*op, mask.v);
}
*/

/*
pySpParVec pySpParVec::SubsRef(const pySpParVec& ri)
{
	return pySpParVec(v(ri.v));
}*/

pyDenseParVec pySpParVec::SubsRef(const pyDenseParVec& ri)
{
	return pyDenseParVec(v(ri.v));
}


double pySpParVec::Reduce(op::BinaryFunction* bf, op::UnaryFunction* uf)
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


pySpParVec pySpParVec::Sort()
{
	pySpParVec ret(0);
	ret.v = v.sort();
	return ret; // Sort is in-place. The return value is the permutation used.
}

pyDenseParVec pySpParVec::TopK(int64_t k)
{
	// FullyDistVec::FullyDistVec(IT glen, NT initval, NT id) 
	FullyDistVec<INDEXTYPE,INDEXTYPE> sel(k, 0, 0);
	
	//void FullyDistVec::iota(IT globalsize, NT first)
	sel.iota(k, v.TotalLength() - k);
	
	FullyDistSpVec<INDEXTYPE,doubleint> sorted(v);
	FullyDistSpVec<INDEXTYPE,INDEXTYPE> perm = sorted.sort();
	
	// FullyDistVec FullyDistSpVec::operator(FullyDistVec & v)
	FullyDistVec<INDEXTYPE,INDEXTYPE> topkind = perm(sel);
	FullyDistVec<INDEXTYPE,doubleint> topkele = v(topkind);
	//return make_pair(topkind, topkele);

	return pyDenseParVec(topkele);


/*
	FullyDistSpVec<INDEXTYPE, INDEXTYPE> sel(k);
	sel.iota(k, 0);

	pySpParVec sorted = copy();
	op::UnaryFunction negate = op::negate();
	sorted.Apply(&negate); // the negation is so that the sort direction is reversed
//	sorted.printall();
	FullyDistSpVec<INDEXTYPE, INDEXTYPE> perm = sorted.v.sort();
//	sorted.printall();
//	perm.DebugPrint();
	sorted.Apply(&negate);

	// return dense	
	return pySpParVec(sorted.v(sel));
*/
}

void pySpParVec::setNumToInd()
{
	v.setNumToInd();
}

pySpParVec pySpParVec::abs()
{
	pySpParVec ret = copy();
	ret.Apply(&op::abs());
	return ret;
}

void pySpParVec::__delitem__(const pyDenseParVec& key)
{
	v = EWiseMult(v, key.v, 1, doubleint(0));
}

void pySpParVec::__delitem__(int64_t key)
{
	v.DelElement(key);
}

double pySpParVec::__getitem__(int64_t key)
{
	doubleint val = v[key];
	
	if (val == v.NOT_FOUND)
	{
		//cout << "Element " << key << " not found." << endl;
		return v.GetZero();
	}
	
	return val;
}

double pySpParVec::__getitem__(double key)
{
	return __getitem__(static_cast<int64_t>(key));
}

/*
pySpParVec pySpParVec::__getitem__(const pySpParVec& key)
{
	return SubsRef(key);
}*/

pyDenseParVec pySpParVec::__getitem__(const pyDenseParVec& key)
{
	return SubsRef(key);
}

void pySpParVec::__setitem__(int64_t key, double value)
{
	v.SetElement(key, doubleint(value));
}

void pySpParVec::__setitem__(double  key, double value)
{
	__setitem__(static_cast<int64_t>(key), value);
}

void pySpParVec::__setitem__(const pyDenseParVec& key, const pyDenseParVec& value)
{
/*
			if self.pySPV.len() != key.pyDPV.len():
				raise KeyError, 'Vector and Key different lengths';
			tmp = PyDenseParVec(self.len(),0);
			tmp = self.dense();
			pcb.EWiseMult_inplacefirst(self.pySPV, key.pyDPV, True, 0);
			tmp.pyDPV += value.pyDPV;
			self.pySPV = tmp.sparse().pySPV;
*/
	if (__len__() != key.__len__())
	{
		cout << "Vector and Key different lengths" << endl;
		// throw
	}
	EWiseMult_inplacefirst(*this, key, 1, 0);
	*this += value;
}

void pySpParVec::__setitem__(const char* key, double value)
{
	if (strcmp(key, "existent") == 0)
	{
		v.Apply(::set<doubleint>(doubleint(value)));
	}
	else
	{
		// throw
	}
}

char* pySpParVec::__repr__()
{
	printall();
	return " ";
}


pySpParVec pySpParVec::zeros(int64_t howmany)
{
	return pySpParVec(howmany);
}

pySpParVec pySpParVec::range(int64_t howmany, int64_t start)
{
	pySpParVec ret(howmany);
	ret.v.iota(howmany, start-1);
	return ret;
}




