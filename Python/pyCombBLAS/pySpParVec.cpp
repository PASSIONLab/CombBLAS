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

pyDenseParVec* pySpParVec::dense() const
{
	pyDenseParVec* ret = new pyDenseParVec(v.TotalLength(), 0);
	ret->v += v;
	return ret;
}

int64_t pySpParVec::getne() const
{
	return v.getnnz();
}

int64_t pySpParVec::getnnz() const
{
	//return Count(bind2nd<int64_t>(not_equal_to<int64_t>(), 0));
	return v.getnnz();
}

int64_t pySpParVec::__len__() const
{
	return v.TotalLength();
}

int64_t pySpParVec::len() const
{
	return v.TotalLength();
}

pySpParVec* pySpParVec::operator+(const pySpParVec& other)
{
	pySpParVec* ret = copy();
	ret->operator+=(other);
	return ret;
}

pySpParVec* pySpParVec::operator-(const pySpParVec& other)
{
	pySpParVec* ret = copy();
	ret->operator-=(other);
	return ret;
}

pySpParVec* pySpParVec::operator+(const pyDenseParVec& other)
{
	pySpParVec* ret = copy();
	ret->operator+=(other);
	return ret;
}

pySpParVec* pySpParVec::operator-(const pyDenseParVec& other)
{
	pySpParVec* ret = copy();
	ret->operator-=(other);
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
	pyDenseParVec* tmpd = dense();
	tmpd->v += other.v;
	pySpParVec* tmps = tmpd->sparse();
	this->v.stealFrom(tmps->v);
	delete tmpd;
	delete tmps;
	return *this;
}

pySpParVec& pySpParVec::operator-=(const pyDenseParVec& other)
{
	pyDenseParVec* tmpd = dense();
	tmpd->v -= other.v;
	pySpParVec* tmps = tmpd->sparse();
	this->v.stealFrom(tmps->v);
	delete tmpd;
	delete tmps;
	return *this;
}

void pySpParVec::SetElement(int64_t index, int64_t numx)	// element-wise assignment
{
	v.SetElement(index, numx);
}

int64_t pySpParVec::GetElement(int64_t index)
{
	int64_t val = v[index];
	
	if (val == v.NOT_FOUND)
	{
		cout << "Element " << index << " not found." << endl;
	}
	
	return val;
}


pySpParVec* pySpParVec::copy()
{
	pySpParVec* ret = new pySpParVec(0);
	ret->v = v;
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

/*
pyDenseParVec* pySpParVec::FindInds_GreaterThan(int64_t value)
{
	pyDenseParVec* ret = new pyDenseParVec();
	
	// cheapen out and use the dense version for now
	pyDenseParVec* temp = dense();
	ret->v = temp->v.FindInds(bind2nd(greater<int64_t>(), value));
	delete temp;
	return ret;
}

pyDenseParVec* pySpParVec::FindInds_NotEqual(int64_t value)
{
	pyDenseParVec* ret = new pyDenseParVec();
	
	// cheapen out and use the dense version for now
	pyDenseParVec* temp = dense();
	ret->v = temp->v.FindInds(bind2nd(not_equal_to<int64_t>(), value));
	delete temp;
	return ret;
}

void pySpParVec::Apply_SetTo(int64_t value)
{
	v.Apply(set<int64_t>(value));
}

*/

/////////////////////////
/*
int64_t pySpParVec::Count(op::UnaryFunction* op)
{
	return v.Count(*op);
}

pySpParVec* pySpParVec::Find(op::UnaryFunction* op)
{
	pySpParVec* ret = new pySpParVec();
	ret->v = v.Find(*op);
	return ret;
}

pyDenseParVec* pySpParVec::FindInds(op::UnaryFunction* op)
{
	pyDenseParVec* ret = new pyDenseParVec();
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


pySpParVec* pySpParVec::SubsRef(const pySpParVec& ri)
{
	pySpParVec* ret = new pySpParVec(0);
	ret->v = v(ri.v);
	return ret;
}


/*int64_t pySpParVec::Reduce_sum()
{
	return v.Reduce(plus<int64_t>(), 0);
}*/

int64_t pySpParVec::Reduce(op::BinaryFunction* f)
{
	if (!f->associative && root())
		cout << "Attempting to Reduce with a non-associative function! Results will be undefined" << endl;

	f->getMPIOp();
	int64_t ret = v.Reduce(*f, 0);
	f->releaseMPIOp();
	return ret;
}

pySpParVec* pySpParVec::Sort()
{
	pySpParVec* ret = new pySpParVec(0);
	ret->v = v.sort();
	return ret;
}


void pySpParVec::setNumToInd()
{
	v.setNumToInd();
}

pySpParVec* pySpParVec::abs()
{
	pySpParVec* ret = copy();
	op::UnaryFunction* a = op::abs();
	ret->Apply(a);
	delete a;
	return ret;
}

void pySpParVec::__delitem__(const pyDenseParVec& key)
{
}

void pySpParVec::__delitem__(int64_t key)
{
}

int64_t pySpParVec::__getitem__(int64_t key)
{
	return GetElement(key);
}

pySpParVec* pySpParVec::__getitem__(const pySpParVec& key)
{
	return SubsRef(key);
}

void pySpParVec::__setitem__(int64_t key, int64_t value)
{
	SetElement(key, value);
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

void pySpParVec::__setitem__(const char* key, int64_t value)
{
	if (strcmp(key, "existent") == 0)
	{
		v.Apply(::set<int64_t>(value));
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


pySpParVec* pySpParVec::zeros(int64_t howmany)
{
	pySpParVec* ret = new pySpParVec(howmany);
	return ret;
}

pySpParVec* pySpParVec::range(int64_t howmany, int64_t start)
{
	pySpParVec* ret = new pySpParVec(howmany);
	ret->v.iota(howmany, start);
	return ret;
}




