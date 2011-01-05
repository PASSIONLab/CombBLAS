#include <mpi.h>

#include <iostream>
#include "pyDenseParVec.h"

pyDenseParVec::pyDenseParVec()
{
}

pyDenseParVec::pyDenseParVec(int64_t size, int64_t id): v(size, id, 0)
{
	/*MPI::Intracomm comm = v.getCommGrid()->GetDiagWorld();
	
	int64_t locsize = 0;
	
	if (comm != MPI::COMM_NULL)
	{
		int nprocs = comm.Get_size();
		int dgrank = comm.Get_rank();
		locsize = (int64_t)floor(static_cast<double>(size)/static_cast<double>(nprocs));
		
		if (dgrank == nprocs-1)
		{
			// this may be shorter than the others
			locsize = size - locsize*(nprocs-1);
		}
	}

	FullyDistVec<int64_t, int64_t> temp(locsize, id, 0);
	v.stealFrom(temp);*/
}

pyDenseParVec::pyDenseParVec(int64_t size, int64_t init, int64_t zero): v(size, init, zero)
{
	/*
	MPI::Intracomm comm = v.getCommGrid()->GetDiagWorld();
	
	int64_t locsize = 0;
	
	if (comm != MPI::COMM_NULL)
	{
		int nprocs = comm.Get_size();
		int dgrank = comm.Get_rank();
		locsize = (int64_t)floor(static_cast<double>(size)/static_cast<double>(nprocs));
		
		if (dgrank == nprocs-1)
		{
			// this may be shorter than the others
			locsize = size - locsize*(nprocs-1);
		}
	}

	FullyDistVec<int64_t, int64_t> temp(locsize, init, zero);
	v.stealFrom(temp);*/
}

//pyDenseParVec::pyDenseParVec(const pySpParMat& commSource, int64_t zero)
//{
//}

pySpParVec* pyDenseParVec::sparse() const
{
	pySpParVec* ret = new pySpParVec(0);
	ret->v = v.Find(bind2nd(not_equal_to<int64_t>(), (int64_t)0));
	return ret;
}

pySpParVec* pyDenseParVec::sparse(int64_t zero) const
{
	pySpParVec* ret = new pySpParVec(0);
	
	ret->v = v.Find(bind2nd(not_equal_to<int64_t>(), zero));
	return ret;
}

int64_t pyDenseParVec::len() const
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

//pyDenseParVec& pyDenseParVec::operator=(const pyDenseParVec & rhs)
//{
//	v.operator=(rhs.v);
//	return *this;
//}

pyDenseParVec* pyDenseParVec::operator+(const pyDenseParVec & rhs)
{
	pyDenseParVec* ret = this->copy();
	*(ret) += rhs;	
	return ret;
}

pyDenseParVec* pyDenseParVec::operator-(const pyDenseParVec & rhs)
{
	pyDenseParVec* ret = this->copy();
	*(ret) -= rhs;	
	return ret;
}

pyDenseParVec* pyDenseParVec::operator+(const pySpParVec & rhs)
{
	pyDenseParVec* ret = this->copy();
	*(ret) += rhs;	
	return ret;
}

pyDenseParVec* pyDenseParVec::operator-(const pySpParVec & rhs)
{
	pyDenseParVec* ret = this->copy();
	*(ret) -= rhs;	
	return ret;
}



pyDenseParVec* pyDenseParVec::copy()
{
	pyDenseParVec* ret = new pyDenseParVec();
	ret->v = v;
	return ret;
}

int64_t pyDenseParVec::Count_GreaterThan(int64_t value)
{
	return v.Count(bind2nd(greater<int64_t>(), value));
}

pySpParVec* pyDenseParVec::Find_totality()
{
	pySpParVec* ret = new pySpParVec();
	ret->v = v.Find(totality<int64_t>());
	return ret;
}

pySpParVec* pyDenseParVec::Find_GreaterThan(int64_t value)
{
	pySpParVec* ret = new pySpParVec();
	ret->v = v.Find(bind2nd(greater<int64_t>(), value));
	return ret;
}

pySpParVec* pyDenseParVec::Find_NotEqual(int64_t value)
{
	pySpParVec* ret = new pySpParVec();
	ret->v = v.Find(bind2nd(not_equal_to<int64_t>(), value));
	return ret;
}

pyDenseParVec* pyDenseParVec::FindInds_GreaterThan(int64_t value)
{
	pyDenseParVec* ret = new pyDenseParVec();
	ret->v = v.FindInds(bind2nd(greater<int64_t>(), value));
	return ret;
}

pyDenseParVec* pyDenseParVec::FindInds_NotEqual(int64_t value)
{
	pyDenseParVec* ret = new pyDenseParVec();
	ret->v = v.FindInds(bind2nd(not_equal_to<int64_t>(), value));
	return ret;
}
	
pyDenseParVec* pyDenseParVec::SubsRef(const pyDenseParVec& ri)
{
	pyDenseParVec* ret = new pyDenseParVec();
	ret->v = v(ri.v);
	return ret;
}


void pyDenseParVec::invert() // "~";  almost equal to logical_not
{
	v.Apply(invert64);
}


void pyDenseParVec::abs()
{
	v.Apply(abs64);
}

void pyDenseParVec::negate()
{
	v.Apply(negate64);
}

int64_t pyDenseParVec::getnnz() const
{
	return v.Count(bind2nd(not_equal_to<int64_t>(), (int64_t)0));
}

int64_t pyDenseParVec::getnz() const
{
	return v.Count(bind2nd(equal_to<int64_t>(), (int64_t)0));
}

void pyDenseParVec::SetElement (int64_t indx, int64_t numx)	// element-wise assignment
{
	v.SetElement(indx, numx);
}

int64_t pyDenseParVec::GetElement (int64_t indx)	// element-wise fetch
{
	return v.GetElement(indx);
}

void pyDenseParVec::RandPerm()
{
	v.RandPerm();
}

void pyDenseParVec::printall()
{
	v.DebugPrint();
}

void pyDenseParVec::ApplyMasked_SetTo(const pySpParVec& mask, int64_t value)
{
	v.Apply(set<int64_t>(value), mask.v);
}

pyDenseParVec* pyDenseParVec::range(int64_t howmany, int64_t start)
{
	pyDenseParVec* ret = new pyDenseParVec();
	ret->v.iota(howmany, start);
	return ret;
}
