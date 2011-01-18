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

	FullyDistSpVec<int64_t, int64_t> temp(locsize);
	v = temp;*/
}


//pySpParVec::pySpParVec(const pySpParMat& commSource): v(commSource.A.commGrid);
//{
//}

//pySpParVec::pySpParVec(SpParVec<int64_t, int64_t> & in_v): v(in_v)
//{
//}

pyDenseParVec* pySpParVec::dense() const
{
	pyDenseParVec* ret = new pyDenseParVec(v.TotalLength(), 0);
	ret->v += v;
	return ret;
}


int64_t pySpParVec::getnnz() const
{
	return v.getnnz();
}

int64_t pySpParVec::len() const
{
	return v.TotalLength();
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


/*
void pySpParVec::invert() // "~";  almost equal to logical_not
{
	v.Apply(invert64);
}

void pySpParVec::abs()
{
	v.Apply(abs64);
}
*/

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

	int64_t ret = v.Reduce(*f, *f->getMPIOp(), 0);
	f->releaseMPIOp();
	return ret;
}

pySpParVec* pySpParVec::sort()
{
	pySpParVec* ret = new pySpParVec(0);
	ret->v = v.sort();
	return ret;
}


void pySpParVec::setNumToInd()
{
	v.setNumToInd();
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




