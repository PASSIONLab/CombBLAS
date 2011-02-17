//#include <mpi.h>
#include <sys/time.h> 

#include <iostream>
#include "pySpParMatBool.h"


pySpParMatBool* EWiseMult(const pySpParMatBool& A1, const pySpParMatBool& A2, bool exclude)
{
	pySpParMatBool* ret = new pySpParMatBool();
	ret->A = EWiseMult(A1.A, A2.A, exclude);
	return ret;
}

pySpParMatBool::pySpParMatBool()
{
}

pySpParMatBool::pySpParMatBool(pySpParMatBool* copyFrom): A(copyFrom->A)
{
}

pySpParMatBool::pySpParMatBool(int64_t m, int64_t n, pyDenseParVec* rows, pyDenseParVec* cols, pyDenseParVec* vals)
{
	FullyDistVec<int64_t, doubleint> irow = rows->v;
	FullyDistVec<int64_t, doubleint> icol = cols->v;
	A = MatType(m, n, irow, icol, vals->v);
}

pySpParMatBool::pySpParMatBool(const pySpParMat& copyFrom)
{
	A = copyFrom.A;
}


int64_t pySpParMatBool::getnee()
{
	return A.getnnz();
}

int64_t pySpParMatBool::getnnz()
{
	// actually count the number of nonzeros

	op::BinaryFunction *p = op::plus();
	op::BinaryFunction *ne = op::not_equal_to();
	op::UnaryFunction *ne0 = op::bind2nd(ne, 0);
	pyDenseParVec* colsums = Reduce(Column(), p, ne0, 0);

	int64_t ret = static_cast<int64_t>(colsums->Reduce(p));

	delete colsums;
	delete ne0;
	delete ne;
	delete p;
	
	return ret;
}

int64_t pySpParMatBool::getnrow()
{
	return A.getnrow();
}

int64_t pySpParMatBool::getncol()
{
	return A.getncol();
}
	
void pySpParMatBool::load(const char* filename)
{
	ifstream input(filename);
	A.ReadDistribute(input, 0);
	input.close();
}

void pySpParMatBool::save(const char* filename)
{
	ofstream output(filename);
	A.put(output);
	output.close();
}

double pySpParMatBool::GenGraph500Edges(int scale)
{
	double k1time = 0;

	int nprocs = MPI::COMM_WORLD.Get_size();


	// COPIED FROM AYDIN'S C++ GRAPH500 CODE ------------
	// this is an undirected graph, so A*x does indeed BFS
	double initiator[4] = {.57, .19, .19, .05};

	DistEdgeList<int64_t> DEL;
	DEL.GenGraph500Data(initiator, scale, 16 * ((int64_t) std::pow(2.0, (double) scale)) / nprocs );
	PermEdges<int64_t>(DEL);
	RenameVertices<int64_t>(DEL);

	// Start Kernel #1
	MPI::COMM_WORLD.Barrier();
	double t1 = MPI_Wtime();

	A = MatType(DEL);	// remove self loops and duplicates (since this is of type boolean)
	MatType AT = A;
	AT.Transpose();
	A += AT;
	
	MPI::COMM_WORLD.Barrier();
	double t2=MPI_Wtime();
	
	// END OF COPY
	
	k1time = t2-t1;
	return k1time;
}

/*
double pySpParMatBool::GenGraph500Edges(int scale, pyDenseParVec& pyDegrees)
{
	double k1time = 0;
	FullyDistVec<INDEXTYPE, doubleint> degrees;

	int nprocs = MPI::COMM_WORLD.Get_size();
	//int rank = MPI::COMM_WORLD.Get_rank();


	// COPIED FROM AYDIN'S C++ GRAPH500 CODE ------------
	// this is an undirected graph, so A*x does indeed BFS
	double initiator[4] = {.57, .19, .19, .05};

	DistEdgeList<int64_t> DEL;
	DEL.GenGraph500Data(initiator, scale, 16 * ((int64_t) std::pow(2.0, (double) scale)) / nprocs );
	PermEdges<int64_t>(DEL);
	RenameVertices<int64_t>(DEL);

	PSpMat_DoubleInt * G = new PSpMat_DoubleInt(DEL, false);	 // conversion from distributed edge list, keep self-loops
	
	op::BinaryFunction* p = op::plus();
	p->getMPIOp();
	degrees = G->Reduce(::Column, *p, doubleint(0)); 
	p->releaseMPIOp();
	delete p;
	
	delete G;

	// Start Kernel #1
	MPI::COMM_WORLD.Barrier();
	double t1 = MPI_Wtime();

	A = MatType(DEL);	// remove self loops and duplicates (since this is of type boolean)
	MatType AT = A;
	AT.Transpose();
	A += AT;
	
	MPI::COMM_WORLD.Barrier();
	double t2=MPI_Wtime();
	
	// END OF COPY
	
	k1time = t2-t1;
	pyDegrees.v.stealFrom(degrees);
	return k1time;
}
*/
pySpParMatBool* pySpParMatBool::copy()
{
	pySpParMatBool* ret = new pySpParMatBool(this);
	return ret;
}

pySpParMatBool& pySpParMatBool::operator+=(const pySpParMatBool& other)
{
	A += other.A;
	return *this;
}

pySpParMatBool& pySpParMatBool::assign(const pySpParMatBool& other)
{
	A = other.A;
	return *this;
}

pySpParMatBool* pySpParMatBool::operator*(const pySpParMatBool& other)
{
	return SpMM(other);
}

pySpParMatBool* pySpParMatBool::SpMM(const pySpParMatBool& other)
{
	pySpParMatBool* ret = new pySpParMatBool();
	ret->A = Mult_AnXBn_Synch<PlusTimesSRing<bool, bool > >(A, other.A);
	return ret;
}

pySpParMatBool* pySpParMatBool::__getitem__(const pyDenseParVec& rows, const pyDenseParVec& cols) const
{
	return SubsRef(rows, cols);
}

pySpParMatBool* pySpParMatBool::SubsRef(const pyDenseParVec& rows, const pyDenseParVec& cols) const
{
	pySpParMatBool* ret = new pySpParMatBool();
	ret->A = A(rows.v, cols.v);
	return ret;
}
	
int64_t pySpParMatBool::removeSelfLoops()
{
	return A.RemoveLoops();
}

void pySpParMatBool::Apply(op::UnaryFunction* op)
{
	A.Apply(*op);
}


void pySpParMatBool::ColWiseApply(const pySpParVec& x, op::BinaryFunction* f)
{
	::ColWiseApply(A, x.v, *f);
}


pySpParMatBool* EWiseApply(const pySpParMatBool& A, const pySpParMatBool& B, op::BinaryFunction *bf, bool notB, double defaultBValue)
{
	pySpParMatBool* ret = new pySpParMatBool();
	ret->A = EWiseApply(A.A, B.A, *bf, notB, bool(defaultBValue));
	return ret;
}

void pySpParMatBool::Prune(op::UnaryFunction* op)
{
	A.Prune(*op);
}

int64_t pySpParMatBool::Count(op::UnaryFunction* pred)
{
	// use Reduce to count along the columns, then reduce the result vector into one value
	op::BinaryFunction *p = op::plus();
	pyDenseParVec* colsums = Reduce(Column(), p, pred, 0);

	int64_t ret = static_cast<int64_t>(colsums->Reduce(p));

	delete colsums;
	delete p;
	
	return ret;
}

	
pyDenseParVec* pySpParMatBool::Reduce(int dim, op::BinaryFunction* f, double identity)
{
	return Reduce(dim, f, NULL, identity);
}

pyDenseParVec* pySpParMatBool::Reduce(int dim, op::BinaryFunction* bf, op::UnaryFunction* uf, double identity)
{
	int64_t len = 1;
	if (dim == ::Row)
		len = getnrow();
	else
		len = getncol();
		
	pyDenseParVec* ret = new pyDenseParVec(len, identity, identity);
	FullyDistVec<INDEXTYPE, doubleint> tmp;
	
	// Make a temporary graph	
	PSpMat_DoubleInt * G = new PSpMat_DoubleInt(A);
	
	bf->getMPIOp();
	if (uf == NULL)
		G->Reduce(tmp, (Dim)dim, *bf, doubleint(identity));
	else
		G->Reduce(tmp, (Dim)dim, *bf, doubleint(identity), *uf);
	bf->releaseMPIOp();

	delete G;
	
	ret->v = tmp;
	
	return ret;
}

void pySpParMatBool::Transpose()
{
	A.Transpose();
}

/*void pySpParMatBool::EWiseMult(pySpParMatBool* rhs, bool exclude)
{
	A.EWiseMult(rhs->A, exclude);
}*/

void pySpParMatBool::Find(pyDenseParVec* outrows, pyDenseParVec* outcols, pyDenseParVec* outvals) const
{
	FullyDistVec<int64_t, int64_t> irows, icols;
	FullyDistVec<int64_t, bool> vals;
	A.Find(irows, icols, vals);
	outrows->v = irows;
	outcols->v = icols;
	outvals->v = vals;
	/*
	cout << "Find::vals:  ";
	for (int i = 0; i < vals.TotalLength(); i++)
	{
		bool v = vals.GetElement(i);
		cout << v << ", ";
	}*/
	//A.Find(outrows->v, outcols->v, outvals->v);
}

pySpParVec* pySpParMatBool::SpMV_PlusTimes(const pySpParVec& x)
{
	pySpParVec* ret = new pySpParVec();
	ret->v = SpMV< PlusTimesSRing<bool, doubleint > >(A, x.v);
	return ret;
}

pySpParVec* pySpParMatBool::SpMV_SelMax(const pySpParVec& x)
{
	pySpParVec* ret = new pySpParVec();
	ret->v = SpMV< SelectMaxSRing<bool, doubleint > >(A, x.v);
	return ret;
}

void pySpParMatBool::SpMV_SelMax_inplace(pySpParVec& x)
{
	x.v = SpMV< SelectMaxSRing<bool, doubleint> >(A, x.v);
}

