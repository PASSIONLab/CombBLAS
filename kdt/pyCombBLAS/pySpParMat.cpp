//#include <mpi.h>
#include <sys/time.h> 

#include <iostream>
#include "pySpParMat.h"

pySpParMat::pySpParMat()
{
}

pySpParMat::pySpParMat(const pySpParMat& copyFrom): A(copyFrom.A)
{
}

pySpParMat::pySpParMat(MatType other): A(other)
{
}

pySpParMat::pySpParMat(int64_t m, int64_t n, pyDenseParVec* rows, pyDenseParVec* cols, pyDenseParVec* vals)
{
	FullyDistVec<int64_t, doubleint> irow = rows->v;
	FullyDistVec<int64_t, doubleint> icol = cols->v;
	A = MatType(m, n, irow, icol, vals->v);
}

int64_t pySpParMat::getnee()
{
	return A.getnnz();
}

int64_t pySpParMat::getnnz()
{
	// actually count the number of nonzeros

	op::BinaryFunction ne = op::not_equal_to();
	op::UnaryFunction ne0 = op::bind2nd(ne, 0);
	return Count(&ne0);
}

int64_t pySpParMat::getnrow()
{
	return A.getnrow();
}

int64_t pySpParMat::getncol()
{
	return A.getncol();
}
	
void pySpParMat::load(const char* filename)
{
	ifstream input(filename);
	A.ReadDistribute(input, 0);
	input.close();
}

void pySpParMat::save(const char* filename)
{
	ofstream output(filename);
	A.put(output);
	output.close();
}

double pySpParMat::GenGraph500Edges(int scale)
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
double pySpParMat::GenGraph500Edges(int scale, pyDenseParVec& pyDegrees)
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
pySpParMat pySpParMat::copy()
{
	return pySpParMat(*this);
}

pySpParMat& pySpParMat::operator+=(const pySpParMat& other)
{
	A += other.A;
	return *this;
}

pySpParMat& pySpParMat::assign(const pySpParMat& other)
{
	A = other.A;
	return *this;
}

pySpParMat pySpParMat::operator*(const pySpParMat& other)
{
	return SpMM(other);
}

pySpParMat pySpParMat::SpMM(const pySpParMat& other)
{
	return pySpParMat( Mult_AnXBn_Synch<PlusTimesSRing<doubleint, doubleint > >(A, other.A) );
}

pySpParMat pySpParMat::__getitem__(const pyDenseParVec& rows, const pyDenseParVec& cols) const
{
	return SubsRef(rows, cols);
}

pySpParMat pySpParMat::SubsRef(const pyDenseParVec& rows, const pyDenseParVec& cols) const
{
	return pySpParMat(A(rows.v, cols.v));
}
	
int64_t pySpParMat::removeSelfLoops()
{
	return A.RemoveLoops();
}

void pySpParMat::Apply(op::UnaryFunction* op)
{
	A.Apply(*op);
}

void pySpParMat::ColWiseApply(const pySpParVec& values, op::BinaryFunction* f)
{
	::ColWiseApply(A, values.v, *f);
}

pySpParMat EWiseMult(const pySpParMat& A1, const pySpParMat& A2, bool exclude)
{
	return pySpParMat(EWiseMult(A1.A, A2.A, exclude));
}

pySpParMat EWiseApply(const pySpParMat& A, const pySpParMat& B, op::BinaryFunction *bf, bool notB, double defaultBValue)
{
	return pySpParMat(EWiseApply(A.A, B.A, *bf, notB, doubleint(defaultBValue)));
}

void pySpParMat::Prune(op::UnaryFunction* op)
{
	A.Prune(*op);
}

int64_t pySpParMat::Count(op::UnaryFunction* pred)
{
	// use Reduce to count along the columns, then reduce the result vector into one value
	op::BinaryFunction p = op::plus();
	return static_cast<int64_t>(Reduce(Column(), &p, pred, 0).Reduce(&p));
}

	
pyDenseParVec pySpParMat::Reduce(int dim, op::BinaryFunction* f, double identity)
{
	return Reduce(dim, f, NULL, identity);
}

pyDenseParVec pySpParMat::Reduce(int dim, op::BinaryFunction* bf, op::UnaryFunction* uf, double identity)
{
	int64_t len = 1;
	if (dim == ::Row)
		len = getnrow();
	else
		len = getncol();
		
	pyDenseParVec ret(len, identity, identity);

	bf->getMPIOp();
	if (uf == NULL)
		A.Reduce(ret.v, (Dim)dim, *bf, doubleint(identity));
	else
		A.Reduce(ret.v, (Dim)dim, *bf, doubleint(identity), *uf);
	bf->releaseMPIOp();
	
	return ret;
}

void pySpParMat::Transpose()
{
	A.Transpose();
}

/*void pySpParMat::EWiseMult(pySpParMat* rhs, bool exclude)
{
	A.EWiseMult(rhs->A, exclude);
}*/

void pySpParMat::Find(pyDenseParVec* outrows, pyDenseParVec* outcols, pyDenseParVec* outvals) const
{
	FullyDistVec<int64_t, int64_t> irows, icols;
	A.Find(irows, icols, outvals->v);
	outrows->v = irows;
	outcols->v = icols;
	//A.Find(outrows->v, outcols->v, outvals->v);
}

pySpParVec pySpParMat::SpMV_PlusTimes(const pySpParVec& x)
{
	return pySpParVec( SpMV< PlusTimesSRing<doubleint, doubleint > >(A, x.v) );
}

pySpParVec pySpParMat::SpMV_SelMax(const pySpParVec& x)
{
	return pySpParVec( SpMV< SelectMaxSRing<doubleint, doubleint > >(A, x.v) );
}

void pySpParMat::SpMV_SelMax_inplace(pySpParVec& x)
{
	x.v = SpMV< SelectMaxSRing<doubleint, doubleint > >(A, x.v);
}


