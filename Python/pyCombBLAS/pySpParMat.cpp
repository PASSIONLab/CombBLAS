//#include <mpi.h>
#include <sys/time.h> 

#include <iostream>
#include "pySpParMat.h"

pySpParMat::pySpParMat()
{
}

pySpParMat::pySpParMat(pySpParMat* copyFrom): A(copyFrom->A)
{
}

pySpParMat::pySpParMat(int64_t m, int64_t n, pyDenseParVec* rows, pyDenseParVec* cols, pyDenseParVec* vals): A(m, n, rows->v, cols->v, vals->v)
{
}


int64_t pySpParMat::getnnz()
{
	return A.getnnz();
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

void pySpParMat::GenGraph500Edges(int scale)
{
	/*
	DistEdgeList<int64_t> DEL;
	
	double a = 0.57;
	double b = 0.19;
	double c = 0.19;
	double d = 1-(a+b+c); // = 0.05
	double abcd[] = {a, b, c, d};
	
	bool mayiprint = false;
	
	if (mayiprint)
		cout << "GenGraph500" << endl;
		
	DEL.GenGraph500Data(abcd, scale, (int64_t)(pow(2., scale)*16));
	
	if (mayiprint)
		cout << "PermEdges" << endl;
	PermEdges<int64_t>(DEL);

	if (mayiprint)
		cout << "RenameVertices" << endl;
	RenameVertices<int64_t>(DEL);
	
	if (mayiprint)
		cout << "Convert To Matrix" << endl;
		
	A = SpParMat<int64_t, int, SpDCCols<int64_t, int> > (DEL);
	*/
	
	int nprocs = MPI::COMM_WORLD.Get_size();
	int rank = MPI::COMM_WORLD.Get_rank();
	
	// this is an undirected graph, so A*x does indeed BFS
	double initiator[4] = {.57, .19, .19, .05};

	DistEdgeList<int64_t> DEL;
	DEL.GenGraph500Data(initiator, scale, 16 * ((int64_t) std::pow(2.0, (double) scale)) / nprocs );
	PermEdges<int64_t>(DEL);
	RenameVertices<int64_t>(DEL);

	A = MatType (DEL);	 // conversion from distributed edge list
	MatType AT = A;
	AT.Transpose();
	
	int64_t nnz = A.getnnz();
	if (rank == 0)
		cout << "Generator: A.getnnz() before A += AT: " << nnz << endl;
	A += AT;
	nnz = A.getnnz();
	if (rank == 0)
		cout << "Generator: A.getnnz() after A += AT: " << nnz << endl;
}

double pySpParMat::GenGraph500Edges(int scale, pyDenseParVec& pyDegrees)
{
	double k1time = 0;
	FullyDistVec<int64_t, int64_t> degrees;

	int nprocs = MPI::COMM_WORLD.Get_size();
	int rank = MPI::COMM_WORLD.Get_rank();


	// COPIED FROM AYDIN'S C++ GRAPH500 CODE ------------
	// this is an undirected graph, so A*x does indeed BFS
	double initiator[4] = {.57, .19, .19, .05};

	DistEdgeList<int64_t> DEL;
	DEL.GenGraph500Data(initiator, scale, 16 * ((int64_t) std::pow(2.0, (double) scale)) / nprocs );
	PermEdges<int64_t>(DEL);
	RenameVertices<int64_t>(DEL);

	PSpMat_Int64 * G = new PSpMat_Int64(DEL, false);	 // conversion from distributed edge list, keep self-loops
	degrees = G->Reduce(::Column, plus<int64_t>(), 0); 
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

pySpParMat* pySpParMat::copy()
{
	pySpParMat* ret = new pySpParMat(this);
	return ret;
}

void pySpParMat::Apply(op::UnaryFunction* op)
{
	A.Apply(*op);
}

void pySpParMat::Prune(op::UnaryFunction* op)
{
	A.Prune(*op);
}
	
pyDenseParVec* pySpParMat::Reduce(int dim, op::BinaryFunction* f, int64_t identity)
{
	return Reduce(dim, f, NULL, identity);
}

pyDenseParVec* pySpParMat::Reduce(int dim, op::BinaryFunction* bf, op::UnaryFunction* uf, int64_t identity)
{
	int64_t len = 1;
	if (dim == ::Row)
		len = getnrow();
	else
		len = getncol();
		
	pyDenseParVec* ret = new pyDenseParVec(len, identity, identity);

	// make a temporary int matrix
	//SpParMat<int64_t, int, SpDCCols<int64_t, int> > * AInt = new SpParMat<int64_t, int, SpDCCols<int64_t, int> >(A);
	
	bf->getMPIOp();
	if (uf == NULL)
		A.Reduce(ret->v, (Dim)dim, *bf, identity);
	else
		A.Reduce(ret->v, (Dim)dim, *bf, identity, *uf);
	bf->releaseMPIOp();
	
	//delete AInt;	// delete temporary
	
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
	A.Find(outrows->v, outcols->v, outvals->v);
}

pySpParVec* pySpParMat::SpMV_PlusTimes(const pySpParVec& x)
{
	pySpParVec* ret = new pySpParVec();
	FullyDistSpVec<int64_t, int64_t> result = SpMV< PlusTimesSRing<bool, int64_t > >(A, x.v);
	ret->v.stealFrom(result);
	return ret;
}

pySpParVec* pySpParMat::SpMV_SelMax(const pySpParVec& x)
{
	pySpParVec* ret = new pySpParVec();
	FullyDistSpVec<int64_t, int64_t> result = SpMV< SelectMaxSRing<bool, int64_t > >(A, x.v);
	ret->v.stealFrom(result);
	return ret;
}

void pySpParMat::SpMV_SelMax_inplace(pySpParVec& x)
{
	x.v = SpMV< SelectMaxSRing<bool, int64_t > >(A, x.v);
}

