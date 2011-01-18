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

	A = PSpMat_Bool (DEL);	 // conversion from distributed edge list
	PSpMat_Bool AT = A;
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

	A = PSpMat_Bool(DEL);	// remove self loops and duplicates (since this is of type boolean)
	PSpMat_Bool AT = A;
	AT.Transpose();
	A += AT;
	
	MPI::COMM_WORLD.Barrier();
	double t2=MPI_Wtime();
	
	// END OF COPY
	
	k1time = t2-t1;
	pyDegrees.v.stealFrom(degrees);
	return k1time;
}

pyDenseParVec* pySpParMat::GenGraph500Candidates(int howmany)
{
	pyDenseParVec* pyCands = FindIndsOfColsWithSumGreaterThan(1);
		
	FullyDistVec<int64_t,int64_t> First64(A.getcommgrid(), -1);
	pyCands->v.RandPerm();

	First64.iota(howmany, 0);			
	pyCands->v = pyCands->v(First64);		

	
	return pyCands;
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
	int64_t len = 1;
	if (dim == ::Row)
		len = getnrow();
	else
		len = getncol();
		
	pyDenseParVec* ret = new pyDenseParVec(len, identity, identity);

	// make a temporary int matrix
	SpParMat<int64_t, int, SpDCCols<int64_t, int> > * AInt = new SpParMat<int64_t, int, SpDCCols<int64_t, int> >(A);
	
	f->getMPIOp();
	AInt->Reduce(ret->v, (Dim)dim, *f, identity);
	f->releaseMPIOp();
	
	delete AInt;	// delete temporary
	
	return ret;
}



pyDenseParVec* pySpParMat::FindIndsOfColsWithSumGreaterThan(int64_t gt)
{
	pyDenseParVec* ret = new pyDenseParVec();
	FullyDistVec<int64_t, int> ColSums;
	
	// make a temporary int matrix
	SpParMat<int64_t, int, SpDCCols<int64_t, int> > * AInt = new SpParMat<int64_t, int, SpDCCols<int64_t, int> >(A);
	AInt->Reduce(ColSums, ::Row, plus<int>(), 0);
	delete AInt;	// save memory	

	ret->v = ColSums.FindInds(bind2nd(greater<int>(), (int)gt));	// only the indices of connected vertices

	
	//DenseParVec<int64_t, int> ColSums = A.Reduce(Column, plus<int>(), 0);
	//cout << "column sums:------------" << endl;
	//ColSums.DebugPrint();
	//ret->v = ColSums.FindInds(bind2nd(greater<int>(), (int)gt));
	return ret;
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

