#include <mpi.h>

#include <iostream>
#include "pySpParMat.h"

pySpParMat::pySpParMat()
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
}

int64_t upcast(int i) {
	return i;
}

//pyDenseParVec* pySpParMat::Reduce_ColumnSums()
//{
//	pyDenseParVec* ret = new pyDenseParVec();
//	ret->v.stealFrom(A.Reduce(Column, plus<int64_t>(), 0, upcast));
//	return ret;
//}

pySpParVec* pySpParMat::FindIndsOfColsWithSumGreaterThan(int64_t gt)
{
	pySpParVec* ret = new pySpParVec();
	DenseParVec<int64_t, int> ColSums = A.Reduce(Column, plus<int>(), 0); 
	ret->v = ColSums.FindInds(bind2nd(greater<int>(), (int)gt));	
	return ret;
}



pySpParVec* pySpParMat::SpMV_SelMax(const pySpParVec& x)
{
	pySpParVec* ret = new pySpParVec();
	ret->v = SpMV< SelectMaxSRing<int, int64_t > >(A, x.v);
	return ret;
}


void init_pyCombBLAS_MPI()
{
	cout << "calling MPI::Init" << endl;
	MPI::Init();
	/*
	int nprocs = MPI::COMM_WORLD.Get_size();
	int myrank = MPI::COMM_WORLD.Get_rank();
	MPI::COMM_WORLD.Barrier();
	
	int sum = 0;
	int one = 1;
	MPI_Reduce(&one, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD); 

	cout << "I am proc " << myrank << " out of " << nprocs << ". Hear me roar!" << endl;
	if (myrank == 0) {
		cout << "We all reduced our ones to get " << sum;
		if (sum == nprocs)
			cout << ". Success! MPI works." << endl;
		else
			cout << ". SHOULD GET #PROCS! MPI is broken!" << endl;
	}
	*/
}

void finalize() {
	cout << "calling MPI::Finalize" << endl;
	MPI::Finalize();
}
