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
	
	// this is an undirected graph, so A*x does indeed BFS
	double initiator[4] = {.57, .19, .19, .05};

	DistEdgeList<int64_t> DEL;
	DEL.GenGraph500Data(initiator, scale, 16 * ((int64_t) std::pow(2.0, (double) scale)) / nprocs );
	PermEdges<int64_t>(DEL);
	RenameVertices<int64_t>(DEL);

	A = SpParMat<int64_t, bool, SpDCCols<int64_t, bool> > (DEL);	 // conversion from distributed edge list
	SpParMat<int64_t, bool, SpDCCols<int64_t, bool> > AT = A;
	AT.Transpose();
	A += AT;
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

void pySpParMat::Apply_SetTo(int64_t value)
{
	A.Apply(set<int64_t>(value));
}


pyDenseParVec* pySpParMat::FindIndsOfColsWithSumGreaterThan(int64_t gt)
{
	pyDenseParVec* ret = new pyDenseParVec();
	
	// make a temporary int matrix
	SpParMat<int64_t, int, SpDCCols<int64_t, int> > * AInt = new SpParMat<int64_t, int, SpDCCols<int64_t, int> >(A);
	DenseParVec<int64_t, int> ColSums = AInt->Reduce(Column, plus<int>(), 0); 
	ret->v = ColSums.FindInds(bind2nd(greater<int>(), (int)gt));	// only the indices of connected vertices
	delete AInt;	// save memory	

	
	//DenseParVec<int64_t, int> ColSums = A.Reduce(Column, plus<int>(), 0);
	//cout << "column sums:------------" << endl;
	//ColSums.DebugPrint();
	//ret->v = ColSums.FindInds(bind2nd(greater<int>(), (int)gt));
	return ret;
}


pySpParVec* pySpParMat::SpMV_PlusTimes(const pySpParVec& x)
{
	pySpParVec* ret = new pySpParVec(0);
	ret->v = SpMV< PlusTimesSRing<int, int64_t > >(A, x.v);
	return ret;
}

pySpParVec* pySpParMat::SpMV_SelMax(const pySpParVec& x)
{
	pySpParVec* ret = new pySpParVec(0);
	ret->v = SpMV< SelectMaxSRing<int, int64_t > >(A, x.v);
	return ret;
}


