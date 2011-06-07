//#include <mpi.h>
#include <sys/time.h> 

#include <iostream>
#include "pySpParMatBool.h"


pySpParMatBool EWiseMult(const pySpParMatBool& A1, const pySpParMatBool& A2, bool exclude)
{
	return pySpParMatBool(EWiseMult(A1.A, A2.A, exclude));
}

pySpParMatBool::pySpParMatBool()
{
}

pySpParMatBool::pySpParMatBool(const pySpParMatBool& copyFrom): A(copyFrom.A)
{
}

pySpParMatBool::pySpParMatBool(MatType other): A(other)
{
}

pySpParMatBool::pySpParMatBool(int64_t m, int64_t n, pyDenseParVec* rows, pyDenseParVec* cols, pyDenseParVec* vals)
{
	/*
	// This should work, but it gives a compile error complaining about FullyDistVec<int64, bool> on that matrix constructor.
	FullyDistVec<INDEXTYPE, INDEXTYPE> irow = rows->v;
	FullyDistVec<INDEXTYPE, INDEXTYPE> icol = cols->v;
	A = MatType(m, n, irow, icol, 1);
	*/
	pySpParMat B(m, n, rows, cols, vals);
	A = B.A;
}

pySpParMatBool::pySpParMatBool(const pySpParMat& copyFrom): A()
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

	op::BinaryFunction ne = op::not_equal_to();
	op::UnaryFunction ne0 = op::bind2nd(ne, 0);
	return Count(&ne0);
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

// Copied directly from Aydin's C++ Graph500 code
template <typename PARMAT>
void Symmetricize(PARMAT & A)
{
	// boolean addition is practically a "logical or"
	// therefore this doesn't destruct any links
	PARMAT AT = A;
	AT.Transpose();
	A += AT;
}

double pySpParMatBool::GenGraph500Edges(int scale, pyDenseParVec* pyDegrees, int EDGEFACTOR)
{
	// Copied directly from Aydin's C++ Graph500 code
	typedef SpParMat < int64_t, bool, SpDCCols<int64_t,bool> > PSpMat_Bool;
	typedef SpParMat < int64_t, int, SpDCCols<int64_t,int> > PSpMat_Int;
	typedef SpParMat < int64_t, int64_t, SpDCCols<int64_t,int64_t> > PSpMat_Int64;
	typedef SpParMat < int32_t, int32_t, SpDCCols<int32_t,int32_t> > PSpMat_Int32;

	// Declare objects
	//PSpMat_Bool A;	
	FullyDistVec<int64_t, int64_t> degrees;	// degrees of vertices (including multi-edges and self-loops)
	FullyDistVec<int64_t, int64_t> nonisov;	// id's of non-isolated (connected) vertices
	bool scramble = true;



	// this is an undirected graph, so A*x does indeed BFS
	double initiator[4] = {.57, .19, .19, .05};
	
	double t01 = MPI_Wtime();
	double t02;
	DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>();
	if(!scramble)
	{
		DEL->GenGraph500Data(initiator, scale, EDGEFACTOR);
		SpParHelper::Print("Generated edge lists\n");
		t02 = MPI_Wtime();
		ostringstream tinfo;
		tinfo << "Generation took " << t02-t01 << " seconds" << endl;
		SpParHelper::Print(tinfo.str());
		
		PermEdges(*DEL);
		SpParHelper::Print("Permuted Edges\n");
		//DEL->Dump64bit("edges_permuted");
		//SpParHelper::Print("Dumped\n");
		
		RenameVertices(*DEL);	// intermediate: generates RandPerm vector, using MemoryEfficientPSort
		SpParHelper::Print("Renamed Vertices\n");
	}
	else	// fast generation
	{
		DEL->GenGraph500Data(initiator, scale, EDGEFACTOR, true, true );
		SpParHelper::Print("Generated renamed edge lists\n");
		t02 = MPI_Wtime();
		ostringstream tinfo;
		tinfo << "Generation took " << t02-t01 << " seconds" << endl;
		SpParHelper::Print(tinfo.str());
	}
	
	// Start Kernel #1
	MPI::COMM_WORLD.Barrier();
	double t1 = MPI_Wtime();
	
	// conversion from distributed edge list, keeps self-loops, sums duplicates
	PSpMat_Int32 * G = new PSpMat_Int32(*DEL, false); 
	delete DEL;	// free memory before symmetricizing
	SpParHelper::Print("Created Sparse Matrix (with int32 local indices and values)\n");
	
	MPI::COMM_WORLD.Barrier();
	double redts = MPI_Wtime();
	G->Reduce(degrees, ::Row, plus<int64_t>(), static_cast<int64_t>(0));	// Identity is 0 
	MPI::COMM_WORLD.Barrier();
	double redtf = MPI_Wtime();
	
	ostringstream redtimeinfo;
	redtimeinfo << "Calculated degrees in " << redtf-redts << " seconds" << endl;
	SpParHelper::Print(redtimeinfo.str());
	A =  PSpMat_Bool(*G);			// Convert to Boolean
	delete G;
	int64_t removed  = A.RemoveLoops();
	
	ostringstream loopinfo;
	loopinfo << "Converted to Boolean and removed " << removed << " loops" << endl;
	SpParHelper::Print(loopinfo.str());
	A.PrintInfo();
	
	FullyDistVec<int64_t, int64_t> * ColSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid(), 0);
	FullyDistVec<int64_t, int64_t> * RowSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid(), 0);
	A.Reduce(*ColSums, ::Column, plus<int64_t>(), static_cast<int64_t>(0)); 	
	A.Reduce(*RowSums, ::Row, plus<int64_t>(), static_cast<int64_t>(0)); 	
	SpParHelper::Print("Reductions done\n");
	ColSums->EWiseApply(*RowSums, plus<int64_t>());
	SpParHelper::Print("Intersection of colsums and rowsums found\n");
	delete RowSums;
	
	nonisov = ColSums->FindInds(bind2nd(greater<int64_t>(), 0));	// only the indices of non-isolated vertices
	delete ColSums;
	
	SpParHelper::Print("Found (and permuted) non-isolated vertices\n");	
	nonisov.RandPerm();	// so that A(v,v) is load-balanced (both memory and time wise)
	A.PrintInfo();
	A(nonisov, nonisov, true);	// in-place permute to save memory	
	SpParHelper::Print("Dropped isolated vertices from input\n");	
	A.PrintInfo();
	
	Symmetricize(A);	// A += A';
	SpParHelper::Print("Symmetricized\n");	
	
	#ifdef THREADED	
	A.ActivateThreading(SPLITS);	
	#endif
	A.PrintInfo();
	
	MPI::COMM_WORLD.Barrier();
	double t2=MPI_Wtime();
	
	//ostringstream k1timeinfo;
	//k1timeinfo << (t2-t1) - (redtf-redts) << " seconds elapsed for Kernel #1" << endl;
	//SpParHelper::Print(k1timeinfo.str());
	
	if (pyDegrees != NULL)
	{
		degrees = degrees(nonisov);	// fix the degrees array too
		pyDegrees->v = degrees;
	}
	return (t2-t1) - (redtf-redts);

/*
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
*/
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
pySpParMatBool pySpParMatBool::copy()
{
	return pySpParMatBool(*this);
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

pySpParMatBool pySpParMatBool::operator*(pySpParMatBool& other)
{
	return SpGEMM(other);
}

pySpParMatBool pySpParMatBool::SpGEMM(pySpParMatBool& other)
{
	return pySpParMatBool( Mult_AnXBn_Synch<PlusTimesSRing<bool, bool > >(A, other.A) );
}

pySpParMatBool pySpParMatBool::__getitem__(const pyDenseParVec& rows, const pyDenseParVec& cols)
{
	return SubsRef(rows, cols);
}

pySpParMatBool pySpParMatBool::SubsRef(const pyDenseParVec& rows, const pyDenseParVec& cols)
{
	return pySpParMatBool( A(rows.v, cols.v) );
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


pySpParMatBool EWiseApply(const pySpParMatBool& A, const pySpParMatBool& B, op::BinaryFunction *bf, bool notB, double defaultBValue)
{
	return pySpParMatBool( EWiseApply(A.A, B.A, *bf, notB, bool(defaultBValue)) );
}

void pySpParMatBool::Prune(op::UnaryFunction* op)
{
	A.Prune(*op);
}

int64_t pySpParMatBool::Count(op::UnaryFunction* pred)
{
	// use Reduce to count along the columns, then reduce the result vector into one value
	op::BinaryFunction p = op::plus();
	return static_cast<int64_t>(Reduce(Column(), &p, pred, 0).Reduce(&p));
}

	
pyDenseParVec pySpParMatBool::Reduce(int dim, op::BinaryFunction* f, double identity)
{
	return Reduce(dim, f, NULL, identity);
}

pyDenseParVec pySpParMatBool::Reduce(int dim, op::BinaryFunction* bf, op::UnaryFunction* uf, double identity)
{
	int64_t len = 1;
	if (dim == ::Row)
		len = getnrow();
	else
		len = getncol();
		
	pyDenseParVec ret(len, identity, identity);
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
	
	ret.v = tmp;
	
	return ret;
}

void pySpParMatBool::Transpose()
{
	A.Transpose();
}

/*void pySpParMatBool::EWiseMult(pySpParMatBool rhs, bool exclude)
{
	A.EWiseMult(rhs->A, exclude);
}*/

void pySpParMatBool::Find(pyDenseParVec* outrows, pyDenseParVec* outcols, pyDenseParVec* outvals) const
{
	FullyDistVec<INDEXTYPE, INDEXTYPE> irows, icols;
	A.Find(irows, icols);
	outrows->v = irows;
	outcols->v = icols;
	
	FullyDistVec<int64_t, doubleint> ones(irows.TotalLength(), 1, 0);
	outvals->v = ones;
	/*
	cout << "Find::vals:  ";
	for (int i = 0; i < vals.TotalLength(); i++)
	{
		bool v = vals.GetElement(i);
		cout << v << ", ";
	}*/
	//A.Find(outrows->v, outcols->v, outvals->v);
}

pySpParVec pySpParMatBool::SpMV_PlusTimes(const pySpParVec& x)
{
	return pySpParVec( SpMV< PlusTimesSRing<bool, doubleint > >(A, x.v) );
}

pySpParVec pySpParMatBool::SpMV_SelMax(const pySpParVec& x)
{
	return pySpParVec( SpMV< SelectMaxSRing<bool, doubleint > >(A, x.v) );
}

void pySpParMatBool::SpMV_SelMax_inplace(pySpParVec& x)
{
	x.v = SpMV< SelectMaxSRing<bool, doubleint> >(A, x.v);
}

