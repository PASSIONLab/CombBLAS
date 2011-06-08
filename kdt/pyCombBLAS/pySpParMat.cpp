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
	FullyDistVec<INDEXTYPE, INDEXTYPE> irow = rows->v;
	FullyDistVec<INDEXTYPE, INDEXTYPE> icol = cols->v;
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

double pySpParMat::GenGraph500Edges(int scale, pyDenseParVec* pyDegrees, int EDGEFACTOR)
{
	typedef SpParMat < int64_t, doubleint, SpDCCols<int64_t,doubleint> > PSpMat_DoubleInt;

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
	A =  PSpMat_DoubleInt(*G);			// Convert to Boolean
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

	DistEdgeList<int64_t> *DEL = new DistEdgeList<int64_t>();
	DEL->GenGraph500Data(initiator, scale, 16 * ((int64_t) std::pow(2.0, (double) scale)) / nprocs );
	PermEdges(*DEL);
	RenameVertices(*DEL);

	// Start Kernel #1
	MPI::COMM_WORLD.Barrier();
	double t1 = MPI_Wtime();

	// conversion from distributed edge list, keeps self-loops, sums duplicates
	A = MatType(*DEL, false);
	delete DEL; // free the distributed edge list before making another copy through the matrix transpose
	

	//int rank;
	//MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	MatType *AT = new MatType(A);
	//cout << rank << ": A: nnz=" << G->getnnz() << ", local nnz=" << G->seq().getnnz() << endl;
	//cout << rank << ": AT: nnz=" << AT->getnnz() << ", local nnz=" << AT->seq().getnnz() << endl;
	AT->Transpose();
	A += *AT;
	delete AT;
	
	MPI::COMM_WORLD.Barrier();
	double t2=MPI_Wtime();
	
	// END OF COPY
	
	k1time = t2-t1;
	return k1time;
*/
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

pySpParMat pySpParMat::operator*(pySpParMat& other)
{
	return SpGEMM(other);
}

pySpParMat pySpParMat::SpGEMM(pySpParMat& other, op::Semiring* sring)
{
	if (sring == NULL)
	{
		return pySpParMat( Mult_AnXBn_Synch<PlusTimesSRing<doubleint, doubleint > >(A, other.A) );
	}
	else
	{
		sring->enableSemiring();
		pySpParMat ret( Mult_AnXBn_Synch<op::SemiringTemplArg<doubleint, doubleint > >(A, other.A) );
		sring->disableSemiring();
		return ret;
	}
}

pySpParMat pySpParMat::__getitem__(const pyDenseParVec& rows, const pyDenseParVec& cols)
{
	return SubsRef(rows, cols);
}

pySpParMat pySpParMat::SubsRef(const pyDenseParVec& rows, const pyDenseParVec& cols)
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

void pySpParMat::DimWiseApply(int dim, const pyDenseParVec& values, op::BinaryFunction* f)
{
	A.DimApply((dim == Column() ? ::Column : ::Row), values.v, *f);
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
	FullyDistVec<INDEXTYPE, INDEXTYPE> irows, icols;
	A.Find(irows, icols, outvals->v);
	outrows->v = irows;
	outcols->v = icols;
	//A.Find(outrows->v, outcols->v, outvals->v);
}

pySpParVec pySpParMat::SpMV_PlusTimes(const pySpParVec& x)
{
	return pySpParVec( ::SpMV< PlusTimesSRing<doubleint, doubleint > >(A, x.v) );
}

pySpParVec pySpParMat::SpMV_SelMax(const pySpParVec& x)
{
	return pySpParVec( ::SpMV< SelectMaxSRing<doubleint, doubleint > >(A, x.v) );
}

void pySpParMat::SpMV_SelMax_inplace(pySpParVec& x)
{
	x.v = ::SpMV< SelectMaxSRing<doubleint, doubleint > >(A, x.v);
}

pySpParVec pySpParMat::SpMV(const pySpParVec& x, op::Semiring* sring)
{
	if (sring == NULL)
	{
		return pySpParVec( ::SpMV< PlusTimesSRing<doubleint, doubleint > >(A, x.v) );
	}
	else if (sring->getType() == op::Semiring::TIMESPLUS)
	{
		return pySpParVec( ::SpMV< PlusTimesSRing<doubleint, doubleint > >(A, x.v) );
	}
	else if (sring->getType() == op::Semiring::SECONDMAX)
	{
		return pySpParVec( ::SpMV< SelectMaxSRing<doubleint, doubleint > >(A, x.v) );
	}
	else
	{
		sring->enableSemiring();
		pySpParVec ret( ::SpMV< op::SemiringTemplArg<doubleint, doubleint > >(A, x.v) );
		sring->disableSemiring();
		return ret;
	}
}

pyDenseParVec pySpParMat::SpMV(const pyDenseParVec& x, op::Semiring* sring)
{
	if (sring == NULL)
	{
		return pyDenseParVec( ::SpMV< PlusTimesSRing<doubleint, doubleint > >(A, x.v) );
	}
	else if (sring->getType() == op::Semiring::TIMESPLUS)
	{
		return pyDenseParVec( ::SpMV< PlusTimesSRing<doubleint, doubleint > >(A, x.v) );
	}
	else if (sring->getType() == op::Semiring::SECONDMAX)
	{
		return pyDenseParVec( ::SpMV< SelectMaxSRing<doubleint, doubleint > >(A, x.v) );
	}
	else
	{
		sring->enableSemiring();
		pyDenseParVec ret( ::SpMV< op::SemiringTemplArg<doubleint, doubleint > >(A, x.v) );
		sring->disableSemiring();
		return ret;
	}
}

void pySpParMat::SpMV_inplace(pySpParVec& x, op::Semiring* sring)
{
	if (sring == NULL)
	{
		x = ::SpMV< PlusTimesSRing<doubleint, doubleint > >(A, x.v);
	}
	else if (sring->getType() == op::Semiring::TIMESPLUS)
	{
		x = ::SpMV< PlusTimesSRing<doubleint, doubleint > >(A, x.v);
	}
	else if (sring->getType() == op::Semiring::SECONDMAX)
	{
		x = ::SpMV< SelectMaxSRing<doubleint, doubleint > >(A, x.v);
	}
	else
	{
		sring->enableSemiring();
		x = ::SpMV< op::SemiringTemplArg<doubleint, doubleint > >(A, x.v);
		sring->disableSemiring();
	}
}

void pySpParMat::SpMV_inplace(pyDenseParVec& x, op::Semiring* sring)
{
	if (sring == NULL)
	{
		x = ::SpMV< PlusTimesSRing<doubleint, doubleint > >(A, x.v);
	}
	else if (sring->getType() == op::Semiring::TIMESPLUS)
	{
		x = ::SpMV< PlusTimesSRing<doubleint, doubleint > >(A, x.v);
	}
	else if (sring->getType() == op::Semiring::SECONDMAX)
	{
		x = ::SpMV< SelectMaxSRing<doubleint, doubleint > >(A, x.v);
	}
	else
	{
		sring->enableSemiring();
		x = ::SpMV< op::SemiringTemplArg<doubleint, doubleint > >(A, x.v);
		sring->disableSemiring();
	}
}
