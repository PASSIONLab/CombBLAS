#include <iostream>
#include "pySpParMatObj2.h"

pySpParMatObj2::pySpParMatObj2()
{
}

pySpParMatObj2::pySpParMatObj2(const pySpParMatObj2& copyFrom): A(copyFrom.A)
{
}

pySpParMatObj2::pySpParMatObj2(MatType other): A(other)
{
}

pySpParMatObj2::pySpParMatObj2(int64_t m, int64_t n, pyDenseParVec* rows, pyDenseParVec* cols, pyDenseParVecObj2* vals)
{
	FullyDistVec<INDEXTYPE, INDEXTYPE> irow = rows->v;
	FullyDistVec<INDEXTYPE, INDEXTYPE> icol = cols->v;
	A = MatType(m, n, irow, icol, vals->v);
}

int64_t pySpParMatObj2::getnee()
{
	return A.getnnz();
}
/*
int64_t pySpParMatObj2::getnnz()
{
	// actually count the number of nonzeros

	op::BinaryFunction ne = op::not_equal_to();
	op::UnaryFunction ne0 = op::bind2nd(ne, 0);
	return Count(&ne0);
}*/

int64_t pySpParMatObj2::getnrow() const
{
	return A.getnrow();
}

int64_t pySpParMatObj2::getncol() const
{
	return A.getncol();
}

class Obj2ReadSaveHandler
{
public:
	Obj2 getNoNum(pySpParMatObj2::INDEXTYPE row, pySpParMatObj2::INDEXTYPE col) { return Obj2(); }

	template <typename c, typename t>
	Obj2 read(std::basic_istream<c,t>& is, pySpParMatObj2::INDEXTYPE row, pySpParMatObj2::INDEXTYPE col)
	{
		Obj2 ret;
		ret.loadCpp(is, row, col);
		return ret;
	}

	template <typename c, typename t>
	void save(std::basic_ostream<c,t>& os, const Obj2& v, pySpParMatObj2::INDEXTYPE row, pySpParMatObj2::INDEXTYPE col)
	{
		v.saveCpp(os);
	}
};

void pySpParMatObj2::load(const char* filename)
{
	ifstream input(filename);
	A.ReadDistribute(input, 0, false, Obj2ReadSaveHandler(), true);
	input.close();
}

void pySpParMatObj2::save(const char* filename)
{
	cout << "Warning! output is transposed" << endl;
	A.SaveGathered(filename, Obj2ReadSaveHandler());
}

#if 0
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

double pySpParMatObj2::GenGraph500Edges(int scale, pyDenseParVec* pyDegrees, int EDGEFACTOR)
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
	A =  PSpMat_Obj2(*G);			// Convert to Boolean
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
#endif
/*
double pySpParMatObj2::GenGraph500Edges(int scale, pyDenseParVec& pyDegrees)
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
pySpParMatObj2 pySpParMatObj2::copy()
{
	return pySpParMatObj2(*this);
}

/*
pySpParMatObj2& pySpParMatObj2::operator+=(const pySpParMatObj2& other)
{
	A += other.A;
	return *this;
}*/

pySpParMatObj2& pySpParMatObj2::assign(const pySpParMatObj2& other)
{
	A = other.A;
	return *this;
}

/*
pySpParMatObj2 pySpParMatObj2::operator*(pySpParMatObj2& other)
{
	return SpGEMM(other);
}
*/
pySpParMat pySpParMatObj2::SpGEMM(pySpParMat& other, op::SemiringObj* sring)
{/*
	sring->enableSemiring();
	pySpParMat ret( Mult_AnXBn_Synch<op::SemiringObjTemplArg>(A, other.A) );
	sring->disableSemiring();
	return ret;*/
	cout << "Mixed-type SpGEMM not supported yet!";
	return pySpParMat();
}

pySpParMatObj2 pySpParMatObj2::SpGEMM(pySpParMatObj2& other, op::SemiringObj* sring)
{
	sring->enableSemiring();
	pySpParMatObj2 ret( Mult_AnXBn_Synch<op::SemiringObjTemplArg<Obj2, Obj2, Obj2> >(A, other.A) );
	sring->disableSemiring();
	return ret;
}

pySpParMatObj1 pySpParMatObj2::SpGEMM(pySpParMatObj1& other, op::SemiringObj* sring)
{
	/*
	sring->enableSemiring();
	pySpParMatObj1 ret( Mult_AnXBn_Synch<op::SemiringObjTemplArg>(A, other.A) );
	sring->disableSemiring();
	return ret;
	*/
	cout << "Mixed-type SpGEMM not supported yet!";
	return pySpParMatObj1();
}

pySpParMatObj2 pySpParMatObj2::__getitem__(const pyDenseParVec& rows, const pyDenseParVec& cols)
{
	return SubsRef(rows, cols);
}

pySpParMatObj2 pySpParMatObj2::SubsRef(const pyDenseParVec& rows, const pyDenseParVec& cols)
{
#ifdef NOPARMATSUBSREF
	cout << "pySpParMatObj2::SubsRef() not implemented!" << endl;
	return copy();
#else
	return pySpParMatObj2(A(rows.v, cols.v));
#endif
}
	
int64_t pySpParMatObj2::removeSelfLoops()
{
	return A.RemoveLoops();
}

void pySpParMatObj2::Apply(op::UnaryFunctionObj* op)
{
	A.Apply(*op);
}

void pySpParMatObj2::DimWiseApply(int dim, const pyDenseParVecObj2& values, op::BinaryFunctionObj* f)
{
	A.DimApply((dim == Column() ? ::Column : ::Row), values.v, *f);
}
/*
pySpParMatObj2 EWiseMult(const pySpParMatObj2& A1, const pySpParMatObj2& A2, bool exclude)
{
	return pySpParMatObj2(EWiseMult(A1.A, A2.A, exclude));
}*/

pySpParMatObj2 EWiseApply(const pySpParMatObj2& A, const pySpParMatObj2& B, op::BinaryFunctionObj *bf, bool notB, Obj2 defaultBValue)
{
	return pySpParMatObj2(EWiseApply<Obj2, pySpParMatObj2::DCColsType>(A.A, B.A, *bf, notB, defaultBValue));
}

pySpParMatObj2 EWiseApply(const pySpParMatObj2& A, const pySpParMatObj1& B, op::BinaryFunctionObj *bf, bool notB, Obj1 defaultBValue)
{
	return pySpParMatObj2(EWiseApply<Obj2, pySpParMatObj2::DCColsType>(A.A, B.A, *bf, notB, defaultBValue));
}

pySpParMatObj2 EWiseApply(const pySpParMatObj2& A, const pySpParMat&     B, op::BinaryFunctionObj *bf, bool notB, double defaultBValue)
{
	return pySpParMatObj2(EWiseApply<Obj2, pySpParMatObj2::DCColsType>(A.A, B.A, *bf, notB, doubleint(defaultBValue)));
}

void pySpParMatObj2::Prune(op::UnaryPredicateObj* pred)
{
	A.Prune(*pred);
}

int64_t pySpParMatObj2::Count(op::UnaryPredicateObj* pred)
{
	// use Reduce to count along the columns, then reduce the result vector into one value
	//op::BinaryFunction p = op::plus();
	//return static_cast<int64_t>(Reduce(Column(), &p, pred, 0).Reduce(&p));
	FullyDistVec<INDEXTYPE, int64_t> colcounts;
	A.Reduce(colcounts, ::Column, ::plus<int64_t>(), static_cast<int64_t>(0), *pred);
	return colcounts.Reduce(::plus<int64_t>(), 0L);
}

pyDenseParVecObj2 pySpParMatObj2::Reduce(int dim, op::BinaryFunctionObj* f, Obj2 identity)
{
	return Reduce(dim, f, NULL, identity);
}

pyDenseParVecObj2 pySpParMatObj2::Reduce(int dim, op::BinaryFunctionObj* bf, op::UnaryFunctionObj* uf, Obj2 identity)
{
	int64_t len = 1;
	if (dim == ::Row)
		len = getnrow();
	else
		len = getncol();
		
	pyDenseParVecObj2 ret(len, identity);

	bf->getMPIOp();
	if (uf == NULL)
		A.Reduce(ret.v, (Dim)dim, *bf, identity);
	else
		A.Reduce(ret.v, (Dim)dim, *bf, identity, *uf);
	bf->releaseMPIOp();
	
	return ret;
}

void pySpParMatObj2::Transpose()
{
	A.Transpose();
}

/*void pySpParMatObj2::EWiseMult(pySpParMatObj2* rhs, bool exclude)
{
	A.EWiseMult(rhs->A, exclude);
}*/

void pySpParMatObj2::Find(pyDenseParVec* outrows, pyDenseParVec* outcols, pyDenseParVecObj2* outvals) const
{
	FullyDistVec<INDEXTYPE, INDEXTYPE> irows, icols;
	A.Find(irows, icols, outvals->v);
	outrows->v = irows;
	outcols->v = icols;
	//A.Find(outrows->v, outcols->v, outvals->v);
}

#define MIXEDOK 1

pySpParVec pySpParMatObj2::SpMV(const pySpParVec& x, op::SemiringObj* sring)
{
	//if (sring == NULL)
#if MIXEDOK == 0
	{
		cout << "Mixed type SpMV not supported yet." << endl;
		//cout << "You must supply a semiring for SpMV!" << endl;
		return pySpParVec(getnrow());
	}
	/*//
	else if (sring->getType() == op::Semiring::SECONDMAX)
	{
		return pySpParVecObj2( ::SpMV< Select2ndSRing>(A, x.v) );
	}
	else*/
#else
	{
		sring->enableSemiring();
		pySpParVec ret(0);
		::SpMV< op::SemiringObjTemplArg<Obj2, doubleint, doubleint> >(A, x.v, ret.v, false );
		//::SpMV< Select2ndSRing<doubleint, doubleint >(A, x.v, ret.v, false );
		sring->disableSemiring();
		return ret;
	}
#endif
}

pySpParVecObj2 pySpParMatObj2::SpMV(const pySpParVecObj2& x, op::SemiringObj* sring)
{
	if (sring == NULL)
	{
		cout << "You must supply a semiring for SpMV!" << endl;
		return pySpParVecObj2(getnrow());
	}
	/*
	else if (sring->getType() == op::Semiring::SECONDMAX)
	{
		return pySpParVecObj2( ::SpMV< Select2ndSRing>(A, x.v) );
	}*/
	else
	{
		sring->enableSemiring();
		pySpParVecObj2 ret( ::SpMV< op::SemiringObjTemplArg<Obj2, Obj2, Obj2> >(A, x.v) );
		sring->disableSemiring();
		return ret;
	}
}

pySpParVecObj1 pySpParMatObj2::SpMV(const pySpParVecObj1& x, op::SemiringObj* sring)
{
	//if (sring == NULL)
	{
		cout << "Mixed type SpMV not supported yet." << endl;
		//cout << "You must supply a semiring for SpMV!" << endl;
		return pySpParVecObj1(getnrow());
	}
	/*//
	else if (sring->getType() == op::Semiring::SECONDMAX)
	{
		return pySpParVecObj2( ::SpMV< Select2ndSRing>(A, x.v) );
	}*/
	/*
	else
	{
		sring->enableSemiring();
		pySpParVecObj1 ret( ::SpMV< op::SemiringObjTemplArg<Obj2, Obj1> >(A, x.v) );
		sring->disableSemiring();
		return ret;
	}*/
}

pyDenseParVec     pySpParMatObj2::SpMV(const pyDenseParVec&     x, op::SemiringObj* sring)
{
	cout << "Mixed type SpMV not supported yet." << endl;
	//cout << "You must supply a semiring for SpMV!" << endl;
	return pyDenseParVec(getnrow(), 0, 0);
}

pyDenseParVecObj2 pySpParMatObj2::SpMV(const pyDenseParVecObj2& x, op::SemiringObj* sring)
{
	if (sring == NULL)
	{
		cout << "You must supply a semiring for SpMV!" << endl;
		return pyDenseParVecObj2(getnrow(), Obj2());
	}
	/*
	else if (sring->getType() == op::Semiring::SECONDMAX)
	{
		return pySpParVecObj2( ::SpMV< Select2ndSRing>(A, x.v) );
	}*/
	else
	{
		sring->enableSemiring();
		pyDenseParVecObj2 ret( ::SpMV< op::SemiringObjTemplArg<Obj2, Obj2, Obj2> >(A, x.v) );
		sring->disableSemiring();
		return ret;
	}
}

pyDenseParVecObj1 pySpParMatObj2::SpMV(const pyDenseParVecObj1& x, op::SemiringObj* sring)
{
	cout << "Mixed type SpMV not supported yet." << endl;
	//cout << "You must supply a semiring for SpMV!" << endl;
	return pyDenseParVecObj1(getnrow(), Obj1());
}


/*
pySpParVec pySpParMatObj2::SpMV_PlusTimes(const pySpParVec& x)
{
	return pySpParVec( ::SpMV< PlusTimesSRing<doubleint, doubleint > >(A, x.v) );
}

pySpParVec pySpParMatObj2::SpMV_SelMax(const pySpParVec& x)
{
	return pySpParVec( ::SpMV< Select2ndSRing<doubleint, doubleint > >(A, x.v) );
}

void pySpParMatObj2::SpMV_SelMax_inplace(pySpParVec& x)
{
	x.v = ::SpMV< Select2ndSRing<doubleint, doubleint > >(A, x.v);
}

pySpParVec pySpParMatObj2::SpMV(const pySpParVec& x, op::Semiring* sring)
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
		return pySpParVec( ::SpMV< Select2ndSRing<doubleint, doubleint > >(A, x.v) );
	}
	else
	{
		sring->enableSemiring();
		pySpParVec ret( ::SpMV< op::SemiringTemplArg<doubleint, doubleint > >(A, x.v) );
		sring->disableSemiring();
		return ret;
	}
}

pyDenseParVec pySpParMatObj2::SpMV(const pyDenseParVec& x, op::Semiring* sring)
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
		return pyDenseParVec( ::SpMV< Select2ndSRing<doubleint, doubleint > >(A, x.v) );
	}
	else
	{
		sring->enableSemiring();
		pyDenseParVec ret( ::SpMV< op::SemiringTemplArg<doubleint, doubleint > >(A, x.v) );
		sring->disableSemiring();
		return ret;
	}
}

void pySpParMatObj2::SpMV_inplace(pySpParVec& x, op::Semiring* sring)
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
		x = ::SpMV< Select2ndSRing<doubleint, doubleint > >(A, x.v);
	}
	else
	{
		sring->enableSemiring();
		x = ::SpMV< op::SemiringTemplArg<doubleint, doubleint > >(A, x.v);
		sring->disableSemiring();
	}
}

void pySpParMatObj2::SpMV_inplace(pyDenseParVec& x, op::Semiring* sring)
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
		x = ::SpMV< Select2ndSRing<doubleint, doubleint > >(A, x.v);
	}
	else
	{
		sring->enableSemiring();
		x = ::SpMV< op::SemiringTemplArg<doubleint, doubleint > >(A, x.v);
		sring->disableSemiring();
	}
}
*/
