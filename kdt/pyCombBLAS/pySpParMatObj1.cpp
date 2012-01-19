#include <iostream>
#include "pySpParMatObj1.h"

pySpParMatObj1::pySpParMatObj1()
{
}

pySpParMatObj1::pySpParMatObj1(const pySpParMatObj1& copyFrom): A(copyFrom.A)
{
}

pySpParMatObj1::pySpParMatObj1(MatType other): A(other)
{
}

pySpParMatObj1::pySpParMatObj1(int64_t m, int64_t n, pyDenseParVec* rows, pyDenseParVec* cols, pyDenseParVecObj1* vals)
{
	FullyDistVec<INDEXTYPE, INDEXTYPE> irow = rows->v;
	FullyDistVec<INDEXTYPE, INDEXTYPE> icol = cols->v;
	A = MatType(m, n, irow, icol, vals->v);
}

int64_t pySpParMatObj1::getnee()
{
	return A.getnnz();
}
/*
int64_t pySpParMatObj1::getnnz()
{
	// actually count the number of nonzeros

	op::BinaryFunction ne = op::not_equal_to();
	op::UnaryFunction ne0 = op::bind2nd(ne, 0);
	return Count(&ne0);
}*/

int64_t pySpParMatObj1::getnrow() const
{
	return A.getnrow();
}

int64_t pySpParMatObj1::getncol() const
{
	return A.getncol();
}

class Obj1ReadSaveHandler
{
public:
	Obj1 getNoNum(pySpParMatObj1::INDEXTYPE row, pySpParMatObj1::INDEXTYPE col) { return Obj1(); }

	template <typename c, typename t>
	Obj1 read(std::basic_istream<c,t>& is, pySpParMatObj1::INDEXTYPE row, pySpParMatObj1::INDEXTYPE col)
	{
		Obj1 ret;
		ret.loadCpp(is, row, col);
		return ret;
	}

	template <typename c, typename t>
	void save(std::basic_ostream<c,t>& os, const Obj1& v, pySpParMatObj1::INDEXTYPE row, pySpParMatObj1::INDEXTYPE col)
	{
		v.saveCpp(os);
	}
};

void pySpParMatObj1::load(const char* filename)
{
	ifstream input(filename);
	A.ReadDistribute(input, 0, false, Obj1ReadSaveHandler(), true);
	input.close();
}

void pySpParMatObj1::save(const char* filename)
{
	A.SaveGathered(filename, Obj1ReadSaveHandler(), true);
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

double pySpParMatObj1::GenGraph500Edges(int scale, pyDenseParVec* pyDegrees, int EDGEFACTOR)
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
	A =  PSpMat_Obj1(*G);			// Convert to Boolean
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
double pySpParMatObj1::GenGraph500Edges(int scale, pyDenseParVec& pyDegrees)
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
pySpParMatObj1 pySpParMatObj1::copy()
{
	return pySpParMatObj1(*this);
}

/*
pySpParMatObj1& pySpParMatObj1::operator+=(const pySpParMatObj1& other)
{
	A += other.A;
	return *this;
}*/

pySpParMatObj1& pySpParMatObj1::assign(const pySpParMatObj1& other)
{
	A = other.A;
	return *this;
}

/*
pySpParMatObj1 pySpParMatObj1::operator*(pySpParMatObj1& other)
{
	return SpGEMM(other);
}
*/

void pySpParMatObj1::Square(op::SemiringObj* sring)
{
	sring->enableSemiring();
	A.Square<op::SemiringObjTemplArg<NUMTYPE, NUMTYPE, NUMTYPE> >();
	sring->disableSemiring();
}

pySpParMat pySpParMatObj1::SpGEMM(pySpParMat& other, op::SemiringObj* sring)
{/*
	sring->enableSemiring();
	pySpParMat ret( Mult_AnXBn_Synch<op::SemiringObjTemplArg>(A, other.A) );
	sring->disableSemiring();
	return ret;*/
	cout << "Mixed-type SpGEMM not supported yet!";
	return pySpParMat();
}

pySpParMatObj1 pySpParMatObj1::SpGEMM(pySpParMatObj1& other, op::SemiringObj* sring)
{
	sring->enableSemiring();
	pySpParMatObj1 ret( Mult_AnXBn_Synch<op::SemiringObjTemplArg<Obj1, Obj1, Obj1> >(A, other.A) );
	sring->disableSemiring();
	return ret;
}

pySpParMatObj2 pySpParMatObj1::SpGEMM(pySpParMatObj2& other, op::SemiringObj* sring)
{
	/*
	sring->enableSemiring();
	pySpParMatObj2 ret( Mult_AnXBn_Synch<op::SemiringObjTemplArg>(A, other.A) );
	sring->disableSemiring();
	return ret;
	*/
	cout << "Mixed-type SpGEMM not supported yet!";
	return pySpParMatObj2();
}

pySpParMatObj1 pySpParMatObj1::__getitem__(const pyDenseParVec& rows, const pyDenseParVec& cols)
{
	return SubsRef(rows, cols);
}

pySpParMatObj1 pySpParMatObj1::SubsRef(const pyDenseParVec& rows, const pyDenseParVec& cols, bool inPlace)
{
	if (inPlace)
	{
		A(rows.v, cols.v, true);
		return pySpParMatObj1();
	}
	else
	{
		return pySpParMatObj1(A(rows.v, cols.v, false));
	}
}
	
int64_t pySpParMatObj1::removeSelfLoops()
{
	return A.RemoveLoops();
}

void pySpParMatObj1::Apply(op::UnaryFunctionObj* op)
{
	A.Apply(*op);
}

void pySpParMatObj1::DimWiseApply(int dim, const pyDenseParVecObj1& values, op::BinaryFunctionObj* f)
{
	A.DimApply((dim == Column() ? ::Column : ::Row), values.v, *f);
}
/*
pySpParMatObj1 EWiseMult(const pySpParMatObj1& A1, const pySpParMatObj1& A2, bool exclude)
{
	return pySpParMatObj1(EWiseMult(A1.A, A2.A, exclude));
}*/

pySpParMatObj1 EWiseApply(const pySpParMatObj1& A, const pySpParMatObj1& B, op::BinaryFunctionObj *bf, bool notB, Obj1 defaultBValue)
{
	return pySpParMatObj1(EWiseApply<Obj1, pySpParMatObj1::DCColsType>(A.A, B.A, *bf, notB, defaultBValue));
}

pySpParMatObj1 EWiseApply(const pySpParMatObj1& A, const pySpParMatObj2& B, op::BinaryFunctionObj *bf, bool notB, Obj2 defaultBValue)
{
	return pySpParMatObj1(EWiseApply<Obj1, pySpParMatObj1::DCColsType>(A.A, B.A, *bf, notB, defaultBValue));
}

pySpParMatObj1 EWiseApply(const pySpParMatObj1& A, const pySpParMat&     B, op::BinaryFunctionObj *bf, bool notB, double defaultBValue)
{
	return pySpParMatObj1(EWiseApply<Obj1, pySpParMatObj1::DCColsType>(A.A, B.A, *bf, notB, doubleint(defaultBValue)));
}

void pySpParMatObj1::Prune(op::UnaryPredicateObj* pred)
{
	A.Prune(*pred);
}

int64_t pySpParMatObj1::Count(op::UnaryPredicateObj* pred)
{
	// use Reduce to count along the columns, then reduce the result vector into one value
	//op::BinaryFunction p = op::plus();
	//return static_cast<int64_t>(Reduce(Column(), &p, pred, 0).Reduce(&p));
	FullyDistVec<INDEXTYPE, int64_t> colcounts;
	A.Reduce(colcounts, ::Column, ::plus<int64_t>(), static_cast<int64_t>(0), *pred);
	return colcounts.Reduce(::plus<int64_t>(), 0L);
}

void pySpParMatObj1::Reduce(int dim, pyDenseParVecObj1 *ret, op::BinaryFunctionObj* bf, op::UnaryFunctionObj* uf, Obj1 identity)
{
	bf->getMPIOp();
	if (uf == NULL)
		A.Reduce(ret->v, (Dim)dim, *bf, identity);
	else
		A.Reduce(ret->v, (Dim)dim, *bf, identity, *uf);
	bf->releaseMPIOp();
}
void pySpParMatObj1::Reduce(int dim, pyDenseParVec     *ret, op::BinaryFunctionObj* bf, op::UnaryFunctionObj* uf, double identity)
{
	bf->getMPIOp();
	if (uf == NULL)
		A.Reduce(ret->v, (Dim)dim, *bf, doubleint(identity));
	else
		A.Reduce(ret->v, (Dim)dim, *bf, doubleint(identity), uf->getRetDoubleVersion());
	bf->releaseMPIOp();
}

void pySpParMatObj1::Transpose()
{
	A.Transpose();
}

/*void pySpParMatObj1::EWiseMult(pySpParMatObj1* rhs, bool exclude)
{
	A.EWiseMult(rhs->A, exclude);
}*/

void pySpParMatObj1::Find(pyDenseParVec* outrows, pyDenseParVec* outcols, pyDenseParVecObj1* outvals) const
{
	FullyDistVec<INDEXTYPE, INDEXTYPE> irows, icols;
	A.Find(irows, icols, outvals->v);
	outrows->v = irows;
	outcols->v = icols;
	//A.Find(outrows->v, outcols->v, outvals->v);
}

template <class VECTYPE, class VEC>
VEC pySpParMatObj1::SpMV_worker(const VEC& x, op::SemiringObj* sring)
{
	if (sring == NULL)
	{
		cout << "You must supply a semiring for SpMV!" << endl;
		return VEC(getnrow());
	}
	/*else if (sring->getType() == op::Semiring::SECONDMAX)
	{
		VEC ret(0);
		::SpMV< Select2ndSRing<NUMTYPE, VECTYPE, VECTYPE> >(A, x.v, ret.v, false );
		return ret;
	}*/
	else
	{
		sring->enableSemiring();
		VEC ret(0);
		::SpMV< op::SemiringObjTemplArg<NUMTYPE, VECTYPE, VECTYPE> >(A, x.v, ret.v, false );
		sring->disableSemiring();
		return ret;
	}
}

template <class VECTYPE, class VEC>
void pySpParMatObj1::SpMV_worker_inplace(VEC& x, op::SemiringObj* sring)
{
	if (sring == NULL)
	{
		cout << "You must supply a semiring for SpMV!" << endl;
	}
	else
	{
		sring->enableSemiring();
		::SpMV< op::SemiringObjTemplArg<NUMTYPE, VECTYPE, VECTYPE> >(A, x.v, x.v, false );
		sring->disableSemiring();
	}
}

pySpParVec     pySpParMatObj1::SpMV(const pySpParVec&     x, op::SemiringObj* sring) { return SpMV_worker<doubleint>(x, sring); }
pySpParVecObj1 pySpParMatObj1::SpMV(const pySpParVecObj1& x, op::SemiringObj* sring) { return SpMV_worker<Obj1>(x, sring); }
pySpParVecObj2 pySpParMatObj1::SpMV(const pySpParVecObj2& x, op::SemiringObj* sring) { return SpMV_worker<Obj2>(x, sring); }

void pySpParMatObj1::SpMV_inplace(pySpParVec&     x, op::SemiringObj* sring) { return SpMV_worker_inplace<doubleint>(x, sring); }
void pySpParMatObj1::SpMV_inplace(pySpParVecObj1& x, op::SemiringObj* sring) { return SpMV_worker_inplace<Obj1>(x, sring); }
void pySpParMatObj1::SpMV_inplace(pySpParVecObj2& x, op::SemiringObj* sring) { return SpMV_worker_inplace<Obj2>(x, sring); }

#if 0
// these don't work yet because the CombBLAS dense vector SpMV hasn't been updated like the sparse vector one has.
pyDenseParVec     pySpParMatObj1::SpMV(const pyDenseParVec&     x, op::SemiringObj* sring) { return SpMV_worker<doubleint>(x, sring); }
pyDenseParVecObj1 pySpParMatObj1::SpMV(const pyDenseParVecObj1& x, op::SemiringObj* sring) { return SpMV_worker<Obj1>(x, sring); }
pyDenseParVecObj2 pySpParMatObj1::SpMV(const pyDenseParVecObj2& x, op::SemiringObj* sring) { return SpMV_worker<Obj2>(x, sring); }
#endif 

void pySpParMatObj1::SpMV_inplace(pyDenseParVec&     x, op::SemiringObj* sring) { cout << "Mixed type dense SpMV not supported yet." << endl; }
void pySpParMatObj1::SpMV_inplace(pyDenseParVecObj1& x, op::SemiringObj* sring)
{
	if (sring == NULL)
	{
		cout << "You must supply a semiring for SpMV!" << endl;
	}
	else
	{
		sring->enableSemiring();
		x.v = ::SpMV< op::SemiringObjTemplArg<Obj1, Obj1, Obj1> >(A, x.v);
		sring->disableSemiring();
	}
}
void pySpParMatObj1::SpMV_inplace(pyDenseParVecObj2& x, op::SemiringObj* sring) { cout << "Mixed type dense SpMV not supported yet." << endl; }

pyDenseParVec     pySpParMatObj1::SpMV(const pyDenseParVec&     x, op::SemiringObj* sring)
{
	cout << "Mixed type dense SpMV not supported yet." << endl;
	//cout << "You must supply a semiring for SpMV!" << endl;
	return pyDenseParVec(getnrow(), 0, 0);
}

pyDenseParVecObj1 pySpParMatObj1::SpMV(const pyDenseParVecObj1& x, op::SemiringObj* sring)
{
	if (sring == NULL)
	{
		cout << "You must supply a semiring for SpMV!" << endl;
		return pyDenseParVecObj1(getnrow(), Obj1());
	}
	/*
	else if (sring->getType() == op::Semiring::SECONDMAX)
	{
		return pySpParVecObj1( ::SpMV< Select2ndSRing>(A, x.v) );
	}*/
	else
	{
		sring->enableSemiring();
		pyDenseParVecObj1 ret( ::SpMV< op::SemiringObjTemplArg<Obj1, Obj1, Obj1> >(A, x.v) );
		sring->disableSemiring();
		return ret;
	}
}

pyDenseParVecObj2 pySpParMatObj1::SpMV(const pyDenseParVecObj2& x, op::SemiringObj* sring)
{
	cout << "Mixed type dense SpMV not supported yet." << endl;
	//cout << "You must supply a semiring for SpMV!" << endl;
	return pyDenseParVecObj2(getnrow(), Obj2());
}

/*
pySpParVec pySpParMatObj1::SpMV_PlusTimes(const pySpParVec& x)
{
	return pySpParVec( ::SpMV< PlusTimesSRing<doubleint, doubleint > >(A, x.v) );
}

pySpParVec pySpParMatObj1::SpMV_SelMax(const pySpParVec& x)
{
	return pySpParVec( ::SpMV< Select2ndSRing<doubleint, doubleint > >(A, x.v) );
}

void pySpParMatObj1::SpMV_SelMax_inplace(pySpParVec& x)
{
	x.v = ::SpMV< Select2ndSRing<doubleint, doubleint > >(A, x.v);
}

pySpParVec pySpParMatObj1::SpMV(const pySpParVec& x, op::Semiring* sring)
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

pyDenseParVec pySpParMatObj1::SpMV(const pyDenseParVec& x, op::Semiring* sring)
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

void pySpParMatObj1::SpMV_inplace(pySpParVec& x, op::Semiring* sring)
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

void pySpParMatObj1::SpMV_inplace(pyDenseParVec& x, op::Semiring* sring)
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
