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
	void binaryfill(FILE * rFile, pySpParMatObj1::INDEXTYPE & row, pySpParMatObj1::INDEXTYPE & col, Obj1 & val) { return; }

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
	A.ReadDistribute(filename, 0, false, Obj1ReadSaveHandler(), true);
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

pySpParMatObj1& pySpParMatObj1::assign(const pySpParMatObj1& other)
{
	A = other.A;
	return *this;
}

pySpParMatObj1 pySpParMatObj1::SubsRef(const pyDenseParVec& rows, const pyDenseParVec& cols, bool inPlace, op::UnaryPredicateObj* matFilter)
{
	if (matFilter == NULL)
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
	else
	{ // The filtering semiring is slightly heavier than the default one, so only use it if needed
		if (inPlace)
		{
			SRFilterHelper<Obj1, Obj1, bool>::setFilterX(matFilter);
			SRFilterHelper<Obj1, bool, Obj1>::setFilterY(matFilter);
			A.SubsRef_SR<PCBBoolCopy1stSRing<Obj1>, PCBBoolCopy2ndSRing<Obj1> >(rows.v, cols.v, true);
			SRFilterHelper<Obj1, Obj1, bool>::setFilterX(NULL);
			SRFilterHelper<Obj1, bool, Obj1>::setFilterY(NULL);
			return pySpParMatObj1();
		}
		else
		{
			SRFilterHelper<Obj1, Obj1, bool>::setFilterX(matFilter);
			SRFilterHelper<Obj1, bool, Obj1>::setFilterY(matFilter);
			pySpParMatObj1 ret(A.SubsRef_SR<PCBBoolCopy1stSRing<Obj1>, PCBBoolCopy2ndSRing<Obj1> >(rows.v, cols.v, false));
			SRFilterHelper<Obj1, Obj1, bool>::setFilterX(NULL);
			SRFilterHelper<Obj1, bool, Obj1>::setFilterY(NULL);
			return ret;
		}
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


pySpParMatObj1 pySpParMatObj1::Prune(op::UnaryPredicateObj* pred, bool inPlace)
{
	return pySpParMatObj1(A.Prune(*pred, inPlace));
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


void pySpParMatObj1::Find(pyDenseParVec* outrows, pyDenseParVec* outcols, pyDenseParVecObj1* outvals) const
{
	FullyDistVec<INDEXTYPE, INDEXTYPE> irows, icols;
	A.Find(irows, icols, outvals->v);
	outrows->v = irows;
	outcols->v = icols;
	//A.Find(outrows->v, outcols->v, outvals->v);
}

/*
Common operations are implemented in one place and shared among the different classes
*/

// the type of this ANullValue
#define NULL_PAR_TYPE   const Obj1&
// how to pass in this ANullValue (i.e. ANull or doubleint(ANull))
#define A_NULL_ARG      ANull

#define MATCLASS pySpParMatObj1
#define MATCLASS_OBJ

#include "pyCommonMatFuncs.cpp"
