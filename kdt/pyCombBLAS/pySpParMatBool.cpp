#include <iostream>
#include "pySpParMatBool.h"

#define MATCLASS pySpParMatBool

pySpParMatBool EWiseMult(const pySpParMatBool& A1, const pySpParMatBool& A2, bool exclude)
{
	return pySpParMatBool(EWiseMult(A1.A, A2.A, exclude));
}

pySpParMatBool::pySpParMatBool(): A(new DCColsType(), commGrid)
{
}

pySpParMatBool::pySpParMatBool(const pySpParMatBool& copyStructureFrom): A(copyStructureFrom.A)
{
}
pySpParMatBool::pySpParMatBool(const pySpParMat    & copyStructureFrom): A(copyStructureFrom.A)
{
}
pySpParMatBool::pySpParMatBool(const pySpParMatObj1& copyStructureFrom): A(copyStructureFrom.A)
{
}
pySpParMatBool::pySpParMatBool(const pySpParMatObj2& copyStructureFrom): A(copyStructureFrom.A)
{
}

pySpParMatBool::pySpParMatBool(MatType other): A(other)
{
}

pySpParMatBool::pySpParMatBool(int64_t m, int64_t n, pyDenseParVec* rows, pyDenseParVec* cols, pyDenseParVec* vals): A(NULL, commGrid)
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
	A.ReadDistribute(filename, 0, false, MatType::ScalarReadSaveHandler(), true);
}

void pySpParMatBool::save(const char* filename)
{
	A.SaveGathered(filename, MatType::ScalarReadSaveHandler(), true);
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

double pySpParMatBool::GenGraph500Edges(int scale, pyDenseParVec* pyDegrees, int EDGEFACTOR, bool delIsolated, double a, double b, double c, double d)
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
	//double initiator[4] = {.57, .19, .19, .05};
	double initiator[4] = {a, b, c, d};
	
	double t01 = MPI_Wtime();
	double t02;
	DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>();
	if(!scramble || !delIsolated)
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
	
	FullyDistVec<int64_t, int64_t> * ColSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid());
	FullyDistVec<int64_t, int64_t> * RowSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid());
	A.Reduce(*ColSums, ::Column, plus<int64_t>(), static_cast<int64_t>(0)); 	
	A.Reduce(*RowSums, ::Row, plus<int64_t>(), static_cast<int64_t>(0)); 	
	SpParHelper::Print("Reductions done\n");
	ColSums->EWiseApply(*RowSums, plus<int64_t>());
	SpParHelper::Print("Intersection of colsums and rowsums found\n");
	delete RowSums;
	
	if (delIsolated)
	{
		nonisov = ColSums->FindInds(bind2nd(greater<int64_t>(), 0));	// only the indices of non-isolated vertices
		
		SpParHelper::Print("Found (and permuted) non-isolated vertices\n");	
		nonisov.RandPerm();	// so that A(v,v) is load-balanced (both memory and time wise)
		A.PrintInfo();
		A(nonisov, nonisov, true);	// in-place permute to save memory	
		SpParHelper::Print("Dropped isolated vertices from input\n");	
		A.PrintInfo();
	}
	delete ColSums;
	
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
}


pySpParMatBool pySpParMatBool::copy()
{
	return pySpParMatBool(*this);
}

pySpParMatBool& pySpParMatBool::assign(const pySpParMatBool& other)
{
	A = other.A;
	return *this;
}

pySpParMatBool pySpParMatBool::SpGEMM(pySpParMatBool& other)
{
	pySpParMatBool ret;
	PSpGEMM<PlusTimesSRing<bool, bool > >(A, other.A, ret.A);
	return ret;
}

pySpParMatBool pySpParMatBool::SubsRef(const pyDenseParVec& rows, const pyDenseParVec& cols, bool inPlace, op::UnaryPredicateObj* matFilter)
{
	if (matFilter == NULL)
	{
		if (inPlace)
		{
			A(rows.v, cols.v, true);
			return pySpParMatBool();
		}
		else
		{
			return pySpParMatBool(A(rows.v, cols.v));
		}
	}
	else
	{ // The filtering semiring is slightly heavier than the default one, so only use it if needed
		if (inPlace)
		{
			SRFilterHelper<bool, bool, bool>::setFilterX(matFilter);
			SRFilterHelper<bool, bool, bool>::setFilterY(matFilter);
			A.SubsRef_SR<PCBBoolCopy1stSRing<bool>, PCBBoolCopy2ndSRing<bool> >(rows.v, cols.v, true);
			SRFilterHelper<bool, bool, bool>::setFilterX(NULL);
			SRFilterHelper<bool, bool, bool>::setFilterY(NULL);
			return pySpParMatBool();
		}
		else
		{
			SRFilterHelper<bool, bool, bool>::setFilterX(matFilter);
			SRFilterHelper<bool, bool, bool>::setFilterY(matFilter);
			pySpParMatBool ret(A.SubsRef_SR<PCBBoolCopy1stSRing<bool>, PCBBoolCopy2ndSRing<bool> >(rows.v, cols.v, false));
			SRFilterHelper<bool, bool, bool>::setFilterX(NULL);
			SRFilterHelper<bool, bool, bool>::setFilterY(NULL);
			return ret;
		}
	}
}
	
int64_t pySpParMatBool::removeSelfLoops()
{
	return A.RemoveLoops();
}

void pySpParMatBool::Apply(op::UnaryFunction* op)
{
	A.Apply(*op);
}

/*
void pySpParMatBool::DimWiseApply(int dim, const pyDenseParVec& values, op::BinaryFunctionObj* f)
{
	A.DimApply((dim == Column() ? ::Column : ::Row), values.v, *f);
}
*/

pySpParMatBool pySpParMatBool::Prune(op::UnaryFunction* op, bool inPlace)
{
	return pySpParMatBool(A.Prune(*op, inPlace));
}

int64_t pySpParMatBool::Count(op::UnaryFunction* pred)
{
	// use Reduce to count along the columns, then reduce the result vector into one value
	op::BinaryFunction p = op::plus();
	return static_cast<int64_t>(Reduce(Column(), &p, pred, 0).Reduce(&p));
}

void pySpParMatBool::Find(pyDenseParVec* outrows, pyDenseParVec* outcols, pyDenseParVec* outvals) const
{
	FullyDistVec<INDEXTYPE, INDEXTYPE> irows, icols;
	A.Find(irows, icols);
	outrows->v = irows;
	outcols->v = icols;
	
	FullyDistVec<int64_t, doubleint> ones(irows.TotalLength(), 1);
	outvals->v = ones;
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
		
	pyDenseParVec ret(len, identity);
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
void pySpParMatBool::Reduce(int dim, pyDenseParVec *ret, op::BinaryFunctionObj* bf, op::UnaryFunctionObj* uf, double identity)
{
	bf->getMPIOp();
	if (uf == NULL)
		A.Reduce(ret->v, (Dim)dim, *bf, doubleint(identity));
	else
		A.Reduce(ret->v, (Dim)dim, *bf, doubleint(identity), *uf);
	bf->releaseMPIOp();
}

// the type of this ANullValue
#define NULL_PAR_TYPE   bool
// how to pass in this ANullValue (i.e. ANull or doubleint(ANull))
#define A_NULL_ARG      ANull

#define MATCLASS pySpParMatBool
//#define MATCLASS_OBJ

#include "pyCommonMatFuncs.cpp"
