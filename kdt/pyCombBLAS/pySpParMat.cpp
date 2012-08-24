#include <iostream>
#include "pySpParMat.h"

pySpParMat::pySpParMat(): A(new DCColsType(), commGrid)
{
}

pySpParMat::pySpParMat(const pySpParMat& copyFrom): A(copyFrom.A)
{
}

pySpParMat::pySpParMat(MatType other): A(other)
{
}

pySpParMat::pySpParMat(const pySpParMatBool& other): A(other.A)
{
}

pySpParMat::pySpParMat(const pySpParMatObj1& other): A(other.A)
{
}

pySpParMat::pySpParMat(const pySpParMatObj2& other): A(other.A)
{
}

pySpParMat::pySpParMat(int64_t m, int64_t n, pyDenseParVec* rows, pyDenseParVec* cols, pyDenseParVec* vals): A(NULL, commGrid)
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
	
void pySpParMat::load(const char* filename, bool pario)
{
	string fn(filename);
	unsigned int dot = fn.find_last_of('.');
	if (strlen(filename) > 4 && strcmp(filename+(strlen(filename)-4), ".bin") == 0)
	{	
		// .bin file
		int mdot = fn.find_last_of('.', dot-1);
		int ndot = fn.find_last_of('.', mdot-1);
		
		string mstr = fn.substr(mdot+1, dot);
		string nstr = fn.substr(ndot+1, mdot);

#ifdef _MSC_VER
#define atoll _atoi64
#endif
		uint64_t m = atoll(mstr.c_str());
		uint64_t n = atoll(nstr.c_str());

        SpParHelper::Print("Detected binary input. Assuming filename format like: xxxxx.NUMVERTS.NUMEDGES.bin\n");
		ostringstream outs;
		outs << "Reading " << fn << " with " << n << " vertices and " << m << " edges" << endl;// << "nstr: " << nstr << " mstr: " << mstr << endl;
		SpParHelper::Print(outs.str());
		DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>(fn.c_str(), n, m);
		
		// conversion from distributed edge list, keeps self-loops, sums duplicates
		A = PSpMat_DoubleInt(*DEL, false);
	}
	else
	{
		// matrix market file
		A.ReadDistribute(filename, 0, false, MatType::ScalarReadSaveHandler(), true, pario);
	}
}

void pySpParMat::save(const char* filename)
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

double pySpParMat::GenGraph500Edges(int scale, pyDenseParVec* pyDegrees, int EDGEFACTOR, bool delIsolated, double a, double b, double c, double d)
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
	//double initiator[4] = {.57, .19, .19, .05};
	double initiator[4] = {a, b, c, d};
	
	double t01 = MPI_Wtime();
	double t02;
	DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>();
	if(!scramble || !delIsolated)
	{
		DEL->GenGraph500Data(initiator, scale, EDGEFACTOR, true, false);
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
	MPI_Barrier(MPI_COMM_WORLD);
	double t1 = MPI_Wtime();
	
	// conversion from distributed edge list, keeps self-loops, sums duplicates
	PSpMat_Int32 * G = new PSpMat_Int32(*DEL, false); 
	delete DEL;	// free memory before symmetricizing
	SpParHelper::Print("Created Sparse Matrix (with int32 local indices and values)\n");
	
	MPI_Barrier(MPI_COMM_WORLD);
	double redts = MPI_Wtime();
	G->Reduce(degrees, ::Row, plus<int64_t>(), static_cast<int64_t>(0));	// Identity is 0 
	MPI_Barrier(MPI_COMM_WORLD);
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
	
	MPI_Barrier(MPI_COMM_WORLD);
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

pySpParMat pySpParMat::copy()
{
	return pySpParMat(*this);
}

pySpParMat& pySpParMat::assign(const pySpParMat& other)
{
	A = other.A;
	return *this;
}

pySpParMat pySpParMat::SubsRef(const pyDenseParVec& rows, const pyDenseParVec& cols, bool inPlace, op::UnaryPredicateObj* matFilter)
{
	if (matFilter == NULL)
	{
		if (inPlace)
		{
			A(rows.v, cols.v, true);
			return pySpParMat();
		}
		else
		{
			return pySpParMat(A(rows.v, cols.v));
		}
	}
	else
	{ // The filtering semiring is slightly heavier than the default one, so only use it if needed
		if (inPlace)
		{
			SRFilterHelper<doubleint, doubleint, bool>::setFilterX(matFilter);
			SRFilterHelper<doubleint, bool, doubleint>::setFilterY(matFilter);
			A.SubsRef_SR<PCBBoolCopy1stSRing<doubleint>, PCBBoolCopy2ndSRing<doubleint> >(rows.v, cols.v, true);
			SRFilterHelper<doubleint, doubleint, bool>::setFilterX(NULL);
			SRFilterHelper<doubleint, bool, doubleint>::setFilterY(NULL);
			return pySpParMat();
		}
		else
		{
			SRFilterHelper<doubleint, doubleint, bool>::setFilterX(matFilter);
			SRFilterHelper<doubleint, bool, doubleint>::setFilterY(matFilter);
			pySpParMat ret(A.SubsRef_SR<PCBBoolCopy1stSRing<doubleint>, PCBBoolCopy2ndSRing<doubleint> >(rows.v, cols.v, false));
			SRFilterHelper<doubleint, doubleint, bool>::setFilterX(NULL);
			SRFilterHelper<doubleint, bool, doubleint>::setFilterY(NULL);
			return ret;
		}
	}
}
	
int64_t pySpParMat::removeSelfLoops()
{
	return A.RemoveLoops();
}

void pySpParMat::Apply(op::UnaryFunction* op)
{
	A.Apply(*op);
}

void pySpParMat::Apply(op::UnaryFunctionObj* op)
{
	A.Apply(*op);
}

void pySpParMat::DimWiseApply(int dim, const pyDenseParVec& values, op::BinaryFunction* f)
{
	A.DimApply((dim == Column() ? ::Column : ::Row), values.v, *f);
}

void pySpParMat::DimWiseApply(int dim, const pyDenseParVec& values, op::BinaryFunctionObj* f)
{
	A.DimApply((dim == Column() ? ::Column : ::Row), values.v, *f);
}

pySpParMat pySpParMat::Keep(op::UnaryPredicateObj* pred, bool inPlace)
{
	return pySpParMat(A.Prune(pcb_logical_not<op::UnaryPredicateObj>(*pred), inPlace));
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
		
	pyDenseParVec ret(len, identity);

	bf->getMPIOp();
	if (uf == NULL)
		A.Reduce(ret.v, (Dim)dim, *bf, doubleint(identity));
	else
		A.Reduce(ret.v, (Dim)dim, *bf, doubleint(identity), *uf);
	bf->releaseMPIOp();
	
	return ret;
}

void pySpParMat::Reduce(int dim, pyDenseParVec* ret, op::BinaryFunctionObj* bf, op::UnaryFunctionObj* uf, double identity)
{
	bf->getMPIOp();
	if (uf == NULL)
		A.Reduce(ret->v, (Dim)dim, *bf, doubleint(identity));
	else
		A.Reduce(ret->v, (Dim)dim, *bf, doubleint(identity), *uf);
	bf->releaseMPIOp();
}


void pySpParMat::Find(pyDenseParVec* outrows, pyDenseParVec* outcols, pyDenseParVec* outvals) const
{
	FullyDistVec<INDEXTYPE, INDEXTYPE> irows, icols;
	A.Find(irows, icols, outvals->v);
	outrows->v = irows;
	outcols->v = icols;
	//A.Find(outrows->v, outcols->v, outvals->v);
}

// the type of this ANullValue
#define NULL_PAR_TYPE   double
// how to pass in this ANullValue (i.e. ANull or doubleint(ANull))
#define A_NULL_ARG      doubleint(ANull)

#define MATCLASS pySpParMat
//#define MATCLASS_OBJ

#include "pyCommonMatFuncs.cpp"
