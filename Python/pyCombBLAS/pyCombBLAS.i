
%module pyCombBLAS

//%typemap(in) int64_t = long long;
//%typemap(out) int64_t = long long;
%apply long long {int64_t}

// This block gets copied verbatim into the header area of the generated wrapper. DiGraph has to
// be defined here somehow. Prefereably we'd #include "DiGraph.h", but that brings in templates which
// cause duplicate definition linker errors. Unless that gets resolved, we just redefine DiGraph
// omitting the templated protected members.

%{
#define SWIG_FILE_WITH_INIT

#include "pySpParMat.h"
#include "pySpParVec.h"
#include "pyDenseParVec.h"
%}


// This block gets called when the module is loaded. It is wrapped in extern "C".
%init %{
init_pyCombBLAS_MPI();
%}

// It's possible to have the generated python code also include some custom code.
// This may be a good place to add an atexit() to call mpi finalize.
%pragma(python) code="import atexit"
%pragma(python) code="atexit.register(DiGraph.finalize())"


// wrapped classes

class pySpParMat {
public:
	pySpParMat();

public:
	int64_t getnnz();
	int64_t getnrow();
	int64_t getncol();
	
public:	
	void load(const char* filename);
	void GenGraph500Edges(int scale);
	
public:
	pyDenseParVec* FindIndsOfColsWithSumGreaterThan(int64_t gt);
	//pyDenseParVec* Reduce_ColumnSums();
	void Apply_SetTo(int64_t v);
	
public:
	pySpParVec* SpMV_PlusTimes(const pySpParVec& v);
	pySpParVec* SpMV_SelMax(const pySpParVec& v);
	void SpMV_SelMax_inplace(pySpParVec& v);
};


class pySpParVec {
public:
	pySpParVec(int64_t length);
	//pySpParVec(const pySpParMat& commSource);

	pyDenseParVec* dense() const;
public:
	int64_t getnnz() const;

	pySpParVec& operator+=(const pySpParVec& other);
	pySpParVec& operator-=(const pySpParVec& other);
	pySpParVec* copy();

	void SetElement(int64_t index, int64_t numx);	// element-wise assignment
	int64_t GetElement(int64_t index);
	
public:	
	void invert(); // "~";  almost equal to logical_not
	void abs();
	
	bool any() const;
	bool all() const;
	
	int64_t intersectSize(const pySpParVec& other);
	
	void printall();

	
public:	
	void load(const char* filename);

public:
	pyDenseParVec* FindInds_GreaterThan(int64_t value);
	pyDenseParVec* FindInds_NotEqual(int64_t value);
	pySpParVec* SubsRef(const pySpParVec& ri);
	void setNumToInd();

public:
	static pySpParVec* zeros(int64_t howmany);
	static pySpParVec* range(int64_t howmany, int64_t start);
	
	
};

//pySpParVec* EWiseMult(const pySpParVec& a, const pySpParVec& b, bool exclude);
pySpParVec* EWiseMult(const pySpParVec& a, const pyDenseParVec& b, bool exclude, int64_t zero);
void EWiseMult_inplacefirst(pySpParVec& a, const pyDenseParVec& b, bool exclude, int64_t zero);

class pyDenseParVec {
public:
	pyDenseParVec(int64_t size, int64_t init);
	pyDenseParVec(int64_t size, int64_t init, int64_t zero);

	pySpParVec* sparse() const;
	pySpParVec* sparse(int64_t zero) const;
	
public:
	int length() const;
	
	void add(const pyDenseParVec& other);
	void add(const pySpParVec& other);
	pyDenseParVec& operator+=(const pyDenseParVec & rhs);
	pyDenseParVec& operator-=(const pyDenseParVec & rhs);
	pyDenseParVec& operator+=(const pySpParVec & rhs);
	pyDenseParVec& operator-=(const pySpParVec & rhs);
	//pyDenseParVec& operator=(const pyDenseParVec & rhs); // SWIG doesn't allow operator=, use copy instead.
	pyDenseParVec* copy();
	
	void SetElement (int64_t indx, int64_t numx);	// element-wise assignment
	int64_t GetElement (int64_t indx);	// element-wise fetch

	void RandPerm();

	void printall();
	
public:
	void invert(); // "~";  almost equal to logical_not
	void abs();
	void negate();
	
	int64_t getnnz() const;
	int64_t getnz() const;

	
public:	
	void load(const char* filename);
	
public:
	int64_t Count_GreaterThan(int64_t value);
	pySpParVec* Find_totality();
	pyDenseParVec* FindInds_GreaterThan(int64_t value);
	pyDenseParVec* FindInds_NotEqual(int64_t value);
	
	pyDenseParVec* SubsRef(const pyDenseParVec& ri);
	void ApplyMasked_SetTo(const pySpParVec& mask, int64_t value);

public:
	static pyDenseParVec* range(int64_t howmany, int64_t start);

};



//void init();
void finalize();
bool root();
