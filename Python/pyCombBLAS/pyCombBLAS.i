
%module pyCombBLAS

//%typemap(in) int64_t = long long;
//%typemap(out) int64_t = long long;
%apply long long {int64_t}

// This block gets copied verbatim into the header area of the generated wrapper.

%{
#define SWIG_FILE_WITH_INIT

#include "pyCombBLAS.h"
%}


// This block gets called when the module is loaded. It is wrapped in extern "C".
%init %{
init_pyCombBLAS_MPI();
%}

// It is possible to have the generated python code also include some custom code.
// This may be a good place to add an atexit() to call mpi finalize.
%pragma(python) code="import atexit"
%pragma(python) code="atexit.register(DiGraph.finalize())"


// Language independent exception handler
%include exception.i    

%exception {
	try {
		$action
	} catch(string& stringReason) {
		const char* sData = (char*)stringReason.c_str();
		SWIG_exception(SWIG_RuntimeError,sData);
		SWIG_exception(SWIG_IndexError,sData);
	} catch(...) {
		SWIG_exception(SWIG_RuntimeError,"Unknown exception");
	}
}


// wrapped classes

class pySpParMat {
public:
	pySpParMat();
	//pySpParMat(int64_t m, int64_t n, pyDenseParVec* rows, pyDenseParVec* cols, pyDenseParVec* vals);

public:
	int64_t getnnz();
	int64_t getnrow();
	int64_t getncol();
	
public:	
	void load(const char* filename);
	void GenGraph500Edges(int scale);
	double GenGraph500Edges(int scale, pyDenseParVec& pyDegrees);
	
public:
	pySpParMat* copy();
	
	void Apply(op::UnaryFunction* op);
	void Prune(op::UnaryFunction* op);
	
	pyDenseParVec* Reduce(int dim, op::BinaryFunction* f, int64_t identity = 0);
	
	void Transpose();
	//void EWiseMult(pySpParMat* rhs, bool exclude);

	//void Find(pyDenseParVec* outrows, pyDenseParVec* outcols, pyDenseParVec* outvals) const;
public:
	pySpParVec* SpMV_PlusTimes(const pySpParVec& v);
	pySpParVec* SpMV_SelMax(const pySpParVec& v);
	void SpMV_SelMax_inplace(pySpParVec& v);
	
public:
	static int Column() { return ::Column; }
	static int Row() { return ::Row; }
};

class pySpParVec {
public:
	pySpParVec(int64_t length);
	//pySpParVec(const pySpParMat& commSource);
	
	pyDenseParVec* dense() const;

public:
	int64_t getne() const;
	int64_t getnnz() const;
	int64_t __len__() const;
	int64_t len() const;

	pySpParVec* operator+(const pySpParVec& other);
	pySpParVec* operator-(const pySpParVec& other);
	pySpParVec* operator+(const pyDenseParVec& other);
	pySpParVec* operator-(const pyDenseParVec& other);

	pySpParVec& operator+=(const pySpParVec& other);
	pySpParVec& operator-=(const pySpParVec& other);
	pySpParVec& operator+=(const pyDenseParVec& other);
	pySpParVec& operator-=(const pyDenseParVec& other);
	pySpParVec* copy();

	void SetElement(int64_t index, int64_t numx);	// element-wise assignment
	int64_t GetElement(int64_t index);
	
public:	
	//void invert(); // "~";  almost equal to logical_not
	//void abs();
	
	bool any() const; // any nonzeros
	bool all() const; // all nonzeros
	
	int64_t intersectSize(const pySpParVec& other);
	
	void printall();
	
public:	
	void load(const char* filename);

public:
	// The functions commented out here presently do not exist in CombBLAS
	int64_t Count(op::UnaryFunction* op);
	//pySpParVec* Find(op::UnaryFunction* op);
	//pyDenseParVec* FindInds(op::UnaryFunction* op);
	void Apply(op::UnaryFunction* op);
	//void ApplyMasked(op::UnaryFunction* op, const pySpParVec& mask);

	pySpParVec* SubsRef(const pySpParVec& ri);
	
	int64_t Reduce(op::BinaryFunction* f);
	
	pySpParVec* Sort();
	
	void setNumToInd();

public:
	static pySpParVec* zeros(int64_t howmany);
	static pySpParVec* range(int64_t howmany, int64_t start);
	
public:
	// Functions from PyCombBLAS
	pySpParVec* abs();
	void __delitem__(const pyDenseParVec& key);
	void __delitem__(int64_t key);
	int64_t __getitem__(int64_t key);
	pySpParVec* __getitem__(const pySpParVec& key);
	void __setitem__(int64_t key, int64_t value);
	void __setitem__(const pyDenseParVec& key, const pyDenseParVec& value);
	//void __setitem__(const pyDenseParVec& key, int64_t value);
	void __setitem__(const char* key, int64_t value);	
	char* __repr__();

};

//      EWiseMult has 2 flavors:
//      - if Exclude is false, will do element-wise multiplication
//      - if Exclude is true, will remove from the result vector all elements
//          whose corresponding element of the second vector is "nonzero"
//          (i.e., not equal to the sparse vector's identity value)  '

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
	int64_t len() const;
	int64_t __len__() const;
	
	void add(const pyDenseParVec& other);
	void add(const pySpParVec& other);
	pyDenseParVec& operator+=(const pyDenseParVec & rhs);
	pyDenseParVec& operator-=(const pyDenseParVec & rhs);
	pyDenseParVec& operator+=(const pySpParVec & rhs);
	pyDenseParVec& operator-=(const pySpParVec & rhs);
	pyDenseParVec& operator*=(const pyDenseParVec& rhs);
	pyDenseParVec& operator*=(const pySpParVec& rhs);
	//pyDenseParVec& operator=(const pyDenseParVec & rhs); // SWIG doesn't allow operator=
	
	pyDenseParVec* operator+(const pyDenseParVec & rhs);
	pyDenseParVec* operator-(const pyDenseParVec & rhs);
	pyDenseParVec* operator+(const pySpParVec & rhs);
	pyDenseParVec* operator-(const pySpParVec & rhs);
	pyDenseParVec* operator*(const pyDenseParVec& rhs);
	pyDenseParVec* operator*(const pySpParVec& rhs);
	
	pyDenseParVec* operator==(const pyDenseParVec& other);
	pyDenseParVec* operator!=(const pyDenseParVec& other);

	pyDenseParVec* copy();
	
	void SetElement (int64_t indx, int64_t numx);	// element-wise assignment
	int64_t GetElement (int64_t indx);	// element-wise fetch
	
	pyDenseParVec* SubsRef(const pyDenseParVec& ri);

	void RandPerm();

	void printall();
	
public:
	
	int64_t getnee() const;
	int64_t getnnz() const;
	int64_t getnz() const;
	bool any() const;
	
public:	
	void load(const char* filename);
	
public:
	int64_t Count(op::UnaryFunction* op);
	pySpParVec* Find(op::UnaryFunction* op);
	pySpParVec* __getitem__(op::UnaryFunction* op);
	pyDenseParVec* FindInds(op::UnaryFunction* op);
	void Apply(op::UnaryFunction* op);
	void ApplyMasked(op::UnaryFunction* op, const pySpParVec& mask);
	void EWiseApply(const pyDenseParVec& other, op::BinaryFunction *f);
	void EWiseApply(const pySpParVec& other, op::BinaryFunction *f, bool doNulls = false, int64_t nullValue = 0);

public:
	static pyDenseParVec* range(int64_t howmany, int64_t start);
	
public:
	// Functions from PyCombBLAS
	pyDenseParVec* abs();
	
	pyDenseParVec& operator+=(int64_t value);
	pyDenseParVec* operator+(int64_t value);
	pyDenseParVec& operator-=(int64_t value);
	pyDenseParVec* operator-(int64_t value);
	
	pyDenseParVec* __and__(const pyDenseParVec& other);
	
	int64_t __getitem__(int64_t key);
	pyDenseParVec* __getitem__(const pyDenseParVec& key);
	void __setitem__(int64_t key, int64_t value);
	void __setitem__(const pySpParVec& key, const pySpParVec& value);
	void __setitem__(const pySpParVec& key, int64_t value);
};

namespace op {

class UnaryFunction {

	protected:
	UnaryFunction(): op(NULL) {}
	public:
	~UnaryFunction() { /*delete op; op = NULL;*/ }
	
	int64_t operator()(const int64_t x) const
	{
		return (*op)(x);
	}
};

UnaryFunction* set(int64_t val);
UnaryFunction* identity();
UnaryFunction* safemultinv();
UnaryFunction* abs();
UnaryFunction* negate();
UnaryFunction* bitwise_not();
UnaryFunction* logical_not();
UnaryFunction* totality();

class BinaryFunction {
	protected:
	BinaryFunction(): op(NULL), commutable(false), associative(false) {}
	public:
	~BinaryFunction() { /*delete op; op = NULL;*/ }
	
	bool commutable;
	bool associative;
	
	int64_t operator()(const int64_t& x, const int64_t& y) const
	{
		return (*op)(x, y);
	}

};

BinaryFunction* plus();
BinaryFunction* minus();
BinaryFunction* multiplies();
BinaryFunction* divides();
BinaryFunction* modulus();

BinaryFunction* max();
BinaryFunction* min();

BinaryFunction* bitwise_and();
BinaryFunction* bitwise_or();
BinaryFunction* bitwise_xor();
BinaryFunction* logical_and();
BinaryFunction* logical_or();
BinaryFunction* logical_xor();

BinaryFunction* equal_to();
BinaryFunction* not_equal_to();
BinaryFunction* greater();
BinaryFunction* less();
BinaryFunction* greater_equal();
BinaryFunction* less_equal();


// Glue functions

UnaryFunction* bind1st(BinaryFunction* op, int64_t val);
UnaryFunction* bind2nd(BinaryFunction* op, int64_t val);
UnaryFunction* compose1(UnaryFunction* f, UnaryFunction* g); // h(x) is the same as f(g(x))
UnaryFunction* compose2(BinaryFunction* f, UnaryFunction* g1, UnaryFunction* g2); // h(x) is the same as f(g1(x), g2(x))
UnaryFunction* not1(UnaryFunction* f);
BinaryFunction* not2(BinaryFunction* f);

} // namespace op




//void init();
void finalize();
bool root();
