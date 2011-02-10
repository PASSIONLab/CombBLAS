#ifndef PY_SP_PAR_VEC_H
#define PY_SP_PAR_VEC_H

#include "pyCombBLAS.h"
class pySpParMat;
class pySpParVec;
class pyDenseParVec;

//INTERFACE_INCLUDE_BEGIN
class pySpParVec {
//INTERFACE_INCLUDE_END
protected:

	typedef int64_t INDEXTYPE;
	FullyDistSpVec<INDEXTYPE, doubleint> v;
	
	//pySpParVec(SpParVec<int64_t, int64_t> & in_v);
	
	friend class pySpParMat;
	friend class pyDenseParVec;
	
	friend pySpParVec* EWiseMult(const pySpParVec& a, const pySpParVec& b, bool exclude);
	friend pySpParVec* EWiseMult(const pySpParVec& a, const pyDenseParVec& b, bool exclude, double zero);
	friend void EWiseMult_inplacefirst(pySpParVec& a, const pyDenseParVec& b, bool exclude, double zero);


	pySpParVec(); // used for initializing temporaries to be returned
/////////////// everything below this appears in python interface:
//INTERFACE_INCLUDE_BEGIN
public:
	pySpParVec(int64_t length);
	//pySpParVec(const pySpParMat& commSource);
	
	pyDenseParVec* dense() const;

public:
	int64_t getnee() const;
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
	
	double Reduce(op::BinaryFunction* f, op::UnaryFunction* uf = NULL);
	
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
	
	double __getitem__(int64_t key);
	double __getitem__(double  key);
	pySpParVec* __getitem__(const pySpParVec& key);
	
	void __setitem__(int64_t key, double value);
	void __setitem__(double  key, double value);
	void __setitem__(const pyDenseParVec& key, const pyDenseParVec& value);
	//void __setitem__(const pyDenseParVec& key, int64_t value);
	void __setitem__(const char* key, double value);	
	
	char* __repr__();

};

//      EWiseMult has 2 flavors:
//      - if Exclude is false, will do element-wise multiplication
//      - if Exclude is true, will remove from the result vector all elements
//          whose corresponding element of the second vector is "nonzero"
//          (i.e., not equal to the sparse vector's identity value)  '

//pySpParVec* EWiseMult(const pySpParVec& a, const pySpParVec& b, bool exclude);
pySpParVec* EWiseMult(const pySpParVec& a, const pyDenseParVec& b, bool exclude, double zero);
void EWiseMult_inplacefirst(pySpParVec& a, const pyDenseParVec& b, bool exclude, double zero);

// compiler can't find the CombBLAS EWiseMult for some strange reason
pySpParMat* EWiseMult(const pySpParMat& A1, const pySpParMat& A2, bool exclude);

//INTERFACE_INCLUDE_END

#endif
