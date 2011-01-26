#ifndef PY_DENSE_PAR_VEC_H
#define PY_DENSE_PAR_VEC_H

#include "pyCombBLAS.h"
class pySpParMat;
class pySpParVec;
class pyDenseParVec;

//INTERFACE_INCLUDE_BEGIN
class pyDenseParVec {
//INTERFACE_INCLUDE_END
protected:

	FullyDistVec<int64_t, int64_t> v;
	
	friend class pySpParVec;
	friend class pySpParMat;
	friend pySpParVec* EWiseMult(const pySpParVec& a, const pyDenseParVec& b, bool exclude, int64_t zero);
	friend void EWiseMult_inplacefirst(pySpParVec& a, const pyDenseParVec& b, bool exclude, int64_t zero);

	pyDenseParVec();
/////////////// everything below this appears in python interface:
//INTERFACE_INCLUDE_BEGIN
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
//INTERFACE_INCLUDE_END

#endif
