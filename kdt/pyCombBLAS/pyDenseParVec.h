#ifndef PY_DENSE_PAR_VEC_H
#define PY_DENSE_PAR_VEC_H

#include "pyCombBLAS.h"

//INTERFACE_INCLUDE_BEGIN
class pyDenseParVec {
//INTERFACE_INCLUDE_END
	typedef int64_t INDEXTYPE;
	typedef FullyDistVec<INDEXTYPE, doubleint> VectType;
	
	public:
	VectType v;

protected:
	
	friend class pySpParVec;
	friend class pySpParMat;
	friend class pySpParMatBool;
	friend pySpParVec EWiseMult(const pySpParVec& a, const pyDenseParVec& b, bool exclude, double zero);
	friend void EWiseMult_inplacefirst(pySpParVec& a, const pyDenseParVec& b, bool exclude, double zero);

	pyDenseParVec();
	pyDenseParVec(VectType other);
/////////////// everything below this appears in python interface:
//INTERFACE_INCLUDE_BEGIN
public:
	pyDenseParVec(int64_t size, double init);
	pyDenseParVec(int64_t size, double init, double zero);
	
	pySpParVec sparse() const;
	pySpParVec sparse(double zero) const;
	
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
	
	pyDenseParVec operator+(const pyDenseParVec & rhs);
	pyDenseParVec operator-(const pyDenseParVec & rhs);
	pyDenseParVec operator+(const pySpParVec & rhs);
	pyDenseParVec operator-(const pySpParVec & rhs);
	pyDenseParVec operator*(const pyDenseParVec& rhs);
	pyDenseParVec operator*(const pySpParVec& rhs);
	
	pyDenseParVec operator==(const pyDenseParVec& other);
	pyDenseParVec operator!=(const pyDenseParVec& other);

	pyDenseParVec copy();
	
	pyDenseParVec SubsRef(const pyDenseParVec& ri);

	void RandPerm(); // Randomly permutes the vector
	pyDenseParVec Sort(); // Does an in-place sort and returns the permutation used in the sort.
	pyDenseParVec TopK(int64_t k); // Returns a vector of the k largest elements.

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
	double Reduce(op::BinaryFunction* f, op::UnaryFunction* uf = NULL);
	pySpParVec Find(op::UnaryFunction* op);
	pySpParVec __getitem__(op::UnaryFunction* op);
	pyDenseParVec FindInds(op::UnaryFunction* op);
	void Apply(op::UnaryFunction* op);
	void ApplyMasked(op::UnaryFunction* op, const pySpParVec& mask);
	void EWiseApply(const pyDenseParVec& other, op::BinaryFunction *f);
	void EWiseApply(const pySpParVec& other, op::BinaryFunction *f, bool doNulls = false, double nullValue = 0);

public:
	static pyDenseParVec range(int64_t howmany, int64_t start);
	
public:
	// Functions from PyCombBLAS
	pyDenseParVec abs();
	
	pyDenseParVec& operator+=(double value);
	pyDenseParVec operator+(double value);
	pyDenseParVec& operator-=(double value);
	pyDenseParVec operator-(double value);
	
	pyDenseParVec __and__(const pyDenseParVec& other);
	
	double __getitem__(int64_t key);
	double __getitem__(double  key);
	pyDenseParVec __getitem__(const pyDenseParVec& key);

	void __setitem__(int64_t key, double value);
	void __setitem__(double  key, double value);
	void __setitem__(const pySpParVec& key, const pySpParVec& value);
	void __setitem__(const pySpParVec& key, double value);
};
//INTERFACE_INCLUDE_END

#endif
