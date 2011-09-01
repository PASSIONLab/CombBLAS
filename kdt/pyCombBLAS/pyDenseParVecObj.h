#ifndef PY_DENSE_PAR_VEC_H
#define PY_DENSE_PAR_VEC_H

#include "pyCombBLAS.h"

//INTERFACE_INCLUDE_BEGIN
class pyDenseParVecObj {
//INTERFACE_INCLUDE_END
	typedef int64_t INDEXTYPE;
	typedef FullyDistVec<INDEXTYPE, doubleint> VectType;
	
	public:
	VectType v;

protected:
	
	friend class pySpParVecObj;
	friend class pySpParMat;
	friend class pySpParMatBool;

	pyDenseParVecObj();
	pyDenseParVecObj(VectType other);
/////////////// everything below this appears in python interface:
//INTERFACE_INCLUDE_BEGIN
public:
	pyDenseParVecObj(int64_t size, double init);
	pyDenseParVecObj(int64_t size, double init, double zero);
	
	pySpParVecObj sparse() const;
	//pySpParVecObj sparse(double zero) const;
	
public:
	int64_t len() const;
	int64_t __len__() const;
	
	/*
	void add(const pyDenseParVecObj& other);
	void add(const pySpParVecObj& other);
	pyDenseParVecObj& operator+=(const pyDenseParVecObj & rhs);
	pyDenseParVecObj& operator-=(const pyDenseParVecObj & rhs);
	pyDenseParVecObj& operator+=(const pySpParVecObj & rhs);
	pyDenseParVecObj& operator-=(const pySpParVecObj & rhs);
	pyDenseParVecObj& operator*=(const pyDenseParVecObj& rhs);
	pyDenseParVecObj& operator*=(const pySpParVecObj& rhs);
	
	pyDenseParVecObj operator+(const pyDenseParVecObj & rhs);
	pyDenseParVecObj operator-(const pyDenseParVecObj & rhs);
	pyDenseParVecObj operator+(const pySpParVecObj & rhs);
	pyDenseParVecObj operator-(const pySpParVecObj & rhs);
	pyDenseParVecObj operator*(const pyDenseParVecObj& rhs);
	pyDenseParVecObj operator*(const pySpParVecObj& rhs);
	*/
	pyDenseParVecObj operator==(const pyDenseParVecObj& other);
	pyDenseParVecObj operator!=(const pyDenseParVecObj& other);

	pyDenseParVecObj copy();
	
	//pyDenseParVecObj SubsRef(const pyDenseParVec& ri);

	void RandPerm(); // Randomly permutes the vector
	//pyDenseParVecObj Sort(); // Does an in-place sort and returns the permutation used in the sort.
	//pyDenseParVecObj TopK(int64_t k); // Returns a vector of the k largest elements.

	void printall();
	
public:
	
	int64_t getnee() const;
	//int64_t getnnz() const;
	//int64_t getnz() const;
	//bool any() const;
	
public:	
	void load(const char* filename);
	
public:
	int64_t Count(op::UnaryFunction* op);
	double Reduce(op::BinaryFunction* f, op::UnaryFunction* uf = NULL);
	pySpParVecObj Find(op::UnaryFunction* op);
	pySpParVecObj __getitem__(op::UnaryFunction* op);
	pyDenseParVecObj FindInds(op::UnaryFunction* op);
	void Apply(op::UnaryFunction* op);
	void ApplyMasked(op::UnaryFunction* op, const pySpParVec& mask);
	void EWiseApply(const pyDenseParVecObj& other, op::BinaryFunction *f);
	void EWiseApply(const pySpParVecObj& other, op::BinaryFunction *f, bool doNulls = false, double nullValue = 0);

public:
	static pyDenseParVecObj range(int64_t howmany, int64_t start);
	
public:
	// Functions from PyCombBLAS

	/*	
	pyDenseParVecObj& operator+=(double value);
	pyDenseParVecObj operator+(double value);
	pyDenseParVecObj& operator-=(double value);
	pyDenseParVecObj operator-(double value);
	
	pyDenseParVecObj __and__(const pyDenseParVecObj& other);
	*/
	
	double __getitem__(int64_t key);
	//pyDenseParVecObj __getitem__(const pyDenseParVecObj& key);

	void __setitem__(int64_t key, double value);
	//void __setitem__(const pySpParVec& key, const pySpParVec& value);
	//void __setitem__(const pySpParVec& key, double value);
};
//INTERFACE_INCLUDE_END

#endif
