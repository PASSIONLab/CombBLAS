#ifndef PY_DENSE_PAR_VEC_Obj1_H
#define PY_DENSE_PAR_VEC_Obj1_H

#include "pyCombBLAS.h"

//INTERFACE_INCLUDE_BEGIN
/*
class pyDenseParVecObj1 {
//INTERFACE_INCLUDE_END
	typedef int64_t INDEXTYPE;
	typedef FullyDistVec<INDEXTYPE, doubleint> VectType;
	
	public:
	VectType v;

protected:
	
	friend class pySpParVecObj1;
	friend class pySpParMat;
	friend class pySpParMatBool;

	pyDenseParVecObj1();
	pyDenseParVecObj1(VectType other);
/////////////// everything below this appears in python interface:
//INTERFACE_INCLUDE_BEGIN
public:
	pyDenseParVecObj1(int64_t size, double init);
	pyDenseParVecObj1(int64_t size, double init, double zero);
	
	pySpParVecObj1 sparse() const;
	//pySpParVecObj1 sparse(double zero) const;
	
public:
	int64_t len() const;
	int64_t __len__() const;
	
	//pyDenseParVecObj1 operator==(const pyDenseParVecObj1& other);
	//pyDenseParVecObj1 operator!=(const pyDenseParVecObj1& other);

	pyDenseParVecObj1 copy();
	
	//pyDenseParVecObj1 SubsRef(const pyDenseParVec& ri);

	void RandPerm(); // Randomly permutes the vector
	//pyDenseParVecObj1 Sort(); // Does an in-place sort and returns the permutation used in the sort.
	//pyDenseParVecObj1 TopK(int64_t k); // Returns a vector of the k largest elements.

	void printall();
	
public:
	
	int64_t getnee() const;
	//int64_t getnnz() const;
	//int64_t getnz() const;
	//bool any() const;
	
public:	
	void load(const char* filename);
	
public:
	int64_t Count(op::UnaryPredicateObj* op);
	double Reduce(op::BinaryFunctionObj* f, op::UnaryFunctionObj* uf = NULL);
	pySpParVecObj1 Find(op::UnaryPredicateObj* op);
	pySpParVecObj1 __getitem__(op::UnaryPredicateObj* op);
	pyDenseParVecObj1 FindInds(op::UnaryPredicateObj* op);
	void Apply(op::UnaryFunctionObj* op);
	void ApplyMasked(op::UnaryFunctionObj* op, const pySpParVec& mask);
	void EWiseApply(const pyDenseParVecObj1& other, op::BinaryFunctionObj *f);
	void EWiseApply(const pySpParVecObj1& other, op::BinaryFunctionObj *f, bool doNulls = false, double nullValue = 0);

public:
	//static pyDenseParVecObj1 range(int64_t howmany, int64_t start);
	
public:
	// Functions from PyCombBLAS
	
	double __getitem__(int64_t key);
	//pyDenseParVecObj1 __getitem__(const pyDenseParVec& key);

	void __setitem__(int64_t key, double value);
	//void __setitem__(const pySpParVec& key, const pySpParVec& value);
	//void __setitem__(const pySpParVec& key, double value);
};
*/
//INTERFACE_INCLUDE_END

#endif
