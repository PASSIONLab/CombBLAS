#ifndef PY_DENSE_PAR_VEC_Obj2_H
#define PY_DENSE_PAR_VEC_Obj2_H

#include "pyCombBLAS.h"

//INTERFACE_INCLUDE_BEGIN
/*
class pyDenseParVecObj2 {
//INTERFACE_INCLUDE_END
	typedef int64_t INDEXTYPE;
	typedef FullyDistVec<INDEXTYPE, doubleint> VectType;
	
	public:
	VectType v;

protected:
	
	friend class pySpParVecObj2;
	friend class pySpParMat;
	friend class pySpParMatBool;

	pyDenseParVecObj2();
	pyDenseParVecObj2(VectType other);
/////////////// everything below this appears in python interface:
//INTERFACE_INCLUDE_BEGIN
public:
	pyDenseParVecObj2(int64_t size, double init);
	pyDenseParVecObj2(int64_t size, double init, double zero);
	
	pySpParVecObj2 sparse() const;
	//pySpParVecObj2 sparse(double zero) const;
	
public:
	int64_t len() const;
	int64_t __len__() const;
	
	//pyDenseParVecObj2 operator==(const pyDenseParVecObj2& other);
	//pyDenseParVecObj2 operator!=(const pyDenseParVecObj2& other);

	pyDenseParVecObj2 copy();
	
	//pyDenseParVecObj2 SubsRef(const pyDenseParVec& ri);

	void RandPerm(); // Randomly permutes the vector
	//pyDenseParVecObj2 Sort(); // Does an in-place sort and returns the permutation used in the sort.
	//pyDenseParVecObj2 TopK(int64_t k); // Returns a vector of the k largest elements.

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
	pySpParVecObj2 Find(op::UnaryPredicateObj* op);
	pySpParVecObj2 __getitem__(op::UnaryPredicateObj* op);
	pyDenseParVecObj2 FindInds(op::UnaryPredicateObj* op);
	void Apply(op::UnaryFunctionObj* op);
	void ApplyMasked(op::UnaryFunctionObj* op, const pySpParVec& mask);
	void EWiseApply(const pyDenseParVecObj2& other, op::BinaryFunctionObj *f);
	void EWiseApply(const pySpParVecObj2& other, op::BinaryFunctionObj *f, bool doNulls = false, double nullValue = 0);

public:
	//static pyDenseParVecObj2 range(int64_t howmany, int64_t start);
	
public:
	// Functions from PyCombBLAS
	
	double __getitem__(int64_t key);
	//pyDenseParVecObj2 __getitem__(const pyDenseParVec& key);

	void __setitem__(int64_t key, double value);
	//void __setitem__(const pySpParVec& key, const pySpParVec& value);
	//void __setitem__(const pySpParVec& key, double value);
};
*/
//INTERFACE_INCLUDE_END

#endif
