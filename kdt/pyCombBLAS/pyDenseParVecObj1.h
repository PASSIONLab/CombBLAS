#ifndef PY_DENSE_PAR_VEC_Obj1_H
#define PY_DENSE_PAR_VEC_Obj1_H

#include "pyCombBLAS.h"

//INTERFACE_INCLUDE_BEGIN

class pyDenseParVecObj1 {
//INTERFACE_INCLUDE_END
	typedef int64_t INDEXTYPE;
	typedef FullyDistVec<INDEXTYPE, Obj1> VectType;
	
	public:
	VectType v;

protected:
	
	friend class pySpParVecObj1;
	friend class pySpParVecObj2;
	friend class pySpParMat;
	friend class pySpParMatObj1;
	friend class pySpParMatObj2;
	friend class pySpParMatBool;

	pyDenseParVecObj1();
	pyDenseParVecObj1(VectType other);
/////////////// everything below this appears in python interface:
//INTERFACE_INCLUDE_BEGIN
public:
	pyDenseParVecObj1(int64_t size, Obj1 init);
	
	pySpParVecObj1 sparse(op::UnaryPredicateObj* keep = NULL) const;
	//pySpParVecObj1 sparse(double zero) const;
	
public:
	int64_t len() const;
	int64_t __len__() const;
	
	//pyDenseParVecObj1 operator==(const pyDenseParVecObj1& other);
	//pyDenseParVecObj1 operator!=(const pyDenseParVecObj1& other);

	pyDenseParVecObj1 copy();
	
	pyDenseParVecObj1 SubsRef(const pyDenseParVec& ri);

	void RandPerm(); // Randomly permutes the vector
	pyDenseParVec Sort(); // Does an in-place sort and returns the permutation used in the sort.
	pyDenseParVecObj1 TopK(int64_t k); // Returns a vector of the k largest elements.

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
	Obj1 Reduce(op::BinaryFunctionObj* f, op::UnaryFunctionObj* uf = NULL);
	pySpParVecObj1 Find(op::UnaryPredicateObj* op);
	pySpParVecObj1 __getitem__(op::UnaryPredicateObj* op);
	pyDenseParVec FindInds(op::UnaryPredicateObj* op);
	void Apply(op::UnaryFunctionObj* op);
	void ApplyMasked(op::UnaryFunctionObj* op, const pySpParVec& mask);
	void EWiseApply(const pyDenseParVecObj1& other, op::BinaryFunctionObj *f, op::BinaryPredicateObj *doOp);
	void EWiseApply(const pyDenseParVecObj2& other, op::BinaryFunctionObj *f, op::BinaryPredicateObj *doOp);
	void EWiseApply(const pyDenseParVec&     other, op::BinaryFunctionObj *f, op::BinaryPredicateObj *doOp);
	void EWiseApply(const pySpParVecObj1& other, op::BinaryFunctionObj *f, op::BinaryPredicateObj *doOp, bool doNulls = false, Obj1 nullValue = Obj1());
	void EWiseApply(const pySpParVecObj2& other, op::BinaryFunctionObj *f, op::BinaryPredicateObj *doOp, bool doNulls = false, Obj2 nullValue = Obj2());
	void EWiseApply(const pySpParVec&     other, op::BinaryFunctionObj *f, op::BinaryPredicateObj *doOp, bool doNulls = false, double nullValue = 0);

	/* NEEDS to return new pyDenseParVec
	void EWiseApply(const pyDenseParVecObj1& other, op::BinaryPredicateObj *f, op::BinaryPredicateObj *doOp);
	void EWiseApply(const pyDenseParVecObj2& other, op::BinaryPredicateObj *f, op::BinaryPredicateObj *doOp);
	void EWiseApply(const pyDenseParVec&     other, op::BinaryPredicateObj *f, op::BinaryPredicateObj *doOp);
	void EWiseApply(const pySpParVecObj1& other, op::BinaryPredicateObj *f, op::BinaryPredicateObj *doOp, bool doNulls = false, Obj1 nullValue = Obj1());
	void EWiseApply(const pySpParVecObj2& other, op::BinaryPredicateObj *f, op::BinaryPredicateObj *doOp, bool doNulls = false, Obj2 nullValue = Obj2());
	void EWiseApply(const pySpParVec&     other, op::BinaryPredicateObj *f, op::BinaryPredicateObj *doOp, bool doNulls = false, double nullValue = 0);
	*/
public:
	//static pyDenseParVecObj1 range(int64_t howmany, int64_t start);
	
public:
	// Functions from PyCombBLAS
	
	Obj1 __getitem__(int64_t key);
	pyDenseParVecObj1 __getitem__(const pyDenseParVec& key);

	void __setitem__(int64_t key, Obj1 * value);
	//void __setitem__(const pySpParVec& key, const pySpParVecObj1& value);
	void __setitem__(const pySpParVec& key, Obj1 * value);
	
	char* __repr__();
};

//INTERFACE_INCLUDE_END

#endif
