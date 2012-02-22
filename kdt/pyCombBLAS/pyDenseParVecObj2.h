#ifndef PY_DENSE_PAR_VEC_Obj2_H
#define PY_DENSE_PAR_VEC_Obj2_H

#include "pyCombBLAS.h"

//INTERFACE_INCLUDE_BEGIN

class pyDenseParVecObj2 {
//INTERFACE_INCLUDE_END
	typedef int64_t INDEXTYPE;
	typedef Obj2 NUMTYPE;
	typedef FullyDistVec<INDEXTYPE, Obj2> VectType;
	
	public:
	VectType v;

protected:
	
	friend class pySpParVecObj2;
	friend class pySpParVecObj1;
	friend class pySpParMat;
	friend class pySpParMatObj2;
	friend class pySpParMatObj1;
	friend class pySpParMatBool;

	pyDenseParVecObj2();
	pyDenseParVecObj2(VectType other);
/////////////// everything below this appears in python interface:
//INTERFACE_INCLUDE_BEGIN
public:
	pyDenseParVecObj2(int64_t size, Obj2 init = Obj2());
	
	pySpParVecObj2 sparse(op::UnaryPredicateObj* keep = NULL) const;
	//pySpParVecObj2 sparse(double zero) const;
	
public:
	int64_t len() const;
	int64_t __len__() const;
	
	//pyDenseParVecObj2 operator==(const pyDenseParVecObj2& other);
	//pyDenseParVecObj2 operator!=(const pyDenseParVecObj2& other);

	pyDenseParVecObj2 copy();
	
	pyDenseParVecObj2 SubsRef(const pyDenseParVec& ri);

	void RandPerm(); // Randomly permutes the vector
	pyDenseParVec Sort(); // Does an in-place sort and returns the permutation used in the sort.
	pyDenseParVecObj2 TopK(int64_t k); // Returns a vector of the k largest elements.

	void printall();
	
public:
	
	int64_t getnee() const;
	//int64_t getnnz() const;
	//int64_t getnz() const;
	//bool any() const;
	
public:	
	void load(const char* filename);
	void save(const char* filename);
	
public:
	int64_t Count(op::UnaryPredicateObj* op);
	Obj2 Reduce(op::BinaryFunctionObj* f, op::UnaryFunctionObj* uf, Obj2 *init);
	double Reduce(op::BinaryFunctionObj* f, op::UnaryFunctionObj* uf, double init);
	pySpParVecObj2 Find(op::UnaryPredicateObj* op);
	pySpParVecObj2 __getitem__(op::UnaryPredicateObj* op);
	pyDenseParVec FindInds(op::UnaryPredicateObj* op);
	void Apply(op::UnaryFunctionObj* op);
	void ApplyMasked(op::UnaryFunctionObj* op, const pySpParVec& mask);
	void ApplyInd(op::BinaryFunctionObj* op);
	void EWiseApply(const pyDenseParVecObj2& other, op::BinaryFunctionObj *f, op::BinaryPredicateObj *doOp);
	void EWiseApply(const pyDenseParVecObj1& other, op::BinaryFunctionObj *f, op::BinaryPredicateObj *doOp);
	void EWiseApply(const pyDenseParVec&     other, op::BinaryFunctionObj *f, op::BinaryPredicateObj *doOp);
	void EWiseApply(const pySpParVecObj2& other, op::BinaryFunctionObj *f, op::BinaryPredicateObj *doOp, bool doNulls, Obj2 *nullValue);
	void EWiseApply(const pySpParVecObj1& other, op::BinaryFunctionObj *f, op::BinaryPredicateObj *doOp, bool doNulls, Obj1 *nullValue);
	void EWiseApply(const pySpParVec&     other, op::BinaryFunctionObj *f, op::BinaryPredicateObj *doOp, bool doNulls, double nullValue);

	/* NEEDS to return new pyDenseParVec
	void EWiseApply(const pyDenseParVecObj2& other, op::BinaryPredicateObj *f, op::BinaryPredicateObj *doOp);
	void EWiseApply(const pyDenseParVecObj1& other, op::BinaryPredicateObj *f, op::BinaryPredicateObj *doOp);
	void EWiseApply(const pyDenseParVec&     other, op::BinaryPredicateObj *f, op::BinaryPredicateObj *doOp);
	void EWiseApply(const pySpParVecObj2& other, op::BinaryPredicateObj *f, op::BinaryPredicateObj *doOp, bool doNulls, Obj2 *nullValue);
	void EWiseApply(const pySpParVecObj1& other, op::BinaryPredicateObj *f, op::BinaryPredicateObj *doOp, bool doNulls, Obj1 *nullValue);
	void EWiseApply(const pySpParVec&     other, op::BinaryPredicateObj *f, op::BinaryPredicateObj *doOp, bool doNulls, double nullValue);
	*/
public:
	//static pyDenseParVecObj2 range(int64_t howmany, int64_t start);
	
public:
	// Functions from PyCombBLAS
	
	Obj2 __getitem__(int64_t key);
	pyDenseParVecObj2 __getitem__(const pyDenseParVec& key);

	void __setitem__(int64_t key, Obj2 * value);
	//void __setitem__(const pySpParVec& key, const pySpParVecObj2& value);
	void __setitem__(const pySpParVec& key, Obj2 * value);
	
	char* __repr__();
};

//INTERFACE_INCLUDE_END

#endif
