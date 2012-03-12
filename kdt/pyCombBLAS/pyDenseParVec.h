#ifndef PY_DENSE_PAR_VEC_H
#define PY_DENSE_PAR_VEC_H

#include "pyCombBLAS.h"

//INTERFACE_INCLUDE_BEGIN
class pyDenseParVec {
//INTERFACE_INCLUDE_END
	typedef int64_t INDEXTYPE;
	typedef doubleint NUMTYPE;
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
	pyDenseParVec(int64_t size, double init = 0);
	
	pySpParVec sparse() const;
	pySpParVec sparse(double zero) const;
	
public:
	int64_t len() const;
	int64_t __len__() const;
	
	pyDenseParVec copy();
	
	pyDenseParVec SubsRef(const pyDenseParVec& ri);

	void RandPerm(); // Randomly permutes the vector
	pyDenseParVec Sort(); // Does an in-place sort and returns the permutation used in the sort.
	pyDenseParVec TopK(int64_t k); // Returns a vector of the k largest elements.

	void printall();
	
public:
	
	int64_t getnee() const;
	
public:	
	void load(const char* filename);
	void save(const char* filename);
	
public:
	int64_t Count(op::UnaryFunction* op);
	int64_t Count(op::UnaryPredicateObj* op);
	double Reduce(op::BinaryFunction* f, op::UnaryFunction* uf = NULL);
	double Reduce(op::BinaryFunctionObj* f, op::UnaryFunctionObj* uf, double init);
	pySpParVec Find(op::UnaryFunction* op);
	pySpParVec Find(op::UnaryPredicateObj* op);
	pySpParVec __getitem__(op::UnaryFunction* op);
	pyDenseParVec FindInds(op::UnaryFunction* op);
	pyDenseParVec FindInds(op::UnaryPredicateObj* op);
	void Apply(op::UnaryFunction* op);
	void Apply(op::UnaryFunctionObj* op);
	void ApplyInd(op::BinaryFunctionObj* op);
	void ApplyMasked(op::UnaryFunction* op, const pySpParVec& mask);
	void ApplyMasked(op::UnaryFunctionObj* op, const pySpParVec& mask);

	void EWiseApply(const pyDenseParVecObj1& other, op::BinaryFunctionObj *op, op::BinaryPredicateObj *doOp, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred);
	void EWiseApply(const pyDenseParVecObj2& other, op::BinaryFunctionObj *op, op::BinaryPredicateObj *doOp, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred);
	void EWiseApply(const pyDenseParVec&     other, op::BinaryFunctionObj *op, op::BinaryPredicateObj *doOp, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred);
	void EWiseApply(const pySpParVecObj1& other, op::BinaryFunctionObj *op, op::BinaryPredicateObj *doOp, bool doNulls, Obj1 nullValue, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred);
	void EWiseApply(const pySpParVecObj2& other, op::BinaryFunctionObj *op, op::BinaryPredicateObj *doOp, bool doNulls, Obj2 nullValue, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred);
	void EWiseApply(const pySpParVec&     other, op::BinaryFunctionObj *op, op::BinaryPredicateObj *doOp, bool doNulls, double nullValue, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred);

	// predicate versions.
	void EWiseApply(const pyDenseParVecObj1& other, op::BinaryPredicateObj *op, op::BinaryPredicateObj *doOp, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred);
	void EWiseApply(const pyDenseParVecObj2& other, op::BinaryPredicateObj *op, op::BinaryPredicateObj *doOp, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred);
	void EWiseApply(const pyDenseParVec&     other, op::BinaryPredicateObj *op, op::BinaryPredicateObj *doOp, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred);
	void EWiseApply(const pySpParVecObj1& other, op::BinaryPredicateObj *op, op::BinaryPredicateObj *doOp, bool doNulls, Obj1 nullValue, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred);
	void EWiseApply(const pySpParVecObj2& other, op::BinaryPredicateObj *op, op::BinaryPredicateObj *doOp, bool doNulls, Obj2 nullValue, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred);
	void EWiseApply(const pySpParVec&     other, op::BinaryPredicateObj *op, op::BinaryPredicateObj *doOp, bool doNulls, double nullValue, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred);

public:
	static pyDenseParVec range(int64_t howmany, int64_t start);
	
public:
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
