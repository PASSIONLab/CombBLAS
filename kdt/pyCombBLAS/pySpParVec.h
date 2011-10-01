#ifndef PY_SP_PAR_VEC_H
#define PY_SP_PAR_VEC_H

#include "pyCombBLAS.h"

//INTERFACE_INCLUDE_BEGIN
class pySpParVec {
//INTERFACE_INCLUDE_END
public:
	typedef int64_t INDEXTYPE;
	typedef FullyDistSpVec<INDEXTYPE, doubleint> VectType;
	VectType v;
	
protected:

	friend class pySpParMat;
	friend class pySpParMatBool;
	friend class pyDenseParVec;
	
	friend pySpParVec EWiseMult(const pySpParVec& a, const pySpParVec& b, bool exclude);
	friend pySpParVec EWiseMult(const pySpParVec& a, const pyDenseParVec& b, bool exclude, double zero);
	friend void EWiseMult_inplacefirst(pySpParVec& a, const pyDenseParVec& b, bool exclude, double zero);

	pySpParVec(); // used for initializing temporaries to be returned
	pySpParVec(VectType other);

/////////////// everything below this appears in python interface:
//INTERFACE_INCLUDE_BEGIN
public:
	pySpParVec(int64_t length);
	
	pyDenseParVec dense() const;

public:
	int64_t getnee() const;
	int64_t getnnz() const;
	int64_t __len__() const;
	int64_t len() const;

	pySpParVec operator+(const pySpParVec& other);
	pySpParVec operator-(const pySpParVec& other);
	pySpParVec operator+(const pyDenseParVec& other);
	pySpParVec operator-(const pyDenseParVec& other);

	pySpParVec& operator+=(const pySpParVec& other);
	pySpParVec& operator-=(const pySpParVec& other);
	pySpParVec& operator+=(const pyDenseParVec& other);
	pySpParVec& operator-=(const pyDenseParVec& other);
	pySpParVec copy();

public:	
	bool any() const; // any nonzeros
	bool all() const; // all nonzeros
	
	int64_t intersectSize(const pySpParVec& other);
	
	void printall();
	
public:	
	void load(const char* filename);

public:
	// The functions commented out here presently do not exist in CombBLAS
	int64_t Count(op::UnaryFunction* op);
	int64_t Count(op::UnaryFunctionObj* op);
	//pySpParVec Find(op::UnaryFunction* op);
	//pyDenseParVec FindInds(op::UnaryFunction* op);
	void Apply(op::UnaryFunction* op);
	void Apply(op::UnaryFunctionObj* op);
	void ApplyInd(op::BinaryFunctionObj* op);
	//void ApplyMasked(op::UnaryFunction* op, const pySpParVec& mask);

	pyDenseParVec SubsRef(const pyDenseParVec& ri);
	
	double Reduce(op::BinaryFunction* f, op::UnaryFunction* uf = NULL);
	double Reduce(op::BinaryFunctionObj* f, op::UnaryFunctionObj* uf = NULL);
	
	pySpParVec Sort(); // Does an in-place sort and returns the permutation used in the sort.
	pyDenseParVec TopK(int64_t k); // Returns a vector of the k largest elements.
	
	void setNumToInd();

public:
	static pySpParVec zeros(int64_t howmany);
	static pySpParVec range(int64_t howmany, int64_t start);
	
public:
	// Functions from PyCombBLAS
	pySpParVec abs();
	void __delitem__(const pyDenseParVec& key);
	void __delitem__(int64_t key);
	
	double __getitem__(int64_t key);
	double __getitem__(double  key);
	pyDenseParVec __getitem__(const pyDenseParVec& key);
	
	void __setitem__(int64_t key, double value);
	void __setitem__(double  key, double value);
	void __setitem__(const pyDenseParVec& key, const pyDenseParVec& value);
	//void __setitem__(const pyDenseParVec& key, int64_t value);
	void __setitem__(const char* key, double value);	
	
	char* __repr__();

	friend pySpParVec EWiseApply(const pySpParVec& a, const pySpParVec&     b, op::BinaryFunction   * op, bool allowANulls, bool allowBNulls);



	friend pySpParVec EWiseApply(const pySpParVec& a, const pySpParVecObj1& b, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, double ANull, Obj1 BNull);
	friend pySpParVec EWiseApply(const pySpParVec& a, const pySpParVecObj2& b, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, double ANull, Obj2 BNull);
	friend pySpParVec EWiseApply(const pySpParVec& a, const pySpParVec&     b, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, double ANull, double BNull);

	friend pySpParVec EWiseApply(const pySpParVec& a, const pySpParVecObj1& b, op::BinaryPredicateObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, double ANull, Obj1 BNull);
	friend pySpParVec EWiseApply(const pySpParVec& a, const pySpParVecObj2& b, op::BinaryPredicateObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, double ANull, Obj2 BNull);
	friend pySpParVec EWiseApply(const pySpParVec& a, const pySpParVec&     b, op::BinaryPredicateObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, double ANull, double BNull);

	// Obj predicates:	
	friend pySpParVec EWiseApply(const pySpParVecObj1& a, const pySpParVecObj1& b, op::BinaryPredicateObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, Obj1 ANull, Obj1 BNull);
	friend pySpParVec EWiseApply(const pySpParVecObj1& a, const pySpParVecObj2& b, op::BinaryPredicateObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, Obj1 ANull, Obj2 BNull);
	friend pySpParVec EWiseApply(const pySpParVecObj1& a, const pySpParVec&     b, op::BinaryPredicateObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, Obj1 ANull, double BNull);

	friend pySpParVec EWiseApply(const pySpParVecObj2& a, const pySpParVecObj1& b, op::BinaryPredicateObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, Obj2 ANull, Obj1 BNull);
	friend pySpParVec EWiseApply(const pySpParVecObj2& a, const pySpParVecObj2& b, op::BinaryPredicateObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, Obj2 ANull, Obj2 BNull);
	friend pySpParVec EWiseApply(const pySpParVecObj2& a, const pySpParVec&     b, op::BinaryPredicateObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, Obj2 ANull, double BNull);
};

pySpParVec EWiseApply(const pySpParVec& a, const pySpParVec&     b, op::BinaryFunction   * op, bool allowANulls = true, bool allowBNulls = true);

pySpParVec EWiseApply(const pySpParVec& a, const pySpParVecObj1& b, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp = NULL, bool allowANulls = true, bool allowBNulls = true, double ANull = 0, Obj1 BNull = Obj1());
pySpParVec EWiseApply(const pySpParVec& a, const pySpParVecObj2& b, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp = NULL, bool allowANulls = true, bool allowBNulls = true, double ANull = 0, Obj2 BNull = Obj2());
pySpParVec EWiseApply(const pySpParVec& a, const pySpParVec&     b, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp = NULL, bool allowANulls = true, bool allowBNulls = true, double ANull = 0, double BNull = 0);

pySpParVec EWiseApply(const pySpParVec& a, const pySpParVecObj1& b, op::BinaryPredicateObj* op, op::BinaryPredicateObj* doOp = NULL, bool allowANulls = true, bool allowBNulls = true, double ANull = 0, Obj1 BNull = Obj1());
pySpParVec EWiseApply(const pySpParVec& a, const pySpParVecObj2& b, op::BinaryPredicateObj* op, op::BinaryPredicateObj* doOp = NULL, bool allowANulls = true, bool allowBNulls = true, double ANull = 0, Obj2 BNull = Obj2());
pySpParVec EWiseApply(const pySpParVec& a, const pySpParVec&     b, op::BinaryPredicateObj* op, op::BinaryPredicateObj* doOp = NULL, bool allowANulls = true, bool allowBNulls = true, double ANull = 0, double BNull = 0);

// Obj predicates:	
pySpParVec EWiseApply(const pySpParVecObj1& a, const pySpParVecObj1& b, op::BinaryPredicateObj* op, op::BinaryPredicateObj* doOp = NULL, bool allowANulls = true, bool allowBNulls = true, Obj1 ANull = Obj1(), Obj1 BNull = Obj1());
pySpParVec EWiseApply(const pySpParVecObj1& a, const pySpParVecObj2& b, op::BinaryPredicateObj* op, op::BinaryPredicateObj* doOp = NULL, bool allowANulls = true, bool allowBNulls = true, Obj1 ANull = Obj1(), Obj2 BNull = Obj2());
pySpParVec EWiseApply(const pySpParVecObj1& a, const pySpParVec&     b, op::BinaryPredicateObj* op, op::BinaryPredicateObj* doOp = NULL, bool allowANulls = true, bool allowBNulls = true, Obj1 ANull = Obj1(), double BNull = 0);

pySpParVec EWiseApply(const pySpParVecObj2& a, const pySpParVecObj1& b, op::BinaryPredicateObj* op, op::BinaryPredicateObj* doOp = NULL, bool allowANulls = true, bool allowBNulls = true, Obj2 ANull = Obj2(), Obj1 BNull = Obj1());
pySpParVec EWiseApply(const pySpParVecObj2& a, const pySpParVecObj2& b, op::BinaryPredicateObj* op, op::BinaryPredicateObj* doOp = NULL, bool allowANulls = true, bool allowBNulls = true, Obj2 ANull = Obj2(), Obj2 BNull = Obj2());
pySpParVec EWiseApply(const pySpParVecObj2& a, const pySpParVec&     b, op::BinaryPredicateObj* op, op::BinaryPredicateObj* doOp = NULL, bool allowANulls = true, bool allowBNulls = true, Obj2 ANull = Obj2(), double BNull = 0);

//      EWiseMult has 2 flavors:
//      - if Exclude is false, will do element-wise multiplication
//      - if Exclude is true, will remove from the result vector all elements
//          whose corresponding element of the second vector is "nonzero"
//          (i.e., not equal to the sparse vector's identity value)  '

//pySpParVec EWiseMult(const pySpParVec& a, const pySpParVec& b, bool exclude);
pySpParVec EWiseMult(const pySpParVec& a, const pyDenseParVec& b, bool exclude, double zero);
void EWiseMult_inplacefirst(pySpParVec& a, const pyDenseParVec& b, bool exclude, double zero);


//INTERFACE_INCLUDE_END

#endif
