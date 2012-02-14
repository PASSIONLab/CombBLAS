#ifndef PY_SP_PAR_VEC_Obj2_H
#define PY_SP_PAR_VEC_Obj2_H

#include "pyCombBLAS.h"

//INTERFACE_INCLUDE_BEGIN
class pySpParVecObj2 {
//INTERFACE_INCLUDE_END
public:
	typedef int64_t INDEXTYPE;
	typedef FullyDistSpVec<INDEXTYPE, Obj2> VectType;
	VectType v;
	
protected:

	friend class pySpParMat;
	friend class pySpParMatObj2;
	friend class pySpParMatObj1;
	friend class pySpParMatBool;
	friend class pyDenseParVecObj2;
	friend class pyDenseParVecObj1;
	
	friend pySpParVecObj2 EWiseMult(const pySpParVecObj2& a, const pySpParVecObj2& b, bool exclude);
	//friend pySpParVecObj2 EWiseMult(const pySpParVecObj2& a, const pyDenseParVec& b, bool exclude, Obj2 zero);
	//friend void EWiseMult_inplacefirst(pySpParVecObj2& a, const pyDenseParVec& b, bool exclude, Obj2 zero);

	pySpParVecObj2(); // used for initializing temporaries to be returned
public:
	pySpParVecObj2(VectType other);

/////////////// everything below this appears in python interface:
//INTERFACE_INCLUDE_BEGIN
public:
	pySpParVecObj2(int64_t length);
	
	pyDenseParVecObj2 dense() const;

public:
	int64_t getnee() const;
	//int64_t getnnz() const;
	int64_t __len__() const;
	int64_t len() const;

	//pySpParVecObj2 operator+(const pySpParVecObj2& other);
	//pySpParVecObj2 operator-(const pySpParVecObj2& other);
	//pySpParVecObj2 operator+(const pyDenseParVec& other);
	//pySpParVecObj2 operator-(const pyDenseParVec& other);

	//pySpParVecObj2& operator+=(const pySpParVecObj2& other);
	//pySpParVecObj2& operator-=(const pySpParVecObj2& other);
	//pySpParVecObj2& operator+=(const pyDenseParVec& other);
	//pySpParVecObj2& operator-=(const pyDenseParVec& other);
	pySpParVecObj2 copy();

public:	
	bool any() const; // any nonzeros
	bool all() const; // all nonzeros
	
	int64_t intersectSize(const pySpParVecObj2& other);
	
	void printall();
	
public:	
	void load(const char* filename);
	void save(const char* filename);

public:
	// The functions commented out here presently do not exist in CombBLAS
	int64_t Count(op::UnaryPredicateObj* op);
	//pySpParVecObj2 Find(op::UnaryFunctionObj* op);
	//pyDenseParVec FindInds(op::UnaryFunctionObj* op);
	void Apply(op::UnaryFunctionObj* op);
	void ApplyInd(op::BinaryFunctionObj* op);
	//void ApplyMasked(op::UnaryFunctionObj* op, const pySpParVecObj2& mask);

	pyDenseParVecObj2 SubsRef(const pyDenseParVec& ri);
	
	Obj2 Reduce(op::BinaryFunctionObj* f, op::UnaryFunctionObj* uf, Obj2 *init);
	double Reduce(op::BinaryFunctionObj* f, op::UnaryFunctionObj* uf, double init);
	
	pySpParVec Sort(); // Does an in-place sort and returns the permutation used in the sort.
	pyDenseParVecObj2 TopK(int64_t k); // Returns a vector of the k largest elements.
	

public:
	//static pySpParVecObj2 zeros(int64_t howmany);
	//static pySpParVecObj2 range(int64_t howmany, int64_t start);
	
public:
	//void __delitem__(const pyDenseParVec& key);
	void __delitem__(int64_t key);
	
	Obj2 __getitem__(int64_t key);
	Obj2 __getitem__(double key);
	pyDenseParVecObj2 __getitem__(const pyDenseParVec& key);
	
	void __setitem__(int64_t key, const Obj2 *value);
	void __setitem__(double key, const Obj2 *value);
	//void __setitem__(const pyDenseParVec& key, const pyDenseParVecObj2& value);
	//void __setitem__(const pyDenseParVec& key, int64_t value);
	void __setitem__(const char* key, const Obj2 *value);	
	
	char* __repr__();
};

pySpParVecObj2 EWiseApply(const pySpParVecObj2& a, const pySpParVecObj2& b, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, Obj2 *ANull, Obj2 *BNull, bool allowIntersect);
pySpParVecObj2 EWiseApply(const pySpParVecObj2& a, const pySpParVecObj1& b, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, Obj2 *ANull, Obj1 *BNull, bool allowIntersect);
pySpParVecObj2 EWiseApply(const pySpParVecObj2& a, const pySpParVec&     b, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, Obj2 *ANull, double BNull, bool allowIntersect);
pySpParVec EWiseApply(const pySpParVecObj2& a, const pySpParVecObj2& b, op::BinaryPredicateObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, Obj2 *ANull, Obj2 *BNull, bool allowIntersect);
pySpParVec EWiseApply(const pySpParVecObj2& a, const pySpParVecObj1& b, op::BinaryPredicateObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, Obj2 *ANull, Obj1 *BNull, bool allowIntersect);
pySpParVec EWiseApply(const pySpParVecObj2& a, const pySpParVec&     b, op::BinaryPredicateObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, Obj2 *ANull, double BNull, bool allowIntersect);

/////////// with Dense

pySpParVecObj2 EWiseApply(const pySpParVecObj2& a, const pyDenseParVec& b, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, Obj2 *ANull);
pySpParVecObj2 EWiseApply(const pySpParVecObj2& a, const pyDenseParVecObj2& b, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, Obj2 *ANull);
pySpParVecObj2 EWiseApply(const pySpParVecObj2& a, const pyDenseParVecObj1& b, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, Obj2 *ANull);
pySpParVec EWiseApply(const pySpParVecObj2& a, const pyDenseParVec& b, op::BinaryPredicateObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, Obj2 *ANull);
pySpParVec EWiseApply(const pySpParVecObj2& a, const pyDenseParVecObj2& b, op::BinaryPredicateObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, Obj2 *ANull);
pySpParVec EWiseApply(const pySpParVecObj2& a, const pyDenseParVecObj1& b, op::BinaryPredicateObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, Obj2 *ANull);

//      EWiseMult has 2 flavors:
//      - if Exclude is false, will do element-wise multiplication
//      - if Exclude is true, will remove from the result vector all elements
//          whose corresponding element of the second vector is "nonzero"
//          (i.e., not equal to the sparse vector's identity value)  '

//pySpParVecObj2 EWiseMult(const pySpParVecObj2& a, const pyDenseParVec& b, bool exclude, Obj2 zero);
//void EWiseMult_inplacefirst(pySpParVecObj2& a, const pyDenseParVec& b, bool exclude, Obj2 zero);


//INTERFACE_INCLUDE_END

#endif
