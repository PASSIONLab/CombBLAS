#ifndef PY_SP_PAR_VEC_Obj1_H
#define PY_SP_PAR_VEC_Obj1_H

#include "pyCombBLAS.h"

//INTERFACE_INCLUDE_BEGIN
class pySpParVecObj1 {
//INTERFACE_INCLUDE_END
public:
	typedef int64_t INDEXTYPE;
	typedef FullyDistSpVec<INDEXTYPE, Obj1> VectType;
	VectType v;
	
protected:

	//friend class pySpParMat;
	//friend class pySpParMatBool;
	friend class pyDenseParVecObj1;
	friend class pyDenseParVecObj2;
	
	friend pySpParVecObj1 EWiseMult(const pySpParVecObj1& a, const pySpParVecObj1& b, bool exclude);
	//friend pySpParVecObj1 EWiseMult(const pySpParVecObj1& a, const pyDenseParVec& b, bool exclude, Obj1 zero);
	//friend void EWiseMult_inplacefirst(pySpParVecObj1& a, const pyDenseParVec& b, bool exclude, Obj1 zero);

	pySpParVecObj1(); // used for initializing temporaries to be returned
	pySpParVecObj1(VectType other);

/////////////// everything below this appears in python interface:
//INTERFACE_INCLUDE_BEGIN
public:
	pySpParVecObj1(int64_t length);
	
	pyDenseParVecObj1 dense() const;

public:
	int64_t getnee() const;
	//int64_t getnnz() const;
	int64_t __len__() const;
	int64_t len() const;

	//pySpParVecObj1 operator+(const pySpParVecObj1& other);
	//pySpParVecObj1 operator-(const pySpParVecObj1& other);
	//pySpParVecObj1 operator+(const pyDenseParVec& other);
	//pySpParVecObj1 operator-(const pyDenseParVec& other);

	//pySpParVecObj1& operator+=(const pySpParVecObj1& other);
	//pySpParVecObj1& operator-=(const pySpParVecObj1& other);
	//pySpParVecObj1& operator+=(const pyDenseParVec& other);
	//pySpParVecObj1& operator-=(const pyDenseParVec& other);
	pySpParVecObj1 copy();

public:	
	bool any() const; // any nonzeros
	bool all() const; // all nonzeros
	
	int64_t intersectSize(const pySpParVecObj1& other);
	
	void printall();
	
public:	
	void load(const char* filename);

public:
	// The functions commented out here presently do not exist in CombBLAS
	int64_t Count(op::UnaryPredicateObj* op);
	//pySpParVecObj1 Find(op::UnaryFunctionObj* op);
	//pyDenseParVec FindInds(op::UnaryFunctionObj* op);
	void Apply(op::UnaryFunctionObj* op);
	//void ApplyMasked(op::UnaryFunctionObj* op, const pySpParVecObj1& mask);

	pyDenseParVecObj1 SubsRef(const pyDenseParVec& ri);
	
	Obj1 Reduce(op::BinaryFunctionObj* f, op::UnaryFunctionObj* uf = NULL);
	
	pySpParVec Sort(); // Does an in-place sort and returns the permutation used in the sort.
	pyDenseParVecObj1 TopK(int64_t k); // Returns a vector of the k largest elements.
	
	void setNumToInd();

public:
	//static pySpParVecObj1 zeros(int64_t howmany);
	//static pySpParVecObj1 range(int64_t howmany, int64_t start);
	
public:
	//void __delitem__(const pyDenseParVec& key);
	void __delitem__(int64_t key);
	
	Obj1 __getitem__(int64_t key);
	Obj1 __getitem__(double key);
	pyDenseParVecObj1 __getitem__(const pyDenseParVec& key);
	
	void __setitem__(int64_t key, const Obj1 *value);
	void __setitem__(double key, const Obj1 *value);
	//void __setitem__(const pyDenseParVec& key, const pyDenseParVecObj1& value);
	//void __setitem__(const pyDenseParVec& key, int64_t value);
	void __setitem__(const char* key, const Obj1 *value);	
	
	char* __repr__();
	
	friend pySpParVecObj1 EWiseApply(const pySpParVecObj1& a, const pySpParVecObj1& b, op::BinaryFunctionObj* op, bool allowANulls, bool allowBNulls);
	friend pySpParVecObj1 EWiseApply(const pySpParVecObj1& a, const pySpParVecObj2& b, op::BinaryFunctionObj* op, bool allowANulls, bool allowBNulls);
	friend pySpParVecObj1 EWiseApply(const pySpParVecObj1& a, const pySpParVec&     b, op::BinaryFunctionObj* op, bool allowANulls, bool allowBNulls);
};

pySpParVecObj1 EWiseApply(const pySpParVecObj1& a, const pySpParVecObj1& b, op::BinaryFunctionObj* op, bool allowANulls = true, bool allowBNulls = true);
pySpParVecObj1 EWiseApply(const pySpParVecObj1& a, const pySpParVecObj2& b, op::BinaryFunctionObj* op, bool allowANulls = true, bool allowBNulls = true);
pySpParVecObj1 EWiseApply(const pySpParVecObj1& a, const pySpParVec&     b, op::BinaryFunctionObj* op, bool allowANulls = true, bool allowBNulls = true);


//      EWiseMult has 2 flavors:
//      - if Exclude is false, will do element-wise multiplication
//      - if Exclude is true, will remove from the result vector all elements
//          whose corresponding element of the second vector is "nonzero"
//          (i.e., not equal to the sparse vector's identity value)  '

//pySpParVecObj1 EWiseMult(const pySpParVecObj1& a, const pyDenseParVec& b, bool exclude, Obj1 zero);
//void EWiseMult_inplacefirst(pySpParVecObj1& a, const pyDenseParVec& b, bool exclude, Obj1 zero);


//INTERFACE_INCLUDE_END

#endif
