#ifndef PY_SP_PAR_VEC_OBJ_H
#define PY_SP_PAR_VEC_OBJ_H

#include "pyCombBLAS.h"

//INTERFACE_INCLUDE_BEGIN
class pySpParVecObj {
//INTERFACE_INCLUDE_END
public:
	typedef int64_t INDEXTYPE;
	typedef FullyDistSpVec<INDEXTYPE, VERTEXTYPE> VectType;
	VectType v;
	
protected:

	//friend class pySpParMat;
	//friend class pySpParMatBool;
	//friend class pyDenseParVec;
	
	friend pySpParVecObj EWiseMult(const pySpParVecObj& a, const pySpParVecObj& b, bool exclude);
	//friend pySpParVecObj EWiseMult(const pySpParVecObj& a, const pyDenseParVec& b, bool exclude, VERTEXTYPE zero);
	//friend void EWiseMult_inplacefirst(pySpParVecObj& a, const pyDenseParVec& b, bool exclude, VERTEXTYPE zero);

	pySpParVecObj(); // used for initializing temporaries to be returned
	pySpParVecObj(VectType other);

/////////////// everything below this appears in python interface:
//INTERFACE_INCLUDE_BEGIN
public:
	pySpParVecObj(int64_t length);
	
	//pyDenseParVec dense() const;

public:
	int64_t getnee() const;
	//int64_t getnnz() const;
	int64_t __len__() const;
	int64_t len() const;

	//pySpParVecObj operator+(const pySpParVecObj& other);
	//pySpParVecObj operator-(const pySpParVecObj& other);
	//pySpParVecObj operator+(const pyDenseParVec& other);
	//pySpParVecObj operator-(const pyDenseParVec& other);

	//pySpParVecObj& operator+=(const pySpParVecObj& other);
	//pySpParVecObj& operator-=(const pySpParVecObj& other);
	//pySpParVecObj& operator+=(const pyDenseParVec& other);
	//pySpParVecObj& operator-=(const pyDenseParVec& other);
	pySpParVecObj copy();

public:	
	bool any() const; // any nonzeros
	bool all() const; // all nonzeros
	
	int64_t intersectSize(const pySpParVecObj& other);
	
	void printall();
	
public:	
	void load(const char* filename);

public:
	// The functions commented out here presently do not exist in CombBLAS
	int64_t Count(op::UnaryFunctionObj* op);
	//pySpParVecObj Find(op::UnaryFunctionObj* op);
	//pyDenseParVec FindInds(op::UnaryFunctionObj* op);
	void Apply(op::UnaryFunctionObj* op);
	//void ApplyMasked(op::UnaryFunctionObj* op, const pySpParVecObj& mask);

	//pyDenseParVec SubsRef(const pyDenseParVec& ri);
	
	//VERTEXTYPE Reduce(op::BinaryFunction* f, op::UnaryFunctionObj* uf = NULL);
	
	pySpParVecObj Sort(); // Does an in-place sort and returns the permutation used in the sort.
	//pyDenseParVec TopK(int64_t k); // Returns a vector of the k largest elements.
	
	void setNumToInd();

public:
	//static pySpParVecObj zeros(int64_t howmany);
	//static pySpParVecObj range(int64_t howmany, int64_t start);
	
public:
	// Functions from PyCombBLAS
	pySpParVecObj abs();
	//void __delitem__(const pyDenseParVec& key);
	void __delitem__(int64_t key);
	
	VERTEXTYPE __getitem__(int64_t key);
	//VERTEXTYPE __getitem__(VERTEXTYPE  key);
	//pyDenseParVec __getitem__(const pyDenseParVec& key);
	
	void __setitem__(int64_t key, const VERTEXTYPE *value);
	//void __setitem__(VERTEXTYPE  key, VERTEXTYPE value);
	//void __setitem__(const pyDenseParVec& key, const pyDenseParVec& value);
	//void __setitem__(const pyDenseParVec& key, int64_t value);
	void __setitem__(const char* key, const VERTEXTYPE *value);	
	
	char* __repr__();

};

//      EWiseMult has 2 flavors:
//      - if Exclude is false, will do element-wise multiplication
//      - if Exclude is true, will remove from the result vector all elements
//          whose corresponding element of the second vector is "nonzero"
//          (i.e., not equal to the sparse vector's identity value)  '


//pySpParVecObj EWiseMult(const pySpParVecObj& a, const pyDenseParVec& b, bool exclude, VERTEXTYPE zero);
//void EWiseMult_inplacefirst(pySpParVecObj& a, const pyDenseParVec& b, bool exclude, VERTEXTYPE zero);


//INTERFACE_INCLUDE_END

#endif
