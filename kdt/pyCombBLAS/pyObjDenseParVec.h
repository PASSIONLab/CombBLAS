#ifndef PY_OBJ_DENSE_PAR_VEC_H
#define PY_OBJ_DENSE_PAR_VEC_H

#include "pyCombBLAS.h"

//INTERFACE_INCLUDE_BEGIN
class pyObjDenseParVec {
//INTERFACE_INCLUDE_END
	typedef int64_t INDEXTYPE;
	typedef FullyDistVec<INDEXTYPE, PyObject*> VectType;
	
	public:
	VectType v;

protected:
	
	friend class pySpParVec;
	friend class pySpParMat;
	friend class pySpParMatBool;
	friend pySpParVec EWiseMult(const pySpParVec& a, const pyObjDenseParVec& b, bool exclude, double zero);
	friend void EWiseMult_inplacefirst(pySpParVec& a, const pyObjDenseParVec& b, bool exclude, double zero);

	pyObjDenseParVec();
	pyObjDenseParVec(VectType other);
/////////////// everything below this appears in python interface:
//INTERFACE_INCLUDE_BEGIN
public:
	pyObjDenseParVec(int64_t size, PyObject* init);
	pyObjDenseParVec(int64_t size, PyObject* init, PyObject* zero);
	
	//pySpParVec sparse() const;
	//pySpParVec sparse(PyObject* zero) const;
	
public:
	int64_t len() const;
	int64_t __len__() const;
	/*
	void add(const pyObjDenseParVec& other);
	void add(const pySpParVec& other);
	pyObjDenseParVec& operator+=(const pyObjDenseParVec & rhs);
	pyObjDenseParVec& operator-=(const pyObjDenseParVec & rhs);
	pyObjDenseParVec& operator+=(const pySpParVec & rhs);
	pyObjDenseParVec& operator-=(const pySpParVec & rhs);
	pyObjDenseParVec& operator*=(const pyObjDenseParVec& rhs);
	pyObjDenseParVec& operator*=(const pySpParVec& rhs);
	
	pyObjDenseParVec operator+(const pyObjDenseParVec & rhs);
	pyObjDenseParVec operator-(const pyObjDenseParVec & rhs);
	pyObjDenseParVec operator+(const pySpParVec & rhs);
	pyObjDenseParVec operator-(const pySpParVec & rhs);
	pyObjDenseParVec operator*(const pyObjDenseParVec& rhs);
	pyObjDenseParVec operator*(const pySpParVec& rhs);
	
	pyObjDenseParVec operator==(const pyObjDenseParVec& other);
	pyObjDenseParVec operator!=(const pyObjDenseParVec& other);
	*/
	pyObjDenseParVec copy();
	
	//pyObjDenseParVec SubsRef(const pyObjDenseParVec& ri);

	//void RandPerm(); // Randomly permutes the vector
	//pyObjDenseParVec Sort(); // Does an in-place sort and returns the permutation used in the sort.
	//pyObjDenseParVec TopK(int64_t k); // Returns a vector of the k largest elements.

	void printall();
	
public:
	
	int64_t getnee() const;
	//int64_t getnnz() const;
	//int64_t getnz() const;
	//bool any() const;
	
public:	
	//void load(const char* filename);
	
public:
	//int64_t Count(op::UnaryFunction* op);
	//double Reduce(op::BinaryFunction* f, op::UnaryFunction* uf = NULL);
	//pySpParVec Find(op::UnaryFunction* op);
	//pySpParVec __getitem__(op::UnaryFunction* op);
	//pyDenseParVec FindInds(op::UnaryFunction* op);
	void Apply(op::ObjUnaryFunction* op);
	//void ApplyMasked(op::UnaryFunction* op, const pySpParVec& mask);
	//void EWiseApply(const pyObjDenseParVec& other, op::BinaryFunction *f);
	//void EWiseApply(const pySpParVec& other, op::BinaryFunction *f, bool doNulls = false, PyObject* nullValue = 0);

public:
	//static pyObjDenseParVec range(int64_t howmany, int64_t start);
	
public:
	// Functions from PyCombBLAS
	/*
	pyObjDenseParVec abs();
	
	pyObjDenseParVec& operator+=(double value);
	pyObjDenseParVec operator+(double value);
	pyObjDenseParVec& operator-=(double value);
	pyObjDenseParVec operator-(double value);
	
	pyObjDenseParVec __and__(const pyObjDenseParVec& other);
	*/
	
	PyObject* __getitem__(int64_t key);
	PyObject* __getitem__(double  key);
	//pyObjDenseParVec __getitem__(const pyObjDenseParVec& key);

	void __setitem__(int64_t key, PyObject* value);
	void __setitem__(double  key, PyObject* value);
	//void __setitem__(const pySpParVec& key, const pySpParVec& value);
	//void __setitem__(const pySpParVec& key, double value);
};
//INTERFACE_INCLUDE_END

#endif
