#ifndef PYOPERATIONOBJ_H
#define PYOPERATIONOBJ_H

#include "pyCombBLAS.h"
#include <functional>
#include <iostream>
#include <math.h>

//INTERFACE_INCLUDE_BEGIN
namespace op {

class UnaryPredicateObj {
//INTERFACE_INCLUDE_END
	public:
	PyObject *callback;
	UnaryPredicateObj(PyObject *pyfunc): callback(pyfunc) { Py_INCREF(callback); }

	public:
	~UnaryPredicateObj() { Py_XDECREF(callback); }

	template <class T>
	bool call(const T& x, swig_type_info *typeinfo) const;
	
//INTERFACE_INCLUDE_BEGIN
	bool operator()(const Obj2& x) const { return call(x, SWIGTYPE_p_Obj2); }
	bool operator()(const Obj1& x) const { return call(x, SWIGTYPE_p_Obj1); }

	protected:
	UnaryPredicateObj() { // should never be called
		printf("UnaryPredicateObj()!!!\n");
		callback = NULL;
	}
};

class UnaryFunctionObj {
//INTERFACE_INCLUDE_END
	public:
	PyObject *callback;
	UnaryFunctionObj(PyObject *pyfunc): callback(pyfunc) { Py_INCREF(callback); }

	public:
	~UnaryFunctionObj() { Py_XDECREF(callback); }
	
	template <class T>
	T call(const T& x, swig_type_info *typeinfo) const;

//INTERFACE_INCLUDE_BEGIN
	Obj2 operator()(const Obj2& x) const { return call(x, SWIGTYPE_p_Obj2); }
	Obj1 operator()(const Obj1& x) const { return call(x, SWIGTYPE_p_Obj1); }
	
	protected:
	UnaryFunctionObj() { // should never be called
		printf("UnaryFunctionObj()!!!\n");
		callback = NULL;
	}
};
//INTERFACE_INCLUDE_END

template <class T>
T UnaryFunctionObj::call(const T& x, swig_type_info *typeinfo) const
{
	PyObject *resultPy;
	T *pret;	

	T tempObj = x;
	PyObject *tempSwigObj = SWIG_NewPointerObj(&tempObj, typeinfo, 0);
	PyObject *vertexArgList = Py_BuildValue("(O)", tempSwigObj);
	
	resultPy = PyEval_CallObject(callback,vertexArgList);  

	if (resultPy && SWIG_IsOK(SWIG_ConvertPtr(resultPy, (void**)&pret, typeinfo,  0  | 0)) && pret != NULL) {
		T ret = T(*pret);
		Py_XDECREF(tempSwigObj);
		Py_XDECREF(vertexArgList);
		Py_XDECREF(resultPy);
		return ret;
	} else
	{
		Py_XDECREF(tempSwigObj);
		Py_XDECREF(vertexArgList);
		cerr << "UnaryFunctionObj::operator() FAILED!" << endl;
		return T();
	}
}

// This function is identical to UnaryFunctionObj::call() except that it returns a boolean instead
// of an object. Please keep the actual calling method the same if you make any changes.
template <class T>
bool UnaryPredicateObj::call(const T& x, swig_type_info *typeinfo) const
{
	PyObject *resultPy;
	T *pret;	

	T tempObj = x;
	PyObject *tempSwigObj = SWIG_NewPointerObj(&tempObj, typeinfo, 0);
	PyObject *vertexArgList = Py_BuildValue("(O)", tempSwigObj);
	
	resultPy = PyEval_CallObject(callback,vertexArgList);  

	if (resultPy) {
		bool ret = PyObject_IsTrue(resultPy);
		Py_XDECREF(tempSwigObj);
		Py_XDECREF(vertexArgList);
		Py_XDECREF(resultPy);
		return ret;
	} else
	{
		Py_XDECREF(tempSwigObj);
		Py_XDECREF(vertexArgList);
		cerr << "UnaryFunctionObj::operator() FAILED!" << endl;
		return false;
	}
}
//INTERFACE_INCLUDE_BEGIN

UnaryFunctionObj unaryObj(PyObject *pyfunc);
UnaryPredicateObj unaryObjPred(PyObject *pyfunc);

#if 0
//INTERFACE_INCLUDE_BEGIN
class BinaryFunctionE {
//INTERFACE_INCLUDE_END
	public:
	ConcreteBinaryFunction<Obj2>* op;
	
	BinaryFunctionE(ConcreteBinaryFunction<Obj2>* opin, bool as, bool com): op(opin), commutable(com), associative(as) {  }

	// for creating an MPI_Op that can be used with MPI Reduce
	static void apply(void * invec, void * inoutvec, int * len, MPI_Datatype *datatype);
	static BinaryFunctionE* currentlyApplied;
	static MPI_Op staticMPIop;
	
	MPI_Op* getMPIOp();
	void releaseMPIOp();
	
//INTERFACE_INCLUDE_BEGIN
	protected:
	BinaryFunctionE(): op(NULL), commutable(false), associative(false) {}
	public:
	~BinaryFunctionE() { /*delete op; op = NULL;*/ }
	
	bool commutable;
	bool associative;
	
	Obj2 operator()(const Obj2& x, const Obj2& y) const
	{
		return (*op)(x, y);
	}

};
class BinaryFunctionV {
//INTERFACE_INCLUDE_END
	public:
	ConcreteBinaryFunction<Obj1>* op;
	
	BinaryFunctionV(ConcreteBinaryFunction<Obj1>* opin, bool as, bool com): op(opin), commutable(com), associative(as) {  }

	// for creating an MPI_Op that can be used with MPI Reduce
	static void apply(void * invec, void * inoutvec, int * len, MPI_Datatype *datatype);
	static BinaryFunctionV* currentlyApplied;
	static MPI_Op staticMPIop;
	
	MPI_Op* getMPIOp();
	void releaseMPIOp();
	
//INTERFACE_INCLUDE_BEGIN
	protected:
	BinaryFunctionV(): op(NULL), commutable(false), associative(false) {}
	public:
	~BinaryFunctionV() { /*delete op; op = NULL;*/ }
	
	bool commutable;
	bool associative;
	
	Obj1 operator()(const Obj1& x, const Obj1& y) const
	{
		return (*op)(x, y);
	}

};

BinaryFunction binaryE(PyObject *pyfunc);
BinaryFunction binaryV(PyObject *pyfunc);

#endif
/*
class Semiring {
//INTERFACE_INCLUDE_END
	public:
	// CUSTOM is a semiring with Python-defined methods
	// The others are pre-implemented in C++ for speed.
	typedef enum {CUSTOM, NONE, TIMESPLUS, PLUSMIN, SECONDMAX} SRingType;

	protected:
	SRingType type;
	
	PyObject *pyfunc_add;
	PyObject *pyfunc_multiply;
	
	BinaryFunction *binfunc_add;
	
	public:
	// CombBLAS' template mechanism means we have to compile in only one C++ semiring.
	// So to support different Python semirings, we have to switch them in.
	void enableSemiring();
	void disableSemiring();
	
	public:
	Semiring(SRingType t): type(t), pyfunc_add(NULL), pyfunc_multiply(NULL), binfunc_add(NULL) {
		//if (t == CUSTOM)
			// scream bloody murder
	}
	
	SRingType getType() { return type; }
	
//INTERFACE_INCLUDE_BEGIN
	protected:
	Semiring(): type(NONE), pyfunc_add(NULL), pyfunc_multiply(NULL), binfunc_add(NULL) {}
	public:
	Semiring(PyObject *add, PyObject *multiply);
	~Semiring();
	
	MPI_Op mpi_op()
	{
		return *(binfunc_add->getMPIOp());
	}
	
	doubleint add(const doubleint & arg1, const doubleint & arg2);	
	doubleint multiply(const doubleint & arg1, const doubleint & arg2);
	void axpy(doubleint a, const doubleint & x, doubleint & y);

};
//INTERFACE_INCLUDE_END

template <class T1, class T2>
struct SemiringTemplArg
{
	static Semiring *currentlyApplied;
	
	typedef typename promote_trait<T1,T2>::T_promote T_promote;
	static T_promote id() { return T_promote();}
	static MPI_Op mpi_op()
	{
		return currentlyApplied->mpi_op();
	}
	
	static T_promote add(const T_promote & arg1, const T_promote & arg2)
	{
		return currentlyApplied->add(arg1, arg2);
	}
	
	static T_promote multiply(const T1 & arg1, const T2 & arg2)
	{
		return currentlyApplied->multiply(arg1, arg2);
	}
	
	static void axpy(T1 a, const T2 & x, T_promote & y)
	{
		currentlyApplied->axpy(a, x, y);
	}
};

//INTERFACE_INCLUDE_BEGIN
Semiring TimesPlusSemiring();
//Semiring MinPlusSemiring();
Semiring SecondMaxSemiring();
*/
} // namespace op


//INTERFACE_INCLUDE_END

// modeled after CombBLAS/Operations.h
// This call is only safe when between BinaryFunction.getMPIOp() and releaseMPIOp() calls.
// That should be safe enough, because this is only called from inside CombBLAS reduce operations,
// which only get called between getMPIOp() and releaseMPIOp().
#if 0
template<> struct MPIOp< op::BinaryFunctionE, Obj2 > {  static MPI_Op op() { return op::BinaryFunctionE::staticMPIop; } };
template<> struct MPIOp< op::BinaryFunctionV, Obj1 > {  static MPI_Op op() { return op::BinaryFunctionV::staticMPIop; } };
#endif

#endif
