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

	template <class T>
	bool call(const T& x) const;

	bool callD(const double& x) const;
	
//INTERFACE_INCLUDE_BEGIN
	bool operator()(const Obj2& x) const { return call(x); }
	bool operator()(const Obj1& x) const { return call(x); }
	bool operator()(const double& x) const { return callD(x); }

	protected:
	UnaryPredicateObj() { // should never be called
		printf("UnaryPredicateObj()!!!\n");
		callback = NULL;
	}

	public:
	~UnaryPredicateObj() { /*Py_XDECREF(callback);*/ }
};

class UnaryFunctionObj {
//INTERFACE_INCLUDE_END
	public:
	PyObject *callback;
	UnaryFunctionObj(PyObject *pyfunc): callback(pyfunc) { Py_INCREF(callback); }

	template <class T>
	T call(const T& x) const;

	double callD(const double& x) const;

//INTERFACE_INCLUDE_BEGIN
	Obj2 operator()(const Obj2& x) const { return call(x); }
	Obj1 operator()(const Obj1& x) const { return call(x); }
	double operator()(const double& x) const { return callD(x); }
	
	protected:
	UnaryFunctionObj() { // should never be called
		printf("UnaryFunctionObj()!!!\n");
		callback = NULL;
	}

	public:
	~UnaryFunctionObj() { /*Py_XDECREF(callback);*/ }
};

//INTERFACE_INCLUDE_END
template <class T>
T UnaryFunctionObj::call(const T& x) const
{
	PyObject *resultPy;
	T *pret;	

	T tempObj = x;
	PyObject *tempSwigObj = SWIG_NewPointerObj(&tempObj, T::SwigTypeInfo, 0);
	PyObject *vertexArgList = Py_BuildValue("(O)", tempSwigObj);
	
	resultPy = PyEval_CallObject(callback,vertexArgList);  

	Py_XDECREF(tempSwigObj);
	Py_XDECREF(vertexArgList);
	if (resultPy && SWIG_IsOK(SWIG_ConvertPtr(resultPy, (void**)&pret, T::SwigTypeInfo,  0  | 0)) && pret != NULL) {
		T ret = T(*pret);
		Py_XDECREF(resultPy);
		return ret;
	} else
	{
		cerr << "UnaryFunctionObj::operator() FAILED!" << endl;
		return T();
	}
}

inline double UnaryFunctionObj::callD(const double& x) const
{
	PyObject *vertexArgList = Py_BuildValue("(d)", x);
	PyObject *resultPy = PyEval_CallObject(callback,vertexArgList);  

	Py_XDECREF(vertexArgList);
	if (resultPy) {
		double dres = PyFloat_AsDouble(resultPy);
		Py_XDECREF(resultPy);
		return dres;
	} else
	{
		cerr << "UnaryFunctionObj::operator() FAILED!" << endl;
		return 0;
	}
}

// This function is identical to UnaryFunctionObj::call() except that it returns a boolean instead
// of an object. Please keep the actual calling method the same if you make any changes.
template <class T>
bool UnaryPredicateObj::call(const T& x) const
{
	PyObject *resultPy;

	T tempObj = x;
	PyObject *tempSwigObj = SWIG_NewPointerObj(&tempObj, T::SwigTypeInfo, 0);
	PyObject *vertexArgList = Py_BuildValue("(O)", tempSwigObj);
	
	resultPy = PyEval_CallObject(callback,vertexArgList);  

	Py_XDECREF(tempSwigObj);
	Py_XDECREF(vertexArgList);
	if (resultPy) {
		bool ret = PyObject_IsTrue(resultPy);
		Py_XDECREF(resultPy);
		return ret;
	} else
	{
		cerr << "UnaryFunctionObj::operator() FAILED!" << endl;
		return false;
	}
}

inline bool UnaryPredicateObj::callD(const double& x) const
{
	PyObject *vertexArgList = Py_BuildValue("(d)", x);
	PyObject *resultPy = PyEval_CallObject(callback,vertexArgList);  

	Py_XDECREF(vertexArgList);
	if (resultPy) {
		bool ret = PyObject_IsTrue(resultPy);
		Py_XDECREF(resultPy);
		return ret;
	} else
	{
		cerr << "UnaryFunctionObj::operator() FAILED!" << endl;
		return 0;
	}
}

//INTERFACE_INCLUDE_BEGIN

UnaryFunctionObj unaryObj(PyObject *pyfunc);
UnaryPredicateObj unaryObjPred(PyObject *pyfunc);

//INTERFACE_INCLUDE_BEGIN
class BinaryFunctionObj {
//INTERFACE_INCLUDE_END
	public:
	PyObject *callback;

	BinaryFunctionObj(PyObject *pyfunc, bool as, bool com): callback(pyfunc), commutable(com), associative(as) { Py_INCREF(callback); }

	// for creating an MPI_Op that can be used with MPI Reduce
	static void apply(void * invec, void * inoutvec, int * len, MPI_Datatype *datatype);
	template <class T1, class T2>
	static void applyWorker(T1 * in, T2 * inout, int * len);

	static BinaryFunctionObj* currentlyApplied;
	static MPI_Op staticMPIop;
	
	MPI_Op* getMPIOp();
	void releaseMPIOp();

	template <class RET, class T1, class T2>
	RET call(const T1& x, const T2& y) const;

	template <class T1>
	double callOD_retD(const T1& x, const double& y) const;
	template <class RET, class T1>
	RET callOD_retO(const T1& x, const double& y) const;

	template <class T2>
	double callDO_retD(const double& x, const T2& y) const;
	template <class RET, class T2>
	RET callDO_retO(const double& x, const T2& y) const;

	inline double callDD(const double& x, const double& y) const;
	
//INTERFACE_INCLUDE_BEGIN
	protected:
	BinaryFunctionObj(): callback(NULL), commutable(false), associative(false) {}
	public:
	~BinaryFunctionObj() { /*Py_XDECREF(callback);*/ }
	
	bool commutable;
	bool associative;
	
	Obj1 operator()(const Obj1& x, const Obj1& y) const { return call<Obj1>(x, y); }
	Obj2 operator()(const Obj2& x, const Obj2& y) const { return call<Obj2>(x, y); }
	Obj1 operator()(const Obj1& x, const Obj2& y) const { return call<Obj1>(x, y); }
	Obj2 operator()(const Obj2& x, const Obj1& y) const { return call<Obj2>(x, y); }

	Obj1 operator()(const Obj1& x, const double& y) const { return callOD_retO<Obj1>(x, y); }
	Obj2 operator()(const Obj2& x, const double& y) const { return callOD_retO<Obj2>(x, y); }
	double operator()(const double& x, const Obj1& y) const { return callDO_retD(x, y); }
	double operator()(const double& x, const Obj2& y) const { return callDO_retD(x, y); }

	double operator()(const double& x, const double& y) const { return callDD(x, y); }


	// These are used by the semiring ops. They do the same thing as the operator() above,
	// but their return type matches the 2nd argument instead of the 1st.
	Obj1 rettype2nd_call(const Obj1& x, const Obj1& y) const { return call<Obj1>(x, y); }
	Obj2 rettype2nd_call(const Obj2& x, const Obj2& y) const { return call<Obj2>(x, y); }
	Obj1 rettype2nd_call(const Obj2& x, const Obj1& y) const { return call<Obj1>(x, y); }
	Obj2 rettype2nd_call(const Obj1& x, const Obj2& y) const { return call<Obj2>(x, y); }

	double rettype2nd_call(const Obj1& x, const double& y) const { return callOD_retD(x, y); }
	double rettype2nd_call(const Obj2& x, const double& y) const { return callOD_retD(x, y); }
	Obj1 rettype2nd_call(const double& x, const Obj1& y) const { return callDO_retO<Obj1>(x, y); }
	Obj2 rettype2nd_call(const double& x, const Obj2& y) const { return callDO_retO<Obj2>(x, y); }

	double rettype2nd_call(const double& x, const double& y) const { return callDD(x, y); }

};

//INTERFACE_INCLUDE_END
template <typename RET, typename T1, typename T2>
RET BinaryFunctionObj::call(const T1& x, const T2& y) const
{
	PyObject *resultPy;
	RET *pret;	

	T1 tempObj1 = x;
	T2 tempObj2 = y;
	PyObject *tempSwigObj1 = SWIG_NewPointerObj(&tempObj1, T1::SwigTypeInfo, 0);
	PyObject *tempSwigObj2 = SWIG_NewPointerObj(&tempObj2, T2::SwigTypeInfo, 0);
	PyObject *vertexArgList = Py_BuildValue("(O O)", tempSwigObj1, tempSwigObj2);
	
	resultPy = PyEval_CallObject(callback,vertexArgList);  

	Py_XDECREF(tempSwigObj1);
	Py_XDECREF(tempSwigObj2);
	Py_XDECREF(vertexArgList);
	if (resultPy && SWIG_IsOK(SWIG_ConvertPtr(resultPy, (void**)&pret, RET::SwigTypeInfo,  0  | 0)) && pret != NULL) {
		RET ret = RET(*pret);
		Py_XDECREF(resultPy);
		return ret;
	} else
	{
		Py_XDECREF(resultPy);
		cerr << "BinaryFunctionObj::operator() FAILED (callOO)!" << endl;
		return RET();
	}
}

template <typename T1>
double BinaryFunctionObj::callOD_retD(const T1& x, const double& y) const
{
	PyObject *resultPy;
	double dres = 0;

	T1 tempObj1 = x;
	PyObject *tempSwigObj1 = SWIG_NewPointerObj(&tempObj1, T1::SwigTypeInfo, 0);
	PyObject *vertexArgList = Py_BuildValue("(O d)", tempSwigObj1, y);
	
	resultPy = PyEval_CallObject(callback,vertexArgList);  

	Py_XDECREF(tempSwigObj1);
	Py_XDECREF(vertexArgList);
	if (resultPy) {                                   // If no errors, return double
		dres = PyFloat_AsDouble(resultPy);
		Py_XDECREF(resultPy);
		return dres;
	} else
	{
		Py_XDECREF(resultPy);
		cerr << "BinaryFunctionObj::operator() FAILED (callOD)!" << endl;
		return 0;
	}
}

template <typename RET, typename T1>
RET BinaryFunctionObj::callOD_retO(const T1& x, const double& y) const
{
	PyObject *resultPy;
	RET *pret;	

	T1 tempObj1 = x;
	PyObject *tempSwigObj1 = SWIG_NewPointerObj(&tempObj1, T1::SwigTypeInfo, 0);
	PyObject *vertexArgList = Py_BuildValue("(O d)", tempSwigObj1, y);
	
	resultPy = PyEval_CallObject(callback,vertexArgList);  

	Py_XDECREF(tempSwigObj1);
	Py_XDECREF(vertexArgList);
	if (resultPy && SWIG_IsOK(SWIG_ConvertPtr(resultPy, (void**)&pret, RET::SwigTypeInfo,  0  | 0)) && pret != NULL) {
		RET ret = RET(*pret);
		Py_XDECREF(resultPy);
		return ret;
	} else
	{
		Py_XDECREF(resultPy);
		cerr << "BinaryFunctionObj::operator() FAILED (callOD)!" << endl;
		return RET();
	}
}

template <typename T2>
double BinaryFunctionObj::callDO_retD(const double& x, const T2& y) const
{
	PyObject *resultPy;
	double dres = 0;

	T2 tempObj2 = y;
	PyObject *tempSwigObj2 = SWIG_NewPointerObj(&tempObj2, T2::SwigTypeInfo, 0);
	PyObject *vertexArgList = Py_BuildValue("(d O)", x, tempSwigObj2);
	
	resultPy = PyEval_CallObject(callback,vertexArgList);  

	Py_XDECREF(tempSwigObj2);
	Py_XDECREF(vertexArgList);
	if (resultPy) {                                   // If no errors, return double
		dres = PyFloat_AsDouble(resultPy);
		Py_XDECREF(resultPy);
		return dres;
	} else
	{
		Py_XDECREF(resultPy);
		cerr << "BinaryFunctionObj::operator() FAILED! (callDO)" << endl;
		return 0;
	}
}

template <typename RET, typename T2>
RET BinaryFunctionObj::callDO_retO(const double& x, const T2& y) const
{
	PyObject *resultPy;
	RET *pret;	

	T2 tempObj2 = y;
	PyObject *tempSwigObj2 = SWIG_NewPointerObj(&tempObj2, T2::SwigTypeInfo, 0);
	PyObject *vertexArgList = Py_BuildValue("(d O)", x, tempSwigObj2);
	
	resultPy = PyEval_CallObject(callback,vertexArgList);  

	Py_XDECREF(tempSwigObj2);
	Py_XDECREF(vertexArgList);
	if (resultPy && SWIG_IsOK(SWIG_ConvertPtr(resultPy, (void**)&pret, RET::SwigTypeInfo,  0  | 0)) && pret != NULL) {
		RET ret = RET(*pret);
		Py_XDECREF(resultPy);
		return ret;
	} else
	{
		Py_XDECREF(resultPy);
		cerr << "BinaryFunctionObj::operator() FAILED! (callDO)" << endl;
		return RET();
	}
}

inline double BinaryFunctionObj::callDD(const double& x, const double& y) const
{
	PyObject *arglist;
	PyObject *resultPy;
	double dres = 0;
	
	arglist = Py_BuildValue("(d d)", x, y);    // Build argument list
	resultPy = PyEval_CallObject(callback,arglist);     // Call Python
	Py_DECREF(arglist);                             // Trash arglist
	if (resultPy) {                                   // If no errors, return double
		dres = PyFloat_AsDouble(resultPy);
		Py_XDECREF(resultPy);
		return dres;
	} else
	{
		cerr << "BinaryFunctionObj::operator() FAILED! (callDD)" << endl;
		return 0;
	}
}

template <class T1, class T2>
void BinaryFunctionObj::applyWorker(T1 * in, T2 * inout, int * len)
{
	for (int i = 0; i < *len; i++)
	{
		inout[i] = (*currentlyApplied)(in[i], inout[i]);
	}
}

//INTERFACE_INCLUDE_BEGIN
class BinaryPredicateObj {
//INTERFACE_INCLUDE_END
	public:
	PyObject *callback;
	BinaryPredicateObj(PyObject *pyfunc): callback(pyfunc) { Py_INCREF(callback); }

	template <class T1, class T2>
	bool call(const T1& x, const T2& y) const;

	template <class T1>
	bool callOD(const T1& x, const double& y) const;
	template <class T2>
	bool callDO(const double& x, const T2& y) const;

	inline bool callDD(const double& x, const double& y) const;
	
//INTERFACE_INCLUDE_BEGIN
	bool operator()(const Obj1& x, const Obj1& y) const { return call(x, y); }
	bool operator()(const Obj1& x, const Obj2& y) const { return call(x, y); }
	bool operator()(const Obj2& x, const Obj2& y) const { return call(x, y); }
	bool operator()(const Obj2& x, const Obj1& y) const { return call(x, y); }

	bool operator()(const Obj1& x, const double& y) const { return callOD(x, y); }
	bool operator()(const Obj2& x, const double& y) const { return callOD(x, y); }
	bool operator()(const double& x, const Obj2& y) const { return callDO(x, y); }
	bool operator()(const double& x, const Obj1& y) const { return callDO(x, y); }

	bool operator()(const double& x, const double& y) const { return callDD(x, y); }

	protected:
	BinaryPredicateObj() { // should never be called
		printf("BinaryPredicateObj()!!!\n");
		callback = NULL;
	}

	public:
	~BinaryPredicateObj() { /*Py_XDECREF(callback);*/ }
};

//INTERFACE_INCLUDE_END
template <class T1, class T2>
bool BinaryPredicateObj::call(const T1& x, const T2& y) const
{
	PyObject *resultPy;

	T1 tempObj1 = x;
	T2 tempObj2 = y;
	PyObject *tempSwigObj1 = SWIG_NewPointerObj(&tempObj1, T1::SwigTypeInfo, 0);
	PyObject *tempSwigObj2 = SWIG_NewPointerObj(&tempObj2, T2::SwigTypeInfo, 0);
	PyObject *vertexArgList = Py_BuildValue("(O, O)", tempSwigObj1, tempSwigObj2);
	
	resultPy = PyEval_CallObject(callback,vertexArgList);  

	Py_XDECREF(tempSwigObj1);
	Py_XDECREF(tempSwigObj2);
	Py_XDECREF(vertexArgList);
	if (resultPy) {
		bool ret = PyObject_IsTrue(resultPy);
		Py_XDECREF(resultPy);
		return ret;
	} else
	{
		cerr << "BinaryPredicateObj::operator() FAILED!" << endl;
		return false;
	}
}

template <class T1>
bool BinaryPredicateObj::callOD(const T1& x, const double& y) const
{
	PyObject *resultPy;

	T1 tempObj1 = x;
	PyObject *tempSwigObj1 = SWIG_NewPointerObj(&tempObj1, T1::SwigTypeInfo, 0);
	PyObject *vertexArgList = Py_BuildValue("(O, d)", tempSwigObj1, y);
	
	resultPy = PyEval_CallObject(callback,vertexArgList);  

	Py_XDECREF(tempSwigObj1);
	Py_XDECREF(vertexArgList);
	if (resultPy) {
		bool ret = PyObject_IsTrue(resultPy);
		Py_XDECREF(resultPy);
		return ret;
	} else
	{
		cerr << "BinaryPredicateObj::operator() FAILED!" << endl;
		return false;
	}
}

template <class T2>
bool BinaryPredicateObj::callDO(const double& x, const T2& y) const
{
	PyObject *resultPy;

	T2 tempObj2 = y;
	PyObject *tempSwigObj2 = SWIG_NewPointerObj(&tempObj2, T2::SwigTypeInfo, 0);
	PyObject *vertexArgList = Py_BuildValue("(d, O)", x, tempSwigObj2);
	
	resultPy = PyEval_CallObject(callback,vertexArgList);  

	Py_XDECREF(tempSwigObj2);
	Py_XDECREF(vertexArgList);
	if (resultPy) {
		bool ret = PyObject_IsTrue(resultPy);
		Py_XDECREF(resultPy);
		return ret;
	} else
	{
		cerr << "BinaryPredicateObj::operator() FAILED!" << endl;
		return false;
	}
}

bool BinaryPredicateObj::callDD(const double& x, const double& y) const
{
	PyObject *resultPy;

	PyObject *vertexArgList = Py_BuildValue("(d, d)", x, y);
	
	resultPy = PyEval_CallObject(callback,vertexArgList);  

	Py_XDECREF(vertexArgList);
	if (resultPy) {
		bool ret = PyObject_IsTrue(resultPy);
		Py_XDECREF(resultPy);
		return ret;
	} else
	{
		cerr << "BinaryPredicateObj::operator() FAILED!" << endl;
		return false;
	}
}

//INTERFACE_INCLUDE_BEGIN
BinaryFunctionObj binaryObj(PyObject *pyfunc, bool comm=false);
BinaryPredicateObj binaryObjPred(PyObject *pyfunc);


class SemiringObj {
//INTERFACE_INCLUDE_END
	public:
	static SemiringObj *currentlyApplied;

	// CUSTOM is a semiring with Python-defined methods
	// The others are pre-implemented in C++ for speed.
	typedef enum {CUSTOM, NONE, TIMESPLUS, PLUSMIN, SECONDMAX} SRingType;

	protected:
	SRingType type;
	
	//PyObject *pyfunc_add;
	//PyObject *pyfunc_multiply;
	
	BinaryFunctionObj *binfunc_add;
	BinaryFunctionObj *binfunc_mul;
	template <class T1, class T2, class OUT>
	friend struct SemiringObjTemplArg;
	
	public:
	// CombBLAS' template mechanism means that we can have only one C++ semiring.
	// Multiple Python semirings are implemented by switching them in.
	void enableSemiring();
	void disableSemiring();
	
	public:
	SemiringObj(SRingType t): type(t)/*, pyfunc_add(NULL), pyfunc_multiply(NULL)*/, binfunc_add(NULL), binfunc_mul(NULL) {
		//if (t == CUSTOM)
			// scream bloody murder
	}
	
	SRingType getType() { return type; }
	
//INTERFACE_INCLUDE_BEGIN
	protected:
	SemiringObj(): type(NONE)/*, pyfunc_add(NULL), pyfunc_multiply(NULL)*/, binfunc_add(NULL), binfunc_mul(NULL) {}
	public:
	SemiringObj(PyObject *add, PyObject *multiply);
	~SemiringObj();
	
	MPI_Op mpi_op()
	{
		return *(binfunc_add->getMPIOp());
	}
	
	//doubleint add(const doubleint & arg1, const doubleint & arg2);	
	//doubleint multiply(const doubleint & arg1, const doubleint & arg2);
	//void axpy(doubleint a, const doubleint & x, doubleint & y);

};
//INTERFACE_INCLUDE_END

template <class T1, class T2, class OUT>
struct SemiringObjTemplArg
{
	static OUT id() { return OUT();}

	static MPI_Op mpi_op()
	{
		return SemiringObj::currentlyApplied->mpi_op();
	}
	
	static OUT add(const OUT & arg1, const OUT & arg2)
	{
		return (*(SemiringObj::currentlyApplied->binfunc_add))(arg1, arg2);
	}
	
	static OUT multiply(const T1 & arg1, const T2 & arg2)
	{
		return SemiringObj::currentlyApplied->binfunc_mul->rettype2nd_call(arg1, arg2);
	}
	
	static void axpy(T1 a, const T2 & x, OUT & y)
	{
		//currentlyApplied->axpy(a, x, y);
		//y = add(y, multiply(a, x));
		y = (*(SemiringObj::currentlyApplied->binfunc_add))(y, SemiringObj::currentlyApplied->binfunc_mul->rettype2nd_call(a, x));
	}
};

//INTERFACE_INCLUDE_BEGIN
//SemiringObj TimesPlusSemiringObj();
//SemiringObj MinPlusSemiringObj();
//SemiringObj SecondMaxSemiringObj();
//SemiringObj SecondSecondSemiringObj();
} // namespace op


//INTERFACE_INCLUDE_END

// modeled after CombBLAS/Operations.h
// This call is only safe when between BinaryFunction.getMPIOp() and releaseMPIOp() calls.
// That should be safe enough, because this is only called from inside CombBLAS reduce operations,
// which only get called between getMPIOp() and releaseMPIOp().
template<typename T> struct MPIOp< op::BinaryFunctionObj, T > {  static MPI_Op op() { return op::BinaryFunctionObj::staticMPIop; } };

#endif
