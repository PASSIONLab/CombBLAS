#ifndef PYOPERATIONOBJ_H
#define PYOPERATIONOBJ_H

#include "pyCombBLAS.h"
#include <functional>
#include <iostream>
#include <math.h>

#ifndef USESEJITS
#define USESEJITS 1
#endif

#if USESEJITS
// use SEJITS-backed workers
#define UnaryPredicateObj_WorkerType UnaryPredicateObj_SEJITS
#define UnaryFunctionObj_WorkerType UnaryFunctionObj_SEJITS
#define BinaryPredicateObj_WorkerType BinaryPredicateObj_SEJITS
#define BinaryFunctionObj_WorkerType BinaryFunctionObj_SEJITS
#else
// use plain-jane Python workers
#define UnaryPredicateObj_WorkerType UnaryPredicateObj_Python
#define UnaryFunctionObj_WorkerType UnaryFunctionObj_Python
#define BinaryPredicateObj_WorkerType BinaryPredicateObj_Python
#define BinaryFunctionObj_WorkerType BinaryFunctionObj_Python
#endif

//INTERFACE_INCLUDE_BEGIN
namespace op {

// WORKERS
//////////////////////

//INTERFACE_INCLUDE_END
class UnaryPredicateObj_Python {
	public:
	PyObject *callback;
	UnaryPredicateObj_Python(PyObject *pyfunc): callback(pyfunc) { Py_INCREF(callback); }

	template <class T>
	bool call(const T& x) const;

	bool callD(const double& x) const;
	
	bool operator()(const Obj2& x) const { return call(x); }
	bool operator()(const Obj1& x) const { return call(x); }
	bool operator()(const double& x) const { return callD(x); }

	UnaryPredicateObj_Python() { // should never be called
		printf("UnaryPredicateObj_Python()!!!\n");
		callback = NULL;
	}

	public:
	~UnaryPredicateObj_Python() { /*Py_XDECREF(callback);*/ }
};

// This class is meant to enable type conversion (for example in Reduce).
// Yes it only allows changing to double, but that will be expanded in the future.
class UnaryDoubleFunctionObj_Python {
	public:
	PyObject *callback;
	UnaryDoubleFunctionObj_Python(PyObject *pyfunc): callback(pyfunc) { Py_INCREF(callback); }

	template <class T>
	double call(const T& x) const;

	double callD(const double& x) const;

	double operator()(const Obj2& x) const { return call(x); }
	double operator()(const Obj1& x) const { return call(x); }
	double operator()(const double& x) const { return callD(x); }
	
	UnaryDoubleFunctionObj_Python() { // should never be called
		printf("UnaryDoubleFunctionObj_Python()!!!\n");
		callback = NULL;
	}

	public:
	~UnaryDoubleFunctionObj_Python() { /*Py_XDECREF(callback);*/ }
};

class UnaryFunctionObj_Python {
	public:
	PyObject *callback;
	UnaryFunctionObj_Python(PyObject *pyfunc): callback(pyfunc) { Py_INCREF(callback); }

	template <class T>
	T call(const T& x) const;

	double callD(const double& x) const;

	Obj2 operator()(const Obj2& x) const { return call(x); }
	Obj1 operator()(const Obj1& x) const { return call(x); }
	double operator()(const double& x) const { return callD(x); }
	
	UnaryFunctionObj_Python() { // should never be called
		printf("UnaryFunctionObj_Python()!!!\n");
		callback = NULL;
	}

	public:
	~UnaryFunctionObj_Python() { /*Py_XDECREF(callback);*/ }
};

class BinaryFunctionObj_Python {
	public:
	PyObject *callback;

	BinaryFunctionObj_Python(PyObject *pyfunc): callback(pyfunc) { Py_INCREF(callback); }

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
	
	BinaryFunctionObj_Python(): callback(NULL) {}
	public:
	~BinaryFunctionObj_Python() { /*Py_XDECREF(callback);*/ }
	
	PyObject* getCallback() const { return callback; }
	
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

class BinaryPredicateObj_Python {
	public:
	PyObject *callback;
	BinaryPredicateObj_Python(PyObject *pyfunc): callback(pyfunc) { Py_INCREF(callback); }

	template <class T1, class T2>
	bool call(const T1& x, const T2& y) const;

	template <class T1>
	bool callOD(const T1& x, const double& y) const;
	template <class T2>
	bool callDO(const double& x, const T2& y) const;

	inline bool callDD(const double& x, const double& y) const;
	
	bool operator()(const Obj1& x, const Obj1& y) const { return call(x, y); }
	bool operator()(const Obj1& x, const Obj2& y) const { return call(x, y); }
	bool operator()(const Obj2& x, const Obj2& y) const { return call(x, y); }
	bool operator()(const Obj2& x, const Obj1& y) const { return call(x, y); }

	bool operator()(const Obj1& x, const double& y) const { return callOD(x, y); }
	bool operator()(const Obj2& x, const double& y) const { return callOD(x, y); }
	bool operator()(const double& x, const Obj2& y) const { return callDO(x, y); }
	bool operator()(const double& x, const Obj1& y) const { return callDO(x, y); }

	bool operator()(const double& x, const double& y) const { return callDD(x, y); }

	BinaryPredicateObj_Python() { // should never be called
		printf("BinaryPredicateObj_Python()!!!\n");
		callback = NULL;
	}

	public:
	~BinaryPredicateObj_Python() { /*Py_XDECREF(callback);*/ }
};

#include "pyOperationsSEJITS.h"

/// DONE WITH WORKER DECLARATIONS
///////////////////////////////////


//INTERFACE_INCLUDE_BEGIN

class UnaryPredicateObj {
//INTERFACE_INCLUDE_END
	public:
	UnaryPredicateObj_WorkerType worker;
	UnaryPredicateObj(PyObject *pyfunc): worker(pyfunc) { }

//INTERFACE_INCLUDE_BEGIN
	bool operator()(const Obj2& x) const { return worker(x); }
	bool operator()(const Obj1& x) const { return worker(x); }
	bool operator()(const double& x) const { return worker(x); }

	protected:
	UnaryPredicateObj() { // should never be called
		printf("UnaryPredicateObj()!!!\n");
	}

	public:
	~UnaryPredicateObj() { }
};

class UnaryFunctionObj {
//INTERFACE_INCLUDE_END
	public:
	UnaryFunctionObj_WorkerType worker;
	
	UnaryFunctionObj(PyObject *pyfunc): worker(pyfunc) { }
	
	UnaryDoubleFunctionObj_Python getRetDoubleVersion() { return UnaryDoubleFunctionObj_Python(worker.callback); }

//INTERFACE_INCLUDE_BEGIN
	Obj2 operator()(const Obj2& x) const { return worker(x); }
	Obj1 operator()(const Obj1& x) const { return worker(x); }
	double operator()(const double& x) const { return worker(x); }
	
	protected:
	UnaryFunctionObj() { // should never be called
		printf("UnaryFunctionObj()!!!\n");
	}

	public:
	~UnaryFunctionObj() { }
};

//INTERFACE_INCLUDE_END
template <class T>
T UnaryFunctionObj_Python::call(const T& x) const
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
		cerr << "UnaryFunctionObj_Python::operator(T) FAILED!" << endl;
		//throw T();
		return T();
	}
}

inline double UnaryFunctionObj_Python::callD(const double& x) const
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
		cerr << "UnaryFunctionObj_Python::operator(double) FAILED!" << endl;
		//throw doubleint();
		return 0;
	}
}
/////////////////////////////

template <class T>
double UnaryDoubleFunctionObj_Python::call(const T& x) const
{
	PyObject *resultPy;
	//T *pret;	

	T tempObj = x;
	PyObject *tempSwigObj = SWIG_NewPointerObj(&tempObj, T::SwigTypeInfo, 0);
	//PyObject *vertexArgList = Py_BuildValue("(O)", tempSwigObj);
	
	resultPy = PyObject_CallFunction(callback,"(O)", tempSwigObj);  

	Py_XDECREF(tempSwigObj);
	//Py_XDECREF(vertexArgList);
	if (resultPy) {
		double dres = PyFloat_AsDouble(resultPy);
		Py_XDECREF(resultPy);
		return dres;
	} else
	{
		cerr << "UnaryDoubleFunctionObj_Python::operator(T) FAILED!" << endl;
		throw doubleint();
		return T();
	}
}

inline double UnaryDoubleFunctionObj_Python::callD(const double& x) const
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
		cerr << "UnaryDoubleFunctionObj_Python::operator(double) FAILED!" << endl;
		throw doubleint();
		return 0;
	}
}


/////////////////////////////

// This function is identical to UnaryFunctionObj_Python::call() except that it returns a boolean instead
// of an object. Please keep the actual calling method the same if you make any changes.
template <class T>
bool UnaryPredicateObj_Python::call(const T& x) const
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
		cerr << "UnaryPredicateObj_Python::operator(T) FAILED!" << endl;
		return false;
	}
}

inline bool UnaryPredicateObj_Python::callD(const double& x) const
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
		cerr << "UnaryPredicateObj_Python::operator(double) FAILED!" << endl;
		return 0;
	}
}

//INTERFACE_INCLUDE_BEGIN

UnaryFunctionObj unaryObj(PyObject *pyfunc);
UnaryPredicateObj unaryObjPred(PyObject *pyfunc);

class BinaryFunctionObj {
//INTERFACE_INCLUDE_END
	public:
	BinaryFunctionObj_WorkerType worker;

	BinaryFunctionObj(PyObject *pyfunc, bool as, bool com): worker(pyfunc), commutable(com), associative(as) { }

	// for creating an MPI_Op that can be used with MPI Reduce
	static void apply(void * invec, void * inoutvec, int * len, MPI_Datatype *datatype);
	template <class T1, class T2>
	static void applyWorker(T1 * in, T2 * inout, int * len);

	static BinaryFunctionObj* currentlyApplied;
	static MPI_Op staticMPIop;
	
	MPI_Op* getMPIOp();
	void releaseMPIOp();

//INTERFACE_INCLUDE_BEGIN
	protected:
	BinaryFunctionObj(): commutable(false), associative(false) {}
	public:
	~BinaryFunctionObj() {  }
	
	PyObject* getCallback() const { return worker.getCallback(); }
	
	bool commutable;
	bool associative;
	
	Obj1 operator()(const Obj1& x, const Obj1& y) const { return worker(x, y); }
	Obj2 operator()(const Obj2& x, const Obj2& y) const { return worker(x, y); }
	Obj1 operator()(const Obj1& x, const Obj2& y) const { return worker(x, y); }
	Obj2 operator()(const Obj2& x, const Obj1& y) const { return worker(x, y); }

	Obj1 operator()(const Obj1& x, const double& y) const { return worker(x, y); }
	Obj2 operator()(const Obj2& x, const double& y) const { return worker(x, y); }
	double operator()(const double& x, const Obj1& y) const { return worker(x, y); }
	double operator()(const double& x, const Obj2& y) const { return worker(x, y); }

	double operator()(const double& x, const double& y) const { return worker(x, y); }


	// These are used by the semiring ops. They do the same thing as the operator() above,
	// but their return type matches the 2nd argument instead of the 1st.
	Obj1 rettype2nd_call(const Obj1& x, const Obj1& y) const { return worker.rettype2nd_call(x, y); }
	Obj2 rettype2nd_call(const Obj2& x, const Obj2& y) const { return worker.rettype2nd_call(x, y); }
	Obj1 rettype2nd_call(const Obj2& x, const Obj1& y) const { return worker.rettype2nd_call(x, y); }
	Obj2 rettype2nd_call(const Obj1& x, const Obj2& y) const { return worker.rettype2nd_call(x, y); }

	double rettype2nd_call(const Obj1& x, const double& y) const { return worker.rettype2nd_call(x, y); }
	double rettype2nd_call(const Obj2& x, const double& y) const { return worker.rettype2nd_call(x, y); }
	Obj1 rettype2nd_call(const double& x, const Obj1& y) const { return worker.rettype2nd_call(x, y); }
	Obj2 rettype2nd_call(const double& x, const Obj2& y) const { return worker.rettype2nd_call(x, y); }

	double rettype2nd_call(const double& x, const double& y) const { return worker.rettype2nd_call(x, y); }

};

//INTERFACE_INCLUDE_END
template <typename RET, typename T1, typename T2>
RET BinaryFunctionObj_Python::call(const T1& x, const T2& y) const
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
		cerr << "BinaryFunctionObj_Python::operator() FAILED (callOO)!" << endl;
		return RET();
	}
}

template <typename T1>
double BinaryFunctionObj_Python::callOD_retD(const T1& x, const double& y) const
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
		cerr << "BinaryFunctionObj_Python::operator() FAILED (callOD)!" << endl;
		return 0;
	}
}

template <typename RET, typename T1>
RET BinaryFunctionObj_Python::callOD_retO(const T1& x, const double& y) const
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
		cerr << "BinaryFunctionObj_Python::operator() FAILED (callOD)!" << endl;
		return RET();
	}
}

template <typename T2>
double BinaryFunctionObj_Python::callDO_retD(const double& x, const T2& y) const
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
		cerr << "BinaryFunctionObj_Python::operator() FAILED! (callDO)" << endl;
		return 0;
	}
}

template <typename RET, typename T2>
RET BinaryFunctionObj_Python::callDO_retO(const double& x, const T2& y) const
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
		cerr << "BinaryFunctionObj_Python::operator() FAILED! (callDO)" << endl;
		return RET();
	}
}

inline double BinaryFunctionObj_Python::callDD(const double& x, const double& y) const
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
		cerr << "BinaryFunctionObj_Python::operator() FAILED! (callDD)" << endl;
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
	BinaryPredicateObj_WorkerType worker;
	BinaryPredicateObj(PyObject *pyfunc): worker(pyfunc) { }

//INTERFACE_INCLUDE_BEGIN
	bool operator()(const Obj1& x, const Obj1& y) const { return worker(x, y); }
	bool operator()(const Obj1& x, const Obj2& y) const { return worker(x, y); }
	bool operator()(const Obj2& x, const Obj2& y) const { return worker(x, y); }
	bool operator()(const Obj2& x, const Obj1& y) const { return worker(x, y); }

	bool operator()(const Obj1& x, const double& y) const { return worker(x, y); }
	bool operator()(const Obj2& x, const double& y) const { return worker(x, y); }
	bool operator()(const double& x, const Obj2& y) const { return worker(x, y); }
	bool operator()(const double& x, const Obj1& y) const { return worker(x, y); }

	bool operator()(const double& x, const double& y) const { return worker(x, y); }

	protected:
	BinaryPredicateObj() { // should never be called
		printf("BinaryPredicateObj()!!!\n");
	}

	public:
	~BinaryPredicateObj() { /*Py_XDECREF(callback);*/ }
};

//INTERFACE_INCLUDE_END
template <class T1, class T2>
bool BinaryPredicateObj_Python::call(const T1& x, const T2& y) const
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
		cerr << "BinaryPredicateObj_Python::operator() FAILED!" << endl;
		return false;
	}
}

template <class T1>
bool BinaryPredicateObj_Python::callOD(const T1& x, const double& y) const
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
		cerr << "BinaryPredicateObj_Python::operator() FAILED!" << endl;
		return false;
	}
}

template <class T2>
bool BinaryPredicateObj_Python::callDO(const double& x, const T2& y) const
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
		cerr << "BinaryPredicateObj_Python::operator() FAILED!" << endl;
		return false;
	}
}

bool BinaryPredicateObj_Python::callDD(const double& x, const double& y) const
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
		cerr << "BinaryPredicateObj_Python::operator() FAILED!" << endl;
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
	UnaryPredicateObj* left_filter;
	UnaryPredicateObj* right_filter;
	template <class T1, class T2, class OUT>
	friend struct SemiringObjTemplArg;
	
	public:
	// CombBLAS' template mechanism means that we can have only one C++ semiring.
	// Multiple Python semirings are implemented by switching them in.
	void enableSemiring();
	void disableSemiring();
	
	public:
	SemiringObj(SRingType t): type(t)/*, pyfunc_add(NULL), pyfunc_multiply(NULL)*/, binfunc_add(NULL), binfunc_mul(NULL), left_filter(NULL), right_filter(NULL) {
		//if (t == CUSTOM)
			// scream bloody murder
	}
	
	SRingType getType() { return type; }
	
//INTERFACE_INCLUDE_BEGIN
	protected:
	SemiringObj(): type(NONE)/*, pyfunc_add(NULL), pyfunc_multiply(NULL)*/, binfunc_add(NULL), binfunc_mul(NULL), left_filter(NULL), right_filter(NULL) {}
	public:
	SemiringObj(PyObject *add, PyObject *multiply, PyObject* left_filter_py = NULL, PyObject* right_filter_py = NULL);
	~SemiringObj();
	
	PyObject* getAddCallback() const { return binfunc_add != NULL ? binfunc_add->getCallback() : NULL; }
	PyObject* getMulCallback() const { return binfunc_mul != NULL ? binfunc_mul->getCallback() : NULL; }
	
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
		// see if we do filtering here
		if (SemiringObj::currentlyApplied->left_filter != NULL || SemiringObj::currentlyApplied->right_filter != NULL)
		{
			// filter the left parameter
			if (SemiringObj::currentlyApplied->left_filter != NULL && !(*(SemiringObj::currentlyApplied->left_filter))(arg1))
				return id();
			// filter the right parameter
			if (SemiringObj::currentlyApplied->right_filter != NULL && !(*(SemiringObj::currentlyApplied->right_filter))(arg2))
				return id();
		}
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
