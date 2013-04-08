#ifndef PYOPERATIONSWORKERS_H
#define PYOPERATIONSWORKERS_H

// WORKERS
//////////////////////

class UnaryPredicateObj_Python {
	public:
	PyObject *callback;
	UnaryPredicateObj_Python(PyObject *pyfunc): callback(pyfunc) { Py_INCREF(callback); }

	PyObject* getCallback() const { return callback; }
	void setCallback(PyObject* c) { callback = c; Py_INCREF(callback); }
	
	template <class T>
	bool call(const T& x) const;

	bool callD(const double& x) const;
	
	bool operator()(const Obj2& x) const { return call(x); }
	bool operator()(const Obj1& x) const { return call(x); }
	bool operator()(const double& x) const { return callD(x); }

	UnaryPredicateObj_Python() {
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

	PyObject* getCallback() const { return callback; }
	void setCallback(PyObject* c) { callback = c; Py_INCREF(callback); }
	
	template <class T>
	double call(const T& x) const;

	double callD(const double& x) const;

	double operator()(const Obj2& x) const { return call(x); }
	double operator()(const Obj1& x) const { return call(x); }
	double operator()(const double& x) const { return callD(x); }
	
	UnaryDoubleFunctionObj_Python() { 
		callback = NULL;
	}

	public:
	~UnaryDoubleFunctionObj_Python() { /*Py_XDECREF(callback);*/ }
};

class UnaryFunctionObj_Python {
	public:
	PyObject *callback;
	UnaryFunctionObj_Python(PyObject *pyfunc): callback(pyfunc) { Py_INCREF(callback); }

	PyObject* getCallback() const { return callback; }
	void setCallback(PyObject* c) { callback = c; Py_INCREF(callback); }
	
	template <class T>
	T call(const T& x) const;

	double callD(const double& x) const;

	Obj2 operator()(const Obj2& x) const { return call(x); }
	Obj1 operator()(const Obj1& x) const { return call(x); }
	double operator()(const double& x) const { return callD(x); }
	
	UnaryFunctionObj_Python() { 
		callback = NULL;
	}

	public:
	~UnaryFunctionObj_Python() { /*Py_XDECREF(callback);*/ }
};

class BinaryFunctionObj_Python {
	public:
	PyObject *callback;

	BinaryFunctionObj_Python(PyObject *pyfunc): callback(pyfunc) { Py_INCREF(callback); }

	BinaryFunctionObj_Python(): callback(NULL) {}
	public:
	~BinaryFunctionObj_Python() { /*Py_XDECREF(callback);*/ }
	
	PyObject* getCallback() const { return callback; }
	void setCallback(PyObject* c) { callback = c; Py_INCREF(callback); }
	
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

	PyObject* getCallback() const { return callback; }
	void setCallback(PyObject* c) { callback = c; Py_INCREF(callback); }
	
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

	BinaryPredicateObj_Python() {
		callback = NULL;
	}

	public:
	~BinaryPredicateObj_Python() { /*Py_XDECREF(callback);*/ }
};

#if USESEJITS
// use SEJITS-backed workers
#define UnaryPredicateObj_WorkerType UnaryPredicateObj_SEJITS
#define UnaryFunctionObj_WorkerType UnaryFunctionObj_SEJITS
#define BinaryPredicateObj_WorkerType BinaryPredicateObj_SEJITS
#define BinaryFunctionObj_WorkerType BinaryFunctionObj_SEJITS

#include "pyOperationsSEJITS.h"

#else
// use plain-jane Python workers
#define UnaryPredicateObj_WorkerType UnaryPredicateObj_Python
#define UnaryFunctionObj_WorkerType UnaryFunctionObj_Python
#define BinaryPredicateObj_WorkerType BinaryPredicateObj_Python
#define BinaryFunctionObj_WorkerType BinaryFunctionObj_Python
#endif

/// DONE WITH WORKER DECLARATIONS
///////////////////////////////////

#endif