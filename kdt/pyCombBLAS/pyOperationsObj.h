#ifndef PYOPERATIONOBJ_H
#define PYOPERATIONOBJ_H

#ifdef PYCOMBBLAS_MPIOK
#define PYOPERATIONOBJ_H_MPIOK PYCOMBBLAS_MPIOK
#else
#define PYOPERATIONOBJ_H_MPIOK 1
#endif

#if PYOPERATIONOBJ_H_MPIOK
#include "pyCombBLAS.h"
#else
#include "pyCombBLAS-NoMPI.h"
#endif

#include <functional>
#include <iostream>
#include <math.h>

#if !defined(_WIN32)
#pragma GCC diagnostic ignored "-Wwrite-strings"
#endif

#ifndef USESEJITS
#define USESEJITS 1
#endif


//INTERFACE_INCLUDE_BEGIN
namespace op {

class CallError {};

//INTERFACE_INCLUDE_END

#include "pyOperationsWorkers.h"

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
 // these are exactly the same as the operator()s, but they are directly callable from Python
 // and make stacked SEJITS filters easier.
 	bool __call__(const Obj2& x) const { return worker(x); }
	bool __call__(const Obj1& x) const { return worker(x); }
	bool __call__(const double& x) const { return worker(x); }

	//protected:
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
	
	UnaryFunctionObj(PyObject *pyfunc): worker(pyfunc) { 
      swig_module_info* module = SWIG_Python_GetModule(NULL);
      swig_type_info* ty = SWIG_TypeQueryModule(module, module, "op::UnaryFunctionObj *");
      
      UnaryFunctionObj * tmp;
      
      if ((SWIG_ConvertPtr(pyfunc, (void**)&tmp, ty, 0)) == 0) {
        // yes, it is a BinaryFunctionObj
        //          printf("UnaryPredicateObj detected, replicating customized callbacks...\n");
          worker.customFunc_double_double = tmp->worker.customFunc_double_double;
          worker.customFunc_Obj2_double = tmp->worker.customFunc_Obj2_double;
      }

    }
	
	UnaryDoubleFunctionObj_Python getRetDoubleVersion() { return UnaryDoubleFunctionObj_Python(worker.callback); }

//INTERFACE_INCLUDE_BEGIN
	Obj2 operator()(const Obj2& x) const { return worker(x); }
	Obj1 operator()(const Obj1& x) const { return worker(x); }
	double operator()(const double& x) const { return worker(x); }
	
    //	protected:
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
	//PyObject *vertexArgList = Py_BuildValue("(O)", tempSwigObj);
	
	resultPy = PyObject_CallFunction(callback,"(O)", tempSwigObj);  

	Py_XDECREF(tempSwigObj);
	//Py_XDECREF(vertexArgList);
	if (resultPy && SWIG_IsOK(SWIG_ConvertPtr(resultPy, (void**)&pret, T::SwigTypeInfo,  0  | 0)) && pret != NULL) {
		T ret = T(*pret);
		Py_XDECREF(resultPy);
		return ret;
	} else
	{
		if (PyErr_Occurred())
		{
			PyErr_Print();
			throw string("UnaryFunctionObj_Python::operator(T) FAILED! (with exception)");
		}
		else
			throw string("UnaryFunctionObj_Python::operator(T) FAILED! (no exception, maybe return value was expected but not found)");
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
		if (PyErr_Occurred())
		{
			PyErr_Print();
			throw string("UnaryFunctionObj_Python::operator(double) FAILED! (with exception)");
		}
		else
			throw string("UnaryFunctionObj_Python::operator(double) FAILED! (no exception, maybe return value was expected but not found)");
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
		if (PyErr_Occurred())
		{
			PyErr_Print();
			throw string("UnaryDoublePredicateObj_Python::operator(T) FAILED! (with exception)");
		}
		else
			throw string("UnaryDoubleFunctionObj_Python::operator(T) FAILED! (no exception, maybe return value was expected but not found)");
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
		if (PyErr_Occurred())
		{
			PyErr_Print();
			throw string("UnaryDoublePredicateObj_Python::operator(double) FAILED! (with exception)");
		}
		else
			throw string("UnaryDoubleFunctionObj_Python::operator(double) FAILED! (no exception, maybe return value was expected but not found)");
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
	//PyObject *vertexArgList = Py_BuildValue("(O)", tempSwigObj);
	
	resultPy = PyObject_CallFunction(callback,"(O)", tempSwigObj);  

	Py_XDECREF(tempSwigObj);
	//Py_XDECREF(vertexArgList);
	if (resultPy) {
		bool ret = PyObject_IsTrue(resultPy);
		Py_XDECREF(resultPy);
		return ret;
	} else
	{
		if (PyErr_Occurred())
		{
			PyErr_Print();
			throw string("UnaryPredicateObj_Python::operator(T) FAILED! (with exception)");
		}
		else
			throw string("UnaryPredicateObj_Python::operator(T) FAILED! (no exception, maybe return value was expected but not found)");
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
		if (PyErr_Occurred())
		{
			PyErr_Print();
			throw string("UnaryPredicateObj_Python::operator(double) FAILED! (with exception)");
		}
		else
			throw string("UnaryPredicateObj_Python::operator(double) FAILED! (no exception, maybe return value was expected but not found)");
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

    //	BinaryFunctionObj(PyObject *pyfunc, bool as, bool com): worker(pyfunc), commutable(com), associative(as) { }
    BinaryFunctionObj(PyObject *pyfunc, bool as, bool com) : worker(pyfunc), commutable(com), associative(as) {
      swig_module_info* module = SWIG_Python_GetModule(NULL);
      swig_type_info* ty = SWIG_TypeQueryModule(module, module, "op::BinaryFunctionObj *");
      
      BinaryFunctionObj * tmp;
      
      if ((SWIG_ConvertPtr(pyfunc, (void**)&tmp, ty, 0)) == 0) {
        // yes, it is a BinaryFunctionObj
        //          printf("UnaryPredicateObj detected, replicating customized callbacks...\n");
          worker.customFunc_doubledouble_double = tmp->worker.customFunc_doubledouble_double;
          worker.customFunc_Obj2double_double = tmp->worker.customFunc_Obj2double_double;
          worker.customFunc_Obj2double_Obj2 = tmp->worker.customFunc_Obj2double_Obj2;
      }
    }
	
	// For dealing with MPI. The prototypes do not mention MPI so that this header can be compiled without MPI (for SEJITS)
	// These functions are actually implemented by the class BinaryFunctionObj_MPI_Interface.
	void getMPIOp();
	void releaseMPIOp();

//INTERFACE_INCLUDE_BEGIN
//	protected:
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

#if PYOPERATIONOBJ_H_MPIOK

class BinaryFunctionObj_MPI_Interface {
public:
	// for creating an MPI_Op that can be used with MPI Reduce
	static void apply(void * invec, void * inoutvec, int * len, MPI_Datatype *datatype);
	template <class T1, class T2>
	static void applyWorker(T1 * in, T2 * inout, int * len);

	static BinaryFunctionObj* currentlyApplied;
	static MPI_Op staticMPIop;

	static MPI_Op* mpi_op();
};

template <class T1, class T2>
void BinaryFunctionObj_MPI_Interface::applyWorker(T1 * in, T2 * inout, int * len)
{
	for (int i = 0; i < *len; i++)
	{
		inout[i] = (*currentlyApplied)(in[i], inout[i]);
	}
}

#endif

template <typename RET, typename T1, typename T2>
RET BinaryFunctionObj_Python::call(const T1& x, const T2& y) const
{
	PyObject *resultPy;
	RET *pret;	

	T1 tempObj1 = x;
	T2 tempObj2 = y;
	PyObject *tempSwigObj1 = SWIG_NewPointerObj(&tempObj1, T1::SwigTypeInfo, 0);
	PyObject *tempSwigObj2 = SWIG_NewPointerObj(&tempObj2, T2::SwigTypeInfo, 0);
	//PyObject *vertexArgList = Py_BuildValue("(O O)", tempSwigObj1, tempSwigObj2);
	
	//resultPy = PyEval_CallObject(callback,vertexArgList);  
	resultPy = PyObject_CallFunction(callback,"(O O)", tempSwigObj1, tempSwigObj2);  

	Py_XDECREF(tempSwigObj1);
	Py_XDECREF(tempSwigObj2);
	//Py_XDECREF(vertexArgList);
	if (resultPy && SWIG_IsOK(SWIG_ConvertPtr(resultPy, (void**)&pret, RET::SwigTypeInfo,  0  | 0)) && pret != NULL) {
		RET ret = RET(*pret);
		Py_XDECREF(resultPy);
		return ret;
	} else
	{
		Py_XDECREF(resultPy);
		if (PyErr_Occurred())
		{
			PyErr_Print();
			throw string("BinaryFunctionObj_Python::operator() FAILED! (callOO) (with exception)");
		}
		else
			throw string("BinaryFunctionObj_Python::operator() FAILED! (no exception, maybe return value was expected but not found) (callOO)");
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
	//PyObject *vertexArgList = Py_BuildValue("(O d)", tempSwigObj1, y);
	
	//resultPy = PyEval_CallObject(callback,vertexArgList);  
	resultPy = PyObject_CallFunction(callback,"(O d)", tempSwigObj1, y);  

	Py_XDECREF(tempSwigObj1);
	//Py_XDECREF(vertexArgList);
	if (resultPy) {                                   // If no errors, return double
		dres = PyFloat_AsDouble(resultPy);
		Py_XDECREF(resultPy);
		return dres;
	} else
	{
		Py_XDECREF(resultPy);
		if (PyErr_Occurred())
		{
			PyErr_Print();
			throw string("BinaryFunctionObj_Python::operator() FAILED! (callOD) (with exception)");
		}
		else
			throw string("BinaryFunctionObj_Python::operator() FAILED! (no exception, maybe return value was expected but not found) (callOD)");
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
	//PyObject *vertexArgList = Py_BuildValue("(O d)", tempSwigObj1, y);
	
	//resultPy = PyEval_CallObject(callback,vertexArgList);  
	resultPy = PyObject_CallFunction(callback,"(O d)", tempSwigObj1, y);  

	Py_XDECREF(tempSwigObj1);
	//Py_XDECREF(vertexArgList);
	if (resultPy && SWIG_IsOK(SWIG_ConvertPtr(resultPy, (void**)&pret, RET::SwigTypeInfo,  0  | 0)) && pret != NULL) {
		RET ret = RET(*pret);
		Py_XDECREF(resultPy);
		return ret;
	} else
	{
		Py_XDECREF(resultPy);
		if (PyErr_Occurred())
		{
			PyErr_Print();
			throw string("BinaryFunctionObj_Python::operator() FAILED! (callOD_retO) (with exception)");
		}
		else
			throw string("BinaryFunctionObj_Python::operator() FAILED! (no exception, maybe return value was expected but not found) (callOD_retO)");
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
		if (PyErr_Occurred())
		{
			PyErr_Print();
			throw string("BinaryFunctionObj_Python::operator() FAILED! (callDO_retD) (with exception)");
		}
		else
			throw string("BinaryFunctionObj_Python::operator() FAILED! (no exception, maybe return value was expected but not found) (callDO)");
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
		if (PyErr_Occurred())
		{
			PyErr_Print();
			throw string("BinaryFunctionObj_Python::operator() FAILED! (callDO_retO) (with exception)");
		}
		else
			throw string("BinaryFunctionObj_Python::operator() FAILED! (no exception, maybe return value was expected but not found) (callDO_retO)");
		return RET();
	}
}

// quick and dirty function. performs:
// SemiringObj::currentlyApplied = NULL;
// That can't be done directly because SemiringObj is still an incomplete type at this point, so
// this prevents some nasty re-ordering of everything in the file.
void clear_SemiringObj_currentlyApplied();
void clear_BinaryFunctionObj_currentlyApplied();

inline double BinaryFunctionObj_Python::callDD(const double& x, const double& y) const
{
	PyObject *arglist;
	PyObject *resultPy;
	double dres = 0;

    /*    printf("Doing callback in callDD\n");
    if (callback == Py_None)
      printf("But callback is NONE\n");
    if (callback == NULL)
      printf("But callback is NULL\n");
    */
	arglist = Py_BuildValue("(d d)", x, y);    // Build argument list
	resultPy = PyEval_CallObject(callback,arglist);     // Call Python
	Py_DECREF(arglist);                             // Trash arglist
	if (resultPy) {                                   // If no errors, return double
		dres = PyFloat_AsDouble(resultPy);
		Py_XDECREF(resultPy);
        return dres;
	} else
	{
		clear_BinaryFunctionObj_currentlyApplied();
		clear_SemiringObj_currentlyApplied();
		if (PyErr_Occurred())
		{
			PyErr_Print();
			throw string("BinaryFunctionObj_Python::operator() FAILED! (callDD) (with exception)");
		}
		else
			throw string("BinaryFunctionObj_Python::operator() FAILED! (no exception, maybe return value was expected but not found) (callDD)");
		return 0;
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

	//protected:
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
		if (PyErr_Occurred())
		{
			PyErr_Print();
			throw string("BinaryPredicateObj_Python::operator() FAILED! (call (OO)) (with exception)");
		}
		else
			throw string("BinaryPredicateObj_Python::operator() FAILED! (no exception, maybe return value was expected but not found) (call (OO))");
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
		if (PyErr_Occurred())
		{
			PyErr_Print();
			throw string("BinaryPredicateObj_Python::operator() FAILED! (callOD) (with exception)");
		}
		else
			throw string("BinaryPredicateObj_Python::operator() FAILED! (no exception, maybe return value was expected but not found) (callOD)");
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
		if (PyErr_Occurred())
		{
			PyErr_Print();
			throw string("BinaryPredicateObj_Python::operator() FAILED! (callDO) (with exception)");
		}
		else
			throw string("BinaryPredicateObj_Python::operator() FAILED! (no exception, maybe return value was expected but not found) (callDO)");
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
		if (PyErr_Occurred())
		{
			PyErr_Print();
			throw string("BinaryPredicateObj_Python::operator() FAILED! (callDD) (with exception)");
		}
		else
			throw string("BinaryPredicateObj_Python::operator() FAILED! (no exception, maybe return value was expected but not found) (callDD)");
		return false;
	}
}

//INTERFACE_INCLUDE_BEGIN
BinaryFunctionObj binaryObj(PyObject *pyfunc, bool comm=false);
BinaryPredicateObj binaryObjPred(PyObject *pyfunc);
//INTERFACE_INCLUDE_END

#if PYOPERATIONOBJ_H_MPIOK

//INTERFACE_INCLUDE_BEGIN
class SemiringObj {
//INTERFACE_INCLUDE_END
	public:
	static SemiringObj *currentlyApplied;

	// CUSTOM is a semiring with Python-defined methods
	// The others are pre-implemented in C++ for speed.
	typedef enum {CUSTOM, NONE, TIMESPLUS, PLUSMIN, SECONDMAX} SRingType;

	//protected: // this used to be protected, but there seems to be some issues with some compilers and friends, so screw it, these fields are now public.
	public:
	SRingType type;
	
	//PyObject *pyfunc_add;
	//PyObject *pyfunc_multiply;
	
	BinaryFunctionObj *binfunc_add;
	BinaryFunctionObj *binfunc_mul;
	UnaryPredicateObj* left_filter;
	UnaryPredicateObj* right_filter;
	//template <class T1, class T2, class OUT>
	//friend struct SemiringObjTemplArg;
	
	public:
	// CombBLAS' template mechanism means that we can have only one C++ semiring.
	// Multiple Python semirings are implemented by switching them in.
	void enableSemiring();
	void disableSemiring();
	
	public:
	SemiringObj(SRingType t): type(t), binfunc_add(NULL), binfunc_mul(NULL), left_filter(NULL), right_filter(NULL) {
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
	
	void setFilters(PyObject* left_filter_py = NULL, PyObject* right_filter_py = NULL);
	
	PyObject* getAddCallback() const { return binfunc_add != NULL ? binfunc_add->getCallback() : NULL; }
	PyObject* getMulCallback() const { return binfunc_mul != NULL ? binfunc_mul->getCallback() : NULL; }
	
	MPI_Op mpi_op()
	{
		binfunc_add->getMPIOp();
		return *(BinaryFunctionObj_MPI_Interface::mpi_op());
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

	// the default argument means that this function can be used like this:
	// if (returnedSAID()) {...}
	// which is how it is called inside CombBLAS routines. That call conveniently clears the flag for us.
	static bool returnedSAID(bool setFlagTo=false)
	{
		static bool flag = false;
		
		bool temp = flag; // save the current flag value to be returned later. Saves an if statement.
		flag = setFlagTo; // set/clear the flag.
		return temp;
	}
	
	static MPI_Op mpi_op()
	{
		return SemiringObj::currentlyApplied->mpi_op();
	}
	
	static OUT add(const OUT & arg1, const OUT & arg2)
	{
      //      printf("calling SEMIRING ADD\n");
		return (*(SemiringObj::currentlyApplied->binfunc_add))(arg1, arg2);
	}
	
	static OUT multiply(const T1 & arg1, const T2 & arg2)
	{
      //printf("calling SEMIRING MUL\n");
		// see if we do filtering here
		if (SemiringObj::currentlyApplied->left_filter != NULL || SemiringObj::currentlyApplied->right_filter != NULL)
		{
			// filter the left parameter
			if (SemiringObj::currentlyApplied->left_filter != NULL && !(*(SemiringObj::currentlyApplied->left_filter))(arg1))
			{
				returnedSAID(true);
				return id();
			}
			// filter the right parameter
			if (SemiringObj::currentlyApplied->right_filter != NULL && !(*(SemiringObj::currentlyApplied->right_filter))(arg2))
			{
				returnedSAID(true);
				return id();
			}
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
SemiringObj TimesPlusSemiringObj();
//SemiringObj MinPlusSemiringObj();
SemiringObj SecondMaxSemiringObj();
//SemiringObj SecondSecondSemiringObj();

//INTERFACE_INCLUDE_END
#endif

//INTERFACE_INCLUDE_BEGIN
} // namespace op
//INTERFACE_INCLUDE_END

#if PYOPERATIONOBJ_H_MPIOK
// modeled after CombBLAS/Operations.h
// This call is only safe when between BinaryFunction.getMPIOp() and releaseMPIOp() calls.
// That should be safe enough, because this is only called from inside CombBLAS reduce operations,
// which only get called between getMPIOp() and releaseMPIOp().
template<typename T> struct MPIOp< op::BinaryFunctionObj, T > {  static MPI_Op op() { return op::BinaryFunctionObj_MPI_Interface::staticMPIop; } };
#endif

#endif
