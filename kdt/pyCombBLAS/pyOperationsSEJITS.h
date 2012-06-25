#ifndef PYOPERATIONSSEJITS_H
#define PYOPERATIONSSEJITS_H

#include "pyCombBLAS-NoMPI.h"
#include "pyOperationsWorkers.h"

class UnaryPredicateObj_SEJITS : public UnaryPredicateObj_Python {
	public:

    // specialized functions, for each possible input type
    bool (*customFuncO1)(const Obj1& x); 
    bool (*customFuncO2)(const Obj2& x); 
    bool (*customFuncD)(const double& x);

	UnaryPredicateObj_SEJITS(PyObject *pyfunc): UnaryPredicateObj_Python(pyfunc) {  
      // set specialized function pointers to NULL.
      // specializer code will replace them
      customFuncO1 = NULL; 
      customFuncO2 = NULL;  
      customFuncD = NULL;

      // now we check if the PyObject is actually a UnaryPredicateObj
      // in disguise
      swig_module_info* module = SWIG_Python_GetModule();
      swig_type_info* ty = SWIG_TypeQueryModule(module, module, "op::UnaryPredicateObj *");

      UnaryPredicateObj_SEJITS* tmp;

      if ((SWIG_ConvertPtr(callback, (void**)&tmp, ty, 0)) == 0) {
        // yes, it is a UnaryPredicateObj
        //printf("UnaryPredicateObj detected, replicating customized callbacks...\n");
        customFuncO1 = tmp->customFuncO1;
        customFuncO2 = tmp->customFuncO2;
        customFuncD = tmp->customFuncD;
      }
        
    }
	
	UnaryPredicateObj_SEJITS() { // should never be called
		printf("UnaryPredicateObj_SEJITS()!!!\n");
	}

    // for each operator, first check whether a specialized function
    // exists and call the specialized version if so.
	bool operator()(const Obj2& x) const { 
      if (customFuncO2 != NULL)
        return (*customFuncO2)(x); 
      else
        return call(x);
    }

	bool operator()(const Obj1& x) const { 
      if (customFuncO1 != NULL)
        return (*customFuncO1)(x);
      else
        return call(x); 
    }

	bool operator()(const double& x) const {
      if (customFuncD != NULL)
        return (*customFuncD)(x);
      else
        return callD(x); 
    }

	public:
	~UnaryPredicateObj_SEJITS() { }
};


class UnaryFunctionObj_SEJITS : public UnaryFunctionObj_Python {
	public:
	UnaryFunctionObj_SEJITS(PyObject *pyfunc): UnaryFunctionObj_Python(pyfunc) { }

	Obj2 operator()(const Obj2& x) const { return call(x); }
	Obj1 operator()(const Obj1& x) const { return call(x); }
	double operator()(const double& x) const { return callD(x); }
	
	UnaryFunctionObj_SEJITS() { // should never be called
		printf("UnaryFunctionObj_SEJITS()!!!\n");
		callback = NULL;
	}

	public:
	~UnaryFunctionObj_SEJITS() { }
};

class BinaryPredicateObj_SEJITS : public BinaryPredicateObj_Python {
	public:
	PyObject *callback;
	BinaryPredicateObj_SEJITS(PyObject *pyfunc): BinaryPredicateObj_Python(pyfunc) { }

	
	bool operator()(const Obj1& x, const Obj1& y) const { return call(x, y); }
	bool operator()(const Obj1& x, const Obj2& y) const { return call(x, y); }
	bool operator()(const Obj2& x, const Obj2& y) const { return call(x, y); }
	bool operator()(const Obj2& x, const Obj1& y) const { return call(x, y); }

	bool operator()(const Obj1& x, const double& y) const { return callOD(x, y); }
	bool operator()(const Obj2& x, const double& y) const { return callOD(x, y); }
	bool operator()(const double& x, const Obj2& y) const { return callDO(x, y); }
	bool operator()(const double& x, const Obj1& y) const { return callDO(x, y); }

	bool operator()(const double& x, const double& y) const { return callDD(x, y); }

	BinaryPredicateObj_SEJITS() { // should never be called
		printf("BinaryPredicateObj_SEJITS()!!!\n");
	}

	public:
	~BinaryPredicateObj_SEJITS() { }
};


class BinaryFunctionObj_SEJITS : public BinaryFunctionObj_Python {
	public:
	PyObject *callback;

	BinaryFunctionObj_SEJITS(PyObject *pyfunc): BinaryFunctionObj_Python(pyfunc) { }

	
	BinaryFunctionObj_SEJITS(): callback(NULL) {}
	public:
	~BinaryFunctionObj_SEJITS() { /*Py_XDECREF(callback);*/ }
	
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

#endif
