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
	UnaryFunctionObj_SEJITS(PyObject *pyfunc): UnaryFunctionObj_Python(pyfunc) { 
    customFunc_double_double = NULL;
    customFunc_Obj2_double = NULL;
  }

	Obj2 operator()(const Obj2& x) const { return call(x); }
	Obj1 operator()(const Obj1& x) const { return call(x); }
	double operator()(const double& x) const { 
      if (customFunc_double_double != NULL)
        return (*customFunc_double_double)(x);
      else
        return callD(x); 
    }

    // specialized functions, for each possible input type and output
    // type
    // convention: name is customFunc_<type1>_<return type>
    double (*customFunc_double_double)(const double& x);     
    double (*customFunc_Obj2_double)(const Obj2& x);     
	
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
  // FIXME: need all possible input combinations
  bool (*customFuncO1O1)(const Obj1& x, const Obj1& y); 
  bool (*customFuncO2O2)(const Obj2& x, const Obj2& y); 
  bool (*customFuncDD)(const double& x, const double& y);

	BinaryPredicateObj_SEJITS(PyObject *pyfunc): BinaryPredicateObj_Python(pyfunc) { 

      // set specialized function pointers to NULL.
      // specializer code will replace them
      customFuncO1O1 = NULL; 
      customFuncO2O2 = NULL;  
      customFuncDD = NULL;

      // now we check if the PyObject is actually a UnaryPredicateObj
      // in disguise
      swig_module_info* module = SWIG_Python_GetModule();
      swig_type_info* ty = SWIG_TypeQueryModule(module, module, "op::BinaryPredicateObj *");

      BinaryPredicateObj_SEJITS* tmp;

      if (module != NULL && ty != NULL && (SWIG_ConvertPtr(pyfunc, (void**)&tmp, ty, 0)) == 0) {
	printf("BinaryPredicateObj_SEJITS detected, replicating callbacks...\n");
        customFuncO1O1 = tmp->customFuncO1O1;
        customFuncO2O2 = tmp->customFuncO2O2;
        customFuncDD = tmp->customFuncDD;
      }
        
    }
	
	bool operator()(const Obj1& x, const Obj1& y) const { 
    if (customFuncO1O1 != NULL)
      return (*customFuncO1O1)(x, y);
    else
      return call(x, y); 
  }

	bool operator()(const Obj1& x, const Obj2& y) const { return call(x, y); }
	bool operator()(const Obj2& x, const Obj1& y) const { return call(x, y); }
	bool operator()(const Obj2& x, const Obj2& y) const { 
    if (customFuncO2O2 != NULL)
      return (*customFuncO2O2)(x, y);
    else
      return call(x, y); 
  }

	bool operator()(const Obj1& x, const double& y) const { return callOD(x, y); }
	bool operator()(const Obj2& x, const double& y) const { return callOD(x, y); }
	bool operator()(const double& x, const Obj2& y) const { return callDO(x, y); }
	bool operator()(const double& x, const Obj1& y) const { return callDO(x, y); }

	bool operator()(const double& x, const double& y) const { 
    if (customFuncDD != NULL)
      { /*printf("using customfunc\n");*/ return (*customFuncDD)(x, y); }
    else
      { /*printf("using python callback\n");*/ return callDD(x, y); }
  }

	BinaryPredicateObj_SEJITS() { // should never be called
		printf("BinaryPredicateObj_SEJITS()!!!\n");
	}

	public:
	~BinaryPredicateObj_SEJITS() { }
};


class BinaryFunctionObj_SEJITS : public BinaryFunctionObj_Python {
	public:
	PyObject *callback;

    // specialized functions, for each possible input type and output
    // type
    // convention: name is customFunc_<type1><type2>_<return type>
    double (*customFunc_doubledouble_double)(const double& x, const double& y);     
    double (*customFunc_Obj2double_double)(const Obj2& x, const double& y);     
    Obj2 (*customFunc_Obj2double_Obj2)(const Obj2& x, const double& y);

	BinaryFunctionObj_SEJITS(PyObject *pyfunc): BinaryFunctionObj_Python(pyfunc) { 
      customFunc_doubledouble_double = NULL;
      customFunc_Obj2double_double = NULL;
      customFunc_Obj2double_Obj2 = NULL;
      /*
      if (pyfunc != Py_None) {
        // now we check if the PyObject is actually a BinaryFunctionObj
        // in disguise
        swig_module_info* module = SWIG_Python_GetModule();
        swig_type_info* ty = SWIG_TypeQueryModule(module, module, "op::BinaryFunctionObj_SEJITS *");

        BinaryFunctionObj_SEJITS* tmp;

        if ((SWIG_ConvertPtr(callback, (void**)&tmp, ty, 0)) == 0) {
          // yes, it is a BinaryFunctionObj
          printf("UnaryPredicateObj detected, replicating customized callbacks...\n");
          customFunc_doubledouble_double = tmp->customFunc_doubledouble_double;
          customFunc_Obj2double_double = tmp->customFunc_Obj2double_double;
          customFunc_Obj2double_Obj2 = tmp->customFunc_Obj2double_Obj2;
        }
      }
      */
      if (pyfunc==NULL)
        printf("WTF GETTING PASSED A NULL!!!!");

    }

	
	BinaryFunctionObj_SEJITS(): callback(NULL) {}
	public:
	~BinaryFunctionObj_SEJITS() { /*Py_XDECREF(callback);*/ }
	
	PyObject* getCallback() const { return callback; }
	
	Obj1 operator()(const Obj1& x, const Obj1& y) const { return call<Obj1>(x, y); }
	Obj2 operator()(const Obj2& x, const Obj2& y) const { return call<Obj2>(x, y); }
	Obj1 operator()(const Obj1& x, const Obj2& y) const { return call<Obj1>(x, y); }
	Obj2 operator()(const Obj2& x, const Obj1& y) const { return call<Obj2>(x, y); }

	Obj1 operator()(const Obj1& x, const double& y) const { return callOD_retO<Obj1>(x, y); }
	Obj2 operator()(const Obj2& x, const double& y) const { 
      /*      printf("O2,d,retO2\n"); 
      if (customFunc_Obj2double_double != NULL)
        printf("    specialized for double f(Obj2, double)\n");
      if (customFunc_doubledouble_double != NULL)
        printf("    specialized for double f(double, double)\n");
      */
      if (customFunc_Obj2double_Obj2 != NULL)
        return (*customFunc_Obj2double_Obj2)(x,y);
      else
        return callOD_retO<Obj2>(x, y); 

    }
	double operator()(const double& x, const Obj1& y) const { return callDO_retD(x, y); }
	double operator()(const double& x, const Obj2& y) const { return callDO_retD(x, y); }

	double operator()(const double& x, const double& y) const { 
      /*printf("d,d,d\n");
      if (customFunc_Obj2double_double != NULL)
        printf("   specialized for double f(Obj2, double)\n");
      if (customFunc_Obj2double_Obj2 != NULL)
        printf("   specialized for Obj2 f(Obj2, double)\n");
      if (callback == Py_None)
        printf("   callback is NONE!\n");
      */
      if (customFunc_doubledouble_double != NULL)
        return (*customFunc_doubledouble_double)(x,y);
      else
        return callDD(x, y); 
    }


	// These are used by the semiring ops. They do the same thing as the operator() above,
	// but their return type matches the 2nd argument instead of the 1st.
	Obj1 rettype2nd_call(const Obj1& x, const Obj1& y) const { return call<Obj1>(x, y); }
	Obj2 rettype2nd_call(const Obj2& x, const Obj2& y) const { return call<Obj2>(x, y); }
	Obj1 rettype2nd_call(const Obj2& x, const Obj1& y) const { return call<Obj1>(x, y); }
	Obj2 rettype2nd_call(const Obj1& x, const Obj2& y) const { return call<Obj2>(x, y); }

	double rettype2nd_call(const Obj1& x, const double& y) const { printf("O1,d\n"); return callOD_retD(x, y); }
	double rettype2nd_call(const Obj2& x, const double& y) const { 
      //      printf("O2, d, retd\n");
      if (customFunc_Obj2double_double != NULL)
        {
          //  printf("using customFunc\n");
          return (*customFunc_Obj2double_double)(x, y);
        }
      else {
        //printf("using interpretation\n");
        return callOD_retD(x, y); 
      }
    }
	Obj1 rettype2nd_call(const double& x, const Obj1& y) const { return callDO_retO<Obj1>(x, y); }
	Obj2 rettype2nd_call(const double& x, const Obj2& y) const { return callDO_retO<Obj2>(x, y); }

	double rettype2nd_call(const double& x, const double& y) const { return operator()(x, y); }

};

#endif
