#ifndef PYOPERATIONSSEJITS_H
#define PYOPERATIONSSEJITS_H

#include "pyCombBLAS-NoMPI.h"
#include "pyOperationsWorkers.h"

class UnaryPredicateObj_SEJITS : public UnaryPredicateObj_Python {
	public:

    // specialized functions, for each possible input type
    bool (*customFuncObj1)(const Obj1& x);
    bool (*customFuncObj2)(const Obj2& x);
    bool (*customFuncdouble)(const double& x);

	UnaryPredicateObj_SEJITS(PyObject *pyfunc): UnaryPredicateObj_Python(pyfunc) {
      // set specialized function pointers to NULL.
      // specializer code will replace them
      customFuncObj1 = NULL;
      customFuncObj2 = NULL;
      customFuncdouble = NULL;

      // now we check if the PyObject is actually a UnaryPredicateObj
      // in disguise
      swig_module_info* module = SWIG_Python_GetModule(NULL);
      swig_type_info* ty = SWIG_TypeQueryModule(module, module, "op::UnaryPredicateObj *");

      UnaryPredicateObj_SEJITS* tmp;

      if ((SWIG_ConvertPtr(callback, (void**)&tmp, ty, 0)) == 0) {
        // yes, it is a UnaryPredicateObj
        //printf("UnaryPredicateObj detected, replicating customized callbacks...\n");
        customFuncObj1 = tmp->customFuncObj1;
        customFuncObj2 = tmp->customFuncObj2;
        customFuncdouble = tmp->customFuncdouble;
      }

    }

	UnaryPredicateObj_SEJITS() {
    // set specialized function pointers to NULL.
    // specializer code will replace them
    customFuncObj1 = NULL;
    customFuncObj2 = NULL;
    customFuncdouble = NULL;
		callback = NULL;
	}

    // for each operator, first check whether a specialized function
    // exists and call the specialized version if so.
	bool operator()(const Obj2& x) const {
      if (customFuncObj2 != NULL)
        return (*customFuncObj2)(x);
      else
        return call(x);
    }

	bool operator()(const Obj1& x) const {
      if (customFuncObj1 != NULL)
        return (*customFuncObj1)(x);
      else
        return call(x);
    }

	bool operator()(const double& x) const {
      if (customFuncdouble != NULL)
        return (*customFuncdouble)(x);
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
        customFunc_Obj1_double = NULL;
        customFunc_Obj2_double = NULL;
        customFunc_double_Obj1 = NULL;
        customFunc_Obj1_Obj1 = NULL;
        customFunc_Obj2_Obj1 = NULL;
        customFunc_double_Obj2 = NULL;
        customFunc_Obj1_Obj2 = NULL;
        customFunc_Obj2_Obj2 = NULL;
    
  }

	Obj2 operator()(const Obj2& x) const { 
    if (customFunc_Obj2_Obj2 != NULL)
      return (*customFunc_Obj2_Obj2)(x);
    else
      return call(x); 
  }

	Obj1 operator()(const Obj1& x) const { 
    if (customFunc_Obj1_Obj1 != NULL)
      return (*customFunc_Obj1_Obj1)(x);
    else
      return call(x); 
  }

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
    double (*customFunc_Obj1_double)(const Obj1& x);
    double (*customFunc_Obj2_double)(const Obj2& x);
    Obj1 (*customFunc_double_Obj1)(const double& x);
    Obj1 (*customFunc_Obj1_Obj1)(const Obj1& x);
    Obj1 (*customFunc_Obj2_Obj1)(const Obj2& x);
    Obj2 (*customFunc_double_Obj2)(const double& x);
    Obj2 (*customFunc_Obj1_Obj2)(const Obj1& x);
    Obj2 (*customFunc_Obj2_Obj2)(const Obj2& x);

	UnaryFunctionObj_SEJITS() {
		callback = NULL;
	}



	public:
	~UnaryFunctionObj_SEJITS() { }
};

class BinaryPredicateObj_SEJITS : public BinaryPredicateObj_Python {
	public:
	PyObject *callback;

  bool (*customFuncObj1Obj1)(const Obj1& x, const Obj1& y);
  bool (*customFuncObj1Obj2)(const Obj1& x, const Obj2& y);
  bool (*customFuncObj1double)(const Obj1& x, const double& y);
  bool (*customFuncObj2Obj2)(const Obj2& x, const Obj2& y);
  bool (*customFuncObj2Obj1)(const Obj2& x, const Obj1& y);
  bool (*customFuncObj2double)(const Obj2& x, const double& y);
  bool (*customFuncdoubledouble)(const double& x, const double& y);
  bool (*customFuncdoubleObj1)(const double& x, const Obj1& y);
  bool (*customFuncdoubleObj2)(const double& x, const Obj2& y);

	BinaryPredicateObj_SEJITS(PyObject *pyfunc): BinaryPredicateObj_Python(pyfunc) {

      // set specialized function pointers to NULL.
      // specializer code will replace them
      customFuncObj1Obj1 = NULL;
      customFuncObj1Obj2 = NULL;
      customFuncObj1double = NULL;
      customFuncObj2Obj2 = NULL;
      customFuncObj2Obj1 = NULL;
      customFuncObj2double = NULL;
      customFuncdoubledouble = NULL;
      customFuncdoubleObj1 = NULL;
      customFuncdoubleObj2 = NULL;


      // now we check if the PyObject is actually a UnaryPredicateObj
      // in disguise
      swig_module_info* module = SWIG_Python_GetModule(NULL);
      swig_type_info* ty = SWIG_TypeQueryModule(module, module, "op::BinaryPredicateObj *");

      BinaryPredicateObj_SEJITS* tmp;

      if (module != NULL && ty != NULL && (SWIG_ConvertPtr(pyfunc, (void**)&tmp, ty, 0)) == 0) {
        printf("BinaryPredicateObj_SEJITS detected, replicating callbacks...\n");

        customFuncObj1Obj1     = tmp->customFuncObj1Obj1    ;
        customFuncObj1Obj2     = tmp->customFuncObj1Obj2    ;
        customFuncObj1double   = tmp->customFuncObj1double  ;
        customFuncObj2Obj2     = tmp->customFuncObj2Obj2    ;
        customFuncObj2Obj1     = tmp->customFuncObj2Obj1    ;
        customFuncObj2double   = tmp->customFuncObj2double  ;
        customFuncdoubledouble = tmp->customFuncdoubledouble;
        customFuncdoubleObj1   = tmp->customFuncdoubleObj1  ;
        customFuncdoubleObj2   = tmp->customFuncdoubleObj2  ;

      }

    }

	bool operator()(const Obj1& x, const Obj1& y) const {
    if (customFuncObj1Obj1 != NULL)
      return (*customFuncObj1Obj1)(x, y);
    else
      return call(x, y);
  }

	bool operator()(const Obj1& x, const Obj2& y) const { 
    if (customFuncObj1Obj2 != NULL)
      return (*customFuncObj1Obj2)(x, y);
    else
      return call(x, y); 
  }

	bool operator()(const Obj2& x, const Obj1& y) const { 
    if (customFuncObj2Obj1 != NULL)
      return (*customFuncObj2Obj1)(x, y);
    else
      return call(x, y); 
  }

	bool operator()(const Obj2& x, const Obj2& y) const {
    if (customFuncObj2Obj2 != NULL)
      return (*customFuncObj2Obj2)(x, y);
    else
      return call(x, y);
  }

	bool operator()(const Obj1& x, const double& y) const { 
    if (customFuncObj1double != NULL)
      return (*customFuncObj1double)(x, y);
    else
      return callOD(x, y); 
  }

	bool operator()(const Obj2& x, const double& y) const { 
    if (customFuncObj2double != NULL)
      return (*customFuncObj2double)(x, y);
    else
      return callOD(x, y); 
  }

	bool operator()(const double& x, const Obj2& y) const { 
    if (customFuncdoubleObj2 != NULL)
      return (*customFuncdoubleObj2)(x, y);
    else
      return callDO(x, y); 
  }

	bool operator()(const double& x, const Obj1& y) const { 
    if (customFuncdoubleObj1 != NULL)
      return (*customFuncdoubleObj1)(x, y);
    else
      return callDO(x, y); 
  }

	bool operator()(const double& x, const double& y) const {
    if (customFuncdoubledouble != NULL)
      { /*printf("using customfunc\n");*/ return (*customFuncdoubledouble)(x, y); }
    else
      { /*printf("using python callback\n");*/ return callDD(x, y); }
  }

	BinaryPredicateObj_SEJITS() {
    // set specialized function pointers to NULL.
    // specializer code will replace them
    customFuncObj1Obj1 = NULL;
    customFuncObj1Obj2 = NULL;
    customFuncObj1double = NULL;
    customFuncObj2Obj2 = NULL;
    customFuncObj2Obj1 = NULL;
    customFuncObj2double = NULL;
    customFuncdoubledouble = NULL;
    customFuncdoubleObj1 = NULL;
    customFuncdoubleObj2 = NULL;
		callback = NULL;
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

    // return type matches first input
    double (*customFunc_doubledouble_double)(const double& x, const double& y);
    double (*customFunc_doubleObj1_double)(const double& x, const Obj1& y);
    double (*customFunc_doubleObj2_double)(const double& x, const Obj2& y);
    Obj1 (*customFunc_Obj1double_Obj1)(const Obj1& x, const double& y);
    Obj1 (*customFunc_Obj1Obj1_Obj1)(const Obj1& x, const Obj1& y);
    Obj1 (*customFunc_Obj1Obj2_Obj1)(const Obj1& x, const Obj2& y);
    Obj2 (*customFunc_Obj2double_Obj2)(const Obj2& x, const double& y);
    Obj2 (*customFunc_Obj2Obj1_Obj2)(const Obj2& x, const Obj1& y);
    Obj2 (*customFunc_Obj2Obj2_Obj2)(const Obj2& x, const Obj2& y);
    // return type matches second input
    Obj1 (*customFunc_doubleObj1_Obj1)(const double& x, const Obj1& y);
    Obj2 (*customFunc_doubleObj2_Obj2)(const double& x, const Obj2& y);
    double (*customFunc_Obj1double_double)(const Obj1& x, const double& y);
    Obj2 (*customFunc_Obj1Obj2_Obj2)(const Obj1& x, const Obj2& y);
    double (*customFunc_Obj2double_double)(const Obj2& x, const double& y);
    Obj1 (*customFunc_Obj2Obj1_Obj1)(const Obj2& x, const Obj1& y);

	BinaryFunctionObj_SEJITS(PyObject *pyfunc): BinaryFunctionObj_Python(pyfunc) {
    customFunc_doubledouble_double = NULL;
    customFunc_doubleObj1_double = NULL;
    customFunc_doubleObj2_double = NULL;
    customFunc_Obj1double_Obj1 = NULL;
    customFunc_Obj1Obj1_Obj1 = NULL;
    customFunc_Obj1Obj2_Obj1 = NULL;
    customFunc_Obj2double_Obj2 = NULL;
    customFunc_Obj2Obj1_Obj2 = NULL;
    customFunc_Obj2Obj2_Obj2 = NULL;
    customFunc_doubleObj1_Obj1 = NULL;
    customFunc_doubleObj2_Obj2 = NULL;
    customFunc_Obj1double_double = NULL;
    customFunc_Obj1Obj2_Obj2 = NULL;
    customFunc_Obj2double_double = NULL;
    customFunc_Obj2Obj1_Obj1 = NULL;
      

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

	//PyObject* getCallback() const { return callback; }

	Obj1 operator()(const Obj1& x, const Obj1& y) const {
    if (customFunc_Obj1Obj1_Obj1 != NULL)
      return (*customFunc_Obj1Obj1_Obj1)(x, y);
    else
      return call<Obj1>(x, y);
  }

	Obj2 operator()(const Obj2& x, const Obj2& y) const {
    if (customFunc_Obj2Obj2_Obj2 != NULL)
      return (*customFunc_Obj2Obj2_Obj2)(x, y);
    else
      return call<Obj2>(x, y); 
  }

	Obj1 operator()(const Obj1& x, const Obj2& y) const {
    if (customFunc_Obj1Obj2_Obj1 != NULL)
      return (*customFunc_Obj1Obj2_Obj1)(x, y);
    else
      return call<Obj1>(x, y); 
  }

	Obj2 operator()(const Obj2& x, const Obj1& y) const {
    if (customFunc_Obj2Obj1_Obj2 != NULL)
      return (*customFunc_Obj2Obj1_Obj2)(x, y);
    else
      return call<Obj2>(x, y); 
  }


	Obj1 operator()(const Obj1& x, const double& y) const {
    if (customFunc_Obj1double_Obj1 != NULL)
      return (*customFunc_Obj1double_Obj1)(x, y);
    else
      return callOD_retO<Obj1>(x, y); 
  }

	Obj2 operator()(const Obj2& x, const double& y) const {
      /*      printf("Obj2,d,retObj2\n");
      if (customFunc_Obj2double_double != NULL)
        printf("    specialized for double f(Obj2, double)\n");
      if (customFunc_doubledouble_double != NULL)
        printf("    specialized for double f(double, double)\n");
      */
      if (customFunc_Obj2double_Obj2 != NULL) {
          printf("+++++ doing call to specialized version\n");
        return (*customFunc_Obj2double_Obj2)(x,y);
        
        }
      else {
          printf("+++++ doing call to non-specialized version\n");
        return callOD_retO<Obj2>(x, y);
        
        }
    }
	double operator()(const double& x, const Obj1& y) const {
    if (customFunc_doubleObj1_double != NULL)
      return (*customFunc_doubleObj1_double)(x, y);
    else
      return callDO_retD(x, y); 
  }

	double operator()(const double& x, const Obj2& y) const {
	    if (customFunc_doubleObj2_double != NULL)
		return (*customFunc_doubleObj2_double)(x,y);
	    else
		return callDO_retD(x, y);
	}

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
	Obj1 rettype2nd_call(const Obj1& x, const Obj1& y) const {
    if (customFunc_Obj1Obj1_Obj1 != NULL)
      return (*customFunc_Obj1Obj1_Obj1)(x, y);
    else
      return call<Obj1>(x, y); 
  }

	Obj2 rettype2nd_call(const Obj2& x, const Obj2& y) const {
    if (customFunc_Obj2Obj2_Obj2 != NULL)
      return (*customFunc_Obj2Obj2_Obj2)(x, y);
    else
      return call<Obj2>(x, y); 
  }

	Obj1 rettype2nd_call(const Obj2& x, const Obj1& y) const {
    if (customFunc_Obj2Obj1_Obj1 != NULL)
      return (*customFunc_Obj2Obj1_Obj1)(x, y);
    else
      return call<Obj1>(x, y); 
  }

	Obj2 rettype2nd_call(const Obj1& x, const Obj2& y) const {
    if (customFunc_Obj1Obj2_Obj2 != NULL)
      return (*customFunc_Obj1Obj2_Obj2)(x, y);
    else
      return call<Obj2>(x, y); 
  }


	double rettype2nd_call(const Obj1& x, const double& y) const {
    if (customFunc_Obj1double_double != NULL)
      return (*customFunc_Obj1double_double)(x, y);
    else
      return callOD_retD(x, y); 
  }

	double rettype2nd_call(const Obj2& x, const double& y) const {
            printf("Obj2, d, retd\n");
      if (customFunc_Obj2double_double != NULL)
        {
            printf("using customFunc\n");
          return (*customFunc_Obj2double_double)(x, y);
        }
      else {
        printf("using interpretation\n");
        return callOD_retD(x, y);
      }
    }
	Obj1 rettype2nd_call(const double& x, const Obj1& y) const {
    if (customFunc_doubleObj1_Obj1 != NULL)
      return (*customFunc_doubleObj1_Obj1)(x, y);
    else
      return callDO_retO<Obj1>(x, y); 
  }

	Obj2 rettype2nd_call(const double& x, const Obj2& y) const {
    if (customFunc_doubleObj2_Obj2 != NULL)
      return (*customFunc_doubleObj2_Obj2)(x, y);
    else
      return callDO_retO<Obj2>(x, y); 
  }


	double rettype2nd_call(const double& x, const double& y) const {
    if (customFunc_doubledouble_double != NULL)
      return (*customFunc_doubledouble_double)(x, y);
    else
      return operator()(x, y); 
  }


};

#endif
