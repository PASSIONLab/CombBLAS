#include "pyOperationsObj.h"
#include <iostream>
#include <math.h>
#include <cstdlib>
#include <Python.h>

namespace op{

/**************************\
| UNARY OPERATIONS
\**************************/


UnaryFunctionObj unaryObj(PyObject *pyfunc)
{
	return UnaryFunctionObj(pyfunc);
}


UnaryPredicateObj unaryObjPred(PyObject *pyfunc)
{
	return UnaryPredicateObj(pyfunc);
}


// Slightly un-standard ops:
#if 0
template<typename T>
struct set_s: public ConcreteUnaryFunction<T>
{
	set_s(T myvalue): value(myvalue) {};
	/** @returns value regardless of x */
	T operator()(const T& x) const
	{
		return value;
	} 
	T value;
};

UnaryFunction set(Obj2* val)
{
	return UnaryFunction(new set_s<Obj2>(Obj2(*val)));
}

UnaryFunction set(Obj1* val)
{
	return UnaryFunction(new set_s<Obj1>(Obj1(*val)));
}
#endif



/**************************\
| BINARY OPERATIONS
\**************************/


BinaryFunctionObj binaryObj(PyObject *pyfunc, bool comm)
{
	// assumed to be associative but not commutative
	return BinaryFunctionObj(pyfunc, true, comm);
}

BinaryPredicateObj binaryObjPred(PyObject *pyfunc)
{
	// assumed to be associative but not commutative
	return BinaryPredicateObj(pyfunc);
}

/**************************\
| METHODS
\**************************/
BinaryFunctionObj* BinaryFunctionObj::currentlyApplied = NULL;
MPI_Op BinaryFunctionObj::staticMPIop;
	
void BinaryFunctionObj::apply(void * invec, void * inoutvec, int * len, MPI_Datatype *datatype)
{
	if (*datatype == MPIType< doubleint >())
		applyWorker(static_cast<doubleint*>(invec), static_cast<doubleint*>(inoutvec), len);
	else if (*datatype == MPIType< Obj1 >())
		applyWorker(static_cast<Obj1*>(invec), static_cast<Obj1*>(inoutvec), len);
	else if (*datatype == MPIType< Obj2 >())
		applyWorker(static_cast<Obj2*>(invec), static_cast<Obj2*>(inoutvec), len);
	else
	{
		cout << "There is an internal error in applying a BinaryFunctionObj: Unknown datatype." << endl;
		std::exit(1);
	}
}

MPI_Op* BinaryFunctionObj::getMPIOp()
{
	//cout << "setting mpi op" << endl;
	if (currentlyApplied != NULL)
	{
		cout << "There is an internal error in creating an MPI version of a BinaryFunctionObj: Conflict between two BFOs." << endl;
		std::exit(1);
	}
	else if (currentlyApplied == this)
	{
		return &staticMPIop;
	}

	currentlyApplied = this;
	MPI_Op_create(BinaryFunctionObj::apply, commutable, &staticMPIop);
	return &staticMPIop;
}

void BinaryFunctionObj::releaseMPIOp()
{
	//cout << "free mpi op" << endl;

	if (currentlyApplied == this)
		currentlyApplied = NULL;
}

/**************************\
| SEMIRING
\**************************/

//template <>
SemiringObj* SemiringObj::currentlyApplied = NULL;

SemiringObj::SemiringObj(PyObject *add, PyObject *multiply, PyObject* left_filter_py, PyObject* right_filter_py)
	: type(CUSTOM)//, pyfunc_add(add), pyfunc_multiply(multiply), binfunc_add(&binary(add))
{
	//Py_INCREF(pyfunc_add);
	//Py_INCREF(pyfunc_multiply);
	
	binfunc_add = new BinaryFunctionObj(add, true, true);
	binfunc_mul = new BinaryFunctionObj(multiply, true, true);
	
	if (left_filter_py != NULL && Py_None != left_filter_py)
		left_filter = new UnaryPredicateObj(left_filter_py);
	else
		left_filter = NULL;

	if (right_filter_py != NULL && Py_None != right_filter_py)
		right_filter = new UnaryPredicateObj(right_filter_py);
	else
		right_filter = NULL;
}
SemiringObj::~SemiringObj()
{
	//Py_XDECREF(pyfunc_add);
	//Py_XDECREF(pyfunc_multiply);
	if (binfunc_add != NULL)
		delete binfunc_add;
	if (binfunc_mul != NULL)
		delete binfunc_mul;
	if (left_filter != NULL)
		delete left_filter;
	if (right_filter != NULL)
		delete right_filter;
	//assert(currentlyApplied != this);
}

void SemiringObj::enableSemiring()
{
	if (currentlyApplied != NULL)
	{
		cout << "There is an internal error in selecting a SemiringObj: Conflict between two Semirings." << endl;
		std::exit(1);
	}
	currentlyApplied = this;
	binfunc_add->getMPIOp();
}

void SemiringObj::disableSemiring()
{
	binfunc_add->releaseMPIOp();
	currentlyApplied = NULL;
}
/*
doubleint SemiringObj::add(const doubleint & arg1, const doubleint & arg2)
{
	PyObject *arglist;
	PyObject *result;
	double dres = 0;
	
	arglist = Py_BuildValue("(d d)", arg1.d, arg2.d);    // Build argument list
	result = PyEval_CallObject(pyfunc_add, arglist);     // Call Python
	Py_DECREF(arglist);                                  // Trash arglist
	if (result) {                                        // If no errors, return double
		dres = PyFloat_AsDouble(result);
	}
	Py_XDECREF(result);
	return doubleint(dres);
}

doubleint SemiringObj::multiply(const doubleint & arg1, const doubleint & arg2)
{
	PyObject *arglist;
	PyObject *result;
	double dres = 0;
	
	arglist = Py_BuildValue("(d d)", arg1.d, arg2.d);         // Build argument list
	result = PyEval_CallObject(pyfunc_multiply, arglist);     // Call Python
	Py_DECREF(arglist);                                       // Trash arglist
	if (result) {                                             // If no errors, return double
		dres = PyFloat_AsDouble(result);
	}
	Py_XDECREF(result);
	return doubleint(dres);
}

void SemiringObj::axpy(doubleint a, const doubleint & x, doubleint & y)
{
	y = add(y, multiply(a, x));
}

SemiringObj TimesPlusSemiringObj()
{
	return SemiringObj(SemiringObj::TIMESPLUS);
}

SemiringObj MinPlusSemiringObj()
{
	return SemiringObj(SemiringObj::PLUSMIN);
}

SemiringObj SecondMaxSemiringObj()
{
	return SemiringObj(SemiringObj::SECONDMAX);
}
*/
} // namespace op
