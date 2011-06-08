#include "pyOperations.h"
#include <iostream>
#include <math.h>
#include <cstdlib>
#include <Python.h>

namespace op{

/**************************\
| UNARY OPERATIONS
\**************************/


#define DECL_UNARY_STRUCT(structname, operation) 						\
	template<typename T>												\
	struct structname : public ConcreteUnaryFunction<T>					\
	{																	\
		T operator()(const T& x) const									\
		{																\
			operation;													\
		}																\
	};
	
#define DECL_UNARY_FUNC(structname, name, operation)					\
	DECL_UNARY_STRUCT(structname, operation)							\
	UnaryFunction name()												\
	{																	\
		return UnaryFunction(new structname<doubleint>());			\
	}																

DECL_UNARY_FUNC(identity_s, identity, return x;)
DECL_UNARY_FUNC(negate_s, negate, return -static_cast<doubleint>(x);)
DECL_UNARY_FUNC(bitwise_not_s, bitwise_not, return ~x;)
DECL_UNARY_FUNC(logical_not_s, logical_not, return !x;)
DECL_UNARY_FUNC(abs_s, abs, return (x < 0) ? -static_cast<doubleint>(x) : x;)
DECL_UNARY_FUNC(totality_s, totality, return 1;)


// Slightly un-standard ops:

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

UnaryFunction set(double val)
{
	return UnaryFunction(new set_s<doubleint>(doubleint(val)));
}

template<typename T>
struct safemultinv_s : public ConcreteUnaryFunction<T>
{
	T operator()(const T& x) const
	{
		T inf = std::numeric_limits<T>::max();
		return (x == 0) ? inf:(1/x);
	}
};

UnaryFunction safemultinv() {
	return UnaryFunction(new safemultinv_s<doubleint>());
}

//// ifthenelse

template<typename T>
struct ifthenelse_s: public ConcreteUnaryFunction<T>
{
	ConcreteUnaryFunction<T> *predicate, *runTrue, *runFalse;

	ifthenelse_s(ConcreteUnaryFunction<T> *pred, ConcreteUnaryFunction<T> *t, ConcreteUnaryFunction<T> *f): predicate(pred), runTrue(t), runFalse(f) {};

	T operator()(const T& x) const
	{
		if ((*predicate)(x))
			return (*runTrue)(x);
		else
			return (*runFalse)(x);
	} 
};

UnaryFunction ifthenelse(UnaryFunction& predicate, UnaryFunction& runTrue, UnaryFunction& runFalse)
{
	return UnaryFunction(new ifthenelse_s<doubleint>(predicate.op, runTrue.op, runFalse.op));
}

//// Custom Python callback
template<typename T>
struct unary_s: public ConcreteUnaryFunction<T>
{
	PyObject *pyfunc;

	unary_s(PyObject *pyfunc_in): pyfunc(pyfunc_in)
	{
		Py_INCREF(pyfunc);
	}
	
	~unary_s()
	{
		Py_DECREF(pyfunc);
	}
	
	T operator()(const T& x) const
	{
		PyObject *arglist;
		PyObject *result;
		double dres = 0;
		
		arglist = Py_BuildValue("(d)", static_cast<double>(x));  // Build argument list
		result = PyEval_CallObject(pyfunc,arglist);              // Call Python
		Py_DECREF(arglist);                                      // Trash arglist
		if (result) {                                            // If no errors, return double
			dres = PyFloat_AsDouble(result);
		}
		Py_XDECREF(result);
		return T(dres);
	} 
};

UnaryFunction unary(PyObject *pyfunc)
{
	return UnaryFunction(new unary_s<doubleint>(pyfunc));
}

////////////////////////////////////////////////
//// Custom Python callback
/*
template<>
struct obj_unary_s: public ConcreteUnaryFunction<PyObject*>
{
	PyObject *pyfunc;
	PyObject *arglist;

	obj_unary_s(PyObject *pyfunc_in): pyfunc(pyfunc_in)
	{
		Py_INCREF(pyfunc);
		arglist = Py_BuildValue("(d)", static_cast<double>(0));  // Build argument list
	}
	
	~obj_unary_s()
	{
		Py_DECREF(arglist);                                      // Trash arglist
		Py_DECREF(pyfunc);
	}
	
	PyObject* operator()(const PyObject*& x) const
	{
		PyObject *arglist;
		PyObject *result;
		double dres = 0;
		
		PyList_SetItem(arglist, 0, x);                           // Set argument
		result = PyEval_CallObject(pyfunc,arglist);              // Call Python
		return result;
	} 
};*/

ObjUnaryFunction::ObjUnaryFunction(PyObject *pyfunc_in): pyfunc(pyfunc_in)
{
	Py_INCREF(pyfunc);
	//arglist = Py_BuildValue("(d)", 1.0);                   // Build argument list
	arglist = PyTuple_New(1);
}

ObjUnaryFunction::~ObjUnaryFunction()
{
	Py_XDECREF(arglist);                                      // Trash arglist
	Py_XDECREF(pyfunc);
}

PyObject* ObjUnaryFunction::operator()(PyObject* x)
{
	//PyObject *arglist;
	PyObject *result;
	
	//arglist = Py_BuildValue("(O)", x);                   // Build argument list
	//Py_INCREF(x);
	//PyList_SetItem(arglist, 0, x);                           // Set argument
	//PyObject* o = PyTuple_GetItem(arglist, 0);
	//cout << "getitem returns: " << o << endl;
	//cout << "PyTuple_Check returns " << PyTuple_Check(arglist) << endl;
	//PyObject* one = Py_BuildValue("d", 1.0);
	//cout << "set tuple returns " << PyTuple_SetItem(arglist, 0, x) << endl;                           // Set argument
	//int tsize = PyTuple_Size(arglist);
	//cout << "size of arglist: " << tsize << endl;
	//result = PyEval_CallObject(pyfunc, arglist);              // Call Python
	//Py_XDECREF(arglist);                                      // Trash arglist


	PyObject *arglist;
	arglist = Py_BuildValue("(O)", x);                   // Build argument list
	result = PyEval_CallObject(pyfunc, arglist);              // Call Python
	Py_XDECREF(arglist);                                      // Trash arglist

	Py_XDECREF(result);
	return result;
} 

ObjUnaryFunction obj_unary(PyObject *pyfunc)
{
	return ObjUnaryFunction(pyfunc);
}
/////////////////////////////////////////////////////




/**************************\
| BINARY OPERATIONS
\**************************/

#define DECL_BINARY_STRUCT(name, operation) 							\
	template<typename T>												\
	struct name : public ConcreteBinaryFunction<T>						\
	{																	\
		T operator()(const T& x, const T& y) const						\
		{																\
			return operation;											\
		}																\
	};
	
#define DECL_BINARY_FUNC(structname, name, as, com, operation)			\
	DECL_BINARY_STRUCT(structname, operation)							\
	BinaryFunction name()												\
	{																	\
		return BinaryFunction(new structname<doubleint>(), as, com);	\
	}																


/*
arguments to DECL_BINARY_FUNC are:
name of implementing structure (not seen by users),
name of operation,
whether it's associative,
whether it's commutative,
implementation code w.r.t. arguments x (left) and y (right)
*/
DECL_BINARY_FUNC(plus_s, plus, true, true, x+y)
DECL_BINARY_FUNC(minus_s, minus, false, false, x-y)
DECL_BINARY_FUNC(multiplies_s, multiplies, true, true, x*y)
DECL_BINARY_FUNC(divides_s, divides, true, false, x/y)
DECL_BINARY_FUNC(modulus_s, modulus, false, false, (double(x)) - (double(y))*std::floor((double(x))/(double(y))) )
DECL_BINARY_FUNC(fmod_s, fmod, false, false, std::fmod(double(x), double(y)))
DECL_BINARY_FUNC(pow_s, pow, true, false, doubleint(::pow(static_cast<double>(x), static_cast<double>(y))) )

DECL_BINARY_FUNC(max_s, max, true, true, std::max<doubleint>(x, y))
DECL_BINARY_FUNC(min_s, min, true, true, std::min<doubleint>(x, y))

DECL_BINARY_FUNC(bitwise_and_s, bitwise_and, true, true, x & y)
DECL_BINARY_FUNC(bitwise_or_s, bitwise_or, true, true, x | y)
DECL_BINARY_FUNC(bitwise_xor_s, bitwise_xor, true, true, x ^ y)

DECL_BINARY_FUNC(logical_and_s, logical_and, true, true, x && y)
DECL_BINARY_FUNC(logical_or_s, logical_or, true, true, x || y)
DECL_BINARY_FUNC(logical_xor_s, logical_xor, true, true, (x || y) && !(x && y))

DECL_BINARY_FUNC(equal_to_s, equal_to, true, true, x == y)
// not sure about the associativity of these
DECL_BINARY_FUNC(not_equal_to_s, not_equal_to, false, true, x != y)
DECL_BINARY_FUNC(greater_s, greater, false, false, x > y)
DECL_BINARY_FUNC(less_s, less, false, false, x < y)
DECL_BINARY_FUNC(greater_equal_s, greater_equal, false, false, x >= y)
DECL_BINARY_FUNC(less_equal_s, less_equal, false, false, x <= y)

//// Custom Python callback
template<typename T>
struct binary_s: public ConcreteBinaryFunction<T>
{
	PyObject *pyfunc;

	binary_s(PyObject *pyfunc_in): pyfunc(pyfunc_in)
	{
		Py_INCREF(pyfunc);
	}
	
	~binary_s()
	{
		Py_DECREF(pyfunc);
	}
	
	T operator()(const T& x, const T& y) const
	{
		PyObject *arglist;
		PyObject *result;
		double dres = 0;
		
		arglist = Py_BuildValue("(d d)", static_cast<double>(x), static_cast<double>(y));    // Build argument list
		result = PyEval_CallObject(pyfunc,arglist);     // Call Python
		Py_DECREF(arglist);                             // Trash arglist
		if (result) {                                   // If no errors, return double
			dres = PyFloat_AsDouble(result);
		}
		Py_XDECREF(result);
		return T(dres);
	} 
};

BinaryFunction binary(PyObject *pyfunc)
{
	// assumed to be associative but not commutative
	return BinaryFunction(new binary_s<doubleint>(pyfunc), true, false);
}

/**************************\
| GLUE OPERATIONS
\**************************/

// BIND
//////////////////////////////////

template<typename T>
struct bind_s : public ConcreteUnaryFunction<T>
{
	int which;
	T bindval;
	ConcreteBinaryFunction<T>* op;
	
	bind_s(ConcreteBinaryFunction<T>* opin, int w, T val): which(w), bindval(val), op(opin) {}
	
	T operator()(const T& x) const
	{
		if (which == 1)
			return (*op)(bindval, x);
		else
			return (*op)(x, bindval);
	}
};

UnaryFunction bind1st(BinaryFunction& op, double val)
{
	return UnaryFunction(new bind_s<doubleint>(op.op, 1, doubleint(val)));
}

UnaryFunction bind2nd(BinaryFunction& op, double val)
{
	return UnaryFunction(new bind_s<doubleint>(op.op, 2, doubleint(val)));
}

// COMPOSE
//////////////////////////////////

// for some reason the regular STL compose1() cannot be found, so doing this manually
template<typename T>
struct compose1_s : public ConcreteUnaryFunction<T>
{
	ConcreteUnaryFunction<T> *f, *g;
	
	compose1_s(ConcreteUnaryFunction<T>* fin, ConcreteUnaryFunction<T>* gin): f(fin), g(gin) {}
	
	T operator()(const T& x) const
	{
		return (*f)((*g)(x));
	}
};

UnaryFunction compose1(UnaryFunction& f, UnaryFunction& g) // h(x) is the same as f(g(x))
{
	//return new UnaryFunction(compose1(f->op, g->op));
	return UnaryFunction(new compose1_s<doubleint>(f.op, g.op));
}


template<typename T>
struct compose2_s : public ConcreteUnaryFunction<T>
{
	ConcreteBinaryFunction<T> *f;
	ConcreteUnaryFunction<T> *g1, *g2;
	
	compose2_s(ConcreteBinaryFunction<T>* fin, ConcreteUnaryFunction<T>* g1in, ConcreteUnaryFunction<T>* g2in): f(fin), g1(g1in), g2(g2in) {}
	
	T operator()(const T& x) const
	{
		return (*f)( (*g1)(x), (*g2)(x) );
	}
};

UnaryFunction compose2(BinaryFunction& f, UnaryFunction& g1, UnaryFunction& g2) // h(x) is the same as f(g1(x), g2(x))
{
	//return new BinaryFunction(compose2(f->op, g1->op, g2->op));
	return UnaryFunction(new compose2_s<doubleint>(f.op, g1.op, g2.op));
}

// NOT
//////////////////////////////////

// Standard STL not1() returns a predicate object, not a unary_function.
// We may want to do it that way as well, but to avoid type problems we keep 'not' as a plain unary function for now.
template<typename T>
struct unary_not_s : public ConcreteUnaryFunction<T>
{
	ConcreteUnaryFunction<T>* op;

	unary_not_s(ConcreteUnaryFunction<T>* operation): op(operation) {}

	T operator()(const T& x) const
	{
		return !(*op)(x);
	}
};

UnaryFunction not1(UnaryFunction& f)
{
	return UnaryFunction(new unary_not_s<doubleint>(f.op));
}


template<typename T>
struct binary_not_s : public ConcreteBinaryFunction<T>
{
	ConcreteBinaryFunction<T>* op;

	binary_not_s(ConcreteBinaryFunction<T>* operation): op(operation) {}

	T operator()(const T& x, const T& y) const
	{
		return !(*op)(x, y);
	}
};

BinaryFunction not2(BinaryFunction& f)
{
	return BinaryFunction(new binary_not_s<doubleint>(f.op), f.associative, f.commutable);
}

/**************************\
| METHODS
\**************************/
BinaryFunction* BinaryFunction::currentlyApplied = NULL;
MPI_Op BinaryFunction::staticMPIop;
	
void BinaryFunction::apply(void * invec, void * inoutvec, int * len, MPI_Datatype *datatype)
{
	doubleint* in = (doubleint*)invec;
	doubleint* inout = (doubleint*)inoutvec;
	
	for (int i = 0; i < *len; i++)
	{
		if (in[i].is_nan() && inout[i].is_nan())
			inout[i] = inout[i];	// both NaN, return NaN
		else if (in[i].is_nan())
			inout[i] = inout[i];	// LHS is NaN, return RHS
		else if (inout[i].is_nan())
			inout[i] = in[i];		//  RHS is Nan, return LHS
		else
			inout[i] = (*currentlyApplied)(in[i], inout[i]);
	}
}

MPI_Op* BinaryFunction::getMPIOp()
{
	//cout << "setting mpi op" << endl;
	if (currentlyApplied != NULL)
	{
		cout << "There is an internal error in creating a MPI version of a BinaryFunction: Conflict between two BFs." << endl;
		std::exit(1);
	}
	else if (currentlyApplied == this)
	{
		return &staticMPIop;
	}

	currentlyApplied = this;
	MPI_Op_create(BinaryFunction::apply, commutable, &staticMPIop);
	return &staticMPIop;
}

void BinaryFunction::releaseMPIOp()
{
	//cout << "free mpi op" << endl;

	if (currentlyApplied == this)
		currentlyApplied = NULL;
}


/**************************\
| SEMIRING
\**************************/
template <>
Semiring* SemiringTemplArg<doubleint, doubleint>::currentlyApplied = NULL;

Semiring::Semiring(PyObject *add, PyObject *multiply)
	: type(CUSTOM), pyfunc_add(add), pyfunc_multiply(multiply), binfunc_add(&binary(add))
{
	Py_INCREF(pyfunc_add);
	Py_INCREF(pyfunc_multiply);
}
Semiring::~Semiring()
{
	Py_XDECREF(pyfunc_add);
	Py_XDECREF(pyfunc_multiply);
	assert((SemiringTemplArg<doubleint, doubleint>::currentlyApplied != this));
}

void Semiring::enableSemiring()
{
	if (SemiringTemplArg<doubleint, doubleint>::currentlyApplied != NULL)
	{
		cout << "There is an internal error in selecting a Semiring: Conflict between two Semirings." << endl;
		std::exit(1);
	}
	SemiringTemplArg<doubleint, doubleint>::currentlyApplied = this;
	binfunc_add->getMPIOp();
}

void Semiring::disableSemiring()
{
	binfunc_add->releaseMPIOp();
	SemiringTemplArg<doubleint, doubleint>::currentlyApplied = NULL;
}

doubleint Semiring::add(const doubleint & arg1, const doubleint & arg2)
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

doubleint Semiring::multiply(const doubleint & arg1, const doubleint & arg2)
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

void Semiring::axpy(doubleint a, const doubleint & x, doubleint & y)
{
	y = add(y, multiply(a, x));
}

Semiring TimesPlusSemiring()
{
	return Semiring(Semiring::TIMESPLUS);
}

Semiring MinPlusSemiring()
{
	return Semiring(Semiring::PLUSMIN);
}

Semiring SecondMaxSemiring()
{
	return Semiring(Semiring::SECONDMAX);
}

} // namespace op
