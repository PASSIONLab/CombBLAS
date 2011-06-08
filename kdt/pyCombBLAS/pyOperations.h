#ifndef PYOPERATION_H
#define PYOPERATION_H

#include "pyCombBLAS.h"
#include <functional>
#include <iostream>
#include <math.h>

namespace op {

template <typename T>
struct ConcreteUnaryFunction : public std::unary_function<T, T>
{
	virtual T operator()(const T& x) const = 0;
	
	virtual ~ConcreteUnaryFunction() {}
};

template <typename T>
struct ConcreteBinaryFunction : public std::binary_function<T, T, T>
{
	virtual T operator()(const T& x, const T& y) const = 0;

	virtual ~ConcreteBinaryFunction() {}
};

template <class T1, class T2>
struct SemiringTemplArg;

}

//INTERFACE_INCLUDE_BEGIN
namespace op {

class UnaryFunction {
//INTERFACE_INCLUDE_END
	public:
	ConcreteUnaryFunction<doubleint>* op;
	
	UnaryFunction(ConcreteUnaryFunction<doubleint>* opin): op(opin) {  }

//INTERFACE_INCLUDE_BEGIN

	protected:
	UnaryFunction(): op(NULL) {}
	public:
	~UnaryFunction() { /*delete op; op = NULL;*/ }
	
	doubleint operator()(const doubleint x) const
	{
		return (*op)(x);
	}
};

UnaryFunction set(double val);
UnaryFunction identity();
UnaryFunction safemultinv();
UnaryFunction abs();
UnaryFunction negate();
UnaryFunction bitwise_not();
UnaryFunction logical_not();
UnaryFunction totality();
UnaryFunction ifthenelse(UnaryFunction& predicate, UnaryFunction& runTrue, UnaryFunction& runFalse);

UnaryFunction unary(PyObject *pyfunc);

//INTERFACE_INCLUDE_END
//////////////////////////////////////////////////////////////////////////////
//INTERFACE_INCLUDE_BEGIN

class ObjUnaryFunction {
//INTERFACE_INCLUDE_END
	public:
	PyObject *pyfunc;
	PyObject *arglist;

//INTERFACE_INCLUDE_BEGIN

	protected:
	ObjUnaryFunction(): pyfunc(NULL), arglist(NULL) {}
	public:
	
	ObjUnaryFunction(PyObject *pyfunc_in);
	
	~ObjUnaryFunction();
	
	PyObject* operator()(PyObject* x);
};

ObjUnaryFunction obj_unary(PyObject *pyfunc);

//INTERFACE_INCLUDE_END

///////////////////////////////////////////////////////////////////////////////

//INTERFACE_INCLUDE_BEGIN
class BinaryFunction {
//INTERFACE_INCLUDE_END
	public:
	ConcreteBinaryFunction<doubleint>* op;
	
	BinaryFunction(ConcreteBinaryFunction<doubleint>* opin, bool as, bool com): op(opin), commutable(com), associative(as) {  }

	// for creating an MPI_Op that can be used with MPI Reduce
	static void apply(void * invec, void * inoutvec, int * len, MPI_Datatype *datatype);
	static BinaryFunction* currentlyApplied;
	static MPI_Op staticMPIop;
	
	MPI_Op* getMPIOp();
	void releaseMPIOp();
	
//INTERFACE_INCLUDE_BEGIN
	protected:
	BinaryFunction(): op(NULL), commutable(false), associative(false) {}
	public:
	~BinaryFunction() { /*delete op; op = NULL;*/ }
	
	bool commutable;
	bool associative;
	
	doubleint operator()(const doubleint& x, const doubleint& y) const
	{
		return (*op)(x, y);
	}

};

BinaryFunction plus();
BinaryFunction minus();
BinaryFunction multiplies();
BinaryFunction divides();
BinaryFunction modulus();
BinaryFunction fmod();
BinaryFunction pow();

BinaryFunction max();
BinaryFunction min();

BinaryFunction bitwise_and();
BinaryFunction bitwise_or();
BinaryFunction bitwise_xor();
BinaryFunction logical_and();
BinaryFunction logical_or();
BinaryFunction logical_xor();

BinaryFunction equal_to();
BinaryFunction not_equal_to();
BinaryFunction greater();
BinaryFunction less();
BinaryFunction greater_equal();
BinaryFunction less_equal();

BinaryFunction binary(PyObject *pyfunc);

// Glue functions

UnaryFunction bind1st(BinaryFunction& op, double val);
UnaryFunction bind2nd(BinaryFunction& op, double val);
UnaryFunction compose1(UnaryFunction& f, UnaryFunction& g); // h(x) is the same as f(g(x))
UnaryFunction compose2(BinaryFunction& f, UnaryFunction& g1, UnaryFunction& g2); // h(x) is the same as f(g1(x), g2(x))
UnaryFunction not1(UnaryFunction& f);
BinaryFunction not2(BinaryFunction& f);



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

} // namespace op


//INTERFACE_INCLUDE_END

// modeled after CombBLAS/Operations.h
// This call is only safe when between BinaryFunction.getMPIOp() and releaseMPIOp() calls.
// That should be safe enough, because this is only called from inside CombBLAS reduce operations,
// which only get called between getMPIOp() and releaseMPIOp().
template<> struct MPIOp< op::BinaryFunction, doubleint > {  static MPI_Op op() { return op::BinaryFunction::staticMPIop; } };

#endif
