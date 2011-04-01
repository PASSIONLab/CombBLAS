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

//INTERFACE_INCLUDE_END


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


// Glue functions

UnaryFunction bind1st(BinaryFunction& op, double val);
UnaryFunction bind2nd(BinaryFunction& op, double val);
UnaryFunction compose1(UnaryFunction& f, UnaryFunction& g); // h(x) is the same as f(g(x))
UnaryFunction compose2(BinaryFunction& f, UnaryFunction& g1, UnaryFunction& g2); // h(x) is the same as f(g1(x), g2(x))
UnaryFunction not1(UnaryFunction& f);
BinaryFunction not2(BinaryFunction& f);

} // namespace op


//INTERFACE_INCLUDE_END

// modeled after CombBLAS/Operations.h
// This call is only safe when between BinaryFunction.getMPIOp() and releaseMPIOp() calls.
// That should be safe enough, because this is only called from inside CombBLAS reduce operations,
// which only get called between getMPIOp() and releaseMPIOp().
template<> struct MPIOp< op::BinaryFunction, doubleint > {  static MPI_Op op() { return op::BinaryFunction::staticMPIop; } };

#endif
