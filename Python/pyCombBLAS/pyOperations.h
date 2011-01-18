#ifndef PYOPERATION_H
#define PYOPERATION_H

#include "pyCombBLAS.h"
#include <functional>
#include <iostream>

namespace op {

template <typename T>
struct ConcreteUnaryFunction : public std::unary_function<T, T>
{
	virtual const T operator()(const T& x) const = 0;
};

template <typename T>
struct ConcreteBinaryFunction : public std::binary_function<T, T, T>
{
	virtual const T operator()(const T& x, const T& y) const = 0;
};

}

//INTERFACE_INCLUDE_BEGIN
namespace op {

class UnaryFunction {
//INTERFACE_INCLUDE_END
	public:
	ConcreteUnaryFunction<int64_t>* op;
	
	UnaryFunction(ConcreteUnaryFunction<int64_t>* opin): op(opin) {  }

//INTERFACE_INCLUDE_BEGIN

	protected:
	UnaryFunction(): op(NULL) {}
	public:
	~UnaryFunction() { /*delete op; op = NULL;*/ }
	
	const int64_t operator()(const int64_t x) const
	{
		return (*op)(x);
	}
};

UnaryFunction* set(int64_t val);
UnaryFunction* identity();
UnaryFunction* safemultinv();
UnaryFunction* abs();
UnaryFunction* negate();
UnaryFunction* bitwise_not();
UnaryFunction* logical_not();

//INTERFACE_INCLUDE_END


//INTERFACE_INCLUDE_BEGIN
class BinaryFunction {
//INTERFACE_INCLUDE_END
	public:
	ConcreteBinaryFunction<int64_t>* op;
	
	BinaryFunction(ConcreteBinaryFunction<int64_t>* opin): op(opin), commutable(false) {  }

	// for creating an MPI_Op that can be used with MPI Reduce
	static void apply(void * invec, void * inoutvec, int * len, MPI_Datatype *datatype);
	static BinaryFunction* currentlyApplied;
	static MPI_Op staticMPIop;
	
	MPI_Op* getMPIOp();
	void releaseMPIOp();
	
//INTERFACE_INCLUDE_BEGIN
	protected:
	BinaryFunction(): op(NULL), commutable(false) {}
	public:
	~BinaryFunction() { /*delete op; op = NULL;*/ }
	
	bool commutable;
	
	const int64_t operator()(const int64_t& x, const int64_t& y) const
	{
		return (*op)(x, y);
	}

};

BinaryFunction* plus();
BinaryFunction* minus();
BinaryFunction* multiplies();
BinaryFunction* divides();
BinaryFunction* modulus();

BinaryFunction* max();
BinaryFunction* min();

BinaryFunction* bitwise_and();
BinaryFunction* bitwise_or();
BinaryFunction* bitwise_xor();
BinaryFunction* logical_and();
BinaryFunction* logical_or();
BinaryFunction* logical_xor();

BinaryFunction* equal_to();
BinaryFunction* not_equal_to();
BinaryFunction* greater();
BinaryFunction* less();
BinaryFunction* greater_equal();
BinaryFunction* less_equal();


// Glue functions

UnaryFunction* bind1st(BinaryFunction* op, int64_t val);
UnaryFunction* bind2nd(BinaryFunction* op, int64_t val);
UnaryFunction* compose1(UnaryFunction* f, UnaryFunction* g); // h(x) is the same as f(g(x))
UnaryFunction* compose2(BinaryFunction* f, UnaryFunction* g1, UnaryFunction* g2); // h(x) is the same as f(g1(x), g2(x))
UnaryFunction* not1(UnaryFunction* f);
BinaryFunction* not2(BinaryFunction* f);

} // namespace op


//INTERFACE_INCLUDE_END


#endif