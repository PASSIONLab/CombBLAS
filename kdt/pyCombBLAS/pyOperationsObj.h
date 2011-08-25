#ifndef PYOPERATIONOBJ_H
#define PYOPERATIONOBJ_H

#include "pyCombBLAS.h"
#include <functional>
#include <iostream>
#include <math.h>

#ifndef NO_SWIGPYRUN
#include "swigpyrun.h"
#endif

extern "C" {
extern swig_type_info *SWIG_VertexTypeInfo;
extern swig_type_info *SWIG_EdgeTypeInfo;
}

#ifndef NO_SWIGPYRUN
#define SWIGTYPE_p_VERTEXTYPE SWIG_VertexTypeInfo
#define SWIGTYPE_p_EDGETYPE SWIG_EdgeTypeInfo
#endif

namespace op {

// Reusing the one from pyOperations.h
/*
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
*/


}

//INTERFACE_INCLUDE_BEGIN
namespace op {

class UnaryFunctionObj {
//INTERFACE_INCLUDE_END
	public:
	//ConcreteUnaryFunction<EDGETYPE>* op;
	//UnaryFunctionObj(ConcreteUnaryFunction<EDGETYPE>* opin): op(opin) {  }
	
	PyObject *callback;
	PyObject *edgeArgList, *vertexArgList;
	PyObject *tempEdgePy, *tempVertexPy;
	EDGETYPE *tempEdge;
	VERTEXTYPE *tempVertex;
	UnaryFunctionObj(PyObject *pyfunc);

//INTERFACE_INCLUDE_BEGIN

	protected:
	UnaryFunctionObj() { // should never be called
		printf("UnaryFunctionObj!!!\n");
		callback = NULL; edgeArgList = NULL; vertexArgList = NULL;
		tempEdgePy = NULL; tempVertexPy = NULL; tempEdge = NULL; tempVertex = NULL;
	}
	public:
	~UnaryFunctionObj();
	
	EDGETYPE operator()(const EDGETYPE& x) const;
	VERTEXTYPE operator()(const VERTEXTYPE& x) const;
};
/*
%pythoncode %{
def set_transform(im,x):
   a = new_mat44()
   for i in range(4):
       for j in range(4):
           mat44_set(a,i,j,x[i][j])
   _example.set_transform(im,a)
   free_mat44(a)
%}*/

//UnaryFunctionObj set(EDGETYPE val);
//UnaryFunctionObj set(VERTEXTYPE val);
//UnaryFunctionObj identityObj();

UnaryFunctionObj unaryObj(PyObject *pyfunc);

#if 0
//INTERFACE_INCLUDE_BEGIN
class BinaryFunctionE {
//INTERFACE_INCLUDE_END
	public:
	ConcreteBinaryFunction<EDGETYPE>* op;
	
	BinaryFunctionE(ConcreteBinaryFunction<EDGETYPE>* opin, bool as, bool com): op(opin), commutable(com), associative(as) {  }

	// for creating an MPI_Op that can be used with MPI Reduce
	static void apply(void * invec, void * inoutvec, int * len, MPI_Datatype *datatype);
	static BinaryFunctionE* currentlyApplied;
	static MPI_Op staticMPIop;
	
	MPI_Op* getMPIOp();
	void releaseMPIOp();
	
//INTERFACE_INCLUDE_BEGIN
	protected:
	BinaryFunctionE(): op(NULL), commutable(false), associative(false) {}
	public:
	~BinaryFunctionE() { /*delete op; op = NULL;*/ }
	
	bool commutable;
	bool associative;
	
	EDGETYPE operator()(const EDGETYPE& x, const EDGETYPE& y) const
	{
		return (*op)(x, y);
	}

};
class BinaryFunctionV {
//INTERFACE_INCLUDE_END
	public:
	ConcreteBinaryFunction<VERTEXTYPE>* op;
	
	BinaryFunctionV(ConcreteBinaryFunction<VERTEXTYPE>* opin, bool as, bool com): op(opin), commutable(com), associative(as) {  }

	// for creating an MPI_Op that can be used with MPI Reduce
	static void apply(void * invec, void * inoutvec, int * len, MPI_Datatype *datatype);
	static BinaryFunctionV* currentlyApplied;
	static MPI_Op staticMPIop;
	
	MPI_Op* getMPIOp();
	void releaseMPIOp();
	
//INTERFACE_INCLUDE_BEGIN
	protected:
	BinaryFunctionV(): op(NULL), commutable(false), associative(false) {}
	public:
	~BinaryFunctionV() { /*delete op; op = NULL;*/ }
	
	bool commutable;
	bool associative;
	
	VERTEXTYPE operator()(const VERTEXTYPE& x, const VERTEXTYPE& y) const
	{
		return (*op)(x, y);
	}

};

BinaryFunction binaryE(PyObject *pyfunc);
BinaryFunction binaryV(PyObject *pyfunc);

#endif
/*
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
*/
} // namespace op


//INTERFACE_INCLUDE_END

// modeled after CombBLAS/Operations.h
// This call is only safe when between BinaryFunction.getMPIOp() and releaseMPIOp() calls.
// That should be safe enough, because this is only called from inside CombBLAS reduce operations,
// which only get called between getMPIOp() and releaseMPIOp().
#if 0
template<> struct MPIOp< op::BinaryFunctionE, EDGETYPE > {  static MPI_Op op() { return op::BinaryFunctionE::staticMPIop; } };
template<> struct MPIOp< op::BinaryFunctionV, VERTEXTYPE > {  static MPI_Op op() { return op::BinaryFunctionV::staticMPIop; } };
#endif

#endif
