#ifndef PYCOMBBLAS_NOMPI_H
#define PYCOMBBLAS_NOMPI_H

/*
This file is for declarations that do not directly depend on MPI. It's so parts of
pyCombBLAS can be compiled with plain g++ instead of mpicxx. The motivation for this
is to allow SEJITS to use the default compiler.
*/

#include <Python.h>
#include <cmath>

#ifndef PYCOMBBLAS_MPIOK
#define PYCOMBBLAS_MPIOK 0
#endif

template <typename T1, typename T2>
bool retTrue(const T1& x, const T2& y)
{
	return true;
}

namespace op{
class UnaryFunction;
class UnaryFunctionObj;
class UnaryPredicateObj;
class BinaryFunction;
class BinaryFunctionObj;
class BinaryPredicateObj;
class Semiring;
class SemiringObj;
}

#ifndef NO_SWIGPYRUN
#include "swigpyrun.h"
#endif

#include "doubleint.h"

class pySpParMat;
class pySpParMatBool;
class pySpParMatObj1;
class pySpParMatObj2;
class pySpParVec;
class pySpParVecObj1;
class pySpParVecObj2;
class pyDenseParVec;
class pyDenseParVecObj1;
class pyDenseParVecObj2;

template <typename RET, typename T1=RET, typename T2=RET>
class use2nd
{
public:
	const RET& operator()(const T1& a, const T2& b) const { return b; }
};

#include "obj.h"

// RNG
double _random();

#endif