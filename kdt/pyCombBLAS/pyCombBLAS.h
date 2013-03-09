#ifndef PYCOMBBLAS_H
#define PYCOMBBLAS_H

#ifdef _MSC_VER
#pragma warning(disable : 4244)
#pragma warning(disable : 4800) // forcing to bool, performance warning.
#endif

#include <Python.h>
#include "../../CombBLAS/CombBLAS.h"

#define PYCOMBBLAS_MPIOK 1
#include "pyCombBLAS-NoMPI.h"

extern shared_ptr<CommGrid> commGrid;


#include "pyOperations.h"
#include "pyOperationsObj.h"
#include "pySemirings.h"
#include "pySpParVec.h"
#include "pySpParVecObj1.h"
#include "pySpParVecObj2.h"
#include "pyDenseParVec.h"
#include "pyDenseParVecObj1.h"
#include "pyDenseParVecObj2.h"
#include "pySpParMat.h"
#include "pySpParMatBool.h"
#include "pySpParMatObj1.h"
#include "pySpParMatObj2.h"

extern "C" {
void init_pyCombBLAS_MPI();
}

// Structure defining the arguments to EWise()
class EWiseArgDescriptor
{
	public:
	enum Type { ITERATOR, GLOBAL_INDEX, PYTHON_OBJ };
	Type type;
	VectorLocalIterator<int64_t, doubleint>* iter; // iterator if this argument needs one
	bool onlyNZ; // if this represents an iterator, is this iterator allowed to point to a null element?
	
	EWiseArgDescriptor(): type(ITERATOR), iter(NULL), onlyNZ(false) {}
	~EWiseArgDescriptor() { delete iter; }
};


// RNG
extern MTRand GlobalMT;

//INTERFACE_INCLUDE_BEGIN

void finalize();
bool root();
void _broadcast(char *outMsg, char *inMsg);
void _barrier();
int _nprocs();
int _rank();
void prnt(const char* str);
double _random();

class NotFoundError {};

class EWiseArg
{
	public:
//INTERFACE_INCLUDE_END
	pyDenseParVec* dptr;
	pySpParVec* sptr;

	enum Type { SPARSE_NZ, SPARSE, DENSE, GLOBAL_INDEX };
	Type type;
//INTERFACE_INCLUDE_BEGIN
	EWiseArg(): dptr(NULL), sptr(NULL), type(SPARSE) {}
};

EWiseArg EWise_Index();
EWiseArg EWise_OnlyNZ(pySpParVec* v);
EWiseArg EWise_OnlyNZ(pyDenseParVec* v); // shouldn't be used, but here for completeness

//INTERFACE_INCLUDE_END
// Swig doesn't seem to handle this, actually
#ifdef SWIG_python
//INTERFACE_INCLUDE_BEGIN
/*
%typemap(in) char ** {
  // Check if is a list
  if (PyList_Check($input)) {
    int size = PyList_Size($input);
    int i = 0;
    $1 = (char **) malloc((size+1)*sizeof(char *));
    for (i = 0; i < size; i++) {
      PyObject *o = PyList_GetItem($input,i);
      if (PyString_Check(o))
	$1[i] = PyString_AsString(PyList_GetItem($input,i));
      else {
	PyErr_SetString(PyExc_TypeError,"list must contain strings");
	free($1);
	return NULL;
      }
    }
    $1[i] = 0;
  } else {
    PyErr_SetString(PyExc_TypeError,"not a list");
    return NULL;
  }
}

// This cleans up the char ** array we malloc'd before the function call
%typemap(freearg) char ** {
  free((char *) $1);
}*/

%typemap(in) (int argc, EWiseArgDescriptor* argv, PyObject *argList) {
	/* Check if is a list */
	if (PyList_Check($input)) {
		int size = PyList_Size($input);
		int i = 0;

		$1 = size;
		$2 = new EWiseArgDescriptor[size];
		$3 = $input;

		pyDenseParVec* dptr;
		pySpParVec* sptr;
		EWiseArg* argptr;
		for (i = 0; i < size; i++)
		{
			PyObject *o = PyList_GetItem($input,i);
			if (SWIG_IsOK(SWIG_ConvertPtr(o, (void**)&dptr, $descriptor(pyDenseParVec *), 0)))
			{
				$2[i].type = EWiseArgDescriptor::ITERATOR;
				$2[i].onlyNZ = false;
				$2[i].iter = new DenseVectorLocalIterator<int64_t, doubleint>(dptr->v);
			}
			else if (SWIG_IsOK(SWIG_ConvertPtr(o, (void**)&sptr, $descriptor(pySpParVec *), 0)))
			{
				$2[i].type = EWiseArgDescriptor::ITERATOR;
				$2[i].onlyNZ = false;
				$2[i].iter = new SparseVectorLocalIterator<int64_t, doubleint>(sptr->v);
			}
			else if (SWIG_IsOK(SWIG_ConvertPtr(o, (void**)&argptr, $descriptor(EWiseArg *), 0)))
			{
				switch (argptr->type)
				{
					case EWiseArg::GLOBAL_INDEX:
						$2[i].type = EWiseArgDescriptor::GLOBAL_INDEX;
						break;
					case EWiseArg::DENSE:
						$2[i].type = EWiseArgDescriptor::ITERATOR;
						$2[i].onlyNZ = false;
						$2[i].iter = new DenseVectorLocalIterator<int64_t, doubleint>(argptr->dptr->v);
						break;
					case EWiseArg::SPARSE:
						$2[i].type = EWiseArgDescriptor::ITERATOR;
						$2[i].onlyNZ = false;
						$2[i].iter = new SparseVectorLocalIterator<int64_t, doubleint>(argptr->sptr->v);
						break;
					case EWiseArg::SPARSE_NZ:
						$2[i].type = EWiseArgDescriptor::ITERATOR;
						$2[i].onlyNZ = true;
						$2[i].iter = new SparseVectorLocalIterator<int64_t, doubleint>(argptr->sptr->v);
						break;
					default:
						cout << "AAAHHH! What are you passing to EWise()?" << endl;
						break;
				}
			}
			else
			{
				// python object
				$2[i].type = EWiseArgDescriptor::PYTHON_OBJ;
			}
		}
		
	} else {
		PyErr_SetString(PyExc_TypeError,"not a list");
		return NULL;
	}
}

// This cleans up the char ** array we malloc'd before the function call
%typemap(freearg) (int argc, EWiseArgDescriptor* argv, PyObject *argList) {
	delete [] $2;
}

//INTERFACE_INCLUDE_END
#endif
//INTERFACE_INCLUDE_BEGIN

void EWise(PyObject *pyewisefunc, int argc, EWiseArgDescriptor* argv, PyObject *argList);
void Graph500VectorOps(pySpParVec& fringe_v, pyDenseParVec& parents_v);

//INTERFACE_INCLUDE_END

// used to reverse predicates. Useful to turn Prune into Keep
template <typename PT>
class pcb_logical_not
{
	public:
	PT& pred;
	pcb_logical_not(PT& p): pred(p) {}
	
	template <typename T>
	bool operator()(T& val)
	{
		return !pred(val);
	}
};

// Helpers for EWiseApply
template <typename NU1, typename NU2>
class EWiseFilterDoOpAdapter
{
	public:
	op::BinaryPredicateObj* plain_binary_op;
	op::UnaryPredicateObj* AFilter;
	op::UnaryPredicateObj* BFilter;
	bool allowANulls, allowBNulls, allowIntersect;
	bool aDense, bDense;
	
	EWiseFilterDoOpAdapter(op::BinaryPredicateObj* op, op::UnaryPredicateObj* AF, op::UnaryPredicateObj* BF, bool aAN, bool aBN, bool aI, bool aD=false, bool bD=false): plain_binary_op(op), AFilter(AF), BFilter(BF), allowANulls(aAN), allowBNulls(aBN), allowIntersect(aI), aDense(aD), bDense(bD) {}
	
	bool operator()(const NU1& a, const NU2& b, bool aIsNull, bool bIsNull)
	{
		// dense semantics mean that filtered-out elements need to pretend to exist like empty values,
		// but the op still needs to happen.
		if (aDense || bDense)
		{
			if (plain_binary_op == NULL)
				return true;
			else
				return (*plain_binary_op)(a, b);
		}
		
		// for all-sparse cases
		bool aPass = aIsNull ? false : (AFilter == NULL ? true : (*AFilter)(a));
		bool bPass = bIsNull ? false : (BFilter == NULL ? true : (*BFilter)(b));
		
		if (!aPass && !bPass)
			return false;
		if (!aPass && !allowANulls)
			return false;
		if (!bPass && !allowBNulls)
			return false;
		
		if (plain_binary_op == NULL)
			return true;
		else
			return (*plain_binary_op)(a, b);
	}
};

template <typename RETT, typename NU1, typename NU2, typename BINOP_T = op::BinaryFunctionObj>
class EWiseFilterOpAdapter
{
	public:
	BINOP_T* plain_binary_op;
	op::UnaryPredicateObj* AFilter;
	op::UnaryPredicateObj* BFilter;
	bool allowANulls, allowBNulls, allowIntersect;
	const NU1& ANull;
	const NU2& BNull;
	
	EWiseFilterOpAdapter(BINOP_T* op, op::UnaryPredicateObj* AF, op::UnaryPredicateObj* BF, bool aAN, bool aBN, bool aI, const NU1& AN, const NU2& BN): plain_binary_op(op), AFilter(AF), BFilter(BF), allowANulls(aAN), allowBNulls(aBN), allowIntersect(aI), ANull(AN), BNull(BN)
	{
		if (plain_binary_op == NULL)
			throw string("bloody murder! don't pass in null binary ops to eWiseApply!");
	}
	
	RETT operator()(const NU1& a, const NU2& b, bool aIsNull, bool bIsNull)
	{
		bool aPass = aIsNull ? false : (AFilter == NULL ? true : (*AFilter)(a));
		bool bPass = bIsNull ? false : (BFilter == NULL ? true : (*BFilter)(b));
		
		if (!aPass && !bPass)
			return (*plain_binary_op)(ANull, BNull); // should only happen in dense cases, the DoOp adapter should prevent this for happenning for sparse/sparse cases.
		else if (!aPass &&  bPass)
			return (*plain_binary_op)(ANull, b);
		else if ( aPass && !bPass)
			return (*plain_binary_op)(a, BNull);
		else
			return (*plain_binary_op)(a, b);
	}
};

#endif
