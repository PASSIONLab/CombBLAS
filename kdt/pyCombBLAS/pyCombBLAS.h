#ifndef PYCOMBBLAS_H
#define PYCOMBBLAS_H

//#include "../../CombBLAS/SpParVec.h"
//#include "../../CombBLAS/SpTuples.h"
//#include "../../CombBLAS/SpDCCols.h"
//#include "../../CombBLAS/SpParMat.h"
//#include "../../CombBLAS/SpParVec.h"
//#include "../../CombBLAS/DenseParMat.h"
//#include "../../CombBLAS/DenseParVec.h"
//#include "../../CombBLAS/ParFriends.h"
//#include "../../CombBLAS/Semirings.h"


/*
#ifdef NOTR1
        #include <boost/tr1/memory.hpp>
#else
        #include <tr1/memory>
#endif
#include "../../CombBLAS/SpTuples.h"
#include "../../CombBLAS/SpDCCols.h"
#include "../../CombBLAS/SpParMat.h"
#include "../../CombBLAS/FullyDistVec.h"
#include "../../CombBLAS/FullyDistSpVec.h"
#include "../../CombBLAS/VecIterator.h"
#include "../../CombBLAS/ParFriends.h"
#include "../../CombBLAS/DistEdgeList.h"
#include "../../CombBLAS/Operations.h"
*/
#include <Python.h>
#include "../../CombBLAS/CombBLAS.h"

namespace op{
class UnaryFunction;
class ObjUnaryFunction;
class BinaryFunction;
class Semiring;
}

#include "doubleint.h"

class pySpParMat;
class pySpParMatBool;
class pySpParVec;
class pyDenseParVec;
class pyObjDenseParVec;

#include "pyOperations.h"
#include "pySpParMat.h"
#include "pySpParMatBool.h"
#include "pySpParVec.h"
#include "pyDenseParVec.h"
#include "pyObjDenseParVec.h"

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


//INTERFACE_INCLUDE_BEGIN

void finalize();
bool root();
int _nprocs();

void testFunc(double (*f)(double));

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

#endif
