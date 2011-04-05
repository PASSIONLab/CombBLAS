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
#include "../../CombBLAS/ParFriends.h"
#include "../../CombBLAS/DistEdgeList.h"
#include "../../CombBLAS/Operations.h"

namespace op{
class UnaryFunction;
class BinaryFunction;
}

#include "doubleint.h"

class pySpParMat;
class pySpParMatBool;
class pySpParVec;
class pyDenseParVec;

#include "pyOperations.h"
#include "pySpParMat.h"
#include "pySpParMatBool.h"
#include "pySpParVec.h"
#include "pyDenseParVec.h"

extern "C" {
void init_pyCombBLAS_MPI();
}

//INTERFACE_INCLUDE_BEGIN

void finalize();
bool root();
int _nprocs();

void testFunc(double (*f)(double));

//INTERFACE_INCLUDE_END

#endif
