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


//double cblas_alltoalltime;
//double cblas_allgathertime;

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

#include "pyOperations.h"
#include "pySpParMat.h"
#include "pySpParVec.h"
#include "pyDenseParVec.h"

class pySpParMat;
class pySpParVec;
class pyDenseParVec;

extern "C" {
void init_pyCombBLAS_MPI();
}

void finalize();
bool root();

#endif
