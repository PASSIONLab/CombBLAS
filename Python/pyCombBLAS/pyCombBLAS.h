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


#include "pySpParMat.h"
#include "pySpParVec.h"
#include "pyDenseParVec.h"

class pySpParMat;
class pySpParVec;
class pyDenseParVec;

int64_t invert64(int64_t v);
int64_t abs64(int64_t v);
int64_t negate64(int64_t);


extern "C" {
void init_pyCombBLAS_MPI();
}

void finalize();
bool root();

#endif
