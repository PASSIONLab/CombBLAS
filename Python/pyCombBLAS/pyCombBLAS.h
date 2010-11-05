#ifndef PYCOMBBLAS_H
#define PYCOMBBLAS_H

#include "pySpParMat.h"
#include "pySpParVec.h"
#include "pyDenseParVec.h"

class pySpParMat;
class pySpParVec;
class pyDenseParVec;

int64_t invert64(int64_t v);
int64_t abs64(int64_t v);
int64_t negate64(int64_t);

bool nonzero64(int64_t);
bool zero64(int64_t);
bool neq64(int64_t test, int64_t val);
bool eq64(int64_t test, int64_t val);

extern "C" {
void init_pyCombBLAS_MPI();
}

void finalize();
bool root();

#endif
