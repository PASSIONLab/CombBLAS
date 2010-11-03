#ifndef PY_DENSE_PAR_VEC_H
#define PY_DENSE_PAR_VEC_H

#include "../../CombBLAS/SpParVec.h"
#include "../../CombBLAS/SpTuples.h"
#include "../../CombBLAS/SpDCCols.h"
#include "../../CombBLAS/SpParMat.h"
#include "../../CombBLAS/SpParVec.h"
#include "../../CombBLAS/DenseParMat.h"
#include "../../CombBLAS/DenseParVec.h"

class pyDenseParVec {
protected:

	DenseParVec<int64_t, int64_t> v;

/////////////// everything below this appears in python interface:
public:
	pyDenseParVec();

public:
	int length() const;
	
public:	
	void load(const char* filename);
	
};

#endif
