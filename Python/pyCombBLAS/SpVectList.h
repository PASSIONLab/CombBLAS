#ifndef SPVECTLIST_H
#define SPVECTLIST_H

#include "../../CombBLAS/SpParVec.h"
#include "../../CombBLAS/SpTuples.h"
#include "../../CombBLAS/SpDCCols.h"
#include "../../CombBLAS/SpParMat.h"
#include "../../CombBLAS/SpParVec.h"
#include "../../CombBLAS/DenseParMat.h"
#include "../../CombBLAS/DenseParVec.h"

class SpVectList {
protected:

	SpParVec<int, int> v;

/////////////// everything below this appears in python interface:
public:
	SpVectList();

public:
	int length() const;
	
public:	
	void load(const char* filename);
	
};

#endif
