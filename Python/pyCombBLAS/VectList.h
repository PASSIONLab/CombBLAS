#ifndef VECTLIST_H
#define VECTLIST_H

#include "../../CombBLAS/SpParVec.h"
#include "../../CombBLAS/SpTuples.h"
#include "../../CombBLAS/SpDCCols.h"
#include "../../CombBLAS/SpParMat.h"
#include "../../CombBLAS/SpParVec.h"
#include "../../CombBLAS/DenseParMat.h"
#include "../../CombBLAS/DenseParVec.h"

class VectList {
protected:

	DenseParVec<int, int> v;

/////////////// everything below this appears in python interface:
public:
	VectList();

public:
	int length() const;
	
public:	
	void load(const char* filename);
	
};

#endif
