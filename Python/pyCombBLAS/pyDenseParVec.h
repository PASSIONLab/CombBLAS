#ifndef PY_DENSE_PAR_VEC_H
#define PY_DENSE_PAR_VEC_H

#include "../../CombBLAS/SpParVec.h"
#include "../../CombBLAS/SpTuples.h"
#include "../../CombBLAS/SpDCCols.h"
#include "../../CombBLAS/SpParMat.h"
#include "../../CombBLAS/SpParVec.h"
#include "../../CombBLAS/DenseParMat.h"
#include "../../CombBLAS/DenseParVec.h"
#include "../../CombBLAS/SpParVec.h"
#include "../../CombBLAS/ParFriends.h"

#include "pySpParMat.h"
#include "pySpParVec.h"

class pySpParMat;
class pySpParVec;

class pyDenseParVec {
protected:

	DenseParVec<int64_t, int64_t> v;
	
	friend class pySpParVec;
	friend pySpParVec* EWiseMult(const pySpParVec& a, const pyDenseParVec& b, bool exclude, int64_t zero);

/////////////// everything below this appears in python interface:
public:
	pyDenseParVec();
	pyDenseParVec(int64_t size, int64_t id);
	//pyDenseParVec(const pySpParMat& commSource, int64_t zero);

public:
	int length() const;
	
	const pyDenseParVec& add(const pyDenseParVec& other);
	const pyDenseParVec& add(const pySpParVec& other);
	
public:	
	void load(const char* filename);
	
};

#endif
