#ifndef PY_SP_PAR_MAT_H
#define PY_SP_PAR_MAT_H

#include "../../CombBLAS/SpParVec.h"
#include "../../CombBLAS/SpTuples.h"
#include "../../CombBLAS/SpDCCols.h"
#include "../../CombBLAS/SpParMat.h"
#include "../../CombBLAS/SpParVec.h"
#include "../../CombBLAS/DenseParMat.h"
#include "../../CombBLAS/DenseParVec.h"
#include "../../CombBLAS/DistEdgeList.h"
#include "../../CombBLAS/ParFriends.h"
#include "../../CombBLAS/Semirings.h"

#include "pyCombBLAS.h"

class pySpParMat;
class pySpParVec;
class pyDenseParVec;


class pySpParMat {
protected:

	typedef SpParMat < int64_t, bool, SpDCCols<int64_t,bool> > PSpMat_Bool;
	typedef SpParMat < int64_t, int, SpDCCols<int64_t,int> > PSpMat_Int;
	typedef SpParMat < int64_t, int64_t, SpDCCols<int64_t,int64_t> > PSpMat_Int64;

	PSpMat_Bool A;
	
	friend class pySpParVec;

/////////////// everything below this appears in python interface:
public:
	pySpParMat();

public:
	int64_t getnnz();
	int64_t getnrow();
	int64_t getncol();
	
public:	
	void load(const char* filename);
	void GenGraph500Edges(int scale);
	double GenGraph500Edges(int scale, pyDenseParVec& pyDegrees);
	pyDenseParVec* GenGraph500Candidates(int howmany);
	
public:
	pyDenseParVec* FindIndsOfColsWithSumGreaterThan(int64_t gt);
	//pyDenseParVec* Reduce_ColumnSums();
	
	void Apply_SetTo(int64_t v);
	
public:
	pySpParVec* SpMV_PlusTimes(const pySpParVec& v);
	pySpParVec* SpMV_SelMax(const pySpParVec& v);
	void SpMV_SelMax_inplace(pySpParVec& v);
	
};

#endif
