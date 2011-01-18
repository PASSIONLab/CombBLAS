#ifndef PY_SP_PAR_MAT_H
#define PY_SP_PAR_MAT_H

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
	
	pySpParMat(pySpParMat* copyFrom);

/////////////// everything below this appears in python interface:
//INTERFACE_INCLUDE_BEGIN
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

	pySpParMat* copy();
	
	void Apply(op::UnaryFunction* op);
	void Prune(op::UnaryFunction* op);
	
	pyDenseParVec* Reduce(int dim, op::BinaryFunction* f, int64_t identity = 0);
public:
	pySpParVec* SpMV_PlusTimes(const pySpParVec& v);
	pySpParVec* SpMV_SelMax(const pySpParVec& v);
	void SpMV_SelMax_inplace(pySpParVec& v);
	
public:
	static int Column() { return ::Column; }
	static int Row() { return ::Row; }
//INTERFACE_INCLUDE_END
};

#endif
