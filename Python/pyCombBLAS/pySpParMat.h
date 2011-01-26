#ifndef PY_SP_PAR_MAT_H
#define PY_SP_PAR_MAT_H

#include "pyCombBLAS.h"

class pySpParMat;
class pySpParVec;
class pyDenseParVec;


//INTERFACE_INCLUDE_BEGIN
class pySpParMat {
//INTERFACE_INCLUDE_END
protected:

	typedef SpParMat < int64_t, bool, SpDCCols<int64_t,bool> > PSpMat_Bool;
	typedef SpParMat < int64_t, int, SpDCCols<int64_t,int> > PSpMat_Int;
	typedef SpParMat < int64_t, int64_t, SpDCCols<int64_t,int64_t> > PSpMat_Int64;

	typedef PSpMat_Int64 MatType;
	MatType A;
	
	friend class pySpParVec;
	
	pySpParMat(pySpParMat* copyFrom);

/////////////// everything below this appears in python interface:
//INTERFACE_INCLUDE_BEGIN
public:
	pySpParMat();
	pySpParMat(int64_t m, int64_t n, pyDenseParVec* rows, pyDenseParVec* cols, pyDenseParVec* vals);

public:
	int64_t getnnz();
	int64_t getnrow();
	int64_t getncol();
	
public:	
	void load(const char* filename);
	void GenGraph500Edges(int scale);
	double GenGraph500Edges(int scale, pyDenseParVec& pyDegrees);
	
public:
	pySpParMat* copy();
	
	void Apply(op::UnaryFunction* op);
	void Prune(op::UnaryFunction* op);
	
	pyDenseParVec* Reduce(int dim, op::BinaryFunction* f, int64_t identity = 0);
	pyDenseParVec* Reduce(int dim, op::BinaryFunction* bf, op::UnaryFunction* uf, int64_t identity = 0);
	
	void Transpose();
	//void EWiseMult(pySpParMat* rhs, bool exclude);

	void Find(pyDenseParVec* outrows, pyDenseParVec* outcols, pyDenseParVec* outvals) const;
public:
	pySpParVec* SpMV_PlusTimes(const pySpParVec& v);
	pySpParVec* SpMV_SelMax(const pySpParVec& v);
	void SpMV_SelMax_inplace(pySpParVec& v);
	
public:
	static int Column() { return ::Column; }
	static int Row() { return ::Row; }
};
//INTERFACE_INCLUDE_END

#endif
