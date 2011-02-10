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

public:
	typedef int64_t INDEXTYPE;
	typedef SpDCCols<INDEXTYPE,doubleint> DCColsType;
	typedef SpParMat < INDEXTYPE, doubleint, DCColsType > PSpMat_DoubleInt;
	typedef PSpMat_DoubleInt MatType;
	
protected:
	
	friend class pySpParVec;
	
	pySpParMat(pySpParMat* copyFrom);

public:
	MatType A;

/////////////// everything below this appears in python interface:
//INTERFACE_INCLUDE_BEGIN
public:
	pySpParMat();
	pySpParMat(int64_t m, int64_t n, pyDenseParVec* rows, pyDenseParVec* cols, pyDenseParVec* vals);

public:
	int64_t getnnz();
	int64_t getnee();
	int64_t getnrow();
	int64_t getncol();
	
public:	
	void load(const char* filename);
	void save(const char* filename);
	
	void GenGraph500Edges(int scale);
	double GenGraph500Edges(int scale, pyDenseParVec& pyDegrees);
	
public:
	pySpParMat* copy();
	pySpParMat& operator+=(const pySpParMat& other);
	pySpParMat& assign(const pySpParMat& other);
	pySpParMat* SpMM(const pySpParMat& other);
	pySpParMat* operator*(const pySpParMat& other);
	
	void Apply(op::UnaryFunction* f);
	void ColWiseApply(const pySpParVec& values, op::BinaryFunction* f);
	void Prune(op::UnaryFunction* f);
	int64_t Count(op::UnaryFunction* pred);
	
	// Be wary of identity value with min()/max()!!!!!!!
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

pySpParMat* EWiseApply(const pySpParMat& A, const pySpParMat& B, op::BinaryFunction *bf, bool notB = false, double defaultBValue = 1);

//INTERFACE_INCLUDE_END


// From CombBLAS/promote.h:
/*
template <class T1, class T2>
struct promote_trait  { };

#define DECLARE_PROMOTE(A,B,C)                  \
    template <> struct promote_trait<A,B>       \
    {                                           \
        typedef C T_promote;                    \
    };
*/
DECLARE_PROMOTE(pySpParMat::MatType, pySpParMat::MatType, pySpParMat::MatType)
DECLARE_PROMOTE(pySpParMat::DCColsType, pySpParMat::DCColsType, pySpParMat::DCColsType)



#endif
