#ifndef PY_SP_PAR_MAT_H
#define PY_SP_PAR_MAT_H

#include "pyCombBLAS.h"

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
	
public:
	
	pySpParMat(MatType other);
	pySpParMat(const pySpParMat& copyFrom);

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
	
	double GenGraph500Edges(int scale, pyDenseParVec* pyDegrees = NULL, int EDGEFACTOR = 16);
	//double GenGraph500Edges(int scale, pyDenseParVec& pyDegrees);
	
public:
	pySpParMat copy();
	pySpParMat& operator+=(const pySpParMat& other);
	pySpParMat& assign(const pySpParMat& other);
	pySpParMat SpGEMM(pySpParMat& other, op::Semiring* sring = NULL);
	pySpParMat operator*(pySpParMat& other);
	pySpParMat SubsRef(const pyDenseParVec& rows, const pyDenseParVec& cols);
	pySpParMat __getitem__(const pyDenseParVec& rows, const pyDenseParVec& cols);
	
	int64_t removeSelfLoops();
	
	void Apply(op::UnaryFunction* f);
	void ColWiseApply(const pySpParVec& values, op::BinaryFunction* f);
	void DimWiseApply(int dim, const pyDenseParVec& values, op::BinaryFunction* f);
	void Prune(op::UnaryFunction* f);
	int64_t Count(op::UnaryFunction* pred);
	
	// Be wary of identity value with min()/max()!!!!!!!
	pyDenseParVec Reduce(int dim, op::BinaryFunction* f, double identity = 0);
	pyDenseParVec Reduce(int dim, op::BinaryFunction* bf, op::UnaryFunction* uf, double identity = 0);
	
	void Transpose();
	//void EWiseMult(pySpParMat* rhs, bool exclude);

	void Find(pyDenseParVec* outrows, pyDenseParVec* outcols, pyDenseParVec* outvals) const;
public:
	pySpParVec SpMV_PlusTimes(const pySpParVec& x);
	pySpParVec SpMV_SelMax(const pySpParVec& x);
	void SpMV_SelMax_inplace(pySpParVec& x);

	pySpParVec SpMV(const pySpParVec& x, op::Semiring* sring);
	pyDenseParVec SpMV(const pyDenseParVec& x, op::Semiring* sring);
	void SpMV_inplace(pySpParVec& x, op::Semiring* sring);
	void SpMV_inplace(pyDenseParVec& x, op::Semiring* sring);
	
public:
	static int Column() { return ::Column; }
	static int Row() { return ::Row; }
};

pySpParMat EWiseMult(const pySpParMat& A1, const pySpParMat& A2, bool exclude);
pySpParMat EWiseApply(const pySpParMat& A, const pySpParMat& B, op::BinaryFunction *bf, bool notB = false, double defaultBValue = 1);

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

template <> struct promote_trait< SpDCCols<int64_t,doubleint> , SpDCCols<int64_t,bool> >       
    {                                           
        typedef SpDCCols<int64_t,doubleint> T_promote;                    
    };

template <> struct promote_trait< SpDCCols<int64_t,bool> , SpDCCols<int64_t,doubleint> >       
    {                                           
        typedef SpDCCols<int64_t,doubleint> T_promote;                    
    };

// Based on what's in CombBLAS/SpDCCols.h:
template <class NIT, class NNT>  struct create_trait< SpDCCols<int64_t, doubleint> , NIT, NNT >
    {
        typedef SpDCCols<NIT,NNT> T_inferred;
    };


#endif
