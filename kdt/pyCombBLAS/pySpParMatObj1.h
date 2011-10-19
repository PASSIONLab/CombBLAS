#ifndef PY_SP_PAR_MAT_Obj1_H
#define PY_SP_PAR_MAT_Obj1_H

#include "pyCombBLAS.h"

//INTERFACE_INCLUDE_BEGIN
class pySpParMatObj1 {
//INTERFACE_INCLUDE_END
protected:

	typedef SpParMat < int64_t, bool, SpDCCols<int64_t,bool> > PSpMat_Bool;
	typedef SpParMat < int64_t, int, SpDCCols<int64_t,int> > PSpMat_Int;
	typedef SpParMat < int64_t, int64_t, SpDCCols<int64_t,int64_t> > PSpMat_Int64;

public:
	typedef int64_t INDEXTYPE;
	typedef SpDCCols<INDEXTYPE,Obj1> DCColsType;
	typedef SpParMat < INDEXTYPE, Obj1, DCColsType > PSpMat_Obj1;
	typedef PSpMat_Obj1 MatType;
	
public:
	
	pySpParMatObj1(MatType other);
	pySpParMatObj1(const pySpParMatObj1& copyFrom);

public:
	MatType A;

/////////////// everything below this appears in python interface:
//INTERFACE_INCLUDE_BEGIN
public:
	pySpParMatObj1();
	pySpParMatObj1(int64_t m, int64_t n, pyDenseParVec* rows, pyDenseParVec* cols, pyDenseParVecObj1* vals);

public:
	//int64_t getnnz();
	int64_t getnee();
	int64_t getnrow();
	int64_t getncol();
	
public:	
	void load(const char* filename);
	void save(const char* filename);
	
	//double GenGraph500Edges(int scale, pyDenseParVec* pyDegrees = NULL, int EDGEFACTOR = 16);
	//double GenGraph500Edges(int scale, pyDenseParVec& pyDegrees);
	
public:
	pySpParMatObj1 copy();
	//pySpParMatObj1& operator+=(const pySpParMatObj1& other);
	pySpParMatObj1& assign(const pySpParMatObj1& other);
	pySpParMat     SpGEMM(pySpParMat&     other, op::SemiringObj* sring);
	pySpParMatObj1 SpGEMM(pySpParMatObj1& other, op::SemiringObj* sring);
	pySpParMatObj2 SpGEMM(pySpParMatObj2& other, op::SemiringObj* sring);
	//pySpParMatObj1 operator*(pySpParMatObj1& other);
#define NOPARMATSUBSREF
	pySpParMatObj1 SubsRef(const pyDenseParVec& rows, const pyDenseParVec& cols);
	pySpParMatObj1 __getitem__(const pyDenseParVec& rows, const pyDenseParVec& cols);
	
	int64_t removeSelfLoops();
	
	void Apply(op::UnaryFunctionObj* f);
	void DimWiseApply(int dim, const pyDenseParVecObj1& values, op::BinaryFunctionObj* f);
	void Prune(op::UnaryPredicateObj* pred);
	int64_t Count(op::UnaryPredicateObj* pred);
	
	// Be wary of identity value with min()/max()!!!!!!!
	pyDenseParVecObj1 Reduce(int dim, op::BinaryFunctionObj* f, Obj1 identity = Obj1());
	pyDenseParVecObj1 Reduce(int dim, op::BinaryFunctionObj* bf, op::UnaryFunctionObj* uf, Obj1 identity = Obj1());
	
	void Transpose();
	//void EWiseMult(pySpParMatObj1* rhs, bool exclude);

	void Find(pyDenseParVec* outrows, pyDenseParVec* outcols, pyDenseParVecObj1* outvals) const;
public:
/*
	pySpParVec SpMV_PlusTimes(const pySpParVec& x);
	pySpParVec SpMV_SelMax(const pySpParVec& x);
	void SpMV_SelMax_inplace(pySpParVec& x);
*/
	pySpParVec     SpMV(const pySpParVec&     x, op::SemiringObj* sring) const;
	pySpParVecObj1 SpMV(const pySpParVecObj1& x, op::SemiringObj* sring);
	pySpParVecObj2 SpMV(const pySpParVecObj2& x, op::SemiringObj* sring);
	pyDenseParVec     SpMV(const pyDenseParVec&     x, op::SemiringObj* sring);
	pyDenseParVecObj1 SpMV(const pyDenseParVecObj1& x, op::SemiringObj* sring);
	pyDenseParVecObj2 SpMV(const pyDenseParVecObj2& x, op::SemiringObj* sring);
//	void SpMV_inplace(pySpParVec& x, op::SemiringObj* sring);
//	void SpMV_inplace(pyDenseParVec& x, op::SemiringObj* sring);

public:
	static int Column() { return ::Column; }
	static int Row() { return ::Row; }
};

//pySpParMat EWiseMult(const pySpParMat& A1, const pySpParMat& A2, bool exclude);
pySpParMatObj1 EWiseApply(const pySpParMatObj1& A, const pySpParMatObj1& B, op::BinaryFunctionObj *bf, bool notB = false, Obj1 defaultBValue = Obj1());
pySpParMatObj1 EWiseApply(const pySpParMatObj1& A, const pySpParMatObj2& B, op::BinaryFunctionObj *bf, bool notB = false, Obj2 defaultBValue = Obj2());
pySpParMatObj1 EWiseApply(const pySpParMatObj1& A, const pySpParMat&     B, op::BinaryFunctionObj *bf, bool notB = false, double defaultBValue = 0);

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
DECLARE_PROMOTE(pySpParMatObj1::MatType, pySpParMatObj1::MatType, pySpParMatObj1::MatType)
DECLARE_PROMOTE(pySpParMatObj1::DCColsType, pySpParMatObj1::DCColsType, pySpParMatObj1::DCColsType)

template <> struct promote_trait< SpDCCols<int64_t,Obj1> , SpDCCols<int64_t,Obj2> >
    {                                           
        typedef SpDCCols<int64_t,Obj2> T_promote;
    };
template <> struct promote_trait< SpDCCols<int64_t,Obj1> , SpDCCols<int64_t,doubleint> >
    {                                           
        typedef SpDCCols<int64_t,doubleint> T_promote;
    };
///////
template <> struct promote_trait< SpDCCols<int64_t,Obj1> , SpDCCols<int64_t,bool> >       
    {                                           
        typedef SpDCCols<int64_t,Obj1> T_promote;                    
    };

template <> struct promote_trait< SpDCCols<int64_t,bool> , SpDCCols<int64_t,Obj1> >       
    {                                           
        typedef SpDCCols<int64_t,Obj1> T_promote;                    
    };

// Based on what's in CombBLAS/SpDCCols.h:
template <class NIT, class NNT>  struct create_trait< SpDCCols<int64_t, Obj1> , NIT, NNT >
    {
        typedef SpDCCols<NIT,NNT> T_inferred;
    };


#endif
