#ifndef PY_SP_PAR_MAT_Obj2_H
#define PY_SP_PAR_MAT_Obj2_H

#include "pyCombBLAS.h"

//INTERFACE_INCLUDE_BEGIN
class pySpParMatObj2 {
//INTERFACE_INCLUDE_END
protected:

	typedef SpParMat < int64_t, bool, SpDCCols<int64_t,bool> > PSpMat_Bool;
	typedef SpParMat < int64_t, int, SpDCCols<int64_t,int> > PSpMat_Int;
	typedef SpParMat < int64_t, int64_t, SpDCCols<int64_t,int64_t> > PSpMat_Int64;

public:
	typedef int64_t INDEXTYPE;
	typedef Obj2 NUMTYPE;
	typedef SpDCCols<INDEXTYPE,NUMTYPE> DCColsType;
	typedef SpParMat < INDEXTYPE, NUMTYPE, DCColsType > PSpMat_Obj2;
	typedef PSpMat_Obj2 MatType;
	
public:
	
	pySpParMatObj2(MatType other);
	pySpParMatObj2(const pySpParMatObj2& copyFrom);

public:
	MatType A;

/////////////// everything below this appears in python interface:
//INTERFACE_INCLUDE_BEGIN
public:
	pySpParMatObj2();
	pySpParMatObj2(int64_t m, int64_t n, pyDenseParVec* rows, pyDenseParVec* cols, pyDenseParVecObj2* vals);

public:
	//int64_t getnnz();
	int64_t getnee();
	int64_t getnrow() const;
	int64_t getncol() const;
	
public:	
	void load(const char* filename, bool pario);
	void save(const char* filename);
	
	//double GenGraph500Edges(int scale, pyDenseParVec* pyDegrees = NULL, int EDGEFACTOR = 16);
	//double GenGraph500Edges(int scale, pyDenseParVec& pyDegrees);
	
public:
	pySpParMatObj2 copy();

	pySpParMatObj2& assign(const pySpParMatObj2& other);
	pySpParMatObj2 SubsRef(const pyDenseParVec& rows, const pyDenseParVec& cols, bool inPlace, op::UnaryPredicateObj* matFilter);
	
	int64_t removeSelfLoops();
	
	void Apply(op::UnaryFunctionObj* f);
	void DimWiseApply(int dim, const pyDenseParVecObj2& values, op::BinaryFunctionObj* f);
	pySpParMatObj2 Keep(op::UnaryPredicateObj* pred, bool inPlace);
	
	// Be wary of identity value with min()/max()!!!!!!!
	void Reduce(int dim, pyDenseParVec     *ret, op::BinaryFunctionObj* bf, op::UnaryFunctionObj* uf, double identity = 0);
	void Reduce(int dim, pyDenseParVecObj2 *ret, op::BinaryFunctionObj* bf, op::UnaryFunctionObj* uf, Obj2 identity = Obj2());
	//void Reduce(int dim, pyDenseParVecObj1 ret, op::BinaryFunctionObj* bf, op::UnaryFunctionObj* uf, Obj1 identity = Obj1());
	
	void Transpose();

	void Find(pyDenseParVec* outrows, pyDenseParVec* outcols, pyDenseParVecObj2* outvals) const;
public:
	pySpParVec     SpMV(const pySpParVec&     x, op::SemiringObj* sring);
	pySpParVecObj2 SpMV(const pySpParVecObj2& x, op::SemiringObj* sring);
	pySpParVecObj1 SpMV(const pySpParVecObj1& x, op::SemiringObj* sring);
	pyDenseParVec     SpMV(const pyDenseParVec&     x, op::SemiringObj* sring);
	pyDenseParVecObj2 SpMV(const pyDenseParVecObj2& x, op::SemiringObj* sring);
	pyDenseParVecObj1 SpMV(const pyDenseParVecObj1& x, op::SemiringObj* sring);

	void SpMV_inplace(pySpParVec&     x, op::SemiringObj* sring);
	void SpMV_inplace(pySpParVecObj2& x, op::SemiringObj* sring);
	void SpMV_inplace(pySpParVecObj1& x, op::SemiringObj* sring);
	void SpMV_inplace(pyDenseParVec&     x, op::SemiringObj* sring);
	void SpMV_inplace(pyDenseParVecObj2& x, op::SemiringObj* sring);
	void SpMV_inplace(pyDenseParVecObj1& x, op::SemiringObj* sring);

	void Square(op::SemiringObj* sring);
	pySpParMat     SpGEMM(pySpParMat     &other, op::SemiringObj* sring);
	pySpParMatBool SpGEMM(pySpParMatBool &other, op::SemiringObj* sring);
	pySpParMatObj2 SpGEMM(pySpParMatObj2 &other, op::SemiringObj* sring);
	pySpParMatObj1 SpGEMM(pySpParMatObj1 &other, op::SemiringObj* sring);

public:
	static int Column() { return ::Column; }
	static int Row() { return ::Row; }
};

pySpParMatObj2 EWiseApply(const pySpParMatObj2& A, const pySpParMat&     B, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, const Obj2& ANull, double BNull, bool allowIntersect, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred);
pySpParMatObj2 EWiseApply(const pySpParMatObj2& A, const pySpParMatBool& B, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, const Obj2& ANull, bool   BNull, bool allowIntersect, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred);
pySpParMatObj2 EWiseApply(const pySpParMatObj2& A, const pySpParMatObj2& B, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, const Obj2& ANull, const Obj2& BNull, bool allowIntersect, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred);
pySpParMatObj2 EWiseApply(const pySpParMatObj2& A, const pySpParMatObj1& B, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, const Obj2& ANull, const Obj1& BNull, bool allowIntersect, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred);

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
DECLARE_PROMOTE(pySpParMatObj2::MatType, pySpParMatObj2::MatType, pySpParMatObj2::MatType)
DECLARE_PROMOTE(pySpParMatObj2::DCColsType, pySpParMatObj2::DCColsType, pySpParMatObj2::DCColsType)

template <> struct promote_trait< SpDCCols<int64_t,Obj2> , SpDCCols<int64_t,Obj1> >
    {                                           
        typedef SpDCCols<int64_t,Obj1> T_promote;
    };
template <> struct promote_trait< SpDCCols<int64_t,Obj2> , SpDCCols<int64_t,doubleint> >
    {                                           
        typedef SpDCCols<int64_t,doubleint> T_promote;
    };
///////
template <> struct promote_trait< SpDCCols<int64_t,Obj2> , SpDCCols<int64_t,bool> >       
    {                                           
        typedef SpDCCols<int64_t,Obj2> T_promote;                    
    };

template <> struct promote_trait< SpDCCols<int64_t,bool> , SpDCCols<int64_t,Obj2> >       
    {                                           
        typedef SpDCCols<int64_t,Obj2> T_promote;                    
    };

// Based on what's in CombBLAS/SpDCCols.h:
template <class NIT, class NNT>  struct create_trait< SpDCCols<int64_t, Obj2> , NIT, NNT >
    {
        typedef SpDCCols<NIT,NNT> T_inferred;
    };


#endif
