#ifndef PY_SP_PAR_MAT_BOOL_H
#define PY_SP_PAR_MAT_BOOL_H

#include "pyCombBLAS.h"

//INTERFACE_INCLUDE_BEGIN
class pySpParMatBool {
//INTERFACE_INCLUDE_END
protected:

	typedef SpParMat < int64_t, bool, SpDCCols<int64_t,bool> > PSpMat_Bool;
	typedef SpParMat < int64_t, int, SpDCCols<int64_t,int> > PSpMat_Int;
	typedef SpParMat < int64_t, int64_t, SpDCCols<int64_t,int64_t> > PSpMat_Int64;

public:
	typedef int64_t INDEXTYPE;
	typedef bool NUMTYPE;
	typedef SpDCCols<INDEXTYPE,NUMTYPE> DCColsType;
	typedef SpParMat < INDEXTYPE, NUMTYPE, DCColsType > PSpMat_DoubleInt;
	typedef PSpMat_DoubleInt MatType;
	
public:
	
	pySpParMatBool(MatType other);

public:
	MatType A;

/////////////// everything below this appears in python interface:
//INTERFACE_INCLUDE_BEGIN
public:
	pySpParMatBool();
	pySpParMatBool(int64_t m, int64_t n, pyDenseParVec* rows, pyDenseParVec* cols, pyDenseParVec* vals);
	
	pySpParMatBool(const pySpParMat    & copyStructureFrom);
	pySpParMatBool(const pySpParMatBool& copyStructureFrom);
	pySpParMatBool(const pySpParMatObj1& copyStructureFrom);
	pySpParMatBool(const pySpParMatObj2& copyStructureFrom);

public:
	int64_t getnnz();
	int64_t getnee();
	int64_t getnrow();
	int64_t getncol();
	
public:	
	void load(const char* filename, bool pario);
	void save(const char* filename);
	
	double GenGraph500Edges(int scale, pyDenseParVec* pyDegrees = NULL, int EDGEFACTOR=16, bool delIsolated=true, double a=.57, double b=.19, double c=.19, double d=.05);
	//double GenGraph500Edges(int scale, pyDenseParVec& pyDegrees);
	
public:
	pySpParMatBool copy();

	pySpParMatBool& assign(const pySpParMatBool& other);
	pySpParMatBool SpGEMM(pySpParMatBool& other);
	pySpParMatBool SubsRef(const pyDenseParVec& rows, const pyDenseParVec& cols, bool inPlace, op::UnaryPredicateObj* matFilter);
	
	int64_t removeSelfLoops();
	
	void Apply(op::UnaryFunction* f);
	//void DimWiseApply(int dim, const pyDenseParVec& values, op::BinaryFunctionObj* f); // Not enough CombBLAS support
	pySpParMatBool Keep(op::UnaryPredicateObj* f, bool inPlace);

	int64_t Count(op::UnaryFunction* pred);
	
	// Be wary of identity value with min()/max()!!!!!!!
	pyDenseParVec Reduce(int dim, op::BinaryFunction* f, double identity = 0);
	pyDenseParVec Reduce(int dim, op::BinaryFunction* bf, op::UnaryFunction* uf, double identity = 0);
	void Reduce(int dim, pyDenseParVec* ret, op::BinaryFunctionObj* bf, op::UnaryFunctionObj* uf, double identity = 0);

	void Transpose();
	//void EWiseMult(pySpParMatBool rhs, bool exclude);

	void Find(pyDenseParVec* outrows, pyDenseParVec* outcols, pyDenseParVec* outvals) const;
public:
	pySpParVec     SpMV(const pySpParVec&     x, op::SemiringObj* sring);
	pySpParVecObj1 SpMV(const pySpParVecObj1& x, op::SemiringObj* sring);
	pySpParVecObj2 SpMV(const pySpParVecObj2& x, op::SemiringObj* sring);
	pyDenseParVec     SpMV(const pyDenseParVec&     x, op::SemiringObj* sring);
	pyDenseParVecObj1 SpMV(const pyDenseParVecObj1& x, op::SemiringObj* sring);
	pyDenseParVecObj2 SpMV(const pyDenseParVecObj2& x, op::SemiringObj* sring);

	void SpMV_inplace(pySpParVec&     x, op::SemiringObj* sring);
	void SpMV_inplace(pySpParVecObj1& x, op::SemiringObj* sring);
	void SpMV_inplace(pySpParVecObj2& x, op::SemiringObj* sring);
	void SpMV_inplace(pyDenseParVec&     x, op::SemiringObj* sring);
	void SpMV_inplace(pyDenseParVecObj1& x, op::SemiringObj* sring);
	void SpMV_inplace(pyDenseParVecObj2& x, op::SemiringObj* sring);

	void Square(op::SemiringObj* sring);
	pySpParMat     SpGEMM(pySpParMat     &other, op::SemiringObj* sring);
	pySpParMatBool SpGEMM(pySpParMatBool &other, op::SemiringObj* sring);
	pySpParMatObj1 SpGEMM(pySpParMatObj1 &other, op::SemiringObj* sring);
	pySpParMatObj2 SpGEMM(pySpParMatObj2 &other, op::SemiringObj* sring);


public:
	static int Column() { return ::Column; }
	static int Row() { return ::Row; }
};

pySpParMatBool EWiseApply(const pySpParMatBool& A, const pySpParMat&     B, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, bool ANull, double BNull, bool allowIntersect, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred);
pySpParMatBool EWiseApply(const pySpParMatBool& A, const pySpParMatBool& B, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, bool ANull, bool   BNull, bool allowIntersect, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred);
pySpParMatBool EWiseApply(const pySpParMatBool& A, const pySpParMatObj1& B, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, bool ANull, const Obj1& BNull, bool allowIntersect, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred);
pySpParMatBool EWiseApply(const pySpParMatBool& A, const pySpParMatObj2& B, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, bool ANull, const Obj2& BNull, bool allowIntersect, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred);

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
//DECLARE_PROMOTE(pySpParMatBool::MatType, pySpParMatBool::MatType, pySpParMatBool::MatType)
//DECLARE_PROMOTE(pySpParMatBool::DCColsType, pySpParMatBool::DCColsType, pySpParMatBool::DCColsType)



#endif
