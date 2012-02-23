// This file implements functions common to all the vectors so that they can be implemented in one place.
// It is included by py*ParVec*.cpp files, and is not to be compiled by itself.

#ifdef VECCLASS

#ifdef DENSE_VEC
//////////////

// dense with dense/sparse function versions

void VECCLASS::EWiseApply(const pyDenseParVecObj1& other, op::BinaryFunctionObj *op, op::BinaryPredicateObj *doOp, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred)
{
	v.EWiseApply(other.v,
		EWiseFilterOpAdapter<VECCLASS::NUMTYPE, VECCLASS::NUMTYPE, Obj1>     (  op, AFilterPred, BFilterPred, true, true, true, VECCLASS::NUMTYPE(), Obj1()),
		EWiseFilterDoOpAdapter<VECCLASS::NUMTYPE, Obj1>                      (doOp, AFilterPred, BFilterPred, true, true, true),
		true);
}

void VECCLASS::EWiseApply(const pyDenseParVecObj2& other, op::BinaryFunctionObj *op, op::BinaryPredicateObj *doOp, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred)
{
	v.EWiseApply(other.v,
		EWiseFilterOpAdapter<VECCLASS::NUMTYPE, VECCLASS::NUMTYPE, Obj2>     (  op, AFilterPred, BFilterPred, true, true, true, VECCLASS::NUMTYPE(), Obj2()),
		EWiseFilterDoOpAdapter<VECCLASS::NUMTYPE, Obj2>                      (doOp, AFilterPred, BFilterPred, true, true, true),
		true);
}

void VECCLASS::EWiseApply(const pyDenseParVec&     other, op::BinaryFunctionObj *op, op::BinaryPredicateObj *doOp, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred)
{
	v.EWiseApply(other.v,
		EWiseFilterOpAdapter<VECCLASS::NUMTYPE, VECCLASS::NUMTYPE, doubleint>     (  op, AFilterPred, BFilterPred, true, true, true, VECCLASS::NUMTYPE(), doubleint()),
		EWiseFilterDoOpAdapter<VECCLASS::NUMTYPE, doubleint>                      (doOp, AFilterPred, BFilterPred, true, true, true),
		true);
}

void VECCLASS::EWiseApply(const pySpParVecObj1& other, op::BinaryFunctionObj *op, op::BinaryPredicateObj *doOp, bool doNulls, Obj1 nullValue, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred)
{
	v.EWiseApply(other.v,
		EWiseFilterOpAdapter<VECCLASS::NUMTYPE, VECCLASS::NUMTYPE, Obj1>     (  op, AFilterPred, BFilterPred, true, true, true, VECCLASS::NUMTYPE(), nullValue),
		EWiseFilterDoOpAdapter<VECCLASS::NUMTYPE, Obj1>                      (doOp, AFilterPred, BFilterPred, true, true, true),
		doNulls, nullValue, true);
}

void VECCLASS::EWiseApply(const pySpParVecObj2& other, op::BinaryFunctionObj *op, op::BinaryPredicateObj *doOp, bool doNulls, Obj2 nullValue, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred)
{
	v.EWiseApply(other.v,
		EWiseFilterOpAdapter<VECCLASS::NUMTYPE, VECCLASS::NUMTYPE, Obj2>     (  op, AFilterPred, BFilterPred, true, true, true, VECCLASS::NUMTYPE(), nullValue),
		EWiseFilterDoOpAdapter<VECCLASS::NUMTYPE, Obj2>                      (doOp, AFilterPred, BFilterPred, true, true, true),
		doNulls, nullValue, true);
}

void VECCLASS::EWiseApply(const pySpParVec&     other, op::BinaryFunctionObj *op, op::BinaryPredicateObj *doOp, bool doNulls, double nullValue, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred)
{
	v.EWiseApply(other.v,
		EWiseFilterOpAdapter<VECCLASS::NUMTYPE, VECCLASS::NUMTYPE, doubleint>     (  op, AFilterPred, BFilterPred, true, true, true, VECCLASS::NUMTYPE(), doubleint(nullValue)),
		EWiseFilterDoOpAdapter<VECCLASS::NUMTYPE, doubleint>                      (doOp, AFilterPred, BFilterPred, true, true, true),
		doNulls, doubleint(nullValue), true);
}

////// dense with dense/sparse predicate versions

#ifndef OBJ_VEC
// These need to exist for Obj vectors somehow as well, but right now there's no clean way to do it because the return type needs to be double but the op is in-place.

void VECCLASS::EWiseApply(const pyDenseParVecObj1& other, op::BinaryPredicateObj *op, op::BinaryPredicateObj *doOp, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred)
{
	v.EWiseApply(other.v,
		EWiseFilterOpAdapter<VECCLASS::NUMTYPE, VECCLASS::NUMTYPE, Obj1, op::BinaryPredicateObj>     (  op, AFilterPred, BFilterPred, true, true, true, VECCLASS::NUMTYPE(), Obj1()),
		EWiseFilterDoOpAdapter<VECCLASS::NUMTYPE, Obj1>                                              (doOp, AFilterPred, BFilterPred, true, true, true),
		true);
/*
	if (doOp != NULL)
		v.EWiseApply(other.v, *op, *doOp);
	else
		v.EWiseApply(other.v, *op, retTrue<VECCLASS::NUMTYPE, Obj1>);
*/
}

void VECCLASS::EWiseApply(const pyDenseParVecObj2& other, op::BinaryPredicateObj *op, op::BinaryPredicateObj *doOp, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred)
{
	v.EWiseApply(other.v,
		EWiseFilterOpAdapter<VECCLASS::NUMTYPE, VECCLASS::NUMTYPE, Obj2, op::BinaryPredicateObj>     (  op, AFilterPred, BFilterPred, true, true, true, VECCLASS::NUMTYPE(), Obj2()),
		EWiseFilterDoOpAdapter<VECCLASS::NUMTYPE, Obj2>                                              (doOp, AFilterPred, BFilterPred, true, true, true),
		true);
}

void VECCLASS::EWiseApply(const pyDenseParVec&     other, op::BinaryPredicateObj *op, op::BinaryPredicateObj *doOp, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred)
{
	v.EWiseApply(other.v,
		EWiseFilterOpAdapter<VECCLASS::NUMTYPE, VECCLASS::NUMTYPE, doubleint, op::BinaryPredicateObj>     (  op, AFilterPred, BFilterPred, true, true, true, VECCLASS::NUMTYPE(), doubleint()),
		EWiseFilterDoOpAdapter<VECCLASS::NUMTYPE, doubleint>                                              (doOp, AFilterPred, BFilterPred, true, true, true),
		true);
}

void VECCLASS::EWiseApply(const pySpParVecObj1& other, op::BinaryPredicateObj *op, op::BinaryPredicateObj *doOp, bool doNulls, Obj1 nullValue, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred)
{
	v.EWiseApply(other.v,
		EWiseFilterOpAdapter<VECCLASS::NUMTYPE, VECCLASS::NUMTYPE, Obj1, op::BinaryPredicateObj>     (  op, AFilterPred, BFilterPred, true, true, true, VECCLASS::NUMTYPE(), nullValue),
		EWiseFilterDoOpAdapter<VECCLASS::NUMTYPE, Obj1>                                              (doOp, AFilterPred, BFilterPred, true, true, true),
		doNulls, nullValue, true);
}

void VECCLASS::EWiseApply(const pySpParVecObj2& other, op::BinaryPredicateObj *op, op::BinaryPredicateObj *doOp, bool doNulls, Obj2 nullValue, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred)
{
	v.EWiseApply(other.v,
		EWiseFilterOpAdapter<VECCLASS::NUMTYPE, VECCLASS::NUMTYPE, Obj2, op::BinaryPredicateObj>     (  op, AFilterPred, BFilterPred, true, true, true, VECCLASS::NUMTYPE(), nullValue),
		EWiseFilterDoOpAdapter<VECCLASS::NUMTYPE, Obj2>                                              (doOp, AFilterPred, BFilterPred, true, true, true),
		doNulls, nullValue, true);
}

void VECCLASS::EWiseApply(const pySpParVec&     other, op::BinaryPredicateObj *op, op::BinaryPredicateObj *doOp, bool doNulls, double nullValue, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred)
{
	v.EWiseApply(other.v,
		EWiseFilterOpAdapter<VECCLASS::NUMTYPE, VECCLASS::NUMTYPE, doubleint, op::BinaryPredicateObj>     (  op, AFilterPred, BFilterPred, true, true, true, VECCLASS::NUMTYPE(), doubleint(nullValue)),
		EWiseFilterDoOpAdapter<VECCLASS::NUMTYPE, doubleint>                                              (doOp, AFilterPred, BFilterPred, true, true, true),
		doNulls, doubleint(nullValue), true);
}

#endif

#else

VECCLASS EWiseApply(const VECCLASS& a, const pySpParVecObj1& b, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, NULL_PAR_TYPE ANull, Obj1 BNull, bool allowIntersect, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred)
{
	return VECCLASS(EWiseApply<VECCLASS::NUMTYPE>(a.v, b.v,
		EWiseFilterOpAdapter<VECCLASS::NUMTYPE, VECCLASS::NUMTYPE, Obj1>     (  op, AFilterPred, BFilterPred, allowANulls, allowBNulls, allowIntersect, A_NULL_ARG, BNull),
		EWiseFilterDoOpAdapter<VECCLASS::NUMTYPE, Obj1>                      (doOp, AFilterPred, BFilterPred, allowANulls, allowBNulls, allowIntersect),
		allowANulls, allowBNulls, A_NULL_ARG, BNull, allowIntersect, true));
}

VECCLASS EWiseApply(const VECCLASS& a, const pySpParVecObj2& b, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, NULL_PAR_TYPE ANull, Obj2 BNull, bool allowIntersect, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred)
{
	return VECCLASS(EWiseApply<VECCLASS::NUMTYPE>(a.v, b.v,
		EWiseFilterOpAdapter<VECCLASS::NUMTYPE, VECCLASS::NUMTYPE, Obj2>     (  op, AFilterPred, BFilterPred, allowANulls, allowBNulls, allowIntersect, A_NULL_ARG, BNull),
		EWiseFilterDoOpAdapter<VECCLASS::NUMTYPE, Obj2>                      (doOp, AFilterPred, BFilterPred, allowANulls, allowBNulls, allowIntersect),
		allowANulls, allowBNulls, A_NULL_ARG, BNull, allowIntersect, true));
}

VECCLASS EWiseApply(const VECCLASS& a, const pySpParVec&     b, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, NULL_PAR_TYPE ANull, double BNull, bool allowIntersect, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred)
{
	return VECCLASS(EWiseApply<VECCLASS::NUMTYPE>(a.v, b.v,
		EWiseFilterOpAdapter<VECCLASS::NUMTYPE, VECCLASS::NUMTYPE, doubleint>     (  op, AFilterPred, BFilterPred, allowANulls, allowBNulls, allowIntersect, A_NULL_ARG, BNull),
		EWiseFilterDoOpAdapter<VECCLASS::NUMTYPE, doubleint>                      (doOp, AFilterPred, BFilterPred, allowANulls, allowBNulls, allowIntersect),
		allowANulls, allowBNulls, A_NULL_ARG, doubleint(BNull), allowIntersect, true));
}

pySpParVec EWiseApply(const VECCLASS& a, const pySpParVecObj1& b, op::BinaryPredicateObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, NULL_PAR_TYPE ANull, Obj1 BNull, bool allowIntersect, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred)
{
	return pySpParVec(EWiseApply<doubleint>(a.v, b.v,
		EWiseFilterOpAdapter<doubleint, VECCLASS::NUMTYPE, Obj1, op::BinaryPredicateObj>     (  op, AFilterPred, BFilterPred, allowANulls, allowBNulls, allowIntersect, A_NULL_ARG, BNull),
		EWiseFilterDoOpAdapter<VECCLASS::NUMTYPE, Obj1>                                      (doOp, AFilterPred, BFilterPred, allowANulls, allowBNulls, allowIntersect),
		allowANulls, allowBNulls, A_NULL_ARG, BNull, allowIntersect, true));
}

pySpParVec EWiseApply(const VECCLASS& a, const pySpParVecObj2& b, op::BinaryPredicateObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, NULL_PAR_TYPE ANull, Obj2 BNull, bool allowIntersect, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred)
{
	return pySpParVec(EWiseApply<doubleint>(a.v, b.v,
		EWiseFilterOpAdapter<doubleint, VECCLASS::NUMTYPE, Obj2, op::BinaryPredicateObj>     (  op, AFilterPred, BFilterPred, allowANulls, allowBNulls, allowIntersect, A_NULL_ARG, BNull),
		EWiseFilterDoOpAdapter<VECCLASS::NUMTYPE, Obj2>                                      (doOp, AFilterPred, BFilterPred, allowANulls, allowBNulls, allowIntersect),
		allowANulls, allowBNulls, A_NULL_ARG, BNull, allowIntersect, true));
}

pySpParVec EWiseApply(const VECCLASS& a, const pySpParVec&     b, op::BinaryPredicateObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, NULL_PAR_TYPE ANull, double BNull, bool allowIntersect, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred)
{
	return pySpParVec(EWiseApply<doubleint>(a.v, b.v,
		EWiseFilterOpAdapter<doubleint, VECCLASS::NUMTYPE, doubleint, op::BinaryPredicateObj>     (  op, AFilterPred, BFilterPred, allowANulls, allowBNulls, allowIntersect, A_NULL_ARG, doubleint(BNull)),
		EWiseFilterDoOpAdapter<VECCLASS::NUMTYPE, doubleint>                                      (doOp, AFilterPred, BFilterPred, allowANulls, allowBNulls, allowIntersect),
		allowANulls, allowBNulls, A_NULL_ARG, doubleint(BNull), allowIntersect, true));
}


/////////// with Dense
VECCLASS EWiseApply(const VECCLASS& a, const pyDenseParVec& b, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, NULL_PAR_TYPE ANull, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred)
{
	return VECCLASS(EWiseApply<VECCLASS::NUMTYPE>(a.v, b.v,
		EWiseFilterOpAdapter<VECCLASS::NUMTYPE, VECCLASS::NUMTYPE, doubleint>     (  op, AFilterPred, BFilterPred, allowANulls, true, true, A_NULL_ARG, doubleint()),
		EWiseFilterDoOpAdapter<VECCLASS::NUMTYPE, doubleint>                      (doOp, AFilterPred, BFilterPred, allowANulls, true, true),
		allowANulls, A_NULL_ARG, true));
}

VECCLASS EWiseApply(const VECCLASS& a, const pyDenseParVecObj1& b, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, NULL_PAR_TYPE ANull, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred)
{
	return VECCLASS(EWiseApply<VECCLASS::NUMTYPE>(a.v, b.v,
		EWiseFilterOpAdapter<VECCLASS::NUMTYPE, VECCLASS::NUMTYPE, Obj1>     (  op, AFilterPred, BFilterPred, allowANulls, true, true, A_NULL_ARG, Obj1()),
		EWiseFilterDoOpAdapter<VECCLASS::NUMTYPE, Obj1>                      (doOp, AFilterPred, BFilterPred, allowANulls, true, true),
		allowANulls, A_NULL_ARG, true));
}

VECCLASS EWiseApply(const VECCLASS& a, const pyDenseParVecObj2& b, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, NULL_PAR_TYPE ANull, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred)
{
	return VECCLASS(EWiseApply<VECCLASS::NUMTYPE>(a.v, b.v,
		EWiseFilterOpAdapter<VECCLASS::NUMTYPE, VECCLASS::NUMTYPE, Obj2>     (  op, AFilterPred, BFilterPred, allowANulls, true, true, A_NULL_ARG, Obj2()),
		EWiseFilterDoOpAdapter<VECCLASS::NUMTYPE, Obj2>                      (doOp, AFilterPred, BFilterPred, allowANulls, true, true),
		allowANulls, A_NULL_ARG, true));
}

pySpParVec EWiseApply(const VECCLASS& a, const pyDenseParVec& b, op::BinaryPredicateObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, NULL_PAR_TYPE ANull, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred)
{
	return pySpParVec(EWiseApply<doubleint>(a.v, b.v,
		EWiseFilterOpAdapter<doubleint, VECCLASS::NUMTYPE, doubleint, op::BinaryPredicateObj>     (  op, AFilterPred, BFilterPred, allowANulls, true, true, A_NULL_ARG, doubleint()),
		EWiseFilterDoOpAdapter<VECCLASS::NUMTYPE, doubleint>                                      (doOp, AFilterPred, BFilterPred, allowANulls, true, true),
		allowANulls, A_NULL_ARG, true));
}

pySpParVec EWiseApply(const VECCLASS& a, const pyDenseParVecObj1& b, op::BinaryPredicateObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, NULL_PAR_TYPE ANull, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred)
{
	return pySpParVec(EWiseApply<doubleint>(a.v, b.v,
		EWiseFilterOpAdapter<doubleint, VECCLASS::NUMTYPE, Obj1, op::BinaryPredicateObj>     (  op, AFilterPred, BFilterPred, allowANulls, true, true, A_NULL_ARG, Obj1()),
		EWiseFilterDoOpAdapter<VECCLASS::NUMTYPE, Obj1>                                      (doOp, AFilterPred, BFilterPred, allowANulls, true, true),
		allowANulls, A_NULL_ARG, true));
}

pySpParVec EWiseApply(const VECCLASS& a, const pyDenseParVecObj2& b, op::BinaryPredicateObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, NULL_PAR_TYPE ANull, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred)
{
	return pySpParVec(EWiseApply<doubleint>(a.v, b.v,
		EWiseFilterOpAdapter<doubleint, VECCLASS::NUMTYPE, Obj2, op::BinaryPredicateObj>     (  op, AFilterPred, BFilterPred, allowANulls, true, true, A_NULL_ARG, Obj2()),
		EWiseFilterDoOpAdapter<VECCLASS::NUMTYPE, Obj2>                                      (doOp, AFilterPred, BFilterPred, allowANulls, true, true),
		allowANulls, A_NULL_ARG, true));
}


#endif


#endif
