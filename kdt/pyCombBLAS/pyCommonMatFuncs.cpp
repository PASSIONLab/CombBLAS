// This file implements functions common to all the matrices so that they can be implemented in one place.
// It is included by pySpParMat*.cpp files, and is not to be compiled by itself.
// There are some #defines that are defined in the file that #includes this one that controls some of the behavior. See the #include locations for what those are.

#ifdef MATCLASS

/******
EWiseApply
******/

MATCLASS EWiseApply(const MATCLASS& A, const pySpParMat&     B, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, NULL_PAR_TYPE ANull, double BNull, bool allowIntersect, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred)
{
	return MATCLASS(EWiseApply<MATCLASS::NUMTYPE, MATCLASS::DCColsType>(A.A, B.A,
		EWiseFilterOpAdapter<MATCLASS::NUMTYPE, MATCLASS::NUMTYPE, doubleint>     (  op, AFilterPred, BFilterPred, allowANulls, allowBNulls, allowIntersect, ANull, BNull),
		EWiseFilterDoOpAdapter<MATCLASS::NUMTYPE, doubleint>                      (doOp, AFilterPred, BFilterPred, allowANulls, allowBNulls, allowIntersect),
		allowANulls, allowBNulls, A_NULL_ARG, doubleint(BNull), allowIntersect, true));
}

MATCLASS EWiseApply(const MATCLASS& A, const pySpParMatBool& B, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, NULL_PAR_TYPE ANull, bool BNull, bool allowIntersect, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred)
{
	return MATCLASS(EWiseApply<MATCLASS::NUMTYPE, MATCLASS::DCColsType>(A.A, B.A,
		EWiseFilterOpAdapter<MATCLASS::NUMTYPE, MATCLASS::NUMTYPE, bool>     (  op, AFilterPred, BFilterPred, allowANulls, allowBNulls, allowIntersect, ANull, BNull),
		EWiseFilterDoOpAdapter<MATCLASS::NUMTYPE, bool>                      (doOp, AFilterPred, BFilterPred, allowANulls, allowBNulls, allowIntersect),
		allowANulls, allowBNulls, A_NULL_ARG, BNull, allowIntersect, true));
}

MATCLASS EWiseApply(const MATCLASS& A, const pySpParMatObj1& B, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, NULL_PAR_TYPE ANull, const Obj1& BNull, bool allowIntersect, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred)
{
	return MATCLASS(EWiseApply<MATCLASS::NUMTYPE, MATCLASS::DCColsType>(A.A, B.A,
		EWiseFilterOpAdapter<MATCLASS::NUMTYPE, MATCLASS::NUMTYPE, Obj1>     (  op, AFilterPred, BFilterPred, allowANulls, allowBNulls, allowIntersect, ANull, BNull),
		EWiseFilterDoOpAdapter<MATCLASS::NUMTYPE, Obj1>                      (doOp, AFilterPred, BFilterPred, allowANulls, allowBNulls, allowIntersect),
		allowANulls, allowBNulls, A_NULL_ARG, BNull, allowIntersect, true));
}

MATCLASS EWiseApply(const MATCLASS& A, const pySpParMatObj2& B, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, NULL_PAR_TYPE ANull, const Obj2& BNull, bool allowIntersect, op::UnaryPredicateObj* AFilterPred, op::UnaryPredicateObj* BFilterPred)
{
	return MATCLASS(EWiseApply<MATCLASS::NUMTYPE, MATCLASS::DCColsType>(A.A, B.A,
		EWiseFilterOpAdapter<MATCLASS::NUMTYPE, MATCLASS::NUMTYPE, Obj2>     (  op, AFilterPred, BFilterPred, allowANulls, allowBNulls, allowIntersect, ANull, BNull),
		EWiseFilterDoOpAdapter<MATCLASS::NUMTYPE, Obj2>                      (doOp, AFilterPred, BFilterPred, allowANulls, allowBNulls, allowIntersect),
		allowANulls, allowBNulls, A_NULL_ARG, BNull, allowIntersect, true));
}


/******
SEMIRING routines, SpMV, SpGEMM, Square
******/

//////// SpMV //////////////

template <class VEC>
void PlusTimesSpMVDoer(const MATCLASS::MatType& A, const VEC& x, VEC& ret)
{
	::SpMV< PlusTimesSRing<MATCLASS::NUMTYPE, MATCLASS::NUMTYPE > >(A, x.v, ret.v, false );
}

template <class VEC>
void PlusTimesSpMVThrower(const MATCLASS::MatType& A, const VEC& x, VEC& ret)
{
	throw string("built-in timesplus semiring not supported for objects.");
}


template <class VECTYPE, class VEC, typename PLUSTIMESFUNC>
VEC SpMV_worker(const MATCLASS::MatType& A, const VEC& x, op::SemiringObj* sring, PLUSTIMESFUNC PlusTimesFunc)
{
	if (sring->getType() == op::SemiringObj::SECONDMAX)
	{
		VEC ret(0);
		// Can't use the CombBLAS Select2ndSRing reliably because it uses MPI_MAX for the MPI_Op, which strictly speaking is not defined for Objs
		//::SpMV< Select2ndSRing<MATCLASS::NUMTYPE, VECTYPE, VECTYPE> >(A, x.v, ret.v, false );
		SRFilterHelper<VECTYPE, MATCLASS::NUMTYPE, VECTYPE>::setFilterX(sring->left_filter);
		SRFilterHelper<VECTYPE, MATCLASS::NUMTYPE, VECTYPE>::setFilterY(sring->right_filter);
		::SpMV< PCBSelect2ndSRing<MATCLASS::NUMTYPE, VECTYPE, VECTYPE> >(A, x.v, ret.v, false );
		SRFilterHelper<VECTYPE, MATCLASS::NUMTYPE, VECTYPE>::setFilterX(NULL);
		SRFilterHelper<VECTYPE, MATCLASS::NUMTYPE, VECTYPE>::setFilterY(NULL);
		return ret;
	}
	else if (sring->getType() == op::SemiringObj::TIMESPLUS)
	{
		VEC ret(0);
#ifndef MATCLASS_OBJ
		PlusTimesFunc(A, x, ret);
#else
		throw string("built-in timesplus semiring not supported for objects.");
#endif
		return ret;
	}
	else
	{
		sring->enableSemiring();
		VEC ret(0);
		::SpMV< op::SemiringObjTemplArg<MATCLASS::NUMTYPE, VECTYPE, VECTYPE> >(A, x.v, ret.v, false );
		sring->disableSemiring();
		return ret;
	}
}

template <class VECTYPE, class VEC, typename PLUSTIMESFUNC>
void SpMV_worker_inplace(const MATCLASS::MatType& A, VEC& x, op::SemiringObj* sring, PLUSTIMESFUNC PlusTimesFunc)
{
	if (sring->getType() == op::SemiringObj::SECONDMAX)
	{
		SRFilterHelper<VECTYPE, MATCLASS::NUMTYPE, VECTYPE>::setFilterX(sring->left_filter);
		SRFilterHelper<VECTYPE, MATCLASS::NUMTYPE, VECTYPE>::setFilterY(sring->right_filter);
		::SpMV< PCBSelect2ndSRing<MATCLASS::NUMTYPE, VECTYPE, VECTYPE> >(A, x.v, x.v, false );
		SRFilterHelper<VECTYPE, MATCLASS::NUMTYPE, VECTYPE>::setFilterX(NULL);
		SRFilterHelper<VECTYPE, MATCLASS::NUMTYPE, VECTYPE>::setFilterY(NULL);
	}
	else if (sring->getType() == op::SemiringObj::TIMESPLUS)
	{
#ifndef MATCLASS_OBJ
		PlusTimesFunc(A, x, x);
#else
		throw string("built-in timesplus semiring not supported for objects.");
#endif
	}
	else
	{
		sring->enableSemiring();
		::SpMV< op::SemiringObjTemplArg<MATCLASS::NUMTYPE, VECTYPE, VECTYPE> >(A, x.v, x.v, false );
		sring->disableSemiring();
	}
}

pySpParVec     MATCLASS::SpMV(const pySpParVec&     x, op::SemiringObj* sring)
{
#ifndef MATCLASS_OBJ
	return ::SpMV_worker<doubleint>(A, x, sring, PlusTimesSpMVDoer<pySpParVec>);
#else
	return ::SpMV_worker<doubleint>(A, x, sring, PlusTimesSpMVThrower<pySpParVec>);
#endif
}
pySpParVecObj1 MATCLASS::SpMV(const pySpParVecObj1& x, op::SemiringObj* sring) { return ::SpMV_worker<Obj1>(A, x, sring, PlusTimesSpMVThrower<pySpParVecObj1>); }
pySpParVecObj2 MATCLASS::SpMV(const pySpParVecObj2& x, op::SemiringObj* sring) { return ::SpMV_worker<Obj2>(A, x, sring, PlusTimesSpMVThrower<pySpParVecObj2>); }

void MATCLASS::SpMV_inplace(pySpParVec&     x, op::SemiringObj* sring)
{
#ifndef MATCLASS_OBJ
	::SpMV_worker_inplace<doubleint>(A, x, sring, PlusTimesSpMVDoer<pySpParVec>);
#else
	::SpMV_worker_inplace<doubleint>(A, x, sring, PlusTimesSpMVThrower<pySpParVec>);
#endif
}
void MATCLASS::SpMV_inplace(pySpParVecObj1& x, op::SemiringObj* sring) { ::SpMV_worker_inplace<Obj1>(A, x, sring, PlusTimesSpMVThrower<pySpParVecObj1>); }
void MATCLASS::SpMV_inplace(pySpParVecObj2& x, op::SemiringObj* sring) { ::SpMV_worker_inplace<Obj2>(A, x, sring, PlusTimesSpMVThrower<pySpParVecObj2>); }

#if 0
// these don't work yet because the CombBLAS dense vector SpMV hasn't been updated like the sparse vector one has.
pyDenseParVec     MATCLASS::SpMV(const pyDenseParVec&     x, op::SemiringObj* sring) { return ::SpMV_worker<doubleint>(A, x, sring); }
pyDenseParVecObj1 MATCLASS::SpMV(const pyDenseParVecObj1& x, op::SemiringObj* sring) { return ::SpMV_worker<Obj1>(A, x, sring); }
pyDenseParVecObj2 MATCLASS::SpMV(const pyDenseParVecObj2& x, op::SemiringObj* sring) { return ::SpMV_worker<Obj2>(A, x, sring); }

void MATCLASS::SpMV_inplace(pyDenseParVec&     x, op::SemiringObj* sring) { ::SpMV_worker_inplace<doubleint>(A, x, sring); }
void MATCLASS::SpMV_inplace(pyDenseParVecObj1& x, op::SemiringObj* sring) { ::SpMV_worker_inplace<Obj1>(A, x, sring); }
void MATCLASS::SpMV_inplace(pyDenseParVecObj2& x, op::SemiringObj* sring) { ::SpMV_worker_inplace<Obj2>(A, x, sring); }

#else
pyDenseParVec     MATCLASS::SpMV(const pyDenseParVec&     x, op::SemiringObj* sring) { throw string("Dense CombBLAS SpMV is broken (not updated to handle mixed types like the sprase vector SpMV was)"); }
pyDenseParVecObj1 MATCLASS::SpMV(const pyDenseParVecObj1& x, op::SemiringObj* sring) { throw string("Dense CombBLAS SpMV is broken (not updated to handle mixed types like the sprase vector SpMV was)"); }
pyDenseParVecObj2 MATCLASS::SpMV(const pyDenseParVecObj2& x, op::SemiringObj* sring) { throw string("Dense CombBLAS SpMV is broken (not updated to handle mixed types like the sprase vector SpMV was)"); }

void MATCLASS::SpMV_inplace(pyDenseParVec&     x, op::SemiringObj* sring) { throw string("Dense CombBLAS SpMV is broken (not updated to handle mixed types like the sprase vector SpMV was)"); }
void MATCLASS::SpMV_inplace(pyDenseParVecObj1& x, op::SemiringObj* sring) { throw string("Dense CombBLAS SpMV is broken (not updated to handle mixed types like the sprase vector SpMV was)"); }
void MATCLASS::SpMV_inplace(pyDenseParVecObj2& x, op::SemiringObj* sring) { throw string("Dense CombBLAS SpMV is broken (not updated to handle mixed types like the sprase vector SpMV was)"); }
#endif 

/*
void MATCLASS::SpMV_inplace(pyDenseParVec&     x, op::SemiringObj* sring) { throw string("Mixed type dense SpMV not supported yet."); }
void MATCLASS::SpMV_inplace(pyDenseParVecObj1& x, op::SemiringObj* sring)
{
	if (sring == NULL)
	{
		cout << "You must supply a semiring for SpMV!" << endl;
	}
	else
	{
		sring->enableSemiring();
		x.v = ::SpMV< op::SemiringObjTemplArg<Obj1, Obj1, Obj1> >(A, x.v);
		sring->disableSemiring();
	}
}
void MATCLASS::SpMV_inplace(pyDenseParVecObj2& x, op::SemiringObj* sring) { throw string("Mixed type dense SpMV not supported yet."); }

pyDenseParVec     MATCLASS::SpMV(const pyDenseParVec&     x, op::SemiringObj* sring)
{
	throw string("Mixed type dense SpMV not supported yet.");
	return pyDenseParVec(getnrow(), 0, 0);
}

pyDenseParVecObj1 MATCLASS::SpMV(const pyDenseParVecObj1& x, op::SemiringObj* sring)
{
	if (sring == NULL)
	{
		cout << "You must supply a semiring for SpMV!" << endl;
		return pyDenseParVecObj1(getnrow(), Obj1());
	}
	else if (sring->getType() == op::SemiringObj::SECONDMAX)
	{
		return pyDenseParVecObj1( ::SpMV< Select2ndSRing<NUMTYPE, NUMTYPE, NUMTYPE > >(A, x.v) );
	}
	else
	{
		sring->enableSemiring();
		pyDenseParVecObj1 ret( ::SpMV< op::SemiringObjTemplArg<Obj1, Obj1, Obj1> >(A, x.v) );
		sring->disableSemiring();
		return ret;
	}
}

pyDenseParVecObj2 MATCLASS::SpMV(const pyDenseParVecObj2& x, op::SemiringObj* sring)
{
	cout << "Mixed type dense SpMV not supported yet." << endl;
	//cout << "You must supply a semiring for SpMV!" << endl;
	return pyDenseParVecObj2(getnrow(), Obj2());
}
*/

//////// Square /////////////

void MATCLASS::Square(op::SemiringObj* sring)
{
	if (sring->getType() == op::SemiringObj::SECONDMAX)
	{
		A.Square< Select2ndSRing<NUMTYPE, NUMTYPE, NUMTYPE > >();
	}
	else if (sring->getType() == op::SemiringObj::TIMESPLUS)
	{
#ifndef MATCLASS_OBJ
		A.Square< PlusTimesSRing<NUMTYPE, NUMTYPE > >();
#else
		throw string("built-in timesplus semiring not supported for objects.");
#endif
	}
	else 
	{
		sring->enableSemiring();
		A.Square<op::SemiringObjTemplArg<NUMTYPE, NUMTYPE, NUMTYPE> >();
		sring->disableSemiring();
	}
}

///////// SpGEMM //////////////

pySpParMat MATCLASS::SpGEMM(pySpParMat& other, op::SemiringObj* sring)
{
	if (sring->getType() == op::SemiringObj::SECONDMAX)
	{
		pySpParMat ret;
		PSpGEMM<Select2ndSRing<MATCLASS::NUMTYPE, doubleint, doubleint > >(A, other.A, ret.A);
		return ret;
	}
	else if (sring->getType() == op::SemiringObj::TIMESPLUS)
	{
		pySpParMat ret;
#ifndef MATCLASS_OBJ
		PSpGEMM<PlusTimesSRing<MATCLASS::NUMTYPE, doubleint > >(A, other.A, ret.A);
#else
		throw string("built-in timesplus semiring not supported for objects.");
#endif
		return ret;
	}
	else
	{
		pySpParMat ret;
		sring->enableSemiring();
		PSpGEMM<op::SemiringObjTemplArg<MATCLASS::NUMTYPE, doubleint, doubleint> >(A, other.A, ret.A);
		sring->disableSemiring();
		return ret;
	}
}

pySpParMatBool MATCLASS::SpGEMM(pySpParMatBool& other, op::SemiringObj* sring)
{
	if (sring->getType() == op::SemiringObj::SECONDMAX)
	{
		pySpParMatBool ret;
		PSpGEMM<Select2ndSRing<MATCLASS::NUMTYPE, bool, bool > >(A, other.A, ret.A);
		return ret;
	}
	else if (sring->getType() == op::SemiringObj::TIMESPLUS)
	{
		pySpParMatBool ret;
#ifndef MATCLASS_OBJ
		PSpGEMM<PlusTimesSRing<MATCLASS::NUMTYPE, bool > >(A, other.A, ret.A);
#else
		throw string("built-in timesplus semiring not supported for objects.");
#endif
		return ret;
	}
	else
	{
		pySpParMatBool ret;
		sring->enableSemiring();
		PSpGEMM<op::SemiringObjTemplArg<MATCLASS::NUMTYPE, bool, bool> >(A, other.A, ret.A);
		sring->disableSemiring();
		return ret;
	}
}

pySpParMatObj1 MATCLASS::SpGEMM(pySpParMatObj1& other, op::SemiringObj* sring)
{
	if (sring->getType() == op::SemiringObj::SECONDMAX)
	{
		pySpParMatObj1 ret;
		PSpGEMM<Select2ndSRing<MATCLASS::NUMTYPE, Obj1, Obj1 > >(A, other.A, ret.A);
		return ret;
	}
	else if (sring->getType() == op::SemiringObj::TIMESPLUS)
	{
		throw string("built-in timesplus semiring not supported for objects.");
		return pySpParMatObj1();
	}
	else
	{
		pySpParMatObj1 ret;
		sring->enableSemiring();
		PSpGEMM<op::SemiringObjTemplArg<MATCLASS::NUMTYPE, Obj1, Obj1> >(A, other.A, ret.A);
		sring->disableSemiring();
		return ret;
	}
}

pySpParMatObj2 MATCLASS::SpGEMM(pySpParMatObj2& other, op::SemiringObj* sring)
{
	if (sring->getType() == op::SemiringObj::SECONDMAX)
	{
		pySpParMatObj2 ret;
		PSpGEMM<Select2ndSRing<MATCLASS::NUMTYPE, Obj2, Obj2 > >(A, other.A, ret.A);
		return ret;
	}
	else if (sring->getType() == op::SemiringObj::TIMESPLUS)
	{
		throw string("built-in timesplus semiring not supported for objects.");
		return pySpParMatObj2();
	}
	else
	{
		pySpParMatObj2 ret;
		sring->enableSemiring();
		PSpGEMM<op::SemiringObjTemplArg<MATCLASS::NUMTYPE, Obj2, Obj2> >(A, other.A, ret.A);
		sring->disableSemiring();
		return ret;
	}
}



/******
OTHER common routines
******/

void MATCLASS::Transpose()
{
	A.Transpose();
}



#endif