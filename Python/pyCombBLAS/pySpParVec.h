#ifndef PY_SP_PAR_VEC_H
#define PY_SP_PAR_VEC_H

#include "../../CombBLAS/SpParVec.h"
#include "../../CombBLAS/SpTuples.h"
#include "../../CombBLAS/SpDCCols.h"
#include "../../CombBLAS/SpParMat.h"
#include "../../CombBLAS/SpParVec.h"
#include "../../CombBLAS/DenseParMat.h"
#include "../../CombBLAS/DenseParVec.h"
#include "../../CombBLAS/ParFriends.h"
#include "../../CombBLAS/Semirings.h"

#include "pySpParMat.h"
#include "pyDenseParVec.h"

class pySpParMat;
class pyDenseParVec;

class pySpParVec {
protected:

	SpParVec<int64_t, int64_t> v;
	
	//pySpParVec(SpParVec<int64_t, int64_t> & in_v);
	
	friend class pySpParMat;
	friend class pyDenseParVec;
	
	friend pySpParVec* EWiseMult(const pySpParVec& a, const pySpParVec& b, bool exclude);
	friend pySpParVec* EWiseMult(const pySpParVec& a, const pyDenseParVec& b, bool exclude, int64_t zero);


/////////////// everything below this appears in python interface:
public:
	pySpParVec();
	//pySpParVec(const pySpParMat& commSource);

public:
	int64_t getnnz() const;
	
public:	
	const pySpParVec& add(const pySpParVec& other);
	void SetElement(int64_t index, int64_t numx);	// element-wise assignment


	const pySpParVec& subtract(const pySpParVec& other);
	void invert(); // "~";  almost equal to logical_not
	void abs();
	
	bool anyNonzeros() const;
	bool allNonzeros() const;
	
	int64_t intersectSize(const pySpParVec& other);

	/*
	def __or__(self, other):
		return blah;

	def __and__(self, other):
		return blah;

	def __eq__(self, other):
		return blah;

	def __ne__(self, other):
		return blah;

	def __getitem__(self, key)
		return blah;

	def __setitem__(self, key, value)
		return blah;

	def __int__(self):
		return blah;

	def bool(self):
		return blah;
	*/
	
public:	
	void load(const char* filename);

public:
	static pySpParVec* zeros(int64_t howmany);
	static pySpParVec* range(int64_t howmany, int64_t start);
	
};

pySpParVec* EWiseMult(const pySpParVec& a, const pySpParVec& b, bool exclude);

pySpParVec* EWiseMult(const pySpParVec& a, const pyDenseParVec& b, bool exclude, int64_t zero);

#endif
