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

#include "pyCombBLAS.h"
class pySpParMat;
class pySpParVec;
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
	pySpParVec(int64_t length);
	//pySpParVec(const pySpParMat& commSource);
	
	pyDenseParVec* dense() const;

public:
	int64_t getnnz() const;

	pySpParVec& operator+=(const pySpParVec& other);
	pySpParVec& operator-=(const pySpParVec& other);
	pySpParVec* copy();

	void SetElement(int64_t index, int64_t numx);	// element-wise assignment
	int64_t GetElement(int64_t index);
	
public:	
	void invert(); // "~";  almost equal to logical_not
	void abs();
	
	bool any() const; // any nonzeros
	bool all() const; // all nonzeros
	
	int64_t intersectSize(const pySpParVec& other);
	
	void printall();

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
	pyDenseParVec* FindInds_GreaterThan(int64_t value);
	
	pySpParVec* SpRef(const pySpParVec& ri);
	
	void setNumToInd();

public:
	static pySpParVec* zeros(int64_t howmany);
	static pySpParVec* range(int64_t howmany, int64_t start);
	
};

pySpParVec* EWiseMult(const pySpParVec& a, const pySpParVec& b, bool exclude);

pySpParVec* EWiseMult(const pySpParVec& a, const pyDenseParVec& b, bool exclude, int64_t zero);

#endif
