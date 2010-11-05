#ifndef PY_DENSE_PAR_VEC_H
#define PY_DENSE_PAR_VEC_H

#include "../../CombBLAS/SpParVec.h"
#include "../../CombBLAS/SpTuples.h"
#include "../../CombBLAS/SpDCCols.h"
#include "../../CombBLAS/SpParMat.h"
#include "../../CombBLAS/SpParVec.h"
#include "../../CombBLAS/DenseParMat.h"
#include "../../CombBLAS/DenseParVec.h"
#include "../../CombBLAS/SpParVec.h"
#include "../../CombBLAS/ParFriends.h"

#include "pyCombBLAS.h"
class pySpParMat;
class pySpParVec;
class pyDenseParVec;

class pyDenseParVec {
protected:

	DenseParVec<int64_t, int64_t> v;
	
	friend class pySpParVec;
	friend class pySpParMat;
	friend pySpParVec* EWiseMult(const pySpParVec& a, const pyDenseParVec& b, bool exclude, int64_t zero);

/////////////// everything below this appears in python interface:
public:
	pyDenseParVec();
	pyDenseParVec(int64_t size, int64_t id);
	//pyDenseParVec(const pySpParMat& commSource, int64_t zero);
	
public:
	int length() const;
	
	void add(const pyDenseParVec& other);
	void add(const pySpParVec& other);
	pyDenseParVec& operator+=(const pyDenseParVec & rhs);
	pyDenseParVec& operator-=(const pyDenseParVec & rhs);
	pyDenseParVec& operator=(const pyDenseParVec & rhs);
	pyDenseParVec* copy();
	
public:
	void invert(); // "~";  almost equal to logical_not
	void abs();
	
	bool anyNonzeros() const;
	bool allNonzeros() const;

	
public:	
	void load(const char* filename);
	
public:
	pySpParVec* FindInds_GreaterThan(int64_t value);
	
};

#endif
