#ifndef PY_SP_PAR_MAT_H
#define PY_SP_PAR_MAT_H

#include "../../CombBLAS/SpParVec.h"
#include "../../CombBLAS/SpTuples.h"
#include "../../CombBLAS/SpDCCols.h"
#include "../../CombBLAS/SpParMat.h"
#include "../../CombBLAS/SpParVec.h"
#include "../../CombBLAS/DenseParMat.h"
#include "../../CombBLAS/DenseParVec.h"
#include "../../CombBLAS/DistEdgeList.h"
#include "../../CombBLAS/ParFriends.h"
#include "../../CombBLAS/Semirings.h"

#include "pyCombBLAS.h"

class pySpParMat;
class pySpParVec;
class pyDenseParVec;


class pySpParMat {
protected:

	template <class NT>
	class PSpMat 
	{ 
	public: 
		typedef SpDCCols < int64_t, NT > DCCols;
		typedef SpParMat < int64_t, NT, DCCols > MPI_DCCols;
	};

	PSpMat<int>::MPI_DCCols A;
	
	friend class pySpParVec;

/////////////// everything below this appears in python interface:
public:
	pySpParMat();

public:
	int64_t getnnz();
	int64_t getnrow();
	int64_t getncol();
	
public:	
	void load(const char* filename);
	void GenGraph500Edges(int scale);
	
public:
	pySpParVec* FindIndsOfColsWithSumGreaterThan(int64_t gt);
	//pyDenseParVec* Reduce_ColumnSums();
	
public:
	pySpParVec* SpMV_SelMax(const pySpParVec& v);
	
};

extern "C" {
void init_pyCombBLAS_MPI();
}

void finalize();

#endif
