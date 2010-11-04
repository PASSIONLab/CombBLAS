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

#include "pySpParVec.h"

class pySpParVec;

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
	int nedges();
	int nverts();
	
public:	
	void load(const char* filename);
	void GenGraph500Edges(int scale);
	
public:
	pySpParVec* SpMV_SelMax(const pySpParVec& v);
	
};

extern "C" {
void init_pyCombBLAS_MPI();
}

void finalize();

#endif
