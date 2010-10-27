#ifndef DIGRAPH_H
#define DIGRAPH_H

#include "../../CombBLAS/SpParVec.h"
#include "../../CombBLAS/SpTuples.h"
#include "../../CombBLAS/SpDCCols.h"
#include "../../CombBLAS/SpParMat.h"
#include "../../CombBLAS/DenseParMat.h"
#include "../../CombBLAS/DenseParVec.h"

class DiGraph {
protected:

	template <class NT>
	class PSpMat 
	{ 
	public: 
		typedef SpDCCols < int, NT > DCCols;
		typedef SpParMat < int, NT, DCCols > MPI_DCCols;
	};

	PSpMat<double>::MPI_DCCols g;

/////////////// everything below this appears in declaration for wrapper:
public:
	static void init();
	static void finalize();

/////////////// everything below this appears in python interface:
public:
	DiGraph();

public:
	int nedges();
	int nverts();
	
public:	
	void load(const char* filename);
	
};

void finalize();

#endif
