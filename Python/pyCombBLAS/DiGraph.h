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
	typedef SpDCCols < int, double > DCCols;
	typedef SpParMat < int, double, SpDCCols < int, double > > MPI_DCCols;
	
	//MPI_DCCols g;

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

#endif
