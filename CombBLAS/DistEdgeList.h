/****************************************************************/
/* Sequential and Parallel Sparse Matrix Multiplication Library  /
/  version 2.3 --------------------------------------------------/
/  date: 10/28/2010 ---------------------------------------------/
/  author: Adam Lugowski (alugowski@cs.ucsb.edu) ----------------/
\****************************************************************/

#ifndef _DIST_EDGE_LIST_H_
#define _DIST_EDGE_LIST_H_

#include <iostream>
#include <fstream>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <iterator>
#ifdef NOTR1
	#include <boost/tr1/memory.hpp>
	#include <boost/tr1/tuple.hpp>
#else
	#include <tr1/memory>	// for shared_ptr
	#include <tr1/tuple>
#endif
#include "SpMat.h"
#include "SpTuples.h"
#include "SpDCCols.h"
#include "CommGrid.h"
#include "MPIType.h"
#include "LocArr.h"
#include "SpDefs.h"
#include "Deleter.h"
#include "SpHelper.h"
#include "SpParHelper.h"
#include "DenseParMat.h"
#include "FullyDistVec.h"
#include "Friends.h"
#include "Operations.h"

using namespace std;
using namespace std::tr1;

template <typename IT>
class DistEdgeList
{
public:	
	// Constructors
	DistEdgeList ();
	~DistEdgeList ();

	void GenGraph500Data(double initiator[4], int log_numverts, IT nedges);
	void CleanupEmpties();
	
	IT getNumRows() const { return numrows; }
	IT getNumCols() const { return numcols; }
	IT getNumLocalEdges() const { return nedges; }
	
private:
	shared_ptr<CommGrid> commGrid; 
	
	IT* edges; // edge list composed of pairs of edge endpoints.
	           // Edge i goes from edges[2*i+0] to edges[2*i+1]
	           
	IT nedges; // number of edges
	IT memedges; // number of edges for which there is space. nedges <= memedges
	
	IT numrows;
	IT numcols;
	
	void SetMemSize(IT ne);
	
	template<typename IU>
	friend void PermEdges(DistEdgeList<IU> & DEL);
	
	template <typename IU>
	friend void RenameVertices(DistEdgeList<IU> & DEL);

	template <class IU, class NU, class UDER>
	friend class SpParMat;
};

template<typename IU>
void PermEdges(DistEdgeList<IU> & DEL);


#include "DistEdgeList.cpp"

#endif
