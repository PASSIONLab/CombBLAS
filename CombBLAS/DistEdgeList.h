/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.1 -------------------------------------------------*/
/* date: 12/25/2010 --------------------------------------------*/
/* authors: Adam Lugowski (alugowski@cs.ucsb.edu), Aydin Buluc--*/
/****************************************************************/

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

/** 
 * From Graph 500 reference implementation v2.1.1
**/
typedef struct packed_edge {
  uint32_t v0_low;
  uint32_t v1_low;
  uint32_t high; /* v1 in high half, v0 in low half */
} packed_edge;

static inline int64_t get_v0_from_edge(const packed_edge* p) {
  return (p->v0_low | ((int64_t)((int16_t)(p->high & 0xFFFF)) << 32));
}

static inline int64_t get_v1_from_edge(const packed_edge* p) {
  return (p->v1_low | ((int64_t)((int16_t)(p->high >> 16)) << 32));
}

static inline void write_edge(packed_edge* p, int64_t v0, int64_t v1) {
  p->v0_low = (uint32_t)v0;
  p->v1_low = (uint32_t)v1;
  p->high = ((v0 >> 32) & 0xFFFF) | (((v1 >> 32) & 0xFFFF) << 16);
}


template <typename IT>
class DistEdgeList
{
public:	
	// Constructors
	DistEdgeList ();
	DistEdgeList (char * filename, IT globaln, IT globalm);	// read from binary in parallel
	~DistEdgeList ();

	void Dump64bit(string filename);
	void Dump32bit(string filename);
	void GenGraph500Data(double initiator[4], int log_numverts, IT nedges, bool scramble =false, bool packed=false);
	void CleanupEmpties();
	
	IT getNumRows() const { return numrows; }
	IT getNumCols() const { return numcols; }
	IT getNumLocalEdges() const { return nedges; }
	
private:
	shared_ptr<CommGrid> commGrid; 
	
	IT* edges; // edge list composed of pairs of edge endpoints.
	           // Edge i goes from edges[2*i+0] to edges[2*i+1]
	packed_edge * pedges;
	           
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
