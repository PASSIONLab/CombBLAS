/****************************************************************/
/* Sequential and Parallel Sparse Matrix Multiplication Library */
/* version 2.3 --------------------------------------------------/
/* date: 10/28/2010 ---------------------------------------------/
/* author: Adam Lugowski (alugowski@cs.ucsb.edu) ----------------/
/****************************************************************/

#include <mpi.h>

#include "SpParMat.h"
#include "ParFriends.h"
#include "Operations.h"

#include "graph500-1.2/generator/graph_generator.h"
#include "graph500-1.2/generator/utils.h"

#include <fstream>
#include <algorithm>
using namespace std;

template <typename IT>
DistEdgeList<IT>::DistEdgeList(): nedges(0), edges(NULL)
{
	comm = MPI::COMM_WORLD.Dup();
}

template <typename IT>
DistEdgeList<IT>::~DistEdgeList()
{
	comm.Free();
	delete [] edges;
}

/** Allocates enough space
*/
template <typename IT>
void DistEdgeList<IT>::SetMemSize(IT ne)
{
	if (edges)
	{
		delete [] edges;
		edges = NULL;
	}
	
	memedges = ne;
	edges = 0;
	
	if (memedges > 0)
		edges = new IT[2*memedges];
}

/** Removes all edges that begin with a -1. 
 * Walks back from the end to tighten the nedges counter, then walks forward and replaces any edge
 * with a -1 source with the last edge.
 */
template <typename IT>
void DistEdgeList<IT>::CleanupEmpties()
{

	// find out how many edges there actually are
	while (nedges > 0 && edges[2*(nedges-1) + 0] == -1)
	{
		nedges--;
	}
	
	
	// remove marked multiplicities or self-loops
	for (int64_t i = 0; i < (nedges-1); i++)
	{
		if (edges[2*i + 0] == -1)
		{
			// the graph500 generator marked this edge as a self-loop or a multiple edge.
			// swap it with the last edge
			edges[2*i + 0] = edges[2*(nedges-1) + 0];
			edges[2*i + 1] = edges[2*(nedges-1) + 1];
			edges[2*(nedges-1) + 0] = -1; // mark this spot as unused
			nedges--;
		}
	}
}


/**
 * Generates a SpParMat which represents a matrix usable for the Graph500 benchmark.
 
 
I think the function is already semantically correct, except that nedges parameter should be
totaledges/np where np is the number of processors, just like the way John did in this M-file.
However, this should go inside the DistEdgeList class instead of SpParMat. The returned data
(int64_t * edges) need not be reformatted and DistEdgeList class should ideally just use that
array as its private data.
[implication: the line "SpTuples<int64_t,NT> A(nedges, n, n, edges);" and anything following that
should be omitted from GenGraph500Data]

Note that GenGraph500Data will return global vertex numbers (from 1... N). The ith edge can be
accessed with edges[2*i] and edges[2*i+1]. There will be duplicates and the data won't be sorted.
Please try to verify that my calls to the reference implementation inside GenGraph500Data are
meaningful.

The header files are "graph500-1.2/generator/graph_generator.h" and  "graph500-1.2/generator/utils.h"

*/


/** Generates an edge list consisting of an RMAT matrix suitable for the Graph500 benchmark.
 * Requires IT = int64_t
*/
template <typename IT>
void DistEdgeList<IT>::GenGraph500Data(double initiator[4], int log_numverts, int64_t nedges_in)
{
	// Spread the two 64-bit numbers into five nonzero values in the correct range
	uint_fast32_t seed[5];
	make_mrg_seed(1, 2, seed);

	SetMemSize(nedges_in);	
	nedges = nedges_in;
	numrows = numcols = (IT)pow((double)2, log_numverts);
	
	// clear the source vertex by setting it to -1
	for (int64_t i = 0; i < nedges; i++)
	{
		edges[2*i+0] = -1;
	}
	
	generate_kronecker(0, 1, seed, log_numverts, nedges, initiator, edges);
}


/**
 * Randomly permutes the distributed edge list.
 
 which is inside "ParFriends.h" and is a friend of SpParVec. 

The differences are:
- your PermEdges will be among all processors instead of being among diagonals only. 
- it will be a friend of your DistEdgeList class. 
- you are likely to use a vector of "tuples<double,IT,IT>" where double is the random number;
and the other two IT's are endpoints of the edgelist. 
- once you call Viral's psort on this vector, everything will go to the right place [tuples are
sorted lexicographically] and you can reconstruct the int64_t * edges in an embarrassingly parallel way. 

As I understood, the entire purpose of this function is to destroy any locality. It does not
rename any vertices and edges are not named anyway. 
For an example, think about the edge (0,1). It will eventually (at the end of kernel 1) be
owned by processor P(0,0). 
However, assume that processor P(r1,c1) has a copy of it before the call to PermEdges. After
this call, some other irrelevant processor P(r2,c2) will own it. So we gained nothing, it is
just a scrambled egg. (John, Steve, am I right?)
*/
template <typename IT>
void PermEdges(DistEdgeList<IT> & DEL)
{
	IT nedges = DEL.memedges;
	
	pair<double, pair<IT,IT> >* vecpair = new pair<double, pair<IT,IT> >[nedges];

	int nproc = DEL.comm.Get_size();
	int rank = DEL.comm.Get_rank();

	IT* dist = new IT[nproc];
	dist[rank] = nedges;
	DEL.comm.Allgather(MPI::IN_PLACE, 0, MPI::DATATYPE_NULL, dist, 1, MPIType<IT>());
	IT lengthuntil = accumulate(dist, dist+rank, 0);

	MTRand M;	// generate random numbers with Mersenne Twister
	for (IT i = 0; i < nedges; i++)
	{
		vecpair[i].first = M.rand();
		vecpair[i].second.first = DEL.edges[2*i + 0];
		vecpair[i].second.second = DEL.edges[2*i + 1];
	}

	// free some space
	DEL.SetMemSize(0);
	
	// less< pair<T1,T2> > works correctly (sorts wrt first elements)	
	psort::parallel_sort (vecpair, vecpair + nedges,  dist, DEL.comm);
	
	// recreate the edge list
	DEL.SetMemSize(nedges);

	for (IT i = 0; i < nedges; i++)
	{
		DEL.edges[2*i + 0] = vecpair[i].second.first;
		DEL.edges[2*i + 1] = vecpair[i].second.second;
	}
	
	delete [] dist;
	delete [] vecpair;
}

/*
(AL3) Rename vertices globally. You first need to do:

SpParVec<int,int> p;
RandPerm(p, A.getlocalrows());

This will create a global permutation vector distributed on diagonal processors. Then the sqrt(p)
round robin algorithm will do the renaming: 

For all diagonal processors P(i,i)
            Broadcast local_p to all p processors
            For j= i*sqrt(p) to min((i+1)*sqrt(p), N)
                      Rename the all j's with local_p(j) inside the edgelist (and mark them
                      "renamed" so that yeach vertex id is renamed only once)


*/
template <typename IU>
void RenameVertices(DistEdgeList<IU> & DEL)
{
	int rank = DEL.comm.Get_rank();
	int size = DEL.comm.Get_size();
	int sqrt_size = (int)sqrt(size);
	if (sqrt_size*sqrt_size != size) {
		if (rank == 0)
			printf("RenameVertices(): BAD ASSUMPTION: number of processors is not square!\n");
	}

	// create permutation
	SpParVec<IU, IU> globalPerm;
	globalPerm.iota(DEL.getNumRows(), 0);
	RandPerm(globalPerm, DEL.getNumRows()/sqrt_size);
	
	// way to mark whether each vertex was already renamed or not
	bool* renamed = new bool[2*DEL.getNumLocalEdges()];
	memset(renamed, 0, sizeof(bool)*2*DEL.getNumLocalEdges());
	
	// permutation for one round
	IU* localPerm = new IU[2*DEL.memedges];
	long permsize;
	long startInd = 0;
	
	for (int round = 0; round < sqrt_size; round++)
	{
		// broadcast the permutation from the one diagonal processor
		int broadcaster = round*sqrt_size + round;
		
		if (rank == broadcaster)
		{
			permsize = globalPerm.getlocnnz();
			copy(globalPerm.num.begin(), globalPerm.num.end(), localPerm);
		}
		
		//if (rank == 0)
		//	printf("on round %d, startInd=%ld\n", round, startInd);
			
		DEL.comm.Bcast(&permsize, 1, MPIType<long>(), broadcaster);
		DEL.comm.Bcast(localPerm, permsize, MPIType<IU>(), broadcaster);

		//if (rank == 0)
		//	printf("on round %d: got permutation of size %ld\n", round, permsize);
		
		for (int64_t j = 0; j < 2*DEL.getNumLocalEdges(); j++)
		{
			// We are renaming vertices, not edges
			if (startInd <= DEL.edges[j] && DEL.edges[j] < (startInd + permsize) && !renamed[j])
			{
				//printf("proc %d: permuting edges[%ld] with localPerm[%ld]\n", rank, j, DEL.edges[j]-startInd);
				DEL.edges[j] = localPerm[DEL.edges[j]-startInd];
				renamed[j] = true;
			}
			fflush(stdout);
			DEL.comm.Barrier();
		}
		
		startInd += permsize;
	}
	
	delete [] localPerm;
	delete [] renamed;
}

/*
Below are what I'll provide from this point [everything timed here]:

A contructor of the form SpParMat(DistEdgeList) that will require an all-to-all communication
to send the data to their eventual owners. 
Before this point, the distribution was based on approximate nonzero count and processors was
potentially holding somebody else's data.  

I have the semi-complete functions to do the local data conversion (after the all-to-all exchange): 
SpTuples<IT,NT>::SpTuples (IT maxnnz, IT nRow, IT nCol, IT * edges)
SpDCCols (const SpTuples<IT,NT> & rhs, bool transpose, MemoryPool * mpool = NULL);

I just need to make them work together and write the parallel logic around them. It will also
remove any duplicates, self-loops, etc.
*/
