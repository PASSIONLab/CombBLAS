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

#ifndef GRAPH_GENERATOR_SEQ
#define GRAPH_GENERATOR_SEQ
#endif

#include "graph500-1.2/generator/graph_generator.h"
#include "graph500-1.2/generator/utils.h"

#include <fstream>
#include <algorithm>
using namespace std;

template <typename IT>
DistEdgeList<IT>::DistEdgeList(): nedges(0), edges(NULL)
{
	commGrid.reset(new CommGrid(MPI::COMM_WORLD, 0, 0));
}

template <typename IT>
DistEdgeList<IT>::~DistEdgeList()
{
	delete [] edges;
}

//! Allocates enough space
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
	for (IT i = 0; i < (nedges-1); i++)
	{
		if (edges[2*i + 0] == -1)
		{
			// the graph500 generator marked this edge as a self-loop or a multiple edge.
			// swap it with the last edge
			edges[2*i + 0] = edges[2*(nedges-1) + 0];
			edges[2*i + 1] = edges[2*(nedges-1) + 1];
			edges[2*(nedges-1) + 0] = -1; // mark this spot as unused

			while (nedges > 0 && edges[2*(nedges-1) + 0] == -1)	// the swapped edge might be -1 too
				nedges--;
		}
	}
}


/**
 * Note that GenGraph500Data will return global vertex numbers (from 1... N). The ith edge can be
 * accessed with edges[2*i] and edges[2*i+1]. There will be duplicates and the data won't be sorted.
 * Generates an edge list consisting of an RMAT matrix suitable for the Graph500 benchmark.
*/
template <typename IT>
void DistEdgeList<IT>::GenGraph500Data(double initiator[4], int log_numverts, IT nedges_in)
{
	// Spread the two 64-bit numbers into five nonzero values in the correct range
	uint_fast32_t seed[5];
	uint64_t seed1 = MPI::COMM_WORLD.Get_rank();
	uint64_t seed2 = time(NULL);
	make_mrg_seed(seed1, seed2, seed);

	SetMemSize(nedges_in);	
	nedges = nedges_in;
	numrows = numcols = (IT)pow((double)2, log_numverts);
	
	// clear the source vertex by setting it to -1
	for (IT i = 0; i < nedges; i++)
		edges[2*i+0] = -1;
	
	generate_kronecker(0, 1, seed, log_numverts, nedges, initiator, edges);
}


/**
 * Randomly permutes the distributed edge list.
 * Once we call Viral's psort on this vector, everything will go to the right place [tuples are
 * sorted lexicographically] and you can reconstruct the int64_t * edges in an embarrassingly parallel way. 
 * As I understood, the entire purpose of this function is to destroy any locality. It does not
 * rename any vertices and edges are not named anyway. 
 * For an example, think about the edge (0,1). It will eventually (at the end of kernel 1) be owned by processor P(0,0). 
 * However, assume that processor P(r1,c1) has a copy of it before the call to PermEdges. After
 * this call, some other irrelevant processor P(r2,c2) will own it. So we gained nothing, it is just a scrambled egg. 
**/
template <typename IT>
void PermEdges(DistEdgeList<IT> & DEL)
{
	IT maxedges = DEL.memedges;	// this can be optimized by calling the clean-up first
	
	// to lower memory consumption, rename in stages
	// this is not "identical" to a full randomization; 
	// but more than enough to destroy any possible locality 
	IT stages = 16;	
	IT perstage = maxedges / stages;
	
	int nproc =(DEL.commGrid)->GetSize();
	int rank = (DEL.commGrid)->GetRank();
	IT* dist = new IT[nproc];

	MTRand M;	// generate random numbers with Mersenne Twister
	for(IT s=0; s< stages; ++s)
	{
		IT n_sofar = s*perstage;
		IT n_thisstage = ((s==(stages-1))? (maxedges - n_sofar): perstage);
		pair<double, pair<IT,IT> >* vecpair = new pair<double, pair<IT,IT> >[n_thisstage];
		dist[rank] = n_thisstage;
		(DEL.commGrid->GetWorld()).Allgather(MPI::IN_PLACE, 1, MPIType<IT>(), dist, 1, MPIType<IT>());
		IT lengthuntil = accumulate(dist, dist+rank, 0);

		for (IT i = 0; i < n_thisstage; i++)
		{
			vecpair[i].first = M.rand();
			vecpair[i].second.first = DEL.edges[2*(i+n_sofar)];
			vecpair[i].second.second = DEL.edges[2*(i+n_sofar)+1];
		}
		if(rank == 0)
			cout << "Calling parallel_sort" << endl;

		// less< pair<T1,T2> > works correctly (sorts w.r.t. first element of type T1)	
		// SpParHelper::MemoryEfficientPSort(pair<KEY,VAL> * array, IT length, IT * dist, MPI::Intracomm & comm)
		SpParHelper::MemoryEfficientPSort(vecpair, n_thisstage, dist, DEL.commGrid->GetWorld());
		for (IT i = 0; i < n_thisstage; i++)
		{
			DEL.edges[2*(i+n_sofar)] = vecpair[i].second.first;
			DEL.edges[2*(i+n_sofar)+1] = vecpair[i].second.second;
		}
		delete [] vecpair;
		if(rank == 0)
			cout << "PermEdges: Stage " << s << " completed" << endl;
	}
	delete [] dist;
}

/**
  * Rename vertices globally. 
  *	You first need to do create a random permutation distributed on all processors. 
  *	Then the p round robin algorithm will do the renaming: 
  * For all processors P(i,i)
  *          Broadcast local_p to all p processors
  *          For j= i*N/p to min((i+1)*N/p, N)
  *                    Rename the all j's with local_p(j) inside the edgelist (and mark them
  *                    "renamed" so that yeach vertex id is renamed only once)
  **/
template <typename IU>
void RenameVertices(DistEdgeList<IU> & DEL)
{
	int nprocs = DEL.commGrid->GetSize();
	int rank = DEL.commGrid->GetRank();
	MPI::Intracomm World = DEL.commGrid->GetWorld(); 

	// create permutation
	FullyDistVec<IU, IU> globalPerm(DEL.commGrid, -1);
	globalPerm.iota(DEL.getNumRows(), 0);
	globalPerm.RandPerm();	// now, randperm can return a 0-based permutation
	IU locrows = globalPerm.MyLocLength(); 
	
	// way to mark whether each vertex was already renamed or not
	IU locedgelist = 2*DEL.getNumLocalEdges();
	bool* renamed = new bool[locedgelist];
	fill_n(renamed, locedgelist, 0);
	
	// permutation for one round
	IU * localPerm;
	IU permsize;
	IU startInd = 0;

	//vector < pair<IU, IU> > vec;
	//for(IU i=0; i< DEL.getNumLocalEdges(); i++)
	//	vec.push_back(make_pair(DEL.edges[2*i], DEL.edges[2*i+1]));
	//sort(vec.begin(), vec.end());
	//vector < pair<IU, IU> > uniqued;
	//unique_copy(vec.begin(), vec.end(), back_inserter(uniqued));
	//cout << "before: " << vec.size() << " and after: " << uniqued.size() << endl;
	
	for (int round = 0; round < nprocs; round++)
	{
		// broadcast the permutation from the one processor
		if (rank == round)
		{
			permsize = locrows;
			localPerm = new IU[permsize];
			copy(globalPerm.arr.begin(), globalPerm.arr.end(), localPerm);
		}
		World.Bcast(&permsize, 1, MPIType<IU>(), round);
		if(rank != round)
		{
			localPerm = new IU[permsize];
		}
		World.Bcast(localPerm, permsize, MPIType<IU>(), round);
	
		// iterate over 	
		for (typename vector<IU>::size_type j = 0; j < locedgelist ; j++)
		{
			// We are renaming vertices, not edges
			if (startInd <= DEL.edges[j] && DEL.edges[j] < (startInd + permsize) && !renamed[j])
			{
				DEL.edges[j] = localPerm[DEL.edges[j]-startInd];
				renamed[j] = true;
			}
		}
		startInd += permsize;
		delete [] localPerm;
	}
	delete [] renamed;
}

