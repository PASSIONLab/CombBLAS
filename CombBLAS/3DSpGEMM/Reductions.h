#ifndef _REDUCTIONS_H_
#define _REDUCTIONS_H_


#include <mpi.h>
#include <sys/time.h>
#include <iostream>
#include <iomanip>
#include <functional>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>


#include "CombBLAS/CombBLAS.h"
#include "Glue.h"
#include "CCGrid.h"

namespace combblas {


/***************************************************************************
 * Distribute a local m/sqrt(p) x n/sqrt(p) matrix (represented by a list of tuples) across layers
 * so that a each processor along the third dimension receives m/sqrt(p) x n/(c*sqrt(p)) submatrices.
 * After receiving c submatrices, they are merged to create one m/sqrt(p) x n/(c*sqrt(p)) matrix.
 * Assumption: input tuples are deleted
 * Inputs:
 *      fibWorld: Communicator along the third dimension
 *      localmerged: input array of tuples, which will be distributed across layers
 *  Output: output array of tuples, after distributing across layers
            and merging locally in the received processor
 *
 ***************************************************************************/

template <typename SR, typename IT, typename NT>
SpTuples<IT,NT> * ParallelReduce_Alltoall_threaded(MPI_Comm & fibWorld, SpTuples<IT,NT> * & localmerged)
{
    double comp_begin, comm_begin, comp_time=0, comm_time=0;
    int fprocs, fibrank;
    MPI_Comm_size(fibWorld,&fprocs);
    MPI_Comm_rank(fibWorld,&fibrank);
    IT mdim = localmerged->getnrow();
    IT ndim = localmerged->getncol();
    if(fprocs == 1)
    {
        return localmerged;
    }
    
    
    // ------------ find splitters to distributed across layers -----------
    comp_begin = MPI_Wtime();
    std::vector<int> send_sizes(fprocs);
    std::vector<int> recv_sizes(fprocs);
    std::vector<int> recv_offsets(fprocs);
    std::vector<int> send_offsets = findColSplitters<int>(localmerged, fprocs);
    for(int i=0; i<fprocs; i++)
    {
        send_sizes[i] = send_offsets[i+1] - send_offsets[i];
    }
    comp_time += (MPI_Wtime() - comp_begin);
    
    
    // ------------ Communicate counts -----------
    comm_begin = MPI_Wtime();
    MPI_Alltoall( send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1, MPI_INT,fibWorld);
    comm_time += (MPI_Wtime() - comm_begin);
    MPI_Datatype MPI_triple;
    MPI_Type_contiguous(sizeof(std::tuple<IT,IT,NT>), MPI_CHAR, &MPI_triple);
    MPI_Type_commit(&MPI_triple);
    
    
    // ------------ Allocate memory to receive data -----------
    comp_begin = MPI_Wtime();
    int recv_count = 0;
    for( int i = 0; i < fprocs; i++ )
    {
        recv_count += recv_sizes[i];
    }
    std::tuple<IT,IT,NT> * recvbuf = static_cast<std::tuple<IT, IT, NT>*> (::operator new (sizeof(std::tuple<IT, IT, NT>[recv_count])));
    
    recv_offsets[0] = 0;
    for( int i = 1; i < fprocs; i++ )
    {
        recv_offsets[i] = recv_offsets[i-1]+recv_sizes[i-1];
    }
    comp_time += (MPI_Wtime() - comp_begin);
    
    
    // ------------ Communicate split tuples -----------
    comm_begin = MPI_Wtime();
    MPI_Alltoallv( localmerged->tuples, send_sizes.data(), send_offsets.data(), MPI_triple, recvbuf, recv_sizes.data(), recv_offsets.data(), MPI_triple, fibWorld); // WARNING: is this big enough?
    comm_time += (MPI_Wtime() - comm_begin);
    
    
    
    // -------- update column indices of split tuples ----------
    comp_begin = MPI_Wtime();
    IT ndimSplit = ndim/fprocs;
    if(fibrank==(fprocs-1))
        ndimSplit = ndim - ndimSplit * fibrank;
    IT coloffset = fibrank * ndimSplit;
#pragma omp parallel for
    for(int k=0; k<recv_count; k++)
    {
        std::get<1>(recvbuf[k]) = std::get<1>(recvbuf[k]) - coloffset;
    }
    
    
    // -------- create vector of SpTuples for MultiwayMerge ----------
    std::vector< SpTuples<IT,NT>* > lists;
    for(int i=0; i< fprocs; ++i)
    {
        SpTuples<IT, NT>* spTuples = new SpTuples<IT, NT> (recv_sizes[i], mdim, ndimSplit, &recvbuf[recv_offsets[i]], true); // If needed pass an empty object of proper dimension
        lists.push_back(spTuples);
    }
    
    // -------- merge received tuples ----------
    SpTuples<IT,NT> * globalmerged = MultiwayMerge<SR>(lists, mdim, ndimSplit, false);
    
    comp_time += (MPI_Wtime() - comp_begin);
    comp_reduce_layer += comp_time;
    comm_reduce += comm_time;
    
    
    ::operator delete(recvbuf);
    delete localmerged; // not sure if we can call ::operator delete here
    
    return  globalmerged;
}


template <typename NT, typename IT>
SpDCCols<IT,NT> * ReduceAll_threaded(std::vector< SpTuples<IT,NT>* > & unreducedC, CCGrid & CMG)
{
	typedef PlusTimesSRing<double, double> PTDD;
    IT mdim = unreducedC[0]->getnrow();
    IT ndim = unreducedC[0]->getncol();
    
    // ------ merge list of tuples from n/sqrt(p) stages of SUMMA -------
    double loc_beg1 = MPI_Wtime();
    //SpTuples<IT, NT>* localmerged = multiwayMerge(unreducedC, true);
    SpTuples<IT, NT>* localmerged = MultiwayMerge<PTDD>(unreducedC, mdim, ndim, true);
    comp_reduce += (MPI_Wtime() - loc_beg1);

    // scatter local tuples across layers
    SpTuples<IT,NT> * mergedSpTuples = ParallelReduce_Alltoall_threaded<PTDD>(CMG.fiberWorld, localmerged);
    
    loc_beg1 = MPI_Wtime();
    // TODO: this is not a good constructor. Change it back to SpTuple-based constructor
    SpDCCols<IT,NT> * reducedC = new SpDCCols<IT,NT>(mergedSpTuples->getnrow(), mergedSpTuples->getncol(), mergedSpTuples->getnnz(), mergedSpTuples->tuples, false);
    comp_result += (MPI_Wtime() - loc_beg1);
    delete mergedSpTuples;  // too expensive
	return reducedC;
}

}

#endif


