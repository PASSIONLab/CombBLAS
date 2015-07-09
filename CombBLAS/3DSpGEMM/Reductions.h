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


#include "../CombBLAS.h"
#include "Glue.h"
#include "CCGrid.h"
#include "mtSpGEMM.h"
#include "OldReductions.h"

/***************************************************************************
 * Find indices of column splitters in a list of tuple in parallel.
 * Inputs:
 *      tuples: an array of SpTuples each tuple is (rowid, colid, val)
 *      nsplits: number of splits requested
 *  Output:
 *      splitters: An array of size (nsplits+1) storing the starts and ends of split tuples.
 ***************************************************************************/
template <typename IT, typename NT>
int * findColSplitters(SpTuples<IT,NT> * & spTuples, int nsplits)
{
    int* splitters = new int[nsplits+1];
    splitters[0] = 0;
    ColLexiCompare<IT,NT> comp;
#pragma omp parallel for
    for(int i=1; i< nsplits; i++)
    {
        IT cur_col = i * (spTuples->getncol()/nsplits);
        tuple<IT,IT,NT> search_tuple(0, cur_col, 0);
        tuple<IT,IT,NT>* it = lower_bound (spTuples->tuples, spTuples->tuples + spTuples->getnnz(), search_tuple, comp);
        splitters[i] = (int) (it - spTuples->tuples);
    }
    splitters[nsplits] = spTuples->getnnz();
    
    return splitters;
}


/***************************************************************************
 * Distribute a local m/sqrt(p) x n/sqrt(p) matrix (represented by a list of tuples) across layers
 * so that a each processor along the third dimension receives m/sqrt(p) x n/(c*sqrt(p)) submatrices.
 * After receiving c submatrices, they are merged to create one m/sqrt(p) x n/(c*sqrt(p)) matrix.
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
    
    if(fprocs == 1)
    {
        return localmerged;
    }
    
    comp_begin = MPI_Wtime();
    int* send_sizes = new int[fprocs];
    int* recv_sizes = new int[fprocs];
    int * send_offsets = findColSplitters(localmerged, fprocs);
    for(int i=0; i<fprocs; i++)
    {
        send_sizes[i] = send_offsets[i+1] - send_offsets[i];
    }
    comp_time += (MPI_Wtime() - comp_begin);
    
    comm_begin = MPI_Wtime();
    MPI_Alltoall( send_sizes, 1, MPI_INT, recv_sizes, 1, MPI_INT,fibWorld);
    comm_time += (MPI_Wtime() - comm_begin);
    
    
    comp_begin = MPI_Wtime();
    int recv_count = 0;
    for( int i = 0; i < fprocs; i++ )
    {
        recv_count += recv_sizes[i];
    }
    tuple<IT,IT,NT> * recvbuf = new tuple<IT,IT,NT>[recv_count];
    int* recv_offsets = new int[fprocs];
    recv_offsets[0] = 0;
    for( int i = 1; i < fprocs; i++ )
    {
        recv_offsets[i] = recv_offsets[i-1]+recv_sizes[i-1];
    }
    comp_time += (MPI_Wtime() - comp_begin);
    
    MPI_Datatype MPI_triple;
    MPI_Type_contiguous(sizeof(tuple<IT,IT,NT>), MPI_CHAR, &MPI_triple);
    MPI_Type_commit(&MPI_triple);
    
    
    comm_begin = MPI_Wtime();
    MPI_Alltoallv( localmerged->tuples, send_sizes, send_offsets, MPI_triple, recvbuf, recv_sizes, recv_offsets, MPI_triple, fibWorld); // WARNING: is this big enough?
    comm_time += (MPI_Wtime() - comm_begin);
    
    
    
    comp_begin = MPI_Wtime();
    
    // -------- update column indices of split tuples ----------
    
    IT local_ncol = localmerged->getncol()/fprocs;
    if(fibrank==(fprocs-1))
        local_ncol = localmerged->getncol() - local_ncol * fibrank;
    IT coloffset = fibrank * local_ncol;
#pragma omp parallel for
    for(int k=0; k<recv_count; k++)
    {
        std::get<1>(recvbuf[k]) = std::get<1>(recvbuf[k]) - coloffset;
    }
    
    
    // -------- create vector of SpTuples for multiwayMerge ----------
    vector< SpTuples<IT,NT>* > lists;
    for(int i=0; i< fprocs; ++i)
    {
        //if(recv_sizes[i] > 0)
        {
            SpTuples<IT, NT>* spTuples = new SpTuples<IT, NT> (recv_sizes[i], localmerged->getnrow(), local_ncol, &recvbuf[recv_offsets[i]], true); // if none received, pass an empty object of proper dimension
            lists.push_back(spTuples);
        }
    }
    
    SpTuples<IT,NT> * globalmerged = multiwayMerge(lists, false);
    
    comp_time += (MPI_Wtime() - comp_begin);
 
    comp_reduce_layer += comp_time;
    comm_reduce += comm_time;
    delete [] send_sizes;
    delete [] recv_sizes;
    delete [] recv_offsets;
    delete [] recvbuf;
    delete localmerged;
    delete [] send_offsets;
    return  globalmerged;
}


template <typename NT, typename IT>
SpDCCols<IT,NT> * ReduceAll_threaded(vector< SpTuples<IT,NT>* > & unreducedC, CCGrid & CMG)
{
	typedef PlusTimesSRing<double, double> PTDD;
	
    // merge list of tuples from n/sqrt(p) stages of SUMMA
    double loc_beg1 = MPI_Wtime();
    SpTuples<IT, NT>* localmerged = multiwayMerge(unreducedC, true);
	comp_reduce += (MPI_Wtime() - loc_beg1);

    // scatter local tuples across layers
    SpTuples<IT,NT> * mergedSpTuples = ParallelReduce_Alltoall_threaded<PTDD>(CMG.fiberWorld, localmerged);
    
    loc_beg1 = MPI_Wtime();
    SpDCCols<IT,NT> * reducedC = new SpDCCols<IT,NT>(mergedSpTuples->getnrow(), mergedSpTuples->getncol(), mergedSpTuples->getnnz(), mergedSpTuples->tuples, false);
    comp_result += (MPI_Wtime() - loc_beg1);
	return reducedC;
}



template <typename NT, typename IT>
SpDCCols<IT,NT> * ReduceAll(vector< SpTuples<IT,NT>* > & unreducedC, CCGrid & CMG)
{
    typedef PlusTimesSRing<double, double> PTDD;
    
    IT C_m = unreducedC[0]->getnrow();
    IT C_n = unreducedC[0]->getncol();
    
    int64_t totrecv;
    tuple<IT,IT,NT> * recvdata;
    
    double loc_beg1 = MPI_Wtime();
    // MergeAll defined in ../CombBLAS/Friends.h
    SpTuples<IT,NT> localmerged =  MergeAll<PTDD>(unreducedC, C_m, C_n,true); // delete unreducedC entries
    comp_reduce += (MPI_Wtime() - loc_beg1);
    
    
    MPI_Datatype MPI_triple;
    MPI_Type_contiguous(sizeof(tuple<IT,IT,NT>), MPI_CHAR, &MPI_triple);
    MPI_Type_commit(&MPI_triple);
    SpDCCols<IT,NT> * locret;
    
    int pre_glmerge = localmerged.getnnz(); // WARNING: is this big enought to hold?
    
    IT outputnnz = 0;
    ParallelReduce_Alltoall<PTDD>(CMG.fiberWorld, localmerged.tuples, MPI_triple, recvdata, localmerged.getnnz(), outputnnz, C_n);
    loc_beg1 = MPI_Wtime();
    locret = new SpDCCols<IT,NT>(SpTuples<IT,NT>(outputnnz, C_m, C_n, recvdata), false);
    //MPI_Barrier(MPI_COMM_WORLD); //needed
    //comp_reduce += (MPI_Wtime() - loc_beg1); //needed
    
    MPI_Type_free(&MPI_triple);
    return locret;
}

#endif


