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

template <typename IT, typename NT>
int * findColSplitters(tuple<IT,IT,NT> * & tuples, IT ntuples, IT ncols, int nsplits)
{
    int* splitters = new int[nsplits+1];  // WARNING: who is deleting this?
    splitters[0] = 0;
    ColLexiCompare<IT,NT> comp;
#pragma omp parallel for  //schedule(dynamic)
    for(int i=1; i< nsplits; i++)
    {
        IT cur_col = i * (ncols/nsplits);
        tuple<IT,IT,NT> search_tuple(0, cur_col, 0);
        tuple<IT,IT,NT>* it = lower_bound (tuples, tuples+ntuples, search_tuple, comp);
        splitters[i] = (int) (it - tuples);
    }
    splitters[nsplits] = ntuples;
    
    return splitters;
}


// localmerged is invalidated in all processes after this function (is this true?)
// globalmerged is valid on all processes upon exit
template <typename SR, typename IT, typename NT>
void ParallelReduce_Alltoall_threaded(MPI_Comm & fibWorld, tuple<IT,IT,NT> * & localmerged,
                             MPI_Datatype & MPI_triple, tuple<IT,IT,NT> * & globalmerged,
                             IT inputnnz, IT & outputnnz, IT ncols)
{
    int fprocs;
    double comp_begin, comm_begin, comp_time=0, comm_time=0;
    
    MPI_Comm_size(fibWorld,&fprocs);
    if(fprocs == 1)
    {
        globalmerged = localmerged;
        localmerged = NULL;
        outputnnz = inputnnz;
        return;
    }
    comp_begin = MPI_Wtime();
    int send_sizes[fprocs];
    int recv_sizes[fprocs];
    int * send_offsets = findColSplitters(localmerged, inputnnz, ncols, fprocs);
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
    int recv_offsets[fprocs];   // WARNING: this is probably not standard C++
    recv_offsets[0] = 0;
    for( int i = 1; i < fprocs; i++ )
    {
        recv_offsets[i] = recv_offsets[i-1]+recv_sizes[i-1];
    }
    comp_time += (MPI_Wtime() - comp_begin);
    
    comm_begin = MPI_Wtime();
    MPI_Alltoallv( localmerged, send_sizes, send_offsets, MPI_triple, recvbuf, recv_sizes, recv_offsets, MPI_triple, fibWorld); // WARNING: is this big enough?
    comm_time += (MPI_Wtime() - comm_begin);
    
    
    comp_begin = MPI_Wtime();
    vector< tuple<IT,IT,NT>* > lists;
    vector<IT> listSizes;
    for(int i=0; i< fprocs; ++i)
    {
        if(recv_sizes[i] > 0)
        {
            lists.push_back(&recvbuf[recv_offsets[i]]);
            listSizes.push_back(recv_sizes[i]);
        }
    }
    globalmerged = multiwayMerge(lists, listSizes, outputnnz, false);
    comp_time += (MPI_Wtime() - comp_begin);
    
    comp_reduce_layer += comp_time;
    comm_reduce += comm_time;
    
    delete [] recvbuf;
    delete [] localmerged;
    localmerged  = NULL;
}


template <typename NT, typename IT>
SpDCCols<IT,NT> * ReduceAll_threaded(vector< SpTuples<IT,NT>* > & unreducedC, CCGrid & CMG)
{
	typedef PlusTimesSRing<double, double> PTDD;
    
	IT C_m = unreducedC[0]->getnrow();
	IT C_n = unreducedC[0]->getncol();
    
    double loc_beg1 = MPI_Wtime();
    
    vector<tuple<IT, IT, NT>*> lists;
    vector<IT> listSizes;
    int localcount = unreducedC.size();
    for(int i=0; i< localcount; ++i)
    {
        if(unreducedC[i]->getnnz() > 0)
        {
            lists.push_back(unreducedC[i]->tuples);
            listSizes.push_back(unreducedC[i]->getnnz());
        }
    }
    
    IT localmerged_size;
    tuple<IT, IT, NT>* localmerged = multiwayMerge(lists, listSizes, localmerged_size,true);
	comp_reduce += (MPI_Wtime() - loc_beg1);

    MPI_Datatype MPI_triple;
    MPI_Type_contiguous(sizeof(tuple<IT,IT,NT>), MPI_CHAR, &MPI_triple);
    MPI_Type_commit(&MPI_triple);
    SpDCCols<IT,NT> * locret;
	int pre_glmerge = localmerged_size;
    tuple<IT,IT,NT> * recvdata;
	
#ifdef PARALLELREDUCE
	IT outputnnz = 0;
    ParallelReduce_Alltoall_threaded<PTDD>(CMG.fiberWorld, localmerged, MPI_triple, recvdata, localmerged_size, outputnnz, C_n);
    
    loc_beg1 = MPI_Wtime();
    locret = new SpDCCols<IT,NT>(C_m, C_n, outputnnz, recvdata, false);
    comp_result += (MPI_Wtime() - loc_beg1);

#else // not multithreaded yet
    int fibsize = CMG.GridLayers;
	if(CMG.layer_grid == 0) // layer_grid = rankinfiber
	{
		int * pst_glmerge = new int[fibsize];	// redundant at non-root
		MPI_Gather(&pre_glmerge, 1, MPI_INT, pst_glmerge, 1, MPI_INT, 0, CMG.fiberWorld);
		int64_t totrecv = std::accumulate(pst_glmerge, pst_glmerge+fibsize, static_cast<int64_t>(0));
		
		int * dpls = new int[fibsize]();
		std::partial_sum(pst_glmerge, pst_glmerge+fibsize-1, dpls+1);
		recvdata = new tuple<IT,IT,NT>[totrecv];
        
		double reduce_beg = MPI_Wtime();
        MPI_Gatherv(localmerged, pre_glmerge, MPI_triple, recvdata, pst_glmerge, dpls, MPI_triple, 0, CMG.fiberWorld);
		comm_reduce += (MPI_Wtime() - reduce_beg);
		double loc_beg2 = MPI_Wtime();
		locret = new SpDCCols<IT,NT>(MergeAllContiguous<PTDD>( recvdata, C_m, C_n, fibsize, pst_glmerge, dpls, true), false);
	}
	else 
	{
		MPI_Gather(&pre_glmerge, 1, MPI_INT, NULL, 1, MPI_INT, 0, CMG.fiberWorld);
		MPI_Gatherv(localmerged, pre_glmerge, MPI_triple, NULL, NULL, NULL, MPI_triple, 0, CMG.fiberWorld);
		locret = new SpDCCols<IT,NT>(); // other layes don't have the data
	}
#endif
    
    MPI_Type_free(&MPI_triple);
	return locret;
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
    
#ifdef PARALLELREDUCE
    IT outputnnz = 0;
    ParallelReduce_Alltoall<PTDD>(CMG.fiberWorld, localmerged.tuples, MPI_triple, recvdata, localmerged.getnnz(), outputnnz, C_n);
    loc_beg1 = MPI_Wtime();
    locret = new SpDCCols<IT,NT>(SpTuples<IT,NT>(outputnnz, C_m, C_n, recvdata), false);
    //MPI_Barrier(MPI_COMM_WORLD); //needed
    //comp_reduce += (MPI_Wtime() - loc_beg1); //needed
#else
    
    int fibsize = CMG.GridLayers;
    if(CMG.layer_grid == 0)	// root of the fibers (i.e. 0th layer)
    {
        int * pst_glmerge = new int[fibsize];	// redundant at non-root
        MPI_Gather(&pre_glmerge, 1, MPI_INT, pst_glmerge, 1, MPI_INT, 0, CMG.fiberWorld);
        int64_t totrecv = std::accumulate(pst_glmerge, pst_glmerge+fibsize, static_cast<int64_t>(0));
        
        int * dpls = new int[fibsize]();       // displacements (zero initialized pid)
        std::partial_sum(pst_glmerge, pst_glmerge+fibsize-1, dpls+1);
        recvdata = new tuple<IT,IT,NT>[totrecv];
        
        //MPI_Barrier(MPI_COMM_WORLD);
        double reduce_beg = MPI_Wtime();
        MPI_Gatherv(localmerged.tuples, pre_glmerge, MPI_triple, recvdata, pst_glmerge, dpls, MPI_triple, 0, CMG.fiberWorld);
        comm_reduce += (MPI_Wtime() - reduce_beg);
        
        // SpTuples<IU,NU> MergeAllContiguous (tuple<IU,IU,NU> * colsortedranges, IU mstar, IU nstar, int hsize, int * nonzeros, int * dpls, bool delarrays)
        // MergeAllContiguous frees the arrays and LOC_SPMAT constructor does not transpose [in this call]
        
        double loc_beg2 = MPI_Wtime();
        locret = new SpDCCols<IT,NT>(MergeAllContiguous<PTDD>( recvdata, C_m, C_n, fibsize, pst_glmerge, dpls, true), false);
        //comp_reduce += (MPI_Wtime() - loc_beg2);
        
    }
    else 
    {
        MPI_Gather(&pre_glmerge, 1, MPI_INT, NULL, 1, MPI_INT, 0, CMG.fiberWorld); // recvbuf is irrelevant on non-root
        MPI_Gatherv(localmerged.tuples, pre_glmerge, MPI_triple, NULL, NULL, NULL, MPI_triple, 0, CMG.fiberWorld);
        locret = new SpDCCols<IT,NT>(); // other layes don't have the data
    }
#endif
    
    MPI_Type_free(&MPI_triple);
    return locret;
}

#endif


