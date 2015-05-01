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
#include "../SpParHelper.h"
#include "Glue.h"
#include "mtSpGEMM.h"

#ifdef BUPC
extern "C" int SUMMALayer(void * A1, void * A2, void * B1, void * B2, void ** C, CCGrid * cmg, bool isBT, bool threaded);
extern "C" void * ReduceAll(void ** C, CCGrid * cmg, int localcount);
extern "C" void DeleteMatrix(void ** A);
extern "C" int64_t GetNNZ(void * A);
#endif

/*
template<class SR, class NUO, class IU, class NU1, class NU2>
SpTuples<IU, NUO> * Tuples_AnXBn2
(const SpDCCols<IU, NU1> & A,
 const SpDCCols<IU, NU2> & B,
 bool clearA = false, bool clearB = false)
{
    IU mdim = A.getnrow();
    IU ndim = A.getncol();
    if(A.isZero() || B.isZero())
    {
        return new SpTuples<IU, NUO>(0, mdim, ndim);
    }
    StackEntry< NUO, pair<IU,IU> > * multstack;
    IU cnz = SpHelper::SpColByCol< SR > (*(A.GetDCSC()), *(B.GetDCSC()), A.getncol(),  multstack);
    
    if(clearA)
        delete const_cast<SpDCCols<IU, NU1> *>(&A);
    if(clearB)
        delete const_cast<SpDCCols<IU, NU2> *>(&B);
    
    return new SpTuples<IU, NUO> (cnz, mdim, ndim, multstack);
}
*/


int64_t GetNNZ(void * A)
{
	SpDCCols<int32_t, double> * castA = (SpDCCols<int32_t, double>*) (A);
	return castA->getnnz();
}

void DeleteMatrix(void ** A)
{
	SpDCCols<int32_t, double> * castA = (SpDCCols<int32_t, double>*) (*A);
	delete castA;
}

// localmerged is invalidated in all processes after this redursive function
// globalmerged is valid on all processes upon exit
template <typename SR>
void ParallelReduce_Alltoall(MPI_Comm & fibWorld, tuple<int32_t,int32_t,double> * & localmerged,
			     MPI_Datatype & MPI_triple, tuple<int32_t,int32_t,double> * & globalmerged,
			     int inputnnz, int & outputnnz, int ncols)
{
    int fprocs;
    MPI_Comm_size(fibWorld,&fprocs);
    if(fprocs == 1)
	{
		globalmerged = localmerged;
		localmerged = NULL;
		outputnnz = inputnnz;
		return;
	}
	int send_sizes[fprocs];
	int recv_sizes[fprocs];
	// this could be made more efficient, either by a binary search or by guessing then correcting
    MPI_Barrier(MPI_COMM_WORLD);
	double loc_beg1 = MPI_Wtime();
	int target = 0;
	int cols_per_proc = (ncols + fprocs - 1) / fprocs;
	int split_point = cols_per_proc;
	int send_offsets[fprocs];
	send_offsets[0] = 0;
	for( int i = 0; i < inputnnz; i++ ) {
	  if( std::get<1>(localmerged[i]) >= split_point ) {
	    if( target == 0 )
	      send_sizes[target] = i;
	    else {
	      send_sizes[target] = i-send_offsets[target];
	    }
	    send_offsets[target+1] = i;
	    target++;
	    split_point += cols_per_proc;
	  }
	}
	send_sizes[fprocs-1] = inputnnz - send_offsets[fprocs-1];
    MPI_Barrier(MPI_COMM_WORLD);
	comp_reduce += (MPI_Wtime() - loc_beg1);

	double reduce_beg = MPI_Wtime();
	MPI_Alltoall( send_sizes, 1, MPI_INT, recv_sizes, 1, MPI_INT,fibWorld);
    MPI_Barrier(MPI_COMM_WORLD);
	comm_reduce += (MPI_Wtime() - reduce_beg);

	int recv_count = 0;
	for( int i = 0; i < fprocs; i++ )
	  recv_count += recv_sizes[i];
	tuple<int32_t,int32_t,double> *recvbuf = (tuple<int32_t,int32_t,double>*) malloc( recv_count * sizeof(tuple<int32_t,int32_t,double>) );
	int recv_offsets[fprocs];
	recv_offsets[0] = 0;
	for( int i = 1; i < fprocs; i++ ) {
	  recv_offsets[i] = recv_offsets[i-1]+recv_sizes[i-1];
	}
    MPI_Barrier(MPI_COMM_WORLD);
	reduce_beg = MPI_Wtime();
	MPI_Alltoallv( localmerged, send_sizes, send_offsets, MPI_triple, recvbuf, recv_sizes, recv_offsets, MPI_triple, fibWorld);
    MPI_Barrier(MPI_COMM_WORLD);
    comm_reduce += (MPI_Wtime() - reduce_beg);
	loc_beg1 = MPI_Wtime();

	int pos[fprocs];
	for( int i = 0; i < fprocs; i++ )
	  pos[i] = recv_offsets[i];
	outputnnz = 0;
	globalmerged = new tuple<int32_t,int32_t,double>[recv_count];

	while( true ) {
	  // find the next entry
	  int nexti = -1;
	  int r = INT_MAX;
	  int c = INT_MAX;
	  for( int i = 0; i < fprocs; i++ ) {
	    if( pos[i] < recv_offsets[i]+recv_sizes[i] ) {
	      if( std::get<1>(recvbuf[pos[i]]) < c ) {
		c = std::get<1>(recvbuf[pos[i]]);
		r = std::get<0>(recvbuf[pos[i]]);
		nexti = i;
	      } else if( (std::get<1>(recvbuf[pos[i]]) == c) && (std::get<0>(recvbuf[pos[i]]) < r) ) {
		r = std::get<0>(recvbuf[pos[i]]);
		nexti = i;
	      }
	    }
	  }
	  if( nexti == -1 ) // merge is finished
	    break;

	  pos[nexti]++;
	  if( outputnnz > 0 && std::get<0>(globalmerged[outputnnz-1]) == std::get<0>(recvbuf[pos[nexti]]) && std::get<1>(globalmerged[outputnnz-1]) == std::get<1>(recvbuf[pos[nexti]]) )
	    // add this one to the previous
	    std::get<2>(globalmerged[outputnnz-1]) = SR::add( std::get<2>(globalmerged[outputnnz-1]), std::get<2>(recvbuf[pos[nexti]]) );
	  else {
	    // make this the next entry in the output
	    globalmerged[outputnnz] = recvbuf[pos[nexti]];
	    outputnnz++;
	  }
	}
    MPI_Barrier(MPI_COMM_WORLD);
	comp_reduce += (MPI_Wtime() - loc_beg1);
	
	free(recvbuf);
	delete [] localmerged;
	localmerged  = NULL;
}


// localmerged is invalidated in all processes after this redursive function
// globalmerged is valid only in fibWorld root (0) upon exit
template <typename SR>
void ParallelReduce(MPI_Comm & fibWorld, tuple<int32_t,int32_t,double> * & localmerged,
						MPI_Datatype & MPI_triple, tuple<int32_t,int32_t,double> * & globalmerged,
						int inputnnz, int & outputnnz)
{
    int fprocs, frank;
    MPI_Comm_size(fibWorld,&fprocs);
    MPI_Comm_rank(fibWorld,&frank);
	if(fprocs == 1)
	{
		globalmerged = localmerged;
		localmerged = NULL;
		outputnnz = inputnnz;
		return;
	}
	else if(fprocs % 2 != 0)
	{	
		SpParHelper::Print("Not even sized neighbors, can't merge\n");
		return;
	}
	
	int color = frank % 2;
    int key   = frank / 2;
    MPI_Comm halfWorld;
    MPI_Comm_split(fibWorld, color, key, &halfWorld); // odd-even split
	
	if(color == 0)  // even numbered - received
	{
		MPI_Status status;
		int hissize = 0;
		
		MPI_Recv(&hissize, 1, MPI_INT, frank+1, 1, fibWorld, &status);
		
		tuple<int32_t,int32_t,double> * recvdata = new tuple<int32_t,int32_t,double>[hissize];
		
		double reduce_beg = MPI_Wtime();
		MPI_Recv(recvdata, hissize, MPI_triple, frank+1, 1, fibWorld, &status);
		comm_reduce += (MPI_Wtime() - reduce_beg);

		
		int i=0, j=0, k = 0;
		tuple<int32_t,int32_t,double> *  mergeddata = new tuple<int32_t,int32_t,double>[inputnnz + hissize];


		while(i < inputnnz && j < hissize)
		{
			// both data are in ascending order w.r.t. first columns then rows
			if(get<1>(localmerged[i]) > get<1>(recvdata[j]))
			{
				mergeddata[k] = recvdata[j++];  
			}
			else if(get<1>(localmerged[i]) < get<1>(recvdata[j]))
			{
				mergeddata[k] = localmerged[i++];
			}
			else // columns are equal 
			{
				if(get<0>(localmerged[i]) > get<0>(recvdata[j]))
				{
					mergeddata[k] = recvdata[j++];
				}
				else if(get<0>(localmerged[i]) < get<0>(recvdata[j]))
				{
					mergeddata[k] = localmerged[i++];
				}
				else  // everything equal
				{
					mergeddata[k] = make_tuple(get<0>(localmerged[i]), get<1>(recvdata[j]), SR::add(get<2>(recvdata[j]), get<2>(localmerged[i])));
					++i; ++j;
				}
			}
			++k;  // in any case, one more entry added to result

		}
			
		delete [] recvdata;
		delete [] localmerged;
		localmerged  = NULL;
		return ParallelReduce<SR>(halfWorld, mergeddata, MPI_triple, globalmerged, k, outputnnz); // k is the new input nnz
		
	}
	else // odd numbered - sender (does not recurse further)
	{
		MPI_Send(&inputnnz, 1, MPI_INT, frank-1, 1, fibWorld);
		MPI_Send(localmerged, inputnnz, MPI_triple, frank-1, 1, fibWorld);
		delete [] localmerged;
		localmerged  = NULL;
	}

}


void * ReduceAll(void ** C, CCGrid * cmg, int localcount)
{
	typedef SpTuples<int32_t, double> SPTUPLE;
	typedef SpDCCols<int32_t, double> LOC_SPMAT;

	typedef PlusTimesSRing<double, double> PTDD;
    MPI_Comm layWorld, fibWorld, rowWorld, colWorld;
    MPI_Comm_split(MPI_COMM_WORLD, cmg->layer_grid, cmg->rankinlayer, &layWorld);
	MPI_Comm_split(MPI_COMM_WORLD, cmg->rankinlayer, cmg->layer_grid, &fibWorld);
	MPI_Comm_split(MPI_COMM_WORLD, cmg->layer_grid * cmg->GRROWS + cmg->rankinlayer / cmg->GRROWS, cmg->RANKINROW, &rowWorld);
	MPI_Comm_split(MPI_COMM_WORLD, cmg->layer_grid * cmg->GRCOLS + cmg->rankinlayer % cmg->GRROWS, cmg->RANKINCOL, &colWorld);

	vector<double> all_merge_time;
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    
	if(myrank == 0)
	{
		all_merge_time.resize(nprocs);
#ifdef DEBUG
		cout << MPI::COMM_WORLD.Get_rank() << "'s tuples: " << endl;
		for(int i=0; i< localcount; ++i)
			cout << "Tuple " << i << " has " << static_cast<SPTUPLE*>(C[i])->getnnz() << " nonzeros" << endl;
#endif
	}
	
#ifdef TIMING // BEGIN TIMING
	double loc_merge_beg = MPI_Wtime();
#endif
	vector<SPTUPLE *> alltuples;
	int C_m = 0;
	int C_n = 0;
	for(int i=0; i< localcount; ++i)
	{
		if(static_cast<SPTUPLE*>(C[i])->getnnz() > 0)
		{
			alltuples.push_back(static_cast<SPTUPLE*>(C[i]));
			C_m = static_cast<SPTUPLE*>(C[i])->getnrow();
			C_n = static_cast<SPTUPLE*>(C[i])->getncol();
		}
	}
	
	int64_t totrecv;
	tuple<int32_t,int32_t,double> * recvdata;
    MPI_Barrier(MPI_COMM_WORLD);
	double loc_beg1 = MPI_Wtime();
	SPTUPLE localmerged = MergeAll<PTDD>(alltuples, C_m, C_n,true); // delete alltuples[] entries
    MPI_Barrier(MPI_COMM_WORLD);
	comp_reduce += (MPI_Wtime() - loc_beg1);

	
#ifdef TIMING // END TIMING
	double loc_merge_time = MPI_Wtime() - loc_merge_beg;
	MPI_Gather(&loc_merge_time, 1, MPI_DOUBLE, SpHelper::p2a(all_merge_time), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if(myrank== 0)
	{
		vector<int> permutation = SpHelper::order(all_merge_time);
		int smallest = permutation[0];
		int largest = permutation[nprocs-1];
		int median = permutation[nprocs/2];
		cout << "Localmerged has " << localmerged.getnnz() << " nonzeros" << endl;
		cout << "Local Merge MEAN: " << accumulate( all_merge_time.begin(), all_merge_time.end(), 0.0 )/ static_cast<double> (nprocs) << endl;
		cout << "Local Merge MAX: " << all_merge_time[largest] << endl;
		cout << "Local Merge MIN: " << all_merge_time[smallest]  << endl;
		cout << "Local Merge MEDIAN: " << all_merge_time[median] << endl;
	}
#endif
	
	MPI_Datatype MPI_triple;
    MPI_Type_contiguous(sizeof(tuple<int32_t,int32_t,double>), MPI_CHAR, &MPI_triple);
    MPI_Type_commit(&MPI_triple);
	int pre_glmerge = localmerged.getnnz();
	LOC_SPMAT * locret;
	
#ifdef PARALLELREDUCE
	int outputnnz = 0;
	//ParallelReduce<PTDD>(fibWorld, localmerged.tuples, MPI_triple, recvdata, (int) localmerged.getnnz(), outputnnz);
	ParallelReduce_Alltoall<PTDD>(fibWorld, localmerged.tuples, MPI_triple, recvdata, (int) localmerged.getnnz(), outputnnz, C_n);
	locret = new LOC_SPMAT(SPTUPLE(outputnnz, C_m, C_n, recvdata), false);
#else
    int fibsize, fibrank;
    MPI_Comm_size(fibWorld,&fibsize);
    MPI_Comm_rank(fibWorld,&fibrank);
    
	if(fibrank == 0)	// root of the fibers (i.e. 0th layer)
	{
		
#ifdef TIMING	// BEGIN TIMING
		double loc_merge_beg = MPI_Wtime();
		vector<double> gl_merge_time;
        
        int layprocs, layrank;
        MPI_Comm_size(layWorld,&layprocs);
        MPI_Comm_rank(layWorld,&layrank);
        if(layrank == 0)
			gl_merge_time.resize(layprocs);
#endif

		int * pst_glmerge = new int[fibsize];	// redundant at non-root
		MPI_Gather(&pre_glmerge, 1, MPI_INT, pst_glmerge, 1, MPI_INT, 0, fibWorld);
		int64_t totrecv = std::accumulate(pst_glmerge, pst_glmerge+fibsize, static_cast<int64_t>(0));
		
		int * dpls = new int[fibsize]();       // displacements (zero initialized pid)
		std::partial_sum(pst_glmerge, pst_glmerge+fibsize-1, dpls+1);
		recvdata = new tuple<int32_t,int32_t,double>[totrecv];

		// IntraComm::GatherV(sendbuf, int sentcnt, sendtype, recvbuf, int * recvcnts, int * displs, recvtype, root)
		    MPI_Barrier(MPI_COMM_WORLD);
		double reduce_beg = MPI_Wtime();
        MPI_Gatherv(localmerged.tuples, pre_glmerge, MPI_triple, recvdata, pst_glmerge, dpls, MPI_triple, 0, fibWorld);
		comm_reduce += (MPI_Wtime() - reduce_beg);
			
		// SpTuples<IU,NU> MergeAllContiguous (tuple<IU,IU,NU> * colsortedranges, IU mstar, IU nstar, int hsize, int * nonzeros, int * dpls, bool delarrays)
		// MergeAllContiguous frees the arrays and LOC_SPMAT constructor does not transpose [in this call]
		
		double loc_beg2 = MPI_Wtime();
		locret = new LOC_SPMAT(MergeAllContiguous<PTDD>( recvdata, C_m, C_n, fibsize, pst_glmerge, dpls, true), false);
		comp_reduce += (MPI_Wtime() - loc_beg2);

		
#ifdef TIMING		// END TIMING
		double loc_merge_time = MPI_Wtime() - loc_merge_beg;
		MPI_Gather(&loc_merge_time, 1, MPI_DOUBLE, SpHelper::p2a(gl_merge_time), 1, MPI_DOUBLE, 0, layWorld);
		
		int64_t mergednnz = locret->getnnz();
		int64_t globalnnz = 0;
		MPI_Reduce(&mergednnz, &globalnnz, 1, MPIType<int64_t>(), MPI_SUM, 0, layWorld);
		
		if(layWorld.Get_rank() == 0)
		{
			vector<int> permutation = SpHelper::order(gl_merge_time);
			int smallest = permutation[0];
			int largest = permutation[layprocs-1];
			int median = permutation[layprocs/2];
			cout << "Global Merge MEAN: " << accumulate( gl_merge_time.begin(), gl_merge_time.end(), 0.0 )/ static_cast<double> (layprocs) << endl;
			cout << "Global Merge MAX: " << gl_merge_time[largest] << endl;
			cout << "Global Merge MIN: " << gl_merge_time[smallest]  << endl;
			cout << "Global Merge MEDIAN: " << gl_merge_time[median] << endl;
			
			cout << layWorld.Get_rank() << "'s final tuple has " << locret->getnnz() << " local nonzeros" << endl;
			cout << "While the total number of nonzeros are " << globalnnz << endl;
		}
#endif
	}
	else 
	{
		MPI_Gather(&pre_glmerge, 1, MPI_INT, NULL, 1, MPI_INT, 0, fibWorld); // recvbuf is irrelevant on non-root
		MPI_Gatherv(localmerged.tuples, pre_glmerge, MPI_triple, NULL, NULL, NULL, MPI_triple, 0, fibWorld);
		locret = new LOC_SPMAT(); // other layes don't have the data
	}
#endif
    MPI_Type_free(&MPI_triple);
    MPI_Comm_free(&fibWorld);
    MPI_Comm_free(&rowWorld);
    MPI_Comm_free(&colWorld);
    MPI_Comm_free(&layWorld);
	return locret;
}
 
// B1 and B2 are already locally transposed
// Returns an array of unmerged lists in C (size 2 * cmg->GRCOLS)
int SUMMALayer (void * A1, void * A2, void * B1, void * B2, void ** C, CCGrid * cmg, bool isBT, bool threaded)
{
	typedef SpDCCols<int32_t, double> LOC_SPMAT;
	typedef SpTuples<int32_t, double> SPTUPLE;
	typedef PlusTimesSRing<double, double> PTDD;

	
	LOC_SPMAT * A1_cast = (LOC_SPMAT *) A1;
	LOC_SPMAT * A2_cast = (LOC_SPMAT *) A2;
	LOC_SPMAT * B1_cast = (LOC_SPMAT *) B1;
	LOC_SPMAT * B2_cast = (LOC_SPMAT *) B2;
    
    MPI_Comm layWorld, fibWorld, rowWorld, colWorld;
    MPI_Comm_split(MPI_COMM_WORLD, cmg->layer_grid, cmg->rankinlayer, &layWorld);
    MPI_Comm_split(MPI_COMM_WORLD, cmg->rankinlayer, cmg->layer_grid, &fibWorld);
    MPI_Comm_split(MPI_COMM_WORLD, cmg->layer_grid * cmg->GRROWS + cmg->rankinlayer / cmg->GRROWS, cmg->RANKINROW, &rowWorld);
    MPI_Comm_split(MPI_COMM_WORLD, cmg->layer_grid * cmg->GRCOLS + cmg->rankinlayer % cmg->GRROWS, cmg->RANKINCOL, &colWorld);
	

	if(cmg->GRROWS != cmg->GRCOLS)
		SpParHelper::Print("Only works on square grids for now\n");
	
	int stages = cmg->GRCOLS;	// total number of "essential" summa stages  - we will do twice as much
    int phases;
    MPI_Comm_size(fibWorld, &phases); // total number of stages will be finished in phases
	int eachphase = stages / phases;
	if(eachphase * phases != stages)
		SpParHelper::Print("Number of layers (c) should devide the grid dimension evenly\n");
	   
	int stage_beg = cmg->layer_grid * eachphase;
	int stage_end = (cmg->layer_grid+1) * eachphase; 
	
	// Data is overallocated (for a full \sqrt{p} x \sqrt{p} grid) but doesn't matter
	int32_t ** ARecvSizes = SpHelper::allocate2D<int32_t>(LOC_SPMAT::esscount, stages);
	int32_t ** BRecvSizes = SpHelper::allocate2D<int32_t>(LOC_SPMAT::esscount, stages);
	
	// Remotely fetched matrices are stored as pointers
	LOC_SPMAT * ARecv; 
	LOC_SPMAT * BRecv;
	SPTUPLE  *** tomerge = (SPTUPLE  ***) C;
	*tomerge = new SPTUPLE * [2*eachphase];
	
	int Aself = cmg->RANKINROW;
	int Bself = cmg->RANKINCOL;	

#ifdef DEBUG
    int layrank;
    MPI_Comm_rank(layWorld, &layrank);
	if(layrank == 0)
		cout << "Layer " << cmg->layer_grid << " is handling iterations from " << stage_beg << " to " << stage_end << endl;
#endif
	
	for(int k=0; k<2; ++k)
	{
		LOC_SPMAT *thisA, *thisB;
		if(k == 0)
		{
			thisA = A1_cast;
			thisB = B1_cast;
		}
		if(k == 1)
		{
			thisA = A2_cast;
			thisB = B2_cast;
		}
		
		// Set the dimensions
		SpParHelper::GetSetSizes( *thisA, ARecvSizes, rowWorld);
		SpParHelper::GetSetSizes( *thisB, BRecvSizes, colWorld);
			
		for(int i = stage_beg; i < stage_end; ++i) 
		{
			double bcast_beg = MPI_Wtime();
			vector<int32_t> ess;	
			if(i == Aself)
			{
				ARecv = thisA;	// shallow-copy 
			}
			else
			{
				ess.resize(LOC_SPMAT::esscount);
				for(int j=0; j< LOC_SPMAT::esscount; ++j)	
					ess[j] = ARecvSizes[j][i];		// essentials of the ith matrix in this row	
		
				ARecv = new LOC_SPMAT();				// first, create the object
			}
			SpParHelper::BCastMatrix(rowWorld, *ARecv, ess, i);	// then, receive its elements	
			ess.clear();	
			if(i == Bself)
			{
				BRecv = thisB;	// shallow-copy
			}
			else
			{
				ess.resize(LOC_SPMAT::esscount);		
				for(int j=0; j< LOC_SPMAT::esscount; ++j)	
				{
					ess[j] = BRecvSizes[j][i];	
				}	
				BRecv = new LOC_SPMAT();
			}
			SpParHelper::BCastMatrix(colWorld, *BRecv, ess, i);	// then, receive its elements
			comm_bcast += (MPI_Wtime() - bcast_beg);
			
            
			double summa_beg = MPI_Wtime();
            SPTUPLE * C_cont;
            if(threaded)
            {
                C_cont = LocalSpGEMM<PTDD, double>
                (*ARecv, *BRecv, // parameters themselves
                 i != Aself, 	// 'delete A' condition
                 i != Bself);	// 'delete B' condition
            }
            else
            {
                C_cont = MultiplyReturnTuples<PTDD, double>
                (*ARecv, *BRecv, // parameters themselves
                 false, isBT,	// transpose information (B is transposed)
                 i != Aself, 	// 'delete A' condition
                 i != Bself);	// 'delete B' condition
            }
		
            comp_summa += (MPI_Wtime() - summa_beg);
		
			(*tomerge)[k*eachphase + i-stage_beg] = C_cont;
		}
	}
	
	SpHelper::deallocate2D(ARecvSizes, LOC_SPMAT::esscount);
	SpHelper::deallocate2D(BRecvSizes, LOC_SPMAT::esscount);
    MPI_Comm_free(&fibWorld);
    MPI_Comm_free(&rowWorld);
    MPI_Comm_free(&colWorld);
    MPI_Comm_free(&layWorld);
	return eachphase;
}

