/*
 * This file contains current obsolete functions
 * Kept for use during future needs (if any)
 */

#ifndef _OLD_REDUCTIONS_H_
#define _OLD_REDUCTIONS_H_

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



// localmerged is invalidated in all processes after this redursive function
// globalmerged is valid on all processes upon exit
template <typename SR, typename IT, typename NT>
void ParallelReduce_Alltoall(MPI_Comm & fibWorld, tuple<IT,IT,NT> * & localmerged,
                             MPI_Datatype & MPI_triple, tuple<IT,IT,NT> * & globalmerged,
                             IT inputnnz, IT & outputnnz, IT ncols)
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
    //MPI_Barrier(MPI_COMM_WORLD);
    double loc_beg1 = MPI_Wtime();
    int target = 0;
    int cols_per_proc = (ncols + fprocs - 1) / fprocs;
    int split_point = cols_per_proc;
    int send_offsets[fprocs+1];
    send_offsets[0] = 0;
    
    for( int i = 0; i < inputnnz; i++ )
    {
        if( std::get<1>(localmerged[i]) >= split_point )
        {
            send_offsets[++target] = i;
            split_point += cols_per_proc;
        }
    }
    while(target < fprocs) send_offsets[++target] = inputnnz;
    for(int i=0; i<fprocs; i++)
    {
        send_sizes[i] = send_offsets[i+1] - send_offsets[i];
    }
    /*
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
     */
    //MPI_Barrier(MPI_COMM_WORLD);
    //comp_reduce += (MPI_Wtime() - loc_beg1);
    
    double reduce_beg = MPI_Wtime();
    MPI_Alltoall( send_sizes, 1, MPI_INT, recv_sizes, 1, MPI_INT,fibWorld);
    //MPI_Barrier(MPI_COMM_WORLD);
    comm_reduce += (MPI_Wtime() - reduce_beg);
    
    int recv_count = 0;
    for( int i = 0; i < fprocs; i++ )
        recv_count += recv_sizes[i];
    tuple<IT,IT,NT> *recvbuf = new tuple<IT,IT,NT>[recv_count];
    
    int recv_offsets[fprocs];
    recv_offsets[0] = 0;
    for( int i = 1; i < fprocs; i++ ) {
        recv_offsets[i] = recv_offsets[i-1]+recv_sizes[i-1];
    }
    //MPI_Barrier(MPI_COMM_WORLD);
    reduce_beg = MPI_Wtime();
    MPI_Alltoallv( localmerged, send_sizes, send_offsets, MPI_triple, recvbuf, recv_sizes, recv_offsets, MPI_triple, fibWorld);
    //MPI_Barrier(MPI_COMM_WORLD);
    comm_reduce += (MPI_Wtime() - reduce_beg);
    loc_beg1 = MPI_Wtime();
    
    int pos[fprocs];
    for( int i = 0; i < fprocs; i++ )
        pos[i] = recv_offsets[i];
    outputnnz = 0;
    globalmerged = new tuple<IT,IT,NT>[recv_count];
    
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
        
        if( outputnnz > 0 && std::get<0>(globalmerged[outputnnz-1]) == std::get<0>(recvbuf[pos[nexti]]) && std::get<1>(globalmerged[outputnnz-1]) == std::get<1>(recvbuf[pos[nexti]]) )
            // add this one to the previous
            std::get<2>(globalmerged[outputnnz-1]) = SR::add( std::get<2>(globalmerged[outputnnz-1]), std::get<2>(recvbuf[pos[nexti]]) );
        else {
            // make this the next entry in the output
            globalmerged[outputnnz] = recvbuf[pos[nexti]];
            outputnnz++;
        }
        
        pos[nexti]++;  // it was a bug since it was placed before the if statement
    }
    //MPI_Barrier(MPI_COMM_WORLD);
    //comp_reduce += (MPI_Wtime() - loc_beg1);
    
    delete [] recvbuf;
    delete [] localmerged;
    localmerged  = NULL;
}

#endif


