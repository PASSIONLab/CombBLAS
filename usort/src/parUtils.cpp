
/**
  @file parUtils.C
  @author Rahul S. Sampath, rahul.sampath@gmail.com
  @author Hari Sundar, hsundar@gmail.com
  */

#include "mpi.h"
#include "binUtils.h"
#include "dtypes.h"
#include "parUtils.h"

#ifdef __DEBUG__
#ifndef __DEBUG_PAR__
#define __DEBUG_PAR__
#endif
#endif

namespace par {

  unsigned int splitCommBinary( MPI_Comm orig_comm, MPI_Comm *new_comm) {
    int npes, rank;

    MPI_Group  orig_group, new_group;

    MPI_Comm_size(orig_comm, &npes);
    MPI_Comm_rank(orig_comm, &rank);

    unsigned int splitterRank = binOp::getPrevHighestPowerOfTwo(npes);

    int *ranksAsc, *ranksDesc;
    //Determine sizes for the 2 groups 
    ranksAsc = new int[splitterRank];
    ranksDesc = new int[( npes - splitterRank)];

    int numAsc = 0;
    int numDesc = ( npes - splitterRank - 1);

    //This is the main mapping between old ranks and new ranks.
    for(int i=0; i<npes; i++) {
      if( static_cast<unsigned int>(i) < splitterRank) {
        ranksAsc[numAsc] = i;
        numAsc++;
      }else {
        ranksDesc[numDesc] = i;
        numDesc--;
      }
    }//end for i

    MPI_Comm_group(orig_comm, &orig_group);

    /* Divide tasks into two distinct groups based upon rank */
    if (static_cast<unsigned int>(rank) < splitterRank) {
      MPI_Group_incl(orig_group, splitterRank, ranksAsc, &new_group);
    }else {
      MPI_Group_incl(orig_group, (npes-splitterRank), ranksDesc, &new_group);
    }

    MPI_Comm_create(orig_comm, new_group, new_comm);

    delete [] ranksAsc;
    ranksAsc = NULL;
    
    delete [] ranksDesc;
    ranksDesc = NULL;

    return splitterRank;
  }//end function

  unsigned int splitCommBinaryNoFlip( MPI_Comm orig_comm, MPI_Comm *new_comm) {
    int npes, rank;

    MPI_Group  orig_group, new_group;

    MPI_Comm_size(orig_comm, &npes);
    MPI_Comm_rank(orig_comm, &rank);

    unsigned int splitterRank =  binOp::getPrevHighestPowerOfTwo(npes);

    int *ranksAsc, *ranksDesc;
    //Determine sizes for the 2 groups 
    ranksAsc = new int[splitterRank];
    ranksDesc = new int[( npes - splitterRank)];

    int numAsc = 0;
    int numDesc = 0; //( npes - splitterRank - 1);

    //This is the main mapping between old ranks and new ranks.
    for(int i = 0; i < npes; i++) {
      if(static_cast<unsigned int>(i) < splitterRank) {
        ranksAsc[numAsc] = i;
        numAsc++;
      }else {
        ranksDesc[numDesc] = i;
        numDesc++;
      }
    }//end for i

    MPI_Comm_group(orig_comm, &orig_group);

    /* Divide tasks into two distinct groups based upon rank */
    if (static_cast<unsigned int>(rank) < splitterRank) {
      MPI_Group_incl(orig_group, splitterRank, ranksAsc, &new_group);
    }else {
      MPI_Group_incl(orig_group, (npes-splitterRank), ranksDesc, &new_group);
    }

    MPI_Comm_create(orig_comm, new_group, new_comm);

    delete [] ranksAsc;
    ranksAsc = NULL;
    
    delete [] ranksDesc;
    ranksDesc = NULL;

    return splitterRank;
  }//end function

  //create Comm groups and remove empty processors...
  int splitComm2way(bool iAmEmpty, MPI_Comm * new_comm, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
    MPI_Barrier(comm);
#endif

      MPI_Group  orig_group, new_group;
    int size;
    MPI_Comm_size(comm, &size);

    bool* isEmptyList = new bool[size];
    par::Mpi_Allgather<bool>(&iAmEmpty, isEmptyList, 1, comm);

    int numActive=0, numIdle=0;
    for(int i = 0; i < size; i++) {
      if(isEmptyList[i]) {
        numIdle++;
      }else {
        numActive++;
      }
    }//end for i

    int* ranksActive = new int[numActive];
    int* ranksIdle = new int[numIdle];

    numActive=0;
    numIdle=0;
    for(int i = 0; i < size; i++) {
      if(isEmptyList[i]) {
        ranksIdle[numIdle] = i;
        numIdle++;
      }else {
        ranksActive[numActive] = i;
        numActive++;
      }
    }//end for i

    delete [] isEmptyList;	
    isEmptyList = NULL;

    /* Extract the original group handle */
    MPI_Comm_group(comm, &orig_group);

    /* Divide tasks into two distinct groups based upon rank */
    if (!iAmEmpty) {
      MPI_Group_incl(orig_group, numActive, ranksActive, &new_group);
    }else {
      MPI_Group_incl(orig_group, numIdle, ranksIdle, &new_group);
    }

    /* Create new communicator */
    MPI_Comm_create(comm, new_group, new_comm);

    delete [] ranksActive;
    ranksActive = NULL;
    
    delete [] ranksIdle;
    ranksIdle = NULL;

  }//end function

  int splitCommUsingSplittingRank(int splittingRank, MPI_Comm* new_comm,
      MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
    MPI_Barrier(comm);
#endif

      MPI_Group  orig_group, new_group;
    int size;
    int rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int* ranksActive = new int[splittingRank];
    int* ranksIdle = new int[size - splittingRank];

    for(int i = 0; i < splittingRank; i++) {
      ranksActive[i] = i;
    }

    for(int i = splittingRank; i < size; i++) {
      ranksIdle[i - splittingRank] = i;
    }

    /* Extract the original group handle */
    MPI_Comm_group(comm, &orig_group);

    /* Divide tasks into two distinct groups based upon rank */
    if (rank < splittingRank) {
      MPI_Group_incl(orig_group, splittingRank, ranksActive, &new_group);
    }else {
      MPI_Group_incl(orig_group, (size - splittingRank), ranksIdle, &new_group);
    }

    /* Create new communicator */
    MPI_Comm_create(comm, new_group, new_comm);

    delete [] ranksActive;
    ranksActive = NULL;
    
    delete [] ranksIdle;
    ranksIdle = NULL;

  }//end function

  //create Comm groups and remove empty processors...
  int splitComm2way(const bool* isEmptyList, MPI_Comm * new_comm, MPI_Comm comm) {
      
    MPI_Group  orig_group, new_group;
    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    int numActive=0, numIdle=0;
    for(int i = 0; i < size; i++) {
      if(isEmptyList[i]) {
        numIdle++;
      }else {
        numActive++;
      }
    }//end for i

    int* ranksActive = new int[numActive];
    int* ranksIdle = new int[numIdle];

    numActive=0;
    numIdle=0;
    for(int i = 0; i < size; i++) {
      if(isEmptyList[i]) {
        ranksIdle[numIdle] = i;
        numIdle++;
      }else {
        ranksActive[numActive] = i;
        numActive++;
      }
    }//end for i

    /* Extract the original group handle */
    MPI_Comm_group(comm, &orig_group);

    /* Divide tasks into two distinct groups based upon rank */
    if (!isEmptyList[rank]) {
      MPI_Group_incl(orig_group, numActive, ranksActive, &new_group);
    }else {
      MPI_Group_incl(orig_group, numIdle, ranksIdle, &new_group);
    }

    /* Create new communicator */
    MPI_Comm_create(comm, new_group, new_comm);

    delete [] ranksActive;
    ranksActive = NULL;
    
    delete [] ranksIdle;
    ranksIdle = NULL;

    return 0;
  }//end function

		
	int AdjustCommunicationPattern(std::vector<int>& send_sizes, std::vector<int>& send_partners, 
				 												 std::vector<int>& recv_sizes, std::vector<int>& recv_partners, MPI_Comm comm) 
	{
    int npes;
    int rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);
		
		unsigned int k = send_sizes.size();
		
		// do scans ...
		DendroIntL lsz[k];
		DendroIntL gsz[k],  gscan[k];
		
		for(size_t i = 0; i < send_sizes.size(); ++i) {
			lsz[i] = send_sizes[i];
		}
		par::Mpi_Scan<DendroIntL>( lsz, gscan, k, MPI_SUM, comm);
		
		if (rank == npes-1) {
			for(size_t i = 0; i < k; ++i) {
				gsz[i] = gscan[i];
			}
		}		
		// broadcast from last proc to get total counts, per segment ...
		par::Mpi_Bcast<DendroIntL>( gsz, k, npes-1, comm);
		
		DendroIntL segment_p0[k];
		for(size_t i = 0; i < k; ++i) {
			segment_p0[i] = (i*npes)/k;
		}
		
		/*
		 * -- Dividing into k segments, so each segment will have npes/k procs.
		 * -- Each proc will have gsz[i]/(npes/k) elements.
		 * -- rank of proc which will get i-th send_buff is,
		 *        -- segment_p0[i] + gscan[i]    
		 */
				
		// figure out send_partners for k sends
		// send_partners.clear();
		for(size_t i = 0; i < k; ++i) {
			int new_part;
			int seg_npes  =   ( (i == k-1) ? npes - segment_p0[i] : segment_p0[i+1]-segment_p0[i] );
			int overhang  =   gsz[i] % seg_npes;
			DendroIntL rank_mid = gscan[i] - lsz[i]/2;
			if ( rank_mid < overhang*(gsz[i]/seg_npes + 1)) {
				new_part = segment_p0[i] + rank_mid/(gsz[i]/seg_npes + 1);
			} else {
				new_part = segment_p0[i] + (rank_mid - overhang)/(gsz[i]/seg_npes);	
			}
			send_partners[i] = new_part; 
		}
		
		int idx=0;
		if (send_partners[0] == rank) {
			send_sizes[0] = 0;
		}
		for(size_t i = 1; i < k; ++i)
		{
			if (send_partners[i] == rank) {
				send_sizes[i] = 0;
				idx = i;
				continue;
			}
			if (send_partners[i] == send_partners[i-1]) {
				send_sizes[idx] += lsz[i];
				send_sizes[i]=0;
			} else {
					idx = i;
			}
		}
		
		// let procs know you will be sending to them ...
	
		// try MPI one sided comm
		MPI_Win win;
		int *rcv;
	  MPI_Alloc_mem(sizeof(int)*npes, MPI_INFO_NULL, &rcv);
		for(size_t i = 0; i < npes; ++i) rcv[i] = 0;
		
		MPI_Win_create(rcv, npes, sizeof(int), MPI_INFO_NULL,  MPI_COMM_WORLD, &win);
		
		
		MPI_Win_fence(MPI_MODE_NOPRECEDE, win);
		for (size_t i = 0; i < send_sizes.size(); i++) 
		{
			if (send_sizes[i]) {
		    MPI_Put(&(send_sizes[i]), 1, MPI_INT, send_partners[i], rank, 1, MPI_INT, win);
			}
		}	 
		MPI_Win_fence((MPI_MODE_NOSTORE | MPI_MODE_NOSUCCEED), win);
		// figure out recv partners and sizes ...
		recv_sizes.clear(); recv_partners.clear();
		for(size_t i = 0; i < npes; ++i)
		{
			if (rcv[i]) {
				recv_partners.push_back(i);
				recv_sizes.push_back(rcv[i]);
			} 
		}
		
		MPI_Win_free(&win);
	  MPI_Free_mem(rcv);
		 
		return 1;
	}

}// end namespace

