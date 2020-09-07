/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.6 -------------------------------------------------*/
/* date: 6/15/2017 ---------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc  --------------------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2017, The Regents of the University of California
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */


#include "usort/parUtils.h"

namespace combblas {

template <typename IT>
void SpParHelper::ReDistributeToVector(int* & map_scnt, std::vector< std::vector< IT > > & locs_send, std::vector< std::vector< std::string > > & data_send, 
					std::vector<std::array<char, MAXVERTNAME>> & distmapper_array, const MPI_Comm & comm)
{
	int nprocs, myrank;
	MPI_Comm_size(comm, &nprocs);
	MPI_Comm_rank(comm, &myrank);

    	int * map_rcnt = new int[nprocs];
    	MPI_Alltoall(map_scnt, 1, MPI_INT, map_rcnt, 1, MPI_INT, comm);  
    	int * map_sdspl = new int[nprocs]();
    	int * map_rdspl = new int[nprocs]();
    	std::partial_sum(map_scnt, map_scnt+nprocs-1, map_sdspl+1);
    	std::partial_sum(map_rcnt, map_rcnt+nprocs-1, map_rdspl+1);
    	IT totmapsend = map_sdspl[nprocs-1] + map_scnt[nprocs-1];
    	IT totmaprecv = map_rdspl[nprocs-1] + map_rcnt[nprocs-1];

    	// sendbuf is a pointer to array of MAXVERTNAME chars. 
    	// Explicit grouping syntax is due to precedence of [] over *
    	// char* sendbuf[MAXVERTNAME] would have declared a MAXVERTNAME-length array of char pointers
    	char (*sendbuf)[MAXVERTNAME];	// each sendbuf[i] is type char[MAXVERTNAME]
    	sendbuf = (char (*)[MAXVERTNAME]) malloc(sizeof(char[MAXVERTNAME])* totmapsend);	// notice that this is allocating a contiguous block of memory
    
    	IT * sendinds =  new IT[totmapsend];
    	for(int i=0; i<nprocs; ++i)
    	{	 
	    int loccnt = 0;
	    for(std::string s:data_send[i])
	    {
		    std::strcpy(sendbuf[map_sdspl[i]+loccnt], s.c_str());
		    loccnt++;
	    }
	    std::vector<std::string>().swap(data_send[i]);	// free memory
    	}
    	for(int i=0; i<nprocs; ++i)	      // sanity check: received indices should be sorted by definition
    	{ 
          std::copy(locs_send[i].begin(), locs_send[i].end(), sendinds+map_sdspl[i]);
        	std::vector<IT>().swap(locs_send[i]);	// free memory
    	}

    	char (*recvbuf)[MAXVERTNAME];	// recvbuf is of type char (*)[MAXVERTNAME]
    	recvbuf = (char (*)[MAXVERTNAME]) malloc(sizeof(char[MAXVERTNAME])* totmaprecv);

   	MPI_Datatype MPI_STRING;	// this is not necessary (we could just use char) but easier for bookkeeping
   	MPI_Type_contiguous(sizeof(char[MAXVERTNAME]), MPI_CHAR, &MPI_STRING);
    	MPI_Type_commit(&MPI_STRING);

    	MPI_Alltoallv(sendbuf, map_scnt, map_sdspl, MPI_STRING, recvbuf, map_rcnt, map_rdspl, MPI_STRING, comm);
    	free(sendbuf);	// can't delete[] so use free
    	MPI_Type_free(&MPI_STRING);

    	IT * recvinds = new IT[totmaprecv];
    	MPI_Alltoallv(sendinds, map_scnt, map_sdspl, MPIType<IT>(), recvinds, map_rcnt, map_rdspl, MPIType<IT>(), comm);
    	DeleteAll(sendinds, map_scnt, map_sdspl, map_rcnt, map_rdspl);

    	if(!std::is_sorted(recvinds, recvinds+totmaprecv))
	    	std::cout << "Assertion failed at proc " << myrank << ": Received indices are not sorted, this is unexpected" << std::endl;

    	for(IT i=0; i< totmaprecv; ++i)
    	{
		assert(i == recvinds[i]);
		std::copy(recvbuf[i], recvbuf[i]+MAXVERTNAME, distmapper_array[i].begin());
    	}
    	free(recvbuf);
    	delete [] recvinds;	
}


template<typename KEY, typename VAL, typename IT>
void SpParHelper::MemoryEfficientPSort(std::pair<KEY,VAL> * array, IT length, IT * dist, const MPI_Comm & comm)
{
	int nprocs, myrank;
	MPI_Comm_size(comm, &nprocs);
	MPI_Comm_rank(comm, &myrank);
	int nsize = nprocs / 2;	// new size
	if(nprocs < 10000)
	{
		bool excluded =  false;
		if(dist[myrank] == 0)	excluded = true;

		int nreals = 0; 
		for(int i=0; i< nprocs; ++i)	
			if(dist[i] != 0) ++nreals;

        //SpParHelper::MemoryEfficientPSort(vecpair, nnz, dist, World);
        
		if(nreals == nprocs)	// general case
		{
            long * dist_in = new long[nprocs];
            for(int i=0; i< nprocs; ++i)    dist_in[i] = (long) dist[i];
            vpsort::parallel_sort (array, array+length,  dist_in, comm);
            delete [] dist_in;
		}
		else
		{
			long * dist_in = new long[nreals];
			int * dist_out = new int[nprocs-nreals];	// ranks to exclude
			int indin = 0;
			int indout = 0;
			for(int i=0; i< nprocs; ++i)	
			{
				if(dist[i] == 0)
					dist_out[indout++] = i;
				else
					dist_in[indin++] = (long) dist[i];	
			}
		
			#ifdef DEBUG	
      std::ostringstream outs;
			outs << "To exclude indices: ";
      std::copy(dist_out, dist_out+indout, std::ostream_iterator<int>(outs, " ")); outs << std::endl;
			SpParHelper::Print(outs.str());
			#endif

			MPI_Group sort_group, real_group;
			MPI_Comm_group(comm, &sort_group);
			MPI_Group_excl(sort_group, indout, dist_out, &real_group);
			MPI_Group_free(&sort_group);

			// The Create() function should be executed by all processes in comm, 
			// even if they do not belong to the new group (in that case MPI_COMM_NULL is returned as real_comm?)
			// MPI::Intracomm MPI::Intracomm::Create(const MPI::Group& group) const;
			MPI_Comm real_comm;
			MPI_Comm_create(comm, real_group, &real_comm);
			if(!excluded)
			{
				vpsort::parallel_sort (array, array+length,  dist_in, real_comm);
				MPI_Comm_free(&real_comm);
			}
			MPI_Group_free(&real_group);
			delete [] dist_in;
			delete [] dist_out;
		}
	}
	else
	{
		IT gl_median = std::accumulate(dist, dist+nsize, static_cast<IT>(0));	// global rank of the first element of the median processor
		sort(array, array+length);	// re-sort because we might have swapped data in previous iterations
		int color = (myrank < nsize)? 0: 1;
		
		std::pair<KEY,VAL> * low = array;
		std::pair<KEY,VAL> * upp = array;
		GlobalSelect(gl_median, low, upp, array, length, comm);
		BipartiteSwap(low, array, length, nsize, color, comm);

		if(color == 1)	dist = dist + nsize;	// adjust for the second half of processors

		// recursive call; two implicit 'spawn's where half of the processors execute different paramaters
		// MPI::Intracomm MPI::Intracomm::Split(int color, int key) const;

		MPI_Comm halfcomm;
		MPI_Comm_split(comm, color, myrank, &halfcomm);	// split into two communicators
		MemoryEfficientPSort(array, length, dist, halfcomm);
	}
    
}


/*
 TODO: This function is just a hack at this moment. 
 The payload (VAL) can only be integer at this moment.
 FIX this.
 */
template<typename KEY, typename VAL, typename IT>
std::vector<std::pair<KEY,VAL>> SpParHelper::KeyValuePSort(std::pair<KEY,VAL> * array, IT length, IT * dist, const MPI_Comm & comm)
{
    int nprocs, myrank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &myrank);
    int nsize = nprocs / 2;	// new size
    
    
    
    bool excluded =  false;
    if(dist[myrank] == 0)	excluded = true;
    
    int nreals = 0;
    for(int i=0; i< nprocs; ++i)
        if(dist[i] != 0) ++nreals;
    
    std::vector<IndexHolder<KEY>> in(length);
#ifdef THREADED
#pragma omp parallel for
#endif
    for(int i=0; i< length; ++i)
    {
        in[i] = IndexHolder<KEY>(array[i].first, static_cast<unsigned long>(array[i].second));
    }
    
    if(nreals == nprocs)	// general case
    {
        par::sampleSort(in, comm);
    }
    else
    {
        long * dist_in = new long[nreals];
        int * dist_out = new int[nprocs-nreals];	// ranks to exclude
        int indin = 0;
        int indout = 0;
        for(int i=0; i< nprocs; ++i)
        {
            if(dist[i] == 0)
                dist_out[indout++] = i;
            else
                dist_in[indin++] = (long) dist[i];
        }
        
#ifdef DEBUG
        std::ostringstream outs;
        outs << "To exclude indices: ";
        std::copy(dist_out, dist_out+indout, std::ostream_iterator<int>(outs, " ")); outs << std::endl;
        SpParHelper::Print(outs.str());
#endif
        
        MPI_Group sort_group, real_group;
        MPI_Comm_group(comm, &sort_group);
        MPI_Group_excl(sort_group, indout, dist_out, &real_group);
        MPI_Group_free(&sort_group);
        
        // The Create() function should be executed by all processes in comm,
        // even if they do not belong to the new group (in that case MPI_COMM_NULL is returned as real_comm?)
        // MPI::Intracomm MPI::Intracomm::Create(const MPI::Group& group) const;
        MPI_Comm real_comm;
        MPI_Comm_create(comm, real_group, &real_comm);
        if(!excluded)
        {
            par::sampleSort(in, real_comm);
            MPI_Comm_free(&real_comm);
        }
        MPI_Group_free(&real_group);
        delete [] dist_in;
        delete [] dist_out;
    }

    std::vector<std::pair<KEY,VAL>> sorted(in.size());
    for(int i=0; i<in.size(); i++)
    {
        sorted[i].second = static_cast<VAL>(in[i].index);
        sorted[i].first = in[i].value;
    }
    return sorted;
}


template<typename KEY, typename VAL, typename IT>
void SpParHelper::GlobalSelect(IT gl_rank, std::pair<KEY,VAL> * & low,  std::pair<KEY,VAL> * & upp, std::pair<KEY,VAL> * array, IT length, const MPI_Comm & comm)
{
	int nprocs, myrank;
	MPI_Comm_size(comm, &nprocs);
	MPI_Comm_rank(comm, &myrank);
	IT begin = 0;
	IT end = length;	// initially everyone is active			
	std::pair<KEY, double> * wmminput = new std::pair<KEY,double>[nprocs];	// (median, #{actives})

	MPI_Datatype MPI_sortType;
	MPI_Type_contiguous (sizeof(std::pair<KEY,double>), MPI_CHAR, &MPI_sortType);
	MPI_Type_commit (&MPI_sortType);

	KEY wmm;	// our median pick
	IT gl_low, gl_upp;	
	IT active = end-begin;				// size of the active range
	IT nacts = 0; 
	bool found = 0;
	int iters = 0;
	
	/* changes by shan : to add begin0 and end0 for exit condition */
	IT begin0, end0;
	do
	{
		iters++;
		begin0 = begin; end0 = end;
		KEY median = array[(begin + end)/2].first; 	// median of the active range
		wmminput[myrank].first = median;
		wmminput[myrank].second = static_cast<double>(active);
		MPI_Allgather(MPI_IN_PLACE, 0, MPI_sortType, wmminput, 1, MPI_sortType, comm);
		double totact = 0;	// total number of active elements
		for(int i=0; i<nprocs; ++i)
			totact += wmminput[i].second;	

		// input to weighted median of medians is a set of (object, weight) pairs
		// the algorithm computes the first set of elements (according to total 
		// order of "object"s), whose sum is still less than or equal to 1/2
		for(int i=0; i<nprocs; ++i)
			wmminput[i].second /= totact ;	// normalize the weights
		
		sort(wmminput, wmminput+nprocs);	// sort w.r.t. medians
		double totweight = 0;
		int wmmloc=0;
		while( wmmloc<nprocs && totweight < 0.5 )
		{
			totweight += wmminput[wmmloc++].second;
		}

        	wmm = wmminput[wmmloc-1].first;	// weighted median of medians

		std::pair<KEY,VAL> wmmpair = std::make_pair(wmm, VAL());
		low =std::lower_bound (array+begin, array+end, wmmpair); 
		upp =std::upper_bound (array+begin, array+end, wmmpair); 
		IT loc_low = low-array;	// #{elements smaller than wmm}
		IT loc_upp = upp-array;	// #{elements smaller or equal to wmm}

		MPI_Allreduce( &loc_low, &gl_low, 1, MPIType<IT>(), MPI_SUM, comm);
		MPI_Allreduce( &loc_upp, &gl_upp, 1, MPIType<IT>(), MPI_SUM, comm);

		if(gl_upp < gl_rank)
		{
			// our pick was too small; only recurse to the right
			begin = (low - array);
		}
		else if(gl_rank < gl_low)
		{
			// our pick was too big; only recurse to the left
			end = (upp - array);
		} 
		else
		{	
			found = true;	
		}
		active = end-begin;
		MPI_Allreduce(&active, &nacts, 1, MPIType<IT>(), MPI_SUM, comm);
		if (begin0 == begin && end0 == end) break;  // ABAB: Active range did not shrink, so we break (is this kosher?)
	} 
	while((nacts > 2*nprocs) && (!found));
	delete [] wmminput;

    	MPI_Datatype MPI_pairType;
	MPI_Type_contiguous (sizeof(std::pair<KEY,VAL>), MPI_CHAR, &MPI_pairType);
	MPI_Type_commit (&MPI_pairType);

	int * nactives = new int[nprocs];
	nactives[myrank] = static_cast<int>(active);	// At this point, actives are small enough
	MPI_Allgather(MPI_IN_PLACE, 0, MPI_INT, nactives, 1, MPI_INT, comm);
	int * dpls = new int[nprocs]();	// displacements (zero initialized pid) 
	std::partial_sum(nactives, nactives+nprocs-1, dpls+1);
	std::pair<KEY,VAL> * recvbuf = new std::pair<KEY,VAL>[nacts];
	low = array + begin;	// update low to the beginning of the active range
	MPI_Allgatherv(low, active, MPI_pairType, recvbuf, nactives, dpls, MPI_pairType, comm);

	std::pair<KEY,int> * allactives = new std::pair<KEY,int>[nacts];
	int k = 0;
	for(int i=0; i<nprocs; ++i)
	{
		for(int j=0; j<nactives[i]; ++j)
		{
			allactives[k] = std::make_pair(recvbuf[k].first, i);
			k++;
		}
	}
	DeleteAll(recvbuf, dpls, nactives);
	sort(allactives, allactives+nacts); 
	MPI_Allreduce(&begin, &gl_low, 1, MPIType<IT>(), MPI_SUM, comm);        // update
	int diff = gl_rank - gl_low;
	for(int k=0; k < diff; ++k)
	{		
		if(allactives[k].second == myrank)	
			++low;	// increment the local pointer
	}
	delete [] allactives;
	begin = low-array;
	MPI_Allreduce(&begin, &gl_low, 1, MPIType<IT>(), MPI_SUM, comm);        // update
}

template<typename KEY, typename VAL, typename IT>
void SpParHelper::BipartiteSwap(std::pair<KEY,VAL> * low, std::pair<KEY,VAL> * array, IT length, int nfirsthalf, int color, const MPI_Comm & comm)
{
	int nprocs, myrank;
	MPI_Comm_size(comm, &nprocs);
	MPI_Comm_rank(comm, &myrank);

	IT * firsthalves = new IT[nprocs];
	IT * secondhalves = new IT[nprocs];	
	firsthalves[myrank] = low-array;
	secondhalves[myrank] = length - (low-array);

	MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IT>(), firsthalves, 1, MPIType<IT>(), comm);
	MPI_Allgather(MPI_IN_PLACE, 0, MPIType<IT>(), secondhalves, 1, MPIType<IT>(), comm);
	
	int * sendcnt = new int[nprocs]();	// zero initialize
	int totrecvcnt = 0; 

	std::pair<KEY,VAL> * bufbegin = NULL;
	if(color == 0)	// first processor half, only send second half of data
	{
		bufbegin = low;
		totrecvcnt = length - (low-array);
		IT beg_oftransfer = std::accumulate(secondhalves, secondhalves+myrank, static_cast<IT>(0));
		IT spaceafter = firsthalves[nfirsthalf];
		int i=nfirsthalf+1;
		while(i < nprocs && spaceafter < beg_oftransfer)
		{
			spaceafter += firsthalves[i++];		// post-incremenet
		}
		IT end_oftransfer = beg_oftransfer + secondhalves[myrank];	// global index (within second half) of the end of my data
		IT beg_pour = beg_oftransfer;
		IT end_pour = std::min(end_oftransfer, spaceafter);
		sendcnt[i-1] = end_pour - beg_pour;
		while( i < nprocs && spaceafter < end_oftransfer )	// find other recipients until I run out of data
		{
			beg_pour = end_pour;
			spaceafter += firsthalves[i];
			end_pour = std::min(end_oftransfer, spaceafter);
			sendcnt[i++] = end_pour - beg_pour;	// post-increment
		}
	}
	else if(color == 1)	// second processor half, only send first half of data
	{
		bufbegin = array;
		totrecvcnt = low-array;
		// global index (within the second processor half) of the beginning of my data
		IT beg_oftransfer = std::accumulate(firsthalves+nfirsthalf, firsthalves+myrank, static_cast<IT>(0));
		IT spaceafter = secondhalves[0];
		int i=1;
		while( i< nfirsthalf && spaceafter < beg_oftransfer)
		{
			//spacebefore = spaceafter;
			spaceafter += secondhalves[i++];	// post-increment
		}
		IT end_oftransfer = beg_oftransfer + firsthalves[myrank];	// global index (within second half) of the end of my data
		IT beg_pour = beg_oftransfer;
		IT end_pour = std::min(end_oftransfer, spaceafter);
		sendcnt[i-1] = end_pour - beg_pour;
		while( i < nfirsthalf && spaceafter < end_oftransfer )	// find other recipients until I run out of data
		{
			beg_pour = end_pour;
			spaceafter += secondhalves[i];
			end_pour = std::min(end_oftransfer, spaceafter);
			sendcnt[i++] = end_pour - beg_pour;	// post-increment
		}
	}
	DeleteAll(firsthalves, secondhalves);
	int * recvcnt = new int[nprocs];
	MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, comm);   // get the recv counts
	// Alltoall is actually unnecessary, because sendcnt = recvcnt
	// If I have n_mine > n_yours data to send, then I can send you only n_yours 
	// as this is your space, and you'll send me identical amount.
	// Then I can only receive n_mine - n_yours from the third processor and
	// that processor can only send n_mine - n_yours to me back. 
	// The proof follows from induction

	MPI_Datatype MPI_valueType;
	MPI_Type_contiguous(sizeof(std::pair<KEY,VAL>), MPI_CHAR, &MPI_valueType);
	MPI_Type_commit(&MPI_valueType);

	std::pair<KEY,VAL> * receives = new std::pair<KEY,VAL>[totrecvcnt];
	int * sdpls = new int[nprocs]();	// displacements (zero initialized pid) 
	int * rdpls = new int[nprocs](); 
	std::partial_sum(sendcnt, sendcnt+nprocs-1, sdpls+1);
	std::partial_sum(recvcnt, recvcnt+nprocs-1, rdpls+1);

	MPI_Alltoallv(bufbegin, sendcnt, sdpls, MPI_valueType, receives, recvcnt, rdpls, MPI_valueType, comm);  // sparse swap
	
	DeleteAll(sendcnt, recvcnt, sdpls, rdpls);
  std::copy(receives, receives+totrecvcnt, bufbegin);
	delete [] receives;
}


template<typename KEY, typename VAL, typename IT>
void SpParHelper::DebugPrintKeys(std::pair<KEY,VAL> * array, IT length, IT * dist, MPI_Comm & World)
{
	int rank, nprocs;
	MPI_Comm_rank(World, &rank);
	MPI_Comm_size(World, &nprocs);
	MPI_File thefile;
    
    char _fn[] = "temp_sortedkeys"; // AL: this is to avoid the problem that C++ string literals are const char* while C string literals are char*, leading to a const warning (technically error, but compilers are tolerant)
	MPI_File_open(World, _fn, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &thefile);    

	// The cast in the last parameter is crucial because the signature of the function is
   	// T accumulate ( InputIterator first, InputIterator last, T init )
	// Hence if init if of type "int", the output also becomes an it (remember C++ signatures are ignorant of return value)
	IT sizeuntil = std::accumulate(dist, dist+rank, static_cast<IT>(0)); 

	MPI_Offset disp = sizeuntil * sizeof(KEY);	// displacement is in bytes
    	MPI_File_set_view(thefile, disp, MPIType<KEY>(), MPIType<KEY>(), "native", MPI_INFO_NULL);

	KEY * packed = new KEY[length];
	for(int i=0; i<length; ++i)
	{
		packed[i] = array[i].first;
	}
	MPI_File_write(thefile, packed, length, MPIType<KEY>(), NULL);
	MPI_File_close(&thefile);
	delete [] packed;
	
	// Now let processor-0 read the file and print
	if(rank == 0)
	{
		FILE * f = fopen("temp_sortedkeys", "r");
                if(!f)
                { 
                        std::cerr << "Problem reading binary input file\n";
                        return;
                }
		IT maxd = *std::max_element(dist, dist+nprocs);
		KEY * data = new KEY[maxd];

		for(int i=0; i<nprocs; ++i)
		{
			// read n_per_proc integers and print them
			fread(data, sizeof(KEY), dist[i],f);

			std::cout << "Elements stored on proc " << i << ": " << std::endl;
			std::copy(data, data+dist[i], std::ostream_iterator<KEY>(std::cout, "\n"));
		}
		delete [] data;
	}
}


/**
  * @param[in,out] MRecv {an already existing, but empty SpMat<...> object}
  * @param[in] essentials {carries essential information (i.e. required array sizes) about ARecv}
  * @param[in] arrwin {windows array of size equal to the number of built-in arrays in the SpMat data structure}
  * @param[in] ownind {processor index (within this processor row/column) of the owner of the matrix to be received}
  * @remark {The communicator information is implicitly contained in the MPI::Win objects}
 **/
template <class IT, class NT, class DER>
void SpParHelper::FetchMatrix(SpMat<IT,NT,DER> & MRecv, const std::vector<IT> & essentials, std::vector<MPI_Win> & arrwin, int ownind)
{
	MRecv.Create(essentials);		// allocate memory for arrays
 
	Arr<IT,NT> arrinfo = MRecv.GetArrays();
	assert( (arrwin.size() == arrinfo.totalsize()));

	// C-binding for MPI::Get
	//	int MPI_Get(void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp,
        //    		int target_count, MPI_Datatype target_datatype, MPI_Win win)

	IT essk = 0;
	for(int i=0; i< arrinfo.indarrs.size(); ++i)	// get index arrays
	{
		//arrwin[essk].Lock(MPI::LOCK_SHARED, ownind, 0);
		MPI_Get( arrinfo.indarrs[i].addr, arrinfo.indarrs[i].count, MPIType<IT>(), ownind, 0, arrinfo.indarrs[i].count, MPIType<IT>(), arrwin[essk++]);
	}
	for(int i=0; i< arrinfo.numarrs.size(); ++i)	// get numerical arrays
	{
		//arrwin[essk].Lock(MPI::LOCK_SHARED, ownind, 0);
		MPI_Get(arrinfo.numarrs[i].addr, arrinfo.numarrs[i].count, MPIType<NT>(), ownind, 0, arrinfo.numarrs[i].count, MPIType<NT>(), arrwin[essk++]);
	}
}


/**
  * @param[in] Matrix {For the root processor, the local object to be sent to all others.
  * 		For all others, it is a (yet) empty object to be filled by the received data}
  * @param[in] essentials {irrelevant for the root}
 **/
template<typename IT, typename NT, typename DER>	
void SpParHelper::BCastMatrix(MPI_Comm & comm1d, SpMat<IT,NT,DER> & Matrix, const std::vector<IT> & essentials, int root)
{
	int myrank;
	MPI_Comm_rank(comm1d, &myrank);
	if(myrank != root)
	{
		Matrix.Create(essentials);		// allocate memory for arrays		
	}

	Arr<IT,NT> arrinfo = Matrix.GetArrays();
	for(unsigned int i=0; i< arrinfo.indarrs.size(); ++i)	// get index arrays
	{
		MPI_Bcast(arrinfo.indarrs[i].addr, arrinfo.indarrs[i].count, MPIType<IT>(), root, comm1d);
	}
	for(unsigned int i=0; i< arrinfo.numarrs.size(); ++i)	// get numerical arrays
	{
		MPI_Bcast(arrinfo.numarrs[i].addr, arrinfo.numarrs[i].count, MPIType<NT>(), root, comm1d);
	}			
}

/**
  * @param[in] Matrix {For the root processor, the local object to be sent to all others.
  * 		For all others, it is a (yet) empty object to be filled by the received data}
  * @param[in] essentials {irrelevant for the root}
 **/
template<typename IT, typename NT, typename DER>	
void SpParHelper::IBCastMatrix(MPI_Comm & comm1d, SpMat<IT,NT,DER> & Matrix, const std::vector<IT> & essentials, int root, std::vector<MPI_Request> & indarrayReq , std::vector<MPI_Request> & numarrayReq)
{
	int myrank;
	MPI_Comm_rank(comm1d, &myrank);
	if(myrank != root)
	{
		Matrix.Create(essentials);		// allocate memory for arrays		
	}

	Arr<IT,NT> arrinfo = Matrix.GetArrays();
	for(unsigned int i=0; i< arrinfo.indarrs.size(); ++i)	// get index arrays
	{
		MPI_Ibcast(arrinfo.indarrs[i].addr, arrinfo.indarrs[i].count, MPIType<IT>(), root, comm1d, &indarrayReq[i]);
	}
	for(unsigned int i=0; i< arrinfo.numarrs.size(); ++i)	// get numerical arrays
	{
		MPI_Ibcast(arrinfo.numarrs[i].addr, arrinfo.numarrs[i].count, MPIType<NT>(), root, comm1d, &numarrayReq[i]);
	}			
}

/**
 * Just a test function to see the time to gather a matrix on an MPI process
 * The ultimate object would be to create the whole matrix on rank 0 (TODO)
 * @param[in] Matrix {For the root processor, the local object to be sent to all others.
 * 		For all others, it is a (yet) empty object to be filled by the received data}
 * @param[in] essentials {irrelevant for the root}
 **/
template<typename IT, typename NT, typename DER>
void SpParHelper::GatherMatrix(MPI_Comm & comm1d, SpMat<IT,NT,DER> & Matrix, int root)
{
    int myrank, nprocs;
    MPI_Comm_rank(comm1d, &myrank);
    MPI_Comm_size(comm1d,&nprocs);

    /*
    if(myrank != root)
    {
        Matrix.Create(essentials);		// allocate memory for arrays
    }
     */
    
    Arr<IT,NT> arrinfo = Matrix.GetArrays();
    std::vector<std::vector<int>> recvcnt_ind(arrinfo.indarrs.size());
    std::vector<std::vector<int>> recvcnt_num(arrinfo.numarrs.size());
    for(unsigned int i=0; i< arrinfo.indarrs.size(); ++i)	// get index arrays
    {
        recvcnt_ind[i].resize(nprocs);
        int lcount = (int)arrinfo.indarrs[i].count;
        MPI_Gather(&lcount, 1, MPI_INT, recvcnt_ind[i].data(),1, MPI_INT, root, comm1d);
    }
    for(unsigned int i=0; i< arrinfo.numarrs.size(); ++i)	// get numerical arrays
    {
        recvcnt_num[i].resize(nprocs);
        int lcount = (int) arrinfo.numarrs[i].count;
        MPI_Gather(&lcount, 1, MPI_INT, recvcnt_num[i].data(),1, MPI_INT, root, comm1d);
    }
    
    // now gather the actual vector
    std::vector<std::vector<int>> recvdsp_ind(arrinfo.indarrs.size());
    std::vector<std::vector<int>> recvdsp_num(arrinfo.numarrs.size());
    std::vector<std::vector<IT>> recvind(arrinfo.indarrs.size());
    std::vector<std::vector<IT>> recvnum(arrinfo.numarrs.size());
    for(unsigned int i=0; i< arrinfo.indarrs.size(); ++i)	// get index arrays
    {
        recvdsp_ind[i].resize(nprocs);
        recvdsp_ind[i][0] = 0;
        for(int j=1; j<nprocs; j++)
            recvdsp_ind[i][j] = recvdsp_ind[i][j-1] + recvcnt_ind[i][j-1];
        recvind[i].resize(recvdsp_ind[i][nprocs-1] + recvcnt_ind[i][nprocs-1]);
        MPI_Gatherv(arrinfo.indarrs[i].addr, arrinfo.indarrs[i].count, MPIType<IT>(), recvind[i].data(),recvcnt_ind[i].data(), recvdsp_ind[i].data(), MPIType<IT>(), root, comm1d);
    }
    
    
    for(unsigned int i=0; i< arrinfo.numarrs.size(); ++i)	// gather num arrays
    {
        recvdsp_num[i].resize(nprocs);
        recvdsp_num[i][0] = 0;
        for(int j=1; j<nprocs; j++)
            recvdsp_num[i][j] = recvdsp_num[i][j-1] + recvcnt_num[i][j-1];
        recvnum[i].resize(recvdsp_num[i][nprocs-1] + recvcnt_num[i][nprocs-1]);
        MPI_Gatherv(arrinfo.numarrs[i].addr, arrinfo.numarrs[i].count, MPIType<NT>(), recvnum[i].data(),recvcnt_num[i].data(), recvdsp_num[i].data(), MPIType<NT>(), root, comm1d);
    }
}


template <class IT, class NT, class DER>
void SpParHelper::SetWindows(MPI_Comm & comm1d, const SpMat< IT,NT,DER > & Matrix, std::vector<MPI_Win> & arrwin) 
{	
	Arr<IT,NT> arrs = Matrix.GetArrays(); 
	 
	// static MPI::Win MPI::Win::create(const void *base, MPI::Aint size, int disp_unit, MPI::Info info, const MPI_Comm & comm);
	// The displacement unit argument is provided to facilitate address arithmetic in RMA operations
	// *** COLLECTIVE OPERATION ***, everybody exposes its own array to everyone else in the communicator
		
	for(int i=0; i< arrs.indarrs.size(); ++i)
	{
	        MPI_Win nWin;
	        MPI_Win_create(arrs.indarrs[i].addr, 
			       arrs.indarrs[i].count * sizeof(IT), sizeof(IT), MPI_INFO_NULL, comm1d, &nWin);
		arrwin.push_back(nWin);
	}
	for(int i=0; i< arrs.numarrs.size(); ++i)
	{
	        MPI_Win nWin;
		MPI_Win_create(arrs.numarrs[i].addr, 
			       arrs.numarrs[i].count * sizeof(NT), sizeof(NT), MPI_INFO_NULL, comm1d, &nWin);
		arrwin.push_back(nWin);
	}	
}

inline void SpParHelper::LockWindows(int ownind, std::vector<MPI_Win> & arrwin)
{
	for(std::vector<MPI_Win>::iterator itr = arrwin.begin(); itr != arrwin.end(); ++itr)
	{
		MPI_Win_lock(MPI_LOCK_SHARED, ownind, 0, *itr);
	}
}

inline void SpParHelper::UnlockWindows(int ownind, std::vector<MPI_Win> & arrwin) 
{
	for(std::vector<MPI_Win>::iterator itr = arrwin.begin(); itr != arrwin.end(); ++itr)
	{
		MPI_Win_unlock( ownind, *itr);
	}
}


/**
 * @param[in] owner {target processor rank within the processor group} 
 * @param[in] arrwin {start access epoch only to owner's arrwin (-windows) }
 */
inline void SpParHelper::StartAccessEpoch(int owner, std::vector<MPI_Win> & arrwin, MPI_Group & group)
{
	/* Now start using the whole comm as a group */
	int acc_ranks[1]; 
	acc_ranks[0] = owner;
	MPI_Group access;
	MPI_Group_incl(group, 1, acc_ranks, &access);	// take only the owner

	// begin the ACCESS epochs for the arrays of the remote matrices A and B
	// Start() *may* block until all processes in the target group have entered their exposure epoch
	for(unsigned int i=0; i< arrwin.size(); ++i)
	       MPI_Win_start(access, 0, arrwin[i]);

	MPI_Group_free(&access);
}

/**
 * @param[in] self {rank of "this" processor to be excluded when starting the exposure epoch} 
 */
inline void SpParHelper::PostExposureEpoch(int self, std::vector<MPI_Win> & arrwin, MPI_Group & group)
{
	// begin the EXPOSURE epochs for the arrays of the local matrices A and B
	for(unsigned int i=0; i< arrwin.size(); ++i)
	       MPI_Win_post(group, MPI_MODE_NOPUT, arrwin[i]);
}

template <class IT, class DER>
void SpParHelper::AccessNFetch(DER * & Matrix, int owner, std::vector<MPI_Win> & arrwin, MPI_Group & group, IT ** sizes)
{
	StartAccessEpoch(owner, arrwin, group);	// start the access epoch to arrwin of owner

	std::vector<IT> ess(DER::esscount);			// pack essentials to a vector
	for(int j=0; j< DER::esscount; ++j)	
		ess[j] = sizes[j][owner];	

	Matrix = new DER();	// create the object first	
	FetchMatrix(*Matrix, ess, arrwin, owner);	// then start fetching its elements
}

template <class IT, class DER>
void SpParHelper::LockNFetch(DER * & Matrix, int owner, std::vector<MPI_Win> & arrwin, MPI_Group & group, IT ** sizes)
{
	LockWindows(owner, arrwin);

	std::vector<IT> ess(DER::esscount);			// pack essentials to a vector
	for(int j=0; j< DER::esscount; ++j)	
		ess[j] = sizes[j][owner];	

	Matrix = new DER();	// create the object first	
	FetchMatrix(*Matrix, ess, arrwin, owner);	// then start fetching its elements
}

/**
 * @param[in] sizes 2D array where 
 *  	sizes[i] is an array of size r/s representing the ith essential component of all local blocks within that row/col
 *	sizes[i][j] is the size of the ith essential component of the jth local block within this row/col
 */
template <class IT, class NT, class DER>
void SpParHelper::GetSetSizes(const SpMat<IT,NT,DER> & Matrix, IT ** & sizes, MPI_Comm & comm1d)
{
	std::vector<IT> essentials = Matrix.GetEssentials();
	int index;
	MPI_Comm_rank(comm1d, &index);

	for(IT i=0; (unsigned)i < essentials.size(); ++i)
	{
		sizes[i][index] = essentials[i]; 
		MPI_Allgather(MPI_IN_PLACE, 1, MPIType<IT>(), sizes[i], 1, MPIType<IT>(), comm1d);
	}	
}

inline void SpParHelper::PrintFile(const std::string & s, const std::string & filename)
{
	int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	if(myrank == 0)
	{
		std::ofstream out(filename.c_str(), std::ofstream::app);
		out << s;
		out.close();
	}
}

inline void SpParHelper::PrintFile(const std::string & s, const std::string & filename, MPI_Comm & world)
{
    int myrank;
    MPI_Comm_rank(world, &myrank);
    if(myrank == 0)
    {
        std::ofstream out(filename.c_str(), std::ofstream::app);
        out << s;
        out.close();
    }
}


inline void SpParHelper::Print(const std::string & s)
{
	int myrank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	if(myrank == 0)
	{
		std::cerr << s;
	}
}

inline void SpParHelper::Print(const std::string & s, MPI_Comm & world)
{
    int myrank;
    MPI_Comm_rank(world, &myrank);
    if(myrank == 0)
    {
        std::cerr << s;
    }
}


inline void SpParHelper::check_newline(int *bytes_read, int bytes_requested, char *buf)
{
    if ((*bytes_read) < bytes_requested) {
        // fewer bytes than expected, this means EOF
        if (buf[(*bytes_read) - 1] != '\n') {
            // doesn't terminate with a newline, add one to prevent infinite loop later
            buf[(*bytes_read) - 1] = '\n';
            std::cout << "Error in Matrix Market format, appending missing newline at end of file" << std::endl;
            (*bytes_read)++;
        }
    }
}


inline bool SpParHelper::FetchBatch(MPI_File & infile, MPI_Offset & curpos, MPI_Offset end_fpos, bool firstcall, std::vector<std::string> & lines, int myrank)
{
    size_t bytes2fetch = ONEMILLION;    // we might read more than needed but no problem as we won't process them
    MPI_Status status;
    int bytes_read;
    if(firstcall && myrank != 0)
    {
        curpos -= 1;    // first byte is to check whether we started at the beginning of a line
        bytes2fetch += 1;
    }
    char * buf = new char[bytes2fetch]; // needs to happen **after** bytes2fetch is updated
    char * originalbuf = buf;   // so that we can delete it later because "buf" will move
    
    MPI_File_read_at(infile, curpos, buf, bytes2fetch, MPI_CHAR, &status);
    MPI_Get_count(&status, MPI_CHAR, &bytes_read);  // MPI_Get_Count can only return 32-bit integers
    if(!bytes_read)
    {
        delete [] originalbuf;
        return true;    // done
    }
    SpParHelper::check_newline(&bytes_read, bytes2fetch, buf);
    if(firstcall && myrank != 0)
    {
        if(buf[0] == '\n')  // we got super lucky and hit the line break
        {
            buf += 1;
            bytes_read -= 1;
            curpos += 1;
        }
        else    // skip to the next line and let the preceeding processor take care of this partial line
        {
            char *c = (char*)memchr(buf, '\n', MAXLINELENGTH); //  return a pointer to the matching byte or NULL if the character does not occur
            if (c == NULL) {
                std::cout << "Unexpected line without a break" << std::endl;
            }
            int n = c - buf + 1;
            bytes_read -= n;
            buf += n;
            curpos += n;
        }
    }
    while(bytes_read > 0 && curpos < end_fpos)  // this will also finish the last line
    {
        char *c = (char*)memchr(buf, '\n', bytes_read); //  return a pointer to the matching byte or NULL if the character does not occur
        if (c == NULL) {
            delete [] originalbuf;
            return false;  // if bytes_read stops in the middle of a line, that line will be re-read next time since curpos has not been moved forward yet
        }
        int n = c - buf + 1;
        
        // string constructor from char * buffer: copies the first n characters from the array of characters pointed by s
        lines.push_back(std::string(buf, n-1));  // no need to copy the newline character
        bytes_read -= n;   // reduce remaining bytes
        buf += n;   // move forward the buffer
        curpos += n;
    }
    delete [] originalbuf;
    if (curpos >= end_fpos) return true;  // don't call it again, nothing left to read
    else    return false;
}


inline void SpParHelper::WaitNFree(std::vector<MPI_Win> & arrwin)
{
	// End the exposure epochs for the arrays of the local matrices A and B
	// The Wait() call matches calls to Complete() issued by ** EACH OF THE ORIGIN PROCESSES ** 
	// that were granted access to the window during this epoch.
	for(unsigned int i=0; i< arrwin.size(); ++i)
	{
		MPI_Win_wait(arrwin[i]);
	}
	FreeWindows(arrwin);
}		
	
inline void SpParHelper::FreeWindows(std::vector<MPI_Win> & arrwin)
{
	for(unsigned int i=0; i< arrwin.size(); ++i)
	{
		MPI_Win_free(&arrwin[i]);
	}
}

}
