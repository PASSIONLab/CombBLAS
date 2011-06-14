/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.2 -------------------------------------------------*/
/* date: 10/06/2011 --------------------------------------------*/
/* authors: Aydin Buluc (abuluc@lbl.gov), Adam Lugowski --------*/
/****************************************************************/
/*
Copyright (c) 2011, Aydin Buluc

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

template<typename KEY, typename VAL, typename IT>
void SpParHelper::MemoryEfficientPSort(pair<KEY,VAL> * array, IT length, IT * dist, const MPI::Intracomm & comm)
{	
	int nprocs = comm.Get_size();
	int myrank = comm.Get_rank();
	int nsize = nprocs / 2;	// new size
	if(nprocs < 1000)
	{
		bool excluded =  false;
		if(dist[myrank] == 0)	excluded = true;

		int nreals = 0; 
		for(int i=0; i< nprocs; ++i)	
			if(dist[i] != 0) ++nreals;

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
			ostringstream outs;
			outs << "To exclude indices: ";
			copy(dist_out, dist_out+indout, ostream_iterator<int>(outs, " ")); outs << endl;
			SpParHelper::Print(outs.str());
			#endif
			// MPI::Group MPI::Comm::Get_group() const;
			MPI::Group sort_group = comm.Get_group();
			MPI::Group real_group = sort_group.Excl(indout, dist_out);
			sort_group.Free();

			// The Create() function should be executed by all processes in comm, 
			// even if they do not belong to the new group (in that case MPI_COMM_NULL is returned as real_comm?)
			// MPI::Intracomm MPI::Intracomm::Create(const MPI::Group& group) const;
			MPI::Intracomm real_comm = comm.Create(real_group);
			if(!excluded)
			{
				vpsort::parallel_sort (array, array+length,  dist_in, real_comm);
				real_comm.Free();
			}
			real_group.Free();
			delete [] dist_in;
			delete [] dist_out;
		}
	}
	else
	{
		IT gl_median = accumulate(dist, dist+nsize, static_cast<IT>(0));	// global rank of the first element of the median processor
		sort(array, array+length);	// re-sort because we might have swapped data in previous iterations
		int color = (myrank < nsize)? 0: 1;
		
		pair<KEY,VAL> * low = array;
		pair<KEY,VAL> * upp = array;
		GlobalSelect(gl_median, low, upp, array, length, comm);
		BipartiteSwap(low, array, length, nsize, color, comm);

		if(color == 1)	dist = dist + nsize;	// adjust for the second half of processors

		// recursive call; two implicit 'spawn's where half of the processors execute different paramaters
		// MPI::Intracomm MPI::Intracomm::Split(int color, int key) const;
		MPI::Intracomm halfcomm = comm.Split(color, myrank);	// split into two communicators
		MemoryEfficientPSort(array, length, dist, halfcomm);
	}
}

template<typename KEY, typename VAL, typename IT>
void SpParHelper::GlobalSelect(IT gl_rank, pair<KEY,VAL> * & low,  pair<KEY,VAL> * & upp, pair<KEY,VAL> * array, IT length, const MPI::Intracomm & comm)
{
//	comm.Barrier();
//	double t1=MPI::Wtime();
			
	int nprocs = comm.Get_size();
	int myrank = comm.Get_rank();
	IT begin = 0;
	IT end = length;	// initially everyone is active			
	pair<KEY, double> * wmminput = new pair<KEY,double>[nprocs];	// (median, #{actives})

    	MPI_Datatype MPI_sortType;
	MPI_Type_contiguous (sizeof(pair<KEY,double>), MPI_CHAR, &MPI_sortType);
	MPI_Type_commit (&MPI_sortType);

	KEY wmm;	// our median pick
	IT gl_low, gl_upp;	
	IT active = end-begin;				// size of the active range
	IT nacts = 0; 
	bool found = 0;
	int iters = 0;
	do
	{
		iters++;
		KEY median = array[(begin + end)/2].first; 	// median of the active range
               	wmminput[myrank].first = median;
		wmminput[myrank].second = static_cast<double>(active);
		comm.Allgather(MPI::IN_PLACE, 0, MPI_sortType, wmminput, 1, MPI_sortType);
		double totact = 0;	// total number of active elements
		for(int i=0; i<nprocs; ++i)
			totact += wmminput[i].second;	

		// input to weighted median of medians is a set of (object, weight) pairs
		// the algorithm computers the first set of elements (according to total 
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

		pair<KEY,VAL> wmmpair = make_pair(wmm, VAL());
		low =lower_bound (array+begin, array+end, wmmpair); 
		upp =upper_bound (array+begin, array+end, wmmpair); 
		IT loc_low = low-array;	// #{elements smaller than wmm}
		IT loc_upp = upp-array;	// #{elements smaller or equal to wmm}
		comm.Allreduce( &loc_low, &gl_low, 1, MPIType<IT>(), MPI::SUM);
		comm.Allreduce( &loc_upp, &gl_upp, 1, MPIType<IT>(), MPI::SUM);

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
		comm.Allreduce(&active, &nacts, 1, MPIType<IT>(), MPI::SUM);
	} 
	while((nacts > 2*nprocs) && (!found));
	delete [] wmminput;

    	MPI_Datatype MPI_pairType;
	MPI_Type_contiguous (sizeof(pair<KEY,VAL>), MPI_CHAR, &MPI_pairType);
	MPI_Type_commit (&MPI_pairType);

	int * nactives = new int[nprocs];
	nactives[myrank] = static_cast<int>(active);	// At this point, actives are small enough
	comm.Allgather(MPI::IN_PLACE, 0, MPI::INT, nactives, 1, MPI::INT);
	int * dpls = new int[nprocs]();	// displacements (zero initialized pid) 
	partial_sum(nactives, nactives+nprocs-1, dpls+1);
	pair<KEY,VAL> * recvbuf = new pair<KEY,VAL>[nacts];
	low = array + begin;	// update low to the beginning of the active range
	comm.Allgatherv(low, active, MPI_pairType, recvbuf, nactives, dpls, MPI_pairType);

	pair<KEY,int> * allactives = new pair<KEY,int>[nacts];
	int k = 0;
	for(int i=0; i<nprocs; ++i)
	{
		for(int j=0; j<nactives[i]; ++j)
		{
			allactives[k] = make_pair(recvbuf[k].first, i);
			k++;
		}
	}
	DeleteAll(recvbuf, dpls, nactives);
	sort(allactives, allactives+nacts); 
	comm.Allreduce(&begin, &gl_low, 1, MPIType<IT>(), MPI::SUM);	// update
	int diff = gl_rank - gl_low;
	for(int k=0; k < diff; ++k)
	{		
		if(allactives[k].second == myrank)	
			++low;	// increment the local pointer
	}
	delete [] allactives;
	begin = low-array;
	comm.Allreduce(&begin, &gl_low, 1, MPIType<IT>(), MPI::SUM); 	// update

//	comm.Barrier();
//	double t2 = MPI::Wtime();
//	if(myrank == 0)
//		fprintf(stdout, "%.6lf seconds and %d iterations elapsed for Median finding\n", t2-t1, iters);
}

template<typename KEY, typename VAL, typename IT>
void SpParHelper::BipartiteSwap(pair<KEY,VAL> * low, pair<KEY,VAL> * array, IT length, int nfirsthalf, int color, const MPI::Intracomm & comm)
{
//	comm.Barrier();
//	double t1=MPI::Wtime();

	int nprocs = comm.Get_size();
	int myrank = comm.Get_rank();
	IT * firsthalves = new IT[nprocs];
	IT * secondhalves = new IT[nprocs];	
	firsthalves[myrank] = low-array;
	secondhalves[myrank] = length - (low-array);
	comm.Allgather(MPI::IN_PLACE, 0, MPIType<IT>(), firsthalves, 1, MPIType<IT>());
	comm.Allgather(MPI::IN_PLACE, 0, MPIType<IT>(), secondhalves, 1, MPIType<IT>());
	
	int * sendcnt = new int[nprocs]();	// zero initialize
	int totrecvcnt = 0; 
	//IT spacebefore = 0;	// receiving part space

	pair<KEY,VAL> * bufbegin = NULL;
	if(color == 0)	// first processor half, only send second half of data
	{
		bufbegin = low;
		totrecvcnt = length - (low-array);
		IT beg_oftransfer = accumulate(secondhalves, secondhalves+myrank, static_cast<IT>(0));
		IT spaceafter = firsthalves[nfirsthalf];
		int i=nfirsthalf+1;
		while(i < nprocs && spaceafter < beg_oftransfer)
		{
			//spacebefore = spaceafter;
			spaceafter += firsthalves[i++];		// post-incremenet
		}
		IT end_oftransfer = beg_oftransfer + secondhalves[myrank];	// global index (within second half) of the end of my data
		IT beg_pour = beg_oftransfer;
		IT end_pour = min(end_oftransfer, spaceafter);
		sendcnt[i-1] = end_pour - beg_pour;
		while( i < nprocs && spaceafter < end_oftransfer )	// find other recipients until I run out of data
		{
			beg_pour = end_pour;
			spaceafter += firsthalves[i];
			end_pour = min(end_oftransfer, spaceafter);
			sendcnt[i++] = end_pour - beg_pour;	// post-increment
		}
	}
	else if(color == 1)	// second processor half, only send first half of data
	{
		bufbegin = array;
		totrecvcnt = low-array;
		// global index (within the second processor half) of the beginning of my data
		IT beg_oftransfer = accumulate(firsthalves+nfirsthalf, firsthalves+myrank, static_cast<IT>(0));
		IT spaceafter = secondhalves[0];
		int i=1;
		while( i< nfirsthalf && spaceafter < beg_oftransfer)
		{
			//spacebefore = spaceafter;
			spaceafter += secondhalves[i++];	// post-increment
		}
		IT end_oftransfer = beg_oftransfer + firsthalves[myrank];	// global index (within second half) of the end of my data
		IT beg_pour = beg_oftransfer;
		IT end_pour = min(end_oftransfer, spaceafter);
		sendcnt[i-1] = end_pour - beg_pour;
		while( i < nfirsthalf && spaceafter < end_oftransfer )	// find other recipients until I run out of data
		{
			beg_pour = end_pour;
			spaceafter += secondhalves[i];
			end_pour = min(end_oftransfer, spaceafter);
			sendcnt[i++] = end_pour - beg_pour;	// post-increment
		}
	}
	DeleteAll(firsthalves, secondhalves);
	int * recvcnt = new int[nprocs];
	comm.Alltoall(sendcnt, 1, MPI::INT, recvcnt, 1, MPI::INT);	// get the recv counts
	// Alltoall is actually unnecessary, because sendcnt = recvcnt
	// If I have n_mine > n_yours data to send, then I can send you only n_yours 
	// as this is your space, and you'll send me identical amount.
	// Then I can only receive n_mine - n_yours from the third processor and
	// that processor can only send n_mine - n_yours to me back. 
	// The proof follows from induction

	MPI::Datatype MPI_valueType = MPI::CHAR.Create_contiguous(sizeof(pair<KEY,VAL>));
	MPI_valueType.Commit();

//	double t2 = MPI::Wtime();
//	if(myrank == 0)
//		fprintf(stdout, "%.6lf seconds elapsed for setting up swap structures on %d procs\n", t2-t1, nprocs);
	
	pair<KEY,VAL> * receives = new pair<KEY,VAL>[totrecvcnt];
	int * sdpls = new int[nprocs]();	// displacements (zero initialized pid) 
	int * rdpls = new int[nprocs](); 
	partial_sum(sendcnt, sendcnt+nprocs-1, sdpls+1);
	partial_sum(recvcnt, recvcnt+nprocs-1, rdpls+1);

	comm.Alltoallv(bufbegin, sendcnt, sdpls, MPI_valueType, receives, recvcnt, rdpls, MPI_valueType);  // sparse swap

//	double t3 = MPI::Wtime();
//	if(myrank == 0)
//		fprintf(stdout, "%.6lf seconds elapsed for actual data swap on %d procs\n", t3-t2, nprocs);
	
	DeleteAll(sendcnt, recvcnt, sdpls, rdpls);
	copy(receives, receives+totrecvcnt, bufbegin);
	delete [] receives;
}


template<typename KEY, typename VAL, typename IT>
void SpParHelper::DebugPrintKeys(pair<KEY,VAL> * array, IT length, IT * dist, MPI::Intracomm & World)
{
    	int rank = World.Get_rank();
    	int nprocs = World.Get_size();
    	MPI::File thefile = MPI::File::Open(World, "temp_sortedkeys", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI::INFO_NULL);    

	// The cast in the last parameter is crucial because the signature of the function is
   	// T accumulate ( InputIterator first, InputIterator last, T init )
	// Hence if init if of type "int", the output also becomes an it (remember C++ signatures are ignorant of return value)
	IT sizeuntil = accumulate(dist, dist+rank, static_cast<IT>(0)); 

	MPI::Offset disp = sizeuntil * sizeof(KEY);	// displacement is in bytes
    	thefile.Set_view(disp, MPIType<KEY>(), MPIType<KEY>(), "native", MPI::INFO_NULL);

	KEY * packed = new KEY[length];
	for(int i=0; i<length; ++i)
	{
		packed[i] = array[i].first;
	}
	thefile.Write(packed, length, MPIType<KEY>());
	thefile.Close();
	delete [] packed;
	
	// Now let processor-0 read the file and print
	if(rank == 0)
	{
		FILE * f = fopen("temp_sortedkeys", "r");
                if(!f)
                { 
                        cerr << "Problem reading binary input file\n";
                        return;
                }
		IT maxd = *max_element(dist, dist+nprocs);
		KEY * data = new KEY[maxd];

		for(int i=0; i<nprocs; ++i)
		{
			// read n_per_proc integers and print them
			fread(data, sizeof(KEY), dist[i],f);

			cout << "Elements stored on proc " << i << ": " << endl;
			copy(data, data+dist[i], ostream_iterator<KEY>(cout, "\n"));
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
void SpParHelper::FetchMatrix(SpMat<IT,NT,DER> & MRecv, const vector<IT> & essentials, vector<MPI::Win> & arrwin, int ownind)
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
		arrwin[essk++].Get(arrinfo.indarrs[i].addr, arrinfo.indarrs[i].count, MPIType<IT>(), ownind, 0, arrinfo.indarrs[i].count, MPIType<IT>());
	}
	for(int i=0; i< arrinfo.numarrs.size(); ++i)	// get numerical arrays
	{
		//arrwin[essk].Lock(MPI::LOCK_SHARED, ownind, 0);
		arrwin[essk++].Get(arrinfo.numarrs[i].addr, arrinfo.numarrs[i].count, MPIType<NT>(), ownind, 0, arrinfo.numarrs[i].count, MPIType<NT>());
	}
}


/**
  * @param[in] Matrix {For the root processor, the local object to be sent to all others.
  * 		For all others, it is a (yet) empty object to be filled by the received data}
  * @param[in] essentials {irrelevant for the root}
 **/
template<typename IT, typename NT, typename DER>	
void SpParHelper::BCastMatrix(MPI::Intracomm & comm1d, SpMat<IT,NT,DER> & Matrix, const vector<IT> & essentials, int root)
{
	if(comm1d.Get_rank() != root)
	{
		Matrix.Create(essentials);		// allocate memory for arrays		
	}

	Arr<IT,NT> arrinfo = Matrix.GetArrays();
	for(unsigned int i=0; i< arrinfo.indarrs.size(); ++i)	// get index arrays
	{
		comm1d.Bcast(arrinfo.indarrs[i].addr, arrinfo.indarrs[i].count, MPIType<IT>(), root);					
	}
	for(unsigned int i=0; i< arrinfo.numarrs.size(); ++i)	// get numerical arrays
	{
		comm1d.Bcast(arrinfo.numarrs[i].addr, arrinfo.numarrs[i].count, MPIType<NT>(), root);					
	}			
}

template <class IT, class NT, class DER>
void SpParHelper::SetWindows(MPI::Intracomm & comm1d, const SpMat< IT,NT,DER > & Matrix, vector<MPI::Win> & arrwin) 
{	
	Arr<IT,NT> arrs = Matrix.GetArrays(); 
	 
	// static MPI::Win MPI::Win::create(const void *base, MPI::Aint size, int disp_unit, MPI::Info info, const MPI::Intracomm & comm);
	// The displacement unit argument is provided to facilitate address arithmetic in RMA operations
	// *** COLLECTIVE OPERATION ***, everybody exposes its own array to everyone else in the communicator
		
	for(int i=0; i< arrs.indarrs.size(); ++i)
	{
		arrwin.push_back(MPI::Win::Create(arrs.indarrs[i].addr, 
			arrs.indarrs[i].count * sizeof(IT), sizeof(IT), MPI::INFO_NULL, comm1d));
	}
	for(int i=0; i< arrs.numarrs.size(); ++i)
	{
		arrwin.push_back(MPI::Win::Create(arrs.numarrs[i].addr, 
			arrs.numarrs[i].count * sizeof(NT), sizeof(NT), MPI::INFO_NULL, comm1d));
	}	
}

inline void SpParHelper::LockWindows(int ownind, vector<MPI::Win> & arrwin)
{
	for(unsigned int i=0; i< arrwin.size(); ++i)
	{
		arrwin[i].Lock(MPI::LOCK_SHARED, ownind, 0);
	}
}

inline void SpParHelper::UnlockWindows(int ownind, vector<MPI::Win> & arrwin) 
{
	for(unsigned int i=0; i< arrwin.size(); ++i)
	{
		arrwin[i].Unlock(ownind);
	}
}


inline void SpParHelper::SetWinErrHandler(vector<MPI::Win> & arrwin)
{
	for(unsigned int i=0; i< arrwin.size(); ++i)
	{
		arrwin[i].Set_errhandler(MPI::ERRORS_THROW_EXCEPTIONS);
	}
}

/**
 * @param[in] owner {target processor rank within the processor group} 
 * @param[in] arrwin {start access epoch only to owner's arrwin (-windows) }
 */
inline void SpParHelper::StartAccessEpoch(int owner, vector<MPI::Win> & arrwin, MPI::Group & group)
{
	/* Now start using the whole comm as a group */
	int acc_ranks[1]; 
	acc_ranks[0] = owner;
	MPI::Group access = group.Incl(1, acc_ranks);	// take only the owner

	// begin the ACCESS epochs for the arrays of the remote matrices A and B
	// Start() *may* block until all processes in the target group have entered their exposure epoch
	for(unsigned int i=0; i< arrwin.size(); ++i)
		arrwin[i].Start(access, 0); 
	access.Free();
}

/**
 * @param[in] self {rank of "this" processor to be excluded when starting the exposure epoch} 
 */
inline void SpParHelper::PostExposureEpoch(int self, vector<MPI::Win> & arrwin, MPI::Group & group)
{
	MPI::Group exposure = group;
	
	// begin the EXPOSURE epochs for the arrays of the local matrices A and B
	for(unsigned int i=0; i< arrwin.size(); ++i)
		arrwin[i].Post(exposure, MPI_MODE_NOPUT);
	exposure.Free();
}

template <class IT, class DER>
void SpParHelper::AccessNFetch(DER * & Matrix, int owner, vector<MPI::Win> & arrwin, MPI::Group & group, IT ** sizes)
{
	StartAccessEpoch(owner, arrwin, group);	// start the access epoch to arrwin of owner

	vector<IT> ess(DER::esscount);			// pack essentials to a vector
	for(int j=0; j< DER::esscount; ++j)	
		ess[j] = sizes[j][owner];	

	Matrix = new DER();	// create the object first	
	FetchMatrix(*Matrix, ess, arrwin, owner);	// then start fetching its elements
}

template <class IT, class DER>
void SpParHelper::LockNFetch(DER * & Matrix, int owner, vector<MPI::Win> & arrwin, MPI::Group & group, IT ** sizes)
{
	LockWindows(owner, arrwin);

	vector<IT> ess(DER::esscount);			// pack essentials to a vector
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
void SpParHelper::GetSetSizes(const SpMat<IT,NT,DER> & Matrix, IT ** & sizes, MPI::Intracomm & comm1d)
{
	vector<IT> essentials = Matrix.GetEssentials();
	int index = comm1d.Get_rank();
	for(IT i=0; (unsigned)i < essentials.size(); ++i)
	{
		sizes[i][index] = essentials[i]; 
		comm1d.Allgather(MPI::IN_PLACE, 1, MPIType<IT>(), sizes[i], 1, MPIType<IT>());
	}	
}


inline void SpParHelper::Print(const string & s)
{
	int myrank = MPI::COMM_WORLD.Get_rank();
	if(myrank == 0)
	{
		cout << s;
	}
}

inline void SpParHelper::WaitNFree(vector<MPI::Win> & arrwin)
{
	// End the exposure epochs for the arrays of the local matrices A and B
	// The Wait() call matches calls to Complete() issued by ** EACH OF THE ORIGIN PROCESSES ** 
	// that were granted access to the window during this epoch.
	for(unsigned int i=0; i< arrwin.size(); ++i)
	{
		arrwin[i].Wait();
	}
	FreeWindows(arrwin);
}		
	
inline void SpParHelper::FreeWindows(vector<MPI::Win> & arrwin)
{
	for(unsigned int i=0; i< arrwin.size(); ++i)
	{
		arrwin[i].Free();
	}
}

