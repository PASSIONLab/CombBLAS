/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.1 -------------------------------------------------*/
/* date: 12/25/2010 --------------------------------------------*/
/* authors: Aydin Buluc (abuluc@lbl.gov), Adam Lugowski --------*/
/****************************************************************/

template<typename KEY, typename VAL, typename IT>
void SpParHelper::MemoryEfficientPSort(pair<KEY,VAL> * array, IT length, IT * dist, MPI::Intracomm & comm)
{	
	int nprocs = comm.Get_size();
	int nsize = nprocs / 2;	// new size
	if(nprocs < 50)
	{
		psort::parallel_sort (array, array+length,  dist, comm);
	}
	else
	{
		IT gl_median = accumulate(dist, dist+nsize, 0);	// global rank of the first element of the median processor

//		ostringstream out;
//		out << "Rank of first element in median processor: " << gl_median << endl;
//		SpParHelper::Print(out.str());
		sort(array, array+length);	// re-sort because we might have swapped data in previous iterations

		int myrank = comm.Get_rank();
		int color = (myrank < nsize)? 0: 1;
		
		pair<KEY,VAL> * low = array;
		pair<KEY,VAL> * upp = array;
		GlobalSelect(gl_median, low, upp, array, length, comm);
		BipartiteSwap(low, array, length, nsize, color, comm);

		if(color == 1)	dist = dist + nsize;	// adjust for the second half of processors

		// recursive call; two implicit 'spawn's where half of the processors execute different paramaters
		MPI::Intracomm halfcomm = comm.Split(color, myrank);	// split into two communicators
		MemoryEfficientPSort(array, length, dist, halfcomm);
	}
}

template<typename KEY, typename VAL, typename IT>
void SpParHelper::GlobalSelect(IT gl_rank, pair<KEY,VAL> * & low,  pair<KEY,VAL> * & upp, pair<KEY,VAL> * array, IT length, MPI::Intracomm & comm)
{
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
	int active = end-begin;				// size of the active range
	int nacts = 0; 
	bool found = 0;
	do
	{
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
//		ostringstream out;
//		out << "Weighted median of medians:" << wmm << endl;
//		SpParHelper::Print(out.str());

		pair<KEY,VAL> wmmpair = make_pair(wmm, VAL());
		low =lower_bound (array+begin, array+end, wmmpair); 
		upp =upper_bound (array+begin, array+end, wmmpair); 
		IT loc_low = low-array;	// #{elements smaller than wmm}
		IT loc_upp = upp-array;	// #{elements smaller or equal to wmm}
		comm.Allreduce( &loc_low, &gl_low, 1, MPIType<IT>(), MPI::SUM);
		comm.Allreduce( &loc_upp, &gl_upp, 1, MPIType<IT>(), MPI::SUM);
//		out.clear();
//		out.str("");
//		out << "GL_LOW: " << gl_low << ", GL_UPP: " << gl_upp << ", GL_RANK: " << gl_rank << endl;
//		SpParHelper::Print(out.str());

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
		comm.Allreduce(&active, &nacts, 1, MPI::INT, MPI::SUM);

//		out.clear();
//		out.str("");
//		out << "Total actives: "<< nacts << endl;
//		SpParHelper::Print(out.str());
	} 
	while((nacts > 2*nprocs) && (!found));
	delete [] wmminput;

    	MPI_Datatype MPI_pairType;
	MPI_Type_contiguous (sizeof(pair<KEY,VAL>), MPI_CHAR, &MPI_pairType);
	MPI_Type_commit (&MPI_pairType);

	int * nactives = new int[nprocs];
	nactives[myrank] = active;
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
			allactives[k++] = make_pair(recvbuf[k].first, i);
		}
	}
	DeleteAll(recvbuf, dpls, nactives);
	sort(allactives, allactives+nacts); 
	comm.Allreduce(&begin, &gl_low, 1, MPIType<IT>(), MPI::SUM);	// update
//	ostringstream out;
//	out << "GL_LOW: " << gl_low << ", GL_RANK: " << gl_rank << endl;
	for(int k=gl_low; k < gl_rank; ++k)
	{		
		if(allactives[k-gl_low].second == myrank)	
			++low;	// increment the local pointer
	}
	delete [] allactives;

//	begin = low-array;
//	comm.Allreduce(&begin, &gl_low, 1, MPIType<IT>(), MPI::SUM); 	// update
//	out << "GL_LOW: " << gl_low << ", GL_RANK: " << gl_rank << endl;
//	SpParHelper::Print(out.str());
}

template<typename KEY, typename VAL, typename IT>
void SpParHelper::BipartiteSwap(pair<KEY,VAL> * low, pair<KEY,VAL> * array, IT length, int nfirsthalf, int color, MPI::Intracomm & comm)
{
	int nprocs = comm.Get_size();
	int myrank = comm.Get_rank();
	IT * firsthalves = new IT[nprocs];
	IT * secondhalves = new IT[nprocs];	
	firsthalves[myrank] = low-array;
	secondhalves[myrank] = length - (low-array);
	comm.Allgather(MPI::IN_PLACE, 0, MPIType<IT>(), firsthalves, 1, MPIType<IT>());
	comm.Allgather(MPI::IN_PLACE, 0, MPIType<IT>(), secondhalves, 1, MPIType<IT>());
	
	//if(myrank == 0)
	//{
	//	copy(firsthalves, firsthalves+nprocs, ostream_iterator<IT>(cout, " ")); cout << endl;
	//	copy(secondhalves, secondhalves+nprocs, ostream_iterator<IT>(cout, " ")); cout << endl;
	//}

	int * sendcnt = new int[nprocs]();	// zero initialize
	vector< tuple<int,IT,IT>  > package;	// recipient, begin index, length
	int totrecvcnt; 
	IT spacebefore = 0;	// receiving part space
	if(color == 0)	// first processor half, only send second half of data
	{
		totrecvcnt = length - (low-array);
		IT beg_oftransfer = accumulate(secondhalves, secondhalves+myrank, 0);
		IT spaceafter = spacebefore+firsthalves[nfirsthalf];
		int i=nfirsthalf;
		while(i < nprocs && spaceafter < beg_oftransfer)
		{
			spacebefore = spaceafter;
			spaceafter += firsthalves[++i];	// pre-incremenet
		}
		IT end_oftransfer = beg_oftransfer + secondhalves[myrank];	// global index (within second half) of the end of my data
		IT beg_pour = beg_oftransfer;
		IT end_pour = min(end_oftransfer, spaceafter);
		package.push_back(make_tuple(i, beg_pour-beg_oftransfer, end_pour-beg_pour));	// (recipient, begin, length)
		sendcnt[i] = end_pour - beg_pour;
		while( i < nprocs && spaceafter < end_oftransfer )	// find other recipients until I run out of data
		{
			beg_pour = end_pour;
			spaceafter += firsthalves[++i];
			end_pour = min(end_oftransfer, spaceafter);
			package.push_back(make_tuple(i, beg_pour-beg_oftransfer, end_pour-beg_pour));
			sendcnt[i] = end_pour - beg_pour;
		}
	}
	else if(color == 1)	// second processor half, only send first half of data
	{
		totrecvcnt = low-array;
		// global index (within the second processor half) of the beginning of my data
		IT beg_oftransfer = accumulate(firsthalves+nfirsthalf, firsthalves+myrank, 0);
		IT spaceafter = spacebefore+secondhalves[0];
		int i=0;
		while( i< nfirsthalf && spaceafter < beg_oftransfer)
		{
			spacebefore = spaceafter;
			spaceafter += secondhalves[++i];	// pre-incremenet
		}
		IT end_oftransfer = beg_oftransfer + firsthalves[myrank];	// global index (within second half) of the end of my data
		IT beg_pour = beg_oftransfer;
		IT end_pour = min(end_oftransfer, spaceafter);
		package.push_back(make_tuple(i, beg_pour-beg_oftransfer, end_pour-beg_pour));	// (recipient, begin, length)
		sendcnt[i] = end_pour - beg_pour;
		while( i < nfirsthalf && spaceafter < end_oftransfer )	// find other recipients until I run out of data
		{
			beg_pour = end_pour;
			spaceafter += secondhalves[++i];
			end_pour = min(end_oftransfer, spaceafter);
			package.push_back(make_tuple(i, beg_pour-beg_oftransfer, end_pour-beg_pour));
			sendcnt[i] = end_pour - beg_pour;
		}
	}
	DeleteAll(firsthalves, secondhalves);
	int * recvcnt = new int[nprocs];
	comm.Alltoall(sendcnt, 1, MPI::INT, recvcnt, 1, MPI::INT);	// get the recv counts

    	MPI_Datatype MPI_valueType;
	MPI_Type_contiguous (sizeof(pair<KEY,VAL>), MPI_CHAR, &MPI_valueType);
	MPI_Type_commit (&MPI_valueType);

	vector< MPI::Request > requests;
	pair<KEY,VAL> * receives = new pair<KEY,VAL>[totrecvcnt];
	int recvsofar = 0;
	for (int i=0; i< nprocs; ++i)
	{
		if(recvcnt[i] > 0)
		{
			MPI::Request req = comm.Irecv(receives + recvsofar, recvcnt[i], MPI_valueType, i, SWAPTAG);
			requests.push_back(req);
			recvsofar += recvcnt[i];
		}
	}
	int sentsofar = 0;
	for(int i=0; i< package.size(); ++i)
	{
		int recipient = tr1::get<0>(package[i]);
		IT beg = tr1::get<1>(package[i]);
		IT len = tr1::get<2>(package[i]);
		MPI::Request req = comm.Isend(array+beg, len, MPI_valueType, recipient, SWAPTAG);
		requests.push_back(req);
		sentsofar += len;
	}
	MPI::Status * status = new MPI::Status[requests.size()];
	try
	{
		MPI::Request::Waitall(requests.size(), &(requests[0]), status);
	}
	catch ( MPI::Exception failure)
	{
		int error = failure.Get_error_code();
		if ( error == MPI::ERR_IN_STATUS )
		{
			for(int i=0; i<requests.size(); ++i)
			{
				if(status[i].Get_error() != MPI::SUCCESS)
				{
					cout << "Error @processor " << myrank<< ": " << status[i].Get_error() << endl;
				}
			}
		}
		else
		{
			cout << "Error @processor " << myrank<< ": " << error << endl;;
		}
	}
	
	DeleteAll(sendcnt, recvcnt, status);
	assert(sentsofar == recvsofar);
	if(color == 0)	array = low;	// no effect on the calling 'array' as the pointer was passed-by-value
	copy(receives, receives+totrecvcnt, array);
	delete [] receives;
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
	for(int i=0; i< arrinfo.indarrs.size(); ++i)	// get index arrays
	{
		comm1d.Bcast(arrinfo.indarrs[i].addr, arrinfo.indarrs[i].count, MPIType<IT>(), root);					
	}
	for(int i=0; i< arrinfo.numarrs.size(); ++i)	// get numerical arrays
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
	for(int i=0; i< arrwin.size(); ++i)
	{
		arrwin[i].Lock(MPI::LOCK_SHARED, ownind, 0);
	}
}

inline void SpParHelper::UnlockWindows(int ownind, vector<MPI::Win> & arrwin) 
{
	for(int i=0; i< arrwin.size(); ++i)
	{
		arrwin[i].Unlock(ownind);
	}
}


inline void SpParHelper::SetWinErrHandler(vector<MPI::Win> & arrwin)
{
	for(int i=0; i< arrwin.size(); ++i)
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
	for(int i=0; i< arrwin.size(); ++i)
		arrwin[i].Start(access, 0); 
}

/**
 * @param[in] self {rank of "this" processor to be excluded when starting the exposure epoch} 
 */
inline void SpParHelper::PostExposureEpoch(int self, vector<MPI::Win> & arrwin, MPI::Group & group)
{
	MPI::Group exposure = group;
	
	// begin the EXPOSURE epochs for the arrays of the local matrices A and B
	for(int i=0; i< arrwin.size(); ++i)
		arrwin[i].Post(exposure, MPI_MODE_NOPUT);
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
	for(IT i=0; i< essentials.size(); ++i)
	{
		sizes[i][index] = essentials[i]; 
		comm1d.Allgather(MPI::IN_PLACE, 1, MPIType<IT>(), sizes[i], 1, MPIType<IT>());
	}	
}


void SpParHelper::Print(const string & s)
{
	int myrank = MPI::COMM_WORLD.Get_rank();
	if(myrank == 0)
	{
		cout << s;
	}
}

void SpParHelper::WaitNFree(vector<MPI::Win> & arrwin)
{
	// End the exposure epochs for the arrays of the local matrices A and B
	// The Wait() call matches calls to Complete() issued by ** EACH OF THE ORIGIN PROCESSES ** 
	// that were granted access to the window during this epoch.
	for(int i=0; i< arrwin.size(); ++i)
	{
		arrwin[i].Wait();
	}
	FreeWindows(arrwin);
}		
	
void SpParHelper::FreeWindows(vector<MPI::Win> & arrwin)
{
	for(int i=0; i< arrwin.size(); ++i)
	{
		arrwin[i].Free();
	}
}

