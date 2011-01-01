#include "FullyDistVec.h"
#include "FullyDistSpVec.h"
#include "Operations.h"

template <class IT, class NT>
FullyDistVec<IT, NT>::FullyDistVec (): zero(0), FullyDist<IT,NT>()
{ }

template <class IT, class NT>
FullyDistVec<IT, NT>::FullyDistVec (NT id): zero(id), FullyDist<IT,NT>()
{ }

template <class IT, class NT>
FullyDistVec<IT, NT>::FullyDistVec (IT globallen, NT initval, NT id): zero(id), FullyDist<IT,NT>(globallen)
{
	arr.resize(MyLocLength(), initval);
}

template <class IT, class NT>
FullyDistVec<IT, NT>::FullyDistVec ( shared_ptr<CommGrid> grid, NT id): zero(id), FullyDist<IT,NT>(grid)
{ }

template <class IT, class NT>
FullyDistVec<IT, NT>::FullyDistVec ( shared_ptr<CommGrid> grid, IT globallen, NT initval, NT id)
: zero(id), FullyDist<IT,NT>(grid,globallen)
{
	arr.resize(MyLocLength(), initval);
}

template <class IT, class NT>
template <typename _BinaryOperation>
NT FullyDistVec<IT,NT>::Reduce(_BinaryOperation __binary_op, NT identity)
{
	// std::accumulate returns identity for empty sequences
	NT localsum = std::accumulate( arr.begin(), arr.end(), identity, __binary_op);

	NT totalsum = identity;
	(commGrid->GetWorld()).Allreduce( &localsum, &totalsum, 1, MPIType<NT>(), MPIOp<_BinaryOperation, NT>::op());
	return totalsum;
}

template <class IT, class NT>
FullyDistVec< IT,NT > &  FullyDistVec<IT,NT>::operator=(const FullyDistSpVec< IT,NT > & rhs)		// FullyDistSpVec->FullyDistVec conversion operator
{
	arr.resize(rhs.MyLocLength());
	std::fill(arr.begin(), arr.end(), zero);	

	IT spvecsize = rhs.getlocnnz();
	for(IT i=0; i< spvecsize; ++i)
	{
		arr[rhs.ind[i]] = rhs.num[i];
	}
	
	return *this;
}


template <class IT, class NT>
FullyDistVec< IT,NT > &  FullyDistVec<IT,NT>::operator=(const DenseParVec< IT,NT > & rhs)		// DenseParVec->FullyDistVec conversion operator
{
	if(*commGrid != *rhs.commGrid) 		
	{
		cout << "Grids are not comparable elementwise addition" << endl; 
		MPI::COMM_WORLD.Abort(GRIDMISMATCH);
	}
	else
	{
		zero = rhs.zero;
		glen = rhs.getTotalLength();
		arr.resize(MyLocLength());	// once glen is set, MyLocLength() works
		fill(arr.begin(), arr.end(), zero);	

		int * sendcnts;
		int * dpls;
		if(rhs.diagonal)
		{
			int proccols = commGrid->GetGridCols();	
        		IT n_perproc = rhs.getLocalLength() / proccols;
			sendcnts = new int[proccols];
			fill(sendcnts, sendcnts+proccols-1, n_perproc);
			sendcnts[proccols-1] = rhs.getLocalLength() - (n_perproc * (proccols-1));
			dpls = new int[proccols]();	// displacements (zero initialized pid) 
			partial_sum(sendcnts, sendcnts+proccols-1, dpls+1);
		}

		int rowroot = commGrid->GetDiagOfProcRow();
		(commGrid->GetRowWorld()).Scatterv(&(rhs.arr[0]),sendcnts, dpls, MPIType<NT>(), &(arr[0]), arr.size(), MPIType<NT>(),rowroot);
	}
	return *this;
}


// Let the compiler create an assignment operator and call base class' 
// assignment operator automatically

template <class IT, class NT>
FullyDistVec< IT,NT > &  FullyDistVec<IT,NT>::stealFrom(FullyDistVec<IT,NT> & victim)
{
	FullyDist<IT,NT>::operator= (victim);	// to update glen and commGrid
	arr.swap(victim.arr);
	zero = victim.zero;
	return *this;
}

template <class IT, class NT>
FullyDistVec< IT,NT > &  FullyDistVec<IT,NT>::operator+=(const FullyDistSpVec< IT,NT > & rhs)		
{
	IT spvecsize = rhs.getlocnnz();
	for(IT i=0; i< spvecsize; ++i)
	{
		if(arr[rhs.ind[i]] == zero) // not set before
			arr[rhs.ind[i]] = rhs.num[i];
		else
			arr[rhs.ind[i]] += rhs.num[i];
	}
	return *this;
}

template <class IT, class NT>
FullyDistVec< IT,NT > &  FullyDistVec<IT,NT>::operator-=(const FullyDistSpVec< IT,NT > & rhs)		
{
	IT spvecsize = rhs.getlocnnz();
	for(IT i=0; i< spvecsize; ++i)
	{
		arr[rhs.ind[i]] -= rhs.num[i];
	}
}


/**
  * Perform __binary_op(*this, v2) for every element in rhs, *this, 
  * which are of the same size. and write the result back to *this
  */ 
template <class IT, class NT>
template <typename _BinaryOperation>	
void FullyDistVec<IT,NT>::EWise(const FullyDistVec<IT,NT> & rhs,  _BinaryOperation __binary_op)
{
	if(zero == rhs.zero)
	{
		transform ( arr.begin(), arr.end(), rhs.arr.begin(), arr.begin(), __binary_op );
	}
	else
	{
		cout << "FullyDistVec objects have different identity (zero) elements..." << endl;
		cout << "Operation didn't happen !" << endl;
	}
};


template <class IT, class NT>
FullyDistVec<IT,NT> & FullyDistVec<IT, NT>::operator+=(const FullyDistVec<IT,NT> & rhs)
{
	if(this != &rhs)		
	{	
		if(!(*commGrid == *rhs.commGrid)) 		
		{
			cout << "Grids are not comparable elementwise addition" << endl; 
			MPI::COMM_WORLD.Abort(GRIDMISMATCH);
		}
		else 
		{
			EWise(rhs, std::plus<NT>());
		} 	
	}	
	return *this;
};

template <class IT, class NT>
FullyDistVec<IT,NT> & FullyDistVec<IT, NT>::operator-=(const FullyDistVec<IT,NT> & rhs)
{
	if(this != &rhs)		
	{	
		if(!(*commGrid == *rhs.commGrid)) 		
		{
			cout << "Grids are not comparable elementwise addition" << endl; 
			MPI::COMM_WORLD.Abort(GRIDMISMATCH);
		}
		else 
		{
			EWise(rhs, std::minus<NT>());
		} 	
	}	
	return *this;
};		

template <class IT, class NT>
bool FullyDistVec<IT,NT>::operator==(const FullyDistVec<IT,NT> & rhs) const
{
	ErrorTolerantEqual<NT> epsilonequal;
	int local = 1;
	local = (int) std::equal(arr.begin(), arr.end(), rhs.arr.begin(), epsilonequal );
	int whole = 1;
	commGrid->GetWorld().Allreduce( &local, &whole, 1, MPI::INT, MPI::BAND);
	return static_cast<bool>(whole);	
}

template <class IT, class NT>
template <typename _Predicate>
IT FullyDistVec<IT,NT>::Count(_Predicate pred) const
{
	IT local = count_if( arr.begin(), arr.end(), pred );
	IT whole = 0;
	commGrid->GetWorld().Allreduce( &local, &whole, 1, MPIType<IT>(), MPI::SUM);
	return whole;	
}

//! Returns a dense vector of global indices 
//! for which the predicate is satisfied
template <class IT, class NT>
template <typename _Predicate>
FullyDistVec<IT,IT> FullyDistVec<IT,NT>::FindInds(_Predicate pred) const
{
	FullyDistVec<IT,IT> found(commGrid, (IT) 0);
	MPI::Intracomm World = commGrid->GetWorld();
	int nprocs = commGrid->GetSize();
	int rank = commGrid->GetRank();

	IT sizelocal = LocArrSize();
	IT sizesofar = LengthUntil();
	for(IT i=0; i<sizelocal; ++i)
	{
		if(pred(arr[i]))
		{
			found.arr.push_back(i+sizesofar);
		}
	}
	IT * dist = new IT[nprocs];
	IT nsize = found.arr.size(); 
	dist[rank] = nsize;
	World.Allgather(MPI::IN_PLACE, 1, MPIType<IT>(), dist, 1, MPIType<IT>());
	IT lengthuntil = accumulate(dist, dist+rank, 0);
	found.glen = accumulate(dist, dist+nprocs, 0);

	// Although the found vector is not reshuffled yet, its glen and commGrid are set
	// We can call the Owner/MyLocLength/LengthUntil functions (to infer future distribution)

	// rebalance/redistribute
	int * sendcnt = new int[nprocs];
	fill(sendcnt, sendcnt+nprocs, 0);
	for(IT i=0; i<nsize; ++i)
	{
		IT locind;
		int owner = found.Owner(i+lengthuntil, locind);	
		++sendcnt[owner];
	}
	int * recvcnt = new int[nprocs];
	World.Alltoall(sendcnt, 1, MPI::INT, recvcnt, 1, MPI::INT); // share the counts 

	int * sdispls = new int[nprocs];
	int * rdispls = new int[nprocs];
	sdispls[0] = 0;
	rdispls[0] = 0;
	for(int i=0; i<nprocs-1; ++i)
	{
		sdispls[i+1] = sdispls[i] + sendcnt[i];
		rdispls[i+1] = rdispls[i] + recvcnt[i];
	}
	IT totrecv = accumulate(recvcnt,recvcnt+nprocs, (IT) 0);
	vector<IT> recvbuf(totrecv);
			
	// data is already in the right order in found.arr
	World.Alltoallv(&(found.arr[0]), sendcnt, sdispls, MPIType<IT>(), &(recvbuf[0]), recvcnt, rdispls, MPIType<IT>()); 
	found.arr.swap(recvbuf);
	DeleteAll(sendcnt, recvcnt, sdispls, rdispls);

	return found;
}


//! Requires no communication because FullyDistSpVec (the return object)
//! is distributed based on length, not nonzero counts
template <class IT, class NT>
template <typename _Predicate>
FullyDistSpVec<IT,NT> FullyDistVec<IT,NT>::Find(_Predicate pred) const
{
	FullyDistSpVec<IT,NT> found(commGrid);
	IT size = arr.size();
	for(IT i=0; i<size; ++i)
	{
		if(pred(arr[i]))
		{
			found.ind.push_back(i);
			found.num.push_back(arr[i]);
		}
	}
	found.glen = glen;
	return found;	
}

template <class IT, class NT>
ifstream& FullyDistVec<IT,NT>::ReadDistribute (ifstream& infile, int master)
{
	FullyDistSpVec<IT,NT> tmpSpVec(commGrid);
	tmpSpVec.ReadDistribute(infile, master);

	*this = tmpSpVec;
	return infile;
}

template <class IT, class NT>
void FullyDistVec<IT,NT>::SetElement (IT indx, NT numx)
{
	int rank = commGrid->GetRank();
	if (glen == 0) 
	{
		if(rank == 0)
			cout << "FullyDistVec::SetElement can't be called on an empty vector." << endl;
		return;
	}
	IT locind;
	int owner = Owner(indx, locind);
	if(commGrid->GetRank() == owner)
	{
		if (locind > (LocArrSize() -1))
		{
			cout << "FullyDistVec::SetElement cannot expand array" << endl;
		}
		else if (locind < 0)
		{
			cout << "FullyDistVec::SetElement local index < 0" << endl;
		}
		else
		{
			arr[locind] = numx;
		}
	}
}

template <class IT, class NT>
NT FullyDistVec<IT,NT>::GetElement (IT indx) const
{
	NT ret;
	MPI::Intracomm World = commGrid->GetWorld();
	int rank = commGrid->GetRank();
	if (glen == 0) 
	{
		if(rank == 0)
			cout << "FullyDistVec::GetElement can't be called on an empty vector." << endl;

		return numeric_limits<NT>::min();
	}
	IT locind;
	int owner = Owner(indx, locind);
	if(commGrid->GetRank() == owner)
	{
		if (locind > (LocArrSize() -1))
		{
			cout << "FullyDistVec::GetElement local index > size" << endl;
			ret = numeric_limits<NT>::min();

		}
		else if (locind < 0)
		{
			cout << "FullyDistVec::GetElement local index < 0" << endl;
			ret = numeric_limits<NT>::min();
		}
		else
		{
			ret = arr[locind];
		}
	}
	World.Bcast(&ret, 1, MPIType<NT>(), owner);
	return ret;
}

// Write to file using MPI-2
template <class IT, class NT>
void FullyDistVec<IT,NT>::DebugPrint()
{
	MPI::Intracomm World = commGrid->GetWorld();
    	int rank = World.Get_rank();
    	int nprocs = World.Get_size();
    	MPI::File thefile = MPI::File::Open(World, "temp_fullydistvec", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI::INFO_NULL);    
	IT lengthuntil = LengthUntil();

	// The disp displacement argument specifies the position 
	// (absolute offset in bytes from the beginning of the file) 
    	thefile.Set_view(lengthuntil * sizeof(NT), MPIType<NT>(), MPIType<NT>(), "native", MPI::INFO_NULL);

	IT count = LocArrSize();	
	thefile.Write(&(arr[0]), count, MPIType<NT>());
	thefile.Close();
	
	// Now let processor-0 read the file and print
	IT * counts = new IT[nprocs];
	World.Gather(&count, 1, MPIType<IT>(), counts, 1, MPIType<IT>(), 0);	// gather at root=0
	if(rank == 0)
	{
		FILE * f = fopen("temp_fullydistvec", "r");
                if(!f)
                {
                        cerr << "Problem reading binary input file\n";
                        return;
                }
		IT maxd = *max_element(counts, counts+nprocs);
		NT * data = new NT[maxd];

		for(int i=0; i<nprocs; ++i)
		{
			// read counts[i] integers and print them
			fread(data, sizeof(NT), counts[i],f);

			cout << "Elements stored on proc " << i << ": {" ;	
			for (int j = 0; j < counts[i]; j++)
			{
				cout << data[j] << ",";
			}
			cout << "}" << endl;
		}
		delete [] data;
		delete [] counts;
	}
}

template <class IT, class NT>
template <typename _UnaryOperation>
void FullyDistVec<IT,NT>::Apply(_UnaryOperation __unary_op, const FullyDistSpVec<IT,NT> & mask)
{
	typename vector< IT >::const_iterator miter = mask.ind.begin();
	while (miter < mask.ind.end())
	{
		IT index = *miter++;
		arr[index] = __unary_op(arr[index]);
	}
}	

// Randomly permutes an already existing vector
template <class IT, class NT>
void FullyDistVec<IT,NT>::RandPerm()
{
	MPI::Intracomm World = commGrid->GetWorld();
	long size = (long) LocArrSize();
	pair<double,IT> * vecpair = new pair<double,IT>[size];

	int nprocs = World.Get_size();
	int rank = World.Get_rank();

	long * dist = new long[nprocs];
	dist[rank] = size;
	World.Allgather(MPI::IN_PLACE, 1, MPIType<long>(), dist, 1, MPIType<long>());
	IT lengthuntil = accumulate(dist, dist+rank, 0);

  	MTRand M;	// generate random numbers with Mersenne Twister
	for(int i=0; i<size; ++i)
	{
		vecpair[i].first = M.rand();
		vecpair[i].second = arr[i];	
	}

	// less< pair<T1,T2> > works correctly (sorts wrt first elements)	
	// SpParHelper::MemoryEfficientPSort(pair<KEY,VAL> * array, IT length, IT * dist, MPI::Intracomm & comm)
	SpParHelper::MemoryEfficientPSort(vecpair, size, dist, World);

	vector< NT > nnum(size);
	for(int i=0; i<size; ++i)
		nnum[i] = vecpair[i].second;

	delete [] vecpair;
	delete [] dist;

	arr.swap(nnum);
}

template <class IT, class NT>
void FullyDistVec<IT,NT>::iota(IT globalsize, NT first)
{
	glen = globalsize;
	IT length = MyLocLength();	// only needs glen to determine length
	arr.resize(length);
	SpHelper::iota(arr.begin(), arr.end(), LengthUntil() + first);	// global across processors
}

template <class IT, class NT>
FullyDistVec<IT,NT> FullyDistVec<IT,NT>::operator() (const FullyDistVec<IT,IT> & ri) const
{
	if(!(*commGrid == *ri.commGrid))
	{
		cout << "Grids are not comparable for dense vector subsref" << endl;
		return FullyDistVec<IT,NT>();
	}

	MPI::Intracomm World = commGrid->GetWorld();
	FullyDistVec<IT,NT> Indexed(commGrid, ri.glen, ri.zero, ri.zero);	// length(Indexed) = length(ri)
	int rank = World.Get_rank();
	int nprocs = World.Get_size();
	vector< vector< IT > > data_req(nprocs);	
	vector< vector< IT > > revr_map(nprocs);	// to put the incoming data to the correct location	

	IT riloclen = ri.LocArrSize();
	for(IT i=0; i < riloclen; ++i)
	{
		IT locind;
		int owner = Owner(ri.arr[i], locind);	// numerical values in ri are 0-based
		data_req[owner].push_back(locind);
		revr_map[owner].push_back(i);
	}
	IT * sendbuf = new IT[riloclen];
	int * sendcnt = new int[nprocs];
	int * sdispls = new int[nprocs];
	for(int i=0; i<nprocs; ++i)
		sendcnt[i] = data_req[i].size();

	int * rdispls = new int[nprocs];
	int * recvcnt = new int[nprocs];
	World.Alltoall(sendcnt, 1, MPI::INT, recvcnt, 1, MPI::INT);	// share the request counts 

	sdispls[0] = 0;
	rdispls[0] = 0;
	for(int i=0; i<nprocs-1; ++i)
	{
		sdispls[i+1] = sdispls[i] + sendcnt[i];
		rdispls[i+1] = rdispls[i] + recvcnt[i];
	}
	IT totrecv = accumulate(recvcnt,recvcnt+nprocs,zero);
	IT * recvbuf = new IT[totrecv];

	for(int i=0; i<nprocs; ++i)
	{
		copy(data_req[i].begin(), data_req[i].end(), sendbuf+sdispls[i]);
		vector<IT>().swap(data_req[i]);
	}

	IT * reversemap = new IT[riloclen];
	for(int i=0; i<nprocs; ++i)
	{
		copy(revr_map[i].begin(), revr_map[i].end(), reversemap+sdispls[i]);	// reversemap array is unique
		vector<IT>().swap(revr_map[i]);
	}
	World.Alltoallv(sendbuf, sendcnt, sdispls, MPIType<IT>(), recvbuf, recvcnt, rdispls, MPIType<IT>());  // request data
		
	// We will return the requested data,
	// our return will be as big as the request 
	// as we are indexing a dense vector, all elements exist
	// so the displacement boundaries are the same as rdispls
	NT * databack = new NT[totrecv];		
	for(int i=0; i<nprocs; ++i)
	{
		for(int j = rdispls[i]; j < rdispls[i] + recvcnt[i]; ++j)	// fetch the numerical values
		{
			databack[j] = arr[recvbuf[j]];
		}
	}
	delete [] recvbuf;
	NT * databuf = new NT[riloclen];

	// the response counts are the same as the request counts 
	World.Alltoallv(databack, recvcnt, rdispls, MPIType<NT>(), databuf, sendcnt, sdispls, MPIType<NT>());  // send data
	DeleteAll(rdispls, recvcnt, databack);

	// Now create the output from databuf
	// Indexed.arr is already allocated in contructor
	for(int i=0; i<nprocs; ++i)
	{
		for(int j=sdispls[i]; j< sdispls[i]+sendcnt[i]; ++j)
		{
			Indexed.arr[reversemap[j]] = databuf[j];
		}
	}
	DeleteAll(sdispls, sendcnt, databuf,reversemap);
	return Indexed;
}

template <class IT, class NT>
void FullyDistVec<IT,NT>::PrintInfo(string vectorname) const
{
	IT totl = TotalLength();
	if (commGrid->GetRank() == 0)		
		cout << "As a whole, " << vectorname << " has length " << totl << endl; 
}
