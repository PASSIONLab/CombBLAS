#include "FullyDistVec.h"
#include "FullyDistSpVec.h"
#include "Operations.h"

template <class IT, class NT>
FullyDistVec<IT, NT>::FullyDistVec ()
{
	zero = static_cast<NT>(0);
	commGrid.reset(new CommGrid(MPI::COMM_WORLD, 0, 0));
}

template <class IT, class NT>
FullyDistVec<IT, NT>::FullyDistVec (NT id): zero(id)
{
	commGrid.reset(new CommGrid(MPI::COMM_WORLD, 0, 0));
}

template <class IT, class NT>
FullyDistVec<IT, NT>::FullyDistVec (IT locallength, NT initval, NT id): zero(id)
{
	commGrid.reset(new CommGrid(MPI::COMM_WORLD, 0, 0));
	arr.resize(locallength, initval);
}

template <class IT, class NT>
FullyDistVec<IT, NT>::FullyDistVec ( shared_ptr<CommGrid> grid, NT id): zero(id)
{
	commGrid.reset(new CommGrid(*grid));		
};

template <class IT, class NT>
FullyDistVec<IT, NT>::FullyDistVec ( shared_ptr<CommGrid> grid, IT locallength, NT initval, NT id): commGrid(grid), zero(id)
{
	arr.resize(locallength, initval);
};

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
	arr.resize(rhs.length);
	std::fill(arr.begin(), arr.end(), zero);	

	IT spvecsize = rhs.ind.size();
	for(IT i=0; i< spvecsize; ++i)
	{
		arr[rhs.ind[i]] = rhs.num[i];
	}
	
	return *this;
}

template <class IT, class NT>
FullyDistVec< IT,NT > &  FullyDistVec<IT,NT>::operator=(const FullyDistVec< IT,NT > & rhs)	
{
	if (this == &rhs)      	// Same object?
      		return *this;   // Yes, so skip assignment, and just return *this.
	commGrid.reset(new CommGrid(*(rhs.commGrid)));		
	arr = rhs.arr;
	zero = rhs.zero;
	return *this;
}

template <class IT, class NT>
FullyDistVec< IT,NT > &  FullyDistVec<IT,NT>::stealFrom(FullyDistVec<IT,NT> & victim)		// FullyDistSpVec->FullyDistVec conversion operator
{
	commGrid.reset(new CommGrid(*(victim.commGrid)));		
	arr.swap(victim.arr);
	zero = victim.zero;
	return *this;
}

template <class IT, class NT>
FullyDistVec< IT,NT > &  FullyDistVec<IT,NT>::operator+=(const FullyDistSpVec< IT,NT > & rhs)		
{
	IT spvecsize = rhs.ind.size();
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
	IT spvecsize = rhs.ind.size();
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
	int rank = World.Get_rank();
	int nprocs = World.Get_size();
	IT old_n_perproc = getTypicalLocLength();
		
	IT size = arr.size();
	for(IT i=0; i<size; ++i)
	{
		if(pred(arr[i]))
		{
			found.arr.push_back(i+old_n_perproc*rank);
		}
	}
	World.Barrier();

	// Since the found vector is not reshuffled yet, we can't use getTypicalLocLength() at this point
	IT n_perproc = found.getTotalLength(World) / nprocs;
	if(n_perproc == 0)	// it has less than p elements, all owned by the last processor
	{
		if(rank != nprocs-1)
		{
			int arrsize = found.arr.size();
			World.Gather(&arrsize, 1, MPI::INT, NULL, 1, MPI::INT, nprocs-1);
			World.Gatherv(&(found.arr[0]), arrsize, MPIType<IT>(), NULL, NULL, NULL, MPIType<IT>(), nprocs-1);
		}
		else
		{	
			int * allnnzs = new int[nprocs];
			allnnzs[rank] = found.arr.size();
			World.Gather(MPI::IN_PLACE, 1, MPI::INT, allnnzs, 1, MPI::INT, nprocs-1);
				
			int * rdispls = new int[nprocs];
			rdispls[0] = 0;
			for(int i=0; i<nprocs-1; ++i)
				rdispls[i+1] = rdispls[i] + allnnzs[i];

			IT totrecv = accumulate(allnnzs, allnnzs+nprocs, 0);
			vector<IT> recvbuf(totrecv);
			World.Gatherv(MPI::IN_PLACE, 1, MPI::INT, &(recvbuf[0]), allnnzs, rdispls, MPIType<IT>(), nprocs-1);

			found.arr.swap(recvbuf);
			DeleteAll(allnnzs, rdispls);
		}
		return found;		// don't execute further
	}

	IT lengthuntil = rank * n_perproc;

	// rebalance/redistribute
	IT nsize = found.arr.size();
	int * sendcnt = new int[nprocs];
	fill(sendcnt, sendcnt+nprocs, 0);
	for(IT i=0; i<nsize; ++i)
	{
		// owner id's are monotonically increasing and continuous
		int owner = std::min(static_cast<int>( (i+lengthuntil) / n_perproc), nprocs-1); 
		sendcnt[owner]++;
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
	MPI::Intracomm World = commGrid->GetWorld();
	int rank = World.Get_rank();
	int nprocs = World.Get_size();
	IT n_perproc = getTypicalLocLength();	
	IT offset = rank * n_perproc;
		
	if (n_perproc == 0) 
	{
		cout << "FullyDistVec::SetElement can't be called on an empty vector." << endl;
		return;
	}
	IT owner = (indx) / n_perproc;	
	IT rec = (owner < nprocs-1) ? owner : nprocs-1;	// find its owner 
	if(rec == rank) // this process is the owner
	{
		IT locindx = indx-offset;	
		if (locindx > arr.size()-1)
		{
			cout << "FullyDistVec::SetElement cannot expand array" << endl;
		}
		else if (locindx < 0)
		{
			cout << "FullyDistVec::SetElement local index < 0" << endl;
		}
		else
		{
			arr[locindx] = numx;
		}
	}
}

template <class IT, class NT>
NT FullyDistVec<IT,NT>::GetElement (IT indx) const
{
	NT ret;
	MPI::Intracomm World = commGrid->GetWorld();
	int rank = World.Get_rank();
	int nprocs = World.Get_size();
	IT n_perproc = getTypicalLocLength();
	IT offset = rank * n_perproc;	
	if (n_perproc == 0 && rank == 0) 
	{
		cout << "FullyDistVec::GetElement can't be called on an empty vector." << endl;
		return numeric_limits<NT>::min();
	}
		
	int owner = (indx) / n_perproc;	
	IT rec = (owner < nprocs-1) ? owner : nprocs-1;	// find its owner 
		
	if(rec == rank) // this process is the owner
	{
		IT locindx = indx-offset;

		if (locindx > arr.size()-1)
		{
			cout << "FullyDistVec::GetElement cannot expand array" << endl;
		}
		else if (locindx < 0)
		{
			cout << "FullyDistVec::GetElement local index < 0" << endl;
		}
		else
		{
			ret = arr[locindx];
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

	IT n_perproc = getTypicalLocLength();
	IT lengthuntil = rank * n_perproc;

	// The disp displacement argument specifies the position 
	// (absolute offset in bytes from the beginning of the file) 
    	thefile.Set_view(lengthuntil * sizeof(NT), MPIType<NT>(), MPIType<NT>(), "native", MPI::INFO_NULL);

	int count = arr.size();
	thefile.Write(&(arr[0]), count, MPIType<NT>());
	thefile.Close();
	
	// Now let processor-0 read the file and print
	IT total = getTotalLength(World);
	if(rank == 0)
	{
		FILE * f = fopen("temp_fullydistvec", "r");
                if(!f)
                {
                        cerr << "Problem reading binary input file\n";
                        return;
                }
		IT maxd = std::max(total-(n_perproc * (nprocs-1)), n_perproc);
		NT * data = new NT[maxd];

		for(int i=0; i<nprocs-1; ++i)
		{
			// read n_perproc integers and print them
			fread(data, sizeof(NT), n_perproc,f);

			cout << "Elements stored on proc " << i << ": {" ;	
			for (int j = 0; j < n_perproc; j++)
			{
				cout << data[j] << ",";
			}
			cout << "}" << endl;
		}
		// read the remaining total-n_perproc integers and print them
		fread(data, sizeof(NT), total-(n_perproc * (nprocs-1)), f);
		
		cout << "Elements stored on proc " << nprocs-1 << ": {" ;	
		for (int j = 0; j < total-(n_perproc * (nprocs-1)); j++)
		{
			cout << data[j] << ",";
		}
		cout << "}" << endl;
		delete [] data;
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
	IT size = arr.size();
	pair<double,IT> * vecpair = new pair<double,IT>[size];

	int nprocs = World.Get_size();
	int rank = World.Get_rank();

	IT * dist = new IT[nprocs];
	dist[rank] = size;
	World.Allgather(MPI::IN_PLACE, 1, MPIType<IT>(), dist, 1, MPIType<IT>());
	IT lengthuntil = accumulate(dist, dist+rank, 0);

  	MTRand M;	// generate random numbers with Mersenne Twister
	for(int i=0; i<size; ++i)
	{
		vecpair[i].first = M.rand();
		vecpair[i].second = arr[i];	
	}

	// less< pair<T1,T2> > works correctly (sorts wrt first elements)	
    	psort::parallel_sort (vecpair, vecpair + size,  dist, World);

	vector< NT > nnum(size);
	for(int i=0; i<size; ++i)
		nnum[i] = vecpair[i].second;

	delete [] vecpair;
	delete [] dist;

	arr.swap(nnum);
}

template <class IT, class NT>
void FullyDistVec<IT,NT>::iota(IT size, NT first)
{
	MPI::Intracomm World = commGrid->GetWorld();
	int rank = World.Get_rank();
	int nprocs = World.Get_size();
	IT n_perproc = size / nprocs;

	IT length = (rank != nprocs-1) ? n_perproc: (size - (n_perproc * (nprocs-1)));
	arr.resize(length);
	SpHelper::iota(arr.begin(), arr.end(), (rank * n_perproc) + first);	// global across processors
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
	FullyDistVec<IT,NT> Indexed(commGrid, zero);	// length(Indexed) = length(ri)
	int rank = World.Get_rank();
	int nprocs = World.Get_size();
	IT n_perproc = getTypicalLocLength();
	vector< vector< IT > > data_req(nprocs);	
	vector< vector< IT > > revr_map(nprocs);	// to put the incoming data to the correct location	
	for(IT i=0; i < ri.arr.size(); ++i)
	{
		int owner = ri.arr[i] / n_perproc;	// numerical values in ri are 0-based
		owner = std::min(owner, nprocs-1);	// find its owner 
		data_req[owner].push_back(ri.arr[i] - (n_perproc * owner));
		revr_map[owner].push_back(i);
	}
	IT * sendbuf = new IT[ri.arr.size()];
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

	IT * reversemap = new IT[ri.arr.size()];
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
	NT * databuf = new NT[ri.arr.size()];

	// the response counts are the same as the request counts 
	World.Alltoallv(databack, recvcnt, rdispls, MPIType<NT>(), databuf, sendcnt, sdispls, MPIType<NT>());  // send data
	DeleteAll(rdispls, recvcnt, databack);

	// Now create the output from databuf
	Indexed.arr.resize(ri.arr.size()); 
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
	IT totl = getTotalLength(commGrid->GetWorld());
	if (commGrid->GetRank() == 0)		// is always on the diagonal
		cout << "As a whole, " << vectorname << " has length " << totl << endl; 
}
