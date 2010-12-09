#include <limits>
#include "FullyDistSpVec.h"
#include "SpDefs.h"
#include "SpHelper.h"
using namespace std;

template <class IT, class NT>
FullyDistSpVec<IT, NT>::FullyDistSpVec ( shared_ptr<CommGrid> grid)
: commGrid(grid), length(zero), NOT_FOUND(numeric_limits<NT>::min())
{ };

template <class IT, class NT>
FullyDistSpVec<IT, NT>::FullyDistSpVec ( shared_ptr<CommGrid> grid, IT loclen)
: commGrid(grid), length(loclen), NOT_FOUND(numeric_limits<NT>::min())
{ };

template <class IT, class NT>
FullyDistSpVec<IT, NT>::FullyDistSpVec (): length(zero), NOT_FOUND(numeric_limits<NT>::min())
{
	commGrid.reset(new CommGrid(MPI::COMM_WORLD, 0, 0));
};

template <class IT, class NT>
FullyDistSpVec<IT, NT>::FullyDistSpVec (IT loclen): length(loclen), NOT_FOUND(numeric_limits<NT>::min())
{
	commGrid.reset(new CommGrid(MPI::COMM_WORLD, 0, 0));
}

template <class IT, class NT>
void FullyDistSpVec<IT,NT>::stealFrom(FullyDistSpVec<IT,NT> & victim)
{
	commGrid.reset(new CommGrid(*(victim.commGrid)));		
	ind.swap(victim.ind);
	num.swap(victim.num);
	length = victim.length;
}

template <class IT, class NT>
NT FullyDistSpVec<IT,NT>::operator[](IT indx) const
{
	MPI::Intracomm World = commGrid->GetWorld();
	int rank = World.Get_rank();
	int nprocs = World.Get_size();
	IT n_perproc = getTypicalLocLength();
	IT offset = rank * n_perproc;
	int owner = std::min(static_cast<int>(indx/n_perproc), nprocs-1);
	NT val;
	if(owner == rank)
	{
		IT locindx = indx-offset; 
		typename vector<IT>::const_iterator it = lower_bound(ind.begin(), ind.end(), locindx);	// ind is a sorted vector
		if(it != ind.end() && locindx == (*it))	// found
		{
			val = num[it-ind.begin()];
		}
		else
		{
			val = NOT_FOUND;	// return NULL
		}
	}
	World.Bcast(&val, 1, MPIType<NT>(), owner);			
	return val;
}

//! Indexing is performed 0-based 
template <class IT, class NT>
void FullyDistSpVec<IT,NT>::SetElement (IT indx, NT numx)
{
	int rank = commGrid->GetRank();
	int nprocs = commGrid->GetSize();	
	IT n_perproc = getTypicalLocLength();
	int owner = std::min(static_cast<int>(indx/n_perproc), nprocs-1);
		
	if(owner == rank) 	// insert if this process is the owner
	{
		IT locindx = indx-(rank*n_perproc); 
		typename vector<IT>::iterator iter = lower_bound(ind.begin(), ind.end(), locindx);	
		if(iter == ind.end())	// beyond limits, insert from back
		{
			ind.push_back(locindx);
			num.push_back(numx);
		}
		else if (locindx < *iter)	// not found, insert in the middle
		{
			// the order of insertions is crucial
			// if we first insert to ind, then ind.begin() is invalidated !
			num.insert(num.begin() + (iter-ind.begin()), numx);
			ind.insert(iter, locindx);
		}
		else // found
		{
			*(num.begin() + (iter-ind.begin())) = numx;
		}
	}
}

/**
 * The distribution and length are inherited from ri
 * Example: This is [{1,n1},{4,n4},{7,n7},{8,n8},{9,n9}] with P_00 owning {1,4} and P_11 rest
 * Assume ri = [{1,4},{2,1}] is distributed as one element per processor
 * Then result has length 2, distrubuted one element per processor
**/
template <class IT, class NT>
FullyDistSpVec<IT,NT> FullyDistSpVec<IT,NT>::operator() (const FullyDistSpVec<IT,IT> & ri) const
{
	MPI::Intracomm World = commGrid->GetWorld();
	FullyDistSpVec<IT,NT> Indexed(commGrid);
	int nprocs = World.Get_size();
	IT n_perproc = getTypicalLocLength();
	vector< vector<IT> > data_req(nprocs);
	for(IT i=0; i < ri.num.size(); ++i)
	{
		int owner = (ri.num[i]) / n_perproc;	// numerical values in ri are 0-based
		owner = std::min(owner, nprocs-1);	// find its owner 
		data_req[owner].push_back(ri.num[i] - (owner * n_perproc));
	}
	IT * sendbuf = new IT[ri.num.size()];
	int * sendcnt = new int[nprocs];
	int * sdispls = new int[nprocs];
	for(int i=0; i<nprocs; ++i)
		sendcnt[i] = data_req[i].size();

	int * rdispls = new int[nprocs];
	int * recvcnt = new int[nprocs];
	DiagWorld.Alltoall(sendcnt, 1, MPI::INT, recvcnt, 1, MPI::INT);	// share the request counts 

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
		copy(data_req[i].begin(), data_req[i].end(), sendbuf+sdispls[i]);

	DiagWorld.Alltoallv(sendbuf, sendcnt, sdispls, MPIType<IT>(), recvbuf, recvcnt, rdispls, MPIType<IT>());  // request data
		
	// We will return the requested data, 
	// our return can be at most as big as the request
	// and smaller if we are missing some elements 
	IT * indsback = new IT[totrecv];
	NT * databack = new NT[totrecv];		

	int * ddispls = new int[nprocs];
	copy(rdispls, rdispls+nprocs, ddispls);
	for(int i=0; i<nprocs; ++i)
	{
		// this is not the most efficient method because it scans ind vector nprocs = sqrt(p) times
		IT * it = set_intersection(recvbuf+rdispls[i], recvbuf+rdispls[i]+recvcnt[i], ind.begin(), ind.end(), indsback+rdispls[i]);
		recvcnt[i] = (it - (indsback+rdispls[i]));	// update with size of the intersection
	
		IT vi = 0;
		for(int j = rdispls[i]; j < rdispls[i] + recvcnt[i]; ++j)	// fetch the numerical values
		{
			// indsback is a subset of ind
			while(indsback[j] > ind[vi]) 
				++vi;
			databack[j] = num[vi++];
		}
	}
		
	DeleteAll(recvbuf, ddispls);
	NT * databuf = new NT[ri.num.size()];

	World.Alltoall(recvcnt, 1, MPI::INT, sendcnt, 1, MPI::INT);	// share the response counts, overriding request counts 
	World.Alltoallv(indsback, recvcnt, rdispls, MPIType<IT>(), sendbuf, sendcnt, sdispls, MPIType<IT>());  // send data
	World.Alltoallv(databack, recvcnt, rdispls, MPIType<NT>(), databuf, sendcnt, sdispls, MPIType<NT>());  // send data
	DeleteAll(rdispls, recvcnt, indsback, databack);

	// Now create the output from databuf 
	for(int i=0; i<nprocs; ++i)
	{
		// data will come globally sorted from processors 
		// i.e. ind owned by proc_i is always smaller than 
		// ind owned by proc_j for j < i
		for(int j=sdispls[i]; j< sdispls[i]+sendcnt[i]; ++j)
		{
			Indexed.ind.push_back(sendbuf[j]);
			Indexed.num.push_back(databuf[j]);
		}
	}
	Indexed.length = ri.length;
	DeleteAll(sdispls, sendcnt, sendbuf, databuf);
	return Indexed;
}

template <class IT, class NT>
void FullyDistSpVec<IT,NT>::iota(IT size, NT first)
{
	int rank = commGrid->GetRank();
	int nprocs = commGrid->GetSize();
	IT n_perproc = size / nprocs;

	length = (rank != nprocs-1) ? n_perproc: (size - (n_perproc * (nprocs-1)));
	ind.resize(length);
	num.resize(length);
	SpHelper::iota(ind.begin(), ind.end(), zero);	// offset'd within processors
	SpHelper::iota(num.begin(), num.end(), (rank * n_perproc) + first);	// global across processors
}

template <class IT, class NT>
FullyDistSpVec<IT, IT> FullyDistSpVec<IT, NT>::sort()
{
	MPI::Intracomm World = commGrid->GetWorld();
	FullyDistSpVec<IT,IT> temp(commGrid);
	IT nnz = ind.size(); 
	pair<IT,IT> * vecpair = new pair<IT,IT>[nnz];

	int nprocs = World.Get_size();
	int rank = World.Get_rank();

	IT * dist = new IT[nprocs];
	dist[rank] = nnz;
	World.Allgather(MPI::IN_PLACE, 1, MPIType<IT>, dist, 1, MPIType<IT>());
	IT lengthuntil = accumulate(dist, dist+rank, 0);

	for(size_t i=0; i<nnz; ++i)
	{
		vecpair[i].first = num[i];	// we'll sort wrt numerical values
		vecpair[i].second = ind[i] + lengthuntil;	
	}

	// less< pair<T1,T2> > works correctly (sorts wrt first elements)	
    	psort::parallel_sort (vecpair, vecpair + nnz,  dist, DiagWorld);

	vector< IT > nind(nnz);
	vector< IT > nnum(nnz);
	for(size_t i=0; i<nnz; ++i)
	{
		num[i] = vecpair[i].first;	// sorted range (change the object itself)
		nind[i] = ind[i];		// make sure the sparsity distribution is the same
		nnum[i] = vecpair[i].second;	// inverse permutation stored as numerical values
	}
	delete [] vecpair;
	delete [] dist;

	temp.length = length;
	temp.ind = nind;
	temp.num = nnum;
	return temp;
}
		
template <class IT, class NT>
FullyDistSpVec<IT,NT> & FullyDistSpVec<IT, NT>::operator+=(const FullyDistSpVec<IT,NT> & rhs)
{
	if(this != &rhs)		
	{	
		IT lsize = ind.size();
		IT rsize = rhs.ind.size();

		vector< IT > nind;
		vector< NT > nnum;
		nind.reserve(lsize+rsize);
		nnum.reserve(lsize+rsize);

		IT i=0, j=0;
		while(i < lsize && j < rsize)
		{
			// assignment won't change the size of vector, push_back is necessary
			if(ind[i] > rhs.ind[j])
			{	
				nind.push_back( rhs.ind[j] );
				nnum.push_back( rhs.num[j++] );
			}
			else if(ind[i] < rhs.ind[j])
			{
				nind.push_back( ind[i] );
				nnum.push_back( num[i++] );
			}
			else
			{
				nind.push_back( ind[i] );
				nnum.push_back( num[i++] + rhs.num[j++] );
			}
		}
		ind.swap(nind);		// ind will contain the elements of nind with capacity shrunk-to-fit size
		num.swap(nnum);
		length = ind.size(); 	
	}	
	return *this;
};	
template <class IT, class NT>
FullyDistSpVec<IT,NT> & FullyDistSpVec<IT, NT>::operator-=(const FullyDistSpVec<IT,NT> & rhs)
{
	if(this != &rhs)		
	{	
		IT lsize = ind.size();
		IT rsize = rhs.ind.size();

		vector< IT > nind;
		vector< NT > nnum;
		nind.reserve(lsize+rsize);
		nnum.reserve(lsize+rsize);

		IT i=0, j=0;
		while(i < lsize && j < rsize)
		{
			// assignment won't change the size of vector, push_back is necessary
			if(ind[i] > rhs.ind[j])
			{	
				nind.push_back( rhs.ind[j] );
				nnum.push_back( -rhs.num[j++] );
			}
			else if(ind[i] < rhs.ind[j])
			{
				nind.push_back( ind[i] );
				nnum.push_back( num[i++] );
			}
			else
			{
				nind.push_back( ind[i] );
				nnum.push_back( num[i++] - rhs.num[j++] );
			}
		}
		ind.swap(nind);		// ind will contain the elements of nind with capacity shrunk-to-fit size
		num.swap(nnum);
		length = ind.size();
	} 		
	return *this;
};	

//! Called on an existing object
template <class IT, class NT>
ifstream& FullyDistSpVec<IT,NT>::ReadDistribute (ifstream& infile, int master)
{
	IT total_n, total_nnz, n_perproc;
	MPI::Intracomm World = commGrid->GetWorld();
	int neighs = World.Get_size();	// number of neighbors (including oneself)
	int buffperneigh = MEMORYINBYTES / (neighs * (sizeof(IT) + sizeof(NT)));

	int * displs = new int[neighs];
	for (int i=0; i<neighs; ++i)
		displs[i] = i*buffperneigh;

	int * curptrs; 
	int recvcount;
	IT * inds; 
	NT * vals;

	int rank = World.Get_rank();	
	if(rank == master)	// 1 processor only
	{		
		inds = new IT [ buffperneigh * neighs ];
		vals = new NT [ buffperneigh * neighs ];
		curptrs = new int[neighs]; 
		fill_n(curptrs, neighs, 0);	// fill with zero
		if (infile.is_open())
		{
			infile.clear();
			infile.seekg(0);
			infile >> total_n >> total_nnz;
			n_perproc = total_n / neighs;	// the last proc gets the extras
			World.Bcast(&total_n, 1, MPIType<IT>(), master);			
	
			IT tempind;
			NT tempval;
			IT cnz = 0;
			while ( (!infile.eof()) && cnz < total_nnz)
			{
				infile >> tempind;
				infile >> tempval;
				tempind--;
				int rec = std::min((int)(tempind / n_perproc), neighs-1);	// recipient processor along the diagonal
				inds[ rec * buffperneigh + curptrs[rec] ] = tempind;
				vals[ rec * buffperneigh + curptrs[rec] ] = tempval;
				++ (curptrs[rec]);				

				if(curptrs[rec] == buffperneigh || (cnz == (total_nnz-1)) )		// one buffer is full, or file is done !
				{
					// first, send the receive counts ...
					World.Scatter(curptrs, 1, MPI::INT, &recvcount, 1, MPI::INT, master);

					// generate space for own recv data ... (use arrays because vector<bool> is cripled, if NT=bool)
					IT * tempinds = new IT[recvcount];
					NT * tempvals = new NT[recvcount];
					
					// then, send all buffers that to their recipients ...
					World.Scatterv(inds, curptrs, displs, MPIType<IT>(), tempinds, recvcount,  MPIType<IT>(), master); 
					World.Scatterv(vals, curptrs, displs, MPIType<NT>(), tempvals, recvcount,  MPIType<NT>(), master); 
	
					// now push what is ours to tuples
					IT offset = master * n_perproc;
					for(IT i=zero; i< recvcount; ++i)
					{					
						ind.push_back( tempinds[i]-offset );
						num.push_back( tempvals[i] );
					}

					// reset current pointers so that we can reuse {inds,vals} buffers
					fill_n(curptrs, neighs, 0);
						DeleteAll(tempinds, tempvals);
				}
				++ cnz;
			}
			assert (cnz == total_nnz);
		
			// Signal the end of file to other processors along the diagonal
			fill_n(curptrs, neighs, numeric_limits<int>::max());	
			World.Scatter(curptrs, 1, MPI::INT, &recvcount, 1, MPI::INT, master);
		}
		else	// input file does not exist !
		{
			total_n = 0;	
			World.Bcast(&total_n, 1, MPIType<IT>(), master);						
		}
		DeleteAll(inds,vals, curptrs);
	}
	else 	 	// all other processors
	{
		World.Bcast(&total_n, 1, MPIType<IT>(), master);
		n_perproc = total_n / neighs;

		while(total_n > 0)	// otherwise, input file do not exist
		{
			// first receive the receive counts ...
			World.Scatter(curptrs, 1, MPI::INT, &recvcount, 1, MPI::INT, master);

			if( recvcount == numeric_limits<int>::max())
				break;
	
			// create space for incoming data ... 
			IT * tempinds = new IT[recvcount];
			NT * tempvals = new NT[recvcount];
				
			// receive actual data ... (first 4 arguments are ignored in the receiver side)
			World.Scatterv(inds, curptrs, displs, MPIType<IT>(), tempinds, recvcount,  MPIType<IT>(), master); 
			World.Scatterv(vals, curptrs, displs, MPIType<NT>(), tempvals, recvcount,  MPIType<NT>(), master); 

			// now push what is ours to tuples
			IT offset = rank * n_perproc;
			for(IT i=zero; i< recvcount; ++i)
			{					
				ind.push_back( tempinds[i]-offset );
				num.push_back( tempvals[i] );
			}
			DeleteAll(tempinds, tempvals);
		}
	}
	delete [] displs;
 	length = (rank != neighs-1) ? n_perproc: (total_n - (n_perproc * (neighs-1)));	
	World.Barrier();
	return infile;
}

template <class IT, class NT>
IT FullyDistSpVec<IT,NT>::getTotalLength(MPI::Intracomm & comm) const
{
	IT totlen = 0;
	if(comm != MPI::COMM_NULL)
	{
		comm.Allreduce( &length, & totlen, 1, MPIType<IT>(), MPI::SUM); 
	}
	return totlen;
}

template <class IT, class NT>
IT FullyDistSpVec<IT,NT>::getTypicalLocLength() const
{
        MPI::Intracomm World = commGrid->GetWorld();
	int rank = World.Get_rank();
	int nprocs = World.Get_size();
	IT n_perproc = length;
	if (rank == nprocs-1 && nprocs > 1)
	{
		// the local length on the last processor will be greater than the others if the vector length is not evenly divisible
		// but for these calculations we need that length
		World.Recv(&n_perproc, 1, MPIType<IT>(), 0, 1);
	}
	else if (dgrank == 0 && nprocs > 1)
	{
		World.Send(&n_perproc, 1, MPIType<IT>(), nprocs-1, 1);
	}
        return n_perproc;
}

template <class IT, class NT>
template <typename _BinaryOperation>
NT FullyDistSpVec<IT,NT>::Reduce(_BinaryOperation __binary_op, NT init)
{
	// std::accumulate returns init for empty sequences
	// the semantics are init + num[0] + ... + num[n]
	NT localsum = std::accumulate( num.begin(), num.end(), init, __binary_op);

	NT totalsum = init;
	(commGrid->GetWorld()).Allreduce( &localsum, &totalsum, 1, MPIType<NT>(), MPIOp<_BinaryOperation, NT>::op());
	return totalsum;
}

template <class IT, class NT>
void FullyDistSpVec<IT,NT>::PrintInfo(string vectorname) const
{
	IT nznz = getnnz();
	IT totl = getTotalLength();

	if (commGrid->GetRank() == 0)	
		cout << "As a whole, " << vectorname << " has: " << nznz << " nonzeros and length " << totl << endl; 
}

template <class IT, class NT>
void FullyDistSpVec<IT,NT>::DebugPrint()
{
	int bufsize, count;
    	MPI::Status status;

	MPI::Intracomm World = commGrid->GetWorld();
    	int rank = World.Get_rank();
    	int nprocs = World.Get_size();
    	MPI::File thefile = MPI::File::Open(World, "temp_fullydistspvec", MMPI_MODE_WRONLY | MPI_MODE_WRONLY, MPI::INFO_NULL);    

	IT n_perproc = getTypicalLocLength();
	IT lengthuntil = rank * n_perproc;

	struct mystruct
	{
		IT ind;
		NT num;
	};
	mystruct data;

	MPI::Aint addr1 = MPI::Get_address(&data.ind);
	MPI::Aint addr2 = MPI::Get_address(&data.num);
	MPI::Aint disp[2];
	disp[0] = 0;
	disp[1] = addr2 - addr1;
	int blocklen[2] = {1, 1}; 

	MPI::Datatype type[2] = { MPIType<IT>(), MPIType<NT>() };
	MPI::Datatype datatype = MPI::Create_struct(3, blocklen, disp, type);
	datatype.commit();
	int dsize = datatype.Get_size();

	// The disp displacement argument specifies the position 
	// (absolute offset in bytes from the beginning of the file) 
    	thefile.Set_view(lengthuntil * dsize, datatype, datatype, "native", MPI::INFO_NULL);

	int count = arr.size();
	mystruct * packed = new mystruct[count];
	for(int i=0; i<count; ++i)
	{
		packed[i].ind = ind[i];
		packed[i].num = num[i];
	}
	thefile.Write(packed, count, datatype, &status);
	thefile.Close();
	delete [] packed;
	
	// Now let processor-0 read the file and print
	IT total = getTotalLength(World); 
	if(rank == 0)
	{
		FILE * f = fopen("temp_fullydistspvec", "r");
                if(!f)
                { 
                        cerr << "Problem reading binary input file\n";
                        return 1;
                }
		IT maxd = std::max(total-n_per_proc, n_per_proc);
		IT * data = new IT[maxd];

		for(int i=0; i<nprocs-1; ++i)
		{
			// read n_per_proc integers and print them
			fread(data, sizeof(IT), n_per_proc,f);

			cout << "Elements stored on proc " << i << ":\n";
			for (int j = 0; j < n_per_proc; j++)
			{
				cout << "Element #" << (j+offset) << ": [" << (ind[j]) << "] = " << num[j] << endl;

				cout << data[j] << ",";
			}
			cout << "}" << endl;
		}
		// read the remaining total-n_per_proc integers and print them
		fread(data, sizeof(IT), total-n_per_proc, f);
	
		cout << "Elements stored on proc " << nprocs-1 << ":\n" ;	
		for (int j = 0; j < total-n_per_proc; j++)
		{
			cout << data[j] << ",";
		}
		delete [] data;
	}
		
}
