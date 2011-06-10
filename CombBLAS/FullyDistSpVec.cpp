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

#include <limits>
#include "FullyDistSpVec.h"
#include "SpDefs.h"
#include "SpHelper.h"
using namespace std;

template <class IT, class NT>
FullyDistSpVec<IT, NT>::FullyDistSpVec ( shared_ptr<CommGrid> grid)
: FullyDist<IT,NT,typename disable_if< is_boolean<NT>::value, NT >::type>(grid), NOT_FOUND(numeric_limits<NT>::min()), zero(0)
{ };

template <class IT, class NT>
FullyDistSpVec<IT, NT>::FullyDistSpVec ( shared_ptr<CommGrid> grid, IT globallen)
: FullyDist<IT,NT,typename disable_if< is_boolean<NT>::value, NT >::type>(grid,globallen), NOT_FOUND(numeric_limits<NT>::min()), zero(0)
{ };

template <class IT, class NT>
FullyDistSpVec<IT,NT>::FullyDistSpVec ()
: FullyDist<IT,NT,typename disable_if< is_boolean<NT>::value, NT >::type>(), NOT_FOUND(numeric_limits<NT>::min()), zero(0)
{ };

template <class IT, class NT>
FullyDistSpVec<IT,NT>::FullyDistSpVec (IT globallen)
: FullyDist<IT,NT,typename disable_if< is_boolean<NT>::value, NT >::type>(globallen), NOT_FOUND(numeric_limits<NT>::min()), zero(0)
{ }


template <class IT, class NT>
FullyDistSpVec<IT,NT> &  FullyDistSpVec<IT,NT>::operator=(const FullyDistSpVec< IT,NT > & rhs)	
{
	if(this != &rhs)		
	{
		FullyDist<IT,NT,typename disable_if< is_boolean<NT>::value, NT >::type>::operator= (rhs);	// to update glen and commGrid
		ind = rhs.ind;
		num = rhs.num;
		zero = rhs.zero;
		NOT_FOUND = rhs.NOT_FOUND;
	}
	return *this;
}

template <class IT, class NT>
FullyDistSpVec<IT,NT>::FullyDistSpVec (const FullyDistVec<IT,NT> & rhs)		// Conversion copy-constructor
{
	*this = rhs;
}

template <class IT, class NT>
FullyDistSpVec<IT,NT> &  FullyDistSpVec<IT,NT>::operator=(const FullyDistVec< IT,NT > & rhs)		// conversion from dense
{
	FullyDist<IT,NT,typename disable_if< is_boolean<NT>::value, NT >::type>::operator= (rhs);	// to update glen and commGrid
	NOT_FOUND = numeric_limits<NT>::min();
	zero = rhs.zero;

	IT vecsize = rhs.LocArrSize();
	for(IT i=0; i< vecsize; ++i)
	{
		if(rhs.arr[i] != rhs.zero)
		{
			ind.push_back(i);
			num.push_back(rhs.arr[i]);
		}
	}
	return *this;
}

template <class IT, class NT>
void FullyDistSpVec<IT,NT>::stealFrom(FullyDistSpVec<IT,NT> & victim)
{
	FullyDist<IT,NT,typename disable_if< is_boolean<NT>::value, NT >::type>::operator= (victim);	// to update glen and commGrid
	ind.swap(victim.ind);
	num.swap(victim.num);
	NOT_FOUND = victim.NOT_FOUND;
	zero = victim.zero;
}

template <class IT, class NT>
NT FullyDistSpVec<IT,NT>::operator[](IT indx) const
{
	NT val;
	IT locind;
	int owner = Owner(indx, locind);
	if(commGrid->GetRank() == owner)
	{
		typename vector<IT>::const_iterator it = lower_bound(ind.begin(), ind.end(), locind);	// ind is a sorted vector
		if(it != ind.end() && locind == (*it))	// found
		{
			val = num[it-ind.begin()];
		}
		else
		{
			val = NOT_FOUND;	// return NULL
		}
	}
	(commGrid->GetWorld()).Bcast(&val, 1, MPIType<NT>(), owner);			
	return val;
}

//! Indexing is performed 0-based 
template <class IT, class NT>
void FullyDistSpVec<IT,NT>::SetElement (IT indx, NT numx)
{
	if(glen == 0)
		SpParHelper::Print("WARNING: SetElement() called on a vector with zero length\n");

	IT locind;
	int owner = Owner(indx, locind);
	if(commGrid->GetRank() == owner)
	{
		typename vector<IT>::iterator iter = lower_bound(ind.begin(), ind.end(), locind);	
		if(iter == ind.end())	// beyond limits, insert from back
		{
			ind.push_back(locind);
			num.push_back(numx);
		}
		else if (locind < *iter)	// not found, insert in the middle
		{
			// the order of insertions is crucial
			// if we first insert to ind, then ind.begin() is invalidated !
			num.insert(num.begin() + (iter-ind.begin()), numx);
			ind.insert(iter, locind);
		}
		else // found
		{
			*(num.begin() + (iter-ind.begin())) = numx;
		}
	}
}

template <class IT, class NT>
void FullyDistSpVec<IT,NT>::DelElement (IT indx)
{
	IT locind;
	int owner = Owner(indx, locind);
	if(commGrid->GetRank() == owner)
	{
		typename vector<IT>::iterator iter = lower_bound(ind.begin(), ind.end(), locind);	
		if(iter != ind.end() && !(locind < *iter))
		{
			num.erase(num.begin() + (iter-ind.begin()));
			ind.erase(iter);
		}
	}
}

/**
 * The distribution and length are inherited from ri
 * Its zero is inherited from *this (because ri is of type IT)
 * Example: This is [{1,n1},{4,n4},{7,n7},{8,n8},{9,n9}] with P_00 owning {1,4} and P_11 rest
 * Assume ri = [4,1,5,7] is distributed as two elements per processor
 * Then result has length 4, distrubuted two element per processor, even though 5 and 7 doesn't exist
 * This is because we are returning a "dense" output, so the absent elements will be padded with 0
**/
template <class IT, class NT>
FullyDistVec<IT,NT> FullyDistSpVec<IT,NT>::operator() (const FullyDistVec<IT,IT> & ri) const
{
	MPI::Intracomm World = commGrid->GetWorld();
	// FullyDistVec ( shared_ptr<CommGrid> grid, IT globallen, NT initval, NT id);
	FullyDistVec<IT,NT> Indexed(ri.commGrid, ri.glen, zero, zero);
	int nprocs = World.Get_size();
        unordered_map<IT, IT> revr_map;       // inverted index that maps indices of *this to indices of output
	vector< vector<IT> > data_req(nprocs);
	IT locnnz = ri.LocArrSize();

	// ABAB: Input sanity check
	int local = 1;
	int whole = 1;
	for(IT i=0; i < locnnz; ++i)
	{
		if(ri.arr[i] >= glen || ri.arr[i] < 0)
		{
			local = 0;
		} 
	}
	World.Allreduce( &local, &whole, 1, MPI::INT, MPI::BAND);
	if(whole == 0)
	{
		throw outofrangeexception();
	}

	for(IT i=0; i < locnnz; ++i)
	{
		IT locind;
		int owner = Owner(ri.arr[i], locind);	// numerical values in ri are 0-based
		data_req[owner].push_back(locind);
                revr_map.insert(typename unordered_map<IT, IT>::value_type(locind, i));
	}
	IT * sendbuf = new IT[locnnz];
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
	IT totrecv = accumulate(recvcnt,recvcnt+nprocs,0);
	IT * recvbuf = new IT[totrecv];

	for(int i=0; i<nprocs; ++i)
	{
		copy(data_req[i].begin(), data_req[i].end(), sendbuf+sdispls[i]);
		vector<IT>().swap(data_req[i]);
	}
	World.Alltoallv(sendbuf, sendcnt, sdispls, MPIType<IT>(), recvbuf, recvcnt, rdispls, MPIType<IT>());  // request data
		
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
	NT * databuf = new NT[ri.LocArrSize()];

	World.Alltoall(recvcnt, 1, MPI::INT, sendcnt, 1, MPI::INT);	// share the response counts, overriding request counts 
	World.Alltoallv(indsback, recvcnt, rdispls, MPIType<IT>(), sendbuf, sendcnt, sdispls, MPIType<IT>());  // send indices
	World.Alltoallv(databack, recvcnt, rdispls, MPIType<NT>(), databuf, sendcnt, sdispls, MPIType<NT>());  // send data
	DeleteAll(rdispls, recvcnt, indsback, databack);

	// Now create the output from databuf (holds numerical values) and sendbuf (holds indices)
	// arr is already resized during its construction
	for(int i=0; i<nprocs; ++i)
	{
		// data will come globally sorted from processors 
		// i.e. ind owned by proc_i is always smaller than 
		// ind owned by proc_j for j < i
		for(int j=sdispls[i]; j< sdispls[i]+sendcnt[i]; ++j)	
		{
			typename unordered_map<IT,IT>::iterator it = revr_map.find(sendbuf[j]);
			Indexed.arr[it->second] = databuf[j];
			// cout << it->second << "(" << sendbuf[j] << "):" << databuf[j] << endl;
		}
	}
	DeleteAll(sdispls, sendcnt, sendbuf, databuf);
	return Indexed;
}

template <class IT, class NT>
void FullyDistSpVec<IT,NT>::iota(IT globalsize, NT first)
{
	glen = globalsize;
	IT length = MyLocLength();
	ind.resize(length);
	num.resize(length);
	SpHelper::iota(ind.begin(), ind.end(), 0);	// offset'd within processors
	SpHelper::iota(num.begin(), num.end(), LengthUntil() + first);	// global across processors
}

template <class IT, class NT>
FullyDistSpVec<IT, IT> FullyDistSpVec<IT, NT>::sort()
{
	MPI::Intracomm World = commGrid->GetWorld();
	FullyDistSpVec<IT,IT> temp(commGrid);
	IT nnz = getlocnnz(); 
	pair<NT,IT> * vecpair = new pair<NT,IT>[nnz];
	int nprocs = World.Get_size();
	int rank = World.Get_rank();

	IT * dist = new IT[nprocs];
	dist[rank] = nnz;
	World.Allgather(MPI::IN_PLACE, 1, MPIType<IT>(), dist, 1, MPIType<IT>());
	IT sizeuntil = accumulate(dist, dist+rank, 0);
	for(IT i=0; i< nnz; ++i)
	{
		vecpair[i].first = num[i];	// we'll sort wrt numerical values
		vecpair[i].second = ind[i] + sizeuntil;	
	}
	SpParHelper::MemoryEfficientPSort(vecpair, nnz, dist, World);

	vector< IT > nind(nnz);
	vector< IT > nnum(nnz);
	for(IT i=0; i< nnz; ++i)
	{
		num[i] = vecpair[i].first;	// sorted range (change the object itself)
		nind[i] = ind[i];		// make sure the sparsity distribution is the same
		nnum[i] = vecpair[i].second;	// inverse permutation stored as numerical values
	}
	delete [] vecpair;
	delete [] dist;

	temp.NOT_FOUND = NOT_FOUND;
	temp.glen = glen;
	temp.ind = nind;
	temp.num = nnum;
	return temp;
}
		
template <class IT, class NT>
FullyDistSpVec<IT,NT> & FullyDistSpVec<IT, NT>::operator+=(const FullyDistSpVec<IT,NT> & rhs)
{
	if(this != &rhs)		
	{	
		if(glen != rhs.glen)
		{
			cerr << "Vector dimensions don't match for addition\n";
			return *this; 	
		}
		IT lsize = getlocnnz();
		IT rsize = rhs.getlocnnz();

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
		while( i < lsize)	// rhs was depleted first
		{
			nind.push_back( ind[i] );
			nnum.push_back( num[i++] );
		}
		while( j < rsize) 	// *this was depleted first
		{
			nind.push_back( rhs.ind[j] );
			nnum.push_back( rhs.num[j++] );
		}
		ind.swap(nind);		// ind will contain the elements of nind with capacity shrunk-to-fit size
		num.swap(nnum);
	}	
	else
	{		
		typename vector<NT>::iterator it;
		for(it = num.begin(); it != num.end(); ++it)
			(*it) *= 2;
	}
	return *this;
};	
template <class IT, class NT>
FullyDistSpVec<IT,NT> & FullyDistSpVec<IT, NT>::operator-=(const FullyDistSpVec<IT,NT> & rhs)
{
	if(this != &rhs)		
	{	
		if(glen != rhs.glen)
		{
			cerr << "Vector dimensions don't match for addition\n";
			return *this; 	
		}
		IT lsize = getlocnnz();
		IT rsize = rhs.getlocnnz();
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
				nnum.push_back( -static_cast<NT>(rhs.num[j++]) );
			}
			else if(ind[i] < rhs.ind[j])
			{
				nind.push_back( ind[i] );
				nnum.push_back( num[i++] );
			}
			else
			{
				nind.push_back( ind[i] );
				nnum.push_back( num[i++] - rhs.num[j++] );	// ignore numerical cancellations
			}
		}
		while( i < lsize)	// rhs was depleted first
		{
			nind.push_back( ind[i] );
			nnum.push_back( num[i++] );
		}
		while( j < rsize) 	// *this was depleted first
		{
			nind.push_back( rhs.ind[j] );
			nnum.push_back( zero - (rhs.num[j++]) );
		}
		ind.swap(nind);		// ind will contain the elements of nind with capacity shrunk-to-fit size
		num.swap(nnum);
	} 		
	else
	{
		ind.clear();
		num.clear();
	}
	return *this;
};	

//! Called on an existing object
template <class IT, class NT>
ifstream& FullyDistSpVec<IT,NT>::ReadDistribute (ifstream& infile, int master)
{
	IT total_nnz;
	MPI::Intracomm World = commGrid->GetWorld();
	int neighs = World.Get_size();	// number of neighbors (including oneself)
	int buffperneigh = MEMORYINBYTES / (neighs * (sizeof(IT) + sizeof(NT)));

	int * displs = new int[neighs];
	for (int i=0; i<neighs; ++i)
		displs[i] = i*buffperneigh;

	int * curptrs = NULL; 
	int recvcount = 0;
	IT * inds = NULL; 
	NT * vals = NULL;
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
			infile >> glen >> total_nnz;
			World.Bcast(&glen, 1, MPIType<IT>(), master);			
	
			IT tempind;
			NT tempval;
			double loadval;
			IT cnz = 0;
			while ( (!infile.eof()) && cnz < total_nnz)
			{
				infile >> tempind;
				//infile >> tempval;
				infile >> loadval;
				tempval = static_cast<NT>(loadval);
				tempind--;
				IT locind;
				int rec = Owner(tempind, locind);	// recipient (owner) processor
				inds[ rec * buffperneigh + curptrs[rec] ] = locind;
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
					for(IT i=0; i< recvcount; ++i)
					{					
						ind.push_back( tempinds[i] );	// already offset'd by the sender
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
			glen = 0;	
			World.Bcast(&glen, 1, MPIType<IT>(), master);						
		}
		DeleteAll(inds,vals, curptrs);
	}
	else 	 	// all other processors
	{
		World.Bcast(&glen, 1, MPIType<IT>(), master);

		while(glen > 0)	// otherwise, input file do not exist
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
			for(IT i=0; i< recvcount; ++i)
			{					
				ind.push_back( tempinds[i] );
				num.push_back( tempvals[i] );
			}
			DeleteAll(tempinds, tempvals);
		}
	}
	delete [] displs;
	World.Barrier();
	return infile;
}

template <class IT, class NT>
template <typename _Predicate>
IT FullyDistSpVec<IT,NT>::Count(_Predicate pred) const
{
	IT local = count_if( num.begin(), num.end(), pred );
	IT whole = 0;
	commGrid->GetWorld().Allreduce( &local, &whole, 1, MPIType<IT>(), MPI::SUM);
	return whole;	
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
template <typename _BinaryOperation, typename _UnaryOperation>
NT FullyDistSpVec<IT,NT>::Reduce(_BinaryOperation __binary_op, NT default_val, _UnaryOperation __unary_op)
{
	// std::accumulate returns identity for empty sequences
	NT localsum = default_val; 
	
	if (num.size() > 0)
	{
		typename vector< NT >::const_iterator iter = num.begin();
		localsum = __unary_op(*iter);
		iter++;
		while (iter < num.end())
		{
			localsum = __binary_op(localsum, __unary_op(*iter));
			iter++;
		}
	}

	NT totalsum = default_val;
	(commGrid->GetWorld()).Allreduce( &localsum, &totalsum, 1, MPIType<NT>(), MPIOp<_BinaryOperation, NT>::op());
	return totalsum;
}

template <class IT, class NT>
void FullyDistSpVec<IT,NT>::PrintInfo(string vectorname) const
{
	IT nznz = getnnz();
	if (commGrid->GetRank() == 0)	
		cout << "As a whole, " << vectorname << " has: " << nznz << " nonzeros and length " << glen << endl; 
}

template <class IT, class NT>
void FullyDistSpVec<IT,NT>::DebugPrint()
{
	MPI::Intracomm World = commGrid->GetWorld();
    	int rank = World.Get_rank();
    	int nprocs = World.Get_size();
    	MPI::File thefile = MPI::File::Open(World, "temp_fullydistspvec", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI::INFO_NULL);    

	IT * dist = new IT[nprocs];
	dist[rank] = getlocnnz();
	World.Allgather(MPI::IN_PLACE, 1, MPIType<IT>(), dist, 1, MPIType<IT>());
	IT sizeuntil = accumulate(dist, dist+rank, 0);

	struct mystruct
	{
		IT ind;
		NT num;
	};
	mystruct data;

	MPI::Datatype datatype = MPI::CHAR.Create_contiguous(sizeof(mystruct));
	datatype.Commit();
	int dsize = datatype.Get_size();

	// The disp displacement argument specifies the position 
	// (absolute offset in bytes from the beginning of the file) 
    	thefile.Set_view(static_cast<int>(sizeuntil * dsize), datatype, datatype, "native", MPI::INFO_NULL);

	int count = ind.size();
	mystruct * packed = new mystruct[count];
	for(int i=0; i<count; ++i)
	{
		packed[i].ind = ind[i];
		packed[i].num = num[i];
	}
	thefile.Write(packed, count, datatype);
	thefile.Close();
	delete [] packed;
	
	// Now let processor-0 read the file and print
	if(rank == 0)
	{
		FILE * f = fopen("temp_fullydistspvec", "r");
                if(!f)
                { 
                        cerr << "Problem reading binary input file\n";
                        return;
                }
		IT maxd = *max_element(dist, dist+nprocs);
		mystruct * data = new mystruct[maxd];

		for(int i=0; i<nprocs; ++i)
		{
			// read n_per_proc integers and print them
			fread(data, dsize, dist[i],f);	

			cout << "Elements stored on proc " << i << ": {";
			for (int j = 0; j < dist[i]; j++)
			{
				cout << "(" << data[j].ind << "," << data[j].num << "), ";
			}
			cout << "}" << endl;
		}
		delete [] data;
		delete [] dist;
	}
}
