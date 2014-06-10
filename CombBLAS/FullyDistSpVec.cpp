/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.4 -------------------------------------------------*/
/* date: 1/17/2014 ---------------------------------------------*/
/* authors: Aydin Buluc (abuluc@lbl.gov), Adam Lugowski --------*/
/****************************************************************/
/*
 Copyright (c) 2010-2014, The Regents of the University of California
 
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
#include "hash.hpp"
using namespace std;

template <class IT, class NT>
FullyDistSpVec<IT, NT>::FullyDistSpVec ( shared_ptr<CommGrid> grid)
: FullyDist<IT,NT,typename CombBLAS::disable_if< CombBLAS::is_boolean<NT>::value, NT >::type>(grid)
{ };

template <class IT, class NT>
FullyDistSpVec<IT, NT>::FullyDistSpVec ( shared_ptr<CommGrid> grid, IT globallen)
: FullyDist<IT,NT,typename CombBLAS::disable_if< CombBLAS::is_boolean<NT>::value, NT >::type>(grid,globallen)
{ };

template <class IT, class NT>
FullyDistSpVec<IT,NT>::FullyDistSpVec ()
: FullyDist<IT,NT,typename CombBLAS::disable_if< CombBLAS::is_boolean<NT>::value, NT >::type>()
{ };

template <class IT, class NT>
FullyDistSpVec<IT,NT>::FullyDistSpVec (IT globallen)
: FullyDist<IT,NT,typename CombBLAS::disable_if< CombBLAS::is_boolean<NT>::value, NT >::type>(globallen)
{ }


template <class IT, class NT>
FullyDistSpVec<IT,NT> &  FullyDistSpVec<IT,NT>::operator=(const FullyDistSpVec< IT,NT > & rhs)
{
	if(this != &rhs)
	{
		FullyDist<IT,NT,typename CombBLAS::disable_if< CombBLAS::is_boolean<NT>::value, NT >::type>::operator= (rhs);	// to update glen and commGrid
		ind = rhs.ind;
		num = rhs.num;
	}
	return *this;
}

template <class IT, class NT>
FullyDistSpVec<IT,NT>::FullyDistSpVec (const FullyDistVec<IT,NT> & rhs)		// Conversion copy-constructor
{
	*this = rhs;
}

template <class IT, class NT>
template <typename _UnaryOperation>
FullyDistSpVec<IT,NT>::FullyDistSpVec (const FullyDistVec<IT,NT> & rhs, _UnaryOperation unop)		// Conversion copy-constructor where unary op is true
{
	FullyDist<IT,NT,typename CombBLAS::disable_if< CombBLAS::is_boolean<NT>::value, NT >::type>::operator= (rhs);	// to update glen and commGrid
    
	vector<IT>().swap(ind);
	vector<NT>().swap(num);
	IT vecsize = rhs.LocArrSize();
	for(IT i=0; i< vecsize; ++i)
	{
		if(unop(rhs.arr[i]))
        {
            ind.push_back(i);
            num.push_back(rhs.arr[i]);
        }
	}
}

template <class IT, class NT>
FullyDistSpVec<IT,NT> &  FullyDistSpVec<IT,NT>::operator=(const FullyDistVec< IT,NT > & rhs)		// conversion from dense
{
	FullyDist<IT,NT,typename CombBLAS::disable_if< CombBLAS::is_boolean<NT>::value, NT >::type>::operator= (rhs);	// to update glen and commGrid

	vector<IT>().swap(ind);
	vector<NT>().swap(num);
	IT vecsize = rhs.LocArrSize();
	for(IT i=0; i< vecsize; ++i)
	{
		// rhs.zero does not exist after CombBLAS 1.2
		ind.push_back(i);
		num.push_back(rhs.arr[i]);
	}
	return *this;
}

template <class IT, class NT>
void FullyDistSpVec<IT,NT>::stealFrom(FullyDistSpVec<IT,NT> & victim)
{
	FullyDist<IT,NT,typename CombBLAS::disable_if< CombBLAS::is_boolean<NT>::value, NT >::type>::operator= (victim);	// to update glen and commGrid
	ind.swap(victim.ind);
	num.swap(victim.num);
}

template <class IT, class NT>
NT FullyDistSpVec<IT,NT>::operator[](IT indx)
{
	NT val;
	IT locind;
	int owner = Owner(indx, locind);
	int found = 0;
	if(commGrid->GetRank() == owner)
	{
		typename vector<IT>::const_iterator it = lower_bound(ind.begin(), ind.end(), locind);	// ind is a sorted vector
		if(it != ind.end() && locind == (*it))	// found
		{
			val = num[it-ind.begin()];
			found = 1;
		}
		else
		{
			val = NT();	// return NULL
			found = 0;
		}
	}
	MPI_Bcast(&found, 1, MPI_INT, owner, commGrid->GetWorld());
	MPI_Bcast(&val, 1, MPIType<NT>(), owner, commGrid->GetWorld());
	wasFound = found;
	return val;
}

template <class IT, class NT>
NT FullyDistSpVec<IT,NT>::GetLocalElement(IT indx)
{
	NT val = NT();
	IT locind;
	int owner = Owner(indx, locind);
	int found = 0;
	typename vector<IT>::const_iterator it = lower_bound(ind.begin(), ind.end(), locind);	// ind is a sorted vector
	if(commGrid->GetRank() == owner) {
		if(it != ind.end() && locind == (*it))	// found
		{
			val = num[it-ind.begin()];
			found = 1;
		}
	}
	wasFound = found;
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
	MPI_Comm World = commGrid->GetWorld();
	FullyDistVec<IT,NT> Indexed(ri.commGrid, ri.glen, NT());	// NT() is the initial value
	int nprocs;
	MPI_Comm_size(World, &nprocs);
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
	MPI_Allreduce( &local, &whole, 1, MPI_INT, MPI_BAND, World);
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
	MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, World);  // share the request counts

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
	MPI_Alltoallv(sendbuf, sendcnt, sdispls, MPIType<IT>(), recvbuf, recvcnt, rdispls, MPIType<IT>(), World);  // request data

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

	MPI_Alltoall(recvcnt, 1, MPI_INT, sendcnt, 1, MPI_INT, World);	// share the response counts, overriding request counts
	MPI_Alltoallv(indsback, recvcnt, rdispls, MPIType<IT>(), sendbuf, sendcnt, sdispls, MPIType<IT>(), World);  // send indices
	MPI_Alltoallv(databack, recvcnt, rdispls, MPIType<NT>(), databuf, sendcnt, sdispls, MPIType<NT>(), World);  // send data
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
	MPI_Comm World = commGrid->GetWorld();
	FullyDistSpVec<IT,IT> temp(commGrid);
	IT nnz = getlocnnz();
	pair<NT,IT> * vecpair = new pair<NT,IT>[nnz];

	int nprocs, rank;
	MPI_Comm_size(World, &nprocs);
	MPI_Comm_rank(World, &rank);

	IT * dist = new IT[nprocs];
	dist[rank] = nnz;
	MPI_Allgather(MPI_IN_PLACE, 1, MPIType<IT>(), dist, 1, MPIType<IT>(), World);
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

	temp.glen = glen;
	temp.ind = nind;
	temp.num = nnum;
	return temp;
}

template <class IT, class NT>
template <typename _BinaryOperation >
FullyDistSpVec<IT,NT> FullyDistSpVec<IT, NT>::UniqAll2All(_BinaryOperation __binary_op, MPI_Op mympiop)
{
    MPI_Comm World = commGrid->GetWorld();
	int nprocs = commGrid->GetSize();
    
    vector< vector< NT > > datsent(nprocs);
	vector< vector< IT > > indsent(nprocs);
    
    IT locind;
    size_t locvec = num.size();     // nnz in local vector
    IT lenuntil = LengthUntil();    // to convert to global index
	for(size_t i=0; i< locvec; ++i)
	{
        uint64_t myhash;    // output of MurmurHash3_x64_64 is 64-bits regardless of the input length
        MurmurHash3_x64_64((const void*) &(num[i]),sizeof(NT), 0, &myhash);
        double range = static_cast<double>(myhash) * static_cast<double>(glen);
        NT mapped = range / static_cast<double>(numeric_limits<uint64_t>::max());   // mapped is in range [0,n)
        int owner = Owner(mapped, locind);
        
        datsent[owner].push_back(num[i]);  // all identical entries will be hashed to the same value -> same processor
        indsent[owner].push_back(ind[i]+lenuntil);
    }
    int * sendcnt = new int[nprocs];
	int * sdispls = new int[nprocs];
	for(int i=0; i<nprocs; ++i)
		sendcnt[i] = (int) datsent[i].size();
    
	int * rdispls = new int[nprocs];
	int * recvcnt = new int[nprocs];
	MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, World);  // share the request counts
	sdispls[0] = 0;
	rdispls[0] = 0;
	for(int i=0; i<nprocs-1; ++i)
	{
		sdispls[i+1] = sdispls[i] + sendcnt[i];
		rdispls[i+1] = rdispls[i] + recvcnt[i];
	}
    NT * datbuf = new NT[locvec];
	for(int i=0; i<nprocs; ++i)
	{
		copy(datsent[i].begin(), datsent[i].end(), datbuf+sdispls[i]);
		vector<NT>().swap(datsent[i]);
	}
    IT * indbuf = new IT[locvec];
    for(int i=0; i<nprocs; ++i)
	{
		copy(indsent[i].begin(), indsent[i].end(), indbuf+sdispls[i]);
		vector<IT>().swap(indsent[i]);
	}
    IT totrecv = accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));
	NT * recvdatbuf = new NT[totrecv];
	MPI_Alltoallv(datbuf, sendcnt, sdispls, MPIType<NT>(), recvdatbuf, recvcnt, rdispls, MPIType<NT>(), World);
    delete [] datbuf;
    
    IT * recvindbuf = new IT[totrecv];
    MPI_Alltoallv(indbuf, sendcnt, sdispls, MPIType<IT>(), recvindbuf, recvcnt, rdispls, MPIType<IT>(), World);
    delete [] indbuf;
    
    vector< pair<NT,IT> > tosort;   // in fact, tomerge would be a better name but it is unlikely to be faster
    
	for(int i=0; i<nprocs; ++i)
	{
		for(int j = rdispls[i]; j < rdispls[i] + recvcnt[i]; ++j)	// fetch the numerical values
		{
            tosort.push_back(make_pair(recvdatbuf[j], recvindbuf[j]));
		}
	}
	DeleteAll(recvindbuf, recvdatbuf);
    std::sort(tosort.begin(), tosort.end());
    //std::unique returns an iterator to the element that follows the last element not removed.
    typename vector< pair<NT,IT> >::iterator last;
    last = std::unique (tosort.begin(), tosort.end(), equal_first<NT,IT>());

    vector< vector< NT > > datback(nprocs);
	vector< vector< IT > > indback(nprocs);
    
    for(typename vector< pair<NT,IT> >::iterator itr = tosort.begin(); itr != last; ++itr)
    {
        IT locind;
        int owner = Owner(itr->second, locind);
        
        datback[owner].push_back(itr->first);
        indback[owner].push_back(locind);
    }
    for(int i=0; i<nprocs; ++i) sendcnt[i] = (int) datback[i].size();
    MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, World);  // share the request counts
    for(int i=0; i<nprocs-1; ++i)
	{
		sdispls[i+1] = sdispls[i] + sendcnt[i];
		rdispls[i+1] = rdispls[i] + recvcnt[i];
	}
    datbuf = new NT[tosort.size()];
	for(int i=0; i<nprocs; ++i)
	{
		copy(datback[i].begin(), datback[i].end(), datbuf+sdispls[i]);
		vector<NT>().swap(datback[i]);
	}
    indbuf = new IT[tosort.size()];
    for(int i=0; i<nprocs; ++i)
	{
		copy(indback[i].begin(), indback[i].end(), indbuf+sdispls[i]);
		vector<IT>().swap(indback[i]);
	}
    totrecv = accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));   // update value
    
    recvdatbuf = new NT[totrecv];
	MPI_Alltoallv(datbuf, sendcnt, sdispls, MPIType<NT>(), recvdatbuf, recvcnt, rdispls, MPIType<NT>(), World);
    delete [] datbuf;
    
    recvindbuf = new IT[totrecv];
    MPI_Alltoallv(indbuf, sendcnt, sdispls, MPIType<IT>(), recvindbuf, recvcnt, rdispls, MPIType<IT>(), World);
    delete [] indbuf;

    FullyDistSpVec<IT,NT> Indexed(commGrid, glen);	// length(Indexed) = length(glen) = length(*this)
    
    vector< pair<IT,NT> > back2sort;
    for(int i=0; i<nprocs; ++i)
	{
		for(int j = rdispls[i]; j < rdispls[i] + recvcnt[i]; ++j)	// fetch the numerical values
		{
            back2sort.push_back(make_pair(recvindbuf[j], recvdatbuf[j]));
		}
	}
    std::sort(back2sort.begin(), back2sort.end());
    for(typename vector< pair<IT,NT> >::iterator itr = back2sort.begin(); itr != back2sort.end(); ++itr)
    {
        Indexed.ind.push_back(itr->first);
        Indexed.num.push_back(itr->second);
    }
    
    DeleteAll(sdispls, rdispls, sendcnt, recvcnt);
    DeleteAll(recvindbuf, recvdatbuf);
    return Indexed;
    
}

//! Only works for cases where the range of *this is [0,n) where n=length(*this)
template <class IT, class NT>
template <typename _BinaryOperation >
FullyDistSpVec<IT,NT> FullyDistSpVec<IT, NT>::Uniq2D(_BinaryOperation __binary_op, MPI_Op mympiop)
{
    // The indices for FullyDistVec are offset'd to 1/p pieces
	// The matrix indices are offset'd to 1/sqrt(p) pieces
	// Add the corresponding offset before sending the data
	IT roffset = RowLenUntil();
	IT rrowlen = MyRowLength();
    
    // Get the right local dimensions
	IT diagneigh = commGrid->GetComplementRank();
	IT ccollen;
	MPI_Status status;
	MPI_Sendrecv(&rrowlen, 1, MPIType<IT>(), diagneigh, TRROWX, &ccollen, 1, MPIType<IT>(), diagneigh, TRROWX, commGrid->GetWorld(), &status);
    
    
	// We create n-by-n matrix B from length-n vector
	// Rows(B): indices of nonzeros in vector
    // Columns(B): values in vector
    // Values(B): same as Rows(B)
    
	IT rowneighs = commGrid->GetGridCols();	// number of neighbors along this processor row (including oneself)
    IT glen = TotalLength();
	IT m_perproccol = glen / rowneighs;
    
	vector< vector<IT> > rowid(rowneighs);
	vector< vector<IT> > colid(rowneighs);
    
	size_t locvec = num.size();	// nnz in local vector
    IT lenuntil = LengthUntil();
	for(size_t i=0; i< locvec; ++i)
	{
		// numerical values (permutation indices) are 0-based
		IT rowrec = (m_perproccol!=0) ? std::min(num[i] / m_perproccol, rowneighs-1) : (rowneighs-1); 	// recipient along processor row
        
		// vector's numerical values give the colids and its indices give rowids
		rowid[rowrec].push_back( ind[i] + roffset);
		colid[rowrec].push_back( num[i] - (rowrec * m_perproccol));
	}
    
	int * sendcnt = new int[rowneighs];
	int * recvcnt = new int[rowneighs];
	for(IT i=0; i<rowneighs; ++i)
		sendcnt[i] = rowid[i].size();
    
	MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, commGrid->GetRowWorld()); // share the counts
	int * sdispls = new int[rowneighs]();
	int * rdispls = new int[rowneighs]();
	partial_sum(sendcnt, sendcnt+rowneighs-1, sdispls+1);
	partial_sum(recvcnt, recvcnt+rowneighs-1, rdispls+1);
	IT p_nnz = accumulate(recvcnt,recvcnt+rowneighs, static_cast<IT>(0));
    
	// create space for incoming data ...
	IT * p_rows = new IT[p_nnz];
	IT * p_cols = new IT[p_nnz];
  	IT * senddata = new IT[locvec];	// re-used for both rows and columns
	for(int i=0; i<rowneighs; ++i)
	{
		copy(rowid[i].begin(), rowid[i].end(), senddata+sdispls[i]);
		vector<IT>().swap(rowid[i]);	// clear memory of rowid
	}
	MPI_Alltoallv(senddata, sendcnt, sdispls, MPIType<IT>(), p_rows, recvcnt, rdispls, MPIType<IT>(), commGrid->GetRowWorld());
    
	for(int i=0; i<rowneighs; ++i)
	{
		copy(colid[i].begin(), colid[i].end(), senddata+sdispls[i]);
		vector<IT>().swap(colid[i]);	// clear memory of colid
	}
	MPI_Alltoallv(senddata, sendcnt, sdispls, MPIType<IT>(), p_cols, recvcnt, rdispls, MPIType<IT>(), commGrid->GetRowWorld());
	delete [] senddata;
    
	tuple<IT,IT,NT> * p_tuples = new tuple<IT,IT,NT>[p_nnz];
    
    int procrows = commGrid->GetGridRows();
	int my_procrow = commGrid->GetRankInProcCol();
	IT n_perprocrow = glen / procrows;	// length on a typical processor row
	for(IT i=0; i< p_nnz; ++i)
		p_tuples[i] = make_tuple(p_rows[i], p_cols[i], p_rows[i]+(n_perprocrow * my_procrow));
    
	DeleteAll(p_rows, p_cols);
    
	SpDCCols<IT,NT> * PSeq = new SpDCCols<IT,NT>();
	PSeq->Create( p_nnz, rrowlen, ccollen, p_tuples);		// square matrix
    SpParMat<IT,NT, SpDCCols<IT,NT> > B (PSeq, commGrid);
    //B.PrintInfo();
    
    FullyDistVec<IT,NT> colmin;
    B.Reduce(colmin, Column, __binary_op, glen+1, mympiop);    // all values are guarenteed to be smaller than "glen" {0,1,...,glen-1}
    //colmin.DebugPrint();
    
    // at this point, colmin[i] is semantically zero iff colmin[i] >= glen
    SetIfNotEqual<NT> setter(glen+1);
    B.DimApply(Column, colmin, setter); // B[i][j] to be pruned if B[i][j] >= glen
    B.Prune(bind2nd(greater<NT>(), glen));
    //B.PrintInfo();
    
    FullyDistVec<IT,NT> colind2val;
    colind2val.iota(B.getncol(), 1);    // start with 1 so that we can prune all zeros
    B.DimApply(Column, colind2val, sel2nd<NT>());
    //B.PrintInfo();
    
    FullyDistVec<IT,NT> pruned;
    B.Reduce(pruned, Row, plus<NT>(), (NT) 0);
    //pruned.DebugPrint();
    
    FullyDistSpVec<IT,NT> UniqInds(pruned, bind2nd(greater<NT>(), 0));    // only retain [< glen] entries
    //UniqInds.DebugPrint();
    
    return EWiseApply<NT>(UniqInds, *this, sel2nd<NT>(), bintotality<NT,NT>(), false, false, (NT) 0, (NT) 0);
}

    

// ABAB: \todo Concept control so it only gets called in integers
template <class IT, class NT>
template <typename _BinaryOperation >
FullyDistSpVec<IT,NT> FullyDistSpVec<IT, NT>::Uniq(_BinaryOperation __binary_op, MPI_Op mympiop)
{
#ifndef _2DUNIQ_
    return UniqAll2All(__binary_op, mympiop);
#else
    
    NT mymax = Reduce(maximum<NT>(), (NT) 0);
    NT mymin = Reduce(minimum<NT>(), TotalLength());
    if(mymax >= TotalLength() || mymin < (NT) 0)
    {
        return UniqAll2All(__binary_op, mympiop);
    }
    else
    {
        return Uniq2D(__binary_op, mympiop);
    }
#endif
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
			nnum.push_back( NT() - (rhs.num[j++]) );
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
template <class HANDLER>
ifstream& FullyDistSpVec<IT,NT>::ReadDistribute (ifstream& infile, int master, HANDLER handler)
{
	IT total_nnz;
	MPI_Comm World = commGrid->GetWorld();
	int neighs = commGrid->GetSize();  // number of neighbors (including oneself)
	int buffperneigh = MEMORYINBYTES / (neighs * (sizeof(IT) + sizeof(NT)));

	int * displs = new int[neighs];
	for (int i=0; i<neighs; ++i)
		displs[i] = i*buffperneigh;

	int * curptrs = NULL;
	int recvcount = 0;
	IT * inds = NULL;
	NT * vals = NULL;
	int rank = commGrid->GetRank();
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
			IT numrows, numcols;
			bool indIsRow = true;
			//infile >> glen >> total_nnz;
			infile >> numrows >> numcols >> total_nnz;
			if (numcols == 1)
			{ // column vector, read vector indices from the row index
				indIsRow = true;
				glen = numrows;
			}
			else
			{ // row vector, read vector indices from the column index
				indIsRow = false;
				glen = numcols;
			}
			MPI_Bcast(&glen, 1, MPIType<IT>(), master, World);

			IT tempind;
			IT temprow, tempcol;
			IT cnz = 0;
			while ( (!infile.eof()) && cnz < total_nnz)
			{
				//infile >> tempind;
				infile >> temprow >> tempcol;
				if (indIsRow)
					tempind = temprow;
				else
					tempind = tempcol;
				tempind--;
				IT locind;
				int rec = Owner(tempind, locind);	// recipient (owner) processor
				inds[ rec * buffperneigh + curptrs[rec] ] = locind;
				vals[ rec * buffperneigh + curptrs[rec] ] = handler.read(infile, tempind);
				++ (curptrs[rec]);

				if(curptrs[rec] == buffperneigh || (cnz == (total_nnz-1)) )		// one buffer is full, or file is done !
				{
					// first, send the receive counts ...
					MPI_Scatter(curptrs, 1, MPI_INT, &recvcount, 1, MPI_INT, master, World);

					// generate space for own recv data ... (use arrays because vector<bool> is cripled, if NT=bool)
					IT * tempinds = new IT[recvcount];
					NT * tempvals = new NT[recvcount];

					// then, send all buffers that to their recipients ...
					MPI_Scatterv(inds, curptrs, displs, MPIType<IT>(), tempinds, recvcount,  MPIType<IT>(), master, World);
					MPI_Scatterv(vals, curptrs, displs, MPIType<NT>(), tempvals, recvcount,  MPIType<NT>(), master, World);

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
			MPI_Scatter(curptrs, 1, MPI_INT, &recvcount, 1, MPI_INT, master, World);
		}
		else	// input file does not exist !
		{
			glen = 0;
			MPI_Bcast(&glen, 1, MPIType<IT>(), master, World);
		}
		DeleteAll(inds,vals, curptrs);
	}
	else 	 	// all other processors
	{
		MPI_Bcast(&glen, 1, MPIType<IT>(), master, World);
		while(glen > 0)	// otherwise, input file do not exist
		{
			// first receive the receive counts ...
			MPI_Scatter(curptrs, 1, MPI_INT, &recvcount, 1, MPI_INT, master, World);

			if( recvcount == numeric_limits<int>::max())
				break;

			// create space for incoming data ...
			IT * tempinds = new IT[recvcount];
			NT * tempvals = new NT[recvcount];

			// receive actual data ... (first 4 arguments are ignored in the receiver side)
			MPI_Scatterv(inds, curptrs, displs, MPIType<IT>(), tempinds, recvcount,  MPIType<IT>(), master, World);
			MPI_Scatterv(vals, curptrs, displs, MPIType<NT>(), tempvals, recvcount,  MPIType<NT>(), master, World);

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
	MPI_Barrier(World);
	return infile;
}

template <class IT, class NT>
template <class HANDLER>
void FullyDistSpVec<IT,NT>::SaveGathered(ofstream& outfile, int master, HANDLER handler, bool printProcSplits)
{
	int rank, nprocs;
	MPI_Comm World = commGrid->GetWorld();
	MPI_Comm_rank(World, &rank);
	MPI_Comm_size(World, &nprocs);
	MPI_File thefile;

	char _fn[] = "temp_fullydistspvec"; // AL: this is to avoid the problem that C++ string literals are const char* while C string literals are char*, leading to a const warning (technically error, but compilers are tolerant)
	int mpi_err = MPI_File_open(World, _fn, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &thefile);
	if(mpi_err != MPI_SUCCESS)
	{
		char mpi_err_str[MPI_MAX_ERROR_STRING];
    		int  mpi_err_strlen;
		MPI_Error_string(mpi_err, mpi_err_str, &mpi_err_strlen);
		printf("MPI_File_open failed (%s)\n", mpi_err_str);
		MPI_Abort(World, 1);
    	}

	IT * dist = new IT[nprocs];
	dist[rank] = getlocnnz();
	MPI_Allgather(MPI_IN_PLACE, 1, MPIType<IT>(), dist, 1, MPIType<IT>(), World);
	IT sizeuntil = accumulate(dist, dist+rank, static_cast<IT>(0));
	IT totalLength = TotalLength();
	IT totalNNZ = getnnz();

	struct mystruct
	{
		IT ind;
		NT num;
	};

	MPI_Datatype datatype;
	MPI_Type_contiguous(sizeof(mystruct), MPI_CHAR, &datatype );
	MPI_Type_commit(&datatype);
	int dsize;
	MPI_Type_size(datatype, &dsize);

	// The disp displacement argument specifies the position
	// (absolute offset in bytes from the beginning of the file)
	char native[] = "native"; // AL: this is to avoid the problem that C++ string literals are const char* while C string literals are char*, leading to a const warning (technically error, but compilers are tolerant)
	MPI_File_set_view(thefile, static_cast<int>(sizeuntil * dsize), datatype, datatype, native, MPI_INFO_NULL);

	int count = ind.size();
	mystruct * packed = new mystruct[count];
	for(int i=0; i<count; ++i)
	{
		packed[i].ind = ind[i] + sizeuntil;
		packed[i].num = num[i];
	}
	
	mpi_err = MPI_File_write(thefile, packed, count, datatype, NULL);
	if(mpi_err != MPI_SUCCESS)
        {
                char mpi_err_str[MPI_MAX_ERROR_STRING];
                int  mpi_err_strlen;
                MPI_Error_string(mpi_err, mpi_err_str, &mpi_err_strlen);
                printf("MPI_File_write failed (%s)\n", mpi_err_str);
                MPI_Abort(World, 1);
        }

	MPI_Barrier(World);
	MPI_File_close(&thefile);
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

		streamsize oldPrecision = outfile.precision();
		outfile.precision(21);
		outfile << totalLength << "\t1\t" << totalNNZ << endl;
		for(int i=0; i<nprocs; ++i)
		{
			// read n_per_proc integers and print them
			if (fread(data, dsize, dist[i], f) < static_cast<size_t>(dist[i]))
			{
			    cout << "fread 660 failed! attempting to continue..." << endl;
			}

			if (printProcSplits)
				outfile << "Elements stored on proc " << i << ":" << endl;

			for (int j = 0; j < dist[i]; j++)
			{
				outfile << data[j].ind+1 << "\t1\t";
				handler.save(outfile, data[j].num, data[j].ind);
				outfile << endl;
			}
		}
		outfile.precision(oldPrecision);
		fclose(f);
		remove("temp_fullydistspvec");
		delete [] data;
		delete [] dist;
	}
	MPI_Barrier(World);
}


template <class IT, class NT>
template <typename _Predicate>
IT FullyDistSpVec<IT,NT>::Count(_Predicate pred) const
{
	IT local = count_if( num.begin(), num.end(), pred );
	IT whole = 0;
	MPI_Allreduce( &local, &whole, 1, MPIType<IT>(), MPI_SUM, commGrid->GetWorld());
	return whole;
}


template <class IT, class NT>
template <typename _BinaryOperation>
NT FullyDistSpVec<IT,NT>::Reduce(_BinaryOperation __binary_op, NT init) const
{
	// std::accumulate returns init for empty sequences
	// the semantics are init + num[0] + ... + num[n]
	NT localsum = std::accumulate( num.begin(), num.end(), init, __binary_op);

	NT totalsum = init;
	MPI_Allreduce( &localsum, &totalsum, 1, MPIType<NT>(), MPIOp<_BinaryOperation, NT>::op(), commGrid->GetWorld());
	return totalsum;
}

template <class IT, class NT>
template <typename OUT, typename _BinaryOperation, typename _UnaryOperation>
OUT FullyDistSpVec<IT,NT>::Reduce(_BinaryOperation __binary_op, OUT default_val, _UnaryOperation __unary_op) const
{
	// std::accumulate returns identity for empty sequences
	OUT localsum = default_val;

	if (num.size() > 0)
	{
		typename vector< NT >::const_iterator iter = num.begin();
		//localsum = __unary_op(*iter);
		//iter++;
		while (iter < num.end())
		{
			localsum = __binary_op(localsum, __unary_op(*iter));
			iter++;
		}
	}

	OUT totalsum = default_val;
	MPI_Allreduce( &localsum, &totalsum, 1, MPIType<OUT>(), MPIOp<_BinaryOperation, OUT>::op(), commGrid->GetWorld());
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
	int rank, nprocs;
	MPI_Comm World = commGrid->GetWorld();
	MPI_Comm_rank(World, &rank);
	MPI_Comm_size(World, &nprocs);
	MPI_File thefile;

	char tfilename[32] = "temp_fullydistspvec";
	MPI_File_open(World, tfilename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &thefile);

	IT * dist = new IT[nprocs];
	dist[rank] = getlocnnz();
	MPI_Allgather(MPI_IN_PLACE, 1, MPIType<IT>(), dist, 1, MPIType<IT>(), World);
	IT sizeuntil = accumulate(dist, dist+rank, static_cast<IT>(0));

	struct mystruct
	{
		IT ind;
		NT num;
	};

	MPI_Datatype datatype;
	MPI_Type_contiguous(sizeof(mystruct), MPI_CHAR, &datatype );
	MPI_Type_commit(&datatype);
	int dsize;
	MPI_Type_size(datatype, &dsize);

	// The disp displacement argument specifies the position
	// (absolute offset in bytes from the beginning of the file)
	char openmode[32] = "native";
    	MPI_File_set_view(thefile, static_cast<int>(sizeuntil * dsize), datatype, datatype, openmode, MPI_INFO_NULL);

	int count = ind.size();
	mystruct * packed = new mystruct[count];
	for(int i=0; i<count; ++i)
	{
		packed[i].ind = ind[i];
		packed[i].num = num[i];
	}
	MPI_File_write(thefile, packed, count, datatype, NULL);
	MPI_Barrier(World);
	MPI_File_close(&thefile);
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
			if (fread(data, dsize, dist[i],f) < static_cast<size_t>(dist[i]))
			{
			    cout << "fread 802 failed! attempting to continue..." << endl;
			}

			cout << "Elements stored on proc " << i << ": {";
			for (int j = 0; j < dist[i]; j++)
			{
				cout << "(" << data[j].ind << "," << data[j].num << "), ";
			}
			cout << "}" << endl;
		}
		fclose(f);
		remove("temp_fullydistspvec");
		delete [] data;
		delete [] dist;
	}
	MPI_Barrier(World);
}

template <class IT, class NT>
void FullyDistSpVec<IT,NT>::Reset()
{
	ind.resize(0);
	num.resize(0);
}

// Assigns given locations their value, needs to be sorted
template <class IT, class NT>
void FullyDistSpVec<IT,NT>::BulkSet(IT inds[], int count) {
	ind.resize(count);
	num.resize(count);
	copy(inds, inds+count, ind.data());
	copy(inds, inds+count, num.data());
}



/*
 ** Create a new sparse vector vout by swaping the indices and values of a sparse vector vin.
 ** the length of vout is globallen, which nust be less than the maximum entry of vin.
 ** nnz(vin) = nnz(vout)
 ** for every nonzero entry vin[k]: vout[vin[k]] = k
 */

template <class IT, class NT>
FullyDistSpVec<IT,NT> FullyDistSpVec<IT,NT>::Invert (IT globallen)
{
    FullyDistSpVec<IT,NT> Inverted(commGrid, globallen);
    IT max_entry = Reduce(maximum<IT>(), (IT) 0 ) ;
    if(max_entry >= globallen)
    {
        cout << "Sparse vector has entries (" << max_entry  << ") larger than requested global vector length " << globallen << endl;
        return Inverted;
    }
	
    
	int nprocs = commGrid->GetSize();
	vector< vector< NT > > datsent(nprocs);
	vector< vector< IT > > indsent(nprocs);
    
	IT ploclen = getlocnnz();
	for(IT k=0; k < ploclen; ++k)
	{
		IT locind;
		int owner = Inverted.Owner(num[k], locind);     // numerical values in rhs are 0-based indices
        IT gind = ind[k] + LengthUntil();
        datsent[owner].push_back(gind);
		indsent[owner].push_back(locind);   // so that we don't need no correction at the recipient
	}
	int * sendcnt = new int[nprocs];
	int * sdispls = new int[nprocs];
	for(int i=0; i<nprocs; ++i)
		sendcnt[i] = (int) datsent[i].size();
    
	int * rdispls = new int[nprocs];
	int * recvcnt = new int[nprocs];
    MPI_Comm World = commGrid->GetWorld();
	MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, World);  // share the request counts
	sdispls[0] = 0;
	rdispls[0] = 0;
	for(int i=0; i<nprocs-1; ++i)
	{
		sdispls[i+1] = sdispls[i] + sendcnt[i];
		rdispls[i+1] = rdispls[i] + recvcnt[i];
	}
    NT * datbuf = new NT[ploclen];
	for(int i=0; i<nprocs; ++i)
	{
		copy(datsent[i].begin(), datsent[i].end(), datbuf+sdispls[i]);
		vector<NT>().swap(datsent[i]);
	}
    IT * indbuf = new IT[ploclen];
    for(int i=0; i<nprocs; ++i)
	{
		copy(indsent[i].begin(), indsent[i].end(), indbuf+sdispls[i]);
		vector<IT>().swap(indsent[i]);
	}
    IT totrecv = accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));
	NT * recvdatbuf = new NT[totrecv];
	MPI_Alltoallv(datbuf, sendcnt, sdispls, MPIType<NT>(), recvdatbuf, recvcnt, rdispls, MPIType<NT>(), World);
    delete [] datbuf;
    
    IT * recvindbuf = new IT[totrecv];
    MPI_Alltoallv(indbuf, sendcnt, sdispls, MPIType<IT>(), recvindbuf, recvcnt, rdispls, MPIType<IT>(), World);
    delete [] indbuf;
    
    
    vector< pair<IT,NT> > tosort;   // in fact, tomerge would be a better name but it is unlikely to be faster
    
	for(int i=0; i<nprocs; ++i)
	{
		for(int j = rdispls[i]; j < rdispls[i] + recvcnt[i]; ++j)	// fetch the numerical values
		{
            tosort.push_back(make_pair(recvindbuf[j], recvdatbuf[j]));
		}
	}
	DeleteAll(recvindbuf, recvdatbuf);
    DeleteAll(sdispls, rdispls, sendcnt, recvcnt);
    std::sort(tosort.begin(), tosort.end());
    
    for(typename vector<pair<IT,NT>>::iterator itr = tosort.begin(); itr != tosort.end(); ++itr)
    {
        Inverted.ind.push_back(itr->first);
        Inverted.num.push_back(itr->second);
	}
	return Inverted;
    
}

