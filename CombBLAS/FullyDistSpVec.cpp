/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.5 -------------------------------------------------*/
/* date: 10/09/2015 ---------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc, Adam Lugowski ------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2015, The Regents of the University of California
 
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
#include "FileHeader.h"

#ifdef GNU_PARALLEL
#include <parallel/algorithm>
#include <parallel/numeric>
#endif

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
FullyDistSpVec<IT,NT>::FullyDistSpVec (const FullyDistVec<IT,NT> & rhs) // Conversion copy-constructor
: FullyDist<IT,NT,typename CombBLAS::disable_if< CombBLAS::is_boolean<NT>::value, NT >::type>(rhs.commGrid,rhs.glen)
{
	*this = rhs;
}

// Conversion copy-constructor where unary op is true
template <class IT, class NT>
template <typename _UnaryOperation>
FullyDistSpVec<IT,NT>::FullyDistSpVec (const FullyDistVec<IT,NT> & rhs, _UnaryOperation unop)
: FullyDist<IT,NT,typename CombBLAS::disable_if< CombBLAS::is_boolean<NT>::value, NT >::type>(rhs.commGrid,rhs.glen)
{
	//FullyDist<IT,NT,typename CombBLAS::disable_if< CombBLAS::is_boolean<NT>::value, NT >::type>::operator= (rhs);	// to update glen and commGrid
    
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



// ABAB: This function probably operates differently than a user would immediately expect
// ABAB: Write a well-posed description for it
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


/************************************************************************
 * Create a sparse vector from index and value vectors (dense vectors)
 * FullyDistSpVec v(globalsize, inds, vals):
 *      nnz(v) = size(inds) = size(vals)
 *      size(v) = globallen
 *      if inds has duplicate entries and SumDuplicates is true then we sum 
 *      the values of duplicate indices. Otherwise, only the first entry is kept.
 ************************************************************************/
template <class IT, class NT>
FullyDistSpVec<IT,NT>::FullyDistSpVec (IT globallen, const FullyDistVec<IT,IT> & inds,  const FullyDistVec<IT,NT> & vals, bool SumDuplicates)
: FullyDist<IT,NT,typename CombBLAS::disable_if< CombBLAS::is_boolean<NT>::value, NT >::type>(inds.commGrid,globallen)
{
    if(*(inds.commGrid) != *(vals.commGrid))
    {
        SpParHelper::Print("Grids are not comparable, FullyDistSpVec() fails !");
        MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
    }
    if(inds.TotalLength() != vals.TotalLength())
    {
        SpParHelper::Print("Index and value vectors have different sizes, FullyDistSpVec() fails !");
        MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
    }
    //commGrid = inds.commGrid;
    //glen = globallen;
    
    IT maxind = inds.Reduce(maximum<IT>(), (IT) 0);
    if(maxind>=globallen)
    {
        SpParHelper::Print("At least one index is greater than globallen, FullyDistSpVec() fails !");
        MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
    }
    

    
    MPI_Comm World = commGrid->GetWorld();
    int nprocs = commGrid->GetSize();
    int * rdispls = new int[nprocs];
    int * recvcnt = new int[nprocs];
    int * sendcnt = new int[nprocs](); // initialize to 0
    int * sdispls = new int[nprocs];
    
    // ----- share count --------
    IT locsize = inds.LocArrSize();
    for(IT i=0; i<locsize; ++i)
    {
        IT locind;
        int owner = Owner(inds.arr[i], locind);
        sendcnt[owner]++;
    }
    MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, World);
    
    
    // ----- compute send and receive displacements --------
    
    sdispls[0] = 0;
    rdispls[0] = 0;
    for(int i=0; i<nprocs-1; ++i)
    {
        sdispls[i+1] = sdispls[i] + sendcnt[i];
        rdispls[i+1] = rdispls[i] + recvcnt[i];
    }

    
    // ----- prepare data to be sent --------
    
    NT * datbuf = new NT[locsize];
    IT * indbuf = new IT[locsize];
    int *count = new int[nprocs](); //current position
    for(IT i=0; i < locsize; ++i)
    {
        IT locind;
        int owner = Owner(inds.arr[i], locind);
        int id = sdispls[owner] + count[owner];
        datbuf[id] = vals.arr[i];
        indbuf[id] = locind;
        count[owner]++;
    }
    delete [] count;
    IT totrecv = accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));
    
    
    // ----- Send and receive indices and values --------
    
    NT * recvdatbuf = new NT[totrecv];
    MPI_Alltoallv(datbuf, sendcnt, sdispls, MPIType<NT>(), recvdatbuf, recvcnt, rdispls, MPIType<NT>(), World);
    delete [] datbuf;
    
    IT * recvindbuf = new IT[totrecv];
    MPI_Alltoallv(indbuf, sendcnt, sdispls, MPIType<IT>(), recvindbuf, recvcnt, rdispls, MPIType<IT>(), World);
    delete [] indbuf;
    
    
    // ------ merge and sort received data ----------
    
    vector< pair<IT,NT> > tosort;
    tosort.resize(totrecv);
    for(int i=0; i<totrecv; ++i)
    {
        tosort[i] = make_pair(recvindbuf[i], recvdatbuf[i]);
    }
    DeleteAll(recvindbuf, recvdatbuf);
    DeleteAll(sdispls, rdispls, sendcnt, recvcnt);
    std::sort(tosort.begin(), tosort.end());
    
    
    // ------ create local sparse vector ----------
    
    ind.reserve(totrecv);
    num.reserve(totrecv);
    IT lastIndex=-1;
    for(auto itr = tosort.begin(); itr != tosort.end(); ++itr)
    {
        if(lastIndex!=itr->first) //if SumDuplicates=false, keep only the first one
        {
            ind.push_back(itr->first);
            num.push_back(itr->second);
            lastIndex = itr->first;
        }
        else if(SumDuplicates)
        {
            num.back() += itr->second;
        }
    }
}


//! Returns a dense vector of nonzero values
//! for which the predicate is satisfied on values
template <class IT, class NT>
template <typename _Predicate>
FullyDistVec<IT,NT> FullyDistSpVec<IT,NT>::FindVals(_Predicate pred) const
{
    FullyDistVec<IT,NT> found(commGrid);
    MPI_Comm World = commGrid->GetWorld();
    int nprocs = commGrid->GetSize();
    int rank = commGrid->GetRank();
    
    IT sizelocal = getlocnnz();
    for(IT i=0; i<sizelocal; ++i)
    {
        if(pred(num[i]))
        {
            found.arr.push_back(num[i]);
        }
    }
    IT * dist = new IT[nprocs];
    IT nsize = found.arr.size();
    dist[rank] = nsize;
    MPI_Allgather(MPI_IN_PLACE, 1, MPIType<IT>(), dist, 1, MPIType<IT>(), World);
    IT lengthuntil = accumulate(dist, dist+rank, static_cast<IT>(0));
    found.glen = accumulate(dist, dist+nprocs, static_cast<IT>(0));
    
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
    MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, World); // share the counts
    
    int * sdispls = new int[nprocs];
    int * rdispls = new int[nprocs];
    sdispls[0] = 0;
    rdispls[0] = 0;
    for(int i=0; i<nprocs-1; ++i)
    {
        sdispls[i+1] = sdispls[i] + sendcnt[i];
        rdispls[i+1] = rdispls[i] + recvcnt[i];
    }
    IT totrecv = accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));
    vector<NT> recvbuf(totrecv);
    
    // data is already in the right order in found.arr
    MPI_Alltoallv(found.arr.data(), sendcnt, sdispls, MPIType<NT>(), recvbuf.data(), recvcnt, rdispls, MPIType<NT>(), World);
    found.arr.swap(recvbuf);
    delete [] dist;
    DeleteAll(sendcnt, recvcnt, sdispls, rdispls);
    
    return found;
}



//! Returns a dense vector of nonzero global indices
//! for which the predicate is satisfied on values
template <class IT, class NT>
template <typename _Predicate>
FullyDistVec<IT,IT> FullyDistSpVec<IT,NT>::FindInds(_Predicate pred) const
{
    FullyDistVec<IT,IT> found(commGrid);
    MPI_Comm World = commGrid->GetWorld();
    int nprocs = commGrid->GetSize();
    int rank = commGrid->GetRank();
    
    IT sizelocal = getlocnnz();
    IT sizesofar = LengthUntil();
    for(IT i=0; i<sizelocal; ++i)
    {
        if(pred(num[i]))
        {
            found.arr.push_back(ind[i]+sizesofar);
        }
    }
    IT * dist = new IT[nprocs];
    IT nsize = found.arr.size();
    dist[rank] = nsize;
    MPI_Allgather(MPI_IN_PLACE, 1, MPIType<IT>(), dist, 1, MPIType<IT>(), World);
    IT lengthuntil = accumulate(dist, dist+rank, static_cast<IT>(0));
    found.glen = accumulate(dist, dist+nprocs, static_cast<IT>(0));
    
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
    MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, World); // share the counts
    
    int * sdispls = new int[nprocs];
    int * rdispls = new int[nprocs];
    sdispls[0] = 0;
    rdispls[0] = 0;
    for(int i=0; i<nprocs-1; ++i)
    {
        sdispls[i+1] = sdispls[i] + sendcnt[i];
        rdispls[i+1] = rdispls[i] + recvcnt[i];
    }
    IT totrecv = accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));
    vector<IT> recvbuf(totrecv);
    
    // data is already in the right order in found.arr
    MPI_Alltoallv(found.arr.data(), sendcnt, sdispls, MPIType<IT>(), recvbuf.data(), recvcnt, rdispls, MPIType<IT>(), World);
    found.arr.swap(recvbuf);
    delete [] dist;
    DeleteAll(sendcnt, recvcnt, sdispls, rdispls);
    
    return found;
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


// - sorts the entries with respect to nonzero values
// - ignores structural zeros
// - keeps the sparsity structure intact
// - returns a permutation representing the mapping from old to new locations
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
    IT until = LengthUntil();
	for(IT i=0; i< nnz; ++i)
	{
		vecpair[i].first = num[i];	// we'll sort wrt numerical values
		vecpair[i].second = ind[i] + until;
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

/**
 * Private subroutine of ReadDistribute.
 */
template <class IT, class NT>
void FullyDistSpVec<IT,NT>::BcastEssentials(MPI_Comm & world, IT & total_m, IT & total_n, IT & total_nnz, int master)
{
	MPI_Bcast(&total_m, 1, MPIType<IT>(), master, world);
	MPI_Bcast(&total_n, 1, MPIType<IT>(), master, world);
	MPI_Bcast(&total_nnz, 1, MPIType<IT>(), master, world);
}

template <class IT, class NT>
void FullyDistSpVec<IT,NT>::AllocateSetBuffers(IT * & myinds, NT * & myvals,  int * & rcurptrs, int * & ccurptrs, int rowneighs, int colneighs, IT buffpercolneigh)
{
	// allocate buffers on the heap as stack space is usually limited
	myinds = new IT [ buffpercolneigh * colneighs ];
	myvals = new NT [ buffpercolneigh * colneighs ];
	
	ccurptrs = new int[colneighs];
	rcurptrs = new int[rowneighs];
	fill_n(ccurptrs, colneighs, 0);	// fill with zero
	fill_n(rcurptrs, rowneighs, 0);
}

/**
 * Private subroutine of ReadDistribute
 * @param[in] rankinrow {Row head's rank in its processor row}
 * Initially temp_xxx arrays carry data received along the proc. column AND needs to be sent along the proc. row
 * After usage, function frees the memory of temp_xxx arrays and then creates and frees memory of all the six arrays itself
 */
template <class IT, class NT>
void FullyDistSpVec<IT,NT>::HorizontalSend(IT * & inds, NT * & vals, IT * & tempinds, NT * & tempvals, vector < pair <IT,NT> > & localpairs,
                                         int * rcurptrs, int * rdispls, IT buffperrowneigh, int recvcount, int rankinrow)
{
    int rowneighs = commGrid->GetGridCols();	// number of neighbors along this processor row (including oneself)

	inds = new IT [ buffperrowneigh * rowneighs ];
	vals = new NT [ buffperrowneigh * rowneighs ];
	
	// prepare to send the data along the horizontal
	for(int i=0; i< recvcount; ++i)
	{
        IT localind;
        int globalowner = Owner(tempinds[i], localind);
        int rowrec = commGrid->GetRankInProcRow(globalowner); // recipient processor along the row
        
		inds[ rowrec * buffperrowneigh + rcurptrs[rowrec] ] = localind; // use the local index
		vals[ rowrec * buffperrowneigh + rcurptrs[rowrec] ] = tempvals[i];
		++ (rcurptrs[rowrec]);
	}
    
#ifdef IODEBUG
	ofstream oput;
	commGrid->OpenDebugFile("Read", oput);
	oput << "To row neighbors: ";
	copy(rcurptrs, rcurptrs+rowneighs, ostream_iterator<int>(oput, " ")); oput << endl;
	oput << "Row displacements were: ";
	copy(rdispls, rdispls+rowneighs, ostream_iterator<int>(oput, " ")); oput << endl;
	oput.close();
#endif
    
	MPI_Scatter(rcurptrs, 1, MPI_INT, &recvcount, 1, MPI_INT, rankinrow, commGrid->GetRowWorld()); // Send the receive counts for horizontal communication
    
	// the data is now stored in rows/cols/vals, can reset temporaries
	// sets size and capacity to *new* recvcount
	DeleteAll(tempinds, tempvals);
	tempinds = new IT[recvcount];
	tempvals = new NT[recvcount];
	
	// then, send all buffers that to their recipients ...
	MPI_Scatterv(inds, rcurptrs, rdispls, MPIType<IT>(), tempinds, recvcount,  MPIType<IT>(), rankinrow, commGrid->GetRowWorld());
	MPI_Scatterv(vals, rcurptrs, rdispls, MPIType<NT>(), tempvals, recvcount,  MPIType<NT>(), rankinrow, commGrid->GetRowWorld());
	
	for(int i=0; i< recvcount; ++i)
	{
		localpairs.push_back( 	make_pair(tempinds[i], tempvals[i]) );  // no need to offset-adjust as we received the local index
	}
	
	fill_n(rcurptrs, rowneighs, 0);
	DeleteAll(inds, vals, tempinds, tempvals);
}


/*
 * Private subroutine of ReadDistribute.
 * @post {rows, cols, vals are pre-allocated on the heap after this call}
 * @post {ccurptrs are set to zero; so that if another call is made to this function without modifying ccurptrs, no data will be send from this procesor}
 */
template <class IT, class NT>
void FullyDistSpVec<IT,NT>::VerticalSend(IT * & inds, NT * & vals, vector< pair<IT,NT> > & localpairs, int * rcurptrs, int * ccurptrs, int * rdispls, int * cdispls,
                                        IT buffperrowneigh, IT buffpercolneigh, int rankinrow)
{
    int colneighs = commGrid->GetGridRows();	// number of neighbors along this processor column (including oneself)
    
	// first, send/recv the counts ...
	int * colrecvdispls = new int[colneighs];
	int * colrecvcounts = new int[colneighs];
	MPI_Alltoall(ccurptrs, 1, MPI_INT, colrecvcounts, 1, MPI_INT, commGrid->GetColWorld());      // share the request counts
	int totrecv = accumulate(colrecvcounts,colrecvcounts+colneighs,0);  // ABAB: This might overflow in 1 MPI thread to 1 node/socket world
	colrecvdispls[0] = 0; 		// receive displacements are exact whereas send displacements have slack
	for(int i=0; i<colneighs-1; ++i)
		colrecvdispls[i+1] = colrecvdispls[i] + colrecvcounts[i];
	
	// generate space for own recv data ... (use arrays because vector<bool> is cripled, if NT=bool)
	IT * tempinds = new IT[totrecv];
	NT * tempvals = new NT[totrecv];
	
	// then, exchange all buffers that to their recipients ...
	MPI_Alltoallv(inds, ccurptrs, cdispls, MPIType<IT>(), tempinds, colrecvcounts, colrecvdispls, MPIType<IT>(), commGrid->GetColWorld());
	MPI_Alltoallv(vals, ccurptrs, cdispls, MPIType<NT>(), tempvals, colrecvcounts, colrecvdispls, MPIType<NT>(), commGrid->GetColWorld());
    
	// finally, reset current pointers !
	fill_n(ccurptrs, colneighs, 0);
	DeleteAll(colrecvdispls, colrecvcounts);
	DeleteAll(inds, vals);
	
	// rcurptrs/rdispls are zero initialized scratch space
	HorizontalSend(inds, vals, tempinds , tempvals, localpairs, rcurptrs, rdispls, buffperrowneigh, totrecv, rankinrow);
	
	// reuse these buffers for the next vertical communication
	inds = new IT [ buffpercolneigh * colneighs ];
	vals = new NT [ buffpercolneigh * colneighs ];
}


/**
 * Private subroutine of ReadDistribute.
 * Executed by p_r processors on the first processor column.
 * @pre {inds, vals are pre-allocated on the heap before this call}
 * @param[in] rankinrow {row head's rank in its processor row - determines the scatter person}
 */
template <class IT, class NT>
template <class HANDLER>
void FullyDistSpVec<IT,NT>::ReadAllMine(FILE * binfile, IT * & inds, NT * & vals, vector< pair<IT,NT> > & localpairs, int * rcurptrs, int * ccurptrs, int * rdispls, int * cdispls,
                                    IT buffperrowneigh, IT buffpercolneigh, IT entriestoread, HANDLER handler, int rankinrow)
{
	assert(entriestoread != 0);
    int colneighs = commGrid->GetGridRows();	// number of neighbors along this processor column (including oneself)
	int rowneighs = commGrid->GetGridCols();	// number of neighbors along this processor row (including oneself)
    
	IT cnz = 0;
	IT tempind;
	NT tempval;
	int finishedglobal = 1;
	while(cnz < entriestoread && !feof(binfile))	// this loop will execute at least once
	{
		handler.binaryfill(binfile, tempind, tempval);
        IT localind;
        int globalowner = Owner(tempind, localind);
        int colrec = commGrid->GetRankInProcCol(globalowner); // recipient processor along the column
        
		size_t commonindex = colrec * buffpercolneigh + ccurptrs[colrec];
		inds[ commonindex ] = tempind;  // keep the indices global for now (row heads will call the ::Owner function later again)
		vals[ commonindex ] = tempval;
		++ (ccurptrs[colrec]);
		if(ccurptrs[colrec] == buffpercolneigh || (cnz == (entriestoread-1)) )		// one buffer is full, or this processor's share is done !
		{
#ifdef IODEBUG
			ofstream oput;
			commGrid->OpenDebugFile("Read", oput);
			oput << "To column neighbors: ";
			copy(ccurptrs, ccurptrs+colneighs, ostream_iterator<int>(oput, " ")); oput << endl;
			oput.close();
#endif
            
			VerticalSend(inds, vals, localpairs, rcurptrs, ccurptrs, rdispls, cdispls, buffperrowneigh, buffpercolneigh, rankinrow);
            
			if(cnz == (entriestoread-1))	// last execution of the outer loop
			{
				int finishedlocal = 1;	// I am done, but let me check others
				MPI_Allreduce( &finishedlocal, &finishedglobal, 1, MPI_INT, MPI_BAND, commGrid->GetColWorld());
				while(!finishedglobal)
				{
#ifdef DEBUG
					ofstream oput;
					commGrid->OpenDebugFile("Read", oput);
					oput << "To column neighbors: ";
					copy(ccurptrs, ccurptrs+colneighs, ostream_iterator<int>(oput, " ")); oput << endl;
					oput.close();
#endif
                    
					// postcondition of VerticalSend: ccurptrs are set to zero
					// if another call is made to this function without modifying ccurptrs, no data will be send from this procesor
					VerticalSend(inds, vals, localpairs, rcurptrs, ccurptrs, rdispls, cdispls, buffperrowneigh, buffpercolneigh, rankinrow);
                    
					MPI_Allreduce( &finishedlocal, &finishedglobal, 1, MPI_INT, MPI_BAND, commGrid->GetColWorld());
				}
			}
			else // the other loop will continue executing
			{
				int finishedlocal = 0;
				MPI_Allreduce( &finishedlocal, &finishedglobal, 1, MPI_INT, MPI_BAND, commGrid->GetColWorld());
			}
		} // end_if for "send buffer is full" case
		++cnz;
	}
    
	// signal the end to row neighbors
	fill_n(rcurptrs, rowneighs, numeric_limits<int>::max());
	int recvcount;
	MPI_Scatter(rcurptrs, 1, MPI_INT, &recvcount, 1, MPI_INT, rankinrow, commGrid->GetRowWorld());
}



//! Called on an existing object
//! New generalized (parallel/binary/etc) version as of 2014
template <class IT, class NT>
template <class HANDLER>
void FullyDistSpVec<IT,NT>::ReadDistribute (const string & filename, int master, HANDLER handler, bool pario)
{
    ifstream infile;
	FILE * binfile = NULL;	// points to "past header" if the file is binary
	int seeklength = 0;
	HeaderInfo hfile;
	if(commGrid->GetRank() == master)	// 1 processor
	{
		hfile = ParseHeader(filename, binfile, seeklength);
	}
    MPI_Bcast(&seeklength, 1, MPI_INT, master, commGrid->GetWorld());
    
	IT total_m, total_n, total_nnz;
	IT m_perproc = 0, n_perproc = 0;
    
	int colneighs = commGrid->GetGridRows();	// number of neighbors along this processor column (including oneself)
	int rowneighs = commGrid->GetGridCols();	// number of neighbors along this processor row (including oneself)
    
	IT buffpercolneigh = MEMORYINBYTES / (colneighs * (sizeof(IT) + sizeof(NT)));
	IT buffperrowneigh = MEMORYINBYTES / (rowneighs * (sizeof(IT) + sizeof(NT)));
	if(pario)
	{
		// since all colneighs will be reading the data at the same time
		// chances are they might all read the data that should go to one
		// in that case buffperrowneigh > colneighs * buffpercolneigh
		// in order not to overflow
		buffpercolneigh /= colneighs;
		if(seeklength == 0)
			SpParHelper::Print("COMBBLAS: Parallel I/O requested but binary header is corrupted\n");
	}


    // make sure that buffperrowneigh >= buffpercolneigh to cover for this patological case:
	//   	-- all data received by a given column head (by vertical communication) are headed to a single processor along the row
	//   	-- then making sure buffperrowneigh >= buffpercolneigh guarantees that the horizontal buffer will never overflow
	buffperrowneigh = std::max(buffperrowneigh, buffpercolneigh);
	if(std::max(buffpercolneigh * colneighs, buffperrowneigh * rowneighs) > numeric_limits<int>::max())
	{
		SpParHelper::Print("COMBBLAS: MPI doesn't support sending int64_t send/recv counts or displacements\n");
	}
    
	int * cdispls = new int[colneighs];
	for (IT i=0; i<colneighs; ++i)  cdispls[i] = i*buffpercolneigh;
	int * rdispls = new int[rowneighs];
	for (IT i=0; i<rowneighs; ++i)  rdispls[i] = i*buffperrowneigh;
    
	// Note: all other column heads that initiate the horizontal communication has the same "rankinrow" with the master
	int rankincol = commGrid->GetRankInProcCol(master);	// get master's rank in its processor column
	int rankinrow = commGrid->GetRankInProcRow(master);
	vector< pair<IT, NT> > localpairs;
    
    int *ccurptrs = NULL, *rcurptrs = NULL;
    int recvcount = 0;
    IT * inds = NULL;
    NT * vals = NULL;
    
    if(commGrid->GetRank() == master)	// 1 processor
	{
		if( !hfile.fileexists )
		{
			SpParHelper::Print( "COMBBLAS: Input file doesn't exist\n");
            total_n = 0; total_m = 0;
			BcastEssentials(commGrid->GetWorld(), total_m, total_n, total_nnz, master);
			return;
		}
		if (hfile.headerexists && hfile.format == 1)
		{
			SpParHelper::Print("COMBBLAS: Ascii input with binary headers is not supported");
			total_n = 0; total_m = 0;
			BcastEssentials(commGrid->GetWorld(), total_m, total_n, total_nnz, master);
			return;
		}
		if ( !hfile.headerexists )	// no header - ascii file (at this point, file exists)
		{
			infile.open(filename.c_str());
			char comment[256];
			infile.getline(comment,256);
			while(comment[0] == '%')
			{
				infile.getline(comment,256);
			}
			stringstream ss;
			ss << string(comment);
			ss >> total_m >> total_n >> total_nnz;
			if(pario)
			{
				SpParHelper::Print("COMBBLAS: Trying to read binary headerless file in parallel, aborting\n");
				total_n = 0; total_m = 0;
				BcastEssentials(commGrid->GetWorld(), total_m, total_n, total_nnz, master);
				return;
			}
		}
		else // hfile.headerexists && hfile.format == 0
		{
			total_m = hfile.m;
			total_n = hfile.n;
			total_nnz = hfile.nnz;
		}
        BcastEssentials(commGrid->GetWorld(), total_m, total_n, total_nnz, master);
        
        bool rowvector = false;
        if( total_m != 1 && total_n != 1)
        {
            SpParHelper::Print("COMBBLAS: One of the dimensions should be 1 for this to be a vector, can't read\n");
            return;
        }
        else if(total_m == 1)
        {
            glen = total_n;
            rowvector = true;
        }
        else
        {
            glen = total_m;
        }
        
        AllocateSetBuffers(inds, vals,  rcurptrs, ccurptrs, rowneighs, colneighs, buffpercolneigh);
        
		if(seeklength > 0 && pario)   // sqrt(p) processors also do parallel binary i/o
		{
			IT entriestoread =  total_nnz / colneighs;
#ifdef IODEBUG
			ofstream oput;
			commGrid->OpenDebugFile("Read", oput);
			oput << "Total nnz: " << total_nnz << " entries to read: " << entriestoread << endl;
			oput.close();
#endif
			ReadAllMine(binfile, inds, vals, localpairs, rcurptrs, ccurptrs, rdispls, cdispls,
                        buffperrowneigh, buffpercolneigh, entriestoread, handler, rankinrow);
		}
		else	// only this (master) is doing I/O (text or binary)
		{
			IT tempind, temprow, tempcol;
			NT tempval;
			char line[1024];
			bool nonumline = false; // ABAB: we don't support vector files without values at the moment
			IT cnz = 0;
			for(; cnz < total_nnz; ++cnz)
			{
				size_t commonindex;
				stringstream linestream;
				if( (!hfile.headerexists) && (!infile.eof()))
				{
					// read one line at a time so that missing numerical values can be detected
					infile.getline(line, 1024);
					linestream << line;
					linestream >> temprow >> tempcol;
                    --temprow;	// file is 1-based where C-arrays are 0-based
					--tempcol;
                    tempind = rowvector ? tempcol : temprow;
					if (!nonumline)
					{
						// see if this line has a value
						linestream >> skipws;
						nonumline = linestream.eof();
					}
    
				}
				else if(hfile.headerexists && (!feof(binfile)) )
				{
					handler.binaryfill(binfile, tempind, tempval);
				}
                IT localind;
                int globalowner = Owner(tempind, localind);
                int colrec = commGrid->GetRankInProcCol(globalowner); // recipient processor along the column
                commonindex = colrec * buffpercolneigh + ccurptrs[colrec];
                
				inds[ commonindex ] = tempind;
				if( (!hfile.headerexists) && (!infile.eof()))
				{
					vals[ commonindex ] = nonumline ? handler.getNoNum(tempind) : handler.read(linestream, tempind); //tempval;
				}
				else if(hfile.headerexists && (!feof(binfile)) )
				{
					vals[ commonindex ] = tempval;
				}
				++ (ccurptrs[colrec]);
				if(ccurptrs[colrec] == buffpercolneigh || (cnz == (total_nnz-1)) )		// one buffer is full, or file is done !
				{
					MPI_Scatter(ccurptrs, 1, MPI_INT, &recvcount, 1, MPI_INT, rankincol, commGrid->GetColWorld()); // first, send the receive counts
                    
					// generate space for own recv data ... (use arrays because vector<bool> is cripled, if NT=bool)
					IT * tempinds = new IT[recvcount];
					NT * tempvals = new NT[recvcount];
					
					// then, send all buffers that to their recipients ...
					MPI_Scatterv(inds, ccurptrs, cdispls, MPIType<IT>(), tempinds, recvcount,  MPIType<IT>(), rankincol, commGrid->GetColWorld());
					MPI_Scatterv(vals, ccurptrs, cdispls, MPIType<NT>(), tempvals, recvcount,  MPIType<NT>(), rankincol, commGrid->GetColWorld());
                    
					fill_n(ccurptrs, colneighs, 0);  				// finally, reset current pointers !
					DeleteAll(inds, vals);
					
					HorizontalSend(inds, vals,tempinds, tempvals, localpairs, rcurptrs, rdispls, buffperrowneigh, recvcount, rankinrow);
					
					if( cnz != (total_nnz-1) )	// otherwise the loop will exit with noone to claim memory back
					{
						// reuse these buffers for the next vertical communication
						inds = new IT [ buffpercolneigh * colneighs ];
						vals = new NT [ buffpercolneigh * colneighs ];
					}
				} // end_if for "send buffer is full" case
			} // end_for for "cnz < entriestoread" case
			assert (cnz == total_nnz);
			
			// Signal the end of file to other processors along the column
			fill_n(ccurptrs, colneighs, numeric_limits<int>::max());
			MPI_Scatter(ccurptrs, 1, MPI_INT, &recvcount, 1, MPI_INT, rankincol, commGrid->GetColWorld());
            
			// And along the row ...
			fill_n(rcurptrs, rowneighs, numeric_limits<int>::max());
			MPI_Scatter(rcurptrs, 1, MPI_INT, &recvcount, 1, MPI_INT, rankinrow, commGrid->GetRowWorld());
		}	// end of "else" (only one processor reads) block
	}	// end_if for "master processor" case
	else if( commGrid->OnSameProcCol(master) ) 	// (r-1) processors
	{
		BcastEssentials(commGrid->GetWorld(), total_m, total_n, total_nnz, master);
        
		if(seeklength > 0 && pario)   // these processors also do parallel binary i/o
		{
			binfile = fopen(filename.c_str(), "rb");
			IT entrysize = handler.entrylength();
			int myrankincol = commGrid->GetRankInProcCol();
			IT perreader = total_nnz / colneighs;
			IT read_offset = entrysize * static_cast<IT>(myrankincol) * perreader + seeklength;
			IT entriestoread = perreader;
			if (myrankincol == colneighs-1)
				entriestoread = total_nnz - static_cast<IT>(myrankincol) * perreader;
			fseek(binfile, read_offset, SEEK_SET);
            
#ifdef IODEBUG
			ofstream oput;
			commGrid->OpenDebugFile("Read", oput);
			oput << "Total nnz: " << total_nnz << " OFFSET : " << read_offset << " entries to read: " << entriestoread << endl;
			oput.close();
#endif
			
			AllocateSetBuffers(inds, vals,  rcurptrs, ccurptrs, rowneighs, colneighs, buffpercolneigh);
			ReadAllMine(binfile, inds, vals, localpairs, rcurptrs, ccurptrs, rdispls, cdispls,
                        buffperrowneigh, buffpercolneigh, entriestoread, handler, rankinrow);
		}
		else // only master does the I/O
		{
			while(total_n > 0 || total_m > 0)	// otherwise input file does not exist !
            {
				
				MPI_Scatter(ccurptrs, 1, MPI_INT, &recvcount, 1, MPI_INT, rankincol, commGrid->GetColWorld());        // first receive the receive counts ...
				if( recvcount == numeric_limits<int>::max()) break;
                
				// create space for incoming data ...
				IT * tempinds = new IT[recvcount];
				NT * tempvals = new NT[recvcount];
				
				// receive actual data ... (first 4 arguments are ignored in the receiver side)
				MPI_Scatterv(inds, ccurptrs, cdispls, MPIType<IT>(), tempinds, recvcount,  MPIType<IT>(), rankincol, commGrid->GetColWorld());
				MPI_Scatterv(vals, ccurptrs, cdispls, MPIType<NT>(), tempvals, recvcount,  MPIType<NT>(), rankincol, commGrid->GetColWorld());
                
				// now, send the data along the horizontal
				rcurptrs = new int[rowneighs]();    // zero initialized via ()
				
				// HorizontalSend frees the memory of temp_xxx arrays and then creates and frees memory of all the four arrays itself
				HorizontalSend(inds, vals, tempinds, tempvals, localpairs, rcurptrs, rdispls,
                               buffperrowneigh, recvcount, rankinrow);
			}
		}
		
		// Signal the end of file to other processors along the row
		fill_n(rcurptrs, rowneighs, numeric_limits<int>::max());
		MPI_Scatter(rcurptrs, 1, MPI_INT, &recvcount, 1, MPI_INT, rankinrow, commGrid->GetRowWorld());
		delete [] rcurptrs;
	}
	else		// r * (s-1) processors that only participate in the horizontal communication step
	{
		BcastEssentials(commGrid->GetWorld(), total_m, total_n, total_nnz, master);
		while(total_n > 0 || total_m > 0)	// otherwise input file does not exist !
		{
			// receive the receive count
			MPI_Scatter(rcurptrs, 1, MPI_INT, &recvcount, 1, MPI_INT, rankinrow, commGrid->GetRowWorld());
			if( recvcount == numeric_limits<int>::max())
				break;
            
			// create space for incoming data ...
			IT * tempinds = new IT[recvcount];
			NT * tempvals = new NT[recvcount];
            
			MPI_Scatterv(inds, rcurptrs, rdispls, MPIType<IT>(), tempinds, recvcount,  MPIType<IT>(), rankinrow, commGrid->GetRowWorld());
			MPI_Scatterv(vals, rcurptrs, rdispls, MPIType<NT>(), tempvals, recvcount,  MPIType<NT>(), rankinrow, commGrid->GetRowWorld());
            
            for(int i=0; i< recvcount; ++i)
            {
                localpairs.push_back( 	make_pair(tempinds[i], tempvals[i]) );  // no need to offset-adjust as we received the local index
            }
            
            DeleteAll(tempinds, tempvals);
		}
	}
    // ABAB: Sort localpairs and copy the sorted result to ind/num
#ifdef IODEBUG
    ofstream oput;
    commGrid->OpenDebugFile("Read", oput);
    for(auto litr = localpairs.begin(); litr != localpairs.end(); litr++)
    {
        oput << "(" << litr->first << "," <<  litr->second << ")" << endl;
    }
    oput.close();
#endif
    
}


//! Called on an existing object
//! ABAB: Obsolete, will be deleted once moved to Github (and becomes independent of KDT)
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
	MPI_File_write(thefile, packed, count, datatype, MPI_STATUS_IGNORE);
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
 ** the length of vout is globallen, which must be greater than the maximum entry of vin.
 ** nnz(vin) = nnz(vout)
 ** for every nonzero entry vin[k]: vout[vin[k]] = k
 */
/*
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
    
    IT lastIndex=-1;
    for(typename vector<pair<IT,NT>>::iterator itr = tosort.begin(); itr != tosort.end(); ++itr)
    {
        if(lastIndex!=itr->first) // avoid duplicate indices
        {
            Inverted.ind.push_back(itr->first);
            Inverted.num.push_back(itr->second);
        }
        lastIndex = itr->first;
        
	}
	return Inverted;
    
}
*/



/*
 ** Create a new sparse vector vout by swaping the indices and values of a sparse vector vin.
 ** the length of vout is globallen, which must be greater than the maximum entry of vin.
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
	
    MPI_Comm World = commGrid->GetWorld();
	int nprocs = commGrid->GetSize();
    int * rdispls = new int[nprocs+1];
	int * recvcnt = new int[nprocs];
    int * sendcnt = new int[nprocs](); // initialize to 0
	int * sdispls = new int[nprocs+1];
    
    
	IT ploclen = getlocnnz();
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for(IT k=0; k < ploclen; ++k)
	{
		IT locind;
		int owner = Inverted.Owner(num[k], locind);
#ifdef _OPENMP
        __sync_fetch_and_add(&sendcnt[owner], 1);
#else
        sendcnt[owner]++;
#endif
	}
    

   	MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, World);  // share the request counts
    
    sdispls[0] = 0;
	rdispls[0] = 0;
	for(int i=0; i<nprocs; ++i)
	{
		sdispls[i+1] = sdispls[i] + sendcnt[i];
		rdispls[i+1] = rdispls[i] + recvcnt[i];
	}
    
    
    
    NT * datbuf = new NT[ploclen];
    IT * indbuf = new IT[ploclen];
    int *count = new int[nprocs](); //current position
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(IT i=0; i < ploclen; ++i)
	{
		IT locind;
        int owner = Inverted.Owner(num[i], locind);
        int id;
#ifdef _OPENMP
        id = sdispls[owner] + __sync_fetch_and_add(&count[owner], 1);
#else
        id = sdispls[owner] + count[owner];
        count[owner]++;
#endif
        datbuf[id] = ind[i] + LengthUntil();
        indbuf[id] = locind;
	}
    delete [] count;
    

    IT totrecv = rdispls[nprocs];
	NT * recvdatbuf = new NT[totrecv];
	MPI_Alltoallv(datbuf, sendcnt, sdispls, MPIType<NT>(), recvdatbuf, recvcnt, rdispls, MPIType<NT>(), World);
    delete [] datbuf;
    
    IT * recvindbuf = new IT[totrecv];
    MPI_Alltoallv(indbuf, sendcnt, sdispls, MPIType<IT>(), recvindbuf, recvcnt, rdispls, MPIType<IT>(), World);
    delete [] indbuf;

    
    vector< pair<IT,NT> > tosort;
    tosort.resize(totrecv);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for(int i=0; i<totrecv; ++i)
	{
        tosort[i] = make_pair(recvindbuf[i], recvdatbuf[i]);
	}
	DeleteAll(recvindbuf, recvdatbuf);
    DeleteAll(sdispls, rdispls, sendcnt, recvcnt);
    
#if defined(GNU_PARALLEL) && defined(_OPENMP)
    __gnu_parallel::sort(tosort.begin(), tosort.end());
#else
    std::sort(tosort.begin(), tosort.end());
#endif
    
    Inverted.ind.reserve(totrecv);
    Inverted.num.reserve(totrecv);
    IT lastIndex=-1;
    
    // not threaded because Inverted.ind is kept sorted
    for(typename vector<pair<IT,NT>>::iterator itr = tosort.begin(); itr != tosort.end(); ++itr)
    {
        if(lastIndex!=itr->first) // avoid duplicate indices
        {
            Inverted.ind.push_back(itr->first);
            Inverted.num.push_back(itr->second);
        }
        lastIndex = itr->first;
        
	}
    
	return Inverted;
    
}



/*
 // generalized invert taking binary operations to define index and values of the inverted vector
 */

template <class IT, class NT>
template <typename _BinaryOperationIdx, typename _BinaryOperationVal>
FullyDistSpVec<IT,NT> FullyDistSpVec<IT,NT>::Invert (IT globallen, _BinaryOperationIdx __binopIdx, _BinaryOperationVal __binopVal)

{
    
    FullyDistSpVec<IT,NT> Inverted(commGrid, globallen);
    
    
    // identify the max index in the composed vector
    IT localmax = (IT) 0;
    for(IT k=0; k < num.size(); ++k)
    {
        localmax = std::max(localmax, __binopIdx(num[k], ind[k] + LengthUntil()));
    }
    IT globalmax = (IT) 0;
    MPI_Allreduce( &localmax, &globalmax, 1, MPIType<IT>(), MPI_MAX, commGrid->GetWorld());
    
    if(globalmax >= globallen)
    {
        cout << "Sparse vector has entries (" << globalmax  << ") larger than requested global vector length " << globallen << endl;
        return Inverted;
    }
    

    
    MPI_Comm World = commGrid->GetWorld();
    int nprocs = commGrid->GetSize();
    int * rdispls = new int[nprocs+1];
    int * recvcnt = new int[nprocs];
    int * sendcnt = new int[nprocs](); // initialize to 0
    int * sdispls = new int[nprocs+1];
    
    
    IT ploclen = getlocnnz();
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(IT k=0; k < ploclen; ++k)
    {
        IT locind;
        IT globind = __binopIdx(num[k], ind[k] + LengthUntil()); // get global index of the inverted vector
        int owner = Inverted.Owner(globind, locind);
        
#ifdef _OPENMP
        __sync_fetch_and_add(&sendcnt[owner], 1);
#else
        sendcnt[owner]++;
#endif
    }
    
    
    MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, World);
    
    sdispls[0] = 0;
    rdispls[0] = 0;
    for(int i=0; i<nprocs; ++i)
    {
        sdispls[i+1] = sdispls[i] + sendcnt[i];
        rdispls[i+1] = rdispls[i] + recvcnt[i];
    }
    
    
    NT * datbuf = new NT[ploclen];
    IT * indbuf = new IT[ploclen];
    int *count = new int[nprocs](); //current position
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(IT i=0; i < ploclen; ++i)
    {
        IT locind;
        IT globind = __binopIdx(num[i], ind[i] + LengthUntil()); // get global index of the inverted vector
        int owner = Inverted.Owner(globind, locind);
        int id;
#ifdef _OPENMP
        id = sdispls[owner] + __sync_fetch_and_add(&count[owner], 1);
#else
        id = sdispls[owner] + count[owner];
        count[owner]++;
#endif
        datbuf[id] = __binopVal(num[i], ind[i] + LengthUntil());
        indbuf[id] = locind;
    }
    delete [] count;
    
    
    IT totrecv = rdispls[nprocs];
    NT * recvdatbuf = new NT[totrecv];
    MPI_Alltoallv(datbuf, sendcnt, sdispls, MPIType<NT>(), recvdatbuf, recvcnt, rdispls, MPIType<NT>(), World);
    delete [] datbuf;
    
    IT * recvindbuf = new IT[totrecv];
    MPI_Alltoallv(indbuf, sendcnt, sdispls, MPIType<IT>(), recvindbuf, recvcnt, rdispls, MPIType<IT>(), World);
    delete [] indbuf;
    
    
    vector< pair<IT,NT> > tosort;
    tosort.resize(totrecv);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<totrecv; ++i)
    {
        tosort[i] = make_pair(recvindbuf[i], recvdatbuf[i]);
    }
    DeleteAll(recvindbuf, recvdatbuf);
    DeleteAll(sdispls, rdispls, sendcnt, recvcnt);
    
#if defined(GNU_PARALLEL) && defined(_OPENMP)
    __gnu_parallel::sort(tosort.begin(), tosort.end());
#else
    std::sort(tosort.begin(), tosort.end());
#endif
    
    Inverted.ind.reserve(totrecv);
    Inverted.num.reserve(totrecv);
    IT lastIndex=-1;
    
    // not threaded because Inverted.ind is kept sorted
    for(typename vector<pair<IT,NT>>::iterator itr = tosort.begin(); itr != tosort.end(); ++itr)
    {
        if(lastIndex!=itr->first) // avoid duplicate indices
        {
            Inverted.ind.push_back(itr->first);
            Inverted.num.push_back(itr->second);
        }
        lastIndex = itr->first;
        
    }
    
    return Inverted;
    
}


// Invert using RMA

template <class IT, class NT>
template <typename _BinaryOperationIdx, typename _BinaryOperationVal>
FullyDistSpVec<IT,NT> FullyDistSpVec<IT,NT>::InvertRMA (IT globallen, _BinaryOperationIdx __binopIdx, _BinaryOperationVal __binopVal)

{
    
    FullyDistSpVec<IT,NT> Inverted(commGrid, globallen);
    int myrank;
    MPI_Comm_rank(commGrid->GetWorld(), &myrank);
    
    // identify the max index in the composed vector
    IT localmax = (IT) 0;
    for(IT k=0; k < num.size(); ++k)
    {
        localmax = std::max(localmax, __binopIdx(num[k], ind[k] + LengthUntil()));
    }
    IT globalmax = (IT) 0;
    MPI_Allreduce( &localmax, &globalmax, 1, MPIType<IT>(), MPI_MAX, commGrid->GetWorld());
    
    if(globalmax >= globallen)
    {
        cout << "Sparse vector has entries (" << globalmax  << ") larger than requested global vector length " << globallen << endl;
        return Inverted;
    }
    
    
    
    MPI_Comm World = commGrid->GetWorld();
    int nprocs = commGrid->GetSize();
    int * rdispls = new int[nprocs+1];
    int * recvcnt = new int[nprocs](); // initialize to 0
    int * sendcnt = new int[nprocs](); // initialize to 0
    int * sdispls = new int[nprocs+1];
    
    
    IT ploclen = getlocnnz();
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(IT k=0; k < ploclen; ++k)
    {
        IT locind;
        IT globind = __binopIdx(num[k], ind[k] + LengthUntil()); // get global index of the inverted vector
        int owner = Inverted.Owner(globind, locind);
        
#ifdef _OPENMP
        __sync_fetch_and_add(&sendcnt[owner], 1);
#else
        sendcnt[owner]++;
#endif
    }
    
    
    MPI_Win win2;
    MPI_Win_create(recvcnt, nprocs * sizeof(MPI_INT), sizeof(MPI_INT), MPI_INFO_NULL, World, &win2);
    for(int i=0; i<nprocs; ++i)
    {
        if(sendcnt[i]>0)
        {
            MPI_Win_lock(MPI_LOCK_SHARED,i,MPI_MODE_NOCHECK,win2);
            MPI_Put(&sendcnt[i], 1, MPI_INT, i, myrank, 1, MPI_INT, win2);
            MPI_Win_unlock(i, win2);
        }
    }
    MPI_Win_free(&win2);
    
    
    
    sdispls[0] = 0;
    rdispls[0] = 0;
    for(int i=0; i<nprocs; ++i)
    {
        sdispls[i+1] = sdispls[i] + sendcnt[i];
        rdispls[i+1] = rdispls[i] + recvcnt[i];
    }
    
    int * rmadispls = new int[nprocs+1];
    MPI_Win win3;
    MPI_Win_create(rmadispls, nprocs * sizeof(MPI_INT), sizeof(MPI_INT), MPI_INFO_NULL, World, &win3);
    for(int i=0; i<nprocs; ++i)
    {
        if(recvcnt[i]>0)
        {
            MPI_Win_lock(MPI_LOCK_SHARED,i,MPI_MODE_NOCHECK,win3);
            MPI_Put(&rdispls[i], 1, MPI_INT, i, myrank, 1, MPI_INT, win3);
            MPI_Win_unlock(i, win3);
        }
    }
    MPI_Win_free(&win3);

    
    NT * datbuf = new NT[ploclen];
    IT * indbuf = new IT[ploclen];
    int *count = new int[nprocs](); //current position
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(IT i=0; i < ploclen; ++i)
    {
        IT locind;
        IT globind = __binopIdx(num[i], ind[i] + LengthUntil()); // get global index of the inverted vector
        int owner = Inverted.Owner(globind, locind);
        int id;
#ifdef _OPENMP
        id = sdispls[owner] + __sync_fetch_and_add(&count[owner], 1);
#else
        id = sdispls[owner] + count[owner];
        count[owner]++;
#endif
        datbuf[id] = __binopVal(num[i], ind[i] + LengthUntil());
        indbuf[id] = locind;
    }
    delete [] count;
    
    
    IT totrecv = rdispls[nprocs];
    NT * recvdatbuf = new NT[totrecv];
    IT * recvindbuf = new IT[totrecv];
    MPI_Win win, win1;
    MPI_Win_create(recvdatbuf, totrecv * sizeof(NT), sizeof(NT), MPI_INFO_NULL, commGrid->GetWorld(), &win);
    MPI_Win_create(recvindbuf, totrecv * sizeof(IT), sizeof(IT), MPI_INFO_NULL, commGrid->GetWorld(), &win1);
    //MPI_Win_fence(0, win);
    //MPI_Win_fence(0, win1);
    for(int i=0; i<nprocs; ++i)
    {
        if(sendcnt[i]>0)
        {
            MPI_Win_lock(MPI_LOCK_SHARED, i, 0, win);
            MPI_Put(&datbuf[sdispls[i]], sendcnt[i], MPIType<NT>(), i, rmadispls[i], sendcnt[i], MPIType<NT>(), win);
            MPI_Win_unlock(i, win);
            
            MPI_Win_lock(MPI_LOCK_SHARED, i, 0, win1);
            MPI_Put(&indbuf[sdispls[i]], sendcnt[i], MPIType<IT>(), i, rmadispls[i], sendcnt[i], MPIType<IT>(), win1);
            MPI_Win_unlock(i, win1);
        }
    }
    //MPI_Win_fence(0, win);
    //MPI_Win_fence(0, win1);
    MPI_Win_free(&win);
    MPI_Win_free(&win1);
    
    delete [] datbuf;
    delete [] indbuf;
    delete [] rmadispls;
    
    
    vector< pair<IT,NT> > tosort;
    tosort.resize(totrecv);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<totrecv; ++i)
    {
        tosort[i] = make_pair(recvindbuf[i], recvdatbuf[i]);
    }
    DeleteAll(recvindbuf, recvdatbuf);
    DeleteAll(sdispls, rdispls, sendcnt, recvcnt);
    
#if defined(GNU_PARALLEL) && defined(_OPENMP)
    __gnu_parallel::sort(tosort.begin(), tosort.end());
#else
    std::sort(tosort.begin(), tosort.end());
#endif
    
    Inverted.ind.reserve(totrecv);
    Inverted.num.reserve(totrecv);
    IT lastIndex=-1;
    
    // not threaded because Inverted.ind is kept sorted
    for(typename vector<pair<IT,NT>>::iterator itr = tosort.begin(); itr != tosort.end(); ++itr)
    {
        if(lastIndex!=itr->first) // avoid duplicate indices
        {
            Inverted.ind.push_back(itr->first);
            Inverted.num.push_back(itr->second);
        }
        lastIndex = itr->first;
        
    }
    
    return Inverted;
    
}



template <typename IT, typename NT>
template <typename NT1, typename _UnaryOperation>
void FullyDistSpVec<IT,NT>::Select (const FullyDistVec<IT,NT1> & denseVec, _UnaryOperation __unop)
{
	if(*commGrid == *(denseVec.commGrid))
	{
		if(TotalLength() != denseVec.TotalLength())
		{
			ostringstream outs;
			outs << "Vector dimensions don't match (" << TotalLength() << " vs " << denseVec.TotalLength() << ") for Select\n";
			SpParHelper::Print(outs.str());
			MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
		}
		else
		{

			IT spsize = getlocnnz();
            IT k = 0;
            // iterate over the sparse vector
            for(IT i=0; i< spsize; ++i)
            {
                if(__unop(denseVec.arr[ind[i]]))
                {
                    ind[k] = ind[i];
                    num[k++] = num[i];
                }
            }
            ind.resize(k);
            num.resize(k);
		}
	}
	else
	{
        ostringstream outs;
        outs << "Grids are not comparable for Select" << endl;
        SpParHelper::Print(outs.str());
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
	}
}


// \todo: Shouldn't this wrap EWiseApply for code maintanence instead?
template <typename IT, typename NT>
template <typename NT1>
void FullyDistSpVec<IT,NT>::Setminus (const FullyDistSpVec<IT,NT1> & other)
{
    if(*commGrid == *(other.commGrid))
    {
        if(TotalLength() != other.TotalLength())
        {
            ostringstream outs;
            outs << "Vector dimensions don't match (" << TotalLength() << " vs " << other.TotalLength() << ") for Select\n";
            SpParHelper::Print(outs.str());
            MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
        }
        else
        {
            
            IT mysize = getlocnnz();
            IT othersize = other.getlocnnz();
            IT k = 0, i=0, j=0;
            // iterate over the sparse vector
            for(; i< mysize && j < othersize;)
            {
                if(other.ind[j] == ind[i]) //skip
                {
                    i++; j++;
                }
                else if(other.ind[j] > ind[i])
                {
                    ind[k] = ind[i];
                    num[k++] = num[i++];
                }
                else j++;
            }
            while(i< mysize)
            {
                ind[k] = ind[i];
                num[k++] = num[i++];
            }

            ind.resize(k);
            num.resize(k);
        }
    }
    else
    {
        ostringstream outs;
        outs << "Grids are not comparable for Select" << endl;
        SpParHelper::Print(outs.str());
        MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
    }
}





//
template <typename IT, typename NT>
template <typename NT1, typename _UnaryOperation1, typename _UnaryOperation2>
FullyDistSpVec<IT,NT1> FullyDistSpVec<IT,NT>::SelectNew (const FullyDistVec<IT,NT1> & denseVec, _UnaryOperation1 __unop1, _UnaryOperation2 __unop2)
{
    FullyDistSpVec<IT,NT1> composed(commGrid, TotalLength());
	if(*commGrid == *(denseVec.commGrid))
	{
		if(TotalLength() != denseVec.TotalLength())
		{
			ostringstream outs;
			outs << "Vector dimensions don't match (" << TotalLength() << " vs " << denseVec.TotalLength() << ") for Select\n";
			SpParHelper::Print(outs.str());
			MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
		}
		else
		{
            
			IT spsize = getlocnnz();
            //IT k = 0;
            // iterate over the sparse vector
            for(IT i=0; i< spsize; ++i)
            {
                if(__unop1(denseVec.arr[ind[i]]))
                {
                    composed.ind.push_back(ind[i]);
                    composed.num.push_back(__unop2(num[i]));
                    //ind[k] = ind[i];
                    //num[k++] = num[i];
                }
            }
            //ind.resize(k);
            //num.resize(k);
		}
	}
	else
	{
        ostringstream outs;
        outs << "Grids are not comparable for Select" << endl;
        SpParHelper::Print(outs.str());
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
	}
    
    return composed;
}




template <typename IT, typename NT>
template <typename NT1, typename _UnaryOperation, typename _BinaryOperation>
void FullyDistSpVec<IT,NT>::SelectApply (const FullyDistVec<IT,NT1> & denseVec, _UnaryOperation __unop, _BinaryOperation __binop)
{
	if(*commGrid == *(denseVec.commGrid))
	{
		if(TotalLength() != denseVec.TotalLength())
		{
			ostringstream outs;
			outs << "Vector dimensions don't match (" << TotalLength() << " vs " << denseVec.TotalLength() << ") for Select\n";
			SpParHelper::Print(outs.str());
			MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
		}
		else
		{
            
			IT spsize = getlocnnz();
            IT k = 0;
            // iterate over the sparse vector
            for(IT i=0; i< spsize; ++i)
            {
                if(__unop(denseVec.arr[ind[i]]))
                {
                    ind[k] = ind[i];
                    num[k++] = __binop(num[i], denseVec.arr[ind[i]]);
                }
            }
            ind.resize(k);
            num.resize(k);
		}
	}
	else
	{
        ostringstream outs;
        outs << "Grids are not comparable for Select" << endl;
        SpParHelper::Print(outs.str());
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
	}
}




template <typename IT, typename NT>
template <typename NT1, typename _UnaryOperation, typename _BinaryOperation>
FullyDistSpVec<IT,NT> FullyDistSpVec<IT,NT>::SelectApplyNew(const FullyDistVec<IT,NT1> & denseVec, _UnaryOperation __unop, _BinaryOperation __binop)
{
    FullyDistSpVec<IT,NT> composed(commGrid, TotalLength());
	if(*commGrid == *(denseVec.commGrid))
	{
		if(TotalLength() != denseVec.TotalLength())
		{
			ostringstream outs;
			outs << "Vector dimensions don't match (" << TotalLength() << " vs " << denseVec.TotalLength() << ") for Select\n";
			SpParHelper::Print(outs.str());
			MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
		}
		else
		{
            
			IT spsize = getlocnnz();
            //IT k = 0;
            // iterate over the sparse vector
            for(IT i=0; i< spsize; ++i)
            {
                if(__unop(denseVec.arr[ind[i]]))
                {
                    composed.ind.push_back(ind[i]);
                    composed.num.push_back( __binop(num[i], denseVec.arr[ind[i]]));
                }
            }
            //ind.resize(k);
            //num.resize(k);
		}
	}
	else
	{
        ostringstream outs;
        outs << "Grids are not comparable for Select" << endl;
        SpParHelper::Print(outs.str());
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
	}
    return composed;
}


// apply an unary function to each nnz and return a new vector
// can be a constrauctor
/*
template <typename IT, typename NT>
template <typename NT1, typename _UnaryOperation>
FullyDistSpVec<IT,NT1> FullyDistSpVec<IT,NT>::Apply(_UnaryOperation __unop)
{
    FullyDistSpVec<IT,NT1> composed(commGrid, TotalLength());
    IT spsize = getlocnnz();
    for(IT i=0; i< spsize; ++i)
    {
        composed.ind.push_back(ind[i]);
        composed.num.push_back( __unop(num[i]));
    }
    return composed;
}
*/



/* exp version
  */
template <class IT, class NT>
template <typename _UnaryOperation>
void FullyDistSpVec<IT,NT>::FilterByVal (FullyDistSpVec<IT,IT> Selector, _UnaryOperation __unop, bool filterByIndex)
{
    if(*commGrid != *(Selector.commGrid))
    {
        ostringstream outs;
        outs << "Grids are not comparable for Filter" << endl;
        SpParHelper::Print(outs.str());
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
    }
    int nprocs = commGrid->GetSize();
    MPI_Comm World = commGrid->GetWorld();
	
    
	int * rdispls = new int[nprocs];
    int sendcnt = Selector.ind.size();
    int * recvcnt = new int[nprocs];
    MPI_Allgather(&sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, World);
    
	rdispls[0] = 0;
	for(int i=0; i<nprocs-1; ++i)
	{
		rdispls[i+1] = rdispls[i] + recvcnt[i];
	}
    
    
    IT * sendbuf = new IT[sendcnt];
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int k=0; k<sendcnt; k++)
    {
        if(filterByIndex)
            sendbuf[k] = Selector.ind[k] + Selector.LengthUntil();
        else
            sendbuf[k] = Selector.num[k];
    }
    
    IT totrecv = accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));
    
    std::vector<IT> recvbuf;
    recvbuf.resize(totrecv);
    
    MPI_Allgatherv(sendbuf, sendcnt, MPIType<IT>(), recvbuf.data(), recvcnt, rdispls, MPIType<IT>(), World);
    delete [] sendbuf;
    DeleteAll(rdispls,recvcnt);
    
     if(!filterByIndex) // need to sort
     {
#if defined(GNU_PARALLEL) && defined(_OPENMP)
    __gnu_parallel::sort(recvbuf.begin(), recvbuf.end());
#else
    std::sort(recvbuf.begin(), recvbuf.end());
#endif
     }
    
    // now perform filter (recvbuf is sorted) // TODO: OpenMP parallel and keep things sorted
    IT k=0;
    
    for(IT i=0; i<num.size(); i++)
    {
        IT val = __unop(num[i]);
        if(!std::binary_search(recvbuf.begin(), recvbuf.end(), val))
        {
            ind[k] = ind[i];
            num[k++] = num[i];
        }
    }
    ind.resize(k);
    num.resize(k);
 
    
}



