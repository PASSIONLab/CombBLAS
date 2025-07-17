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


#include <limits>
#include "FullyDistSpVec.h"
#include "SpDefs.h"
#include "SpHelper.h"
#include "hash.hpp"
#include "FileHeader.h"
#include <sys/types.h>
#include <sys/stat.h>

#ifdef GNU_PARALLEL
#include <parallel/algorithm>
#include <parallel/numeric>
#endif

#include "usort/parUtils.h"

namespace combblas {

template <class IT, class NT>
FullyDistSpVec<IT, NT>::FullyDistSpVec ( std::shared_ptr<CommGrid> grid)
: FullyDist<IT,NT,typename combblas::disable_if< combblas::is_boolean<NT>::value, NT >::type>(grid)
{ };

template <class IT, class NT>
FullyDistSpVec<IT, NT>::FullyDistSpVec ( std::shared_ptr<CommGrid> grid, IT globallen)
: FullyDist<IT,NT,typename combblas::disable_if< combblas::is_boolean<NT>::value, NT >::type>(grid,globallen)
{ };

template <class IT, class NT>
FullyDistSpVec<IT,NT>::FullyDistSpVec ()
: FullyDist<IT,NT,typename combblas::disable_if< combblas::is_boolean<NT>::value, NT >::type>()
{ };

template <class IT, class NT>
FullyDistSpVec<IT,NT>::FullyDistSpVec (IT globallen)
: FullyDist<IT,NT,typename combblas::disable_if< combblas::is_boolean<NT>::value, NT >::type>(globallen)
{ }


template <class IT, class NT>
FullyDistSpVec<IT,NT> &  FullyDistSpVec<IT,NT>::operator=(const FullyDistSpVec< IT,NT > & rhs)
{
	if(this != &rhs)
	{
		FullyDist<IT,NT,typename combblas::disable_if< combblas::is_boolean<NT>::value, NT >::type>::operator= (rhs);	// to update glen and commGrid
		ind = rhs.ind;
		num = rhs.num;
	}
	return *this;
}

template <class IT, class NT>
FullyDistSpVec<IT,NT>::FullyDistSpVec (const FullyDistVec<IT,NT> & rhs) // Conversion copy-constructor
: FullyDist<IT,NT,typename combblas::disable_if< combblas::is_boolean<NT>::value, NT >::type>(rhs.commGrid,rhs.glen)
{
	*this = rhs;
}

// Conversion copy-constructor where unary op is true
template <class IT, class NT>
template <typename _UnaryOperation>
FullyDistSpVec<IT,NT>::FullyDistSpVec (const FullyDistVec<IT,NT> & rhs, _UnaryOperation unop)
: FullyDist<IT,NT,typename combblas::disable_if< combblas::is_boolean<NT>::value, NT >::type>(rhs.commGrid,rhs.glen)
{
	//FullyDist<IT,NT,typename combblas::disable_if< combblas::is_boolean<NT>::value, NT >::type>::operator= (rhs);	// to update glen and commGrid

	std::vector<IT>().swap(ind);
	std::vector<NT>().swap(num);
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



// create a sparse vector from local vectors
template <class IT, class NT>
FullyDistSpVec<IT,NT>::FullyDistSpVec (std::shared_ptr<CommGrid> grid, IT globallen, const std::vector<IT>& indvec, const std::vector<NT> & numvec, bool SumDuplicates, bool sorted)
: FullyDist<IT,NT,typename combblas::disable_if< combblas::is_boolean<NT>::value, NT >::type>(grid, globallen)
{

    assert(indvec.size()==numvec.size());
    IT vecsize = indvec.size();
    if(!sorted)
    {
        std::vector< std::pair<IT,NT> > tosort(vecsize);
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(IT i=0; i<vecsize; ++i)
        {
            tosort[i] = std::make_pair(indvec[i], numvec[i]);
        }

#if defined(GNU_PARALLEL) && defined(_OPENMP)
        __gnu_parallel::sort(tosort.begin(), tosort.end());
#else
        std::sort(tosort.begin(), tosort.end());
#endif


        ind.reserve(vecsize);
        num.reserve(vecsize);
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
    else
    {

        ind.reserve(vecsize);
        num.reserve(vecsize);
        IT lastIndex=-1;

        for(IT i=0; i< vecsize; ++i)
        {
            if(lastIndex!=indvec[i]) //if SumDuplicates=false, keep only the first one
            {
                ind.push_back(indvec[i]);
                num.push_back(numvec[i]);
                lastIndex = indvec[i];
            }
            else if(SumDuplicates)
            {
                num.back() += numvec[i];
            }
        }
    }
}



// ABAB: This function probably operates differently than a user would immediately expect
// ABAB: Write a well-posed description for it
template <class IT, class NT>
FullyDistSpVec<IT,NT> &  FullyDistSpVec<IT,NT>::operator=(const FullyDistVec< IT,NT > & rhs)		// conversion from dense
{
	FullyDist<IT,NT,typename combblas::disable_if< combblas::is_boolean<NT>::value, NT >::type>::operator= (rhs);	// to update glen and commGrid

	std::vector<IT>().swap(ind);
	std::vector<NT>().swap(num);
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
: FullyDist<IT,NT,typename combblas::disable_if< combblas::is_boolean<NT>::value, NT >::type>(inds.commGrid,globallen)
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
    IT totrecv = std::accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));


    // ----- Send and receive indices and values --------

    NT * recvdatbuf = new NT[totrecv];
    MPI_Alltoallv(datbuf, sendcnt, sdispls, MPIType<NT>(), recvdatbuf, recvcnt, rdispls, MPIType<NT>(), World);
    delete [] datbuf;

    IT * recvindbuf = new IT[totrecv];
    MPI_Alltoallv(indbuf, sendcnt, sdispls, MPIType<IT>(), recvindbuf, recvcnt, rdispls, MPIType<IT>(), World);
    delete [] indbuf;


    // ------ merge and sort received data ----------

    std::vector< std::pair<IT,NT> > tosort;
    tosort.resize(totrecv);
    for(int i=0; i<totrecv; ++i)
    {
        tosort[i] = std::make_pair(recvindbuf[i], recvdatbuf[i]);
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
    IT lengthuntil = std::accumulate(dist, dist+rank, static_cast<IT>(0));
    found.glen = std::accumulate(dist, dist+nprocs, static_cast<IT>(0));

    // Although the found vector is not reshuffled yet, its glen and commGrid are set
    // We can call the Owner/MyLocLength/LengthUntil functions (to infer future distribution)

    // rebalance/redistribute
    int * sendcnt = new int[nprocs];
    std::fill(sendcnt, sendcnt+nprocs, 0);
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
    IT totrecv = std::accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));
    std::vector<NT> recvbuf(totrecv);

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
    IT lengthuntil = std::accumulate(dist, dist+rank, static_cast<IT>(0));
    found.glen = std::accumulate(dist, dist+nprocs, static_cast<IT>(0));

    // Although the found vector is not reshuffled yet, its glen and commGrid are set
    // We can call the Owner/MyLocLength/LengthUntil functions (to infer future distribution)

    // rebalance/redistribute
    int * sendcnt = new int[nprocs];
    std::fill(sendcnt, sendcnt+nprocs, 0);
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
    IT totrecv = std::accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));
    std::vector<IT> recvbuf(totrecv);

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
	FullyDist<IT,NT,typename combblas::disable_if< combblas::is_boolean<NT>::value, NT >::type>::operator= (victim);	// to update glen and commGrid
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
		typename std::vector<IT>::const_iterator it = std::lower_bound(ind.begin(), ind.end(), locind);	// ind is a sorted vector
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
	typename std::vector<IT>::const_iterator it = std::lower_bound(ind.begin(), ind.end(), locind);	// ind is a sorted vector
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
		typename std::vector<IT>::iterator iter = std::lower_bound(ind.begin(), ind.end(), locind);
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
		typename std::vector<IT>::iterator iter = std::lower_bound(ind.begin(), ind.end(), locind);
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
	std::unordered_map<IT, IT> revr_map;       // inverted index that maps indices of *this to indices of output
	std::vector< std::vector<IT> > data_req(nprocs);
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
                revr_map.insert(typename std::unordered_map<IT, IT>::value_type(locind, i));
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
	IT totrecv = std::accumulate(recvcnt,recvcnt+nprocs,0);
	IT * recvbuf = new IT[totrecv];

	for(int i=0; i<nprocs; ++i)
	{
    std::copy(data_req[i].begin(), data_req[i].end(), sendbuf+sdispls[i]);
		std::vector<IT>().swap(data_req[i]);
	}
	MPI_Alltoallv(sendbuf, sendcnt, sdispls, MPIType<IT>(), recvbuf, recvcnt, rdispls, MPIType<IT>(), World);  // request data

	// We will return the requested data,
	// our return can be at most as big as the request
	// and smaller if we are missing some elements
	IT * indsback = new IT[totrecv];
	NT * databack = new NT[totrecv];

	int * ddispls = new int[nprocs];
	std::copy(rdispls, rdispls+nprocs, ddispls);
	for(int i=0; i<nprocs; ++i)
	{
		// this is not the most efficient method because it scans ind vector nprocs = sqrt(p) times
		IT * it = std::set_intersection(recvbuf+rdispls[i], recvbuf+rdispls[i]+recvcnt[i], ind.begin(), ind.end(), indsback+rdispls[i]);
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
			typename std::unordered_map<IT,IT>::iterator it = revr_map.find(sendbuf[j]);
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


//! iota over existing nonzero entries
template <class IT, class NT>
void FullyDistSpVec<IT,NT>::nziota(NT first)
{
    std::iota(num.begin(), num.end(), NnzUntil() + first);	// global across processors
}

//! Returns the number of nonzeros until this processor
template <class IT, class NT>
IT FullyDistSpVec<IT,NT>::NnzUntil() const
{
    IT mynnz = ind.size();
    IT prevnnz = 0;
    MPI_Scan(&mynnz, &prevnnz, 1, MPIType<IT>(), MPI_SUM, commGrid->GetWorld());
    return (prevnnz - mynnz);
}



/* old version
// - sorts the entries with respect to nonzero values
// - ignores structural zeros
// - keeps the sparsity structure intact
// - returns a permutation representing the mapping from old to new locations
template <class IT, class NT>
FullyDistSpVec<IT, IT> FullyDistSpVec<IT, NT>::sort()
{
    MPI_Comm World = commGrid->GetWorld();
    FullyDistSpVec<IT,IT> temp(commGrid);
    if(getnnz()==0) return temp;
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
*/


/*
 TODO: This function is just a hack at this moment.
 The indices of the return vector is not correct.
 FIX this
 */
// - sorts the entries with respect to nonzero values
// - ignores structural zeros
// - keeps the sparsity structure intact
// - returns a permutation representing the mapping from old to new locations
template <class IT, class NT>
FullyDistSpVec<IT, IT> FullyDistSpVec<IT, NT>::sort()
{
	MPI_Comm World = commGrid->GetWorld();
	FullyDistSpVec<IT,IT> temp(commGrid);
    if(getnnz()==0) return temp;
	IT nnz = getlocnnz();
	std::pair<NT,IT> * vecpair = new std::pair<NT,IT>[nnz];



	int nprocs, rank;
	MPI_Comm_size(World, &nprocs);
	MPI_Comm_rank(World, &rank);

	IT * dist = new IT[nprocs];
	dist[rank] = nnz;
	MPI_Allgather(MPI_IN_PLACE, 1, MPIType<IT>(), dist, 1, MPIType<IT>(), World);
    IT until = LengthUntil();
#ifdef THREADED
#pragma omp parallel for
#endif
	for(IT i=0; i< nnz; ++i)
	{
		vecpair[i].first = num[i];	// we'll sort wrt numerical values
		vecpair[i].second = ind[i] + until;

	}
	std::vector<std::pair<NT,IT>> sorted = SpParHelper::KeyValuePSort(vecpair, nnz, dist, World);

    nnz = sorted.size();
    temp.num.resize(nnz);
    temp.ind.resize(nnz);

#ifdef THREADED
#pragma omp parallel for
#endif
	for(IT i=0; i< nnz; ++i)
	{
		//num[i] = sorted[i].first;	// sorted range (change the object itself)
		//nind[i] = ind[i];		// make sure the sparsity distribution is the same
		temp.num[i] = sorted[i].second;	// inverse permutation stored as numerical values
        temp.ind[i] = i; // we are not using this information at this moment
	}

	delete [] vecpair;
	delete [] dist;

	temp.glen = glen;
	return temp;
}



/*
// - sorts the entries with respect to nonzero values
// - ignores structural zeros
// - keeps the sparsity structure intact
// - returns a permutation representing the mapping from old to new locations
template <class IT, class NT>
FullyDistSpVec<IT, IT> FullyDistSpVec<IT, NT>::sort()
{
    MPI_Comm World = commGrid->GetWorld();
    FullyDistSpVec<IT,IT> temp(commGrid);
    if(getnnz()==0) return temp;
    IT nnz = getlocnnz();
    //pair<NT,IT> * vecpair = new pair<NT,IT>[nnz];
    vector<IndexHolder<NT>> in(nnz);
    //vector<IndexHolder<NT>> out;



    int nprocs, rank;
    MPI_Comm_size(World, &nprocs);
    MPI_Comm_rank(World, &rank);

    //IT * dist = new IT[nprocs];
    //dist[rank] = nnz;
    //MPI_Allgather(MPI_IN_PLACE, 1, MPIType<IT>(), dist, 1, MPIType<IT>(), World);
    IT until = LengthUntil();
    for(IT i=0; i< nnz; ++i)
    {
        //vecpair[i].first = num[i];	// we'll sort wrt numerical values
        //vecpair[i].second = ind[i] + until;

        in[i] = IndexHolder<NT>(num[i], static_cast<unsigned long>(ind[i] + until));
    }
    //SpParHelper::MemoryEfficientPSort(vecpair, nnz, dist, World);

    //MPI_Barrier(World);
    //cout << "before sorting " << in.size() << endl;
    par::sampleSort(in, World);
    //MPI_Barrier(World);
    //cout << "after sorting " << in.size() << endl;
    //MPI_Barrier(World);

    //vector< IT > nind(out.size());
    //vector< IT > nnum(out.size());

    temp.ind.resize(in.size());
    temp.num.resize(in.size());
    for(IT i=0; i< in.size(); ++i)
    {
        //num[i] = vecpair[i].first;	// sorted range (change the object itself)
        //nind[i] = ind[i];		// make sure the sparsity distribution is the same
        //nnum[i] = vecpair[i].second;	// inverse permutation stored as numerical values

        //num[i] = out[i].value;	// sorted range (change the object itself)
        //nind[i] = ind[i];		// make sure the sparsity distribution is the same
        //nnum[i] = static_cast<IT>(out[i].index);	// inverse permutation stored as numerical values
        temp.num[i] = static_cast<IT>(in[i].index);	// inverse permutation stored as numerical values
        //cout << temp.num[i] << " " ;
    }

    temp.glen = glen;
    return temp;
}
 */


template <class IT, class NT>
template <typename _BinaryOperation >
FullyDistSpVec<IT,NT> FullyDistSpVec<IT, NT>::UniqAll2All(_BinaryOperation __binary_op, MPI_Op mympiop)
{
    MPI_Comm World = commGrid->GetWorld();
	int nprocs = commGrid->GetSize();

    std::vector< std::vector< NT > > datsent(nprocs);
	std::vector< std::vector< IT > > indsent(nprocs);

    IT locind;
    size_t locvec = num.size();     // nnz in local vector
    IT lenuntil = LengthUntil();    // to convert to global index
	for(size_t i=0; i< locvec; ++i)
	{
        uint64_t myhash;    // output of MurmurHash3_x64_64 is 64-bits regardless of the input length
        MurmurHash3_x64_64((const void*) &(num[i]),sizeof(NT), 0, &myhash);
        double range = static_cast<double>(myhash) * static_cast<double>(glen);
        NT mapped = range / static_cast<double>(std::numeric_limits<uint64_t>::max());   // mapped is in range [0,n)
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
    std::copy(datsent[i].begin(), datsent[i].end(), datbuf+sdispls[i]);
		std::vector<NT>().swap(datsent[i]);
	}
    IT * indbuf = new IT[locvec];
    for(int i=0; i<nprocs; ++i)
	{
    std::copy(indsent[i].begin(), indsent[i].end(), indbuf+sdispls[i]);
		std::vector<IT>().swap(indsent[i]);
	}
    IT totrecv = std::accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));
	NT * recvdatbuf = new NT[totrecv];
	MPI_Alltoallv(datbuf, sendcnt, sdispls, MPIType<NT>(), recvdatbuf, recvcnt, rdispls, MPIType<NT>(), World);
    delete [] datbuf;

    IT * recvindbuf = new IT[totrecv];
    MPI_Alltoallv(indbuf, sendcnt, sdispls, MPIType<IT>(), recvindbuf, recvcnt, rdispls, MPIType<IT>(), World);
    delete [] indbuf;

    std::vector< std::pair<NT,IT> > tosort;   // in fact, tomerge would be a better name but it is unlikely to be faster

	for(int i=0; i<nprocs; ++i)
	{
		for(int j = rdispls[i]; j < rdispls[i] + recvcnt[i]; ++j)	// fetch the numerical values
		{
            tosort.push_back(std::make_pair(recvdatbuf[j], recvindbuf[j]));
		}
	}
	DeleteAll(recvindbuf, recvdatbuf);
    std::sort(tosort.begin(), tosort.end());
    //std::unique returns an iterator to the element that follows the last element not removed.
    typename std::vector< std::pair<NT,IT> >::iterator last;
    last = std::unique (tosort.begin(), tosort.end(), equal_first<NT,IT>());

    std::vector< std::vector< NT > > datback(nprocs);
	std::vector< std::vector< IT > > indback(nprocs);

    for(typename std::vector< std::pair<NT,IT> >::iterator itr = tosort.begin(); itr != last; ++itr)
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
		std::copy(datback[i].begin(), datback[i].end(), datbuf+sdispls[i]);
		std::vector<NT>().swap(datback[i]);
	}
    indbuf = new IT[tosort.size()];
    for(int i=0; i<nprocs; ++i)
	{
		std::copy(indback[i].begin(), indback[i].end(), indbuf+sdispls[i]);
		std::vector<IT>().swap(indback[i]);
	}
    totrecv = std::accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));   // update value

    recvdatbuf = new NT[totrecv];
	MPI_Alltoallv(datbuf, sendcnt, sdispls, MPIType<NT>(), recvdatbuf, recvcnt, rdispls, MPIType<NT>(), World);
    delete [] datbuf;

    recvindbuf = new IT[totrecv];
    MPI_Alltoallv(indbuf, sendcnt, sdispls, MPIType<IT>(), recvindbuf, recvcnt, rdispls, MPIType<IT>(), World);
    delete [] indbuf;

    FullyDistSpVec<IT,NT> Indexed(commGrid, glen);	// length(Indexed) = length(glen) = length(*this)

    std::vector< std::pair<IT,NT> > back2sort;
    for(int i=0; i<nprocs; ++i)
	{
		for(int j = rdispls[i]; j < rdispls[i] + recvcnt[i]; ++j)	// fetch the numerical values
		{
            back2sort.push_back(std::make_pair(recvindbuf[j], recvdatbuf[j]));
		}
	}
    std::sort(back2sort.begin(), back2sort.end());
    for(typename std::vector< std::pair<IT,NT> >::iterator itr = back2sort.begin(); itr != back2sort.end(); ++itr)
    {
        Indexed.ind.push_back(itr->first);
        Indexed.num.push_back(itr->second);
    }

    DeleteAll(sdispls, rdispls, sendcnt, recvcnt);
    DeleteAll(recvindbuf, recvdatbuf);
    return Indexed;
}


// ABAB: \todo Concept control so it only gets called in integers
template <class IT, class NT>
template <typename _BinaryOperation >
FullyDistSpVec<IT,NT> FullyDistSpVec<IT, NT>::Uniq(_BinaryOperation __binary_op, MPI_Op mympiop)
{
    return UniqAll2All(__binary_op, mympiop);
}

template <class IT, class NT>
FullyDistSpVec<IT,NT> & FullyDistSpVec<IT, NT>::operator+=(const FullyDistSpVec<IT,NT> & rhs)
{
	if(this != &rhs)
	{
		if(glen != rhs.glen)
		{
			std::cerr << "Vector dimensions don't match for addition\n";
			return *this;
		}
		IT lsize = getlocnnz();
		IT rsize = rhs.getlocnnz();

		std::vector< IT > nind;
		std::vector< NT > nnum;
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
		typename std::vector<NT>::iterator it;
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
			std::cerr << "Vector dimensions don't match for addition\n";
			return *this;
		}
		IT lsize = getlocnnz();
		IT rsize = rhs.getlocnnz();
		std::vector< IT > nind;
		std::vector< NT > nnum;
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


template <class IT, class NT>
template <typename _BinaryOperation>
void FullyDistSpVec<IT,NT>::SparseCommon(std::vector< std::vector < std::pair<IT,NT> > > & data, _BinaryOperation BinOp)
{
	int nprocs = commGrid->GetSize();
	int * sendcnt = new int[nprocs];
	int * recvcnt = new int[nprocs];
	for(int i=0; i<nprocs; ++i)
		sendcnt[i] = data[i].size();	// sizes are all the same

	MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, commGrid->GetWorld()); // share the counts
	int * sdispls = new int[nprocs]();
	int * rdispls = new int[nprocs]();
	std::partial_sum(sendcnt, sendcnt+nprocs-1, sdispls+1);
	std::partial_sum(recvcnt, recvcnt+nprocs-1, rdispls+1);
	IT totrecv = rdispls[nprocs-1]+recvcnt[nprocs-1];
	IT totsend = sdispls[nprocs-1]+sendcnt[nprocs-1];


  	std::pair<IT,NT> * senddata = new std::pair<IT,NT>[totsend];	// re-used for both rows and columns
	for(int i=0; i<nprocs; ++i)
	{
    std::copy(data[i].begin(), data[i].end(), senddata+sdispls[i]);
		std::vector< std::pair<IT,NT> >().swap(data[i]);	// clear memory
	}
	MPI_Datatype MPI_pair;
	MPI_Type_contiguous(sizeof(std::tuple<IT,NT>), MPI_CHAR, &MPI_pair);
	MPI_Type_commit(&MPI_pair);

	std::pair<IT,NT> * recvdata = new std::pair<IT,NT>[totrecv];
	MPI_Alltoallv(senddata, sendcnt, sdispls, MPI_pair, recvdata, recvcnt, rdispls, MPI_pair, commGrid->GetWorld());

	DeleteAll(senddata, sendcnt, recvcnt, sdispls, rdispls);
	MPI_Type_free(&MPI_pair);

	if(!is_sorted(recvdata, recvdata+totrecv))
		std::sort(recvdata, recvdata+totrecv);

	ind.push_back(recvdata[0].first);
	num.push_back(recvdata[0].second);
	for(IT i=1; i< totrecv; ++i)
       	{
		if(ind.back() == recvdata[i].first)
	   	{
		      	num.back() = BinOp(num.back(), recvdata[i].second);
	      	}
	      	else
	      	{
			ind.push_back(recvdata[i].first);
			num.push_back(recvdata[i].second);
	      	}
	}
	delete [] recvdata;
}

template <class IT, class NT>
template <typename _BinaryOperation>
void FullyDistSpVec<IT,NT>::ParallelRead (const std::string & filename, bool onebased, _BinaryOperation BinOp)
{
    int64_t gnnz;	// global nonzeros (glen is already declared as part of this class's private data)

    FILE *f;
    int myrank = commGrid->GetRank();
    int nprocs = commGrid->GetSize();
    if(myrank == 0)
    {
	if((f = fopen(filename.c_str(),"r"))==NULL)
	{
		std::cout << "File failed to open\n";
		MPI_Abort(commGrid->commWorld, NOFILE);
	}
	else
	{
		fscanf(f,"%ld %ld\n", &glen, &gnnz);
	}
        std::cout << "Total number of nonzeros expected across all processors is " << gnnz << std::endl;

    }
    MPI_Bcast(&glen, 1, MPIType<int64_t>(), 0, commGrid->commWorld);
    MPI_Bcast(&gnnz, 1, MPIType<int64_t>(), 0, commGrid->commWorld);


    struct stat st;     // get file size
    if (stat(filename.c_str(), &st) == -1)
    {
        MPI_Abort(commGrid->commWorld, NOFILE);
    }
    int64_t file_size = st.st_size;
    MPI_Offset fpos, end_fpos;
    if(myrank == 0)    // the offset needs to be for this rank
    {
        std::cout << "File is " << file_size << " bytes" << std::endl;
        fpos = ftell(f);
        fclose(f);
    }
    else
    {
        fpos = myrank * file_size / nprocs;
    }
    if(myrank != (nprocs-1)) end_fpos = (myrank + 1) * file_size / nprocs;
    else end_fpos = file_size;
    MPI_Barrier(commGrid->commWorld);

    MPI_File mpi_fh;
    MPI_File_open (commGrid->commWorld, const_cast<char*>(filename.c_str()), MPI_MODE_RDONLY, MPI_INFO_NULL, &mpi_fh);

    std::vector< std::vector < std::pair<IT,NT> > > data(nprocs);	// data to send

    std::vector<std::string> lines;

    SpParHelper::Print("Fetching first piece\n");

    MPI_Barrier(commGrid->commWorld);
    bool finished = SpParHelper::FetchBatch(mpi_fh, fpos, end_fpos, true, lines, myrank);
    int64_t entriesread = lines.size();
    SpParHelper::Print("Fetched first piece\n");

    MPI_Barrier(commGrid->commWorld);

    IT ii;
    NT vv;
    for (auto itr=lines.begin(); itr != lines.end(); ++itr)
    {
	std::stringstream ss(*itr);
	ss >> ii >> vv;
	if(onebased)	ii--;
	IT locind;
	int owner = Owner(ii, locind);       // recipient (owner) processor
        data[owner].push_back(std::make_pair(locind,vv));
    }

    while(!finished)
    {
        finished = SpParHelper::FetchBatch(mpi_fh, fpos, end_fpos, false, lines, myrank);
        entriesread += lines.size();
        for (auto itr=lines.begin(); itr != lines.end(); ++itr)
    	{
		std::stringstream ss(*itr);
		ss >> ii >> vv;
		if(onebased)	ii--;
		IT locind;
		int owner = Owner(ii, locind);       // recipient (owner) processor
        	data[owner].push_back(std::make_pair(locind,vv));
    	}
    }
    int64_t allentriesread;
    MPI_Reduce(&entriesread, &allentriesread, 1, MPIType<int64_t>(), MPI_SUM, 0, commGrid->commWorld);
#ifdef COMBBLAS_DEBUG
    if(myrank == 0)
        std::cout << "Reading finished. Total number of entries read across all processors is " << allentriesread << std::endl;
#endif

    SparseCommon(data, BinOp);
}

template <class IT, class NT>
template <class HANDLER>
void FullyDistSpVec<IT,NT>::ParallelWrite(const std::string & filename, bool onebased, HANDLER handler, bool includeindices, bool includeheader)
{
       	int myrank = commGrid->GetRank();
    	int nprocs = commGrid->GetSize();
	IT totalLength = TotalLength();
	IT totalNNZ = getnnz();

	std::stringstream ss;
	if(includeheader && myrank == 0)
	{
		ss << totalLength << '\t' << totalNNZ << '\n';	// rank-0 has the header
	}
	IT entries =  getlocnnz();
	IT sizeuntil = 0;
	MPI_Exscan( &entries, &sizeuntil, 1, MPIType<IT>(), MPI_SUM, commGrid->GetWorld() );
	if(myrank == 0) sizeuntil = 0;	// because MPI_Exscan says the recvbuf in process 0 is undefined

	if(includeindices)
	{
		if(onebased)	sizeuntil += 1;	// increment by 1

		for(IT i=0; i< entries; ++i)
		{
			ss << ind[i]+sizeuntil << '\t';
			handler.save(ss, num[i], ind[i]+sizeuntil);
			ss << '\n';
		}
	}
	else	// the base doesn't matter if we don't include indices
	{
		IT dummy = 0;	// dummy because we don't want indices to be printed
		for(IT i=0; i< entries; ++i)
		{
			handler.save(ss, num[i], dummy);
			ss << '\n';
		}
	}

	std::string text = ss.str();

	int64_t * bytes = new int64_t[nprocs];
    	bytes[myrank] = text.size();
    	MPI_Allgather(MPI_IN_PLACE, 1, MPIType<int64_t>(), bytes, 1, MPIType<int64_t>(), commGrid->GetWorld());
	int64_t bytesuntil = std::accumulate(bytes, bytes+myrank, static_cast<int64_t>(0));
	int64_t bytestotal = std::accumulate(bytes, bytes+nprocs, static_cast<int64_t>(0));

    	if(myrank == 0)	// only leader rights the original file with no content
    	{
		std::ofstream ofs(filename.c_str(), std::ios::binary | std::ios::out);
#ifdef COMBBLAS_DEBUG
    std::cout << "Creating file with " << bytestotal << " bytes" << std::endl;
#endif
    		ofs.seekp(bytestotal - 1);
    		ofs.write("", 1);	// this will likely create a sparse file so the actual disks won't spin yet
		ofs.close();
   	}
     	MPI_Barrier(commGrid->GetWorld());

	struct stat st;     // get file size
    	if (stat(filename.c_str(), &st) == -1)
    	{
       		MPI_Abort(commGrid->GetWorld(), NOFILE);
    	}
	if(myrank == nprocs-1)	// let some other processor do the testing
	{
#ifdef COMBBLAS_DEBUG
    std::cout << "File is actually " << st.st_size << " bytes seen from process " << myrank << std::endl;
#endif
	}

    	FILE *ffinal;
	if ((ffinal = fopen(filename.c_str(), "rb+")) == NULL)	// then everyone fills it
        {
		printf("COMBBLAS: Vector output file %s failed to open at process %d\n", filename.c_str(), myrank);
            	MPI_Abort(commGrid->GetWorld(), NOFILE);
       	}
	fseek (ffinal , bytesuntil , SEEK_SET );
	fwrite(text.c_str(),1, bytes[myrank] ,ffinal);
	fflush(ffinal);
	fclose(ffinal);
	delete [] bytes;
}

//! Called on an existing object
//! ABAB: Obsolete, will be deleted once moved to Github (and becomes independent of KDT)
template <class IT, class NT>
template <class HANDLER>
std::ifstream& FullyDistSpVec<IT,NT>::ReadDistribute (std::ifstream& infile, int master, HANDLER handler)
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
		std::fill_n(curptrs, neighs, 0);	// fill with zero
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
				int rec = Owner(tempind, locind);	// recipient (owner) processor  (ABAB: But if the length is not set yet, this should be wrong)
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
					std::fill_n(curptrs, neighs, 0);
					DeleteAll(tempinds, tempvals);
				}
				++ cnz;
			}
			assert (cnz == total_nnz);

			// Signal the end of file to other processors along the diagonal
			std::fill_n(curptrs, neighs, std::numeric_limits<int>::max());
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

			if( recvcount == std::numeric_limits<int>::max())
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
void FullyDistSpVec<IT,NT>::SaveGathered(std::ofstream& outfile, int master, HANDLER handler, bool printProcSplits)
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
	IT sizeuntil = std::accumulate(dist, dist+rank, static_cast<IT>(0));
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
			std::cerr << "Problem reading binary input file\n";
			return;
	        }
		IT maxd = *std::max_element(dist, dist+nprocs);
		mystruct * data = new mystruct[maxd];

		std::streamsize oldPrecision = outfile.precision();
		outfile.precision(21);
		outfile << totalLength << "\t1\t" << totalNNZ << std::endl;
		for(int i=0; i<nprocs; ++i)
		{
			// read n_per_proc integers and print them
			if (fread(data, dsize, dist[i], f) < static_cast<size_t>(dist[i]))
			{
			    std::cout << "fread 660 failed! attempting to continue..." << std::endl;
			}

			if (printProcSplits)
				outfile << "Elements stored on proc " << i << ":" << std::endl;

			for (int j = 0; j < dist[i]; j++)
			{
				outfile << data[j].ind+1 << "\t1\t";
				handler.save(outfile, data[j].num, data[j].ind);
				outfile << std::endl;
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
		typename std::vector< NT >::const_iterator iter = num.begin();
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
void FullyDistSpVec<IT,NT>::PrintInfo(std::string vectorname) const
{
	IT nznz = getnnz();
	if (commGrid->GetRank() == 0)
		std::cout << "As a whole, " << vectorname << " has: " << nznz << " nonzeros and length " << glen << std::endl;
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
	IT sizeuntil = std::accumulate(dist, dist+rank, static_cast<IT>(0));

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
                        std::cerr << "Problem reading binary input file\n";
                        return;
                }
		IT maxd = *std::max_element(dist, dist+nprocs);
		mystruct * data = new mystruct[maxd];

		for(int i=0; i<nprocs; ++i)
		{
			// read n_per_proc integers and print them
			if (fread(data, dsize, dist[i],f) < static_cast<size_t>(dist[i]))
			{
			    std::cout << "fread 802 failed! attempting to continue..." << std::endl;
			}

			std::cout << "Elements stored on proc " << i << ": {";
			for (int j = 0; j < dist[i]; j++)
			{
				std::cout << "(" << data[j].ind << "," << data[j].num << "), ";
			}
			std::cout << "}" << std::endl;
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
	std::copy(inds, inds+count, ind.data());
	std::copy(inds, inds+count, num.data());
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
        std::cout << "Sparse vector has entries (" << max_entry  << ") larger than requested global vector length " << globallen << std::endl;
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


    std::vector< std::pair<IT,NT> > tosort;
    tosort.resize(totrecv);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for(int i=0; i<totrecv; ++i)
	{
        tosort[i] = std::make_pair(recvindbuf[i], recvdatbuf[i]);
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
    for(typename std::vector<std::pair<IT,NT>>::iterator itr = tosort.begin(); itr != tosort.end(); ++itr)
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
 // _BinaryOperationDuplicate: function to reduce duplicate entries
 */

template <class IT, class NT>
template <typename _BinaryOperationIdx, typename _BinaryOperationVal, typename _BinaryOperationDuplicate>
FullyDistSpVec<IT,NT> FullyDistSpVec<IT,NT>::Invert (IT globallen, _BinaryOperationIdx __binopIdx, _BinaryOperationVal __binopVal, _BinaryOperationDuplicate __binopDuplicate)

{

    FullyDistSpVec<IT,NT> Inverted(commGrid, globallen);


    // identify the max index in the composed vector
    IT localmax = (IT) 0;
    for(size_t k=0; k < num.size(); ++k)
    {
        localmax = std::max(localmax, __binopIdx(num[k], ind[k] + LengthUntil()));
    }
    IT globalmax = (IT) 0;
    MPI_Allreduce( &localmax, &globalmax, 1, MPIType<IT>(), MPI_MAX, commGrid->GetWorld());

    if(globalmax >= globallen)
    {
        std::cout << "Sparse vector has entries (" << globalmax  << ") larger than requested global vector length " << globallen << std::endl;
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


    std::vector< std::pair<IT,NT> > tosort;
    tosort.resize(totrecv);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<totrecv; ++i)
    {
        tosort[i] = std::make_pair(recvindbuf[i], recvdatbuf[i]);
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


    // not threaded because Inverted.ind is kept sorted
    for(typename std::vector<std::pair<IT,NT>>::iterator itr = tosort.begin(); itr != tosort.end(); )
    {
        IT ind = itr->first;
        NT val = itr->second;
        ++itr;

        while(itr != tosort.end() && itr->first == ind)
        {
            val = __binopDuplicate(val, itr->second);
            ++itr;
        }


        Inverted.ind.push_back(ind);
        Inverted.num.push_back(val);

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
    for(size_t k=0; k < num.size(); ++k)
    {
        localmax = std::max(localmax, __binopIdx(num[k], ind[k] + LengthUntil()));
    }
    IT globalmax = (IT) 0;
    MPI_Allreduce( &localmax, &globalmax, 1, MPIType<IT>(), MPI_MAX, commGrid->GetWorld());

    if(globalmax >= globallen)
    {
        std::cout << "Sparse vector has entries (" << globalmax  << ") larger than requested global vector length " << globallen << std::endl;
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


    std::vector< std::pair<IT,NT> > tosort;
    tosort.resize(totrecv);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<totrecv; ++i)
    {
        tosort[i] = std::make_pair(recvindbuf[i], recvdatbuf[i]);
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
    for(typename std::vector<std::pair<IT,NT>>::iterator itr = tosort.begin(); itr != tosort.end(); ++itr)
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
			std::ostringstream outs;
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
        std::ostringstream outs;
        outs << "Grids are not comparable for Select" << std::endl;
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
            std::ostringstream outs;
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
        std::ostringstream outs;
        outs << "Grids are not comparable for Select" << std::endl;
        SpParHelper::Print(outs.str());
        MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
    }
}



template <typename IT, typename NT>
template <typename NT1, typename _UnaryOperation, typename _BinaryOperation>
void FullyDistSpVec<IT,NT>::SelectApply (const FullyDistVec<IT,NT1> & denseVec, _UnaryOperation __unop, _BinaryOperation __binop)
{
	if(*commGrid == *(denseVec.commGrid))
	{
		if(TotalLength() != denseVec.TotalLength())
		{
			std::ostringstream outs;
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
        std::ostringstream outs;
        outs << "Grids are not comparable for Select" << std::endl;
        SpParHelper::Print(outs.str());
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
	}
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
        std::ostringstream outs;
        outs << "Grids are not comparable for Filter" << std::endl;
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

    IT totrecv = std::accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));

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

}
