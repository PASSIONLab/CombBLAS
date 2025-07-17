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

#include "FullyDistVec.h"
#include "FullyDistSpVec.h"
#include "Operations.h"

namespace combblas {

template <class IT, class NT>
FullyDistVec<IT, NT>::FullyDistVec ()
: FullyDist<IT,NT,typename combblas::disable_if< combblas::is_boolean<NT>::value, NT >::type>()
{ 
}

template <class IT, class NT>
FullyDistVec<IT, NT>::FullyDistVec (IT globallen, NT initval)
:FullyDist<IT,NT,typename combblas::disable_if< combblas::is_boolean<NT>::value, NT >::type>(globallen)
{
	arr.resize(MyLocLength(), initval);
}


template <class IT, class NT>
FullyDistVec<IT, NT>::FullyDistVec ( std::shared_ptr<CommGrid> grid)
: FullyDist<IT,NT,typename combblas::disable_if< combblas::is_boolean<NT>::value, NT >::type>(grid)
{ }

template <class IT, class NT>
FullyDistVec<IT, NT>::FullyDistVec ( std::shared_ptr<CommGrid> grid, IT globallen, NT initval)
: FullyDist<IT,NT,typename combblas::disable_if< combblas::is_boolean<NT>::value, NT >::type>(grid,globallen)
{
	arr.resize(MyLocLength(), initval);
}

template <class IT, class NT>
FullyDistVec<IT,NT>::FullyDistVec (const FullyDistSpVec<IT,NT> & rhs)		// Conversion copy-constructor
: FullyDist<IT,NT,typename combblas::disable_if< combblas::is_boolean<NT>::value, NT >::type>(rhs.commGrid,rhs.glen)
{
	*this = rhs;
}


template <class IT, class NT>
template <class ITRHS, class NTRHS>
FullyDistVec<IT, NT>::FullyDistVec ( const FullyDistVec<ITRHS, NTRHS>& rhs )
: FullyDist<IT,NT,typename combblas::disable_if< combblas::is_boolean<NT>::value, NT >::type>(rhs.commGrid, static_cast<IT>(rhs.glen))
{
	arr.resize(static_cast<IT>(rhs.arr.size()), NT());
	
	for(IT i=0; (unsigned)i < arr.size(); ++i)
	{
		arr[i] = static_cast<NT>(rhs.arr[static_cast<ITRHS>(i)]);
	}
}

/**
  * Initialize a FullyDistVec with a separate vector from each processor
  * \pre{fillarr sizes does not have to exactly follow how CombBLAS would distribute a FullyDistVec}
  * \warning{However, they should be consecutive across processors. I.e., fillarr.front() at proc i+1 immediately follows fillarr.back() at proc i.}
  */
template <class IT, class NT>
FullyDistVec<IT, NT>::FullyDistVec ( const std::vector<NT> & fillarr, std::shared_ptr<CommGrid> grid ) 
: FullyDist<IT,NT,typename combblas::disable_if< combblas::is_boolean<NT>::value, NT >::type>(grid)
{
	MPI_Comm World = commGrid->GetWorld();
	int nprocs = commGrid->GetSize();
	int rank = commGrid->GetRank();
	
	IT * sizes = new IT[nprocs];
	IT nsize = fillarr.size(); 
	sizes[rank] = nsize;
	MPI_Allgather(MPI_IN_PLACE, 1, MPIType<IT>(), sizes, 1, MPIType<IT>(), World);
	glen = std::accumulate(sizes, sizes+nprocs, static_cast<IT>(0));

	IT lengthuntil = std::accumulate(sizes, sizes+rank, static_cast<IT>(0));
		
	// Although the found vector is not reshuffled yet, its glen and commGrid are set
	// We can call the Owner/MyLocLength/LengthUntil functions (to infer future distribution)
		
	// rebalance/redistribute
	int * sendcnt = new int[nprocs](); // no need to std::fill as this type of new[] with () will initialize PODs correctly
	for(IT i=0; i<nsize; ++i)
	{
		IT locind;
		int owner = Owner(i+lengthuntil, locind);	
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
	arr.resize(totrecv);
		
	// data is already in the right order in found.arr
	MPI_Alltoallv(fillarr.data(), sendcnt, sdispls, MPIType<NT>(), arr.data(), recvcnt, rdispls, MPIType<NT>(), World);
	DeleteAll(sendcnt, recvcnt, sdispls, rdispls);
	delete [] sizes;	
}



template <class IT, class NT>
std::pair<IT, NT> FullyDistVec<IT,NT>::MinElement() const
{
    auto it = min_element(arr.begin(), arr.end());
    NT localMin = *it;
    NT globalMin;
    MPI_Allreduce( &localMin, &globalMin, 1, MPIType<NT>(), MPI_MIN, commGrid->GetWorld());
    
    IT localMinIdx = TotalLength();
    if(globalMin==localMin)
    {
        localMinIdx = distance(arr.begin(), it) + LengthUntil(); 
    }
    IT globalMinIdx;
    MPI_Allreduce( &localMinIdx, &globalMinIdx, 1, MPIType<IT>(), MPI_MIN, commGrid->GetWorld()); // it can be MPI_MAX or anything

    return std::make_pair(globalMinIdx, globalMin);
}


template <class IT, class NT>
template <typename _BinaryOperation>
NT FullyDistVec<IT,NT>::Reduce(_BinaryOperation __binary_op, NT identity) const
{
	// std::accumulate returns identity for empty sequences
	NT localsum = std::accumulate( arr.begin(), arr.end(), identity, __binary_op);

	NT totalsum = identity;
	MPI_Allreduce( &localsum, &totalsum, 1, MPIType<NT>(), MPIOp<_BinaryOperation, NT>::op(), commGrid->GetWorld());
	return totalsum;
}

template <class IT, class NT>
template <typename OUT, typename _BinaryOperation, typename _UnaryOperation>
OUT FullyDistVec<IT,NT>::Reduce(_BinaryOperation __binary_op, OUT default_val, _UnaryOperation __unary_op) const
{
	// std::accumulate returns identity for empty sequences
	OUT localsum = default_val; 
	
	if (arr.size() > 0)
	{
		typename std::vector< NT >::const_iterator iter = arr.begin();
		//localsum = __unary_op(*iter);
		//iter++;
		while (iter < arr.end())
		{
			localsum = __binary_op(localsum, __unary_op(*iter));
			iter++;
		}
	}

	OUT totalsum = default_val;
	MPI_Allreduce( &localsum, &totalsum, 1, MPIType<OUT>(), MPIOp<_BinaryOperation, OUT>::op(), commGrid->GetWorld());
	return totalsum;
}


//! ABAB: Put concept check, NT should be integer for this to make sense
template<class IT, class NT>
void FullyDistVec<IT,NT>::SelectCandidates(double nver)
{
#ifdef DETERMINISTIC
	MTRand M(1);
#else
	MTRand M;	// generate random numbers with Mersenne Twister 
#endif     

	IT length = TotalLength();
	std::vector<double> loccands(length);
	std::vector<NT> loccandints(length);
	MPI_Comm World = commGrid->GetWorld();
	int myrank = commGrid->GetRank();
	if(myrank == 0)
	{
		for(int i=0; i<length; ++i)
			loccands[i] = M.rand();
		std::transform(loccands.begin(), loccands.end(), loccands.begin(),
      	[nver](double val) { return val * nver; });
		
		for(int i=0; i<length; ++i)
			loccandints[i] = static_cast<NT>(loccands[i]);
	}
	MPI_Bcast(&(loccandints[0]), length, MPIType<NT>(),0, World);
	for(IT i=0; i<length; ++i)
		SetElement(i,loccandints[i]);
}

template <class IT, class NT>
template <class ITRHS, class NTRHS>
FullyDistVec< IT,NT > &  FullyDistVec<IT,NT>::operator=(const FullyDistVec< ITRHS,NTRHS > & rhs)	
{
	if(static_cast<const void*>(this) != static_cast<const void*>(&rhs))		
	{
		//FullyDist<IT,NT>::operator= (rhs);	// to update glen and commGrid
		glen = static_cast<IT>(rhs.glen);
		commGrid = rhs.commGrid;
		
		arr.resize(rhs.arr.size(), NT());
		for(IT i=0; (unsigned)i < arr.size(); ++i)
		{
			arr[i] = static_cast<NT>(rhs.arr[static_cast<ITRHS>(i)]);
		}
	}
	return *this;
}	

template <class IT, class NT>
FullyDistVec< IT,NT > &  FullyDistVec<IT,NT>::operator=(const FullyDistVec< IT,NT > & rhs)	
{
	if(this != &rhs)		
	{
		FullyDist<IT,NT,typename combblas::disable_if< combblas::is_boolean<NT>::value, NT >::type>::operator= (rhs);	// to update glen and commGrid
		arr = rhs.arr;
	}
	return *this;
}	

template <class IT, class NT>
FullyDistVec< IT,NT > &  FullyDistVec<IT,NT>::operator=(const FullyDistSpVec< IT,NT > & rhs)		// FullyDistSpVec->FullyDistVec conversion operator
{
	FullyDist<IT,NT,typename combblas::disable_if< combblas::is_boolean<NT>::value, NT >::type>::operator= (rhs);	// to update glen and commGrid
	arr.resize(rhs.MyLocLength());
	std::fill(arr.begin(), arr.end(), NT());	

	IT spvecsize = rhs.getlocnnz();
	for(IT i=0; i< spvecsize; ++i)
	{
		//if(rhs.ind[i] > arr.size())
		//	cout << "rhs.ind[i]: " << rhs.ind[i] <<  endl;
		arr[rhs.ind[i]] = rhs.num[i];
	}
	return *this;
}



/**
 ** Let the compiler create an assignment operator and call base class'
 ** assignment operator automatically
 **/


template <class IT, class NT>
FullyDistVec< IT,NT > &  FullyDistVec<IT,NT>::operator+=(const FullyDistSpVec< IT,NT > & rhs)		
{
	IT spvecsize = rhs.getlocnnz();
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for(IT i=0; i< spvecsize; ++i)
	{
		if(arr[rhs.ind[i]] == NT()) // not set before
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
	return *this;
}


/**
  * Perform __binary_op(*this[i], rhs[i]) for every element in rhs, *this,
  * which are of the same size. and write the result back to *this
  */ 
template <class IT, class NT>
template <typename _BinaryOperation>	
void FullyDistVec<IT,NT>::EWise(const FullyDistVec<IT,NT> & rhs,  _BinaryOperation __binary_op)
{
  std::transform ( arr.begin(), arr.end(), rhs.arr.begin(), arr.begin(), __binary_op );
};

/**
 * Perform __binary_op(*this[i], rhs[i]) for every element in rhs and *this, which are of the same size.
 * write the result output vector, hence the name EWiseOut
 */
template <class IT, class NT>
template <typename _BinaryOperation, typename OUT>
void FullyDistVec<IT,NT>::EWiseOut(const FullyDistVec<IT,NT> & rhs, _BinaryOperation __binary_op, FullyDistVec<IT,OUT> & result)
{
  std::transform ( arr.begin(), arr.end(), rhs.arr.begin(), result.arr.begin(), __binary_op );
};


template <class IT, class NT>
FullyDistVec<IT,NT> & FullyDistVec<IT, NT>::operator+=(const FullyDistVec<IT,NT> & rhs)
{
	if(this != &rhs)		
	{	
		if(!(*commGrid == *rhs.commGrid)) 		
		{
			std::cout << "Grids are not comparable elementwise addition" << std::endl; 
			MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
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
			std::cout << "Grids are not comparable elementwise addition" << std::endl; 
			MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
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
    MPI_Allreduce( &local, &whole, 1, MPI_INT, MPI_BAND, commGrid->GetWorld());
    return static_cast<bool>(whole);	
}

template <class IT, class NT>
template <typename _Predicate>
IT FullyDistVec<IT,NT>::Count(_Predicate pred) const
{
	IT local = count_if( arr.begin(), arr.end(), pred );
	IT whole = 0;
	MPI_Allreduce( &local, &whole, 1, MPIType<IT>(), MPI_SUM, commGrid->GetWorld());
	return whole;	
}

//! Returns a dense vector of global indices 
//! for which the predicate is satisfied
template <class IT, class NT>
template <typename _Predicate>
FullyDistVec<IT,IT> FullyDistVec<IT,NT>::FindInds(_Predicate pred) const
{
	FullyDistVec<IT,IT> found(commGrid);
	MPI_Comm World = commGrid->GetWorld();
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
	MPI_Alltoallv(&(found.arr[0]), sendcnt, sdispls, MPIType<IT>(), &(recvbuf[0]), recvcnt, rdispls, MPIType<IT>(), World);
	found.arr.swap(recvbuf);
	delete [] dist;
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
	size_t size = arr.size();
	for(size_t i=0; i<size; ++i)
	{
		if(pred(arr[i]))
		{
			found.ind.push_back( (IT) i);
			found.num.push_back(arr[i]);
		}
	}
	found.glen = glen;
	return found;	
}


//! Retain a sparse vector with indices where the supplied value is found
template <class IT, class NT>
FullyDistSpVec<IT,NT> FullyDistVec<IT,NT>::Find(NT val) const
{
    FullyDistSpVec<IT,NT> found(commGrid);
    size_t size = arr.size();
    for(size_t i=0; i<size; ++i)
    {
        if(arr[i]==val)
        {
            found.ind.push_back( (IT) i);
            found.num.push_back(val);
        }
    }
    found.glen = glen;
    return found;	
}


template <class IT, class NT>
template <class HANDLER>
std::ifstream& FullyDistVec<IT,NT>::ReadDistribute (std::ifstream& infile, int master, HANDLER handler)
{
	FullyDistSpVec<IT,NT> tmpSpVec(commGrid);
	tmpSpVec.ReadDistribute(infile, master, handler);

	*this = tmpSpVec;
	return infile;
}

template <class IT, class NT>
template <class HANDLER>
void FullyDistVec<IT,NT>::SaveGathered(std::ofstream& outfile, int master, HANDLER handler, bool printProcSplits)
{
	FullyDistSpVec<IT,NT> tmpSpVec = *this;
	tmpSpVec.SaveGathered(outfile, master, handler, printProcSplits);
}

template <class IT, class NT>
void FullyDistVec<IT,NT>::SetElement (IT indx, NT numx)
{
	int rank = commGrid->GetRank();
	if (glen == 0) 
	{
		if(rank == 0)
			std::cout << "FullyDistVec::SetElement can't be called on an empty vector." << std::endl;
		return;
	}
	IT locind;
	int owner = Owner(indx, locind);
	if(commGrid->GetRank() == owner)
	{
		if (locind > (LocArrSize() -1))
		{
			std::cout << "FullyDistVec::SetElement cannot expand array" << std::endl;
		}
		else if (locind < 0)
		{
			std::cout << "FullyDistVec::SetElement local index < 0" << std::endl;
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
	MPI_Comm World = commGrid->GetWorld();
	int rank = commGrid->GetRank();
	if (glen == 0) 
	{
		if(rank == 0)
			std::cout << "FullyDistVec::GetElement can't be called on an empty vector." << std::endl;

		return NT();
	}
	IT locind;
	int owner = Owner(indx, locind);
	if(commGrid->GetRank() == owner)
	{
		if (locind > (LocArrSize() -1))
		{
			std::cout << "FullyDistVec::GetElement local index > size" << std::endl;
			ret = NT();

		}
		else if (locind < 0)
		{
			std::cout << "FullyDistVec::GetElement local index < 0" << std::endl;
			ret = NT();
		}
		else
		{
			ret = arr[locind];
		}
	}
	MPI_Bcast(&ret, 1, MPIType<NT>(), owner, World);
	return ret;
}

// Write to file using MPI-2
template <class IT, class NT>
void FullyDistVec<IT,NT>::DebugPrint()
{
	int nprocs, rank;
	MPI_Comm World = commGrid->GetWorld();
	MPI_Comm_rank(World, &rank);
	MPI_Comm_size(World, &nprocs);
	MPI_File thefile;
	char _fn[] = "temp_fullydistvec"; // AL: this is to avoid the problem that C++ string literals are const char* while C string literals are char*, leading to a const warning (technically error, but compilers are tolerant)
	MPI_File_open(World, _fn, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &thefile);    
	IT lengthuntil = LengthUntil();

	// The disp displacement argument specifies the position 
	// (absolute offset in bytes from the beginning of the file) 
	char native[] = "native"; // AL: this is to avoid the problem that C++ string literals are const char* while C string literals are char*, leading to a const warning (technically error, but compilers are tolerant)
	MPI_File_set_view(thefile, int64_t(lengthuntil * sizeof(NT)), MPIType<NT>(), MPIType<NT>(), native, MPI_INFO_NULL);

	IT count = LocArrSize();	
	MPI_File_write(thefile, &(arr[0]), count, MPIType<NT>(), MPI_STATUS_IGNORE);
	MPI_File_close(&thefile);
	
	// Now let processor-0 read the file and print
	IT * counts = new IT[nprocs];
	MPI_Gather(&count, 1, MPIType<IT>(), counts, 1, MPIType<IT>(), 0, World);	// gather at root=0
	if(rank == 0)
	{
		FILE * f = fopen("temp_fullydistvec", "r");
		if(!f)
		{
				std::cerr << "Problem reading binary input file\n";
				return;
		}
		IT maxd = *std::max_element(counts, counts+nprocs);
		NT * data = new NT[maxd];

		for(int i=0; i<nprocs; ++i)
		{
			// read counts[i] integers and print them
			size_t result = fread(data, sizeof(NT), counts[i],f);
			if (result != (unsigned)counts[i]) { std::cout << "Error in fread, only " << result << " entries read" << std::endl; }

			std::cout << "Elements stored on proc " << i << ": {" ;	
			for (int j = 0; j < counts[i]; j++)
			{
				std::cout << data[j] << ",";
			}
			std::cout << "}" << std::endl;
		}
		delete [] data;
		delete [] counts;
	}
}

template <class IT, class NT>
template <typename _UnaryOperation, typename IRRELEVANT_NT>
void FullyDistVec<IT,NT>::Apply(_UnaryOperation __unary_op, const FullyDistSpVec<IT,IRRELEVANT_NT> & mask)
{
	typename std::vector< IT >::const_iterator miter = mask.ind.begin();
	while (miter < mask.ind.end())
	{
		IT index = *miter++;
		arr[index] = __unary_op(arr[index]);
	}
}	

template <class IT, class NT>
template <typename _BinaryOperation, typename _BinaryPredicate, class NT2>
void FullyDistVec<IT,NT>::EWiseApply(const FullyDistVec<IT,NT2> & other, _BinaryOperation __binary_op, _BinaryPredicate _do_op, const bool useExtendedBinOp)
{
	if(*(commGrid) == *(other.commGrid))	
	{
		if(glen != other.glen)
		{
			std::ostringstream outs;
			outs << "Vector dimensions don't match (" << glen << " vs " << other.glen << ") for FullyDistVec::EWiseApply\n";
			SpParHelper::Print(outs.str());
			MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
		}
		else
		{
			typename std::vector< NT >::iterator thisIter = arr.begin();
			typename std::vector< NT2 >::const_iterator otherIter = other.arr.begin();
			while (thisIter < arr.end())
			{
				if (_do_op(*thisIter, *otherIter, false, false))
					*thisIter = __binary_op(*thisIter, *otherIter, false, false);
				thisIter++;
				otherIter++;
			}
		}
	}
	else
	{
		std::ostringstream outs;
		outs << "Grids are not comparable for FullyDistVec<IT,NT>::EWiseApply" << std::endl; 
		SpParHelper::Print(outs.str());
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
	}
}	


// Note (Ariful): multithreded implemented only when applyNulls=false.
// TODO: employ multithreding when applyNulls=true
template <class IT, class NT>
template <typename _BinaryOperation, typename _BinaryPredicate, class NT2>
void FullyDistVec<IT,NT>::EWiseApply(const FullyDistSpVec<IT,NT2> & other, _BinaryOperation __binary_op, _BinaryPredicate _do_op, bool applyNulls, NT2 nullValue, const bool useExtendedBinOp)
{
	if(*(commGrid) == *(other.commGrid))	
	{
		if(glen != other.glen)
		{
			std::cerr << "Vector dimensions don't match (" << glen << " vs " << other.glen << ") for FullyDistVec::EWiseApply\n";
			MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
		}
		else
		{
			typename std::vector< IT >::const_iterator otherInd = other.ind.begin();
			typename std::vector< NT2 >::const_iterator otherNum = other.num.begin();
			
			if (applyNulls) // scan the entire dense vector and apply sparse elements as they appear
			{
				for(IT i=0; (unsigned)i < arr.size(); ++i)
				{
					if (otherInd == other.ind.end() || i < *otherInd)
					{
						if (_do_op(arr[i], nullValue, false, true))
							arr[i] = __binary_op(arr[i], nullValue, false, true);
					}
					else
					{
						if (_do_op(arr[i], *otherNum, false, false))
							arr[i] = __binary_op(arr[i], *otherNum, false, false);
						otherInd++;
						otherNum++;
					}
				}
			}
			else // scan the sparse vector only
			{
                /*
                for(otherInd = other.ind.begin(); otherInd < other.ind.end(); otherInd++, otherNum++)
				{
					if (_do_op(arr[*otherInd], *otherNum, false, false))
						arr[*otherInd] = __binary_op(arr[*otherInd], *otherNum, false, false);
				}*/
                
                IT spsize = other.ind.size();
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for(IT i=0; i< spsize; i++)
                {
                    if (_do_op(arr[other.ind[i]], other.num[i], false, false))
                        arr[other.ind[i]] = __binary_op(arr[other.ind[i]], other.num[i], false, false);
                }
			}
		}
	}
	else
	{
		std::cout << "Grids are not comparable elementwise apply" << std::endl; 
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
	}
}	


template <class IT, class NT>
FullyDistVec<IT, IT> FullyDistVec<IT, NT>::sort()
{
	MPI_Comm World = commGrid->GetWorld();
	FullyDistVec<IT,IT> temp(commGrid);
	IT nnz = LocArrSize(); 
	std::pair<NT,IT> * vecpair = new std::pair<NT,IT>[nnz];
	int nprocs = commGrid->GetSize();
	int rank = commGrid->GetRank();

	IT * dist = new IT[nprocs];
	dist[rank] = nnz;
	MPI_Allgather(MPI_IN_PLACE, 1, MPIType<IT>(), dist, 1, MPIType<IT>(), World);
	IT sizeuntil = LengthUntil();	// size = length, for dense vectors
	for(IT i=0; i< nnz; ++i)
	{
		vecpair[i].first = arr[i];	// we'll sort wrt numerical values
		vecpair[i].second = i + sizeuntil;	
	}
	SpParHelper::MemoryEfficientPSort(vecpair, nnz, dist, World);

	std::vector< IT > narr(nnz);
	for(IT i=0; i< nnz; ++i)
	{
		arr[i] = vecpair[i].first;	// sorted range (change the object itself)
		narr[i] = vecpair[i].second;	// inverse permutation stored as numerical values
	}
	delete [] vecpair;
	delete [] dist;

	temp.glen = glen;
	temp.arr = narr;
	return temp;
}
		

// Randomly permutes an already existing vector
template <class IT, class NT>
void FullyDistVec<IT,NT>::RandPerm(uint64_t seed)
{
	if(seed == 1383098845){
        // If seed is equal to the default value it is assumed that no seed was provided
#ifdef DETERMINISTIC
        seed = 1383098845;
#else
        seed = time(NULL);
#endif
    }
    else{
        // If seed is not equal to the default value it means user is providing some
        // value as the seed. So don't touch it.
    }
    
	MTRand M(seed);	// generate random numbers with Mersenne Twister
    MPI_Comm World = commGrid->GetWorld();
	int nprocs = commGrid->GetSize();
	int rank = commGrid->GetRank();
    IT size = LocArrSize();

#ifdef COMBBLAS_LEGACY
  std::pair<double,NT> * vecpair = new std::pair<double,NT>[size];
	IT * dist = new IT[nprocs];
	dist[rank] = size;
	MPI_Allgather(MPI_IN_PLACE, 1, MPIType<IT>(), dist, 1, MPIType<IT>(), World);
	for(int i=0; i<size; ++i)
	{
		vecpair[i].first = M.rand();
		vecpair[i].second = arr[i];	
	}
	// less< pair<T1,T2> > works correctly (sorts wrt first elements)	
	SpParHelper::MemoryEfficientPSort(vecpair, size, dist, World);
  std::vector< NT > nnum(size);
	for(int i=0; i<size; ++i) nnum[i] = vecpair[i].second;
    DeleteAll(vecpair, dist);
	arr.swap(nnum);
#else
    std::vector< std::vector< NT > > data_send(nprocs);
	for(int i=0; i<size; ++i)
    {
		// send each entry to a random process
		uint32_t dest = M.randInt(nprocs-1);
		data_send[dest].push_back(arr[i]);
	}
    int * sendcnt = new int[nprocs];
    int * sdispls = new int[nprocs];
    for(int i=0; i<nprocs; ++i) sendcnt[i] = (int) data_send[i].size();
    
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
    IT totrecv = std::accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));
	if(totrecv > std::numeric_limits<int>::max())
	{
		std::cout << "COMBBLAS_WARNING: total data to receive exceeds max int: " << totrecv << std::endl;
	}
    std::vector<NT>().swap(arr);  // make space for temporaries
    
	NT * sendbuf = new NT[size];
    for(int i=0; i<nprocs; ++i)
    {
        std::copy(data_send[i].begin(), data_send[i].end(), sendbuf+sdispls[i]);
        std::vector<NT>().swap(data_send[i]);	// free memory
    }
	NT * recvbuf = new NT[totrecv];
    MPI_Alltoallv(sendbuf, sendcnt, sdispls, MPIType<NT>(), recvbuf, recvcnt, rdispls, MPIType<NT>(), World);
	//std::random_shuffle(recvbuf, recvbuf+ totrecv);
    std::default_random_engine gen(seed);
    std::shuffle(recvbuf, recvbuf+ totrecv,gen); // locally shuffle data
    
	int64_t * localcounts = new int64_t[nprocs];
	localcounts[rank] = totrecv;
	MPI_Allgather(MPI_IN_PLACE, 1, MPI_LONG_LONG, localcounts, 1, MPI_LONG_LONG, World);
	int64_t glenuntil = std::accumulate(localcounts, localcounts+rank, static_cast<int64_t>(0));
    
	std::vector< std::vector< IT > > locs_send(nprocs);
	for(IT i=0; i< totrecv; ++i)    // determine new locations w/ prefix sums
	{
		IT remotelocind;
		int owner = Owner(glenuntil+i, remotelocind);
		locs_send[owner].push_back(remotelocind);
		data_send[owner].push_back(recvbuf[i]);
	}
    
    for(int i=0; i<nprocs; ++i) sendcnt[i] = (int) data_send[i].size(); // = locs_send.size()
    MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, World);
    sdispls[0] = 0;
    rdispls[0] = 0;
    for(int i=0; i<nprocs-1; ++i)
    {
        sdispls[i+1] = sdispls[i] + sendcnt[i];
        rdispls[i+1] = rdispls[i] + recvcnt[i];
    }
    IT newsize = std::accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));
	if(newsize > std::numeric_limits<int>::max())
	{
		std::cout << "COMBBLAS_WARNING: total data to receive exceeds max int: " << newsize << std::endl;
	}
	// re-use the receive buffer as sendbuf of second stage
    IT totalsend = std::accumulate(sendcnt, sendcnt+nprocs, static_cast<IT>(0));
    if(totalsend != totrecv || newsize != size)
    {
        std::cout << "COMBBLAS_WARNING: sending different sized data than received: " << totalsend << "=" << totrecv << " , " << newsize << "=" << size << std::endl;
    }
    for(int i=0; i<nprocs; ++i)
    {
        std::copy(data_send[i].begin(), data_send[i].end(), recvbuf+sdispls[i]);
        std::vector<NT>().swap(data_send[i]);	// free memory
    }
    // re-use the send buffer as receive buffer of second stage
    MPI_Alltoallv(recvbuf, sendcnt, sdispls, MPIType<NT>(), sendbuf, recvcnt, rdispls, MPIType<NT>(), World);
    delete [] recvbuf;
    IT * newinds = new IT[totalsend];
    for(int i=0; i<nprocs; ++i)
    {
        std::copy(locs_send[i].begin(), locs_send[i].end(), newinds+sdispls[i]);
        std::vector<IT>().swap(locs_send[i]);	// free memory
    }
    IT * indsbuf = new IT[size];
	MPI_Alltoallv(newinds, sendcnt, sdispls, MPIType<IT>(), indsbuf, recvcnt, rdispls, MPIType<IT>(), World);
    DeleteAll(newinds, sendcnt, sdispls, rdispls, recvcnt);
    arr.resize(size);
    for(IT i=0; i<size; ++i)
    {
        arr[indsbuf[i]] = sendbuf[i];
    }
#endif
}

// ABAB: In its current form, unless LengthUntil returns NT
// this won't work reliably for anything other than integers
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
		std::cout << "Grids are not comparable for dense vector subsref" << std::endl;
		return FullyDistVec<IT,NT>();
	}

	MPI_Comm World = commGrid->GetWorld();
	FullyDistVec<IT,NT> Indexed(commGrid, ri.glen, NT());	// length(Indexed) = length(ri)
	int nprocs = commGrid->GetSize();
	std::vector< std::vector< IT > > data_req(nprocs);	
	std::vector< std::vector< IT > > revr_map(nprocs);	// to put the incoming data to the correct location	

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
		sendcnt[i] = (int) data_req[i].size();

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
	IT totrecv = std::accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));
	for(int i=0; i<nprocs; ++i)
	{
    std::copy(data_req[i].begin(), data_req[i].end(), sendbuf+sdispls[i]);
		std::vector<IT>().swap(data_req[i]);
	}

	IT * reversemap = new IT[riloclen];
	for(int i=0; i<nprocs; ++i)
	{
    std::copy(revr_map[i].begin(), revr_map[i].end(), reversemap+sdispls[i]);	// reversemap array is unique
		std::vector<IT>().swap(revr_map[i]);
	}

	IT * recvbuf = new IT[totrecv];
	MPI_Alltoallv(sendbuf, sendcnt, sdispls, MPIType<IT>(), recvbuf, recvcnt, rdispls, MPIType<IT>(), World);  // request data
	delete [] sendbuf;
		
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
	MPI_Alltoallv(databack, recvcnt, rdispls, MPIType<NT>(), databuf, sendcnt, sdispls, MPIType<NT>(), World);  // send data
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
void FullyDistVec<IT,NT>::PrintInfo(std::string vectorname) const
{
	IT totl = TotalLength();
	if (commGrid->GetRank() == 0)		
		std::cout << "As a whole, " << vectorname << " has length " << totl << std::endl; 
}





template <class IT, class NT>
void FullyDistVec<IT,NT>::Set(const FullyDistSpVec< IT,NT > & other)
{
    if(*(commGrid) == *(other.commGrid))
    {
        if(glen != other.glen)
        {
            std::cerr << "Vector dimensions don't match (" << glen << " vs " << other.glen << ") for FullyDistVec::Set\n";
            MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
        }
        else
        {
            
            IT spvecsize = other.getlocnnz();
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(IT i=0; i< spvecsize; ++i)
            {
                arr[other.ind[i]] = other.num[i];
            }
        }
    }
    else
    {
        std::cout << "Grids are not comparable for Set" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
    }
}




// General purpose set operation on dense vector by a sparse vector

template <class IT, class NT>
template <class NT1, typename _BinaryOperationIdx, typename _BinaryOperationVal>
void FullyDistVec<IT,NT>::GSet (const FullyDistSpVec<IT,NT1> & spVec, _BinaryOperationIdx __binopIdx, _BinaryOperationVal __binopVal, MPI_Win win)
{
    if(*(commGrid) != *(spVec.commGrid))
    {
        std::cout << "Grids are not comparable for GSet" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
    }
    
    IT spVecSize = spVec.getlocnnz();
    if(spVecSize==0) return;
    
    
    MPI_Comm World = commGrid->GetWorld();
    int nprocs = commGrid->GetSize();
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);


    std::vector< std::vector< NT > > datsent(nprocs);
    std::vector< std::vector< IT > > indsent(nprocs);
    IT lengthUntil = spVec.LengthUntil();
   
    for(IT k=0; k < spVecSize; ++k)
    {
        IT locind;
        // get global index of the dense vector from the value. Most often a select operator.
        // If the first operand is selected, then invert; otherwise, EwiseApply.
        IT globind = __binopIdx(spVec.num[k], spVec.ind[k] + lengthUntil);
        int owner = Owner(globind, locind); // get local index
        NT val = __binopVal(spVec.num[k], spVec.ind[k] + lengthUntil);
        if(globind < glen) // prevent index greater than size of the composed vector
        {
            datsent[owner].push_back(val);
            indsent[owner].push_back(locind);   // so that we don't need no correction at the recipient
        }
    }
    
    
    for(int j = 0; j < datsent[myrank].size(); ++j)	// directly set local entries
    {
        arr[indsent[myrank][j]] = datsent[myrank][j];
    }
    
    
    //MPI_Win win;
    //MPI_Win_create(&arr[0], LocArrSize() * sizeof(NT), sizeof(NT), MPI_INFO_NULL, World, &win);
    //MPI_Win_fence(0, win);
    for(int i=0; i<nprocs; ++i)
    {
        if(i!=myrank)
        {
            MPI_Win_lock(MPI_LOCK_SHARED,i,MPI_MODE_NOCHECK,win);
            for(int j = 0; j < datsent[i].size(); ++j)
            {
                MPI_Put(&datsent[i][j], 1, MPIType<NT>(), i, indsent[i][j], 1, MPIType<NT>(), win);
            }
            MPI_Win_unlock(i, win);
        }
    }
    //MPI_Win_fence(0, win);
    //MPI_Win_free(&win);

}



// General purpose get operation on dense vector by a sparse vector
// Get the element of the dense vector indexed by the value of the sparse vector
// invert and get might not work in the presence of repeated values

template <class IT, class NT>
template <class NT1, typename _BinaryOperationIdx>
 FullyDistSpVec<IT,NT> FullyDistVec<IT,NT>::GGet (const FullyDistSpVec<IT,NT1> & spVec, _BinaryOperationIdx __binopIdx, NT nullValue)
{
    if(*(commGrid) != *(spVec.commGrid))
    {
        std::cout << "Grids are not comparable for GGet" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
    }
    
    MPI_Comm World = commGrid->GetWorld();
    int nprocs = commGrid->GetSize();
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    
    
    std::vector< std::vector< NT > > spIdx(nprocs);
    std::vector< std::vector< IT > > indsent(nprocs);
    IT lengthUntil = spVec.LengthUntil();
    IT spVecSize = spVec.getlocnnz();
    
    FullyDistSpVec<IT, NT> res(spVec.commGrid, spVec.TotalLength());
    res.ind.resize(spVecSize);
    res.num.resize(spVecSize);
    
    
    for(IT k=0; k < spVecSize; ++k)
    {
        IT locind;
        // get global index of the dense vector from the value. Most often a select operator.
        // If the first operand is selected, then invert; otherwise, EwiseApply.
        IT globind = __binopIdx(spVec.num[k], spVec.ind[k] + lengthUntil);
        int owner = Owner(globind, locind); // get local index
        //NT val = __binopVal(spVec.num[k], spVec.ind[k] + lengthUntil);
        if(globind < glen) // prevent index greater than size of the composed vector
        {
            spIdx[owner].push_back(k); // position of spVec
            indsent[owner].push_back(locind);   // so that we don't need no correction at the recipient
        }
        else
            res.num[k] = nullValue;
        res.ind[k] = spVec.ind[k];
    }
    
    
    for(int j = 0; j < indsent[myrank].size(); ++j)	// directly get local entries
    {
        res.num[spIdx[myrank][j]] = arr[indsent[myrank][j]];
    }
    
    
    MPI_Win win;
    MPI_Win_create(&arr[0], LocArrSize() * sizeof(NT), sizeof(NT), MPI_INFO_NULL, World, &win);
    for(int i=0; i<nprocs; ++i)
    {
        if(i!=myrank)
        {
            MPI_Win_lock(MPI_LOCK_SHARED,i,MPI_MODE_NOCHECK,win);
            for(int j = 0; j < indsent[i].size(); ++j)
            {
                MPI_Get(&res.num[spIdx[i][j]], 1, MPIType<NT>(), i, indsent[i][j], 1, MPIType<NT>(), win);
            }
            MPI_Win_unlock(i, win);
        }
    }
    MPI_Win_free(&win);
    
    return res;
}

}
