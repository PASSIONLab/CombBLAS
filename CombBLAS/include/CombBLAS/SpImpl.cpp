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


#include "SpImpl.h"
#include "SpParHelper.h"
#include "PBBS/radixSort.h"
#include "Tommy/tommyhashdyn.h"

namespace combblas {

/**
 * Base template version [full use of the semiring add() and multiply()]
 * @param[in] indx { vector that practically keeps column numbers requested from A }
 *
 *
 * Roughly how the below function works:
 * Let's say our sparse vector has entries at 3, 7 and 9.
 * FillColInds() creates a vector of pairs that contain the
 * start and end indices (into matrix.ir and matrix.numx arrays).
 * pair.first is the start index, pair.second is the end index.
 *
 * Here's how we merge these adjacencies of 3,7 and 9:
 * We keep a heap of size 3 and push the first entries in adj{3}, adj{7}, adj{9} onto the heap wset.
 * That happens in the first for loop.
 *
 * Then as we pop from the heap we push the next entry from the previously popped adjacency (i.e. matrix column).
 * The heap ensures the output comes out sorted without using a SPA.
 * that's why indy.back() == wset[hsize-1].key is enough to ensure proper merging.
 **/
template <class SR, class IT, class NUM, class IVT, class OVT>
void SpImpl<SR,IT,NUM,IVT,OVT>::SpMXSpV(const Dcsc<IT,NUM> & Adcsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen,  
			std::vector<int32_t> & indy, std::vector< OVT > & numy)
{
	int32_t hsize = 0;		
	// colinds dereferences A.ir (valid from colinds[].first to colinds[].second)
	std::vector< std::pair<IT,IT> > colinds( (IT) veclen);		
	Adcsc.FillColInds(indx, (IT) veclen, colinds, NULL, 0);	// csize is irrelevant if aux is NULL	

	if(sizeof(NUM) > sizeof(OVT))	// ABAB: include a filtering based runtime choice as well?
	{
		HeapEntry<IT, OVT> * wset = new HeapEntry<IT, OVT>[veclen]; 
		for(IT j =0; j< veclen; ++j)		// create the initial heap 
		{
			while(colinds[j].first != colinds[j].second )	// iterate until finding the first entry within this column that passes the filter
			{
				OVT mrhs = SR::multiply(Adcsc.numx[colinds[j].first], numx[j]);
				if(SR::returnedSAID())
				{
					++(colinds[j].first);	// increment the active row index within the jth column
				}
				else
				{
					wset[hsize++] = HeapEntry< IT,OVT > ( Adcsc.ir[colinds[j].first], j, mrhs);
					break;	// this column successfully inserted an entry to the heap
				}
			} 
		}
		std::make_heap(wset, wset+hsize);
		while(hsize > 0)
		{
			std::pop_heap(wset, wset + hsize);         	// result is stored in wset[hsize-1]
			IT locv = wset[hsize-1].runr;		// relative location of the nonzero in sparse column vector 
			if((!indy.empty()) && indy.back() == wset[hsize-1].key)	
			{
				numy.back() = SR::add(numy.back(), wset[hsize-1].num);
			}
			else
			{
				indy.push_back( (int32_t) wset[hsize-1].key);
				numy.push_back(wset[hsize-1].num);	
			}
			bool pushed = false;
			// invariant: if ++(colinds[locv].first) == colinds[locv].second, then locv will not appear again in the heap
			while ( (++(colinds[locv].first)) != colinds[locv].second )	// iterate until finding another passing entry
			{
				OVT mrhs =  SR::multiply(Adcsc.numx[colinds[locv].first], numx[locv]);
				if(!SR::returnedSAID())
                                {
					wset[hsize-1].key = Adcsc.ir[colinds[locv].first];
					wset[hsize-1].num = mrhs;
					std::push_heap(wset, wset+hsize);	// runr stays the same
					pushed = true;
					break;
				}
			}
			if(!pushed)	--hsize;
		}
		delete [] wset;
	}
	
	else
	{
		HeapEntry<IT, NUM> * wset = new HeapEntry<IT, NUM>[veclen]; 
		for(IT j =0; j< veclen; ++j)		// create the initial heap 
		{
			if(colinds[j].first != colinds[j].second)	// current != end
			{
				wset[hsize++] = HeapEntry< IT,NUM > ( Adcsc.ir[colinds[j].first], j, Adcsc.numx[colinds[j].first]);  // HeapEntry(key, run, num)
			} 
		}	
		std::make_heap(wset, wset+hsize);
		while(hsize > 0)
		{
			std::pop_heap(wset, wset + hsize);         	// result is stored in wset[hsize-1]
			IT locv = wset[hsize-1].runr;		// relative location of the nonzero in sparse column vector 
			OVT mrhs = SR::multiply(wset[hsize-1].num, numx[locv]);	
		
			if (!SR::returnedSAID())
			{
				if((!indy.empty()) && indy.back() == wset[hsize-1].key)	
				{
					numy.back() = SR::add(numy.back(), mrhs);
				}
				else
				{
					indy.push_back( (int32_t) wset[hsize-1].key);
					numy.push_back(mrhs);	
				}
			}

			if( (++(colinds[locv].first)) != colinds[locv].second)	// current != end
			{
				// runr stays the same !
				wset[hsize-1].key = Adcsc.ir[colinds[locv].first];
				wset[hsize-1].num = Adcsc.numx[colinds[locv].first];  
				std::push_heap(wset, wset+hsize);
			}
			else		--hsize;
		}
		delete [] wset;
	}
}



/**
  * One of the two versions of SpMXSpV with on boolean matrix [uses only Semiring::add()]
  * This version is likely to be more memory efficient than the other one (the one that uses preallocated memory buffers)
  * Because here we don't use a dense accumulation vector but a heap. It will probably be slower though. 
**/
template <class SR, class IT, class IVT, class OVT>
void SpImpl<SR,IT,bool,IVT,OVT>::SpMXSpV(const Dcsc<IT,bool> & Adcsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen,  
			std::vector<int32_t> & indy, std::vector<OVT> & numy)
{   
	IT inf = std::numeric_limits<IT>::min();
	IT sup = std::numeric_limits<IT>::max(); 
	KNHeap< IT, IVT > sHeap(sup, inf); 	// max size: flops

	IT k = 0; 	// index to indx vector
	IT i = 0; 	// index to columns of matrix
	while(i< Adcsc.nzc && k < veclen)
	{
		if(Adcsc.jc[i] < indx[k]) ++i;
		else if(indx[k] < Adcsc.jc[i]) ++k;
		else
		{
			for(IT j=Adcsc.cp[i]; j < Adcsc.cp[i+1]; ++j)	// for all nonzeros in this column
			{
				sHeap.insert(Adcsc.ir[j], numx[k]);	// row_id, num
			}
			++i;
			++k;
		}
	}

	IT row;
	IVT num;
	if(sHeap.getSize() > 0)
	{
		sHeap.deleteMin(&row, &num);
		indy.push_back( (int32_t) row);
		numy.push_back( num );
	}
	while(sHeap.getSize() > 0)
	{
		sHeap.deleteMin(&row, &num);
		if(indy.back() == row)
		{
			numy.back() = SR::add(numy.back(), num);
		}
		else
		{
			indy.push_back( (int32_t) row);
			numy.push_back(num);
		}
	}		
}


/**
 * @param[in,out]   indy,numy,cnts 	{preallocated arrays to be filled}
 * @param[in] 		dspls	{displacements to preallocated indy,numy buffers}
 * This version determines the receiving column neighbor and adjust the indices to the receiver's local index
 * If IVT and OVT are different, then OVT should allow implicit conversion from IVT
**/
template <typename SR, typename IT, typename IVT, class OVT>
void SpImpl<SR,IT,bool,IVT,OVT>::SpMXSpV(const Dcsc<IT,bool> & Adcsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen,  
			int32_t * indy, OVT * numy, int * cnts, int * dspls, int p_c)
{   
	OVT * localy = new OVT[mA];
	BitMap isthere(mA);
	std::vector< std::vector<int32_t> > nzinds(p_c);	// nonzero indices		

	int32_t perproc = mA / p_c;	
	int32_t k = 0; 	// index to indx vector
	IT i = 0; 	// index to columns of matrix
	while(i< Adcsc.nzc && k < veclen)
	{
		if(Adcsc.jc[i] < indx[k]) ++i;
		else if(indx[k] < Adcsc.jc[i]) ++k;
		else
		{
			for(IT j=Adcsc.cp[i]; j < Adcsc.cp[i+1]; ++j)	// for all nonzeros in this column
			{
				int32_t rowid = (int32_t) Adcsc.ir[j];
				if(!isthere.get_bit(rowid))
				{
					int32_t owner = std::min(rowid / perproc, static_cast<int32_t>(p_c-1)); 			
					localy[rowid] = numx[k];	// initial assignment, requires implicit conversion if IVT != OVT
					nzinds[owner].push_back(rowid);
                    isthere.set_bit(rowid);
				}
				else	
				{
					localy[rowid] = SR::add(localy[rowid], numx[k]);
				}	
			}
			++i;
			++k;
		}
	}

	for(int p = 0; p< p_c; ++p)
	{
		sort(nzinds[p].begin(), nzinds[p].end());
		cnts[p] = nzinds[p].size();
		int32_t * locnzinds = &nzinds[p][0];
		int32_t offset = perproc * p;
		for(int i=0; i< cnts[p]; ++i)
		{
			indy[dspls[p]+i] = locnzinds[i] - offset;	// convert to local offset
			numy[dspls[p]+i] = localy[locnzinds[i]]; 	
		}
	}
	delete [] localy;
}


// this version is still very good with splitters
template <typename SR, typename IT, typename IVT, typename OVT>
void SpImpl<SR,IT,bool,IVT,OVT>::SpMXSpV_ForThreading(const Dcsc<IT,bool> & Adcsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen,
                                                      std::vector<int32_t> & indy, std::vector<OVT> & numy, int32_t offset)
{
    std::vector<OVT> localy(mA);
    BitMap isthere(mA);
    std::vector<uint32_t> nzinds;	// nonzero indices
    
    SpMXSpV_ForThreading(Adcsc, mA, indx, numx, veclen, indy, numy, offset, localy, isthere, nzinds);
}



//! We can safely use a SPA here because Adcsc is short (::RowSplit() has already been called on it)
template <typename SR, typename IT, typename IVT, typename OVT>
void SpImpl<SR,IT,bool,IVT,OVT>::SpMXSpV_ForThreading(const Dcsc<IT,bool> & Adcsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen, std::vector<int32_t> & indy, std::vector<OVT> & numy, int32_t offset, std::vector<OVT> & localy, BitMap & isthere, std::vector<uint32_t> & nzinds)
{
	// The following piece of code is not general, but it's more memory efficient than FillColInds
	int32_t k = 0; 	// index to indx vector
	IT i = 0; 	// index to columns of matrix
	while(i< Adcsc.nzc && k < veclen)
	{
		if(Adcsc.jc[i] < indx[k]) ++i;
		else if(indx[k] < Adcsc.jc[i]) ++k;
		else
		{
			for(IT j=Adcsc.cp[i]; j < Adcsc.cp[i+1]; ++j)	// for all nonzeros in this column
			{
				uint32_t rowid = (uint32_t) Adcsc.ir[j];
				if(!isthere.get_bit(rowid))
				{
					localy[rowid] = numx[k];	// initial assignment
					nzinds.push_back(rowid);
					isthere.set_bit(rowid);
				}
				else
				{
					localy[rowid] = SR::add(localy[rowid], numx[k]);
				}	
			}
			++i; ++k;
		}
	}
    int nnzy = nzinds.size();
    integerSort(nzinds.data(), nnzy);
	indy.resize(nnzy);
	numy.resize(nnzy);
	for(int i=0; i< nnzy; ++i)
	{
		indy[i] = nzinds[i] + offset;	// return column-global index and let gespmv determine the receiver's local index
		numy[i] = localy[nzinds[i]]; 	
	}
}





/**
 * SpMXSpV with HeapSort
 * Simply insert entries from columns corresponsing to nonzeros of the input vector into a minHeap
 * Then extract entries from the minHeap
 * Complexity: O(flops*log(flops))
 * offset is the offset of indices in the matrix in case the matrix is split
 * This version is likely to be more memory efficient than the other one (the one that uses preallocated memory buffers)
 * Because here we don't use a dense accumulation vector but a heap. It will probably be slower though.
 **/

template <typename SR, typename IT, typename NT, typename IVT, typename OVT>
void SpMXSpV_HeapSort(const Csc<IT,NT> & Acsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen, std::vector<int32_t> & indy, std::vector<OVT> & numy, int32_t offset)
{
    IT inf = std::numeric_limits<IT>::min();
    IT sup = std::numeric_limits<IT>::max();
    KNHeap< IT, OVT > sHeap(sup, inf);
    
    
    for (int32_t k = 0; k < veclen; ++k)
    {
        IT colid = indx[k];
        for(IT j=Acsc.jc[colid]; j < Acsc.jc[colid+1]; ++j)
        {
            OVT val = SR::multiply( Acsc.num[j], numx[k]);
            sHeap.insert(Acsc.ir[j], val);
        }
    }
    
    IT row;
    OVT num;
    if(sHeap.getSize() > 0)
    {
        sHeap.deleteMin(&row, &num);
        row += offset;
        indy.push_back( (int32_t) row);
        numy.push_back( num );
    }
    while(sHeap.getSize() > 0)
    {
        sHeap.deleteMin(&row, &num);
        row += offset;
        if(indy.back() == row)
        {
            numy.back() = SR::add(numy.back(), num);
        }
        else
        {
            indy.push_back( (int32_t) row);
            numy.push_back(num);
        }
    }
}



template <typename SR, typename IT, typename NT, typename IVT, typename OVT>
void SpMXSpV_Bucket(const Csc<IT,NT> & Acsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen,
                         std::vector<int32_t> & indy, std::vector< OVT > & numy, PreAllocatedSPA<OVT> & SPA)
{
    if(veclen==0)
        return;
    
    double tstart = MPI_Wtime();
    int nthreads=1;
    int rowSplits = SPA.buckets; // SPA must be initialized as checked in SpImpl.h
#ifdef _OPENMP
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
#endif
    if(rowSplits < nthreads)
    {
        std::ostringstream outs;
        outs << "Warning in SpMXSpV_Bucket: " << rowSplits << " buckets are supplied for " << nthreads << " threads\n";
        outs << "4 times the number of threads are recommended when creating PreAllocatedSPA\n";
        SpParHelper::Print(outs.str());
    }
    
    int32_t rowPerSplit = mA / rowSplits;
    
    
    //------------------------------------------------------
    // Step1: count the nnz in each rowsplit of the matrix,
    // because we don't want to waste memory
    // False sharing is not a big problem because it is written outside of the main loop
    //------------------------------------------------------
    
    std::vector<std::vector<int32_t>> bSize(rowSplits, std::vector<int32_t> ( rowSplits, 0));
    std::vector<std::vector<int32_t>> bOffset(rowSplits, std::vector<int32_t> ( rowSplits, 0));
    std::vector<int32_t> sendSize(rowSplits);
    double t0, t1, t2, t3, t4;
#ifdef BENCHMARK_SPMSPV
    t0 = MPI_Wtime();
#endif
    
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for(int b=0; b<rowSplits; b++)
    {
        // evenly balance nnz of x among threads
        int perBucket = veclen/rowSplits;
        int spill = veclen%rowSplits;
        int32_t xstart = b*perBucket + std::min(spill, b);
        int32_t xend = (b+1)*perBucket + std::min(spill, b+1);
        std::vector<int32_t> temp(rowSplits,0);
        for (int32_t i = xstart; i < xend; ++i)
        {
            IT colid = indx[i];
            for(IT j=Acsc.jc[colid]; j < Acsc.jc[colid+1]; ++j)
            {
                uint32_t rowid = (uint32_t) Acsc.ir[j];
                int32_t splitId = rowSplits-1;
                if(rowPerSplit!=0) splitId = (rowid/rowPerSplit > rowSplits-1) ? rowSplits-1 : rowid/rowPerSplit;
                //bSize[b][splitId]++;
                temp[splitId]++;
            }
        }
        int32_t totSend = 0;
        for(int k=0; k<rowSplits; k++)
        {
            bSize[b][k] = temp[k];
            totSend += temp[k];
        }
        sendSize[b] = totSend;
    }
    
    
#ifdef BENCHMARK_SPMSPV
    t1 = MPI_Wtime() - t0;
    t0 = MPI_Wtime();
#endif
    
    
    
    // keep it sequential to avoid fault sharing
    for(int i=1; i<rowSplits; i++)
    {
        for(int j=0; j<rowSplits; j++)
        {
            bOffset[i][j] = bOffset[i-1][j] + bSize[i-1][j];
            bSize[i-1][j] = 0;
        }
    }
    
    std::vector<uint32_t> disp(rowSplits+1);
    int maxBucketSize = -1; // maximum size of a bucket
    disp[0] = 0;
    for(int j=0; j<rowSplits; j++)
    {
        int thisBucketSize = bOffset[rowSplits-1][j] + bSize[rowSplits-1][j];
        disp[j+1] = disp[j] + thisBucketSize;
        bSize[rowSplits-1][j] = 0;
        maxBucketSize = std::max(thisBucketSize, maxBucketSize);
    }
    
    
    
#ifdef BENCHMARK_SPMSPV
    double  tseq = MPI_Wtime() - t0;
#endif
    //------------------------------------------------------
    // Step2: The matrix is traversed column by column and
    // nonzeros each rowsplit of the matrix are compiled together
    //------------------------------------------------------
    // Thread private buckets should fit in L2 cache
#ifndef L2_CACHE_SIZE
#define L2_CACHE_SIZE 256000
#endif
    int THREAD_BUF_LEN = 256;
    int itemsize = sizeof(int32_t) + sizeof(OVT);
    while(true)
    {
        int bufferMem = THREAD_BUF_LEN * rowSplits * itemsize + 8 * rowSplits;
        if(bufferMem>L2_CACHE_SIZE ) THREAD_BUF_LEN/=2;
        else break;
    }
    THREAD_BUF_LEN = std::min(maxBucketSize+1,THREAD_BUF_LEN);
    
#ifdef BENCHMARK_SPMSPV
    t0 = MPI_Wtime();
#endif
    
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        int32_t* tIndSplitA = new int32_t[rowSplits*THREAD_BUF_LEN];
        OVT* tNumSplitA = new OVT[rowSplits*THREAD_BUF_LEN];
        std::vector<int32_t> tBucketSize(rowSplits);
        std::vector<int32_t> tOffset(rowSplits);
#ifdef _OPENMP
#pragma omp for schedule(dynamic,1)
#endif
        for(int b=0; b<rowSplits; b++)
        {
            
            std::fill(tBucketSize.begin(), tBucketSize.end(), 0);
            std::fill(tOffset.begin(), tOffset.end(), 0);
            int perBucket = veclen/rowSplits;
            int spill = veclen%rowSplits;
            int32_t xstart = b*perBucket + std::min(spill, b);
            int32_t xend = (b+1)*perBucket + std::min(spill, b+1);
            
            for (int32_t i = xstart; i < xend; ++i)
            {
                IT colid = indx[i];
                for(IT j=Acsc.jc[colid]; j < Acsc.jc[colid+1]; ++j)
                {
                    OVT val = SR::multiply( Acsc.num[j], numx[i]);
                    uint32_t rowid = (uint32_t) Acsc.ir[j];
                    int32_t splitId = rowSplits-1;
                    if(rowPerSplit!=0) splitId = (rowid/rowPerSplit > rowSplits-1) ? rowSplits-1 : rowid/rowPerSplit;
                    if (tBucketSize[splitId] < THREAD_BUF_LEN)
                    {
                        tIndSplitA[splitId*THREAD_BUF_LEN + tBucketSize[splitId]] = rowid;
                        tNumSplitA[splitId*THREAD_BUF_LEN  + tBucketSize[splitId]++] = val;
                    }
                    else
                    {
                        std::copy(tIndSplitA + splitId*THREAD_BUF_LEN, tIndSplitA + (splitId+1)*THREAD_BUF_LEN, &SPA.indSplitA[disp[splitId] + bOffset[b][splitId]] + tOffset[splitId]);
                        std::copy(tNumSplitA + splitId*THREAD_BUF_LEN, tNumSplitA + (splitId+1)*THREAD_BUF_LEN, &SPA.numSplitA[disp[splitId] + bOffset[b][splitId]] + tOffset[splitId]);
                        tIndSplitA[splitId*THREAD_BUF_LEN] = rowid;
                        tNumSplitA[splitId*THREAD_BUF_LEN] = val;
                        tOffset[splitId] += THREAD_BUF_LEN ;
                        tBucketSize[splitId] = 1;
                    }
                }
            }
            
            for(int splitId=0; splitId<rowSplits; ++splitId)
            {
                if(tBucketSize[splitId]>0)
                {
                    std::copy(tIndSplitA + splitId*THREAD_BUF_LEN, tIndSplitA + splitId*THREAD_BUF_LEN + tBucketSize[splitId], &SPA.indSplitA[disp[splitId] + bOffset[b][splitId]] + tOffset[splitId]);
                    std::copy(tNumSplitA + splitId*THREAD_BUF_LEN, tNumSplitA + splitId*THREAD_BUF_LEN + tBucketSize[splitId], &SPA.numSplitA[disp[splitId] + bOffset[b][splitId]] + tOffset[splitId]);
                }
            }
        }
        delete [] tIndSplitA;
        delete [] tNumSplitA;
    }
    
#ifdef BENCHMARK_SPMSPV
    t2 = MPI_Wtime() - t0;
    t0 = MPI_Wtime();
#endif
    std::vector<uint32_t> nzInRowSplits(rowSplits);
    uint32_t* nzinds = new uint32_t[disp[rowSplits]];
    
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,1)
#endif
    for(int rs=0; rs<rowSplits; ++rs)
    {
        
        for(int i=disp[rs]; i<disp[rs+1] ; i++)
        {
            int32_t lrowid = SPA.indSplitA[i] - rs * rowPerSplit;
            SPA.V_isthereBool[rs][lrowid] = false;
        }
        uint32_t tMergeDisp = disp[rs];
        for(int i=disp[rs]; i<disp[rs+1] ; i++)
        {
            int32_t rowid = SPA.indSplitA[i];
            int32_t lrowid = rowid - rs * rowPerSplit;
            if(!SPA.V_isthereBool[rs][lrowid])// there is no conflict across threads
            {
                SPA.V_localy[0][rowid] = SPA.numSplitA[i];
                nzinds[tMergeDisp++] = rowid;
                SPA.V_isthereBool[rs][lrowid]=true;
            }
            else
            {
                SPA.V_localy[0][rowid] = SR::add(SPA.V_localy[0][rowid], SPA.numSplitA[i]);
            }
        }
        
        integerSort(nzinds + disp[rs], tMergeDisp - disp[rs]);
        nzInRowSplits[rs] = tMergeDisp - disp[rs];
        
    }
    
#ifdef BENCHMARK_SPMSPV
    t3 = MPI_Wtime() - t0;
#endif
    // prefix sum
    std::vector<uint32_t> dispRowSplits(rowSplits+1);
    dispRowSplits[0] = 0;
    for(int i=0; i<rowSplits; i++)
    {
        dispRowSplits[i+1] = dispRowSplits[i] + nzInRowSplits[i];
    }
    
#ifdef BENCHMARK_SPMSPV
    t0 = MPI_Wtime();
#endif
    int nnzy = dispRowSplits[rowSplits];
    indy.resize(nnzy);
    numy.resize(nnzy);
#ifdef BENCHMARK_SPMSPV
    tseq = MPI_Wtime() - t0;
    t0 = MPI_Wtime();
#endif
    
    int  maxNnzInSplit = *std::max_element(nzInRowSplits.begin(),nzInRowSplits.end());
    THREAD_BUF_LEN = std::min(maxNnzInSplit+1,256);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        OVT* tnumy = new OVT [THREAD_BUF_LEN];
        int32_t* tindy = new int32_t [THREAD_BUF_LEN];
       	int curSize, tdisp;
#ifdef _OPENMP
#pragma omp for schedule(dynamic,1)
#endif
        for(int rs=0; rs<rowSplits; rs++)
        {
            curSize = 0;
            tdisp = 0;
            uint32_t * thisind = nzinds + disp[rs];
            std::copy(nzinds+disp[rs], nzinds+disp[rs]+nzInRowSplits[rs], indy.begin()+dispRowSplits[rs]);
            for(int j=0; j<nzInRowSplits[rs]; j++)
            {
                
                if ( curSize < THREAD_BUF_LEN)
                {
                    tnumy[curSize++] = SPA.V_localy[0][thisind[j]];
                }
                else
                {
                    std::copy(tnumy, tnumy+curSize, numy.begin()+dispRowSplits[rs]+tdisp);
                    tdisp += curSize;
                    tnumy[0] = SPA.V_localy[0][thisind[j]];
                    curSize = 1;
                }
            }
            if ( curSize > 0)
            {
                std::copy(tnumy, tnumy+curSize, numy.begin()+dispRowSplits[rs]+tdisp);
            }
        }
        delete [] tnumy;
        delete [] tindy;
    }
    
    
#ifdef BENCHMARK_SPMSPV
    t4 = MPI_Wtime() - t0;
#endif
    
    delete[] nzinds;
    
    
    
#ifdef BENCHMARK_SPMSPV
    double tall = MPI_Wtime() - tstart;
    std::ostringstream outs1;
    outs1 << "Time breakdown of SpMSpV-bucket." << std::endl;
    outs1 << "Estimate buckets: "<< t1 << " Bucketing: " << t2 << " SPA-merge: " << t3 << " Output: " << t4  << " Total: "<< tall << std::endl;
    SpParHelper::Print(outs1.str());
#endif
    
}

}
