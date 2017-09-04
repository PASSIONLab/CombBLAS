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
#include "Deleter.h"
#include "SpParHelper.h"
#include "PBBS/radixSort.h"
#include "Tommy/tommyhashdyn.h"
using namespace std;


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
			vector<int32_t> & indy, vector< OVT > & numy)
{
	int32_t hsize = 0;		
	// colinds dereferences A.ir (valid from colinds[].first to colinds[].second)
	vector< pair<IT,IT> > colinds( (IT) veclen);		
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
		make_heap(wset, wset+hsize);
		while(hsize > 0)
		{
			pop_heap(wset, wset + hsize);         	// result is stored in wset[hsize-1]
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
					push_heap(wset, wset+hsize);	// runr stays the same
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
		make_heap(wset, wset+hsize);
		while(hsize > 0)
		{
			pop_heap(wset, wset + hsize);         	// result is stored in wset[hsize-1]
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
				push_heap(wset, wset+hsize);
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
			vector<int32_t> & indy, vector<OVT> & numy)
{   
	IT inf = numeric_limits<IT>::min();
	IT sup = numeric_limits<IT>::max(); 
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
	vector< vector<int32_t> > nzinds(p_c);	// nonzero indices		

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
					int32_t owner = min(rowid / perproc, static_cast<int32_t>(p_c-1)); 			
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
			indy[dspls[p]+i] = locnzinds[i] - offset;	// conver to local offset
			numy[dspls[p]+i] = localy[locnzinds[i]]; 	
		}
	}
	delete [] localy;
}


// this version is still very good with splitters
template <typename SR, typename IT, typename IVT, typename OVT>
void SpImpl<SR,IT,bool,IVT,OVT>::SpMXSpV_ForThreading(const Dcsc<IT,bool> & Adcsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen,
                                                      vector<int32_t> & indy, vector<OVT> & numy, int32_t offset)
{
    vector<OVT> localy(mA);
    BitMap isthere(mA);
    vector<uint32_t> nzinds;	// nonzero indices
    
    SpMXSpV_ForThreading(Adcsc, mA, indx, numx, veclen, indy, numy, offset, localy, isthere, nzinds);
}



//! We can safely use a SPA here because Adcsc is short (::RowSplit() has already been called on it)
template <typename SR, typename IT, typename IVT, typename OVT>
void SpImpl<SR,IT,bool,IVT,OVT>::SpMXSpV_ForThreading(const Dcsc<IT,bool> & Adcsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen, vector<int32_t> & indy, vector<OVT> & numy, int32_t offset, vector<OVT> & localy, BitMap & isthere, vector<uint32_t> & nzinds)
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








template <typename OVT>
struct tommy_object {
    tommy_node node;
    OVT value;
    uint32_t index;
    
    tommy_object(OVT val, uint32_t ind):value(val), index(ind){}; // constructor
};

template <typename OVT>
int compare(const void* arg, const void* obj)
{
    return *(const uint32_t*)arg != ((const tommy_object<OVT> *)obj)->index;
}

//#define USE_TOMMY


// this version is still very good with splitters
template <typename SR, typename IT, typename IVT, typename OVT>
void SpImpl<SR,IT,bool,IVT,OVT>::SpMXSpV_ForThreading(const Csc<IT,bool> & Acsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen,
                                                      vector<int32_t> & indy, vector<OVT> & numy, int32_t offset)
{
#ifdef USE_TOMMY
    vector<OVT> localy; // won't be used
#else
    vector<OVT> localy(mA);
#endif
    BitMap isthere(mA);
    vector<uint32_t> nzinds;	// nonzero indices
    
    SpMXSpV_ForThreading(Acsc, mA, indx, numx, veclen, indy, numy, offset, localy, isthere, nzinds);
}

/*
template <typename SR, typename IT, typename IVT, typename OVT>
void SpImpl<SR,IT,bool,IVT,OVT>::SpMXSpV_ForThreading(const Csc<IT,bool> & Acsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen,
                                                      vector<int32_t> & indy, vector<OVT> & numy, int32_t offset,
                                                      vector<OVT> & localy, BitMap & isthere, vector<uint32_t> & nzinds)    // these three are pre-allocated buffers
{

#ifdef USE_TOMMY
    tommy_hashdyn hashdyn;
    tommy_hashdyn_init(&hashdyn);
#endif
    
    for (int32_t k = 0; k < veclen; ++k)
    {
        IT colid = indx[k];
        for(IT j=Acsc.jc[colid]; j < Acsc.jc[colid+1]; ++j)	// for all nonzeros in this column
        {
            uint32_t rowid = (uint32_t) Acsc.ir[j];
            if(!isthere.get_bit(rowid))
            {
            #ifdef USE_TOMMY
                tommy_object<OVT> * obj = new tommy_object<OVT>(numx[k], rowid);
                tommy_hashdyn_insert(&hashdyn, &(obj->node), obj, tommy_inthash_u32(obj->index));
            #else
                localy[rowid] = numx[k];	// initial assignment
            #endif
                nzinds.push_back(rowid);
                isthere.set_bit(rowid);
            }
            else
            {
            #ifdef USE_TOMMY
                tommy_object<OVT> * obj = (tommy_object<OVT> *) tommy_hashdyn_search(&hashdyn, compare<OVT>, &rowid, tommy_inthash_u32(rowid));
                obj->value = SR::add(obj->value, numx[k]);
            #else
                localy[rowid] = SR::add(localy[rowid], numx[k]);
            #endif
            }
        }
    }
    int nnzy = nzinds.size();
    integerSort(nzinds.data(), nnzy);
    indy.resize(nnzy);
    numy.resize(nnzy);
    
    for(int i=0; i< nnzy; ++i)
    {
        indy[i] = nzinds[i] + offset;	// return column-global index and let gespmv determine the receiver's local index
    #ifdef USE_TOMMY
        tommy_object<OVT> * obj = (tommy_object<OVT> *) tommy_hashdyn_search(&hashdyn, compare<OVT>, &(nzinds[i]), tommy_inthash_u32(nzinds[i]));
        numy[i] = obj->value;
    #else
        numy[i] = localy[nzinds[i]];
    #endif
    }
    isthere.reset();
    nzinds.clear(); // not necessarily reclaim memory, just make size=0
    
#ifdef USE_TOMMY
    tommy_hashdyn_foreach(&hashdyn, operator delete);
    tommy_hashdyn_done(&hashdyn);
#endif

}
 */



// Heap based approach, just for testing for IPDPS paper
template <typename SR, typename IT, typename IVT, typename OVT>
void SpImpl<SR,IT,bool,IVT,OVT>::SpMXSpV_ForThreading(const Csc<IT,bool> & Acsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen,
                                                      vector<int32_t> & indy, vector<OVT> & numy, int32_t offset,
                                                      vector<OVT> & localy, BitMap & isthere, vector<uint32_t> & nzinds)    // these three are pre-allocated buffers

//template <class SR, class IT, class IVT, class OVT>
//void SpImpl<SR,IT,bool,IVT,OVT>::SpMXSpV(const Csc<IT,bool> & Acsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen, vector<int32_t> & indy, vector<OVT> & numy)
{
    IT inf = numeric_limits<IT>::min();
    IT sup = numeric_limits<IT>::max();
    KNHeap< IT, IVT > sHeap(sup, inf); 	// max size: flops
    
    
    for (int32_t k = 0; k < veclen; ++k)
    {
        IT colid = indx[k];
        for(IT j=Acsc.jc[colid]; j < Acsc.jc[colid+1]; ++j)	// for all nonzeros in this column
        {
            sHeap.insert(Acsc.ir[j], numx[k]);
        }
    }
    
    IT row;
    IVT num;
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





/*
 // previous working version with vectors
template <typename SR, typename IT, typename IVT, typename OVT>
void SpImpl<SR,IT,bool,IVT,OVT>::SpMXSpV_Threaded_2D(const Csc<IT,bool> & Acsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen,
                                                     vector<int32_t> & indy, vector<OVT> & numy, PreAllocatedSPA<IT,OVT> & SPA)
{
    
    int rowSplits = 1, nthreads=1;
#ifdef _OPENMP
#pragma omp parallel
    {
        
        nthreads = omp_get_num_threads();
        rowSplits = nthreads * 4;
    }
#endif
    int32_t rowPerSplit = mA / rowSplits;
    
    
    
    
    
    
    //------------------------------------------------------
    // Step2: The matrix is traversed column by column and
    // nonzeros each rowsplit of the matrix are compiled together
    //------------------------------------------------------
    
    vector<int> bucketSize(rowSplits,0);
    int THREAD_BUF_LEN = 256;
    
    double t0 = MPI_Wtime();
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        double tt = MPI_Wtime();
        int thisThread = omp_get_thread_num();
        vector<int32_t> tIndSplitA(rowSplits*THREAD_BUF_LEN);
        vector<OVT> tNumSplitA(rowSplits*THREAD_BUF_LEN);
        vector<int> tBucketSize(rowSplits,0);
        
#ifdef _OPENMP
#pragma omp for
#endif
        for (int32_t k = 0; k < veclen; ++k)
        {
            IT colid = indx[k];
            for(IT j=Acsc.jc[colid]; j < Acsc.jc[colid+1]; ++j)	// for all nonzeros in this column
            {
                uint32_t rowid = (uint32_t) Acsc.ir[j];
                int32_t splitId = (rowid/rowPerSplit > rowSplits-1) ? rowSplits-1 : rowid/rowPerSplit;
                if (tBucketSize[splitId] < THREAD_BUF_LEN)
                {
                    tIndSplitA[splitId*THREAD_BUF_LEN + tBucketSize[splitId]] = rowid;
                    tNumSplitA[splitId*THREAD_BUF_LEN  + tBucketSize[splitId]++] = numx[k];
                }
                else
                {
                    uint32_t voff = __sync_fetch_and_add (&bucketSize[splitId], THREAD_BUF_LEN);
                    for (uint32_t vk = 0; vk < THREAD_BUF_LEN; ++vk)
                    {
                        SPA.indSplitA[SPA.disp[splitId] + voff + vk] = tIndSplitA[splitId*THREAD_BUF_LEN + vk];
                        SPA.numSplitA[SPA.disp[splitId] + voff + vk] = tNumSplitA[splitId*THREAD_BUF_LEN + vk];
                    }
                    tIndSplitA[splitId*THREAD_BUF_LEN] = rowid;
                    tNumSplitA[splitId*THREAD_BUF_LEN] = numx[k];
                    tBucketSize[splitId] = 1;
                }
            }
        }
        
        for(int rs=0; rs<rowSplits; ++rs)
        {
            if(tBucketSize[rs]>0)
            {
                uint32_t voff = __sync_fetch_and_add (&bucketSize[rs], tBucketSize[rs]);
                for (uint32_t vk = 0; vk < tBucketSize[rs]; ++vk)
                {
                    SPA.indSplitA[SPA.disp[rs] + voff + vk] = tIndSplitA[rs*THREAD_BUF_LEN + vk];
                    SPA.numSplitA[SPA.disp[rs] + voff + vk] = tNumSplitA[rs*THREAD_BUF_LEN + vk];
                }
            }
        }
        cout << MPI_Wtime() - tt << endl;
    }
    double t1 = MPI_Wtime() - t0;
    
    // prefix sum
    vector<uint32_t> disp(rowSplits+1);
    disp[0] = 0;
    for(int i=0; i<rowSplits; i++)
    {
        disp[i+1] = disp[i] + bucketSize[i];
    }
    
    
    
    vector<uint32_t> nzInRowSplits(rowSplits);
    // Ariful: somehow I was not able to make SPA.C_inds working. See the dirty version
    uint32_t* nzinds = static_cast<uint32_t*> (::operator new (sizeof(uint32_t)*disp[rowSplits]));
    

    t0 = MPI_Wtime();
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        
#ifdef _OPENMP
#pragma omp for
#endif
        for(int rs=0; rs<rowSplits; ++rs)
        {
            for(int i=0; i<disp[rs+1]-disp[rs] ; i++)
            //for(int i=disp[rs]; i<disp[rs+1] ; i++)
            {
                SPA.V_isthereBool[0][SPA.indSplitA[SPA.disp[rs]+i]] = false;
            }
            uint32_t tMergeDisp = disp[rs];
            for(int i=0; i<disp[rs+1]-disp[rs] ; i++)
            //for(int i=disp[rs]; i<disp[rs+1] ; i++)
            {
                int32_t rowid = SPA.indSplitA[SPA.disp[rs]+i];
                if(!SPA.V_isthereBool[0][rowid])// there is no conflict across threads
                {
                    SPA.V_localy[0][rowid] = SPA.numSplitA[SPA.disp[rs]+i];
                    nzinds[tMergeDisp++] = rowid;
                    SPA.V_isthereBool[0][rowid]=true;
                }
                else
                {
                    SPA.V_localy[0][rowid] = SR::add(SPA.V_localy[0][rowid], SPA.numSplitA[SPA.disp[rs]+i]);
                }
            }
            
            integerSort(nzinds + disp[rs], tMergeDisp - disp[rs]);
            nzInRowSplits[rs] = tMergeDisp - disp[rs];
            
        }
        
    }
    double t2 = MPI_Wtime ()- t0;
    // prefix sum
    vector<uint32_t> dispRowSplits(rowSplits+1);
    dispRowSplits[0] = 0;
    for(int i=0; i<rowSplits; i++)
    {
        dispRowSplits[i+1] = dispRowSplits[i] + nzInRowSplits[i];
    }
    

    
    int nnzy = dispRowSplits[rowSplits];
    indy.resize(nnzy);
    numy.resize(nnzy);
    
    t0 = MPI_Wtime();
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int rs=0; rs<rowSplits; rs++)
    {
        copy(nzinds+disp[rs], nzinds+disp[rs]+nzInRowSplits[rs], indy.data()+dispRowSplits[rs]);
        for(int j=0; j<nzInRowSplits[rs]; j++)
        {
            numy[j+dispRowSplits[rs]] = SPA.V_localy[0][nzinds[disp[rs]+j]];
            
        }
    }
    double t3 = MPI_Wtime()- t0;
    
    ::operator delete(nzinds);
    
    cout << t1 << " " << t2 << " " << t3 << endl;
}

*/



/*


template <class SR, class IT, class IVT, class OVT>
void SpImpl<SR,IT,bool,IVT,OVT>::SpMXSpV(const Csc<IT,bool> & Acsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen,
                                         vector<int32_t> & indy, vector<OVT> & numy)
{
    IT inf = numeric_limits<IT>::min();
    IT sup = numeric_limits<IT>::max();
    KNHeap< IT, IVT > sHeap(sup, inf); 	// max size: flops
    
    
    for (int32_t k = 0; k < veclen; ++k)
    {
        IT colid = indx[k];
        for(IT j=Acsc.jc[colid]; j < Acsc.jc[colid+1]; ++j)	// for all nonzeros in this column
        {
            sHeap.insert(Acsc.ir[j], numx[k]);
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


*/



// generalized version for non-boolean matrices
// This exactly same as the boolean one. just replaced bool with NT

template <typename SR, typename IT, typename NT, typename IVT, typename OVT>
void SpImpl<SR,IT,NT,IVT,OVT>::SpMXSpV_Threaded_2D(const Csc<IT,NT> & Acsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen,
                                                     int32_t* & indy, OVT* & numy, int & nnzy, PreAllocatedSPA<IT,OVT> & SPA)
{
    if(veclen==0)
    {
        nnzy=0;
        // just to avoid delete crash elsewhere!!
        // TODO: improve memeory management
        indy = new int32_t[nnzy];
        numy = new OVT[nnzy];
        return;
    }
    double tstart = MPI_Wtime();
    int rowSplits = 1, nthreads=1;
#ifdef _OPENMP
#pragma omp parallel
    {
        
        nthreads = omp_get_num_threads();
    }
#endif
    rowSplits = nthreads * 4;
    //	rowSplits = 48;
    int32_t rowPerSplit = mA / rowSplits;
    
    
    //------------------------------------------------------
    // Step1: count the nnz in each rowsplit of the matrix,
    // because we don't want to waste memory
    // False sharing is not a big problem because it is written outside of the main loop
    //------------------------------------------------------
    
    vector<vector<int32_t>> bSize(rowSplits, std::vector<int32_t> ( rowSplits, 0));
    vector<vector<int32_t>> bOffset(rowSplits, std::vector<int32_t> ( rowSplits, 0));
    vector<int32_t> sendSize(rowSplits);
    double t0, t1, t2, t3, t4;
#ifdef TIMING
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
        int32_t xstart = b*perBucket + min(spill, b);
        int32_t xend = (b+1)*perBucket + min(spill, b+1);
        vector<int32_t> temp(rowSplits,0);
        for (int32_t i = xstart; i < xend; ++i)
        {
            IT colid = indx[i];
            for(IT j=Acsc.jc[colid]; j < Acsc.jc[colid+1]; ++j)
            {
                uint32_t rowid = (uint32_t) Acsc.ir[j];
                int32_t splitId = (rowid/rowPerSplit > rowSplits-1) ? rowSplits-1 : rowid/rowPerSplit;
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
    
    
#ifdef TIMING
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
    
    vector<uint32_t> disp(rowSplits+1);
    int maxBucketSize = -1; // maximum size of a bucket
    disp[0] = 0;
    for(int j=0; j<rowSplits; j++)
    {
        int thisBucketSize = bOffset[rowSplits-1][j] + bSize[rowSplits-1][j];
        disp[j+1] = disp[j] + thisBucketSize;
        bSize[rowSplits-1][j] = 0;
        maxBucketSize = max(thisBucketSize, maxBucketSize);
    }
    
    
    
#ifdef TIMING
    double  tseq = MPI_Wtime() - t0;
#endif
    //------------------------------------------------------
    // Step2: The matrix is traversed column by column and
    // nonzeros each rowsplit of the matrix are compiled together
    //------------------------------------------------------
    // A good discussion about memory initialization is discussed here
    // http://stackoverflow.com/questions/7546620/operator-new-initializes-memory-to-zero
    
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
    THREAD_BUF_LEN = min(maxBucketSize+1,THREAD_BUF_LEN);
    
#ifdef TIMING
    t0 = MPI_Wtime();
#endif
    
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        int32_t* tIndSplitA = new int32_t[rowSplits*THREAD_BUF_LEN];
        OVT* tNumSplitA = new OVT[rowSplits*THREAD_BUF_LEN];
        vector<int32_t> tBucketSize(rowSplits);
        vector<int32_t> tOffset(rowSplits);
#ifdef _OPENMP
#pragma omp for schedule(dynamic,1)
#endif
        for(int b=0; b<rowSplits; b++)
        {
            
            fill(tBucketSize.begin(), tBucketSize.end(), 0);
            fill(tOffset.begin(), tOffset.end(), 0);
            int perBucket = veclen/rowSplits;
            int spill = veclen%rowSplits;
            int32_t xstart = b*perBucket + min(spill, b);
            int32_t xend = (b+1)*perBucket + min(spill, b+1);
            
            for (int32_t i = xstart; i < xend; ++i)
            {
                IT colid = indx[i];
                for(IT j=Acsc.jc[colid]; j < Acsc.jc[colid+1]; ++j)
                {
                    OVT val = SR::multiply( Acsc.num[j], numx[i]);
                    uint32_t rowid = (uint32_t) Acsc.ir[j];
                    int32_t splitId = (rowid/rowPerSplit > rowSplits-1) ? rowSplits-1 : rowid/rowPerSplit;
                    if (tBucketSize[splitId] < THREAD_BUF_LEN)
                    {
                        tIndSplitA[splitId*THREAD_BUF_LEN + tBucketSize[splitId]] = rowid;
                        tNumSplitA[splitId*THREAD_BUF_LEN  + tBucketSize[splitId]++] = val;
                    }
                    else
                    {
                        copy(tIndSplitA + splitId*THREAD_BUF_LEN, tIndSplitA + (splitId+1)*THREAD_BUF_LEN, &SPA.indSplitA[disp[splitId] + bOffset[b][splitId]] + tOffset[splitId]);
                        copy(tNumSplitA + splitId*THREAD_BUF_LEN, tNumSplitA + (splitId+1)*THREAD_BUF_LEN, &SPA.numSplitA[disp[splitId] + bOffset[b][splitId]] + tOffset[splitId]);
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
                    copy(tIndSplitA + splitId*THREAD_BUF_LEN, tIndSplitA + splitId*THREAD_BUF_LEN + tBucketSize[splitId], &SPA.indSplitA[disp[splitId] + bOffset[b][splitId]] + tOffset[splitId]);
                    copy(tNumSplitA + splitId*THREAD_BUF_LEN, tNumSplitA + splitId*THREAD_BUF_LEN + tBucketSize[splitId], &SPA.numSplitA[disp[splitId] + bOffset[b][splitId]] + tOffset[splitId]);
                }
            }
        }
        delete [] tIndSplitA;
        delete [] tNumSplitA;
    }
    
#ifdef TIMING
    t2 = MPI_Wtime() - t0;
    t0 = MPI_Wtime();
#endif
    vector<uint32_t> nzInRowSplits(rowSplits);
    // Ariful: somehow I was not able to make SPA.C_inds working. See the dirty version
    uint32_t* nzinds = new uint32_t[disp[rowSplits]];
    
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,1)
#endif
    for(int rs=0; rs<rowSplits; ++rs)
    {
        
        for(int i=disp[rs]; i<disp[rs+1] ; i++)
        {
            SPA.V_isthereBool[0][SPA.indSplitA[i]] = false;
        }
        uint32_t tMergeDisp = disp[rs];
        for(int i=disp[rs]; i<disp[rs+1] ; i++)
        {
            int32_t rowid = SPA.indSplitA[i];
            if(!SPA.V_isthereBool[0][rowid])// there is no conflict across threads
            {
                SPA.V_localy[0][rowid] = SPA.numSplitA[i];
                nzinds[tMergeDisp++] = rowid;
                SPA.V_isthereBool[0][rowid]=true;
            }
            else
            {
                SPA.V_localy[0][rowid] = SR::add(SPA.V_localy[0][rowid], SPA.numSplitA[i]);
            }
        }
        
#ifndef BENCHMARK_SPMSPV
        integerSort(nzinds + disp[rs], tMergeDisp - disp[rs]);
#endif
        nzInRowSplits[rs] = tMergeDisp - disp[rs];
        
    }
    
#ifdef TIMING
    t3 = MPI_Wtime() - t0;
#endif
    // prefix sum
    vector<uint32_t> dispRowSplits(rowSplits+1);
    dispRowSplits[0] = 0;
    for(int i=0; i<rowSplits; i++)
    {
        dispRowSplits[i+1] = dispRowSplits[i] + nzInRowSplits[i];
    }
    
#ifdef TIMING
    t0 = MPI_Wtime();
#endif
    nnzy = dispRowSplits[rowSplits];
    indy = new int32_t[nnzy];
    numy = new OVT[nnzy];
#ifdef TIMING
    tseq = MPI_Wtime() - t0;
    t0 = MPI_Wtime();
#endif
    
    int  maxNnzInSplit = *std::max_element(nzInRowSplits.begin(),nzInRowSplits.end());
    THREAD_BUF_LEN = min(maxNnzInSplit+1,256);
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
            copy(nzinds+disp[rs], nzinds+disp[rs]+nzInRowSplits[rs], indy+dispRowSplits[rs]);
            for(int j=0; j<nzInRowSplits[rs]; j++)
            {
                
                if ( curSize < THREAD_BUF_LEN)
                {
                    //tindy[curSize] = thisind[j];
                    tnumy[curSize++] = SPA.V_localy[0][thisind[j]];
                }
                else
                {
                    //copy(tindy, tindy+curSize, indy+dispRowSplits[rs]+tdisp);
                    copy(tnumy, tnumy+curSize, numy+dispRowSplits[rs]+tdisp);
                    tdisp += curSize;
                    tnumy[0] = SPA.V_localy[0][thisind[j]];
                    curSize = 1;
                }
                //numy[j+dispRowSplits[rs]] = SPA.V_localy[0][nzinds[disp[rs]+j]];
                
            }
            if ( curSize > 0)
            {
                copy(tnumy, tnumy+curSize, numy+dispRowSplits[rs]+tdisp);
                //copy(tindy, tindy+curSize, indy+dispRowSplits[rs]+tdisp);
            }
            //copy(nzinds+disp[rs], nzinds+disp[rs]+nzInRowSplits[rs], numy+dispRowSplits[rs]);
        }
        delete [] tnumy;
        delete [] tindy;
    }
    
    
#ifdef TIMING
    t4 = MPI_Wtime() - t0;
#endif
    
    delete[] nzinds;
    
    
    
#ifdef TIMING
    double tall = MPI_Wtime() - tstart;
    cout << t1 << " " << t2 << " " << t3 << " " << t4  << tall << endl;
#endif
    
}











template <typename SR, typename IT, typename IVT, typename OVT>
void SpImpl<SR,IT,bool,IVT,OVT>::SpMXSpV_Threaded_2D(const Dcsc<IT,bool> & Adcsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen, int32_t* & indy, OVT* & numy, int & nnzy, PreAllocatedSPA<IT,OVT> & SPA)
{
    SpParHelper::Print("2D SpMSpV is not supported for Dcsc yet!\n");
}


template <typename SR, typename IT, typename IVT, typename OVT>
void SpImpl<SR,IT,bool,IVT,OVT>::SpMXSpV_Threaded_2D(const Csc<IT,bool> & Acsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen,
                                                     int32_t* & indy, OVT* & numy, int & nnzy, PreAllocatedSPA<IT,OVT> & SPA)
{
    if(veclen==0)
    {
        nnzy=0;
        // just to avoid delete crash elsewhere!!
        // TODO: improve memeory management
        indy = new int32_t[nnzy];
        numy = new OVT[nnzy];
        return;
    }
    double tstart = MPI_Wtime();
    int rowSplits = 1, nthreads=1;
#ifdef _OPENMP
#pragma omp parallel
    {
        
        nthreads = omp_get_num_threads();
    }
#endif
    rowSplits = nthreads * 4;
    //	rowSplits = 48;
    int32_t rowPerSplit = mA / rowSplits;
    
    
    //------------------------------------------------------
    // Step1: count the nnz in each rowsplit of the matrix,
    // because we don't want to waste memory
    // False sharing is not a big problem because it is written outside of the main loop
    //------------------------------------------------------
    
    vector<vector<int32_t>> bSize(rowSplits, std::vector<int32_t> ( rowSplits, 0));
    vector<vector<int32_t>> bOffset(rowSplits, std::vector<int32_t> ( rowSplits, 0));
    vector<int32_t> sendSize(rowSplits);
    double t0, t1, t2, t3, t4;
#ifdef TIMING
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
        int32_t xstart = b*perBucket + min(spill, b);
        int32_t xend = (b+1)*perBucket + min(spill, b+1);
        vector<int32_t> temp(rowSplits,0);
        for (int32_t i = xstart; i < xend; ++i)
        {
            IT colid = indx[i];
            for(IT j=Acsc.jc[colid]; j < Acsc.jc[colid+1]; ++j)
            {
                uint32_t rowid = (uint32_t) Acsc.ir[j];
                int32_t splitId = (rowid/rowPerSplit > rowSplits-1) ? rowSplits-1 : rowid/rowPerSplit;
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
    
    
#ifdef TIMING
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
    
    vector<uint32_t> disp(rowSplits+1);
    int maxBucketSize = -1; // maximum size of a bucket
    disp[0] = 0;
    for(int j=0; j<rowSplits; j++)
    {
        int thisBucketSize = bOffset[rowSplits-1][j] + bSize[rowSplits-1][j];
        disp[j+1] = disp[j] + thisBucketSize;
        bSize[rowSplits-1][j] = 0;
        maxBucketSize = max(thisBucketSize, maxBucketSize);
    }

    
    
#ifdef TIMING
    double  tseq = MPI_Wtime() - t0;
#endif
    //------------------------------------------------------
    // Step2: The matrix is traversed column by column and
    // nonzeros each rowsplit of the matrix are compiled together
    //------------------------------------------------------
    // A good discussion about memory initialization is discussed here
    // http://stackoverflow.com/questions/7546620/operator-new-initializes-memory-to-zero

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
    THREAD_BUF_LEN = min(maxBucketSize+1,THREAD_BUF_LEN);
    
#ifdef TIMING
    t0 = MPI_Wtime();
#endif
    
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        int32_t* tIndSplitA = new int32_t[rowSplits*THREAD_BUF_LEN];
        OVT* tNumSplitA = new OVT[rowSplits*THREAD_BUF_LEN];
        vector<int32_t> tBucketSize(rowSplits);
        vector<int32_t> tOffset(rowSplits);
#ifdef _OPENMP
#pragma omp for schedule(dynamic,1)
#endif
        for(int b=0; b<rowSplits; b++)
        {
            
            fill(tBucketSize.begin(), tBucketSize.end(), 0);
            fill(tOffset.begin(), tOffset.end(), 0);
            int perBucket = veclen/rowSplits;
            int spill = veclen%rowSplits;
            int32_t xstart = b*perBucket + min(spill, b);
            int32_t xend = (b+1)*perBucket + min(spill, b+1);
            
            for (int32_t i = xstart; i < xend; ++i)
            {
                IT colid = indx[i];
                for(IT j=Acsc.jc[colid]; j < Acsc.jc[colid+1]; ++j)
                {
                    OVT val = SR::multiply( Acsc.num[j], numx[i]);
                    uint32_t rowid = (uint32_t) Acsc.ir[j];
                    int32_t splitId = (rowid/rowPerSplit > rowSplits-1) ? rowSplits-1 : rowid/rowPerSplit;
                    if (tBucketSize[splitId] < THREAD_BUF_LEN)
                    {
                        tIndSplitA[splitId*THREAD_BUF_LEN + tBucketSize[splitId]] = rowid;
                        tNumSplitA[splitId*THREAD_BUF_LEN  + tBucketSize[splitId]++] = val;
                    }
                    else
                    {
                        copy(tIndSplitA + splitId*THREAD_BUF_LEN, tIndSplitA + (splitId+1)*THREAD_BUF_LEN, &SPA.indSplitA[disp[splitId] + bOffset[b][splitId]] + tOffset[splitId]);
                        copy(tNumSplitA + splitId*THREAD_BUF_LEN, tNumSplitA + (splitId+1)*THREAD_BUF_LEN, &SPA.numSplitA[disp[splitId] + bOffset[b][splitId]] + tOffset[splitId]);
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
                    copy(tIndSplitA + splitId*THREAD_BUF_LEN, tIndSplitA + splitId*THREAD_BUF_LEN + tBucketSize[splitId], &SPA.indSplitA[disp[splitId] + bOffset[b][splitId]] + tOffset[splitId]);
                    copy(tNumSplitA + splitId*THREAD_BUF_LEN, tNumSplitA + splitId*THREAD_BUF_LEN + tBucketSize[splitId], &SPA.numSplitA[disp[splitId] + bOffset[b][splitId]] + tOffset[splitId]);
                }
            }
        }
        delete [] tIndSplitA;
        delete [] tNumSplitA;
    }
    
#ifdef TIMING
    t2 = MPI_Wtime() - t0;
    t0 = MPI_Wtime();
#endif
    vector<uint32_t> nzInRowSplits(rowSplits);
    // Ariful: somehow I was not able to make SPA.C_inds working. See the dirty version
    uint32_t* nzinds = new uint32_t[disp[rowSplits]];
    
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,1)
#endif
    for(int rs=0; rs<rowSplits; ++rs)
    {
        
        for(int i=disp[rs]; i<disp[rs+1] ; i++)
        {
            SPA.V_isthereBool[0][SPA.indSplitA[i]] = false;
        }
        uint32_t tMergeDisp = disp[rs];
        for(int i=disp[rs]; i<disp[rs+1] ; i++)
        {
            int32_t rowid = SPA.indSplitA[i];
            if(!SPA.V_isthereBool[0][rowid])// there is no conflict across threads
            {
                SPA.V_localy[0][rowid] = SPA.numSplitA[i];
                nzinds[tMergeDisp++] = rowid;
                SPA.V_isthereBool[0][rowid]=true;
            }
            else
            {
                SPA.V_localy[0][rowid] = SR::add(SPA.V_localy[0][rowid], SPA.numSplitA[i]);
            }
        }
        
#ifndef BENCHMARK_SPMSPV
        integerSort(nzinds + disp[rs], tMergeDisp - disp[rs]);
#endif
        nzInRowSplits[rs] = tMergeDisp - disp[rs];
        
    }
    
#ifdef TIMING
    t3 = MPI_Wtime() - t0;
#endif
    // prefix sum
    vector<uint32_t> dispRowSplits(rowSplits+1);
    dispRowSplits[0] = 0;
    for(int i=0; i<rowSplits; i++)
    {
        dispRowSplits[i+1] = dispRowSplits[i] + nzInRowSplits[i];
    }

#ifdef TIMING
    t0 = MPI_Wtime();
#endif
    nnzy = dispRowSplits[rowSplits];
    indy = new int32_t[nnzy];
    numy = new OVT[nnzy];
#ifdef TIMING
    tseq = MPI_Wtime() - t0;
    t0 = MPI_Wtime();
#endif
    
    int  maxNnzInSplit = *std::max_element(nzInRowSplits.begin(),nzInRowSplits.end());
    THREAD_BUF_LEN = min(maxNnzInSplit+1,256);
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
            copy(nzinds+disp[rs], nzinds+disp[rs]+nzInRowSplits[rs], indy+dispRowSplits[rs]);
            for(int j=0; j<nzInRowSplits[rs]; j++)
            {
                
                if ( curSize < THREAD_BUF_LEN)
                {
                    //tindy[curSize] = thisind[j];
                    tnumy[curSize++] = SPA.V_localy[0][thisind[j]];
                }
                else
                {
                    //copy(tindy, tindy+curSize, indy+dispRowSplits[rs]+tdisp);
                    copy(tnumy, tnumy+curSize, numy+dispRowSplits[rs]+tdisp);
                    tdisp += curSize;
                    tnumy[0] = SPA.V_localy[0][thisind[j]];
                    curSize = 1;
                }
                //numy[j+dispRowSplits[rs]] = SPA.V_localy[0][nzinds[disp[rs]+j]];
                
            }
            if ( curSize > 0)
            {
                copy(tnumy, tnumy+curSize, numy+dispRowSplits[rs]+tdisp);
                //copy(tindy, tindy+curSize, indy+dispRowSplits[rs]+tdisp);
            }
            //copy(nzinds+disp[rs], nzinds+disp[rs]+nzInRowSplits[rs], numy+dispRowSplits[rs]);
        }
        delete [] tnumy;
        delete [] tindy;
    }
    
    
#ifdef TIMING
    t4 = MPI_Wtime() - t0;
#endif
    
    delete[] nzinds;
    
    
    
#ifdef TIMING
    double tall = MPI_Wtime() - tstart;
    cout << t1 << " " << t2 << " " << t3 << " " << t4  << tall << endl;
#endif
    
}








