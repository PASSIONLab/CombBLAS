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

#include "SpImpl.h"

//! base template version [full use of the semiring add() and multiply()]
//! indx vector practically keeps column numbers requested from A
template <class SR, class IT, class NUM, class IVT, class OVT>
void SpImpl<SR,IT,NUM,IVT,OVT>::SpMXSpV(const Dcsc<IT,NUM> & Adcsc, IT mA, const IT * indx, const IVT * numx, IT veclen,  
			vector<IT> & indy, vector< OVT > & numy)
{
	HeapEntry<IT, IVT> * wset = new HeapEntry<IT, IVT>[veclen]; 

	// colinds dereferences A.ir (valid from colinds[].first to colinds[].second)
	vector< pair<IT,IT> > colinds(veclen);		
	Adcsc.FillColInds(indx, veclen, colinds, NULL, 0);	// csize is irrelevant if aux is NULL	
	IT hsize = 0;		
	for(IT j =0; j< veclen; ++j)		// create the initial heap 
	{
		if(colinds[j].first != colinds[j].second)	// current != end
		{
			// HeapEntry(key, run, num)
			wset[hsize++] = HeapEntry< IT,IVT > (Adcsc.ir[colinds[j].first], j, Adcsc.numx[colinds[j].first]);
		} 
	}	
	make_heap(wset, wset+hsize);

	while(hsize > 0)
	{
		pop_heap(wset, wset + hsize);         	// result is stored in wset[hsize-1]
		IT locv = wset[hsize-1].runr;		// relative location of the nonzero in sparse column vector 
		OVT mrhs = SR::multiply(wset[hsize-1].num, numx[locv]);
		if((!indy.empty()) && indy.back() == wset[hsize-1].key)	
		{
			numy.back() = SR::add(numy.back(), mrhs);
		}
		else
		{
			indy.push_back(wset[hsize-1].key);
			numy.push_back(mrhs);	
		}
		if( (++(colinds[locv].first)) != colinds[locv].second)	// current != end
		{
			// runr stays the same !
			wset[hsize-1].key = Adcsc.ir[colinds[locv].first];
			wset[hsize-1].num = Adcsc.numx[colinds[locv].first];  
			push_heap(wset, wset+hsize);
		}
		else
		{
			--hsize;
		}
	}
	delete [] wset;
}


/**
  * One of the two versions of SpMXSpV with on boolean matrix [uses only Semiring::add()]
  * This version is likely to be more memory efficient than the other one (the one that uses preallocated memory buffers)
  * Because here we don't use a dense accumulation vector but a heap. It will probably be slower though. 
**/
template <class SR, class IT, class IVT, class OVT>
void SpImpl<SR,IT,bool,IVT,OVT>::SpMXSpV(const Dcsc<IT,bool> & Adcsc, IT mA, const IT * indx, const IVT * numx, IT veclen,  
			vector<IT> & indy, vector<OVT> & numy)
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
		indy.push_back(row);
		numy.push_back(num);
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
			indy.push_back(row);
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
	bool * isthere = new bool[mA];
	fill(isthere, isthere+mA, false);
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
				if(!isthere[rowid])
				{
					int32_t owner = min(rowid / perproc, static_cast<int32_t>(p_c-1)); 			
					localy[rowid] = numx[k];	// initial assignment, requires implicit conversion if IVT != OVT
					nzinds[owner].push_back(rowid);
					isthere[rowid] = true;
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
	delete [] isthere;
}



template <typename SR, typename IT, typename IVT, typename OVT>
void SpImpl<SR,IT,bool,IVT,OVT>::SpMXSpV_ForThreading(const Dcsc<IT,bool> & Adcsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen,  
			vector<int32_t> & indy, vector<OVT> & numy, int32_t offset)
{   
	OVT * localy = new OVT[mA];
	bool * isthere = new bool[mA];
	fill(isthere, isthere+mA, false);
	vector<int32_t> nzinds;	// nonzero indices		

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
				int32_t rowid = (int32_t) Adcsc.ir[j];
				if(!isthere[rowid])
				{
					localy[rowid] = numx[k];	// initial assignment
					nzinds.push_back(rowid);
					isthere[rowid] = true;
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

	sort(nzinds.begin(), nzinds.end());
	int nnzy = nzinds.size();
	indy.resize(nnzy);
	numy.resize(nnzy);
	for(int i=0; i< nnzy; ++i)
	{
		indy[i] = nzinds[i] + offset;	// return column-global index and let gespmv determine the receiver's local index
		numy[i] = localy[nzinds[i]]; 	
	}
	delete [] localy;
	delete [] isthere;
}


