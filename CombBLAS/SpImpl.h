#ifndef _SP_IMPL_H_
#define _SP_IMPL_H_


#include <iostream>
#include <vector>
#include "promote.h"
using namespace std;


template <class IT, class NT>
class Dcsc;

template <class SR, class IT, class NT1, class NT2>
struct SpImpl;


template <class SR, class IT, class NT1, class NT2>
void SpMXSpV(const Dcsc<IT,NT1> & Adcsc, IT mA, const IT * indx, const NT2 * numx, IT veclen,  
		vector<IT> & indy, vector< typename promote_trait<NT1,NT2>::T_promote > & numy)
{
	SpImpl<SR,IT,NT1,NT2>::SpMXSpV(Adcsc, mA, indx, numx, veclen, indy, numy);	// don't touch this
};


template <class SR, class IT, class NT1, class NT2>
void SpMXSpV(const Dcsc<IT,NT1> & Adcsc, IT mA, const IT * indx, const NT2 * numx, IT veclen,  
		IT * indy, typename promote_trait<NT1,NT2>::T_promote * numy, int * cnts, int * dspls, int p_c)
{
	SpImpl<SR,IT,NT1,NT2>::SpMXSpV(Adcsc, mA, indx, numx, veclen, indy, numy, cnts, dspls,p_c);	// don't touch this
};

template <class SR, class IT, class NT1, class NT2>
void SpMXSpV_ForThreading(const Dcsc<IT,NT1> & Adcsc, IT mA, const IT * indx, const NT2 * numx, IT veclen,  
		vector<IT> & indy, vector< typename promote_trait<NT1,NT2>::T_promote > & numy, IT offset)
{
	SpImpl<SR,IT,NT1,NT2>::SpMXSpV_ForThreading(Adcsc, mA, indx, numx, veclen, indy, numy, offset);	// don't touch this
};


template <class SR, class IT, class NT1, class NT2>
struct SpImpl
{
	static void SpMXSpV(const Dcsc<IT,NT1> & Adcsc, IT mA, const IT * indx, const NT2 * numx, IT veclen,  
			vector<IT> & indy, vector< typename promote_trait<NT1,NT2>::T_promote > & numy);	// specialize this

	static void SpMXSpV(const Dcsc<IT,NT1> & Adcsc, IT mA, const IT * indx, const NT2 * numx, IT veclen,  
			IT * indy, typename promote_trait<NT1,NT2>::T_promote * numy, int * cnts, int * dspls, int p_c);

	static void SpMXSpV_ForThreading(const Dcsc<IT,NT1> & Adcsc, IT mA, const IT * indx, const NT2 * numx, IT veclen,  
			vector<IT> & indy, vector< typename promote_trait<NT1,NT2>::T_promote > & numy, IT offset);
};


template <class SR, class IT, class NT>
struct SpImpl<SR,IT,bool, NT>
{
	static void SpMXSpV(const Dcsc<IT,bool> & Adcsc, IT mA, const IT * indx, const NT * numx, IT veclen,  
			vector<IT> & indy, vector< NT > & numy);	

	static void SpMXSpV(const Dcsc<IT,bool> & Adcsc, IT mA, const IT * indx, const NT * numx, IT veclen,  
			IT * indy, NT * numy, int * cnts, int * dspls, int p_c);

	static void SpMXSpV_ForThreading(const Dcsc<IT,bool> & Adcsc, IT mA, const IT * indx, const NT * numx, IT veclen,  
			vector<IT> & indy, vector<NT> & numy, IT offset);
};


// base template version
// indx vector practically keeps column numbers requested from A
template <typename SR, typename IT, typename NT1, typename NT2>
void SpImpl<SR,IT,NT1,NT2>::SpMXSpV(const Dcsc<IT,NT1> & Adcsc, IT mA, const IT * indx, const NT2 * numx, IT veclen,  
			vector<IT> & indy, vector< typename promote_trait<NT1,NT2>::T_promote > & numy)
{
	typedef typename promote_trait<NT1,NT2>::T_promote T_promote;     
	HeapEntry<IT, NT1> * wset = new HeapEntry<IT, NT1>[veclen]; 

	// colinds dereferences A.ir (valid from colinds[].first to colinds[].second)
	vector< pair<IT,IT> > colinds(veclen);		
	Adcsc.FillColInds(indx, veclen, colinds, NULL, 0);	// csize is irrelevant if aux is NULL	
	IT hsize = 0;		
	for(IT j =0; j< veclen; ++j)		// create the initial heap 
	{
		if(colinds[j].first != colinds[j].second)	// current != end
		{
			// HeapEntry(key, run, num)
			wset[hsize++] = HeapEntry< IT,NT1 > (Adcsc.ir[colinds[j].first], j, Adcsc.numx[colinds[j].first]);
		} 
	}	
	make_heap(wset, wset+hsize);

	while(hsize > 0)
	{
		pop_heap(wset, wset + hsize);         	// result is stored in wset[hsize-1]
		IT locv = wset[hsize-1].runr;		// relative location of the nonzero in sparse column vector 
		T_promote mrhs = SR::multiply(wset[hsize-1].num, numx[locv]);
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


// This version is likely to be more memory efficient than the other signature (the one that uses preallocated memory buffers)
// Because here we don't use a dense accumulation vector but a heap
template <typename SR, typename IT, typename NT>
void SpImpl<SR,IT,bool,NT>::SpMXSpV(const Dcsc<IT,bool> & Adcsc, IT mA, const IT * indx, const NT * numx, IT veclen,  
			vector<IT> & indy, vector<NT> & numy)
{   
	IT inf = numeric_limits<IT>::min();
	IT sup = numeric_limits<IT>::max(); 
       	KNHeap< IT, NT > sHeap(sup, inf); 	// max size: flops

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
	NT num;
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
 * @param[in,out]   	indy,numy,cnts 	{preallocated arrays to be filled}
 * @param[in] 		dspls	{displacements to preallocated indy,numy buffers}
 * This version determines the receiving column neighbor and adjust the indices to the receiver's local index
**/
template <typename SR, typename IT, typename NT>
void SpImpl<SR,IT,bool,NT>::SpMXSpV(const Dcsc<IT,bool> & Adcsc, IT mA, const IT * indx, const NT * numx, IT veclen,  
			IT * indy, NT * numy, int * cnts, int * dspls, int p_c)
{   
	NT * localy = new NT[mA];
	bool * isthere = new bool[mA];
	fill(isthere, isthere+mA, false);
	vector< vector<IT> > nzinds(p_c);	// nonzero indices		

	IT perproc = mA / static_cast<IT>(p_c);	
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
				IT rowid = Adcsc.ir[j];
				if(!isthere[rowid])
				{
					localy[rowid] = numx[k];	// initial assignment
					IT owner = min(rowid / perproc, static_cast<IT>(p_c-1)); 			
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
		IT * locnzinds = &nzinds[p][0];
		IT offset = perproc * p;
		for(int i=0; i< cnts[p]; ++i)
		{
			indy[dspls[p]+i] = locnzinds[i] - offset;	// conver to local offset
			numy[dspls[p]+i] = localy[locnzinds[i]]; 	
		}
	}
	delete [] localy;
	delete [] isthere;
}


template <typename SR, typename IT, typename NT>
void SpImpl<SR,IT,bool,NT>::SpMXSpV_ForThreading(const Dcsc<IT,bool> & Adcsc, IT mA, const IT * indx, const NT * numx, IT veclen,  
			vector<IT> & indy, vector<NT> & numy, IT offset)
{   
	NT * localy = new NT[mA];
	bool * isthere = new bool[mA];
	fill(isthere, isthere+mA, false);
	vector<IT> nzinds;	// nonzero indices		

	// The following piece of code is not general, but it's more memory efficient than FillColInds
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
				IT rowid = Adcsc.ir[j];
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

//////////////////////////////////////////////////////////////////////////////////
// Apply

// based on base SpMV above
template <typename _BinaryOperation, typename IT, typename NT1, typename NT2>
void SpColWiseApply(const Dcsc<IT,NT1> & Adcsc, IT mA, IT nA, const IT * indx, const NT2 * numx, IT veclen, _BinaryOperation __binary_op)
{
	// colinds dereferences A.ir (valid from colinds[].first to colinds[].second)
	vector< pair<IT,IT> > colinds(veclen);		

	float cf  = static_cast<float>(nA+1) / static_cast<float>(Adcsc.nzc);
	IT csize = static_cast<IT>(ceil(cf));   // chunk size

#ifdef UNIQUEMATRIXSPMV
	static IT* spmvaux = NULL;	// set only once over the whole program execution
	if(spmvaux == NULL)
	{
		Adcsc.ConstructAux(nA, spmvaux);
	}
#else
	// ABAB: Short term fix (may have performance hit or not)
	// Long term idea is to piggyback aux to dcsc in a proper way.
	IT* spmvaux = NULL;
#endif

	Adcsc.FillColInds(indx, veclen, colinds, spmvaux, csize);	// csize is irrelevant if aux is NULL	
	for(IT j =0; j< veclen; ++j)		// create the initial heap 
	{
		while(colinds[j].first != colinds[j].second)	// current != end
		{
			Adcsc.numx[colinds[j].first] = __binary_op(Adcsc.numx[colinds[j].first], numx[j]);
			++(colinds[j].first);
		} 
	}	
}



#endif
