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
void SpMXSpV(const Dcsc<IT,NT1> & Adcsc, IT nA, const IT * indx, const NT2 * numx, IT veclen,  vector<IT> & indy, vector< typename promote_trait<NT1,NT2>::T_promote > & numy)
{
	SpImpl<SR,IT,NT1,NT2>::SpMXSpV(Adcsc, nA, indx, numx, veclen, indy, numy);	// don't touch this
};

template <class SR, class IT, class NT1, class NT2>
struct SpImpl
{
	static void SpMXSpV(const Dcsc<IT,NT1> & Adcsc, IT nA, const IT * indx, const NT2 * numx, IT veclen,  
			vector<IT> & indy, vector< typename promote_trait<NT1,NT2>::T_promote > & numy);	// specialize this
};


template <class SR, class IT, class NT>
struct SpImpl<SR,IT,bool, NT>
{
	static void SpMXSpV(const Dcsc<IT,bool> & Adcsc, IT nA, const IT * indx, const NT * numx, IT veclen,  
			vector<IT> & indy, vector< NT > & numy);	// specialize this
};

static int64_t * spmvaux = NULL;

// base template version
// indx vector practically keeps column numbers requested from A
template <typename SR, typename IT, typename NT1, typename NT2>
void SpImpl<SR,IT,NT1,NT2>::SpMXSpV(const Dcsc<IT,NT1> & Adcsc, IT nA, const IT * indx, const NT2 * numx, IT veclen,  
			vector<IT> & indy, vector< typename promote_trait<NT1,NT2>::T_promote > & numy)
{
	typedef typename promote_trait<NT1,NT2>::T_promote T_promote;     
	HeapEntry<IT, NT1> * wset = new HeapEntry<IT, NT1>[veclen]; 

	// colnums vector keeps column numbers requested from A
	vector<IT> colnums(veclen);

	// colinds.first vector keeps indices to A.cp, i.e. it dereferences "colnums" vector (above),
	// colinds.second vector keeps the end indices (i.e. it gives the index to the last valid element of A.cpnack)
	vector< pair<IT,IT> > colinds(veclen);		

	float cf  = static_cast<float>(nA+1) / static_cast<float>(Adcsc.nzc);
        IT csize = static_cast<IT>(ceil(cf));   // chunk size
	if(spmvaux == NULL)
	{
		IT auxsize = Adcsc.ConstructAux(nA, spmvaux);
	}

	Adcsc.FillColInds(indx, veclen, colinds, spmvaux, csize);	// csize is irrelevant if aux is NULL	
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


template <typename SR, typename IT, typename NT>
void SpImpl<SR,IT,bool,NT>::SpMXSpV(const Dcsc<IT,bool> & Adcsc, IT nA, const IT * indx, const NT * numx, IT veclen,  
			vector<IT> & indy, vector<NT> & numy)
{   
	IT inf = numeric_limits<IT>::min();
	IT sup = numeric_limits<IT>::max(); 
        KNHeap< IT, IT > sHeap(sup, inf); 
	// pair<IT, IT> * wset = new pair<IT, IT>[veclen];

	// colnums vector keeps column numbers requested from A
	vector<IT> colnums(veclen);

	// colinds.first vector keeps indices to A.cp, i.e. it dereferences "colnums" vector (above),
	// colinds.second vector keeps the end indices (i.e. it gives the index to the last valid element of A.cpnack)
	vector< pair<IT,IT> > colinds(veclen);		

	float cf  = static_cast<float>(nA+1) / static_cast<float>(Adcsc.nzc);
        IT csize = static_cast<IT>(ceil(cf));   // chunk size
	if(spmvaux == NULL)
	{
		IT auxsize = Adcsc.ConstructAux(nA, spmvaux);
	}
	Adcsc.FillColInds(indx, veclen, colinds, spmvaux, csize);	// csize is irrelevant if aux is NULL	
	IT hsize = 0;		
	for(IT j =0; j< veclen; ++j)		// create the initial heap 
	{
		if(colinds[j].first != colinds[j].second)	// current != end
		{
			// heap (key, run)
			sHeap.insert(Adcsc.ir[colinds[j].first], j);
			++hsize;
			// wset[hsize++] = make_pair(Adcsc.ir[colinds[j].first], j);
		} 
	}	
	//make_heap(wset, wset+hsize);

	IT key, locv;
	while(hsize > 0)
	{
		sHeap.deleteMin(&key, &locv);
		// pop_heap(wset, wset + hsize);         	// result is stored in wset[hsize-1]
		// IT locv = wset[hsize-1].second;		// relative location of the nonzero in sparse column vector 
		NT mrhs = numx[locv];			// no need to call multiply when we know the matrix is boolean
		// ABAB: For BFS, we don't need numx as well since numx[locv] == wset[hsize-1].first

		if((!indy.empty()) && indy.back() == key)	
		{
			numy.back() = SR::add(numy.back(), mrhs);
		}
		else
		{
			indy.push_back(key);
			numy.push_back(mrhs);	
		}
		if( (++(colinds[locv].first)) != colinds[locv].second)	// current != end
		{
			// run stays the same !
			// wset[hsize-1].first = Adcsc.ir[colinds[locv].first];
			// push_heap(wset, wset+hsize);
			sHeap.insert(Adcsc.ir[colinds[locv].first], locv);
		}
		else
		{
			// don't push, one of the lists has been depleted
			--hsize;
		}
	}
}
#endif
