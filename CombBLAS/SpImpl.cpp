#include "SpImpl.h"

//! base template version [full use of the semiring add() and multiply()]
//! indx vector practically keeps column numbers requested from A
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


/**
  * One of the two versions of SpMXSpV with on boolean matrix [uses only Semiring::add()]
  * This version is likely to be more memory efficient than the other one (the one that uses preallocated memory buffers)
  * Because here we don't use a dense accumulation vector but a heap. It will probably be slower though. 
**/
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
 * @param[in,out]   indy,numy,cnts 	{preallocated arrays to be filled}
 * @param[in] 		dspls	{displacements to preallocated indy,numy buffers}
 * This version determines the receiving column neighbor and adjust the indices to the receiver's local index
**/
template <typename SR, typename IT, typename NT>
void SpImpl<SR,IT,bool,NT>::SpMXSpV(const Dcsc<IT,bool> & Adcsc, int32_t mA, const int32_t * indx, const NT * numx, int32_t veclen,  
			int32_t * indy, NT * numy, int * cnts, int * dspls, int p_c)
{   
	NT * localy = new NT[mA];
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
					localy[rowid] = numx[k];	// initial assignment
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


template <typename SR, typename IT, typename NT>
void SpImpl<SR,IT,bool,NT>::SpMXSpV_ForThreadingNoMatch(const Dcsc<IT,bool> & Adcsc, int32_t mA, const int32_t * indx, const NT * numx, int32_t veclen,  
			vector<int32_t> & indy, vector<NT> & numy, int32_t offset)
{   
	NT * localy = new NT[mA];
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


