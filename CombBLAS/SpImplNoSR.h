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

#ifndef _SP_IMPL_NOSR_H_
#define _SP_IMPL_NOSR_H_

template <class IT, class NT>
class Dcsc;

template <class IT, class NT1, class NT2>
struct SpImplNoSR;

//! Version without the Semiring (for BFS)
template <class IT, class NT1, class NT2>
void SpMXSpV(const Dcsc<IT,NT1> & Adcsc, int32_t mA, const int32_t * indx, const NT2 * numx, int32_t veclen,  
			 int32_t * indy, typename promote_trait<NT1,NT2>::T_promote * numy, int * cnts, int * dspls, int p_c)
{
	SpImplNoSR<IT,NT1,NT2>::SpMXSpV(Adcsc, mA, indx, numx, veclen, indy, numy, cnts, dspls,p_c);	// don't touch this
};


template <class IT, class NT1, class NT2>
struct SpImplNoSR
{
	static void SpMXSpV(const Dcsc<IT,NT1> & Adcsc, int32_t mA, const int32_t * indx, const NT2 * numx, int32_t veclen,  
						int32_t * indy, typename promote_trait<NT1,NT2>::T_promote * numy, int * cnts, int * dspls, int p_c);
};

template <class IT, class NT>
struct SpImplNoSR<IT,bool, NT>	// specialization
{
	static void SpMXSpV(const Dcsc<IT,bool> & Adcsc, int32_t mA, const int32_t * indx, const NT * numx, int32_t veclen,  
						int32_t * indy, NT * numy, int * cnts, int * dspls, int p_c);
};


/**
 * @param[in,out]   indy,numy,cnts 	{preallocated arrays to be filled}
 * @param[in] 		dspls	{displacements to preallocated indy,numy buffers}
 * This version determines the receiving column neighbor and adjust the indices to the receiver's local index
 * It also by passes-SPA by relying on the fact that x (rhs vector) is sorted and values are indices
 * (If they are not sorted, it'd still work but be non-deterministic)
 * Hence, Semiring operations are not needed (no add or multiply)
 * Also allows the vector's indices to be different than matrix's (for transition only) \TODO: Disable
 **/
template <typename IT, typename NT>
void SpImplNoSR<IT,bool,NT>::SpMXSpV(const Dcsc<IT,bool> & Adcsc, int32_t mA, const int32_t * indx, const NT * numx, int32_t veclen,  
										   int32_t * indy, NT * numy, int * cnts, int * dspls, int p_c)
{   
	bool * isthere = new bool[mA];
	fill(isthere, isthere+mA, false);
	vector< vector< pair<int32_t,NT> > > nzinds_vals(p_c);	// nonzero indices + associated parent assignments
	
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
					nzinds_vals[owner].push_back( make_pair(rowid, numx[k]) );
					isthere[rowid] = true;
				}
				// skip existing entries
			}
			++i;
			++k;
		}
	}
	
	for(int p = 0; p< p_c; ++p)
	{
		sort(nzinds_vals[p].begin(), nzinds_vals[p].end());
		cnts[p] = nzinds_vals[p].size();
		int32_t offset = perproc * p;
		for(int i=0; i< cnts[p]; ++i)
		{
			indy[dspls[p]+i] = nzinds_vals[p][i].first - offset;	// conver to local offset
			numy[dspls[p]+i] = nzinds_vals[p][i].second; 	
		}
	}
	delete [] isthere;
}

#endif
