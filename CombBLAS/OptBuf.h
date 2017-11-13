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


#ifndef _OPT_BUF_H
#define _OPT_BUF_H
#include "BitMap.h"

namespace combblas {

/**
  * This special data structure is used for optimizing BFS iterations
  * by providing a fixed sized buffer for communication
  * the contents of the buffer are irrelevant until SpImpl:SpMXSpV starts
  * hence the copy constructor that doesn't copy contents
  */
template <class IT, class NT>
class OptBuf
{
public:
	OptBuf(): isthere(NULL), p_c(0), totmax(0), localm(0) {};
	void MarkEmpty()
	{
		if(totmax > 0)
		{
			isthere->reset();
		}
	}
	
	void Set(const std::vector<int> & maxsizes, int mA) 
	{
		p_c =  maxsizes.size(); 
		totmax = std::accumulate(maxsizes.begin(), maxsizes.end(), 0);
		inds = new IT[totmax];
		std::fill_n(inds, totmax, -1);
		nums = new NT[totmax];
		dspls = new int[p_c]();
    std::partial_sum(maxsizes.begin(), maxsizes.end()-1, dspls+1);
		localm = mA;
		
		isthere = new BitMap(localm);
	};
	~OptBuf()
	{	if(localm > 0)
		{
			delete isthere;
		}
		
		if(totmax > 0)
		{
			delete [] inds;
			delete [] nums;
		}
		if(p_c > 0)
			delete [] dspls;
	}
	OptBuf(const OptBuf<IT,NT> & rhs)
	{
		p_c = rhs.p_c;
		totmax = rhs.totmax;
		localm = rhs.localm;
		inds = new IT[totmax];
		nums = new NT[totmax];
		dspls = new int[p_c]();
		isthere = new BitMap(localm);
	}
	OptBuf<IT,NT> & operator=(const OptBuf<IT,NT> & rhs)
	{
		if(this != &rhs)
		{
			if(localm > 0)
			{
				delete isthere;
			}
			if(totmax > 0)
			{
				delete [] inds;
				delete [] nums;
			}
			if(p_c > 0)
				delete [] dspls;
	
			p_c = rhs.p_c;
			totmax = rhs.totmax;
			localm = rhs.localm;
			inds = new IT[totmax];
			nums = new NT[totmax];
			dspls = new int[p_c]();	
			isthere = new BitMap(*(rhs.isthere));
		}
		return *this;
	}
	
	IT * inds;	
	NT * nums;	
	int * dspls;
	BitMap * isthere;
	int p_c;
	int totmax;
	int localm;
};

}

#endif

