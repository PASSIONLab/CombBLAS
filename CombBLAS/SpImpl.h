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
void SpMXSpV(const Dcsc<IT,NT1> & Adcsc, int32_t mA, const int32_t * indx, const NT2 * numx, int32_t veclen,  
			 int32_t * indy, typename promote_trait<NT1,NT2>::T_promote * numy, int * cnts, int * dspls, int p_c)
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
void SpMXSpV_ForThreadingNoMatch(const Dcsc<IT,NT1> & Adcsc, int32_t mA, const int32_t * indx, const NT2 * numx, int32_t veclen,  
		vector<int32_t> & indy, vector< typename promote_trait<NT1,NT2>::T_promote > & numy, int32_t offset)
{
	SpImpl<SR,IT,NT1,NT2>::SpMXSpV_ForThreadingNoMatch(Adcsc, mA, indx, numx, veclen, indy, numy, offset);	// don't touch this
};


template <class SR, class IT, class NT1, class NT2>
struct SpImpl
{
	static void SpMXSpV(const Dcsc<IT,NT1> & Adcsc, IT mA, const IT * indx, const NT2 * numx, IT veclen,  
			vector<IT> & indy, vector< typename promote_trait<NT1,NT2>::T_promote > & numy);	// specialize this

	static void SpMXSpV(const Dcsc<IT,NT1> & Adcsc, int32_t mA, const int32_t * indx, const NT2 * numx, int32_t veclen,  
			IT * indy, typename promote_trait<NT1,NT2>::T_promote * numy, int * cnts, int * dspls, int p_c);
};



template <class SR, class IT, class NT>
struct SpImpl<SR,IT,bool, NT>	// specialization
{
	static void SpMXSpV(const Dcsc<IT,bool> & Adcsc, IT mA, const IT * indx, const NT * numx, IT veclen,  
			vector<IT> & indy, vector< NT > & numy);	

	static void SpMXSpV(const Dcsc<IT,bool> & Adcsc, int32_t mA, const int32_t * indx, const NT * numx, int32_t veclen,  
			int32_t * indy, NT * numy, int * cnts, int * dspls, int p_c);

	static void SpMXSpV_ForThreading(const Dcsc<IT,bool> & Adcsc, IT mA, const IT * indx, const NT * numx, IT veclen,  	// version where Dcsc and vector types match
			vector<IT> & indy, vector<NT> & numy, IT offset);

	static void SpMXSpV_ForThreadingNoMatch(const Dcsc<IT,bool> & Adcsc, int32_t mA, const int32_t * indx, const NT * numx, int32_t veclen,  // version for which they don't have to match
			vector<int32_t> & indy, vector<NT> & numy, int32_t offset);
};

#include "SpImpl.cpp"
#endif
