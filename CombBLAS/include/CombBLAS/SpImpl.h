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


#ifndef _SP_IMPL_H_
#define _SP_IMPL_H_

#include <iostream>
#include <vector>
#include "PreAllocatedSPA.h"
#include "Deleter.h"

namespace combblas {

template <class IT, class NT>
class Dcsc;

template <class IT, class NT>
class Csc;

template <class SR, class IT, class NUM, class IVT, class OVT>
struct SpImpl;

//! Overload #1: DCSC
template <class SR, class IT, class NUM, class IVT, class OVT>
void SpMXSpV(const Dcsc<IT,NUM> & Adcsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen,
			 std::vector<int32_t> & indy, std::vector< OVT > & numy, PreAllocatedSPA<OVT> & SPA)
{
	// ignoring SPA for now. However, a branching similar to the CSC case can be implemented
    SpImpl<SR,IT,NUM,IVT,OVT>::SpMXSpV(Adcsc, mA, indx, numx, veclen, indy, numy);	// don't touch this
};


//! Overload #2: DCSC
template <class SR, class IT, class NUM, class IVT, class OVT>
void SpMXSpV(const Dcsc<IT,NUM> & Adcsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen,
			 int32_t * indy, OVT * numy, int * cnts, int * dspls, int p_c)
{
	SpImpl<SR,IT,NUM,IVT,OVT>::SpMXSpV(Adcsc, mA, indx, numx, veclen, indy, numy, cnts, dspls,p_c);	// don't touch this
};


//! Overload #3: DCSC
template <class SR, class IT, class NUM, class IVT, class OVT>
void SpMXSpV_ForThreading(const Dcsc<IT,NUM> & Adcsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen,
                          std::vector<int32_t> & indy, std::vector< OVT > & numy, int32_t offset)
{
    SpImpl<SR,IT,NUM,IVT,OVT>::SpMXSpV_ForThreading(Adcsc, mA, indx, numx, veclen, indy, numy, offset);	// don't touch this
};

//! Overload #4: DCSC w/ preallocated SPA
template <class SR, class IT, class NUM, class IVT, class OVT>
void SpMXSpV_ForThreading(const Dcsc<IT,NUM> & Adcsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen,
                          std::vector<int32_t> & indy, std::vector< OVT > & numy, int32_t offset, std::vector<OVT> & localy, BitMap & isthere, std::vector<uint32_t> & nzinds)
{
    SpImpl<SR,IT,NUM,IVT,OVT>::SpMXSpV_ForThreading(Adcsc, mA, indx, numx, veclen, indy, numy, offset, localy, isthere, nzinds);
};









/*
 The following two functions are base CSC implementation. All overloaded function calls will be routed to these functions.
 */
// all CSC will fall to this
template <typename SR, typename IT, typename NUM, typename IVT, typename OVT>
void SpMXSpV_HeapSort(const Csc<IT,NUM> & Acsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen, std::vector<int32_t> & indy, std::vector<OVT> & numy, int32_t offset);

// all PreAllocatedSPA will fall to this
template <class SR, class IT, class NUM, class IVT, class OVT>
void SpMXSpV_Bucket(const Csc<IT,NUM> & Acsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen,std::vector<int32_t> & indy, std::vector< OVT > & numy, PreAllocatedSPA<OVT> & SPA);



//! Overload #1: CSC
template <class SR, class IT, class NUM, class IVT, class OVT>
void SpMXSpV(const Csc<IT,NUM> & Acsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen,
             int32_t * indy, OVT * numy, int * cnts, int * dspls, int p_c)
{
        std::cout << "Optbuf enabled version is not yet supported with CSC matrices" << std::endl;
};


//! Overload #2: CSC
template <class SR, class IT, class NUM, class IVT, class OVT>
void SpMXSpV(const Csc<IT,NUM> & Acsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen,
             std::vector<int32_t> & indy, std::vector< OVT > & numy, PreAllocatedSPA<OVT> & SPA)
{
    if(SPA.initialized)
        SpMXSpV_Bucket<SR>(Acsc, mA, indx, numx, veclen, indy, numy, SPA);
    else
        SpMXSpV_HeapSort<SR>(Acsc, mA, indx, numx, veclen, indy, numy, 0);

};

//! Overload #3: CSC
template <class SR, class IT, class NUM, class IVT, class OVT>
void SpMXSpV_ForThreading(const Csc<IT,NUM> & Acsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen,
                          std::vector<int32_t> & indy, std::vector< OVT > & numy, int32_t offset)
{
    SpMXSpV_HeapSort<SR>(Acsc, mA, indx, numx, veclen, indy, numy, offset);
};

//! Overload #4: CSC w/ preallocated SPA
template <class SR, class IT, class NUM, class IVT, class OVT>
void SpMXSpV_ForThreading(const Csc<IT,NUM> & Acsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen,
                          std::vector<int32_t> & indy, std::vector< OVT > & numy, int32_t offset, std::vector<OVT> & localy, BitMap & isthere, std::vector<uint32_t> & nzinds)
{

    SpMXSpV_HeapSort<SR>(Acsc, mA, indx, numx, veclen, indy, numy, offset);
    // We can eventually call SpMXSpV_HeapMerge or SpMXSpV_SPA (not implemented for CSC yet)
};





/**
 * IT: The sparse matrix index type. Sparse vector index type is fixed to be int32_t
 * It is the caller function's (inside ParFriends/Friends) job to convert any different types
 * and ensure correctness. Rationale is efficiency, and the fact that we know for sure
 * that 32-bit LOCAL indices are sufficient for all reasonable concurrencies and data sizes (as of 2011)
 * \todo: As of 2015, this might not be true!!! (ABAB)
 **/
template <class SR, class IT, class NUM, class IVT, class OVT>
struct SpImpl
{
    static void SpMXSpV(const Dcsc<IT,NUM> & Adcsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen,
                        std::vector<int32_t> & indy, std::vector< OVT > & numy);	// specialize this

    static void SpMXSpV(const Dcsc<IT,NUM> & Adcsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen,
                        int32_t * indy, OVT * numy, int * cnts, int * dspls, int p_c)
    {
        std::cout << "Optbuf enabled version is not yet supported with general (non-boolean) matrices" << std::endl;
    };


    static void SpMXSpV_ForThreading(const Dcsc<IT,NUM> & Adcsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen,
                                     std::vector<int32_t> & indy, std::vector<OVT> & numy, int32_t offset)
    {
        std::cout << "Threaded version is not yet supported with general (non-boolean) matrices" << std::endl;
    };
	static void SpMXSpV_ForThreading(const Dcsc<IT,NUM> & Acsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen,
									 std::vector<int32_t> & indy, std::vector<OVT> & numy, int32_t offset, std::vector<OVT> & localy, BitMap & isthere, std::vector<uint32_t> & nzinds)
	{
		std::cout << "Threaded version is not yet supported with general (non-boolean) matrices" << std::endl;
	};
};




template <class SR, class IT, class IVT, class OVT>
struct SpImpl<SR,IT,bool, IVT, OVT>	// specialization
{
    static void SpMXSpV(const Dcsc<IT,bool> & Adcsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen,
                        std::vector<int32_t> & indy, std::vector< OVT > & numy);

    static void SpMXSpV(const Dcsc<IT,bool> & Adcsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen,
                        int32_t * indy, OVT * numy, int * cnts, int * dspls, int p_c);

    //! Dcsc and vector index types do not need to match
    static void SpMXSpV_ForThreading(const Dcsc<IT,bool> & Adcsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen,
                                     std::vector<int32_t> & indy, std::vector<OVT> & numy, int32_t offset);
    //! Dcsc and vector index types do not need to match
    static void SpMXSpV_ForThreading(const Dcsc<IT,bool> & Adcsc, int32_t mA, const int32_t * indx, const IVT * numx, int32_t veclen,
                                     std::vector<int32_t> & indy, std::vector<OVT> & numy, int32_t offset, std::vector<OVT> & localy, BitMap & isthere, std::vector<uint32_t> & nzinds);
};

}

#include "SpImpl.cpp"

#endif