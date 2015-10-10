/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.5 -------------------------------------------------*/
/* date: 10/09/2015 ---------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc, Adam Lugowski ------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2015, The Regents of the University of California
 
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


#ifndef _FRIENDS_CSC_H_
#define _FRIENDS_CSC_H_

#include <iostream>
#include "SpMat.h"	// Best to include the base class first
#include "CombBLAS.h"
using namespace std;

template <class IU, class NU>
class SpTuples;

template <class IU, class NU>
class SpCCols;

template <class IU, class NU>
class Csc;

/**
 * Multithreaded SpMV with sparse vector
 * the assembly of outgoing buffers sendindbuf/sendnumbuf are done here
 */
template <typename SR, typename IU, typename NUM, typename IVT, typename OVT>
int csc_gespmv_sparse (const SpCCCols<IU, NUM> & A, const int32_t * indx, const IVT * numx, int32_t nnzx,
                          int32_t * & sendindbuf, OVT * & sendnumbuf, int * & sdispls, int p_c)
{
    // ABAB: implement
}

#endif