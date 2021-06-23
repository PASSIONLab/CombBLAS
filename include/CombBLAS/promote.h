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


#ifndef _PROMOTE_H_
#define _PROMOTE_H_

#include "myenableif.h"

namespace combblas {

template <class T1, class T2, class Enable = void>
struct promote_trait  { };

// typename disable_if< is_boolean<NT>::value, NT >::type won't work, 
// because then it will send Enable=NT which is different from the default template parameter
template <class NT> struct promote_trait< NT , bool, typename combblas::disable_if< combblas::is_boolean<NT>::value >::type >      
{                                           
	typedef NT T_promote;                    
};
template <class NT> struct promote_trait< bool , NT, typename combblas::disable_if< combblas::is_boolean<NT>::value >::type >      
{                                           
	typedef NT T_promote;                    
};

template<class NT> struct promote_trait<NT,NT>	// always allow self promotion
{
	typedef NT T_promote;
};

#define DECLARE_PROMOTE(A,B,C)                  \
    template <> struct promote_trait<A,B>       \
    {                                           \
        typedef C T_promote;                    \
    };
DECLARE_PROMOTE(int64_t, bool, int64_t);
DECLARE_PROMOTE(int64_t, int, int64_t);
DECLARE_PROMOTE(bool, int64_t, int64_t);
DECLARE_PROMOTE(int, int64_t, int64_t);
DECLARE_PROMOTE(int64_t, int64_t, int64_t);
DECLARE_PROMOTE(int, bool,int);
DECLARE_PROMOTE(short, bool,short);
DECLARE_PROMOTE(unsigned, bool, unsigned);
DECLARE_PROMOTE(float, bool, float);
DECLARE_PROMOTE(double, bool, double);
DECLARE_PROMOTE(unsigned long long, bool, unsigned long long);
DECLARE_PROMOTE(bool, int, int);
DECLARE_PROMOTE(bool, short, short);
DECLARE_PROMOTE(bool, unsigned, unsigned);
DECLARE_PROMOTE(bool, float, float);
DECLARE_PROMOTE(bool, double, double);
DECLARE_PROMOTE(bool, unsigned long long, unsigned long long);
DECLARE_PROMOTE(bool, bool, bool);
DECLARE_PROMOTE(float, int, float);
DECLARE_PROMOTE(double, int, double);
DECLARE_PROMOTE(int, float, float);
DECLARE_PROMOTE(int, double, double);
DECLARE_PROMOTE(double, int64_t, double);
DECLARE_PROMOTE(int64_t, double, double);
DECLARE_PROMOTE(double, uint64_t, double);
DECLARE_PROMOTE(uint64_t, double, double);
DECLARE_PROMOTE(float, float, float);
DECLARE_PROMOTE(double, double, double);
DECLARE_PROMOTE(int, int, int);
DECLARE_PROMOTE(unsigned, unsigned, unsigned);
DECLARE_PROMOTE(unsigned long long, unsigned long long, unsigned long long);

}

#endif
