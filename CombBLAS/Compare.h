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

#ifndef _COMPARE_H_
#define _COMPARE_H_

#include <cmath>
#include "SpDefs.h"

#ifdef NOTR1
	#include <boost/tr1/tuple.hpp>
#else
	#include <tr1/tuple>
#endif
using namespace std;
using namespace std::tr1;

template <class T>
struct ErrorTolerantEqual:
	public binary_function< T, T, bool >
	{
		inline bool operator() (const T & a, const T & b) const
		{
			return ( std::abs(a - b) < EPSILON ) ; 
		}
	};
	

template<class IT, class NT>
struct TupleEqual:
	public binary_function< tuple<IT, IT, NT>, tuple<IT, IT, NT>, bool >
	{
		inline bool operator()(const tuple<IT, IT, NT> & lhs, const tuple<IT, IT, NT> & rhs) const
		{
			return ( (get<0>(lhs) == get<0>(rhs)) && (get<1>(lhs) == get<1>(rhs)) );
		} 
	};


/**
 ** Functor class
 ** \return bool, whether lhs precedes rhs in column-sorted order
 ** @pre {No elements with same (i,j) pairs exist in the input}
 **/
template <class IT, class NT>
struct ColLexiCompare:  // struct instead of class so that operator() is public
        public binary_function< tuple<IT, IT, NT>, tuple<IT, IT, NT>, bool >  // (par1, par2, return_type)
        {
                inline bool operator()(const tuple<IT, IT, NT> & lhs, const tuple<IT, IT, NT> & rhs) const
                {
                        if(get<1>(lhs) == get<1>(rhs))
                        {
                                return get<0>(lhs) < get<0>(rhs);
                        }
                        else
                        {
                                return get<1>(lhs) < get<1>(rhs);
                        }
                }
        };

// Non-lexicographical, just compares columns
template <class IT, class NT>
struct ColCompare:  // struct instead of class so that operator() is public
        public binary_function< tuple<IT, IT, NT>, tuple<IT, IT, NT>, bool >  // (par1, par2, return_type)
        {
                inline bool operator()(const tuple<IT, IT, NT> & lhs, const tuple<IT, IT, NT> & rhs) const
                {
			return get<1>(lhs) < get<1>(rhs);
                }
        };

// Non-lexicographical, just compares columns
template <class IT, class NT>
struct RowCompare:  // struct instead of class so that operator() is public
        public binary_function< tuple<IT, IT, NT>, tuple<IT, IT, NT>, bool >  // (par1, par2, return_type)
        {
                inline bool operator()(const tuple<IT, IT, NT> & lhs, const tuple<IT, IT, NT> & rhs) const
                {
			return get<0>(lhs) < get<0>(rhs);
                }
        };

template <class IT, class NT>
struct ColLexiCompareWithID:  // struct instead of class so that operator() is public
        public binary_function< pair< tuple<IT, IT, NT> , int > , pair< tuple<IT, IT, NT> , int>, bool >  // (par1, par2, return_type)
        {
                inline bool operator()(const pair< tuple<IT, IT, NT> , int > & lhs, const pair< tuple<IT, IT, NT> , int > & rhs) const
                {
                        if(get<1>(lhs.first) == get<1>(rhs.first))
                        {
                                return get<0>(lhs.first) < get<0>(rhs.first);
                        }
                        else
                        {
                                return get<1>(lhs.first) < get<1>(rhs.first);
                        }
                }
        };



#endif
