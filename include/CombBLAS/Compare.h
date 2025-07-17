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

#ifndef _COMPARE_H_
#define _COMPARE_H_

#include <cmath>
#include <type_traits>
#include "SpDefs.h"
#include "CombBLAS.h"

namespace combblas {

// third parameter of compare is about floating point-ness
template <class T>
inline bool compare(const T & a, const T & b, std::false_type) 	// not floating point
{
	return (a == b); 			
}

template <class T>
inline bool compare(const T & a, const T & b, std::true_type) 	//  floating point
{
	// According to the IEEE 754 standard, negative zero and positive zero should 
	// compare as equal with the usual (numerical) comparison operators, like the == operators of C++ 

	if(a == b) return true;		// covers the "division by zero" case as well: max(a,b) can't be zero if it fails
	else return ( std::abs(a - b) < EPSILON || (std::abs(a - b) / std::max(std::abs(a), std::abs(b))) < EPSILON ) ;     // Fine if either absolute or relative error is small
}


template <class T>
struct ErrorTolerantEqual
	{	
		bool operator() (const T & a, const T & b) const
		{
			return compare(a,b, std::is_floating_point<T>());
		}
	};

template < typename T >
struct absdiff
{
        T operator () ( T const &arg1, T const &arg2 ) const
        {
                using std::abs;
                return abs( arg1 - arg2 );
        }
};
	

template<class IT, class NT>
struct TupleEqual
	{
		inline bool operator()(const std::tuple<IT, IT, NT> & lhs, const std::tuple<IT, IT, NT> & rhs) const
		{
			return ( (std::get<0>(lhs) == std::get<0>(rhs)) && (std::get<1>(lhs) == std::get<1>(rhs)) );
		} 
	};


/**
 ** Functor class
 ** \return bool, whether lhs precedes rhs in column-sorted order
 ** @pre {No elements with same (i,j) pairs exist in the input}
 **/
template <class IT, class NT>
struct ColLexiCompare
        {
                inline bool operator()(const std::tuple<IT, IT, NT> & lhs, const std::tuple<IT, IT, NT> & rhs) const
                {
                        if(std::get<1>(lhs) == std::get<1>(rhs))
                        {
                                return std::get<0>(lhs) < std::get<0>(rhs);
                        }
                        else
                        {
                                return std::get<1>(lhs) < std::get<1>(rhs);
                        }
                }
        };

template <class IT, class NT>
struct RowLexiCompare
        {
                inline bool operator()(const std::tuple<IT, IT, NT> & lhs, const std::tuple<IT, IT, NT> & rhs) const
                {
                        if(std::get<0>(lhs) == std::get<0>(rhs))
                        {
                                return std::get<1>(lhs) < std::get<1>(rhs);
                        }
                        else
                        {
                                return std::get<0>(lhs) < std::get<0>(rhs);
			}
		}
	};


// Non-lexicographical, just compares columns
template <class IT, class NT>
struct ColCompare
        {
                inline bool operator()(const std::tuple<IT, IT, NT> & lhs, const std::tuple<IT, IT, NT> & rhs) const
                {
			return std::get<1>(lhs) < std::get<1>(rhs);
                }
        };

// Non-lexicographical, just compares columns
template <class IT, class NT>
struct RowCompare
        {
                inline bool operator()(const std::tuple<IT, IT, NT> & lhs, const std::tuple<IT, IT, NT> & rhs) const
                {
			return std::get<0>(lhs) < std::get<0>(rhs);
                }
        };

template <class IT, class NT>
struct ColLexiCompareWithID
        {
                inline bool operator()(const std::pair< std::tuple<IT, IT, NT> , int > & lhs, const std::pair< std::tuple<IT, IT, NT> , int > & rhs) const
                {
                        if(std::get<1>(lhs.first) == std::get<1>(rhs.first))
                        {
                                return std::get<0>(lhs.first) < std::get<0>(rhs.first);
                        }
                        else
                        {
                                return std::get<1>(lhs.first) < std::get<1>(rhs.first);
                        }
                }
        };



}        

#endif
