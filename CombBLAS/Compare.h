#ifndef _COMPARE_H_
#define _COMPARE_H_

#ifdef NOTR1
	#include <boost/tr1/tuple.hpp>
#else
	#include <tr1/tuple.hpp>
#endif
using namespace std;
using namespace std::tr1;

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
