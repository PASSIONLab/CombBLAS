#ifndef _COMPARE_H_
#define _COMPARE_H

/**
 ** Functor class
 ** \return bool, whether lhs precedes rhs in column-sorted order
 ** @pre {No elements with same (i,j) pairs exist in the input}
 **/
template <class IT, class NT>
struct ColSortCompare:  // struct instead of class so that operator() is public
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

#endif
