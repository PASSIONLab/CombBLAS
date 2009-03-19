
#ifndef _SEMIRINGS_H_
#define _SEMIRINGS_H_

#include <utility>
#include <climits>
#include <cmath>
#include "promote.h"

template <typename T>
struct inf_plus{
  T operator()(const T& a, const T& b) const {
	T inf = std::numeric_limits<T>::max();
    	if (a == inf || b == inf){
      		return inf;
    	}
    	return a + b;
  }
};


template <class T1, class T2>
struct PlusTimesSRing
{
	promote_trait<T1,T2>::T_promote add(const T1 & arg1, const T2 & arg2) const
	{
		return (static_cast<promote_trait<T1,T2>::T_promote>(arg1) +  
			static_cast<promote_trait<T1,T2>::T_promote>(arg2) );
	}
	promote_trait<T1,T2>::T_promote multiply(const T1 & arg1, const T2 & arg2) const
	{
		return (static_cast<promote_trait<T1,T2>::T_promote>(arg1) * 
			static_cast<promote_trait<T1,T2>::T_promote>(arg2) );

	}
};


template <class T1, class T2>
struct MinPlusSRing
{
	promote_trait<T1,T2>::T_promote add(const T1 & arg1, const T2 & arg2) const
	{
		return min<promote_trait<T1,T2>::T_promote> 
		(static_cast<promote_trait<T1,T2>::T_promote>(arg1), static_cast<promote_trait<T1,T2>::T_promote>(arg2));
	}
	promote_trait<T1,T2>::T_promote multiply(const T1 & arg1, const T2 & arg2) const
	{
		return inf_plus< promote_trait<T1,T2>::T_promote > 
		(static_cast<promote_trait<T1,T2>::T_promote>(arg1), static_cast<promote_trait<T1,T2>::T_promote>(arg2));
	}
};


#endif
