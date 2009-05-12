
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
	typedef typename promote_trait<T1,T2>::T_promote T_promote;

	static T_promote add(const T1 & arg1, const T2 & arg2)
	{
		return (static_cast<T_promote>(arg1) +  
			static_cast<T_promote>(arg2) );
	}
	static T_promote multiply(const T1 & arg1, const T2 & arg2)
	{
		return (static_cast<T_promote>(arg1) * 
			static_cast<T_promote>(arg2) );

	}
};


template <class T1, class T2>
struct MinPlusSRing
{
	typedef typename promote_trait<T1,T2>::T_promote T_promote;

	static T_promote add(const T1 & arg1, const T2 & arg2)
	{
		return std::min<T_promote> 
		(static_cast<T_promote>(arg1), static_cast<T_promote>(arg2));
	}
	static T_promote multiply(const T1 & arg1, const T2 & arg2)
	{
		return inf_plus< T_promote > 
		(static_cast<T_promote>(arg1), static_cast<T_promote>(arg2));
	}
};


#endif
