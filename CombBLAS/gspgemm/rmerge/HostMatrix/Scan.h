#pragma once

#include "HostMatrix/HostVector.h"

#include <thrust/detail/static_assert.h>
#include <thrust/scan.h>

//Parallel prefix sum.
//dst must be longer than x by 1.
//dst[0] is set to zero.
//dst[i] is set to the sum of x[0:i-1]
template<typename T>
static void Scan(HostVector<T> dst, HostVector<T> x){
	Verify(dst.IsSimple() && x.IsSimple(),FileAndLine);
	Verify(dst.Length()==x.Length()+1,FileAndLine);
	dst[0]=T(0);
	T sum(0);
	for(int64 i=0;i<x.Length();i++)
	{
		sum+=x[i];
		dst[i+1]=sum;
	}
}

template<typename T>
static HostVector<T> Scan(HostVector<T> x){
	HostVector<T> dst(x.Length()+1);
	Scan(dst,x);
	return dst;
}

template<typename T>
void ScanExclusive(HostVector<T> x){
	Verify(x.IsSimple(),FileAndLine);
	thrust::exclusive_scan(x.Data(),x.Data()+x.Length(),x.Data(),T(0));
}

template<typename T>
void ScanInclusive(HostVector<T> x){
	Verify(x.IsSimple(),FileAndLine);
	thrust::inclusive_scan(x.Data(),x.Data()+x.Length(),x.Data(),T(0));
}


