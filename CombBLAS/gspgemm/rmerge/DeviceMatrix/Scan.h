#pragma once

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/detail/static_assert.h>
#include "DeviceMatrix/DeviceVector.h"

#ifdef __CUDACC__

//Parallel prefix sum.
//dst must be longer than x by 1.
//dst[0] is set to zero.
//dst[i] is set to the sum of x[0:i-1]
template<typename T>
void __cdecl CudaScan(DeviceVector<T> dst, DeviceVector<T> x){
	Verify(dst.IsSimple() && x.IsSimple(),"sd435344");
	Verify(dst.Length()==x.Length()+1,"g42t4234");
	dst.Set(0,T(0));	
	thrust::device_ptr<T> xBegin(x.Data());
	thrust::device_ptr<T> xEnd(x.Data()+x.Length());
	thrust::device_ptr<T> dstBegin(dst.Data()+1);
	thrust::inclusive_scan(xBegin,xEnd,dstBegin);
}

template<typename T>
void __cdecl CudaScanExclusive(DeviceVector<T> x){
	thrust::device_ptr<T> xBegin(x.Data());
	thrust::device_ptr<T> xEnd(x.Data()+x.Length());
	thrust::exclusive_scan(xBegin,xEnd,xBegin);
}

template<typename T>
void __cdecl CudaScanInclusive(DeviceVector<T> x){
	thrust::device_ptr<T> xBegin(x.Data());
	thrust::device_ptr<T> xEnd(x.Data()+x.Length());
	thrust::inclusive_scan(xBegin,xEnd,xBegin);
}

template<typename T>
void __cdecl ScanMaxInclusive(DeviceVector<T> x){
	thrust::device_ptr<T> xBegin(x.Data());
	thrust::device_ptr<T> xEnd(x.Data()+x.Length());
	thrust::inclusive_scan(xBegin,xEnd,xBegin,thrust::maximum<T>());
}

#else

template<typename T> void __cdecl ScanMaxInclusive(DeviceVector<T> x);

template<typename T> void __cdecl CudaScan(DeviceVector<T> dst, DeviceVector<T> x);
template<typename T> void __cdecl CudaScanExclusive(DeviceVector<T> x);
template<typename T> void __cdecl CudaScanInclusive(DeviceVector<T> x);

#endif

template<typename T>
static void Scan(DeviceVector<T> dst, DeviceVector<T> x){CudaScan(dst,x);}


template<typename T>
static void ScanInclusive(DeviceVector<T> x){
	Verify(x.IsSimple(),"45v9nn4987598475n");
	CudaScanInclusive(x);
}	

template<typename T>
static void ScanExclusive(DeviceVector<T> x){
	Verify(x.IsSimple(),"45v9nn4987598475n");
	CudaScanExclusive(x);
}	