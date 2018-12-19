#pragma once

#include "DeviceMatrix/DeviceVector.h"

#ifdef __CUDACC__

#include <thrust/detail/static_assert.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

template<typename T, typename K> 
void StableSortByKey(DeviceVector<T> x, DeviceVector<K> keys){
	thrust::stable_sort_by_key(thrust::device_ptr<K>(keys.Data())
		,thrust::device_ptr<K>(keys.Data()+keys.Length())
		,thrust::device_ptr<T>(x.Data()));
}

#ifdef INSTANTIATE_0
void Ugabuga(){
	StableSortByKey(DeviceVector<uint>(),DeviceVector<uint>());
	StableSortByKey(DeviceVector<int>(),DeviceVector<int>());
	StableSortByKey(DeviceVector<int>(),DeviceVector<uint>());
}
#endif

#else
template<typename T, typename K> 
void StableSortByKey(DeviceVector<T> x, DeviceVector<K> keys);

#endif
