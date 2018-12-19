#pragma once

//To provide a backward compatible __ldg() implementation. 
//Check this out: https://github.com/BryanCatanzaro/generics

//This bypasses the L1 cache to use the L2 cache. This should be used for read only data only.
//The L2 cache fetches 32byte blocks instead of 128byte blocks.
//On new architectures (Pascal) it reverts to return *p
template<typename T>
static __device__ __inline__ T ldg(T* p){

	#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 320) // || __CUDA_ARCH__ >= 600
	//not defined on older architectures
	//not useful on newer architectures (Pascal)
	return *p;
	#else
	return __ldg(p);//bypass L1 cache	
	#endif
}


