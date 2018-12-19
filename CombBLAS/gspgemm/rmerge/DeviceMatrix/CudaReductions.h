#pragma once

#include "HostMatrix/Intrinsics.h"
#include "HostMatrix/ElementFunctors.h"
#include "HostMatrix/ReduceFunctors.h"
#include "DeviceMatrix/WarpReduction.h"

#ifdef __CUDACC__
#define REDUCE_BLOCKSIZE 512
#define REDUCE_WARPSIZE 32

template<typename Dst, typename Src, typename ReduceFunctor, typename Transform>
__global__ void __cdecl CudaReduceTransformedKernel(Dst* dst, Src* src, int n, ReduceFunctor reduceFunctor, Transform transform, Dst neutral){
		__shared__ Dst shared[REDUCE_BLOCKSIZE];
		int thread=threadIdx.x;
		Dst threadSum(neutral);
		Dst tmp;
		for(int i=thread;i<n;i+=REDUCE_BLOCKSIZE){
			transform(tmp,src[i]);
			reduceFunctor(threadSum,threadSum,tmp);
		}
		shared[thread]=threadSum;
		__syncthreads();
		BlockReduce<REDUCE_BLOCKSIZE>(shared, shared, thread, reduceFunctor);
		if(thread==0)
			dst[0]=shared[0];
}

template<typename Dst, typename A, typename B, typename ReduceFunctor, typename Combine>
__global__ void __cdecl CudaReduceCombinedKernel(Dst* dst, A* a, B* b, int n, ReduceFunctor reduceFunctor, Combine combine, Dst neutral){
		__shared__ Dst shared[REDUCE_BLOCKSIZE];
		int thread=threadIdx.x;
		Dst threadSum(neutral);
		Dst tmp;
		for(int i=thread;i<n;i+=REDUCE_BLOCKSIZE){
			combine(tmp,a[i], b[i]);
			reduceFunctor(threadSum,threadSum,tmp);
		}
		shared[thread]=threadSum;
		__syncthreads();
		BlockReduce<REDUCE_BLOCKSIZE>(shared, shared, thread, reduceFunctor);
		if(thread==0)
			dst[0]=shared[0];
}

template<typename Dst, typename A, typename B, typename ReduceFunctor, typename Combine>
__global__ void __cdecl CudaReduceCombinedKernel(Dst* dst, A* a, int aStride, B* b, int bStride, int n, ReduceFunctor reduceFunctor, Combine combine, Dst neutral){
		__shared__ Dst shared[REDUCE_BLOCKSIZE];
		int thread=threadIdx.x;
		Dst threadSum(neutral);
		Dst tmp;
		for(int i=thread;i<n;i+=REDUCE_BLOCKSIZE){
			combine(tmp,a[i*aStride], b[i*bStride]);
			reduceFunctor(threadSum,threadSum,tmp);
		}
		shared[thread]=threadSum;
		__syncthreads();
		BlockReduce<REDUCE_BLOCKSIZE>(shared, shared, thread, reduceFunctor);
		if(thread==0)
			dst[0]=shared[0];
}


//****************************************

template<typename Dst, typename Src, typename ReduceFunctor, typename Transform>
void __cdecl CudaReduceTransformed(Dst* dst, Src* src, int n, ReduceFunctor reduceFunctor, Transform transform, Dst neutral){
	dim3 gridDim(1,1,1);
	dim3 blockDim(REDUCE_BLOCKSIZE,1,1);	
	CudaReduceTransformedKernel<<< gridDim, blockDim, 0>>>(dst,src,n,reduceFunctor,transform,neutral);
}

template<typename Dst, typename A, typename B, typename ReduceFunctor, typename Combine>
void __cdecl CudaReduceCombined(Dst* dst, A* a, B* b, int n, ReduceFunctor reduceFunctor, Combine combine, Dst neutral){
	dim3 gridDim(1,1,1);
	dim3 blockDim(REDUCE_BLOCKSIZE,1,1);
	CudaReduceCombinedKernel<<< gridDim, blockDim, 0>>>(dst,a,b,n,reduceFunctor,combine,neutral);
}

template<typename Dst, typename A, typename B, typename ReduceFunctor, typename Combine>
void __cdecl CudaReduceCombined(Dst* dst, A* a, int aStride, B* b, int bStride, int n, ReduceFunctor reduceFunctor, Combine combine, Dst neutral){
	dim3 gridDim(1,1,1);
	dim3 blockDim(REDUCE_BLOCKSIZE,1,1);
	CudaReduceCombinedKernel<<< gridDim, blockDim, 0>>>(dst,a,aStride,b,bStride,n,reduceFunctor,combine,neutral);
}

#else

template<typename Dst, typename Src, typename ReduceFunctor, typename Transform>
void __cdecl CudaReduceTransformed(Dst* dst, Src* src, int n, ReduceFunctor reduceFunctor, Transform transform, Dst neutral);

template<typename Dst, typename A, typename B, typename ReduceFunctor, typename Combine>
void __cdecl CudaReduceCombined(Dst* dst, A* a, B* b, int n, ReduceFunctor reduceFunctor, Combine combine, Dst neutral);

template<typename Dst, typename A, typename B, typename ReduceFunctor, typename Combine>
void __cdecl CudaReduceCombined(Dst* dst, A* a, int aStride, B* b, int bStride, int n, ReduceFunctor reduceFunctor, Combine combine, Dst neutral);


#endif
