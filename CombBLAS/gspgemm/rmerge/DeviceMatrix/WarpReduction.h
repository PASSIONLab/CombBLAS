#pragma once

#include "HostMatrix/ReduceFunctors.h"
#ifdef __CUDACC__
//Reduces an array of size BlockSize.
//requires shared mem of length BlockSize (power of 2 and max 1024)
template<int BlockSize, typename Dst, typename Src, typename ReduceFunctor>
static __device__ void BlockReduce(Dst* shared, Src* src, int thread, ReduceFunctor f){
	if(BlockSize>512 && thread<512){f(shared[thread],shared[thread],shared[thread+512]);__syncthreads();}
	if(BlockSize>256 && thread<256){f(shared[thread],shared[thread],shared[thread+256]);__syncthreads();}
	if(BlockSize>128 && thread<128){f(shared[thread],shared[thread],shared[thread+128]);__syncthreads();}
	if(BlockSize> 64 && thread< 64){f(shared[thread],shared[thread],shared[thread+ 64]);__syncthreads();}
	if(BlockSize> 32 && thread< 32){f(shared[thread],shared[thread],shared[thread+ 32]);__syncthreads();}
	if(BlockSize> 16 && thread< 16){f(shared[thread],shared[thread],shared[thread+ 16]);__syncthreads();}
	if(BlockSize>  8 && thread<  8){f(shared[thread],shared[thread],shared[thread+  8]);__syncthreads();}
	if(BlockSize>  4 && thread<  4){f(shared[thread],shared[thread],shared[thread+  4]);__syncthreads();}
	if(BlockSize>  2 && thread<  2){f(shared[thread],shared[thread],shared[thread+  2]);__syncthreads();}
	if(BlockSize>  1 && thread<  1){f(shared[thread],shared[thread],shared[thread+  1]);__syncthreads();}
}

//requires shared mem of length BlockSize (power of 2 and max 1024)
template<int BlockSize, typename Dst, typename Src, typename ReduceFunctor, typename Transform>
static __device__ void BlockReduce(Dst* shared, Src* src, int n, int thread, ReduceFunctor f, Transform transform){
	//first load through transform and reduce to BlockSize
	Dst threadSum(0);
	Dst tmp;
	for(int i=thread;i<n;i+=BlockSize){
		transform(tmp,src[i]);
		threadSum=f(threadSum,tmp);
	}
	shared[thread]=threadSum;
	__syncthreads();
	//now we have a vector of BlockSize elements to be reduced.
	BlockReduce<BlockSize>(shared,shared,thread,f);
}

//All these function work with warps or subsets of warps, 
//i.e. no synchronization is required.
//The Warpsize must be a power of 2

//Computes the sum of a vector of length WarpSize
template<int WarpSize, typename T, typename Functor>
static __device__ T WarpReduceOld(volatile T* shared, int thread, Functor f){
	if(thread<WarpSize/2){
		if(WarpSize>16)f(shared[thread],shared[thread+16]);
		if(WarpSize>8 )f(shared[thread],shared[thread+8]);
		if(WarpSize>4 )f(shared[thread],shared[thread+4]);
		if(WarpSize>2 )f(shared[thread],shared[thread+2]);
		if(WarpSize>1 )f(shared[thread],shared[thread+1]);
	}
	return shared[0];
}

template<int WarpSize, typename T, typename Functor>
static __device__ T WarpReduce(T value, volatile T* shared, int thread, Functor f){
	/*
	shared[thread]=value;
	if(WarpSize>16)f(shared[thread],shared[thread^16]);
	if(WarpSize>8 )f(shared[thread],shared[thread^8]);
	if(WarpSize>4 )f(shared[thread],shared[thread^4]);
	if(WarpSize>2 )f(shared[thread],shared[thread^2]);
	if(WarpSize>1 )f(shared[thread],shared[thread^1]);	
	return shared[0];
	*/
	
	/*
	if(WarpSize>16)f(value,__shfl_xor(value, 16));
	if(WarpSize>8 )f(value,__shfl_xor(value, 8));
	if(WarpSize>4 )f(value,__shfl_xor(value, 4));
	if(WarpSize>2 )f(value,__shfl_xor(value, 2));
	if(WarpSize>1 )f(value,__shfl_xor(value, 1));	
	return value;
	*/
	
	if(WarpSize>16)f(value,value,__shfl_xor(value, 16));	
	if(WarpSize>8 )f(value,value,__shfl_xor(value, 8));
	if(WarpSize>4 )f(value,value,__shfl_xor(value, 4));	
	if(WarpSize>2 )f(value,value,__shfl_xor(value, 2));
	if(WarpSize>1 )f(value,value,__shfl_xor(value, 1));
	return value;
}

//requires shared mem of size WarpSize
template<int WarpSize, typename T, typename ReduceFunctor>
static __device__ T WarpReduce(T* x,int length,T* tmpShared, int thread, ReduceFunctor f=ReduceFunctor()){
	T threadSum;
	if(thread<length)
		threadSum=x[thread];
	for(int i=WarpSize+thread;i<length;i+=WarpSize)
		threadSum=f(threadSum,x[i]);
	tmpShared[thread]=threadSum;
	return WarpReduce<WarpSize,T,ReduceFunctor>(tmpShared,thread,f);
}

template<int ThreadCount, typename T>
static __device__ T WarpSum(T* x,int length,T* tmp,int thread){
	return WarpReduce<ThreadCount,T,ReduceFunctors::AddFunctor>(x,length,tmp,thread);
}

/*
//Implementation for double
//Requires Compute capability 3.0
static __device__ __inline__ double __shfl_xor(double x, int lane)
{
	#if defined(__CUDACC__) && __CUDA_ARCH__ >= 300
	// Split the double number into 2 32b registers.
	//int lo, hi;
	//asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(x));
	int lo = __double2loint(x);
	int hi = __double2hiint(x);
	// Shuffle the two 32b registers.
	lo = __shfl_xor(lo, lane);
	hi = __shfl_xor(hi, lane);
	// Recreate the 64b number.	
	//asm volatile("mov.b64 %0, {%1,%2};" : "=d"(x) : "r"(lo), "r"(hi));
	x = __hiloint2double(hi,lo);
	#endif
	return x;
}

*/

//works only for Kepler GPUs or higher
//Warpsize must be a power of 2 and <= 32
template<int WarpSize, typename T>
static __device__ __host__ T WarpMin(T value){
	if(WarpSize==1)
		return value;
	else{
		// Use XOR mode to perform butterfly reduction	
		for (int i=WarpSize/2; i>=1; i>>=1){
			T tmp=__shfl_xor(value, i);
			//value=Min(value,tmp);
			value=min(value,tmp);
		}
		return value;
	}
}

//works only for Kepler GPUs or higher
//Warpsize must be a power of 2 and <= 32
template<int WarpSize, typename T>
static __device__ __host__ T WarpMax(T value){
	if(WarpSize==1)
		return value;
	else{
		// Use XOR mode to perform butterfly reduction	
		for (int i=WarpSize/2; i>=1; i>>=1){
			T tmp=__shfl_xor(value, i);
			//value=Min(value,tmp);
			value=max(value,tmp);
		}
		return value;
	}
}

//works only for Kepler GPUs 
//Warpsize must be a power of 2 and <= 32
template<int WarpSize, typename T>
static __device__ __host__ T WarpSum(T value){
	if(WarpSize==1)
		return value;
	else{
		for (int i=WarpSize/2; i>=1; i>>=1)
			value+=__shfl_xor(value, i);
		return value;	
	}
}

template<int WarpSize, typename T>
static __device__ T WarpSum(T* tmp, int n, int thread){
	T sum(0);
	for(int i=thread;i<n;i+=WarpSize)
		sum+=tmp[i];
	return WarpSum<WarpSize>(sum);

}

template<int WarpSize, typename T>
static __device__ T WarpSum(T val, T* tmp,int thread){
	//#if defined(__CUDACC__) && __CUDA_ARCH__ >= 300
	//return WarpSum<WarpSize>(val);
	//#else	
	return WarpReduce<WarpSize>(val,tmp,thread,ReduceFunctors::AddFunctor());
	//#endif
}

template<int WarpSize, typename T>
static __device__ T WarpMin(T val, T* tmp,int thread){
	//#if defined(__CUDACC__) && __CUDA_ARCH__ >= 300
	//return WarpMin<WarpSize>(val);
	//#else
	return WarpReduce<WarpSize>(val,tmp,thread,ReduceFunctors::MinFunctor());
	//#endif
}


#endif