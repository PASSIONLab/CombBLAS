//  Project AC-SpGEMM
//  https://www.tugraz.at/institute/icg/research/team-steinberger/
//
//  Copyright (C) 2018 Institute for Computer Graphics and Vision,
//                     Graz University of Technology
//
//  Author(s):  Martin Winter - martin.winter (at) icg.tugraz.at
//              Daniel Mlakar - daniel.mlakar (at) icg.tugraz.at
//              Rhaleb Zayer - rzayer (at) mpi-inf.mpg.de
//              Hans-Peter Seidel - hpseidel (at) mpi-inf.mpg.de
//              Markus Steinberger - steinberger ( at ) icg.tugraz.at
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//  THE SOFTWARE.
//

#pragma once


#include "meta_utils.h"
#include "common.h"
#include <cuda_runtime.h>
#include <type_traits>



/////////////// HELPERS /////////////////////////

const uint32_t WARP_SIZE = 32;




template<int END, int BEGIN = 0>
struct ConditionalIteration
{
	template<typename F>
	__device__
		static void iterate(F f)
	{
		bool res = f(BEGIN);
		if (res)
			ConditionalIteration<END, BEGIN + 1>::iterate(f);
	}
};

template<uint32_t END>
struct ConditionalIteration<END, END>
{
	template<typename F>
	__device__
		static void iterate(F f)
	{
	}
};


template<int Bytes>
struct VecLoadTypeImpl;

template<>
struct VecLoadTypeImpl<4>
{
	using type = unsigned int;
};
template<>
struct VecLoadTypeImpl<8>
{
	using type = uint2;
};
template<>
struct VecLoadTypeImpl<16>
{
	using type = uint4;
};

template<typename T, int N>
struct VecLoadType
{
	using type = typename VecLoadTypeImpl<sizeof(T)*N>::type;
	union
	{
		T data[N];
		type vec;
	};

	__device__ __forceinline__ VecLoadType() = default;
	__device__ __forceinline__ VecLoadType(type v) : vec(v) {};
};

template<int VecSize, class T, int N>
__device__ __forceinline__ void warp_load_vectorized(T (&out)[N], const T* in)
{
	static_assert(static_popcnt<N>::value == 1, "load_vectorized only works for pow 2 elements");
	
	using LoadType = VecLoadType<T, VecSize>;
	const typename LoadType::type* vec_in = reinterpret_cast<const typename LoadType::type*>(in + (threadIdx.x/WARP_SIZE)*WARP_SIZE*N) + laneid();

	//TODO: get rid of UB by doing an explicit unroll and just use the vec type
	#pragma unroll
	for (int i = 0; i < N / VecSize; ++i)
	{
		LoadType loaded;
		loaded.vec = vec_in[i*WARP_SIZE];
		#pragma unroll
		for (int j = 0; j < VecSize; ++j)
			out[i*VecSize + j] = loaded.data[j];
	}
}

template<int VecSize, class T, int N>
__device__ __forceinline__ void vectorized_to_blocked(T(&data)[N])
{
	const int Vecs = N / VecSize;

	//rotate
	#pragma unroll
	for (int k = 0; k < Vecs - 1; ++k)
	{
		if (laneid() % Vecs > k)
		{
			T tmp[VecSize];
			#pragma unroll
			for (int i = 0; i < VecSize; ++i)
				tmp[i] = data[(Vecs - 1)*VecSize + i];

			#pragma unroll
			for (int j = Vecs - 1; j > 0; --j)
				#pragma unroll
				for (int i = 0; i < VecSize; ++i)
					data[j*VecSize + i] = data[(j - 1)*VecSize + i];

			#pragma unroll
			for (int i = 0; i < VecSize; ++i)
				data[i] = tmp[i];
		}
	}

	//shfl
	int pad_offset = Vecs - (laneid() * Vecs) / WARP_SIZE;
	int section_offset = (laneid() * Vecs) % WARP_SIZE;

	#pragma unroll
	for (int j = 0; j < Vecs; ++j)
	{
		int shfl_offset = section_offset + ((pad_offset + j) % Vecs);
		#pragma unroll
		for (int i = 0; i < VecSize; ++i)
			data[j*VecSize + i] = __shfl(data[j*VecSize + i], shfl_offset);
	}
	
	//rotate back
	#pragma unroll
	for (int k = 0; k < Vecs - 1; ++k)
	{
		if ((laneid() * Vecs) / WARP_SIZE > k)
		{
			T tmp[VecSize];
			#pragma unroll
			for (int i = 0; i < VecSize; ++i)
				tmp[i] = data[i];

			#pragma unroll
			for (int j = 1; j < Vecs; ++j)
				#pragma unroll
				for (int i = 0; i < VecSize; ++i)
					data[(j - 1)*VecSize + i] = data[j*VecSize + i];

			#pragma unroll
			for (int i = 0; i < VecSize; ++i)
				data[(Vecs - 1)*VecSize + i] = tmp[i];
		}
	}
}


template<class COMP, int LO, int N, int R>
struct ThreadOddEvenMerge;

template<class COMP, int LO, int N, int R, int M, bool FULL>
struct ThreadOddEvenMergeImpl;

template<class T>
__device__ __forceinline__ void swap(T& a, T& b)
{
	T temp = a;
	a = b;
	b = temp;
}

template<class COMP, int LO, int N, int R, int M>
struct ThreadOddEvenMergeImpl<COMP, LO, N, R, M, true>
{
	template<class K, int L>
	__device__ __forceinline__ static void run(K(&key)[L])
	{
		ThreadOddEvenMerge<COMP, LO, N, M>::run(key);
		ThreadOddEvenMerge<COMP, LO + R, N, M>::run(key);
#pragma unroll
		for (int i = LO + R; i + R < LO + N; i += M)
			if (COMP::comp(key[i], key[i + R]))
				swap(key[i], key[i + R]); 
	}
	template<class K, class V, int L>
	__device__ __forceinline__ static void run(K(&key)[L], V(&value)[L])
	{
		ThreadOddEvenMerge<COMP, LO, N, M>::run(key, value);
		ThreadOddEvenMerge<COMP, LO + R, N, M>::run(key, value);
#pragma unroll
		for (int i = LO + R; i + R < LO + N; i += M)
			if (COMP::comp(key[i], key[i + R]))
				swap(key[i], key[i + R]),
				swap(value[i], value[i + R]);
	}
};
template<class COMP, int LO, int N, int R, int M>
struct ThreadOddEvenMergeImpl<COMP, LO, N, R, M, false>
{
	template<class K, int L>
	__device__ __forceinline__ static void run(K(&key)[L])
	{
		if (COMP::comp(key[LO], key[LO + R]))
			swap(key[LO], key[LO + R]);
	}
	template<class K, class V, int L>
	__device__ __forceinline__ static void run(K(&key)[L], V(&value)[L])
	{
		if (COMP::comp(key[LO], key[LO + R]))
			swap(key[LO], key[LO + R]),
			swap(value[LO], value[LO + R]);
	}
};


template<class COMP, int LO, int N, int R>
struct ThreadOddEvenMerge : public ThreadOddEvenMergeImpl<COMP, LO, N, R, 2 * R, (2 * R < N)>
{
};

template<class COMP, int LO, int N>
struct ThreadOddEvenMergeSort
{
	template<class K, int L>
	__device__ __forceinline__ static void run(K(&key)[L])
	{
		ThreadOddEvenMergeSort<COMP, LO, N / 2>::run(key);
		ThreadOddEvenMergeSort<COMP, LO + N / 2, N / 2>::run(key);
		ThreadOddEvenMerge<COMP, LO, N, 1>::run(key);
	}
	template<class K, class V, int L>
	__device__ __forceinline__ static void run(K(&key)[L], V(&value)[L])
	{
		ThreadOddEvenMergeSort<COMP, LO, N / 2>::run(key, value);
		ThreadOddEvenMergeSort<COMP, LO + N / 2, N / 2>::run(key, value);
		ThreadOddEvenMerge<COMP, LO, N, 1>::run(key, value);
	}
};

template<class COMP, int LO>
struct ThreadOddEvenMergeSort<COMP, LO, 1>
{
	template<class K, int L>
	__device__ __forceinline__ static void run(K (&key)[L])
	{ }
	template<class K, class V, int L>
	__device__ __forceinline__ static void run(K (&key)[L], V(&value)[L])
	{ }
};

template<class COMP, class K, int L>
__device__ __forceinline__ void threadOddEvenMergeSort(K(&key)[L])
{
	ThreadOddEvenMergeSort<COMP, 0, L>::run(key);
}
template<class COMP, class K, class V, int L>
__device__ __forceinline__ void threadOddEvenMergeSort(K(&key)[L], V(&value)[L])
{
	ThreadOddEvenMergeSort<COMP, 0, L>::run(key, value);
}

struct SortAscending
{
	template<class T>
	__device__ __forceinline__ static bool comp(T a, T b)
	{
		return a > b;
	}
};

struct SortDescending
{
	template<class T>
	__device__ __forceinline__ static bool comp(T a, T b)
	{
		return a < b;
	}
};

__device__ __forceinline__ inline uint32_t laneid()
{
	uint32_t mylaneid;
	asm("mov.u32 %0, %laneid;" : "=r" (mylaneid));
	return mylaneid;
}
