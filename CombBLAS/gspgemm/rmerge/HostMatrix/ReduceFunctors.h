#pragma once

#include "HostMatrix/Intrinsics.h"

namespace ReduceFunctors{	
	class AddFunctor{
		char dummy;
	public:
		template<typename Dst, typename A, typename B>
		__host__ __device__ void operator()(Dst& dst, A a, B b){
			dst=a+b;
		}
		template<typename Dst, typename A>
		__host__ __device__ void operator()(Dst& dst, A a){
			dst+=a;
		}
		template<typename T>
		__device__ T Zero(){return T(0);}
	};

	class MaxFunctor{
		char dummy;
	public:
		template<typename Dst, typename A, typename B>
		__host__ __device__ void operator()(Dst& dst, A a, B b){
			dst=Max(a,b);
		}
		template<typename Dst, typename A>
		__host__ __device__ void operator()(Dst& dst, A a){
			dst=Max(dst,a);
		}
	};

	class MinFunctor{
		char dummy;
	public:
		template<typename Dst, typename A, typename B>
		__host__ __device__ void operator()(Dst& dst, A a, B b){
			dst=Min(a,b);
		}
		template<typename Dst, typename A>
		__host__ __device__ void operator()(Dst& dst, A a){
			dst=Min(dst,a);
		}
	};

	class AndFunctor{
		char dummy;
	public:
		template<typename Dst, typename A, typename B>
		__host__ __device__ void operator()(Dst& dst, A a, B b){
			dst=a&&b;
		}
		template<typename Dst, typename A>
		__host__ __device__ void operator()(Dst& dst, A a){
			dst=dst&&a;
		}
	};

	class OrFunctor{
		char dummy;
	public:
		template<typename Dst, typename A, typename B>
		__host__ __device__ void operator()(Dst& dst, A a, B b){
			return dst=a||b;
		}
		template<typename Dst, typename A>
		__host__ __device__ void operator()(Dst& dst, A a){
			dst=dst||a;
		}
	};

}
