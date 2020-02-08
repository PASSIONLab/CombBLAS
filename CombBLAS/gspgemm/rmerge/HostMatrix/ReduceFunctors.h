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
			dst=Max_rmerge(a,b);
		}
		template<typename Dst, typename A>
		__host__ __device__ void operator()(Dst& dst, A a){
			dst=Max_rmerge(dst,a);
		}
	};

	class MinFunctor{
		char dummy;
	public:
		template<typename Dst, typename A, typename B>
		__host__ __device__ void operator()(Dst& dst, A a, B b){
			dst=Min_rmerge(a,b);
		}
		template<typename Dst, typename A>
		__host__ __device__ void operator()(Dst& dst, A a){
			dst=Min_rmerge(dst,a);
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
