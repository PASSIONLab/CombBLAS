#pragma once

template<typename T>
class StrideIter{
	T* p;
	int stride;
public:
	__device__ StrideIter(T* p,int stride):p(p),stride(stride){}
	__device__ void operator++(){p+=stride;}
	__device__ void operator++(int){p+=stride;}
	__device__ T& operator[](int i){return p[i*stride];}
	__device__ const T& operator *(){return *p;}
};

template<typename T>
class StrideIterConst{
	const T* p;
	int stride;
public:
	__device__ StrideIterConst(const T* p,int stride):p(p),stride(stride){}
	__device__ void operator++(){p+=stride;}
	__device__ void operator++(int){p+=stride;}
	__device__ const T& operator[](int i){return p[i*stride];}
	__device__ const T& operator *(){return *p;}
};


template<typename T, typename Iter1, typename Iter2>
__device__ void Dot(T& sum, Iter1 a, Iter2 b, int n){
	for(int i=0;i<n;i++){
		sum+=(*a)*(*b);
		a++;
		b++;
	}
}
