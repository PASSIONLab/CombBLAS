#pragma once

#include "HostMatrix/devicehost.h"

typedef unsigned int uint;

template<typename T>
class CSparseVector{
public:	
	T* values;	
	uint* indices;
	int length;
	int nonZeroCount;
public:
	__device__ __host__ CSparseVector(T* values, uint* indices, int length, int nonZeroCount):values(values),indices(indices),length(length),nonZeroCount(nonZeroCount){}

	
	__device__ __host__ int Length()const{return length;}
	__device__ __host__ int DimX()const{return length;}
	__device__ __host__ int NonZeroCount()const{return nonZeroCount;}
	__device__ __host__ T* Values(){return values;}
	__device__ __host__ const T* Values()const{return values;}
	__device__ __host__ uint* Indices(){return indices;}
	__device__ __host__ const uint* Indices()const{return indices;}
	__device__ __host__ T& Value(int i){return values[i];}
	__device__ __host__ const T& Value(int i)const{return values[i];}
	__device__ __host__ uint& Index(int i){return indices[i];}
	__device__ __host__ const uint& Index(int i)const{return indices[i];}
};