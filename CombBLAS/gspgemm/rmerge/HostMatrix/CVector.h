#pragma once

#include "HostMatrix/devicehost.h"
#include "HostMatrix/VectorTypes.h"
#include "HostMatrix/StrideIter.h"
#include "HostMatrix/int64.h"

template<typename T>
class CVector{
public:
	T* data;
	int64 length;
	int stride;
public:
	__device__ __host__ CVector():length(0),stride(1),data(0){}
	//__device__ __host__ CVector(const CVector& other):length(other.length),stride(other.stride),data(other.data){}
	__device__ __host__ CVector(T* data,int64 length):data(data),length(length),stride(1){}
	__device__ __host__ CVector(T* data,int64 length, int stride):data(data),length(length),stride(stride){}		
	__device__ __host__ int64 Index(int64 i)const{return i*stride;}
	__device__ __host__ int Index(int i)const{return i*stride;}
	__device__ __host__ T& operator[](int i){return data[i*stride];}
	__device__ __host__ T& operator[](unsigned int i){return data[i*stride];}
	__device__ __host__ T& operator[](int64 i){return data[i*stride];}
	__device__ __host__ T& operator[](uint64 i){return data[i*stride];}
	__device__ __host__ const T& operator[](int i)const{return data[i*stride];}
	__device__ __host__ const T& operator[](unsigned int i)const{return data[i*stride];}
	__device__ __host__ const T& operator[](int64 i)const{return data[i*stride];}
	__device__ __host__ int64 Length()const{return length;}
	__device__ __host__ int Length32()const{return (int)length;}
	__device__ __host__ int64 DimX()const{return length;}
	__device__ __host__ int64 Size()const{return length;}
	__device__ __host__ int Stride()const{return stride;}
	__device__ __host__ bool IsSimple()const{return stride==1;}
	__device__ __host__ T* Data(){return data;}
	__device__ __host__ const T* Data()const{return data;}
	__device__ __host__ CVector SubVector(int start, int length){return CVector(data+start*stride,length,stride);}
	__device__ __host__ CVector SubVector(int64 start, int64 length){return CVector(data+start*stride,length,stride);}
	__device__ __host__ StrideIter<T> begin(){return StrideIter<T>(data,stride);}
	__device__ __host__ StrideIter<T> end(){return StrideIter<T>(data+length*stride,stride);}
};