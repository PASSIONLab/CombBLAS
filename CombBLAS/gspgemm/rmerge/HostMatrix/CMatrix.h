#pragma once

#include "HostMatrix/VectorTypes.h"
#include "HostMatrix/CVector.h"
#include "HostMatrix/StrideIter.h"

template<typename T>
class CMatrix{
	T* data;
	int dimX;
	int dimY;	
	int stride;	
public:
	__device__ __host__ CMatrix():data(0),dimX(0),dimY(0),stride(1){}
	//__device__ __host__ CMatrix(const CMatrix& other):dimX(other.dimX),dimY(other.dimY),stride(other.stride),data(other.data){}	
	__device__ __host__ CMatrix(T* data, int dimX, int dimY, int stride):dimX(dimX),dimY(dimY),stride(stride),data(data){}	
	__device__ __host__ CMatrix(T* data,int dimX, int dimY):dimX(dimX),dimY(dimY),stride(dimX),data(data){}	
	__device__ __host__ CMatrix(T* data,Int2 size):dimX(size.x),dimY(size.y),stride(size.x),data(data){}	
	__device__ __host__ bool IsSimple()const{return stride==dimX;}
	__device__ __host__ CVector<T> GetSimple(){return CVector<T>(data,dimX*(int64)dimY);}
	__device__ __host__ int Index(int x, int y)const{return y*(int64)stride+x;}
	__device__ __host__ void Set(int x, int y, T e){data[Index(x,y)]=e;}
	__device__ __host__ T Get(int x, int y){return data[Index(x,y)];}
	__device__ __host__ const T& operator()(int x, int y)const{return data[Index(x,y)];}
	__device__ __host__ T& operator()(int x, int y){return data[Index(x,y)];}
	__device__ __host__ const T& operator()(Int2 pos)const{return data[Index(pos.x,pos.y)];}
	__device__ __host__ T& operator()(Int2 pos){return data[Index(pos.x,pos.y)];}
	__device__ __host__ CVector<T> Row(int y){return CVector<T>(data+y*(int64)stride,dimX,1);}
	__device__ __host__ CVector<T> Column(int x){return CVector<T>(data+x,dimY,stride);}
	__device__ __host__ StrideIter<T> ColumnIterator(int x){return StrideIter<T>(data+x,stride);}
	__device__ __host__ CVector<T> Diagonal(){return CVector<T>(data,dimY<dimX?dimY:dimX,stride+1);}
	__device__ __host__ CMatrix<T> SubMatrix(int startY, int dimY){return CMatrix<T>(data+stride*(int64)startY,dimX, dimY, stride);}
	__device__ __host__ CMatrix<T> SubMatrix(int x, int y, int dimX, int dimY){return CMatrix<T>(data+stride*(int64)y+x,dimX, dimY, stride);}
	__device__ __host__ T* RowPointer(int y){return data+y*(int64)stride;}
	__device__ __host__ T* RowPointerX(int y){return data+y*(int64)stride;}
	__device__ __host__ T* RowPointerY(int x){return data+x;}
	__device__ __host__ T* Data(){return data;}
	__device__ __host__ const T* Data()const{return data;}
	__device__ __host__ int Width() const {return dimX;}
	__device__ __host__ int Height() const {return dimY;}
	__device__ __host__ int DimX() const {return dimX;}
	__device__ __host__ int DimY() const {return dimY;}
	__device__ __host__ Int2 Size() const {Int2 size;size.x=dimX;size.y=dimY;return size;}
	__device__ __host__ int Stride() const {return stride;}
	__device__ __host__ int RowStride() const {return stride;}
};