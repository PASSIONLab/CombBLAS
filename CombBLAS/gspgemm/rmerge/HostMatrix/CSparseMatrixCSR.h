#pragma once

#include "HostMatrix/devicehost.h"
#include "HostMatrix/CSparseVector.h"
#include "HostMatrix/VectorTypes.h"
typedef unsigned int uint;

//terminated CSR
template<typename T>
class CSparseMatrixCSR{
	int width;
	int height;
	T* values;//length is number of nonzeros	
	uint* colIndices;//length is number of nonzeros
	uint* rowStarts;//length is height+1(terminated CSR)
	int nonZeroCount;
public:
	//__device__ __host__ CSparseMatrixCSR(){}
	__device__ __host__ CSparseMatrixCSR(int width, int height,	T* values, uint* colIndices,uint* rowStarts, int nonZeroCount)
		:width(width),height(height),values(values),colIndices(colIndices),rowStarts(rowStarts),nonZeroCount(nonZeroCount){}

	__device__ __host__ int Width()const{return width;}	
	__device__ __host__ int Height()const{return height;}
	__device__ __host__ Int2 Size()const{return Int2(width,height);}
	__device__ __host__ int NonZeroCount()const{return nonZeroCount;}
	__device__ __host__ uint* RowStarts(){return rowStarts;}
	__device__ __host__ const uint* RowStarts()const{return rowStarts;}
	__device__ __host__ uint* ColIndices(){return colIndices;}
	__device__ __host__ const uint* ColIndices()const{return colIndices;}
	__device__ __host__ T* Values(){return values;}
	__device__ __host__ const T* Values()const{return values;}

	__device__ __host__ int RowLength(int r)const{
		uint rowStart=rowStarts[r];
		int rowLength=rowStarts[r+1]-rowStart;
		return rowLength;
	}

	__device__ __host__ unsigned int RowStart(int r)const{return rowStarts[r];}
	__device__ __host__ void GetRow(int r, T*& rowValues, uint*& rowIndices, int& rowLength){
		uint rowStart=rowStarts[r];
		rowLength=rowStarts[r+1]-rowStart;
		rowValues=values+rowStart;
		rowIndices=colIndices+rowStart;
	}	
	__device__ __host__ void GetRow(int r, const T* & rowValues, const uint* & rowIndices, int& rowLength)const{
		uint rowStart=rowStarts[r];
		rowLength=rowStarts[r+1]-rowStart;
		rowValues=values+rowStart;
		rowIndices=colIndices+rowStart;
	}
	__device__ __host__ CSparseVector<T> GetRow(int r){
		uint rowStart=rowStarts[r];
		int nonZeros=rowStarts[r+1]-rowStart;
		return CSparseVector<T>(values+rowStart,colIndices+rowStart,width,nonZeros);
	}
};