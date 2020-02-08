#pragma once

#include "HostMatrix/CVector.h"
#include "HostMatrix/Intrinsics.h"

#ifdef __CUDACC__

//Extracts a sparse subset of a dense vector
//src is dense, dst is sparse.
template<int ThreadCount, int PerThread, typename T>
__global__ void __cdecl CudaExtractSparseKernel(CVector<T> sparse, CVector<T> dense, CVector<uint> indices){
	uint sparseCount=(uint)sparse.Length();
	unsigned int start=blockIdx.x*ThreadCount*PerThread+threadIdx.x;
	unsigned int end=Min_rmerge(start+ThreadCount*PerThread,sparseCount);
	for(uint i=start;i<end;i+=ThreadCount){
		uint index=indices[i];
		sparse[i]=dense[index];
	}
}

template<bool AddUp, int ThreadCount, int PerThread, typename A, typename B>
__global__ void __cdecl CudaInjectSparseKernel(CVector<A> dense, CVector<B> sparse, CVector<uint> indices){
	uint sparseCount=(uint)sparse.Length();
	uint start=blockIdx.x*ThreadCount*PerThread+threadIdx.x;
	uint end=Min_rmerge(start+ThreadCount*PerThread,sparseCount);
	for(uint i=start;i<end;i+=ThreadCount){
		uint index=indices[i];
		if(AddUp)
			dense[index]+=sparse[i];
		else
			dense[index]=sparse[i];
	}
}


template<int WarpSize, int PerWarp, int BlockSizeY, typename T>
__global__ void __cdecl CudaExtractRowsKernel(CMatrix<T> dst, CMatrix<T> src, CVector<uint> rowIndices){
	uint sparseCount=(uint)rowIndices.Length();
	uint start=blockIdx.x*BlockSizeY*PerWarp+threadIdx.y;
	uint end=Min_rmerge(start+BlockSizeY*PerWarp,sparseCount);
	for(uint y=start;y<end;y+=BlockSizeY){
		uint srcIndex=rowIndices[y];
		//Copy row
		for(int x=threadIdx.x;x<dst.Width();x+=WarpSize){
			dst(x,y)=src(x,srcIndex);
		}
	}
}

template<int WarpSize, int PerWarp, int BlockSizeY, typename T>
__global__ void __cdecl CudaInjectRowsKernel(CMatrix<T> dst, CMatrix<T> src, CVector<uint> rowIndices){
	uint sparseCount=(uint)rowIndices.Length();
	uint start=blockIdx.x*BlockSizeY*PerWarp+threadIdx.y;
	uint end=Min_rmerge(start+BlockSizeY*PerWarp,sparseCount);
	for(uint y=start;y<end;y+=BlockSizeY){
		uint rowIndex=rowIndices[y];
		//Copy row
		for(int x=threadIdx.x;x<dst.Width();x+=WarpSize){
			dst(x,rowIndex)=src(x,y);
		}
	}
}

//****************************************
#define SP_BLOCK 1024
#define SP_PERTHREAD 64

template<typename T>
void __cdecl CudaExtractSparse(CVector<T> sparse, CVector<T> dense, CVector<uint> indices){
	dim3 gridDim(DivUp(uint(sparse.Length()),uint(SP_BLOCK*SP_PERTHREAD)),1,1);
	dim3 blockDim(SP_BLOCK,1,1);
	CudaExtractSparseKernel<SP_BLOCK,SP_PERTHREAD> <<< gridDim, blockDim, 0>>>(sparse,dense,indices);
}

template<typename T>
void __cdecl CudaExtractRows(CMatrix<T> dst, CMatrix<T> src, CVector<uint> rowIndices){
	const int WarpSize=16;
	const int BlockSizeY=32;
	const int PerWarp=32;
	dim3 gridDim(DivUp(rowIndices.Length32(),BlockSizeY*PerWarp),1,1);
	dim3 blockDim(WarpSize,BlockSizeY,1);
	CudaExtractRowsKernel<WarpSize,PerWarp,BlockSizeY> <<< gridDim, blockDim, 0>>>(dst,src,rowIndices);
}

template<typename T>
void __cdecl CudaInjectRows(CMatrix<T> dst, CMatrix<T> src, CVector<uint> rowIndices){
	const int WarpSize=16;
	const int BlockSizeY=32;
	const int PerWarp=32;
	dim3 gridDim(DivUp(rowIndices.Length32(),BlockSizeY*PerWarp),1,1);
	dim3 blockDim(WarpSize,BlockSizeY,1);
	CudaInjectRowsKernel<WarpSize,PerWarp,BlockSizeY> <<< gridDim, blockDim, 0>>>(dst,src,rowIndices);
}

template<bool AddUp, typename A, typename B>
void __cdecl CudaInjectSparse(CVector<A> dense, CVector<B> sparse, CVector<uint> indices){
	dim3 gridDim(DivUp(uint(sparse.Length()),uint(SP_BLOCK*SP_PERTHREAD)),1,1);
	dim3 blockDim(SP_BLOCK,1,1);
	CudaInjectSparseKernel<AddUp,SP_BLOCK,SP_PERTHREAD> <<< gridDim, blockDim, 0>>>(dense,sparse,indices);
}

#else

template<typename T> void __cdecl CudaExtractSparse(CVector<T> sparse, CVector<T> dense, CVector<uint> indices);
template<bool AddUp, typename A, typename B> void __cdecl CudaInjectSparse(CVector<A> dense, CVector<B> sparse, CVector<uint> indices);
template<typename T> void __cdecl CudaExtractRows(CMatrix<T> dst, CMatrix<T> src, CVector<uint> rowIndices);
template<typename T> void __cdecl CudaInjectRows(CMatrix<T> dst, CMatrix<T> src, CVector<uint> rowIndices);

#endif
