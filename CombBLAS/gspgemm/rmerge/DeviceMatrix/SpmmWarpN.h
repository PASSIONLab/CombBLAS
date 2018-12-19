#pragma once

//This file contains code for RMerge1.1, which is an extension of 
//RMerge1 (Gremse et al. 2015 http://epubs.siam.org/doi/abs/10.1137/130948811)

#include "DeviceMatrix/SparseDeviceMatrixCSR.h"
#include "DeviceMatrix/Scan.h"
#include "DeviceMatrix/DeviceReductions.h"
#include "HostMatrix/Intrinsics.h"
#include "HostMatrix/MinMaxValues.h"
#include "HostMatrix/IO.h"
#include "DeviceMatrix/ldg.h"
#include "DeviceMatrix/WarpReduction.h"
#include "DeviceMatrix/MulWarp.h"
#include "DeviceMatrix/MulWarpN.h"
#include "DeviceMatrix/SparseDeviceMatrixCSROperations.h"

//Sparse matrix matrix multiplication by row merging
//Each warp (can be a subwarp) computes c=a*B for a single row a of A.
//Each thread of the warp maintains RowsPerThread rows of B.
//The parameters AssumeOnes is used for iterative row merging, i.e. is set to false only at the first iteration
template<int WarpSize, int RowsPerThread, int BlockDimY, bool AssumeOnes, typename T>
__global__ void __cdecl SpmmWarpNKernel(CSparseMatrixCSR<T> C, CSparseMatrixCSR<T> A, CSparseMatrixCSR<T> B){	
	int r=threadIdx.y+blockIdx.x*BlockDimY;
	if(r>=C.Height())
		return;	
	CSparseVector<T> c=C.GetRow(r);
	CSparseVector<T> a=A.GetRow(r);
	if(RowsPerThread==1)
		MulWarp<WarpSize,AssumeOnes>(c,a,B,threadIdx.x,A.RowStart(r));	
	else
		MulWarpN<WarpSize,RowsPerThread,AssumeOnes>(c,a,B,threadIdx.x,A.RowStart(r));
}

//Sparse matrix matrix multiplication by row merging
//Each warp (can be a subwarp) computes c=a*B for a single row a of A.
//Each thread of the warp maintains RowsPerThread rows of B.
template<int WarpSize, int RowsPerThread, typename T>
void __cdecl SpmmWarpN(SparseDeviceMatrixCSR<T> C, SparseDeviceMatrixCSR<T> A, SparseDeviceMatrixCSR<T> B, bool assumeOnes)
#ifdef __CUDACC__
{
	const int WarpCount=256/WarpSize;
	dim3 blockDim(WarpSize,WarpCount,1);
	dim3 gridDim(DivUp(C.Height(),(int)blockDim.y),1,1);
	if(assumeOnes)
		SpmmWarpNKernel<WarpSize,RowsPerThread,WarpCount,true> <<< gridDim, blockDim, 0>>>(C.GetC(),A.GetC(),B.GetC());
	else
		SpmmWarpNKernel<WarpSize,RowsPerThread,WarpCount,false> <<< gridDim, blockDim, 0>>>(C.GetC(),A.GetC(),B.GetC());
}
#else
;
#endif

//Size prediction for sparse matrix matrix multiplication by row merging.
//Each warp computes the sized of c=a*B for a single row a of A.
//Each thread of the warp maintains RowsPerThread rows of B.
template<int WarpSize, int RowsPerThread, int BlockDimY,bool AssumeOnes, typename T>
__global__ void __cdecl SpmmWarpPredictSizeNKernel(uint* dstLengths, CSparseMatrixCSR<T> A, CSparseMatrixCSR<T> B){	
	int r=threadIdx.y+blockIdx.x*BlockDimY;
	if(r>=A.Height())
		return;
	uint dst;
	if(RowsPerThread==1)
		dst=MulWarpPredictSize<WarpSize,AssumeOnes>(A.GetRow(r),B,threadIdx.x,A.RowStart(r));
	else
		dst=MulWarpPredictSizeN<WarpSize,RowsPerThread,AssumeOnes>(A.GetRow(r),B,threadIdx.x,A.RowStart(r));
	if(threadIdx.x==0)
		dstLengths[r]=dst;
}

//Size prediction for sparse matrix matrix multiplication by row merging.
//Each warp computes the sized of c=a*B for a single row a of A.
//Each thread of the warp maintains RowsPerThread rows of B.
template<int WarpSize, int RowsPerThread, typename T>
void __cdecl SpmmWarpPredictSizeN(DeviceVector<uint> dstLengths, SparseDeviceMatrixCSR<T> A, SparseDeviceMatrixCSR<T> B,bool assumeOnes)
#ifdef __CUDACC__
{	
	const int WarpCount=256/WarpSize;
	dim3 blockDim(WarpSize,WarpCount,1);
	dim3 gridDim(DivUp(A.Height(),(int)blockDim.y),1,1);
	if(assumeOnes)
		SpmmWarpPredictSizeNKernel<WarpSize,RowsPerThread,WarpCount,true> <<< gridDim, blockDim, 0>>>(dstLengths.Data(),A.GetC(),B.GetC());
	else
		SpmmWarpPredictSizeNKernel<WarpSize,RowsPerThread,WarpCount,false> <<< gridDim, blockDim, 0>>>(dstLengths.Data(),A.GetC(),B.GetC());
}
#else
;
#endif

//Compute the size of C=A*B and allocate the memory.
template<int WarpSize, int RowsPerThread, typename T>
static SparseDeviceMatrixCSR<T> MulLimitedTPrepareN(SparseDeviceMatrixCSR<T> A, SparseDeviceMatrixCSR<T> B,bool assumeOnes){		
	//Compute dst size
	DeviceVector<uint> rowStarts(A.Height()+1);	
	SpmmWarpPredictSizeN<WarpSize,RowsPerThread>(rowStarts.SubVector(0,A.Height()),A,B,assumeOnes);	
	ScanExclusive(rowStarts);
	uint nonZeros=rowStarts[rowStarts.Length()-1];	
	SparseDeviceMatrixCSR<T> dst(B.Width(),A.Height(),DeviceVector<T>(nonZeros),DeviceVector<uint>(nonZeros),rowStarts);
	return dst;
}

//compute C=A*B but A must have row lengths of maximal MergeFactor
template<int WarpSize, int RowsPerThread, typename T>
static SparseDeviceMatrixCSR<T> MulLimitedTN(SparseDeviceMatrixCSR<T> A, SparseDeviceMatrixCSR<T> B, bool assumeOnes=false){
	SparseDeviceMatrixCSR<T> dst=MulLimitedTPrepareN<WarpSize,RowsPerThread>(A,B,assumeOnes);
	SpmmWarpN<WarpSize,RowsPerThread>(dst,A,B,assumeOnes);	
	return dst;
}

//Performs multiplication of A*B but A must have maximum row length of mergeFactor.
template<typename T>
static SparseDeviceMatrixCSR<T> MulLimitedN(SparseDeviceMatrixCSR<T> A, SparseDeviceMatrixCSR<T> B, int maxRowLength, bool assumeOnes=false){	
	if(maxRowLength<=2)
		return MulLimitedTN<2,1>(A,B,assumeOnes);
	else if(maxRowLength<=4)
		return MulLimitedTN<4,1>(A,B,assumeOnes);
	else if(maxRowLength<=8)
		return MulLimitedTN<8,1>(A,B,assumeOnes);
	else if(maxRowLength<=12)
		return MulLimitedTN<8,2>(A,B,assumeOnes);
	else if(maxRowLength<=16)
		return MulLimitedTN<8,2>(A,B,assumeOnes);
	else if(maxRowLength<=24)
		return MulLimitedTN<8,3>(A,B,assumeOnes);
	else if(maxRowLength<=32)
		return MulLimitedTN<16,2>(A,B,assumeOnes);
	else if(maxRowLength<=48)
		return MulLimitedTN<16,3>(A,B,assumeOnes);
	else if(maxRowLength<=64)
		return MulLimitedTN<32,2>(A,B,assumeOnes);
	else if(maxRowLength<=96)
		return MulLimitedTN<32,3>(A,B,assumeOnes);
	else if(maxRowLength<=128)
		return MulLimitedTN<32,4>(A,B,assumeOnes);
	else if(maxRowLength<=192)
		return MulLimitedTN<32,6>(A,B,assumeOnes);
	else if(maxRowLength<=256)
		return MulLimitedTN<32,8>(A,B,assumeOnes);
	else throw std::runtime_error("mergeFactor must be 2,4,8,16,... or 256");
}

//Compute C=A*B similar to RMerge1 except that the merge factor can be as high as 256.
//Iteratively split A into A1*A2 and multiply A2 with B.
template<typename T>
static SparseDeviceMatrixCSR<T> MulRMerge11(SparseDeviceMatrixCSR<T> A, SparseDeviceMatrixCSR<T> B, int mergeFactor=256){
	SparseDeviceMatrixCSR<T> left=A;
	SparseDeviceMatrixCSR<T> right=B;	
	int maxRowLength=(int)MaxRowLength(left);	
	bool assumeOnes=false;
	while(maxRowLength>mergeFactor){
		SparseDeviceMatrixCSR<T> tmp1;
		SparseDeviceMatrixCSR<T> tmp2;		
		SpmmWarpDecompose(tmp1,tmp2,left,mergeFactor);		
		left=tmp1;
		right=MulLimitedN(tmp2,right,mergeFactor,assumeOnes);
		assumeOnes=true;//in later iterations, we can assume that the weights are ones
		maxRowLength=DivUp(maxRowLength,mergeFactor);
	}
	return MulLimitedN(left,right,maxRowLength,assumeOnes);
}
