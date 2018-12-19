#pragma once

//This file contains code for RMerge1 (Gremse et al. 2015 http://epubs.siam.org/doi/abs/10.1137/130948811)

#include "DeviceMatrix/SparseDeviceMatrixCSR.h"
#include "DeviceMatrix/Scan.h"
#include "DeviceMatrix/DeviceReductions.h"
#include "HostMatrix/Intrinsics.h"
#include "HostMatrix/MinMaxValues.h"
#include "HostMatrix/IO.h"
#include "DeviceMatrix/ldg.h"
#include "DeviceMatrix/WarpReduction.h"

#include "DeviceMatrix/MulWarp.h"
#include "DeviceMatrix/SpmmWarpDecompose.h"
#include "DeviceMatrix/SparseDeviceMatrixCSROperations.h"


//Sparse matrix matrix multiplication by row merging
template<int WarpSize, int BlockDimY, bool AssumeOnes, typename T>
__global__ void __cdecl SpmmWarpKernel(CSparseMatrixCSR<T> C, CSparseMatrixCSR<T> A, CSparseMatrixCSR<T> B){	
	int r=threadIdx.y+blockIdx.x*BlockDimY;
	if(r>=C.Height())
		return;	
	CSparseVector<T> c=C.GetRow(r);
	CSparseVector<T> a=A.GetRow(r);
	MulWarp<WarpSize,AssumeOnes>(c,a,B,threadIdx.x,A.RowStart(r));
}


//Sparse matrix matrix multiplication C=A*B.
//Rows of A must have at most WarpSize elements.
//Each warp computes the product c=aB, where a is one row of A and c ist a row of C.
//C must be preallocated.
template<int WarpSize,typename T>
void __cdecl SpmmWarp(SparseDeviceMatrixCSR<T> C, SparseDeviceMatrixCSR<T> A, SparseDeviceMatrixCSR<T> B,bool assumeOnes)
#ifdef __CUDACC__
{	
	const int WarpCount=128/WarpSize;
	dim3 blockDim(WarpSize,WarpCount,1);
	dim3 gridDim(DivUp(C.Height(),(int)blockDim.y),1,1);
	if(assumeOnes)
		SpmmWarpKernel<WarpSize,WarpCount,true> <<<gridDim, blockDim, 0>>>(C.GetC(),A.GetC(),B.GetC());
	else
		SpmmWarpKernel<WarpSize,WarpCount,false> <<<gridDim, blockDim, 0>>>(C.GetC(),A.GetC(),B.GetC());
}
#else
;
#endif

//***************************************************************************************

//Predict row lengths of C=A*B (kernel instantiation)
template<int WarpSize, int BlockDimY, bool AssumeOnes, typename T>
__global__ void __cdecl SpmmWarpPredictSizeKernel(uint* dstLengths, CSparseMatrixCSR<T> A, CSparseMatrixCSR<T> B){	
	int r=threadIdx.y+blockIdx.x*BlockDimY;//not enough blocks possible?
	if(r>=A.Height())
		return;
	uint dstLength=MulWarpPredictSize<WarpSize,AssumeOnes>(A.GetRow(r),B,threadIdx.x,A.RowStart(r));
	if(threadIdx.x==0)
		dstLengths[r]=dstLength;
}

//Predict row lengths of C=A*B
template<int WarpSize, typename T>
void __cdecl SpmmWarpPredictSize(DeviceVector<uint> dstLengths, SparseDeviceMatrixCSR<T> A, SparseDeviceMatrixCSR<T> B, bool assumeOnes)
#ifdef __CUDACC__
{	
	const int WarpCount=256/WarpSize;
	dim3 blockDim(WarpSize,WarpCount,1);
	dim3 gridDim(DivUp(A.Height(),(int)blockDim.y),1,1);
	if(assumeOnes)
		SpmmWarpPredictSizeKernel<WarpSize,WarpCount,true> <<< gridDim, blockDim, 0>>>(dstLengths.Data(),A.GetC(),B.GetC());
	else
		SpmmWarpPredictSizeKernel<WarpSize,WarpCount,false> <<< gridDim, blockDim, 0>>>(dstLengths.Data(),A.GetC(),B.GetC());
}
#else
;
#endif

//Predict row lengths of C=A*B and allocates the matrix C
template<int MergeFactor, typename T>
static SparseDeviceMatrixCSR<T> MulLimitedTPrepare(SparseDeviceMatrixCSR<T> A, SparseDeviceMatrixCSR<T> B,bool assumeOnes=false){		
	//Compute dst size
	DeviceVector<uint> rowStarts(A.Height()+1);	
	SpmmWarpPredictSize<MergeFactor>(rowStarts.SubVector(0,A.Height()),A,B,assumeOnes);	
	ScanExclusive(rowStarts);
	uint nonZeros=rowStarts[rowStarts.Length()-1];	
	SparseDeviceMatrixCSR<T> dst(B.Width(),A.Height(),DeviceVector<T>(nonZeros),DeviceVector<uint>(nonZeros),rowStarts);
	return dst;
}

//Compute A*B but A must have row lengths of maximal MergeFactor
template<int MergeFactor, typename T>
static SparseDeviceMatrixCSR<T> MulLimitedT(SparseDeviceMatrixCSR<T> A, SparseDeviceMatrixCSR<T> B, bool assumeOnes=false){
	SparseDeviceMatrixCSR<T> dst=MulLimitedTPrepare<MergeFactor>(A,B,assumeOnes);
	SpmmWarp<MergeFactor>(dst,A,B,assumeOnes);
	return dst;
}

//Performs multiplication of A*B but A must have maximum row length of mergeFactor.
template<typename T>
static SparseDeviceMatrixCSR<T> MulLimited(SparseDeviceMatrixCSR<T> A, SparseDeviceMatrixCSR<T> B, int mergeFactor, bool assumeOnes){	
	if(mergeFactor==2)
		return MulLimitedT<2>(A,B,assumeOnes);
	else if(mergeFactor==4)
		return MulLimitedT<4>(A,B,assumeOnes);
	else if(mergeFactor==8)
		return MulLimitedT<8>(A,B,assumeOnes);
	else if(mergeFactor==16)
		return MulLimitedT<16>(A,B,assumeOnes);
	else if(mergeFactor==32)
		return MulLimitedT<32>(A,B,assumeOnes);	
	else throw std::runtime_error("mergeFactor must be 2,4,8,16 or 32");
}

//Iteratively split A into A1*A2 and multiply A2 with B.
template<typename T>
static SparseDeviceMatrixCSR<T> MulRMerge1(SparseDeviceMatrixCSR<T> A, SparseDeviceMatrixCSR<T> B, int mergeFactor=16, bool adjust=true){
	SparseDeviceMatrixCSR<T> left=A;
	SparseDeviceMatrixCSR<T> right=B;	
	int maxRowLength=(int)MaxRowLength(left);	
	bool assumeOnes=false;
	while(maxRowLength>mergeFactor){
		SparseDeviceMatrixCSR<T> tmp1;
		SparseDeviceMatrixCSR<T> tmp2;		
		SpmmWarpDecompose(tmp1,tmp2,left,mergeFactor);		
		left=tmp1;
		right=MulLimited(tmp2,right,mergeFactor,assumeOnes);
		assumeOnes=true;//in later iterations, we can assume that the weights are ones
		maxRowLength=DivUp(maxRowLength,mergeFactor);		
	}
	//Compute the final multiplication but use the smallest possible subwarp size.
	if(adjust)
		mergeFactor=SufficientSubWarpSize(maxRowLength);
	return MulLimited(left,right,mergeFactor,assumeOnes);
}


