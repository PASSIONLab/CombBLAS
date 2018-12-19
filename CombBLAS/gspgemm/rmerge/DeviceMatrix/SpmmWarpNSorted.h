#pragma once

//This file contains code for RMerge2, an extension of 
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
#include "DeviceMatrix/MulWarpMany.h"
#include "DeviceMatrix/SparseDeviceMatrixCSROperations.h"
#include "DeviceMatrix/Sort.h"
#include "DeviceMatrix/Range.h"
#include "DeviceMatrix/SpmmCases.h"

//Returns the Cuda compute capability (major).
//Kepler is 3
//Maxwell is 5
//Pascal is 6
//Volta is 7?
static int CudaComputeCapabilityMajor() {

	cudaDeviceProp props;
	int device;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&props, device);
	return props.major;
}

#ifdef __CUDACC__
//Performes row scaling, i.e. row merging of only one row.
//The function is meant for a thread block. 
//Each row is scaled by one sub warp.
//A permution p is provided, as well as number n of rows.  
template<uint SubWarpSize, uint BlockSize, typename T>
__device__ void ScaleRows(CSparseMatrixCSR<T>& C, CSparseMatrixCSR<T>& A, const CSparseMatrixCSR<T>& B, uint* p, int n, uint start, uint thread){
	//Each subwarp computes one row
	uint subWarp=thread/SubWarpSize;
	uint threadx=thread%SubWarpSize;
	if(start+subWarp>=n)
		return;
	uint r=p[start+subWarp];
	CSparseVector<T> c=C.GetRow(r);
	CSparseVector<T> a=A.GetRow(r);
	
	const T* rowValues;const uint *rowIndices;int rowLength;
	int rB=a.Index(0);
	T weight=a.Value(0);
	B.GetRow(rB,rowValues,rowIndices,rowLength);
	for(int i=threadx;i<rowLength;i+=SubWarpSize){
		c.Index(i)=rowIndices[i];
		c.Value(i)=weight*rowValues[i];
	}	
}
#endif

#ifdef __CUDACC__
//Turns a block into subwarps. 
//Then predicts the size.
//p has length of BlockSize
template<uint SubWarpSize, uint RowsPerThread, uint BlockSize, typename T>
__device__ void MulWarpsBlock(CSparseMatrixCSR<T>& C, CSparseMatrixCSR<T>& A, const CSparseMatrixCSR<T>& B, uint* p, int n, uint start, uint thread){
	//Each subwarp computes one row
	uint subWarp=thread/SubWarpSize;
	uint threadx=thread%SubWarpSize;
	if(start+subWarp>=n)
		return;
	uint r=p[start+subWarp];
	CSparseVector<T> c=C.GetRow(r);
	if(RowsPerThread==1)
		MulWarp<SubWarpSize,false>(c,A.GetRow(r),B,threadx,A.RowStart(r));
	else
		MulWarpN<SubWarpSize,RowsPerThread,false>(c,A.GetRow(r),B,threadx,A.RowStart(r));
}
#endif

//Returns the number of cases
static int CaseCount(){return 24;}

//These are the merging capabilities of the cases, listed in Table 2.1.
class MergingCapabilities {
public:
	static const int MC_01 = 32 * 1024;
	static const int MC_02 = 16 * 1024;
	static const int MC_03 = 8 * 1024;
	static const int MC_04 = 4 * 1024;
	static const int MC_05 = 2 * 1024;
	static const int MC_06 = 1024;
	static const int MC_07 = 512;
	static const int MC_08 = 384;
	static const int MC_09 = 256;
	static const int MC_10 = 192;
	static const int MC_11 = 128;
	static const int MC_12 = 96;
	static const int MC_13 = 64;
	static const int MC_14 = 48;
	static const int MC_15 = 32;
	static const int MC_16 = 24;
	static const int MC_17 = 16;
	static const int MC_18 = 12;
	static const int MC_19 = 8;
	static const int MC_20 = 4;
	static const int MC_21 = 2;
	static const int MC_22 = 1;
	static const int MC_23 = 0;
};

//Returns the case for a given row length.
static __device__ __host__ uint RowLengthToCase(uint rowLength){	
	if (rowLength == MergingCapabilities::MC_23) { return 23; }
	else if (rowLength == MergingCapabilities::MC_22) { return 22; }
	else if (rowLength <= MergingCapabilities::MC_21) { return 21; }
	else if (rowLength <= MergingCapabilities::MC_20) { return 20; }
	else if (rowLength <= MergingCapabilities::MC_19) { return 19; }
	else if (rowLength <= MergingCapabilities::MC_18) { return 18; }
	else if (rowLength <= MergingCapabilities::MC_17) { return 17; }
	else if (rowLength <= MergingCapabilities::MC_16) { return 16; }
	else if (rowLength <= MergingCapabilities::MC_15) { return 15; }
	else if (rowLength <= MergingCapabilities::MC_14) { return 14; }
	else if (rowLength <= MergingCapabilities::MC_13) { return 13; }
	else if (rowLength <= MergingCapabilities::MC_12) { return 12; }
	else if (rowLength <= MergingCapabilities::MC_11) { return 11; }
	else if (rowLength <= MergingCapabilities::MC_10) { return 10; }
	else if (rowLength <= MergingCapabilities::MC_09) { return 9; }
	else if (rowLength <= MergingCapabilities::MC_08) { return 8; }
	else if (rowLength <= MergingCapabilities::MC_07) { return 7; }
	else if (rowLength <= MergingCapabilities::MC_06) { return 6; }
	else if (rowLength <= MergingCapabilities::MC_05) { return 5; }
	else if (rowLength <= MergingCapabilities::MC_04) { return 4; }
	else if (rowLength <= MergingCapabilities::MC_03) { return 3; }
	else if (rowLength <= MergingCapabilities::MC_02) { return 2; }
	else if (rowLength <= MergingCapabilities::MC_01) { return 1; }
	else return 0;//too large
}

//Scale rows. This is equivalent to merge size 1, i.e. one row is merged. The set of rows in A (and C) is defined by the index vector p, with length n. 
//Each such row of A must have exactly one nonzero entry which defines the weight and row in B.
template<int BlockSize, int WarpSize, typename T>
__global__ void __cdecl ScaleRowsKernel(CSparseMatrixCSR<T> C, CSparseMatrixCSR<T> A, CSparseMatrixCSR<T> B, uint* p, int n, int rowsPerBlock) {
	uint start = rowsPerBlock*blockIdx.x;
	ScaleRows<WarpSize, BlockSize>(C, A, B, p, n, start, threadIdx.x);
}

//Scale rows
template<int Case, int BlockSize, int WarpSize, typename T>
void __cdecl ScaleRows(SparseDeviceMatrixCSR<T>& C, SparseDeviceMatrixCSR<T>& A, SparseDeviceMatrixCSR<T>& B, DeviceVector<uint>& p, HostVector<uint>& rowsPerCase, HostVector<uint>& cumulativeRowsPerCase, HostVector<cudaStream_t>& streams, int& counter)
#ifdef __CUDACC__
{
	int rowCount = rowsPerCase[Case];
	if (rowCount == 0)
		return;
	dim3 blockDim(BlockSize, 1, 1);
	int warps = BlockSize / 32;
	int rowsPerBlock = warps*(32 / WarpSize);
	int rowStart = cumulativeRowsPerCase[Case] - rowCount;
	DeviceVector<uint> sub = p.SubVector(rowStart, rowCount);
	dim3 gridDim(DivUp(rowCount, rowsPerBlock), 1, 1);
	cudaStream_t stream = streams[counter%streams.Length32()];
	ScaleRowsKernel<BlockSize, WarpSize> << < gridDim, blockDim, 0, stream >> >(C.GetC(), A.GetC(), B.GetC(), sub.Pointer(), sub.Length32(), rowsPerBlock);
	counter++;
}
#else
;
#endif


//Sparse matrix matrix multiplication by row merging using a whole block.
//Warpsize must be 32. Each thread maintains multiple input rows. The results of the warps are merged by the first warp and stored in C.
template<int BlockSize, int WarpSize, int WarpCount, int RPT, typename T>
__global__ void __cdecl SpmmNSortedMulLimitedCaseBlockKernel(CSparseMatrixCSR<T> C, CSparseMatrixCSR<T> A, CSparseMatrixCSR<T> B, uint* p, int n, int rowsPerBlock) {
	uint start = rowsPerBlock*blockIdx.x;
	const int W = 32;
	//Some low amount of shared memory is needed for the block-based method for larger numbers of rows.
	__shared__ byte s[W * sizeof(T) + W * sizeof(uint) + sizeof(uint)];
	
	int r = p[start]; 
	CSparseVector<T> c = C.GetRow(r);	
	MulWarpMany<WarpSize, WarpCount, RPT>(c, A.GetRow(r), B, threadIdx.x % W, threadIdx.x / W, s);
}

//Compute C=A*B but only for a subset of rows of C. 
//The subset is indicated by the permutation p in combination with rowsPerCase and cumulativeRowsPerCase.
//The kernel is assigned to streams[counter% streamCount] and counter is increased.
template<int Case, int MergingCapability, int BlockSize, int SM, typename T>
void __cdecl SpmmNSortedMulLimitedCaseBlock(SparseDeviceMatrixCSR<T>& C, SparseDeviceMatrixCSR<T>& A, SparseDeviceMatrixCSR<T>& B, DeviceVector<uint>& p, HostVector<uint>& rowsPerCase, HostVector<uint>& cumulativeRowsPerCase, HostVector<cudaStream_t>& streams, int& counter)
#ifdef __CUDACC__
{
	int rowCount = rowsPerCase[Case];
	if (rowCount == 0)
		return;
	dim3 blockDim(BlockSize, 1, 1);
	int rowsPerBlock = 1;
	int rowStart = cumulativeRowsPerCase[Case] - rowCount;
	DeviceVector<uint> sub = p.SubVector(rowStart, rowCount);
	dim3 gridDim(rowCount, 1, 1);
	cudaStream_t stream = streams[counter%streams.Length32()];
	int sharedSize = SM;
	const int WarpSize = 32;
	const int WarpCount = BlockSize / WarpSize;
	const int RPT = (MergingCapability + BlockSize - 1) / BlockSize;
	SpmmNSortedMulLimitedCaseBlockKernel<BlockSize, WarpSize, WarpCount, RPT> << < gridDim, blockDim, sharedSize, stream >> >(C.GetC(), A.GetC(), B.GetC(), sub.Pointer(), sub.Length32(), rowsPerBlock);
	counter++;
}
#else
;
#endif

//Compute C=A*B but only for a subset of rows of C, indicated by the index vector p with length n.
//Each output row c of C is computed using one warp (or subwarp). Each thread maintains RPT rows in B (or less).
//This function requires that rows a of A have at most WarpSize*RPT nonzero elements.
template<int BlockSize, int WarpSize, int RPT, typename T>
__global__ void __cdecl SpmmNSortedMulLimitedCaseWarpKernel(CSparseMatrixCSR<T> C, CSparseMatrixCSR<T> A, CSparseMatrixCSR<T> B, uint* p, int n, int rowsPerBlock) {
	uint start = rowsPerBlock*blockIdx.x;		
	MulWarpsBlock<WarpSize, RPT, BlockSize>(C, A, B, p, n, start, threadIdx.x);	
}

//Compute C=A*B but only for a subset of rows of C. 
//The subset is indicated by the permutation p in combination with rowsPerCase and cumulativeRowsPerCase.
//The kernel is assigned to streams[counter% streamCount] and counter is increased.
template<int Case, int MergingCapability, int BlockSize, int WarpSize, int SM, typename T>
void __cdecl SpmmNSortedMulLimitedCaseWarp(SparseDeviceMatrixCSR<T>& C, SparseDeviceMatrixCSR<T>& A, SparseDeviceMatrixCSR<T>& B, DeviceVector<uint>& p, HostVector<uint>& rowsPerCase, HostVector<uint>& cumulativeRowsPerCase, HostVector<cudaStream_t>& streams, int& counter)
#ifdef __CUDACC__
{
	int rowCount = rowsPerCase[Case];
	if (rowCount == 0)
		return;
	dim3 blockDim(BlockSize, 1, 1);
	const int warps = BlockSize / 32;
	const int rowsPerBlock=warps*(32 / WarpSize);//TODO: switch to BlockSize/WarpSize
	int rowStart = cumulativeRowsPerCase[Case] - rowCount;
	DeviceVector<uint> sub = p.SubVector(rowStart, rowCount);
	dim3 gridDim(DivUp(rowCount, rowsPerBlock), 1, 1);
	cudaStream_t stream = streams[counter%streams.Length32()];	
	int sharedSize = SM;
	const int RPT = (MergingCapability + WarpSize - 1) / WarpSize;
	SpmmNSortedMulLimitedCaseWarpKernel<BlockSize, WarpSize, RPT> << < gridDim, blockDim, sharedSize, stream >> >(C.GetC(), A.GetC(), B.GetC(), sub.Pointer(), sub.Length32(), rowsPerBlock);
	counter++;
}
#else
;
#endif

//Computes C=A*B.
//The computation is split into several kernel launches (one for each case), which is determined by the permutation p in
//combination with rowsPerCase and cumulativeRowsPerCase.
//Kernels are assigned to the streams in round-robin way.
template<typename T>
void SpmmNSortedMulLimitedKepler(SparseDeviceMatrixCSR<T>& C, SparseDeviceMatrixCSR<T>& A, SparseDeviceMatrixCSR<T>& B, DeviceVector<uint>& p, HostVector<uint>& rowsPerCase, HostVector<uint>& cumulativeRowsPerCase, HostVector<cudaStream_t>& streams){
	int counter=0;
	SpmmNSortedMulLimitedCaseBlock<1, MergingCapabilities::MC_01, 1024, 12000>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseBlock<2, MergingCapabilities::MC_02, 1024, 12000>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseBlock<3, MergingCapabilities::MC_03, 1024, 12000>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseBlock<4, MergingCapabilities::MC_04, 512, 12000>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseBlock<5, MergingCapabilities::MC_05, 512, 12000>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseBlock<6, MergingCapabilities::MC_06, 256, 12000>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseWarp<7, MergingCapabilities::MC_07, 256, 32, 12000>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseWarp<8, MergingCapabilities::MC_08, 256, 32, 12000>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseWarp<9, MergingCapabilities::MC_09, 256, 32, 12000>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseWarp<10, MergingCapabilities::MC_10, 256, 32, 12000>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseWarp<11, MergingCapabilities::MC_11, 256, 32, 12000>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseWarp<12, MergingCapabilities::MC_12, 256, 16, 12000>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseWarp<13, MergingCapabilities::MC_13, 256, 16, 12000>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseWarp<14, MergingCapabilities::MC_14, 256, 16, 12000>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseWarp<15, MergingCapabilities::MC_15, 256, 16, 12000>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseWarp<16, MergingCapabilities::MC_16, 256, 8, 12000>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseWarp<17, MergingCapabilities::MC_17, 256, 8, 12000>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseWarp<18, MergingCapabilities::MC_18, 256, 8, 12000>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseWarp<19, MergingCapabilities::MC_19, 256, 8, 0>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseWarp<20, MergingCapabilities::MC_20, 256, 4, 0>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseWarp<21, MergingCapabilities::MC_21, 256, 2, 0>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	ScaleRows<22, 256, 16>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	//Case 23 does nothing
}

//Computes C=A*B.
//The computation is split into several kernel launches (one for each case), which is determined by the permutation p in
//combination with rowsPerCase and cumulativeRowsPerCase.
//Kernels are assigned to the streams in round-robin way.
template<typename T>
void SpmmNSortedMulLimitedPascal(SparseDeviceMatrixCSR<T>& C, SparseDeviceMatrixCSR<T>& A, SparseDeviceMatrixCSR<T>& B, DeviceVector<uint>& p, HostVector<uint>& rowsPerCase, HostVector<uint>& cumulativeRowsPerCase, HostVector<cudaStream_t>& streams) {
	int counter = 0;
	static const int BSM=20000;

	static const int WSM = 20000;
	static const int WBS = 256;
	SpmmNSortedMulLimitedCaseBlock<1, MergingCapabilities::MC_01, 1024, BSM>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseBlock<2, MergingCapabilities::MC_02, 1024, BSM>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseBlock<3, MergingCapabilities::MC_03, 1024, BSM>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseBlock<4, MergingCapabilities::MC_04, 512, BSM>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseBlock<5, MergingCapabilities::MC_05, 512, BSM>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseBlock<6, MergingCapabilities::MC_06, 256, BSM>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);

	SpmmNSortedMulLimitedCaseWarp<7, MergingCapabilities::MC_07, WBS, 32, WSM>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseWarp<8, MergingCapabilities::MC_08, WBS, 32, WSM>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseWarp<9, MergingCapabilities::MC_09, WBS, 32, WSM>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseWarp<10, MergingCapabilities::MC_10, WBS, 32, WSM>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseWarp<11, MergingCapabilities::MC_11, WBS, 32, WSM>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseWarp<12, MergingCapabilities::MC_12, WBS, 32, WSM>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseWarp<13, MergingCapabilities::MC_13, WBS, 32, WSM>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseWarp<14, MergingCapabilities::MC_14, WBS, 16, WSM>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseWarp<15, MergingCapabilities::MC_15, WBS, 32, WSM>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseWarp<16, MergingCapabilities::MC_16, WBS, 8, WSM>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseWarp<17, MergingCapabilities::MC_17, WBS, 16, WSM>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseWarp<18, MergingCapabilities::MC_18, WBS, 8, WSM>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseWarp<19, MergingCapabilities::MC_19, WBS, 8, 0>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseWarp<20, MergingCapabilities::MC_20, WBS, 4, 0>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmNSortedMulLimitedCaseWarp<21, MergingCapabilities::MC_21, WBS, 2, 0>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	ScaleRows<22, 256, 16>(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	//Case 23 does nothing
}

//Compute C=A*B with preallocated C. 
//Depending on the architecture, different versions are called.
template<typename T>
void SpmmNSortedMulLimited(SparseDeviceMatrixCSR<T>& C, SparseDeviceMatrixCSR<T>& A, SparseDeviceMatrixCSR<T>& B, DeviceVector<uint>& p, HostVector<uint>& rowsPerCase, HostVector<uint>& cumulativeRowsPerCase, HostVector<cudaStream_t>& streams, int architecture) {
	if(architecture<=3)
		SpmmNSortedMulLimitedKepler(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams);	
	else
		SpmmNSortedMulLimitedPascal(C, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams);
}


#ifdef __CUDACC__
//Computes the exact output size of c=a*B (by each warp). 
//Turns a block into subwarps. 
//Then predicts the size.
//p has length of BlockSize
template<int SubWarpSize, int RowsPerThread,int BlockSize, typename T>
__device__ void PredictSizesWarps(uint* dstLengths, CSparseMatrixCSR<T> A, CSparseMatrixCSR<T> B, uint* p, int n, int start, int thread){
	//Each subwarp computes one row
	int subWarp=thread/SubWarpSize;
	int threadx=thread%SubWarpSize;
	if(start+subWarp>=n)
		return;
	int r=p[start+subWarp];
	uint dst;	
	if(RowsPerThread==1)//TODO: Check if removing hurts
		dst=MulWarpPredictSize<SubWarpSize,false>(A.GetRow(r),B,threadx,A.RowStart(r));
	else
		dst=MulWarpPredictSizeN<SubWarpSize,RowsPerThread,false>(A.GetRow(r),B,threadx,A.RowStart(r));
	if(threadx==0)
		dstLengths[r]=dst;
}
#endif

#ifdef __CUDACC__
//Predicts the size of C=A*B. Only for a subset of n rows indicated by the vector p.
template<int BlockSize, int RPT, typename T>
__global__ void __cdecl SpmmWarpPredictSizeNSortedCaseBlockKernel(uint* dstLengths, CSparseMatrixCSR<T> A, CSparseMatrixCSR<T> B, uint* p, int n, int rowsPerBlock) {
	int start = blockIdx.x*rowsPerBlock;//the start for a single thread block
	const int WarpSize = 32;
	//Some small amount of shared memory is needed for communication
	__shared__ uint s[WarpSize + 1];
	int r = p[start];
	const int WarpCount = BlockSize / WarpSize;
	MulWarpPredictSizeMany<WarpSize, WarpCount, RPT>(dstLengths[r], A.GetRow(r), B, int(threadIdx.x % WarpSize), int(threadIdx.x / WarpSize), s);
}
#endif

#ifdef __CUDACC__
//Predicts the size of C=A*B. Only for a subset of n rows indicated by the vector p.
template<int BlockSize, int WarpSize, int RPT, typename T>
__global__ void __cdecl SpmmWarpPredictSizeNSortedCaseWarpKernel(uint* dstLengths, CSparseMatrixCSR<T> A, CSparseMatrixCSR<T> B, uint* p, int n, int rowsPerBlock) {
	int start = blockIdx.x*rowsPerBlock;//the start for a single thread block
		
	PredictSizesWarps<WarpSize,RPT,BlockSize>(dstLengths, A, B, p, n, start, threadIdx.x);
}
#endif

//Computes output size of C=A*B for a given case with fixed MergingCapability, BlockSize and shared memory.
//This is the block-based version.
template<int Case, int MergingCapability, int BlockSize, int SM, typename T>
void __cdecl SpmmWarpPredictSizeNSortedCaseBlock(DeviceVector<uint>& dstLengths, SparseDeviceMatrixCSR<T>& A, SparseDeviceMatrixCSR<T>& B, DeviceVector<uint>& p, HostVector<uint>& rowsPerCase, HostVector<uint>& cumulativeRowsPerCase, HostVector<cudaStream_t>& streams, int& counter)
#ifdef __CUDACC__
{
	int rowCount = rowsPerCase[Case];
	if (rowCount == 0)
		return;
	dim3 blockDim(BlockSize, 1, 1);
	int rowsPerBlock = 1;
	int rowStart = cumulativeRowsPerCase[Case] - rowCount;
	DeviceVector<uint> sub = p.SubVector(rowStart, rowCount);
	dim3 gridDim(rowCount, 1, 1);
	cudaStream_t stream = streams[counter%streams.Length32()];
	int sharedSize = SM;
	const int RPT = MergingCapability / BlockSize;
	SpmmWarpPredictSizeNSortedCaseBlockKernel<BlockSize, RPT> << < gridDim, blockDim, sharedSize, stream >> >(dstLengths.Data(), A.GetC(), B.GetC(), sub.Pointer(), sub.Length32(), rowsPerBlock);
	counter++;
}
#else
;
#endif

//Compute row lengths of C=A*B but only for a subset of rows indicated by permutation p ( subset defined by rowsPerCase and cumulativeRowsPerCase).  
//One GPU kernel is started in streams[counter] to compute this subset. Then counter is increased.
template<int Case, int MergingCapability, int BlockSize, int WarpSize, int SM, typename T>
void __cdecl SpmmWarpPredictSizeNSortedCaseWarp(DeviceVector<uint>& dstLengths, SparseDeviceMatrixCSR<T>& A, SparseDeviceMatrixCSR<T>& B, DeviceVector<uint>& p, HostVector<uint>& rowsPerCase, HostVector<uint>& cumulativeRowsPerCase, HostVector<cudaStream_t>& streams, int& counter)
#ifdef __CUDACC__
{
	int rowCount = rowsPerCase[Case];
	if (rowCount == 0)
		return;
	dim3 blockDim(BlockSize, 1, 1);
	const int warps = BlockSize / 32;
	const int rowsPerBlock = warps*(32 / WarpSize);//TODO: switch to BlockSize/WarpSize	
	int rowStart = cumulativeRowsPerCase[Case] - rowCount;
	DeviceVector<uint> sub = p.SubVector(rowStart, rowCount);
	dim3 gridDim(DivUp(rowCount, rowsPerBlock), 1, 1);
	cudaStream_t stream = streams[counter%streams.Length32()];
	int sharedSize = SM;
	const int RPT = (MergingCapability + WarpSize - 1) / WarpSize;
	SpmmWarpPredictSizeNSortedCaseWarpKernel<BlockSize, WarpSize, RPT> << < gridDim, blockDim, sharedSize, stream >> >(dstLengths.Data(), A.GetC(), B.GetC(), sub.Pointer(), sub.Length32(), rowsPerBlock);
	counter++;
}
#else
;
#endif

//Computes row lengths of C=A*B.
//One kernel is called for each case.
//Subsets of rows are indicated by permutation p, rowsPerCase and cumulativeRowsPerCase
//Kernels are assigned to streams in round-robin mode.
template<typename T>
static void SpmmWarpPredictSizeNSortedKepler(DeviceVector<uint>& dstLengths, SparseDeviceMatrixCSR<T>& A, SparseDeviceMatrixCSR<T>& B, DeviceVector<uint>& p, HostVector<uint>& rowsPerCase, HostVector<uint>& cumulativeRowsPerCase, HostVector<cudaStream_t>& streams) {
	int counter = 0;
	SpmmWarpPredictSizeNSortedCaseBlock<1, MergingCapabilities::MC_01, 1024, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseBlock<2, MergingCapabilities::MC_02, 1024, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseBlock<3, MergingCapabilities::MC_03, 1024, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseBlock<4, MergingCapabilities::MC_04, 512, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseBlock<5, MergingCapabilities::MC_05, 512, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseBlock<6, MergingCapabilities::MC_06, 256, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);

	SpmmWarpPredictSizeNSortedCaseWarp< 7, MergingCapabilities::MC_07, 256, 32, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseWarp< 8, MergingCapabilities::MC_08, 256, 32, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseWarp< 9, MergingCapabilities::MC_09, 256, 32, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseWarp<10, MergingCapabilities::MC_10, 256, 32, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseWarp<11, MergingCapabilities::MC_11, 256, 32, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseWarp<12, MergingCapabilities::MC_12, 256, 16, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseWarp<13, MergingCapabilities::MC_13, 256, 16, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseWarp<14, MergingCapabilities::MC_14, 256, 16, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseWarp<15, MergingCapabilities::MC_15, 256, 16, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseWarp<16, MergingCapabilities::MC_16, 256, 8, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseWarp<17, MergingCapabilities::MC_17, 256, 8, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseWarp<18, MergingCapabilities::MC_18, 256, 8, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseWarp<19, MergingCapabilities::MC_19, 256, 8, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseWarp<20, MergingCapabilities::MC_20, 256, 4, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseWarp<21, MergingCapabilities::MC_21, 256, 2, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseWarp<22, MergingCapabilities::MC_22, 256, 16, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseWarp<23, 1, 256, 1, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);	
}

//Computes row lengths of C=A*B.
//One kernel is called for each case.
//Subsets of rows are indicated by permutation p, rowsPerCase and cumulativeRowsPerCase
//Kernels are assigned to streams in round-robin mode.
template<typename T>
static void SpmmWarpPredictSizeNSortedPascal(DeviceVector<uint>& dstLengths, SparseDeviceMatrixCSR<T>& A, SparseDeviceMatrixCSR<T>& B, DeviceVector<uint>& p, HostVector<uint>& rowsPerCase, HostVector<uint>& cumulativeRowsPerCase, HostVector<cudaStream_t>& streams) {
	int counter = 0;
	SpmmWarpPredictSizeNSortedCaseBlock<1, MergingCapabilities::MC_01, 1024, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseBlock<2, MergingCapabilities::MC_02, 1024, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseBlock<3, MergingCapabilities::MC_03, 1024, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseBlock<4, MergingCapabilities::MC_04, 512, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseBlock<5, MergingCapabilities::MC_05, 512, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseBlock<6, MergingCapabilities::MC_06, 256, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseWarp< 7, MergingCapabilities::MC_07, 256, 32, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseWarp< 8, MergingCapabilities::MC_08, 256, 32, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseWarp< 9, MergingCapabilities::MC_09, 256, 32, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseWarp<10, MergingCapabilities::MC_10, 256, 32, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseWarp<11, MergingCapabilities::MC_11, 256, 32, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseWarp<12, MergingCapabilities::MC_12, 256, 32, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseWarp<13, MergingCapabilities::MC_13, 256, 32, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseWarp<14, MergingCapabilities::MC_14, 256, 16, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseWarp<15, MergingCapabilities::MC_15, 256, 32, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseWarp<16, MergingCapabilities::MC_16, 256, 8, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseWarp<17, MergingCapabilities::MC_17, 256, 16, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseWarp<18, MergingCapabilities::MC_18, 256, 8, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseWarp<19, MergingCapabilities::MC_19, 256, 8, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseWarp<20, MergingCapabilities::MC_20, 256, 4, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseWarp<21, MergingCapabilities::MC_21, 256, 2, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseWarp<22, MergingCapabilities::MC_22, 256, 16, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
	SpmmWarpPredictSizeNSortedCaseWarp<23, 1, 256, 1, 0>(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams, counter);
}

//Computes the output size of C=A*B.
//Different versions are called depending on the architecture parameter.
template<typename T>
static void SpmmWarpPredictSizeNSorted(DeviceVector<uint>& dstLengths, SparseDeviceMatrixCSR<T>& A, SparseDeviceMatrixCSR<T>& B, DeviceVector<uint>& p, HostVector<uint>& rowsPerCase, HostVector<uint>& cumulativeRowsPerCase, HostVector<cudaStream_t>& streams,int architecture){	
	if(architecture<=3)
		SpmmWarpPredictSizeNSortedKepler(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams);	
	else
		SpmmWarpPredictSizeNSortedPascal(dstLengths, A, B, p, rowsPerCase, cumulativeRowsPerCase, streams);
}

//Computes the output size of C=A*B.
//Maximum row nnz of A must be 32k. 
template<typename T>
static SparseDeviceMatrixCSR<T> MulLimitedPrepareNSorted(DeviceVector<uint>& rowStarts, SparseDeviceMatrixCSR<T>& A, SparseDeviceMatrixCSR<T>& B, DeviceVector<uint>& p, HostVector<uint>& rowsPerCase, HostVector<uint>& cumulativeRowsPerCase, HostVector<cudaStream_t>& streams, int architecture){		
	DeviceVector<uint> sub=rowStarts.SubVector(0,A.Height());
	SpmmWarpPredictSizeNSorted(sub,A,B,p,rowsPerCase,cumulativeRowsPerCase,streams,architecture);
	ScanExclusive(rowStarts);
	uint nonZeros=rowStarts[rowStarts.Length()-1];
	SparseDeviceMatrixCSR<T> dst(B.Width(),A.Height(),DeviceVector<T>(nonZeros),DeviceVector<uint>(nonZeros),rowStarts);
	return dst;
}

//Computes C=A*B
//Maximum row nnz of A must be 32k, hence the term "Limited"
template<typename T>
static SparseDeviceMatrixCSR<T> MulLimitedNSorted(SparseDeviceMatrixCSR<T>& A, SparseDeviceMatrixCSR<T>& B, DeviceVector<uint>& p, HostVector<uint>& rowsPerCase, HostVector<uint>& cumulativeRowsPerCase,HostVector<cudaStream_t>& streams, int architecture){
	SparseDeviceMatrixCSR<T> dst=MulLimitedPrepareNSorted(A,B,p,rowsPerCase,cumulativeRowsPerCase,streams, architecture);
	SpmmNSortedMulLimited(dst,A,B,p,rowsPerCase,cumulativeRowsPerCase,streams, architecture);
	return dst;
}

//Calls the function RowLengthToCase for each row of A, i.e. computes the cases.
#ifdef __CUDACC__
template<int BlockSize, typename T>
__global__ void __cdecl ComputeCasesKernel(CVector<uchar> cases, CSparseMatrixCSR<T> A){	
	int r=blockIdx.x*BlockSize+threadIdx.x;
	if(r>=cases.Length())
		return;
	cases[r]=RowLengthToCase(A.RowLength(r));
}
#endif

//Calls the function RowLengthToCase for each row of A, i.e. computes the cases.
template<typename T>
void __cdecl ComputeCases(DeviceVector<uchar>& cases, SparseDeviceMatrixCSR<T>& A)
#ifdef __CUDACC__
{	
	const int BlockSize=256;
	dim3 blockDim(BlockSize,1,1);	
	dim3 gridDim(DivUp(cases.Length32(),BlockSize),1,1);
	ComputeCasesKernel<BlockSize> <<< gridDim,blockDim,0>>>(cases.GetC(),A.GetC());
}
#else
;
#endif

//Preparations for RMerge2
//Computes the cases, sorts them and computes how many rows per case.
template<typename T>
static void SpmmRMergeNSortedCases(DeviceVector<uint>& permutation, HostVector<uint>& rowsPerCase, HostVector<uint>& cumulativeRowsPerCase, SparseDeviceMatrixCSR<T>& A){
	DeviceVector<uchar> cases(A.Height());
	Range(permutation);
	ComputeCases(cases,A);
	StableSortByKey(permutation,cases);
	RowsPerCase(rowsPerCase,cumulativeRowsPerCase,CaseCount(),cases);
}

template<typename T>
static SparseDeviceMatrixCSR<T> SpmmRMergeNSortedPrepare(SparseDeviceMatrixCSR<T>& A, SparseDeviceMatrixCSR<T>& B, DeviceVector<int>& permutation, HostVector<int>& rowsPerCase, HostVector<int>& cumulativeRowsPerCase, HostVector<cudaStream_t>& streams, int architecture){
	SpmmRMergeNSortedCases(permutation,rowsPerCase,cumulativeRowsPerCase,A);
	SparseDeviceMatrixCSR<T> dst=MulLimitedPrepareNSorted(A,B,permutation,rowsPerCase,cumulativeRowsPerCase,streams,architecture);
	return dst;
}

static int StreamCount(){return 6;}

//Creates n streams. If n=1 the stream is set to the default stream
static HostVector<cudaStream_t> CreateStreams(int streamCount=StreamCount()){
	HostVector<cudaStream_t> streams(streamCount);
	if(streamCount==1){
		streams[0]=0;//Default Stream
	}
	else{
		for (int i = 0; i < streamCount; ++i)
			cudaStreamCreate(&streams[i]);
	}
	return streams;
}

//Destroys the streams created by CreateStreams.
static void DestroyStreams(HostVector<cudaStream_t> streams){
	if(streams.Length()==1)
		return;//Default stream does not need to be destroyed
	for (int i = 0; i < streams.Length32(); ++i)
		cudaStreamDestroy(streams[i]);
}

template<typename T>
static SparseDeviceMatrixCSR<T> MulRMerge2(SparseDeviceMatrixCSR<T> A, SparseDeviceMatrixCSR<T> B, HostVector<cudaStream_t> streams, bool prepareOnly=false, int architecture = 6){
	HostVector<uint> rowsPerCase(CaseCount());
	HostVector<uint> cumulativeRowsPerCase(CaseCount());
	DeviceVector<uint> permutation(A.Height());
	DeviceVector<uint> rowStarts(A.Height()+1);
	SpmmRMergeNSortedCases(permutation,rowsPerCase,cumulativeRowsPerCase,A);
	if(rowsPerCase[0]>0)
		return MulRMerge11(A,B);		
	SparseDeviceMatrixCSR<T> dst=MulLimitedPrepareNSorted(rowStarts, A, B, permutation, rowsPerCase, cumulativeRowsPerCase, streams, architecture);	
	if (!prepareOnly)
		SpmmNSortedMulLimited(dst, A, B, permutation, rowsPerCase, cumulativeRowsPerCase, streams, architecture);
	return dst;
}

template<typename T>
static SparseDeviceMatrixCSR<T> MulRMerge2(SparseDeviceMatrixCSR<T> A, SparseDeviceMatrixCSR<T> B, int streamCount=StreamCount(), bool prepareOnly=false, int architecture = 6){
	HostVector<cudaStream_t> streams=CreateStreams(streamCount);
	SparseDeviceMatrixCSR<T> dst=MulRMerge2(A,B,streams,prepareOnly, architecture);
	DestroyStreams(streams);
	return dst;
}