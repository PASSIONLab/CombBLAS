#pragma once

//This file contains code for block-based row merging (in the context of sparse matrix-matrix multiplication). 
//Many rows (e.g. 32k) can be merged because each thread maintains more than one row of the right-hand-side.

#include "DeviceMatrix/ldg.h"
#include "HostMatrix/CSparseMatrixCSR.h"
#include "HostMatrix/CSparseVector.h"


//Multiply sparse vector with sparse matrix using a block of threads.
//Merge up to BlockSize*RowsPerThread rows. 
//Each thread pulls from RowsPerThread rows. 
//Each 32-warp merges 32 rows. Then the first warp does the final merge and stores into dst.
//a must have at most BlockSize*RowsPerThread elements
//Result (dst) must be pre-allocated.
//Requires shared memory of size WarpSize*sizeof(T)+WarpSize*sizeof(uint)+sizeof(uint)
template<uint WarpSize, uint WarpCount, uint RowsPerThread, typename T>
static __device__ void MulWarpMany(CSparseVector<T>& dst, const CSparseVector<T>& a, const CSparseMatrixCSR<T>& B,uint threadx,uint warp, byte *s){
	uint nnz=a.NonZeroCount();
	const uint BlockSize=WarpSize*WarpCount;
	uint thread=warp*WarpSize+threadx;
	if(nnz==0)//nothing to do
		return;
	else if(nnz==1){//simply scale the vector (faster)
		T weight=a.Value(0);
		const T* rowValues;const uint *rowIndices;int rowLength;
		uint r=a.Index(0);
		B.GetRow(r,rowValues,rowIndices,rowLength);
		for(uint i=thread;i<dst.NonZeroCount();i+=BlockSize){
			dst.Index(i)=rowIndices[i];
			dst.Value(i)=weight*rowValues[i];
		}
		return;
	}
	const uint intMax=2147483647;//used to signal that a row is finished		

	//The rows for the thread	
	uint rowStart[RowsPerThread];
	uint rowEnd[RowsPerThread];
	uint frontIndex[RowsPerThread];//The front index of the row. intMax means that the row ended.		

	const uint* pBC=B.ColIndices();
	const T* pBV=B.Values();

	//int rowsPerThread=DivUp(nnz,BlockSize);
	
	//Load the rows of the threads
	for(uint i=0;i<RowsPerThread;i++){
		if(thread+i*BlockSize<nnz){
			uint r=a.Index(thread+i*BlockSize);
			rowStart[i]=B.RowStart(r);
			rowEnd[i]=B.RowStart(r+1);
		}
		else{
			rowStart[i]=0;			
			rowEnd[i]=0;			
		}
		frontIndex[i]=intMax;
		
		if(rowStart[i]<rowEnd[i])//Load the front index and value of each row
			frontIndex[i]=ldg(pBC+rowStart[i]);//ldg: explicit cache usage		
		else
			frontIndex[i]=intMax;

	}
	uint minFrontThread=MinThread<RowsPerThread>(frontIndex);
	uint minFrontWarp=WarpMin<WarpSize>(minFrontThread);
	
	//These arrays are used to communicate between all warps and the first warp	
	T* warpSum=(T*)(&s[0]);//To gather the warpsums
	uint* warpMin=(uint*)(&s[0]+WarpSize*sizeof(T));//To gather the warpmins	
	uint* pTotalMin=(uint*)(&s[0]+WarpSize*sizeof(T)+WarpSize*sizeof(uint));//To distribute the total min to the warps

	if(warp==0){//Initialize because we may have fewer warps than 32
		warpMin[threadx]=intMax;
		warpSum[threadx]=T(0);
	}
	__syncthreads();

	if(threadx==0)
		warpMin[warp]=minFrontWarp;
	__syncthreads();
	if(warp==0)
		pTotalMin[0]=WarpMin<WarpSize>(warpMin[threadx]);
	__syncthreads();
	int minFront=pTotalMin[0];
	__syncthreads();
	int dstPos=0;
	//Results are stored into a "buffer" of registers.
	//When WarpSize results are available, the buffer is saved to global mem (coalesced)
	uint bufferedIndex;//Thread i stores result i in its register
	T bufferedValue;
	uint bufferPos=0;//how many elements are currently in the buffer
	while(minFront!=intMax){//Compute one element per iteration		

		//Only update some warps
		if(minFront==minFrontWarp){
			T tmp=0.0;//Used to compute the value of the thread
			for(int i=0;i<RowsPerThread;i++){
				if(frontIndex[i]==minFront){//put these into tmp and load next elements					
					//load next					
					T frontValue=ldg(pBV+rowStart[i]);//ldg: explicit cache usage
					T w=a.Value(thread+i*BlockSize);
					frontValue*=w;
					tmp+=frontValue;
					rowStart[i]++;
					if(rowStart[i]<rowEnd[i])
						frontIndex[i]=ldg(pBC+rowStart[i]);//ldg: explicit cache usage					
					else//out of the game
						frontIndex[i]=intMax;
				}
			}
			minFrontThread=MinThread<RowsPerThread>(frontIndex);
			minFrontWarp=WarpMin<WarpSize>(minFrontThread);
			T sumOfWarp=WarpSum<WarpSize>(tmp);
			if(threadx==0){
				warpMin[warp]=minFrontWarp;	
				warpSum[warp]=sumOfWarp;
			}
		}
		__syncthreads();

		if(warp==0){
			pTotalMin[0]=WarpMin<WarpSize>(warpMin[threadx]);

			T sum=WarpSum<WarpSize>(warpSum[threadx]);
			warpSum[threadx]=T(0);						
			
			if(threadx==bufferPos){//Save into buffer
				bufferedIndex=(uint)minFront;
				bufferedValue=sum;
			}
			bufferPos++;
			if(bufferPos==WarpSize){//Save buffer to global memory (coalesced)
				dst.Indices()[dstPos+threadx]=bufferedIndex;
				dst.Values()[dstPos+threadx]=bufferedValue;
				dstPos+=WarpSize;
				bufferPos=0;
			}
		}		
		__syncthreads();
		minFront=pTotalMin[0];		
	}

	if(thread<bufferPos){//Flush buffer
		dst.Indices()[dstPos+thread]=bufferedIndex;
		dst.Values()[dstPos+thread]=bufferedValue;
	}
}

//Requires shared memory  warpMin uint[WarpSize+1]
//WarpSize must be 32
template<int WarpSize, int WarpCount, int RowsPerThread, typename T>
static __device__ void MulWarpPredictSizeMany(uint& dst, const CSparseVector<T>& a, const CSparseMatrixCSR<T>& B,int threadx, int warp, uint* warpMin){
	int nnz=a.NonZeroCount();
	if(nnz==0){
		if(threadx==0 && warp==0)
			dst=0;		
		return;
	}
	if(nnz==1){
		if(threadx==0 && warp==0)
			dst=B.RowLength(a.Index(0));
		return;
	}
	const int intMax=2147483647;//used to signal that a row is finished
	//The rows for the thread
	uint rowStart[RowsPerThread];
	uint rowEnd[RowsPerThread];

	int frontIndex[RowsPerThread];//The front index of the row. intMax means that the row ended.
	const int BlockSize=WarpSize*WarpCount;
	int thread=threadx+WarpSize*warp;
	const uint* pBC=B.ColIndices();
	//int rowsPerThread=DivUp(nnz,BlockSize);
	for(int i=0;i<RowsPerThread;i++){
		if(thread+i*BlockSize<nnz){
			int r=a.Index(thread+i*BlockSize);
			rowStart[i]=B.RowStart(r);
			rowEnd[i]=B.RowStart(r+1);
		}
		else{
			rowStart[i]=0;
			rowEnd[i]=0;
		}		
		frontIndex[i]=intMax;
		
		if(rowStart[i]<rowEnd[i]){//Load the front index of 
			frontIndex[i]=ldg(pBC+rowStart[i]);//ldg: explicit cache usage
			rowStart[i]++;
		}
	}

	int minFrontThread=MinThread<RowsPerThread>(frontIndex);
	int minFrontWarp=WarpMin<WarpSize>(minFrontThread);
		
	uint* pTotalMin=&warpMin[0]+WarpSize;//to tell the total min to the warps
	if(warp==0)
		warpMin[threadx]=intMax;
	__syncthreads();
	if(threadx==0)
		warpMin[warp]=minFrontWarp;
	__syncthreads();
	if(warp==0)
		pTotalMin[0]=WarpMin<WarpSize>(warpMin[threadx]);	
	__syncthreads();
	int minFront=pTotalMin[0];

	int dstPos=0;
	while(minFront!=intMax){//Compute one element per iteration		
		//Only update the warp if necessary
		if(minFrontWarp==minFront){
			for(int i=0;i<RowsPerThread;i++){
				if(frontIndex[i]==minFront){
					//load next
					if(rowStart[i]<rowEnd[i]){
						frontIndex[i]=(int)ldg(pBC+rowStart[i]);//ldg: explicit cache usage
						rowStart[i]++;
					}
					else{//out of the game
						frontIndex[i]=intMax;
					}
				}
			}
			minFrontThread=MinThread<RowsPerThread>(frontIndex);
			minFrontWarp=WarpMin<WarpSize>(minFrontThread);
			if(threadx==0)
				warpMin[warp]=minFrontWarp;
		}
		__syncthreads();
		if(warp==0)
			pTotalMin[0]=WarpMin<WarpSize>(warpMin[threadx]);	
		__syncthreads();
		minFront=pTotalMin[0];
		dstPos++;
	}
	if(threadx==0 && warp==0)
		dst=dstPos;
}
