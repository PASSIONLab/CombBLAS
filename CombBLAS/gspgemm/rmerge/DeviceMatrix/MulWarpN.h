#pragma once

//This file contains code for warp-based row merging (in the context of sparse matrix-matrix multiplication). 
//More than 32 rows can be merged because each thread maintains more than one row of the right-hand-side.

#include "DeviceMatrix/ldg.h"
#include "HostMatrix/CSparseMatrixCSR.h"
#include "HostMatrix/CSparseVector.h"
#include "DeviceMatrix/WarpReduction.h"

//Computes the min of n values. Only one thread is used
template<int n, typename T>
__device__ T MinThread(T values[n]){
	T a=values[0];
	for(int i=1;i<n;i++)
		a=Min_rmerge(a,values[i]);
	return a;
}

//Multiply sparse vector with sparse matrix using a subwarp of threads.
//Merge up to WarpSize*RowsPerThread rows. Each thread pulls from RowsPerThread rows. 
//rowA must have at most WarpSize*RowsPerThread elements
//Result (dst) must be pre-allocated.
template<uint WarpSize, uint RowsPerThread, bool AssumeOnes, typename T>
static __device__ void MulWarpN(CSparseVector<T>& dst, const CSparseVector<T>& a, const CSparseMatrixCSR<T>& B,uint thread, uint nnzBeforeRow){
	uint nnz=a.NonZeroCount();
	if(nnz==0)//nothing to do
		return;
	else if(nnz==1){//simply scale the vector (faster)
		T weight=AssumeOnes?1.0:a.Value(0);		
		const T* rowValues;const uint *rowIndices;int rowLength;
		uint r=AssumeOnes ? nnzBeforeRow : a.Index(0);
		B.GetRow(r,rowValues,rowIndices,rowLength);
		for(uint i=thread;i<dst.NonZeroCount();i+=WarpSize){
			dst.Index(i)=rowIndices[i];
			dst.Value(i)=AssumeOnes?rowValues[i]:weight*rowValues[i];
		}
		return;
	}
	const uint intMax=4294967295;//used to signal that a row is finished
	//The rows for the thread
	uint rowStart[RowsPerThread];
	uint rowEnd[RowsPerThread];
	uint frontIndex[RowsPerThread];//The front index of the row. intMax means that the row ended.	
	const uint* __restrict__ pBC=B.ColIndices();
	const T* __restrict__ pBV=B.Values();
	const uint* __restrict__ rowStartsB=B.RowStarts();
	const uint* __restrict__ a_indices=a.Indices();
	for(uint i=0;i<RowsPerThread;i++){
		uint tmpRowStart=0;
		uint tmpRowEnd=0;
		uint tmpFrontIndex=intMax;
		uint t=i*WarpSize+thread;
		if(t<nnz){
			uint r;
			if(AssumeOnes)
				r=nnzBeforeRow+t;
			else
				r=ldg(a_indices+t);
			tmpRowStart=ldg(rowStartsB+r);//B.RowStart(r);
			tmpRowEnd=ldg(rowStartsB+(r+1));//B.RowStart(r+1);
			if(tmpRowStart<tmpRowEnd)
				tmpFrontIndex=ldg(pBC+tmpRowStart);
		}
		rowStart[i]=tmpRowStart;
		rowEnd[i]=tmpRowEnd;
		frontIndex[i]=tmpFrontIndex;
	}

	uint minFront=MinThread<RowsPerThread>(frontIndex);//TODO: integrate this into the above loop
	minFront=WarpMin<WarpSize>(minFront);
	uint dstPos=0;

	//Results are stored into a "buffer" of registers.
	//When WarpSize results are available, the buffer is saved to global mem (coalesced)
	uint bufferedIndex;//Thread i stores result i in its register
	T bufferedValue;
	uint bufferPos=0;//how many elements are in the buffer
	const T* __restrict__ aValues=a.Values();
	while(minFront!=intMax){//Compute one element per iteration
		T tmp=0.0;//Used to compute the value
		for(uint i=0;i<RowsPerThread;i++){
			uint t=i*WarpSize+thread;
			if(frontIndex[i]==minFront){//put these into tmp and load next elements				
				//load value							
				T frontValue=ldg(pBV+rowStart[i]);				
				if(!AssumeOnes){
					T w=ldg(aValues+t);					
					frontValue*=w;
				}
				tmp+=frontValue;

				//Load next
				rowStart[i]++;
				if(rowStart[i]<rowEnd[i])
					frontIndex[i]=(int)ldg(pBC+rowStart[i]);//ldg: explicit cache usage
				else//out of the game
					frontIndex[i]=intMax;
			}
		}

		T sum=WarpSum<WarpSize>(tmp);

		if(thread==bufferPos){//Save into buffer
			bufferedIndex=(uint)minFront;
			bufferedValue=sum;
		}
		bufferPos++;
		if(bufferPos==WarpSize || WarpSize==1){//Save buffer to global memory (coalesced)
			dst.Indices()[dstPos+thread]=bufferedIndex;
			dst.Values()[dstPos+thread]=bufferedValue;
			dstPos+=WarpSize;
			bufferPos=0;
		}
		minFront=MinThread<RowsPerThread>(frontIndex);//TODO: integrate this into the above loop
		minFront=WarpMin<WarpSize>(minFront);
	}
	if(thread<bufferPos){//Flush buffer
		dst.Indices()[dstPos+thread]=bufferedIndex;
		dst.Values()[dstPos+thread]=bufferedValue;
		dstPos+=bufferPos;
		bufferPos=0;
	}
}

//Predict the size. Similar to the function above but more simple.
template<uint WarpSize, uint RowsPerThread, bool AssumeOnes, typename T>
static __device__ uint MulWarpPredictSizeN(const CSparseVector<T>& a, const CSparseMatrixCSR<T>& B,uint thread, uint nnzBeforeRow){
	uint nnz=a.NonZeroCount();
	if(nnz==0)
		return 0;
	if(nnz==1)
		return B.RowLength(AssumeOnes?nnzBeforeRow:a.Index(0));
	const uint intMax=4294967295;//used to signal that a row is finished
	//The rows for the thread	
	uint rowStart[RowsPerThread];
	uint rowEnd[RowsPerThread];
	uint frontIndex[RowsPerThread];//The front index of the row. intMax means that the row ended.
	const uint* pBC=B.ColIndices();
	for(uint i=0;i<RowsPerThread;i++){
		if(thread+i*WarpSize<nnz){
			uint r=AssumeOnes?(nnzBeforeRow+thread+i*WarpSize):ldg(a.Indices()+thread+i*WarpSize);
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

	uint minFrontThread=MinThread<RowsPerThread>(frontIndex);
	uint minFront=WarpMin<WarpSize>(minFrontThread);
	uint dstPos=0;

	while(minFront!=intMax){//Compute one element per iteration		
		for(uint i=0;i<RowsPerThread;i++){
			if(frontIndex[i]==minFront){
				//load next
				if(rowStart[i]<rowEnd[i]){
					frontIndex[i]=(int)ldg(pBC+rowStart[i]);
					rowStart[i]++;
				}
				else{//out of the game
					frontIndex[i]=intMax;
				}
			}
		}
		minFrontThread=MinThread<RowsPerThread>(frontIndex);
		minFront=WarpMin<WarpSize>(minFrontThread);
		dstPos++;
	}
	return dstPos;
}
