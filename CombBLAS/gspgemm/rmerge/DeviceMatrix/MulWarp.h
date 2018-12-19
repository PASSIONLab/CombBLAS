#pragma once

#include "DeviceMatrix/ldg.h"
#include "HostMatrix/CSparseMatrixCSR.h"
#include "HostMatrix/CSparseVector.h"

//Multiply sparse vector with sparse matrix using a warp of threads.
//Merge up to WarpSize rows. Each thread pulls from his row. 
//rowA must have at most WarpSize elements
//Result (dst) must be pre-allocated.
template<int WarpSize, bool AssumeOnes, typename T>
static __device__ void MulWarp(CSparseVector<T>& dst, const CSparseVector<T>& a, const CSparseMatrixCSR<T>& B,int thread, uint nnzBeforeRow){
	if(a.NonZeroCount()==0)//nothing to do
		return;
	else if(a.NonZeroCount()==1){//simply scale the vector (faster)
		T weight=AssumeOnes?1.0:a.Value(0);
		const T* rowValues;const uint *rowIndices;int rowLength;
		int r=AssumeOnes ? nnzBeforeRow : a.Index(0);
		B.GetRow(r,rowValues,rowIndices,rowLength);
		for(int i=thread;i<dst.NonZeroCount();i+=WarpSize){
			dst.Index(i)=rowIndices[i];
			dst.Value(i)=AssumeOnes?rowValues[i]:weight*rowValues[i];
		}
		return;
	}

	const int intMax=2147483647;//used to signal that a row is finished
	const T* rowValues;const uint* rowIndices;int rowLength=0;//The row for the thread
	T weight=0;//The weight for the row
	if(thread<a.NonZeroCount()){
		uint r= AssumeOnes? (nnzBeforeRow + thread) : ldg(a.Indices()+thread);//uint rowIndex=a.Index(thread);	
		uint rowStart=ldg(B.RowStarts()+r);
		rowLength=ldg(B.RowStarts()+r+1)-rowStart;
		rowValues=B.Values()+rowStart;
		rowIndices=B.ColIndices()+rowStart;
		//B.GetRow(rowIndex,rowValues,rowIndices,rowLength);
		weight=AssumeOnes?1.0:ldg(a.Values()+thread);//a.Value(thread);
	}

	int rowPos=0;//Current position into row
	int frontIndex=intMax;//The front index of the row. intMax means that the row ended.
	T frontValue(0);//the front of the row of the thread
	if(rowPos<rowLength){//Load the front index and row
		frontIndex=ldg(rowIndices+rowPos);//ldg: explicit cache usage
		frontValue=AssumeOnes?ldg(rowValues+rowPos):ldg(rowValues+rowPos)*weight;//ldg: explicit cache usage
		rowPos++;
	}

	int minFront=WarpMin<WarpSize>(frontIndex);//The smallest index
	int dstPos=0;

	//Results are stored into a "buffer" of registers.
	//When WarpSize results are available, the buffer is saved to global mem (coalesced)
	uint bufferedIndex;//Thread i stores result i in its register
	T bufferedValue;
	int bufferPos=0;//how many elements are in the buffer
	while(minFront!=intMax){//Compute one element per iteration
		T tmp=0.0;//Used to compute the value
		if(frontIndex==minFront){//put these into tmp and load next elements
			tmp=frontValue;
			//load next
			if(rowPos<rowLength){
				frontValue=AssumeOnes?ldg(rowValues+rowPos):ldg(rowValues+rowPos)*weight;//ldg: explicit cache usage
				frontIndex=(int)ldg(rowIndices+rowPos);//ldg: explicit cache usage
				rowPos++;
			}
			else//out of the game
				frontIndex=intMax;
		}
		T sum=WarpSum<WarpSize>(tmp);
		if(thread==bufferPos){//Save into buffer
			bufferedIndex=(uint)minFront;
			bufferedValue=sum;
		}
		minFront=WarpMin<WarpSize>(frontIndex);
		bufferPos++;		
		if(bufferPos==WarpSize || (minFront==intMax && thread<bufferPos)){//Save buffer to global memory (coalesced)
			dst.Indices()[dstPos+thread]=bufferedIndex;
			dst.Values()[dstPos+thread]=bufferedValue;
			dstPos+=WarpSize;
			bufferPos=0;
		}		
	}
}


//Similar to MulWarp but only computes the size.
template<int WarpSize, bool AssumeOnes, typename T>
static __device__ uint MulWarpPredictSize(const CSparseVector<T>& a, const CSparseMatrixCSR<T>& B,int thread, uint nnzBeforeRow){
	if(a.NonZeroCount()==0)
		return 0;
	if(a.NonZeroCount()==1)
		return B.RowLength(AssumeOnes ? nnzBeforeRow : a.Index(0));
	const int intMax=B.Width();	
	const T* rowValues;const uint* rowIndices;int rowLength=0;//The row for the thread	
	if(thread<a.NonZeroCount())
		B.GetRow(AssumeOnes ? (nnzBeforeRow + thread) : a.Index(thread),rowValues,rowIndices,rowLength);

	int rowPos=0;//position into row
	int frontIndex=intMax;//Means that the row ended
	if(rowPos<rowLength){
		frontIndex=ldg(rowIndices+rowPos);		
		rowPos++;
	}
	int minFront=WarpMin<WarpSize>(frontIndex);	
	int dstPos=0;

	while(minFront!=intMax){		
		if(frontIndex==minFront){			
			//load next
			if(rowPos<rowLength){				
				frontIndex=(int)ldg(rowIndices+rowPos);
				rowPos++;
			}
			else//out of the game
				frontIndex=intMax;
		}
		minFront=WarpMin<WarpSize>(frontIndex);
		dstPos++;
	}
	return dstPos;
}
