#pragma once

//This file contains code to decompose matrices for sparse matrix-matrix multiplication
//using iterative row merging as in RMerge1

#include "DeviceMatrix/SparseDeviceMatrixCSR.h"
#include "DeviceMatrix/Scan.h"
#include "DeviceMatrix/DeviceReductions.h"
#include "HostMatrix/Intrinsics.h"
#include "HostMatrix/MinMaxValues.h"
#include "HostMatrix/IO.h"
#include "DeviceMatrix/ldg.h"
#include "DeviceMatrix/WarpReduction.h"
#include "DeviceMatrix/SparseDeviceMatrixCSROperations.h"


#ifdef __CUDACC__
//Computes: B.RowStarts()
//Requires: A.RowStarts(),A.Values(), B.Values(),B.ColIndices(),M
template<int WarpSize, int ThreadCount, typename T>
__global__ void SpmmWarpFillDecompositionKernel(CSparseMatrixCSR<T> A, CSparseMatrixCSR<T> B, CSparseMatrixCSR<T> M){
	//Each thread splits a row of M into several rows of B.
	uint r=blockIdx.x*ThreadCount+threadIdx.x;
	if(r>=M.Height())
		return;
	CSparseVector<T> mRow=M.GetRow(r);
	uint parts=A.RowLength(r);
	uint* rowStartsA=A.RowStarts();
	uint* rowStartsB=B.RowStarts();	
	uint startM=M.RowStart(r);//This is the start of the first row in B.
	uint rowStartA=rowStartsA[r];
	for(int i=0;i<parts;i++)
		rowStartsB[rowStartA+i]=startM+i*WarpSize;
}
#endif

//A*B==M
template<int WarpSize, typename T>
void __cdecl SpmmWarpFillDecomposition(CSparseMatrixCSR<T> A, CSparseMatrixCSR<T> B, CSparseMatrixCSR<T> M)
#ifdef __CUDACC__
{
	const int ThreadCount=512;
	int blocks=DivUp(M.Height(),ThreadCount);
	SpmmWarpFillDecompositionKernel<WarpSize,ThreadCount> <<<dim3(blocks,1,1),dim3(ThreadCount,1,1)>>>(A,B,M);
}
#else
;
#endif

//Compute the row lengths of A
template<typename T>
static void ComputeRowLengths(DeviceVector<uint> rowLengths, SparseDeviceMatrixCSR<T> A){	
	Verify(rowLengths.Length()==A.Height(),FileAndLine);	
	DeviceVector<uint> rowStarts=A.RowStarts();	
	BinaryComponentWise(rowLengths,rowStarts.SubVector(1,A.Height()),rowStarts.SubVector(0,A.Height()),BinaryFunctors::Subtract());	
}

//Compute the smallest possible merge factor
static int SufficientSubWarpSize(int maxRowLength){
	if(maxRowLength<=2)
		return 2;
	if(maxRowLength<=4)
		return 4;
	if(maxRowLength<=8)
		return 8;
	if(maxRowLength<=16)
		return 16;
	if(maxRowLength<=32)
		return 32;
	if(maxRowLength<=64)
		return 64;
	if(maxRowLength<=128)
		return 128;
	if(maxRowLength<=256)
		return 256;
	throw std::runtime_error("Merge factor too high");
}


//Decomposes M into M=A*B where B has up to WarpSize elements per row.
//The interpretation is that B pre-merges up to WarpSize rows of M into one row, whereas A merges these.
//The use case for this function is sparse matrix matrix mul of M with any other matrix R. 
//Assume M is factorized in A*B*C*D. Then M*R is A*B*C*D*R and can be computed as A*(B*(C*(D*R))).
//Data of M is reused.
template<typename T>
static void SpmmWarpDecompose(SparseDeviceMatrixCSR<T>& A, SparseDeviceMatrixCSR<T>& B, SparseDeviceMatrixCSR<T> M, int mergeFactor){
	DeviceVector<uint> rowStartsA(M.Height()+1);
	DeviceVector<uint> rowLengthsM=rowStartsA.SubVector(0,M.Height());
	ComputeRowLengths(rowLengthsM,M);
	//uint nonZerosM=(uint)M.NonZeroCount();
	uint nonZerosM=M.RowStarts()[M.Height()];
	//Each row r of M creates DivUp(r.NonzeroCount(),WarpSize) rows with WarpSize elements each (except the last one), together having r.Length() elements.	
	ComponentWiseInline(rowLengthsM,ElementFunctors::DivUpByConst<uint>(mergeFactor));
	ScanExclusive(rowStartsA);
	uint nonZerosA=rowStartsA[rowStartsA.Length()-1];
	//B has the same number of nonzeros as M, same width, but larger height. The values of B are the same as M and can be reused.	
	DeviceVector<uint> rowStartsB(nonZerosA+1);
	//We can reuse the memory of M here, i.e. values and colIndices.
	B=SparseDeviceMatrixCSR<T>(M.Width(),(int)nonZerosA,M.Values(),M.ColIndices(),rowStartsB);	
	DeviceVector<T> aValues;//Allocate empty array for the values.
	DeviceVector<uint> aColIndices; //allocate empty array for indices
	A=SparseDeviceMatrixCSR<T>((int)nonZerosA,M.Height(),aValues,aColIndices,rowStartsA);	
	if(mergeFactor==2)
		SpmmWarpFillDecomposition<2>(A.GetC(),B.GetC(),M.GetC());
	else if(mergeFactor==4)
		SpmmWarpFillDecomposition<4>(A.GetC(),B.GetC(),M.GetC());
	else if(mergeFactor==8)
		SpmmWarpFillDecomposition<8>(A.GetC(),B.GetC(),M.GetC());
	else if(mergeFactor==16)
		SpmmWarpFillDecomposition<16>(A.GetC(),B.GetC(),M.GetC());
	else if(mergeFactor==32)
		SpmmWarpFillDecomposition<32>(A.GetC(),B.GetC(),M.GetC());
	else if(mergeFactor==64)
		SpmmWarpFillDecomposition<64>(A.GetC(),B.GetC(),M.GetC());
	else if(mergeFactor==128)
		SpmmWarpFillDecomposition<128>(A.GetC(),B.GetC(),M.GetC());
	else if(mergeFactor==256)
		SpmmWarpFillDecomposition<256>(A.GetC(),B.GetC(),M.GetC());
	else
		throw std::runtime_error("Merge factor too high");
	B.RowStarts().Set(nonZerosA,nonZerosM);//the last value
}
