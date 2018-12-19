#pragma once

#include "HostMatrix/HostVector.h"
#include "HostMatrix/SparseHostVector.h"
#include "HostMatrix/Intrinsics.h"

//terminated CSR
template<typename T>
class SparseHostMatrixCSR{	
	int width;
	int height;
	HostVector<T> values;//length is number of nonzeros
	HostVector<uint> colIndices;//length is number of nonzeros
	HostVector<uint> rowStarts;//length is height+1(terminated CSR)
public:
	SparseHostMatrixCSR():width(0),height(0){}
	SparseHostMatrixCSR(int width, int height,	HostVector<T> values, HostVector<uint> colIndices,HostVector<uint> rowStarts)
		:width(width),height(height),values(values),colIndices(colIndices),rowStarts(rowStarts){
			Verify(values.Length()==colIndices.Length(),"v87r8498n8749");
			Verify(rowStarts.Length()==height+1,"4535c635hj5");
	}

	int Width()const{return width;}
	int Height()const{return height;}
	int DimX()const{return width;}
	int DimY()const{return height;}

	int64 NonZeroCount()const{return values.Length();}

	int RowLength(int r)const{
		unsigned int rowStart=rowStarts[r];
		int rowLength=rowStarts[r+1]-rowStart;
		return rowLength;
	}

	//void GetRow(int r, T*& rowValues, unsigned int*& rowIndices, int& rowLength){
	//	unsigned int rowStart=rowStarts[r];
	//	rowLength=rowStarts[r+1]-rowStart;
	//	rowValues=values.Data()+rowStart;
	//	rowIndices=colIndices.Data()+rowStart;
	//}

	void GetRowPointer(int r, T*& rowValues, unsigned int*& rowIndices, int& rowLength){
		unsigned int rowStart=rowStarts[r];
		rowLength=rowStarts[r+1]-rowStart;
		rowValues=values.Data()+rowStart;
		rowIndices=colIndices.Data()+rowStart;
	}

	void GetRow(int r, T*& rowValues, unsigned int*& rowIndices, int& rowLength){
		unsigned int rowStart=rowStarts[r];
		rowLength=rowStarts[r+1]-rowStart;
		rowValues=values.Data()+rowStart;
		rowIndices=colIndices.Data()+rowStart;
	}

	CSparseVector<T> GetRow(int r){
		unsigned int rowStart=rowStarts[r];
		int nonZeros=rowStarts[r+1]-rowStart;
		return CSparseVector<T>(values.Data()+rowStart,colIndices.Data()+rowStart,width,nonZeros);
	}

	SparseHostVector<T> Row(int r){
		unsigned int rowStart=rowStarts[r];
		int length=rowStarts[r+1]-rowStart;
		return SparseHostVector<T>(values.SubVector(rowStart,length), colIndices.SubVector(rowStart,length),width);
	}

	HostVector<T> Values(){return values;}
	HostVector<uint> ColIndices(){return colIndices;}
	HostVector<uint> RowStarts(){return rowStarts;}
};
