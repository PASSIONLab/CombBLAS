#pragma once

#include "DeviceMatrix/DeviceVector.h"
#include "DeviceMatrix/SparseDeviceVector.h"
#include "HostMatrix/CSparseMatrixCSR.h"

//terminated CRS
template<typename T>
class SparseDeviceMatrixCSR{	
	int width;
	int height;
	DeviceVector<T> values;//length is number of nonzeros
	DeviceVector<unsigned int> colIndices;//length is number of nonzeros
	DeviceVector<unsigned int> rowStarts;//length is height+1(terminated CSR)
public:
	SparseDeviceMatrixCSR():width(0),height(0){}
	SparseDeviceMatrixCSR(int width, int height, DeviceVector<T> values,DeviceVector<unsigned int> colIndices,DeviceVector<unsigned int> rowStarts)
		:width(width),height(height),values(values),colIndices(colIndices),rowStarts(rowStarts){
			Verify(values.Length()==colIndices.Length(),"34t5t34gh56");
			Verify(rowStarts.Length()==height+1,"45v64564567");
	}

	int Width()const{return width;}
	int Height()const{return height;}
	int DimX()const{return width;}
	int DimY()const{return height;}
	Int2 Size()const{return Int2(width,height);}
	int64 NonZeroCount()const{return values.Length();/*rowStarts[height];*/}
	DeviceVector<T> Values(){return values;}
	DeviceVector<unsigned int> ColIndices(){return colIndices;}
	DeviceVector<unsigned int> RowStarts(){return rowStarts;}
	CSparseMatrixCSR<T> GetC(){return CSparseMatrixCSR<T>(width,height,values.Data(),colIndices.Data(),rowStarts.Data(),(int)NonZeroCount());}
};

