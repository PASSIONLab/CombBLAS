#pragma once

#include "DeviceMatrix/DeviceVector.h"

template<typename T>
class SparseDeviceMatrixCOO{	
	int width;
	int height;
	DeviceVector<T> values;//length is number of nonzeros
	DeviceVector<uint> colIndices;//length is number of nonzeros
	DeviceVector<uint> rowIndices;//length is number of nonzeros
public:
	SparseDeviceMatrixCOO():width(0),height(0){}
	SparseDeviceMatrixCOO(int width, int height, DeviceVector<T> values,DeviceVector<uint> colIndices,DeviceVector<uint> rowIndices)
		:width(width),height(height),values(values),colIndices(colIndices),rowIndices(rowIndices){}

	int Width()const{return width;}
	int Height()const{return height;}
	int DimX()const{return width;}
	int DimY()const{return height;}
	Int2 Size()const{return Int2(width,height);}
	int64 NonZeroCount()const{return values.Length();}
	DeviceVector<T> Values(){return values;}
	DeviceVector<uint> ColIndices(){return colIndices;}
	DeviceVector<uint> RowIndices(){return rowIndices;}
};