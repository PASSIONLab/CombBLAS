#pragma once

#include "DeviceMatrix/SparseDeviceMatrixCSR.h"

//contains A and Transpose(A)
template<typename T>
class SparseDeviceMatrixCSRCSC{
	SparseDeviceMatrixCSR<T> A;
	SparseDeviceMatrixCSR<T> AT;

public:
	SparseDeviceMatrixCSRCSC(){}
	SparseDeviceMatrixCSRCSC(SparseDeviceMatrixCSR<T> A, SparseDeviceMatrixCSR<T> AT):A(A),AT(AT){}
	int Width()const{return A.Width();}
	int Height()const{return A.Height();}
	SparseDeviceMatrixCSR<T> GetA(){return A;}
	SparseDeviceMatrixCSR<T> GetAT(){return AT;}
};