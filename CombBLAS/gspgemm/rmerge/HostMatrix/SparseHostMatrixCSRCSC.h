#pragma once

#include "HostMatrix/SparseHostMatrixCSR.h"

//contains A and Transpose(A) which may be beneficial for some operations
template<typename T>
class SparseHostMatrixCSRCSC{
	SparseHostMatrixCSR<T> A;
	SparseHostMatrixCSR<T> AT;

public:
	SparseHostMatrixCSRCSC(){}
	SparseHostMatrixCSRCSC(SparseHostMatrixCSR<T> A, SparseHostMatrixCSR<T> AT):A(A),AT(AT){}
	int Width()const{return A.Width();}
	int Height()const{return A.Height();}
	SparseHostMatrixCSR<T> GetA(){return A;}
	SparseHostMatrixCSR<T> GetAT(){return AT;}
};