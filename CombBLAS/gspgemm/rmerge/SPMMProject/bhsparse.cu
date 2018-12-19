#include "bhsparse.h"
#include "DeviceMatrix/SparseDeviceMatrixCSR.h"

//This code instantiates the bhsparse algorithm.

/*
The MIT License(MIT)

Copyright(c) 2015

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

//Sparse matrix-matrix multiplication using bhsparse
SparseDeviceMatrixCSR<double> Mul_bhsparse(SparseDeviceMatrixCSR<double> A, SparseDeviceMatrixCSR<double> B){
	bhsparse b;
	b.init();
	b.initData(A.Height(),A.Width(),B.Width(),
		A.NonZeroCount(),
		A.Values().Pointer(),(int*)A.RowStarts().Pointer(),(int*)A.ColIndices().Pointer(),
		B.NonZeroCount(),
		B.Values().Pointer(),(int*)B.RowStarts().Pointer(),(int*)B.ColIndices().Pointer());
	
	b.spgemm();	
	b.free_mem();	

	DeviceVector<double> vals(b.get_ValC(),b.get_nnzC());	
	DeviceVector<unsigned int> rowStarts((unsigned int*)b.get_RowPtrC(),(int64)A.Height()+1);
	DeviceVector<unsigned int> colIindices((unsigned int*)b.get_ColIndC(),(int64)b.get_nnzC());		
	return SparseDeviceMatrixCSR<double>(B.Width(),A.Height(),vals,colIindices,rowStarts);
}

SparseDeviceMatrixCSR<float> Mul_bhsparse(SparseDeviceMatrixCSR<float> A, SparseDeviceMatrixCSR<float> B){
	throw std::runtime_error("Not implemented");
}