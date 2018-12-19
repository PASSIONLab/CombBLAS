#include "cuda_runtime_api.h"

//This file instantiates the Cuda code for Cusp and Cusparse

#define IGNORE_MKL
#define INSTANTIATE_0

// #include "MulCusp.h"
#include "SPMMProject/Conversions.h"

#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
// #include <cusp/gallery/poisson.h>
#include <cusp/io/matrix_market.h>
// #include <cusp/multiply.h>

#include <cuda_runtime.h>
#include <cusparse_v2.h>

typedef unsigned int uint;

/*
template<typename T>
SparseHostMatrixCSR<T> LoadMatrixCusp(std::string file){
	cusp::csr_matrix<uint,T,cusp::host_memory> InputMatrix;
	try { cusp::io::read_matrix_market_file(InputMatrix, file); } catch(std::exception ex){std::cout<<"Couldn't read file "<<ex.what();};
	return ToSparseHostMatrixCSR(InputMatrix);
}

template<typename T>
void Copy(DeviceVector<T>& x, cusp::array1d<T, cusp::device_memory>& y){
	Verify(x.Length()==y.size(),FileAndLine);
	cudaMemcpy(x.Data(),y.data().get(),sizeof(T)*x.Length(),cudaMemcpyDeviceToDevice);
}

template<typename T>
void Copy(cusp::array1d<T, cusp::device_memory>& x, DeviceVector<T>& y){
	Verify(y.Length()==x.size(),FileAndLine);
	cudaMemcpy(x.data().get(),y.Data(),sizeof(T)*y.Length(),cudaMemcpyDeviceToDevice);
}

template<typename T>
SparseDeviceMatrixCSR<T> MulCUSP(SparseDeviceMatrixCSR<T> A, SparseDeviceMatrixCSR<T> B) {
	thrust::device_ptr<uint>	wrapped_device_ArowStarts(A.RowStarts().v.Data());
	thrust::device_ptr<uint>	wrapped_device_AcolInd(A.ColIndices().v.Data());
    thrust::device_ptr<T>				wrapped_device_Aval(A.Values().v.Data());
	thrust::device_ptr<uint>	wrapped_device_BrowStarts(B.RowStarts().v.Data());
	thrust::device_ptr<uint>	wrapped_device_BcolInd(B.ColIndices().v.Data());
    thrust::device_ptr<T>				wrapped_device_Bval(B.Values().v.Data());	
    typedef typename cusp::array1d_view<thrust::device_ptr<uint> > DeviceIndexArrayView;
    typedef typename cusp::array1d_view<thrust::device_ptr<T> > DeviceValueArrayView;	
    typedef cusp::csr_matrix_view<DeviceIndexArrayView, DeviceIndexArrayView, DeviceValueArrayView> DeviceView;	

	DeviceIndexArrayView ArowStarts		(wrapped_device_ArowStarts, wrapped_device_ArowStarts + A.RowStarts().Length());
	DeviceIndexArrayView AcolInd		(wrapped_device_AcolInd, wrapped_device_AcolInd + A.ColIndices().Length());
	DeviceValueArrayView Aval			(wrapped_device_Aval, wrapped_device_Aval + A.Values().Length());
	DeviceView A_(A.Height(), A.Width(), A.NonZeroCount(), ArowStarts, AcolInd, Aval);
	DeviceIndexArrayView BrowStarts		(wrapped_device_BrowStarts, wrapped_device_BrowStarts + B.RowStarts().Length());
	DeviceIndexArrayView BcolInd		(wrapped_device_BcolInd, wrapped_device_BcolInd + B.ColIndices().Length());
	DeviceValueArrayView Bval			(wrapped_device_Bval, wrapped_device_Bval + B.Values().Length());
	DeviceView B_(B.Height(), B.Width(), B.NonZeroCount(), BrowStarts, BcolInd, Bval);	
	
	cusp::csr_matrix<uint,T,cusp::device_memory> C_;
	cusp::multiply(A_, B_, C_);
	CudaCheckError();
	
	DeviceVector<uint> CColIndices(C_.column_indices.size());	
	cudaMemcpy(CColIndices.v.Data(),C_.column_indices.data().get(),sizeof(uint)*C_.column_indices.size(),cudaMemcpyDeviceToDevice);	

	DeviceVector<uint> CRowStarts(C_.row_offsets.size());	
	cudaMemcpy(CRowStarts.v.Data(),C_.row_offsets.data().get(),sizeof(uint)*C_.row_offsets.size(),cudaMemcpyDeviceToDevice);	

	DeviceVector<T> CVals(C_.values.size());	
	cudaMemcpy(CVals.v.Data(),C_.values.data().get(),sizeof(T)*C_.values.size(),cudaMemcpyDeviceToDevice);
	
	return SparseDeviceMatrixCSR<T>(B.Width(),A.Height(),CVals,CColIndices,CRowStarts);
}

void InstantiateSpmmmTestOther(){
	SparseHostMatrixCSR<double> m=LoadMatrixCusp<double>("");
	MulCUSP(SparseDeviceMatrixCSR<double>(),SparseDeviceMatrixCSR<double>());
	MulCUSP(SparseDeviceMatrixCSR<float>(),SparseDeviceMatrixCSR<float>());	
}
*/

