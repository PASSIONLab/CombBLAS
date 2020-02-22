#pragma once 

/* #include <cusp/coo_matrix.h> */
/* #include <cusp/csr_matrix.h> */
/* #include <cusp/gallery/poisson.h> */
/* #include <cusp/io/matrix_market.h> */

#include "HostMatrix/SparseHostMatrixCSR.h"
#include "DeviceMatrix/SparseDeviceMatrixCSR.h"
#include "HostMatrix/SparseHostMatrixCSROperations.h"
#include "DeviceMatrix/SparseDeviceMatrixCSROperations.h"
#include "DeviceMatrix/DeviceTransfers.h"


/* template<typename T> */
/* static SparseHostMatrixCSR<T> ToSparseHostMatrixCSR(cusp::csr_matrix<uint,T,cusp::host_memory> A) { */
/* 	HostVector<T> Avalues(A.num_entries); */
/* 	for(int i = 0; i < A.num_entries; i++) Avalues[i] = A.values[i]; */

/* 	HostVector<uint> AcollIndices(A.num_entries); */
/* 	for (int i = 0; i < A.num_entries; i++) AcollIndices[i] = A.column_indices[i]; */
/* 	HostVector<uint> ArowStarts(A.num_rows+1); */
/* 	for (int i = 0; i < A.num_rows+1; i++) ArowStarts[i] = A.row_offsets[i]; */

/* 	SparseHostMatrixCSR<T> Aexmi((int)A.num_cols, (int)A.num_rows, Avalues, AcollIndices, ArowStarts); */
/* 	return Aexmi; */
/* } */

/* template<typename T> */
/* static cusp::csr_matrix<uint,T,cusp::host_memory> ToCuspHostCSR(SparseHostMatrixCSR<T> A) { */
/* 	cusp::csr_matrix<uint,T,cusp::host_memory> Acusp(A.Height(), A.Width(), A.NonZeroCount()); */

/* 	for (int i = 0; i < A.NonZeroCount(); i++) { */
/* 		Acusp.column_indices[i] = A.ColIndices()[i]; */
/* 		Acusp.values[i] = A.Values()[i]; */
/* 	} */

/* 	for (int i = 0; i < A.Height()+1; i++) { */
/* 		Acusp.row_offsets[i] = A.RowStarts()[i]; */
/* 	} */
/* 	return Acusp; */
/* } */

/* template<typename T> */
/* static cusp::array1d<T,cusp::host_memory> ToCuspArray1d(HostVector<T> x) { */
/* 	cusp::array1d<T,cusp::host_memory> xCusp(x.Length()); */

/* 	for (int i = 0; i < x.Length(); i++) { */
/* 		xCusp[i] = x[i]; */
/* 	} */

/* 	return xCusp; */
/* } */

template<typename T>
static void SparseHostMatrixCSRToCusparseCSR(SparseHostMatrixCSR<T> A, int* rowPtr, int* colInd, T* val) {
	rowPtr = (int*) malloc((A.Height()+1)*sizeof(rowPtr[0]));
	colInd = (int*) malloc(A.NonZeroCount()*sizeof(colInd[0]));
	val = (T*) malloc(A.NonZeroCount()*sizeof(val[0]));

	for (int i = 0; i < A.NonZeroCount(); i++) {
		colInd[i] = A.ColIndices()[i];
		val[i] = A.Values()[i];
	}

	for (int i = 0; i < A.Height()+1; i++) {
		rowPtr[i] = A.RowStarts()[i];
	}
}
