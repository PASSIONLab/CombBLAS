#pragma once
#include "HostMatrix/SparseHostMatrixCSR.h"

#ifndef IGNORE_MKL

#include "mkl.h"
#include "mkl_spblas.h"

#endif

template<typename T> 
static void ToOneBased(SparseHostMatrixCSR<T>& A){
	HostVector<uint> rowStarts=A.RowStarts();
	#pragma omp parallel for
	for(int64 i = 0; i < rowStarts.Length(); i++)
		++rowStarts[i];
	HostVector<uint> colIndices=A.ColIndices();
	#pragma omp parallel for
	for(int64 i = 0; i < colIndices.Length(); i++)
		++colIndices[i];
}

template<typename T> 
static void ToZeroBased(SparseHostMatrixCSR<T>& A){
	HostVector<uint> rowStarts=A.RowStarts();
	#pragma omp parallel for
	for(int64 i = 0; i < rowStarts.Length(); i++)
		--rowStarts[i];
	HostVector<uint> colIndices=A.ColIndices();
	#pragma omp parallel for
	for(int64 i = 0; i < colIndices.Length(); i++)
		--colIndices[i];
}

#ifndef IGNORE_MKL

static SparseHostMatrixCSR<double> MulMKLOneBased(SparseHostMatrixCSR<double> A, SparseHostMatrixCSR<double> B){
	HostVector<uint> CRowStarts(A.Height()+1);
	HostVector<uint> CColIndices;
	HostVector<double> CVals;
	int info, request=1, sort=0, nzmax=0, m=A.Height(), n=A.Width(), k=B.Width();
	char trans='N';
	mkl_dcsrmultcsr(&trans, &request, &sort, &m, &n, &k, A.Values().Data(), (int*)A.ColIndices().Data(), (int*)A.RowStarts().Data(), B.Values().Data(), (int*)B.ColIndices().Data(), (int*)B.RowStarts().Data(), CVals.Data(), (int*)CColIndices.Data(), (int*)CRowStarts.Data(), &nzmax, &info);
	CColIndices = HostVector<uint>(CRowStarts[A.Height()]-1);
	CVals = HostVector<double>(CRowStarts[A.Height()]-1);
	request=2;
	mkl_dcsrmultcsr(&trans, &request, &sort, &m, &n, &k, A.Values().Data(), (int*)A.ColIndices().Data(), (int*)A.RowStarts().Data(), B.Values().Data(), (int*)B.ColIndices().Data(), (int*)B.RowStarts().Data(), CVals.Data(), (int*)CColIndices.Data(), (int*)CRowStarts.Data(), &nzmax, &info);	
	SparseHostMatrixCSR<double> C(B.Width(),A.Height(),CVals,CColIndices,CRowStarts);
	return C;
}
static SparseHostMatrixCSR<float> MulMKLOneBased(SparseHostMatrixCSR<float> A, SparseHostMatrixCSR<float> B){
	HostVector<uint> CRowStarts(A.Height()+1);
	HostVector<uint> CColIndices;
	HostVector<float> CVals;
	int info, request=1, sort=0, nzmax=0, m=A.Height(), n=A.Width(), k=B.Width();
	char trans='N';
	mkl_scsrmultcsr(&trans, &request, &sort, &m, &n, &k, A.Values().Data(), (int*)A.ColIndices().Data(), (int*)A.RowStarts().Data(), B.Values().Data(), (int*)B.ColIndices().Data(), (int*)B.RowStarts().Data(), CVals.Data(), (int*)CColIndices.Data(), (int*)CRowStarts.Data(), &nzmax, &info);
	CColIndices = HostVector<uint>(CRowStarts[A.Height()]-1);
	CVals = HostVector<float>(CRowStarts[A.Height()]-1);
	request=2;
	mkl_scsrmultcsr(&trans, &request, &sort, &m, &n, &k, A.Values().Data(), (int*)A.ColIndices().Data(), (int*)A.RowStarts().Data(), B.Values().Data(), (int*)B.ColIndices().Data(), (int*)B.RowStarts().Data(), CVals.Data(), (int*)CColIndices.Data(), (int*)CRowStarts.Data(), &nzmax, &info);	
	SparseHostMatrixCSR<float> C(B.Width(),A.Height(),CVals,CColIndices,CRowStarts);
	return C;
}

template<typename T>
static SparseHostMatrixCSR<T> MulMKL(SparseHostMatrixCSR<T> A, SparseHostMatrixCSR<T> B){
	ToOneBased(A);
	ToOneBased(B);
	SparseHostMatrixCSR<T> C=MulMKLOneBased(A,B);
	ToZeroBased(A);
	ToZeroBased(B);
	ToZeroBased(C);
	return C;
}

#else

template<typename T>
static SparseHostMatrixCSR<T> MulMKLOneBased(SparseHostMatrixCSR<T> A, SparseHostMatrixCSR<T> B){
	throw std::runtime_error("Not implemented");
}

template<typename T>
static SparseHostMatrixCSR<T> MulMKL(SparseHostMatrixCSR<T> A, SparseHostMatrixCSR<T> B){	
	return Mul(A,B);	
}

#endif
