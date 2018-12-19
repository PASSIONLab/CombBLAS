#pragma once

#ifndef IGNORE_MKL

#include "mkl_spblas.h"
#include "HostMatrix/HostComponentWise.h"
#include "HostMatrix/ComponentWiseNames.h"
#include "HostMatrix/HostTransfers.h"
#include "HostMatrix/SparseHostMatrixCSR.h"

static void Mul(HostVector<float> y, SparseHostMatrixCSR<float> A, HostVector<float> x) {
	Verify(y.IsSimple() && x.IsSimple(), FileAndLine);
	char transa = 'n', matdescra = 'G';
	int m = A.DimY(), k = A.DimX();
	float alpha = 1.0f, beta = 0.0f;
	mkl_scsrmv(&transa,&m,&k,&alpha,&matdescra,A.Values().Data(),(int*)A.ColIndices().Data(),(int*)A.RowStarts().Data(),((int*)A.RowStarts().Data())+1,x.Data(),&beta,y.Data());
}

static void Mul(HostVector<double> y, SparseHostMatrixCSR<double> A, HostVector<double> x){
	Verify(y.IsSimple() && x.IsSimple(), FileAndLine);
	char transa = 'n';
	char matdescra[4]; 
	matdescra[0] = 'G';
	matdescra[1] = 'L';
	matdescra[2] = 'N';
	matdescra[3] = 'C';
	int m = A.DimY(), k = A.DimX();
	double alpha = 1.0, beta = 0.0;
	mkl_dcsrmv(&transa,&m,&k,&alpha,matdescra,A.Values().Data(),(int*)A.ColIndices().Data(),(int*)A.RowStarts().Data(),(int*)A.RowStarts().Data()+1,x.Data(),&beta,y.Data());
}

static void Mul(HostMatrix<double> Y, SparseHostMatrixCSR<double> A, HostMatrix<double> X) {
	char transa = 'n';
	int m = A.DimY(), k = A.DimX();
	int n = X.DimX(), ldb = X.Stride();
	int ldc = Y.Stride();
	double alpha = 1.0, beta = 0.0;
	char matdescra[4]; 
	matdescra[0] = 'G';
	matdescra[1] = 'L';
	matdescra[2] = 'N';
	matdescra[3] = 'C';
	mkl_dcsrmm(&transa,&m,&n,&k,&alpha,matdescra,A.Values().Data(),(int*)A.ColIndices().Data(),(int*)A.RowStarts().Data(),(int*)A.RowStarts().Data()+1,X.Data(),&ldb,&beta,Y.Data(),&ldc);
}

static void Mul(HostMatrix<float> Y, SparseHostMatrixCSR<float> A, HostMatrix<float> X) {
	char transa = 'n';
	int m = A.DimY(), k = A.DimX();
	int n = X.DimX(), ldb = X.Stride();
	int ldc = Y.Stride();
	float alpha = 1.0, beta = 0.0;
	char matdescra[4]; 
	matdescra[0] = 'G';
	matdescra[1] = 'L';
	matdescra[2] = 'N';
	matdescra[3] = 'C';
	mkl_scsrmm(&transa,&m,&n,&k,&alpha,matdescra,A.Values().Data(),(int*)A.ColIndices().Data(),(int*)A.RowStarts().Data(),(int*)A.RowStarts().Data()+1,X.Data(),&ldb,&beta,Y.Data(),&ldc);
}
#include "HostMatrix/MulMKL.h"
static SparseHostMatrixCSR<double> MulTmp(SparseHostMatrixCSR<double> A, SparseHostMatrixCSR<double> X) {
	Verify(A.Width()==X.Height(),FileAndLine);
	Verify(A.ColIndices().Data() != X.ColIndices().Data(),FileAndLine);//must not overlap
	// Transform to 1-based index
	ComponentWiseAddUpConstant(A.ColIndices(),1);
	ComponentWiseAddUpConstant(A.RowStarts(),1);
	ComponentWiseAddUpConstant(X.ColIndices(),1);
	ComponentWiseAddUpConstant(X.RowStarts(),1);

	char transa = 'n';
	int job = 1, sort = 0, nzmax = 0, info;
	int m = A.DimY(), n = A.DimX();
	int k = X.DimX();
	HostVector<uint> YRowStarts(m+1);
	HostVector<int> dummy1;
	HostVector<double> dummy2;
	mkl_dcsrmultcsr(&transa,&job,&sort,&m,&n,&k, A.Values().Data(), (int*)A.ColIndices().Data(), (int*)A.RowStarts().Data(), 
												 X.Values().Data(), (int*)X.ColIndices().Data(), (int*)X.RowStarts().Data(),
												 dummy2.Data(),      dummy1.Data(),			     (int*)YRowStarts.Data(), &nzmax, &info);
	HostVector<uint> YColIndices(YRowStarts[m]-1);
	HostVector<double> YValues(YRowStarts[m]-1);
	job = 2;
	mkl_dcsrmultcsr(&transa,&job,&sort,&m,&n,&k, A.Values().Data(), (int*)A.ColIndices().Data(), (int*)A.RowStarts().Data(), 
												 X.Values().Data(), (int*)X.ColIndices().Data(), (int*)X.RowStarts().Data(),
												 YValues.Data(),    (int*)YColIndices.Data(),    (int*)YRowStarts.Data(), &nzmax, &info);
	SparseHostMatrixCSR<double> Y(k,m,YValues,YColIndices,YRowStarts);

	// Transform to 0-based index
	ComponentWiseAddUpConstant(A.ColIndices(),-1);	//necessary for A and X?
	ComponentWiseAddUpConstant(A.RowStarts(),-1);
	ComponentWiseAddUpConstant(X.ColIndices(),-1);
	ComponentWiseAddUpConstant(X.RowStarts(),-1);
	ComponentWiseAddUpConstant(Y.ColIndices(),-1);
	ComponentWiseAddUpConstant(Y.RowStarts(),-1);
	Verify(Y.Width()==X.Width() && Y.Height()==A.Height(),FileAndLine);
	return Y;
}

static SparseHostMatrixCSR<float> Mul(SparseHostMatrixCSR<float> A, SparseHostMatrixCSR<float> B) {
	return MulMKL(A,B);
}

#endif