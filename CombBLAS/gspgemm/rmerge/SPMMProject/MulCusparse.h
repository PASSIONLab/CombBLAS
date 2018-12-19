#pragma once
#include "DeviceMatrix/SparseDeviceMatrixCSR.h"
#include <cusparse_v2.h>

extern cusparseHandle_t cusparseHandle;
void CreateCusparseHandle();
void DestroyCusparseHandle();

static SparseDeviceMatrixCSR<float> MulCusparse(SparseDeviceMatrixCSR<float> A, SparseDeviceMatrixCSR<float> B) {
	throw std::runtime_error("Not implemented");
}

static SparseDeviceMatrixCSR<double> MulCusparse(SparseDeviceMatrixCSR<double> A, SparseDeviceMatrixCSR<double> B) {
	CreateCusparseHandle();
	cusparseMatDescr_t Adescr = 0, Bdescr = 0, Cdescr = 0;

	//cusparse matrix representations, A and B in CSR with int ptrs
	cusparseCreateMatDescr(&Adescr); cusparseSetMatType(Adescr,CUSPARSE_MATRIX_TYPE_GENERAL); cusparseSetMatIndexBase(Adescr,CUSPARSE_INDEX_BASE_ZERO);
	cusparseCreateMatDescr(&Bdescr); cusparseSetMatType(Bdescr,CUSPARSE_MATRIX_TYPE_GENERAL); cusparseSetMatIndexBase(Bdescr,CUSPARSE_INDEX_BASE_ZERO);
	cusparseCreateMatDescr(&Cdescr); cusparseSetMatType(Cdescr,CUSPARSE_MATRIX_TYPE_GENERAL); cusparseSetMatIndexBase(Cdescr,CUSPARSE_INDEX_BASE_ZERO);
	
	DeviceVector<uint> CRowPtr(A.Height()+1);
	int Cnnz;

	cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_HOST);
	cusparseXcsrgemmNnz(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, A.Height(), B.Width(), A.Width(), 
							Adescr, (int)A.NonZeroCount(), (int*)A.RowStarts().Data(), (int*)A.ColIndices().Data(), 
							Bdescr, (int)B.NonZeroCount(), (int*)B.RowStarts().Data(), (int*)B.ColIndices().Data(), 
							Cdescr, (int*)CRowPtr.Data(), &Cnnz);

	DeviceVector<double> CVal(Cnnz);
	DeviceVector<uint> CColInd(Cnnz);

	cusparseDcsrgemm(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, A.Height(), B.Width(), A.Width(), 
							Adescr, (int)A.NonZeroCount(), A.Values().Data(), (int*)A.RowStarts().Data(), (int*)A.ColIndices().Data(), 
							Bdescr, (int)B.NonZeroCount(), B.Values().Data(), (int*)B.RowStarts().Data(), (int*)B.ColIndices().Data(), 
							Cdescr, CVal.Data(), (int*)CRowPtr.Data(), (int*)CColInd.Data());
		
	SparseDeviceMatrixCSR<double> C = SparseDeviceMatrixCSR<double>(B.Width(),A.Height(),CVal,CColInd,CRowPtr);
	DestroyCusparseHandle();
	return C;
}

