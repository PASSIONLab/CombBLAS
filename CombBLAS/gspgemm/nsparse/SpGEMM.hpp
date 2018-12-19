#include <iostream>

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <nsparse.hpp>
#include <nsparse_asm.hpp>
#include <CSR.hpp>

#ifndef SPGEMM_H
#define SPGEMM_H

template <class idType>
__global__ void set_flop_per_row(idType *d_arpt, idType *d_acol, const idType* __restrict__ d_brpt, long long int *d_flop_per_row, idType nrow)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nrow) {
        return;
    }
    idType flop_per_row = 0;
    idType j;
    for (j = d_arpt[i]; j < d_arpt[i + 1]; j++) {
        flop_per_row += d_brpt[d_acol[j] + 1] - d_brpt[d_acol[j]];
    }
    d_flop_per_row[i] = flop_per_row;
}

template <class idType, class valType>
void get_spgemm_flop(CSR<idType, valType> a, CSR<idType, valType> b, long long int &flop)
{
    int GS, BS;
    long long int *d_flop_per_row;

    BS = MAX_LOCAL_THREAD_NUM;
    checkCudaErrors(cudaMalloc((void **)&(d_flop_per_row), sizeof(long long int) * (1 + a.nrow)));
  
    GS = div_round_up(a.nrow, BS);
    set_flop_per_row<<<GS, BS>>>(a.d_rpt, a.d_colids, b.d_rpt, d_flop_per_row, a.nrow);
  
    long long int *tmp = (long long int *)malloc(sizeof(long long int) * a.nrow);
    cudaMemcpy(tmp, d_flop_per_row, sizeof(long long int) * a.nrow, cudaMemcpyDeviceToHost);
    flop = thrust::reduce(thrust::device, d_flop_per_row, d_flop_per_row + a.nrow);

    flop *= 2;
    cudaFree(d_flop_per_row);

}

template <class idType, class valType>
cusparseStatus_t SpGEMM_cuSPARSE_numeric(CSR<idType, valType> a, CSR<idType, valType> b, CSR<idType, valType> &c, cusparseHandle_t cusparseHandle, cusparseOperation_t trans_a, cusparseOperation_t trans_b, cusparseMatDescr_t descr_a, cusparseMatDescr_t descr_b, cusparseMatDescr_t descr_c);

template <>
cusparseStatus_t SpGEMM_cuSPARSE_numeric<int, float>(CSR<int, float> a, CSR<int, float> b, CSR<int, float> &c, cusparseHandle_t cusparseHandle, cusparseOperation_t trans_a, cusparseOperation_t trans_b, cusparseMatDescr_t descr_a, cusparseMatDescr_t descr_b, cusparseMatDescr_t descr_c)
{
    return cusparseScsrgemm(cusparseHandle, trans_a, trans_b, a.nrow, b.ncolumn, a.ncolumn, descr_a, a.nnz, a.d_values, a.d_rpt, a.d_colids, descr_b, b.nnz, b.d_values, b.d_rpt, b.d_colids, descr_c, c.d_values, c.d_rpt, c.d_colids);
}

template <>
cusparseStatus_t SpGEMM_cuSPARSE_numeric<int, double>(CSR<int, double> a, CSR<int, double> b, CSR<int, double> &c, cusparseHandle_t cusparseHandle, cusparseOperation_t trans_a, cusparseOperation_t trans_b, cusparseMatDescr_t descr_a, cusparseMatDescr_t descr_b, cusparseMatDescr_t descr_c)
{
    return cusparseDcsrgemm(cusparseHandle, trans_a, trans_b, a.nrow, b.ncolumn, a.ncolumn, descr_a, a.nnz, a.d_values, a.d_rpt, a.d_colids, descr_b, b.nnz, b.d_values, b.d_rpt, b.d_colids, descr_c, c.d_values, c.d_rpt, c.d_colids);
}

template <class idType, class valType>
void SpGEMM_cuSPARSE_kernel(CSR<idType, valType> a, CSR<idType, valType> b, CSR<idType, valType> &c, cusparseHandle_t cusparseHandle, cusparseOperation_t trans_a, cusparseOperation_t trans_b, cusparseMatDescr_t descr_a, cusparseMatDescr_t descr_b, cusparseMatDescr_t descr_c)
{
    cusparseStatus_t status;
    c.nrow = a.nrow;
    c.ncolumn = b.ncolumn;
    c.devise_malloc = true;
    cudaMalloc((void **)&(c.d_rpt), sizeof(idType) * (c.nrow + 1));

    status = cusparseXcsrgemmNnz(cusparseHandle, trans_a, trans_b, a.nrow, b.ncolumn, a.ncolumn, descr_a, a.nnz, a.d_rpt, a.d_colids, descr_b, b.nnz, b.d_rpt, b.d_colids, descr_c, c.d_rpt, &(c.nnz));
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cout << "cuSPARSE failed at Symbolic phase" << endl;
    }

    cudaMalloc((void **)&(c.d_colids), sizeof(idType) * (c.nnz));
    cudaMalloc((void **)&(c.d_values), sizeof(valType) * (c.nnz));
        
    status = SpGEMM_cuSPARSE_numeric(a, b, c, cusparseHandle, trans_a, trans_b, descr_a, descr_b, descr_c);
    
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cout << "cuSPARSE failed at Numeric phase" << endl;
    }
}

template <class idType, class valType>
void SpGEMM_cuSPARSE(CSR<idType, valType> a, CSR<idType, valType> b, CSR<idType, valType> &c)
{
    cusparseHandle_t cusparseHandle;
    cusparseMatDescr_t descr_a, descr_b, descr_c;
    cusparseOperation_t trans_a, trans_b;

    trans_a = trans_b = CUSPARSE_OPERATION_NON_TRANSPOSE;
  
    /* Set up cuSPARSE Library */
    cusparseCreate(&cusparseHandle);
    cusparseCreateMatDescr(&descr_a);
    cusparseCreateMatDescr(&descr_b);
    cusparseCreateMatDescr(&descr_c);
    cusparseSetMatType(descr_a, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatType(descr_b, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatType(descr_c, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr_a, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatIndexBase(descr_b, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatIndexBase(descr_c, CUSPARSE_INDEX_BASE_ZERO);
  
    /* Execution of SpMV on Device */
    SpGEMM_cuSPARSE_kernel(a, b, c,
                           cusparseHandle,
                           trans_a, trans_b,
                           descr_a, descr_b, descr_c);
    cudaThreadSynchronize();
    
    c.memcpyDtH();

    c.release_csr();
    cusparseDestroy(cusparseHandle);
}

#endif

