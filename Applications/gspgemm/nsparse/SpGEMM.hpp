#include <iostream>

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "nsparse.hpp"
#include "nsparse_asm.hpp"
// #include "nsparse_util.hpp"
#include "CSR.hpp"

#ifndef SPGEMM_H
#define SPGEMM_H



template <class idType>
__global__
void set_flop_per_row (idType *d_arpt,
					   idType *d_acol,
					   const idType* __restrict__ d_brpt,
					   long long int *d_flop_per_row,
					   idType nrow)
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



template <class idType,
		  class valType>
void
get_spgemm_flop (CSR<idType, valType> a,
				 CSR<idType, valType> b,
				 long long int &flop)
{
    int GS, BS;
    long long int *d_flop_per_row;

    BS = MAX_LOCAL_THREAD_NUM;
    cudaMalloc((void **)&(d_flop_per_row), sizeof(long long int) * (1 + a.nrow));
  
    GS = div_round_up(a.nrow, BS);
    set_flop_per_row<<<GS, BS>>>(a.d_rpt, a.d_colids, b.d_rpt, d_flop_per_row, a.nrow);
  
    long long int *tmp = (long long int *)malloc(sizeof(long long int) * a.nrow);
    cudaMemcpy(tmp, d_flop_per_row, sizeof(long long int) * a.nrow, cudaMemcpyDeviceToHost);
    flop = thrust::reduce(thrust::device, d_flop_per_row, d_flop_per_row + a.nrow);

    flop *= 2;
    cudaFree(d_flop_per_row);

}

#endif

