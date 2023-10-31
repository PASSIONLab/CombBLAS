#include "hip/hip_runtime.h"
#ifndef BIN_H
#define BIN_H

#include <iostream>

#include "hip/hip_runtime.h"


template <class idType,
		  int BIN_NUM>
class BIN
{
public:
    BIN(idType nrow):max_flop(0), max_nz(0)
    {
        stream = (hipStream_t *)malloc(sizeof(hipStream_t) * BIN_NUM);
        for (int i = 0; i < BIN_NUM; ++i) {
            hipStreamCreate(&(stream[i]));
        }
        bin_size = (idType *)malloc(sizeof(idType) * BIN_NUM);
        bin_offset = (idType *)malloc(sizeof(idType) * BIN_NUM);
        hipMalloc((void **)&(d_permutation), sizeof(idType) * nrow);
        hipMalloc((void **)&(d_count), sizeof(idType) * (nrow + 1));
        hipMalloc((void **)&(d_max), sizeof(idType));
        hipMalloc((void **)&(d_bin_size), sizeof(idType) * BIN_NUM);
        hipMalloc((void **)&(d_bin_offset), sizeof(idType) * BIN_NUM);
    }


	
    ~BIN()
    {
        hipFree(d_count);
        hipFree(d_permutation);
        hipFree(d_max);
        hipFree(d_bin_size);
        hipFree(d_bin_offset);
        free(bin_size);
        free(bin_offset);
        for (int i = 0; i < BIN_NUM; i++) {
            hipStreamDestroy(stream[i]);
        }
        free(stream);
    }
    
    void set_max_bin(idType *d_arpt, idType *d_acol, idType *d_brpt, idType M, int TS_S_P, int TS_S_T);
    void set_min_bin(idType M, int TS_N_P, int TS_N_T);
    void set_min_bin(idType *d_rpt, idType M, int TS_N_P, int TS_N_T);

    hipStream_t *stream;
    idType *bin_size;
    idType *bin_offset;
    idType *d_bin_size;
    idType *d_bin_offset;
    idType *d_count;
    idType *d_permutation;
    idType *d_max;
    
    idType max_flop;
    idType max_nz;
};



template <class idType>
__global__
void set_flop_per_row(idType *d_arpt,
					  idType *d_acol,
					  const idType* __restrict__ d_brpt,
					  idType *d_count,
					  idType *d_max_flop,
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
    d_count[i] = flop_per_row;
    atomicMax(d_max_flop, flop_per_row);
}



template <class idType,
		  int BIN_NUM>
__global__
void set_bin(idType *d_count,
			 idType *d_bin_size,
			 idType *d_max,
			 idType nrow,
			 idType min,
			 idType mmin)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nrow) {
        return;
    }
    idType nz_per_row = d_count[i];
    atomicMax(d_max, nz_per_row);

    idType j = 0;
    for (j = 0; j < BIN_NUM - 2; j++) {
        if (nz_per_row <= (min << j)) {
            if (nz_per_row <= (mmin)) {
                atomicAdd(d_bin_size + j, 1);
            }
            else {
                atomicAdd(d_bin_size + j + 1, 1);
            }
            return;
        }
    }
    atomicAdd(d_bin_size + BIN_NUM - 1, 1);
}



template <class idType>
__global__
void
init_row_perm(idType *d_permutation,
			  idType M)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= M) {
        return;
    }
  
    d_permutation[i] = i;
}



template <class idType,
		  int BIN_NUM>
__global__
void
set_row_perm(idType *d_bin_size,
			 idType *d_bin_offset,
			 idType *d_max_row_nz,
			 idType *d_permutation,
			 idType M,
			 idType min,
			 idType mmin)
{
    idType i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= M) {
        return;
    }

    idType nz_per_row = d_max_row_nz[i];
    idType dest;
  
    idType j = 0;
    for (j = 0; j < BIN_NUM - 2; j++) {
        if (nz_per_row <= (min << j)) {
            if (nz_per_row <= mmin) {
                dest = atomicAdd(d_bin_size + j, 1);
                d_permutation[d_bin_offset[j] + dest] = i;
            }
            else {
                dest = atomicAdd(d_bin_size + j + 1, 1);
                d_permutation[d_bin_offset[j + 1] + dest] = i;
            }
            return;
        }
    }
    dest = atomicAdd(d_bin_size + BIN_NUM - 1, 1);
    d_permutation[d_bin_offset[BIN_NUM - 1] + dest] = i;
}



template <class idType,
		  int BIN_NUM>
void
BIN<idType, BIN_NUM>::set_max_bin(idType *d_arpt,
								  idType *d_acol,
								  idType *d_brpt,
								  idType M,
								  int TS_S_P,
								  int TS_S_T)
{
    idType i;
    idType GS, BS;
  
    for (i = 0; i < BIN_NUM; i++) {
        bin_size[i] = 0;
        bin_offset[i] = 0;
    }
  
    hipMemcpy(d_bin_size, bin_size, sizeof(idType) * BIN_NUM, hipMemcpyHostToDevice);
    hipMemcpy(d_max, &(max_flop), sizeof(idType), hipMemcpyHostToDevice);
  
    BS = 1024;
    GS = div_round_up(M, BS);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(set_flop_per_row<idType>), dim3(GS), dim3(BS), 0, 0, d_arpt, d_acol, d_brpt, d_count, d_max, M);
    hipMemcpy(&(max_flop), d_max, sizeof(idType), hipMemcpyDeviceToHost);

	// cout << "max flop " << max_flop << endl;
  
    if (max_flop > TS_S_P) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(set_bin<idType, BIN_NUM>), dim3(GS), dim3(BS), 0, 0, d_count, d_bin_size, d_max, M, TS_S_T, TS_S_P);
  
        hipMemcpy(bin_size, d_bin_size, sizeof(idType) * BIN_NUM, hipMemcpyDeviceToHost);
        hipMemcpy(d_bin_size, bin_offset, sizeof(idType) * BIN_NUM, hipMemcpyHostToDevice);

        for (i = 0; i < BIN_NUM - 1; i++) {
            bin_offset[i + 1] = bin_offset[i] + bin_size[i];
        }
        hipMemcpy(d_bin_offset, bin_offset, sizeof(idType) * BIN_NUM, hipMemcpyHostToDevice);

        hipLaunchKernelGGL(HIP_KERNEL_NAME(set_row_perm<idType, BIN_NUM>), dim3(GS), dim3(BS), 0, 0, d_bin_size, d_bin_offset, d_count, d_permutation, M, TS_S_T, TS_S_P);
    }
    else {
        bin_size[0] = M;
        for (i = 1; i < BIN_NUM; i++) {
            bin_size[i] = 0;
        }
        bin_offset[0] = 0;
        for (i = 1; i < BIN_NUM; i++) {
            bin_offset[i] = M;
        }
        hipLaunchKernelGGL(init_row_perm, dim3(GS), dim3(BS), 0, 0, d_permutation, M);
    }
}



template <class idType,
		  int BIN_NUM>
void
BIN<idType, BIN_NUM>::set_min_bin(idType M,
								  int TS_N_P,
								  int TS_N_T)
{
    idType i;
    idType GS, BS;
  
    for (i = 0; i < BIN_NUM; i++) {
        bin_size[i] = 0;
        bin_offset[i] = 0;
    }
  
    hipMemcpy(d_bin_size, bin_size, sizeof(idType) * BIN_NUM, hipMemcpyHostToDevice);
    hipMemcpy(d_max, &(max_nz), sizeof(idType), hipMemcpyHostToDevice);
  
    BS = 1024;
    GS = div_round_up(M, BS);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(set_bin<idType, BIN_NUM>), dim3(GS), dim3(BS), 0, 0, d_count, d_bin_size,
                                         d_max,
                                         M, TS_N_T, TS_N_P);
  
    hipMemcpy(&(max_nz), d_max, sizeof(idType), hipMemcpyDeviceToHost);
    if (max_nz > TS_N_P) {
        hipMemcpy(bin_size, d_bin_size, sizeof(idType) * BIN_NUM, hipMemcpyDeviceToHost);
        hipMemcpy(d_bin_size, bin_offset, sizeof(idType) * BIN_NUM, hipMemcpyHostToDevice);

        for (i = 0; i < BIN_NUM - 1; i++) {
            bin_offset[i + 1] = bin_offset[i] + bin_size[i];
        }
        hipMemcpy(d_bin_offset, bin_offset, sizeof(idType) * BIN_NUM, hipMemcpyHostToDevice);
  
        hipLaunchKernelGGL(HIP_KERNEL_NAME(set_row_perm<idType, BIN_NUM>), dim3(GS), dim3(BS), 0, 0, d_bin_size, d_bin_offset, d_count, d_permutation, M, TS_N_T, TS_N_P);
    }

    else {
        bin_size[0] = M;
        for (i = 1; i < BIN_NUM; i++) {
            bin_size[i] = 0;
        }
        bin_offset[0] = 0;
        for (i = 1; i < BIN_NUM; i++) {
            bin_offset[i] = M;
        }
        BS = 1024;
        GS = div_round_up(M, BS);
        hipLaunchKernelGGL(init_row_perm, dim3(GS), dim3(BS), 0, 0, d_permutation, M);
    }
}

template <class idType>
__global__ void set_nnz_per_row_from_rpt(idType *d_rpt, idType *d_count, idType nrow)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nrow) {
        return;
    }
    d_count[i] = d_rpt[i + 1] - d_rpt[i];
}

template <class idType, int BIN_NUM>
void BIN<idType, BIN_NUM>::set_min_bin(idType *d_rpt, idType M, int TS_N_P, int TS_N_T)
{
    idType i;
    idType GS, BS;
  
    for (i = 0; i < BIN_NUM; i++) {
        bin_size[i] = 0;
        bin_offset[i] = 0;
    }
  
    hipMemcpy(d_bin_size, bin_size, sizeof(idType) * BIN_NUM, hipMemcpyHostToDevice);
    hipMemcpy(d_max, &(max_nz), sizeof(idType), hipMemcpyHostToDevice);
  
    BS = 1024;
    GS = div_round_up(M, BS);
    hipLaunchKernelGGL(set_nnz_per_row_from_rpt, dim3(GS), dim3(BS), 0, 0, d_rpt, d_count, M);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(set_bin<idType, BIN_NUM>), dim3(GS), dim3(BS), 0, 0, d_count, d_bin_size,
                                         d_max,
                                         M, TS_N_T, TS_N_P);
    
    hipMemcpy(&(max_nz), d_max, sizeof(idType), hipMemcpyDeviceToHost);
    if (max_nz > TS_N_P) {
        hipMemcpy(bin_size, d_bin_size, sizeof(idType) * BIN_NUM, hipMemcpyDeviceToHost);
        hipMemcpy(d_bin_size, bin_offset, sizeof(idType) * BIN_NUM, hipMemcpyHostToDevice);

        for (i = 0; i < BIN_NUM - 1; i++) {
            bin_offset[i + 1] = bin_offset[i] + bin_size[i];
        }
        hipMemcpy(d_bin_offset, bin_offset, sizeof(idType) * BIN_NUM, hipMemcpyHostToDevice);
  
        hipLaunchKernelGGL(HIP_KERNEL_NAME(set_row_perm<idType, BIN_NUM>), dim3(GS), dim3(BS), 0, 0, d_bin_size, d_bin_offset, d_count, d_permutation, M, TS_N_T, TS_N_P);
    }

    else {
        bin_size[0] = M;
        for (i = 1; i < BIN_NUM; i++) {
            bin_size[i] = 0;
        }
        bin_offset[0] = 0;
        for (i = 1; i < BIN_NUM; i++) {
            bin_offset[i] = M;
        }
        BS = 1024;
        GS = div_round_up(M, BS);
        hipLaunchKernelGGL(init_row_perm, dim3(GS), dim3(BS), 0, 0, d_permutation, M);
    }
}

#endif
