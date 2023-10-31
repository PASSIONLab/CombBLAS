#include "hip/hip_runtime.h"
#include <iostream>

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "nsparse.hpp"
#include "nsparse_asm.hpp"
#include "CSR.hpp"
#include "BIN.hpp"

#ifndef HASHSPGEMM_H
#define HASHSPGEMM_H

#define BIN_NUM 7
#define PWARP 4
#define TS_S_P 32 //Table Size for Symbolic in PWARP per row
#define TS_N_P 16 //Table Size for Numeric in PWARP per row
#define TS_S_T 512 //Table Size for Symbolic in Thread block per row
#define TS_N_T 256 //Table Size for Numeric in Thread block per row

#define SHARED_S_P 4096 // Total table sizes required by one thread block in PWARP Symbolic
#define SHARED_N_P 2048 // Total table sizes required by one thread block in PWARP Numeric
#define HASH_SCAL 107

#define ORIGINAL_HASH



template <class idType>
__global__
void
print_arr (idType *arr,
		   idType sz,
		   idType low,
		   idType high)
{
	idType i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= sz)
		return;

	if (i >= low && i <= high)
		printf("thread %d val %d\n", i, arr[i]);
}



template <class idType>
__global__
void
init_id_table(idType *d_id_table,
			  idType nnz)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) {
        return;
    }
    d_id_table[i] = -1;
}



template <class idType,
		  class valType>
__global__
void init_value_table(valType *d_values,
					  idType nnz)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) {
        return;
    }
    d_values[i] = 0;
}



template <class idType>
__global__
void
hash_symbolic_pwarp(const idType *d_arpt,
					const idType *d_acolids,
					const idType* __restrict__ d_brpt,
					const idType* __restrict__ d_bcolids,
					const idType *d_permutation,
					idType *d_row_nz,
					idType bin_offset,
					idType M)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;
    idType rid = i / PWARP;
    idType tid = i % PWARP;
    idType local_rid = rid % (blockDim.x / PWARP);
  
    idType j, k;
    idType soffset;
    idType acol, bcol, key, hash, adr, nz, old;
    __shared__ idType id_table[SHARED_S_P];
  
    soffset = local_rid * TS_S_P;
  
    for (j = tid; j < TS_S_P; j += PWARP) {
        id_table[soffset + j] = -1;
    }
    if (rid >= M) {
        return;
    }

    rid = d_permutation[rid + bin_offset];
    nz = 0;
    for (j = d_arpt[rid] + tid; j < d_arpt[rid + 1]; j += PWARP) {
        // acol = ld_gbl_col(d_acolids + j); // @OGUZ-BEWARE @OGUZ-TODO
		acol = d_acolids[j];
        for (k = d_brpt[acol]; k < d_brpt[acol + 1]; k++) {
            bcol = d_bcolids[k];
            key = bcol;
            hash = (bcol * HASH_SCAL) & (TS_S_P - 1);
            adr = soffset + hash;
            while (1) {
                if (id_table[adr] == key) {
                    break;
                }
                else if (id_table[adr] == -1) {
                    old = atomicCAS(id_table + adr, -1, key);
                    if (old == -1) {
                        nz++;
                        break;
                    }
                }
                else {
                    hash = (hash + 1) & (TS_S_P - 1);
                    adr = soffset + hash;
                }
            }
        }
    }
  
    for (j = PWARP / 2; j >= 1; j /= 2) {
		nz += __shfl_xor(nz, j); // @OGUZ-BEWARE
    }

    if (tid == 0) {
        d_row_nz[rid] = nz;
    }
}



template <class idType,
		  int SH_ROW>
__global__
void
hash_symbolic_tb(const idType *d_arpt,
				 const idType *d_acolids,
				 const idType* __restrict__ d_brpt,
				 const idType* __restrict__ d_bcolids,
				 idType *d_permutation,
				 idType *d_row_nz,
				 idType bin_offset,
				 idType M)
{
    idType rid = blockIdx.x;
    idType tid = threadIdx.x & (nsp_warp_sz - 1);
    idType wid = threadIdx.x / nsp_warp_sz;
    idType wnum = blockDim.x / nsp_warp_sz;
    idType j, k;
    idType bcol, key, hash, old;
    idType nz, adr;
    idType acol;

    __shared__ idType id_table[SH_ROW];
    for (j = threadIdx.x; j < SH_ROW; j += blockDim.x) {
        id_table[j] = -1;
    }
  
    if (rid >= M) {
        return;
    }
  
    __syncthreads();

    nz = 0;
    rid = d_permutation[rid + bin_offset];
    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum) {
        // acol = ld_gbl_col(d_acolids + j); // @OGUZ-BEWARE @OGUZ-TODO
		acol = d_acolids[j];
        for (k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k += nsp_warp_sz) {
            bcol = d_bcolids[k];
            key = bcol;
            hash = (bcol * HASH_SCAL) & (SH_ROW - 1);
            adr = hash;
            while (1) {
                if (id_table[adr] == key) {
                    break;
                }
                else if (id_table[adr] == -1) {
                    old = atomicCAS(id_table + adr, -1, key);
                    if (old == -1) {
                        nz++;
                        break;
                    }
                }
                else {
                    hash = (hash + 1) & (SH_ROW - 1);
                    adr = hash;
                }
            }
        }
    }

    for (j = nsp_warp_sz / 2; j >= 1; j /= 2) {
        nz += __shfl_xor(nz, j); // @OGUZ-BEWARE
    }
  
    __syncthreads();
    if (threadIdx.x == 0) {
        id_table[0] = 0;
    }
    __syncthreads();

    if (tid == 0) {
        atomicAdd(id_table, nz);
    }
    __syncthreads();
  
    if (threadIdx.x == 0) {
        d_row_nz[rid] = id_table[0];
    }
}



template <class idType,
		  int SH_ROW>
__global__
void
hash_symbolic_tb_large(const idType *d_arpt,
					   const idType *d_acolids,
					   const idType* __restrict__ d_brpt,
					   const idType* __restrict__ d_bcolids,
					   idType *d_permutation,
					   idType *d_row_nz,
					   idType *d_fail_count,
					   idType *d_fail_perm,
					   idType bin_offset,
					   idType M)
{
    idType rid = blockIdx.x;
    idType tid = threadIdx.x & (nsp_warp_sz - 1);
    idType wid = threadIdx.x / nsp_warp_sz;
    idType wnum = blockDim.x / nsp_warp_sz;
    idType j, k;
    idType bcol, key, hash, old;
    idType adr;
    idType acol;

    __shared__ idType id_table[SH_ROW];
    __shared__ idType snz[1];
    for (j = threadIdx.x; j < SH_ROW; j += blockDim.x) {
        id_table[j] = -1;
    }
    if (threadIdx.x == 0) {
        snz[0] = 0;
    }
  
    if (rid >= M) {
        return;
    }
  
    __syncthreads();
  
    rid = d_permutation[rid + bin_offset];
    idType count = 0;
    idType border = SH_ROW >> 1;
    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum) {
        // acol = ld_gbl_col(d_acolids + j); // @OGUZ-BEWARE @OGUZ-TODO
		acol = d_acolids[j];
        for (k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k += nsp_warp_sz) {
            bcol = d_bcolids[k];
            key = bcol;
            hash = (bcol * HASH_SCAL) & (SH_ROW - 1);
            adr = hash;
            while (count < border && snz[0] < border) {
                if (id_table[adr] == key) {
                    break;
                }
                else if (id_table[adr] == -1) {
                    old = atomicCAS(id_table + adr, -1, key);
                    if (old == -1) {
                        atomicAdd(snz, 1);
                        break;
                    }
                }
                else {
                    hash = (hash + 1) & (SH_ROW - 1);
                    adr = hash;
                    count++;
                }
            }
            if (count >= border || snz[0] >= border) {
                break;
            }
        }
        if (count >= border || snz[0] >= border) {
            break;
        }
    }
  
    __syncthreads();
    if (count >= border || snz[0] >= border) {
        if (threadIdx.x == 0) {
            idType d = atomicAdd(d_fail_count, 1);
            d_fail_perm[d] = rid;
        }
    }
    else {
        if (threadIdx.x == 0) {
            d_row_nz[rid] = snz[0];
        }
    }
}



template <class idType>
__global__
void hash_symbolic_gl(const idType *d_arpt,
					  const idType *d_acol,
					  const idType* __restrict__ d_brpt,
					  const idType* __restrict__ d_bcol,
					  const idType *d_permutation,
					  idType *d_row_nz,
					  idType *d_id_table,
					  idType max_row_nz,
					  idType bin_offset,
					  idType M)
{
    idType rid = blockIdx.x;
    idType tid = threadIdx.x & (nsp_warp_sz - 1);
    idType wid = threadIdx.x / nsp_warp_sz;
    idType wnum = blockDim.x / nsp_warp_sz;
    idType j, k;
    idType bcol, key, hash, old;
    idType nz, adr;
    idType acol;
    idType offset = rid * max_row_nz;

    __shared__ idType snz[1];
    if (threadIdx.x == 0) {
        snz[0] = 0;
    }
    __syncthreads();
  
    if (rid >= M) {
        return;
    }
  
    nz = 0;
    rid = d_permutation[rid + bin_offset];
    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum) {
        // acol = ld_gbl_col(d_acol + j); // @OGUZ-BEWARE @OGUZ-TODO
		acol = d_acol[j];
        for (k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k += nsp_warp_sz) {
            bcol = d_bcol[k];
            key = bcol;
            hash = (bcol * HASH_SCAL) % max_row_nz;
            adr = offset + hash;
            while (1) {
                if (d_id_table[adr] == key) {
                    break;
                }
                else if (d_id_table[adr] == -1) {
                    old = atomicCAS(d_id_table + adr, -1, key);
                    if (old == -1) {
                        nz++;
                        break;
                    }
                }
                else {
                    hash = (hash + 1) % max_row_nz;
                    adr = offset + hash;
                }
            }
        }
    }
  
    for (j = nsp_warp_sz / 2; j >= 1; j /= 2) {
        nz += __shfl_xor(nz, j); // @OGUZ-BEWARE
    }
  
    if (tid == 0) {
        atomicAdd(snz, nz);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        d_row_nz[rid] = snz[0];
    }
}



template <class idType>
__global__
void
hash_symbolic_gl2(const idType *d_arpt,
				  const idType *d_acol,
				  const idType* __restrict__ d_brpt,
				  const idType* __restrict__ d_bcol,
				  const idType *d_permutation,
				  idType *d_row_nz,
				  idType *d_id_table,
				  idType max_row_nz,
				  idType bin_offset,
				  idType total_row_num,
				  idType conc_row_num)
{
    idType rid = blockIdx.x;
    idType tid = threadIdx.x & (nsp_warp_sz - 1);
    idType wid = threadIdx.x / nsp_warp_sz;
    idType wnum = blockDim.x / nsp_warp_sz;
    idType j, k;
    idType bcol, key, hash, old;
    idType nz, adr;
    idType acol, offset;
    idType target;

    offset = rid * max_row_nz;
    __shared__ idType snz[1];
  
    for (; rid < total_row_num; rid += conc_row_num) {
        for (j = threadIdx.x; j < max_row_nz; j += blockDim.x) {
            d_id_table[offset + j] = -1;
        }
        if (threadIdx.x == 0) {
            snz[0] = 0;
        }
        __syncthreads();

        nz = 0;
        target = d_permutation[rid + bin_offset];
        for (j = d_arpt[target] + wid; j < d_arpt[target + 1]; j += wnum) {
            // acol = ld_gbl_col(d_acol + j); // @OGUZ-BEWARE @OGUZ-TODO
			acol = d_acol[j];
            for (k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k += nsp_warp_sz) {
                bcol = d_bcol[k];
                key = bcol;
                hash = (bcol * HASH_SCAL) % max_row_nz;
                adr = offset + hash;
                while (1) {
                    if (d_id_table[adr] == key) {
                        break;
                    }
                    else if (d_id_table[adr] == -1) {
                        old = atomicCAS(d_id_table + adr, -1, key);
                        if (old == -1) {
                            nz++;
                            break;
                        }
                    }
                    else {
                        hash = (hash + 1) % max_row_nz;
                        adr = offset + hash;
                    }
                }
            }
        }
  
        for (j = nsp_warp_sz / 2; j >= 1; j /= 2) {
            nz += __shfl_xor(nz, j); // @OGUZ-BEWARE
        }
  
        if (tid == 0) {
            atomicAdd(snz, nz);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            d_row_nz[target] = snz[0];
        }
        __syncthreads();
    }
}




template <class idType,
		  class valType>
void
hash_symbolic(CSR<idType, valType> a,
			  CSR<idType, valType> b,
			  CSR<idType, valType> &c,
			  BIN<idType, BIN_NUM> &bin)
{
    idType i;
    idType GS, BS;

    for (i = BIN_NUM - 1; i >= 0; i--)
	{
		// std::cout << "bin " << i << " size " << bin.bin_size[i] << std::endl;
        if (bin.bin_size[i] > 0)
		{
            switch (i) {
            case 0 :
				// std::cout << "case 0 calling hash_symbolic_pwarp" << std::endl;
                BS = 512;
                GS = div_round_up(bin.bin_size[i] * PWARP, BS);
                hipLaunchKernelGGL(hash_symbolic_pwarp, dim3(GS), dim3(BS), 0, bin.stream[i],
								   a.d_rpt, a.d_colids, b.d_rpt, b.d_colids, bin.d_permutation,
								   bin.d_count, bin.bin_offset[i], bin.bin_size[i]);
                break;
            case 1 :
				// std::cout << "case 1 calling hash_symbolic_tb-512" << std::endl;
                BS = 64;
                GS = bin.bin_size[i];
            	hipLaunchKernelGGL(HIP_KERNEL_NAME(hash_symbolic_tb<idType, 512>), dim3(GS), dim3(BS), 0, bin.stream[i],
								   a.d_rpt, a.d_colids, b.d_rpt, b.d_colids, bin.d_permutation,
								   bin.d_count, bin.bin_offset[i], bin.bin_size[i]);
            	break;
            case 2 :
				// std::cout << "case 2 calling hash_symbolic_tb-1024" << std::endl;
                BS = 128;
                GS = bin.bin_size[i];
            	hipLaunchKernelGGL(HIP_KERNEL_NAME(hash_symbolic_tb<idType, 1024>), dim3(GS), dim3(BS), 0, bin.stream[i],
								   a.d_rpt, a.d_colids, b.d_rpt, b.d_colids, bin.d_permutation,
								   bin.d_count, bin.bin_offset[i], bin.bin_size[i]);
            	break;
            case 3 :
				// std::cout << "case 3 calling hash_symbolic_tb-2048" << std::endl;
                BS = 256;
                GS = bin.bin_size[i];
            	hipLaunchKernelGGL(HIP_KERNEL_NAME(hash_symbolic_tb<idType, 2048>), dim3(GS), dim3(BS), 0, bin.stream[i],
								   a.d_rpt, a.d_colids, b.d_rpt, b.d_colids, bin.d_permutation,
								   bin.d_count, bin.bin_offset[i], bin.bin_size[i]);
            	break;
            case 4 :
				// std::cout << "case 4 calling hash_symbolic_tb-4096" << std::endl;
                BS = 512;
                GS = bin.bin_size[i];
            	hipLaunchKernelGGL(HIP_KERNEL_NAME(hash_symbolic_tb<idType, 4096>), dim3(GS), dim3(BS), 0, bin.stream[i],
								   a.d_rpt, a.d_colids, b.d_rpt, b.d_colids, bin.d_permutation,
								   bin.d_count, bin.bin_offset[i], bin.bin_size[i]);
            	break;
            case 5 :
				// std::cout << "case 5 calling hash_symbolic_tb-8192" << std::endl;
                BS = 1024;
                GS = bin.bin_size[i];
            	hipLaunchKernelGGL(HIP_KERNEL_NAME(hash_symbolic_tb<idType, 8192>), dim3(GS), dim3(BS), 0, bin.stream[i],
								   a.d_rpt, a.d_colids, b.d_rpt, b.d_colids, bin.d_permutation,
								   bin.d_count, bin.bin_offset[i], bin.bin_size[i]);
            	break;
            case 6 :
                {
#ifdef ORIGINAL_HASH
					// std::cout << "case 6 ORIGINAL_HASH version" << std::endl;
            	    idType fail_count;
            	    idType *d_fail_count, *d_fail_perm;
            	    fail_count = 0;
            	    HIPCHECK(hipMalloc((void **)&d_fail_count, sizeof(idType)));
            	    HIPCHECK(hipMalloc((void **)&d_fail_perm, sizeof(idType) * bin.bin_size[i]));
            	    hipMemcpy(d_fail_count, &fail_count, sizeof(idType), hipMemcpyHostToDevice);
            	    BS = 1024;
            	    GS = bin.bin_size[i];
					// std::cout << "case 6 calling hash_symbolic_tb_large-8192" << std::endl;
            	    hipLaunchKernelGGL(HIP_KERNEL_NAME(hash_symbolic_tb_large<idType, 8192>), dim3(GS), dim3(BS), 0, bin.stream[i],
									   a.d_rpt, a.d_colids, b.d_rpt, b.d_colids, bin.d_permutation,
									   bin.d_count, d_fail_count, d_fail_perm, bin.bin_offset[i], bin.bin_size[i]);
            	    hipMemcpy(&fail_count, d_fail_count, sizeof(idType), hipMemcpyDeviceToHost);
            	    if (fail_count > 0) {
						// std::cout << "fail_count > 0, calling hash_symbolic_gl" << std::endl;
              	        idType max_row_nz = bin.max_flop;
            	        size_t table_size = (size_t)max_row_nz * fail_count;
            	        idType *d_id_table;
            	        HIPCHECK(hipMalloc((void **)&(d_id_table), sizeof(idType) * table_size));
            	        BS = 1024;
            	        GS = div_round_up(table_size, BS);
            	        hipLaunchKernelGGL(HIP_KERNEL_NAME(init_id_table<idType>), dim3(GS), dim3(BS), 0, bin.stream[i],
										   d_id_table, table_size);
            	        GS = bin.bin_size[i];
	                    hipLaunchKernelGGL(HIP_KERNEL_NAME(hash_symbolic_gl<idType>), dim3(GS), dim3(BS), 0, bin.stream[i],
										   a.d_rpt, a.d_colids, b.d_rpt, b.d_colids, d_fail_perm, bin.d_count, d_id_table, max_row_nz, 0, fail_count);
                        hipFree(d_id_table);
  	                }
                    hipFree(d_fail_count);
                    hipFree(d_fail_perm);
#else
					// std::cout << "case 6 not ORIGINAL_HASH version" << std::endl;
                    idType max_row_nz = bin.max_flop;
                    idType conc_row_num = min(56 * 2, bin.bin_size[i]);
                    idType table_size = max_row_nz * conc_row_num;
                    while (table_size * sizeof(idType) > 1024 * 1024 * 1024) {
                        conc_row_num /= 2;
                        table_size = max_row_nz * conc_row_num;
                    }
                    idType *d_id_table;
                    HIPCHECK(hipMalloc((void **)&d_id_table, sizeof(idType) * table_size));
                    BS = 1024;
                    // GS = div_round_up(table_size, BS);
                    // hipLaunchKernelGGL(HIP_KERNEL_NAME(init_id_table<idType>), dim3(GS), dim3(BS), 0, bin.stream[i], d_id_table, table_size);
					// std::cout << "case 6 calling hash_symbolic_gl2" << std::endl;
                    GS = conc_row_num;
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(hash_symbolic_gl2<idType>), dim3(GS), dim3(BS), 0, bin.stream[i],
									   a.d_rpt, a.d_colids, b.d_rpt, b.d_colids, bin.d_permutation, bin.d_count,
									   d_id_table, max_row_nz, bin.bin_offset[i], bin.bin_size[i], conc_row_num);
                    hipFree(d_id_table);
#endif
                }
                break;
            default:
                ;
            }
        }
    }
	
    hipDeviceSynchronize();
    thrust::exclusive_scan(thrust::device, bin.d_count, bin.d_count + (a.nrow + 1), c.d_rpt, 0);
    hipMemcpy(&(c.nnz), c.d_rpt + c.nrow, sizeof(idType), hipMemcpyDeviceToHost);
}



template <class idType,
		  class valType,
		  bool sort>
__global__
void
hash_numeric_pwarp(const idType *d_arpt,
				   const idType *d_acol,
				   const valType *d_aval,
				   const idType* __restrict__ d_brpt,
				   const idType* __restrict__ d_bcol,
				   const valType* __restrict__ d_bval,
				   idType *d_crpt,
				   idType *d_ccol,
				   valType *d_cval,
				   const idType *d_permutation,
				   idType *d_nz,
				   idType bin_offset,
				   idType bin_size)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;
    idType rid = i / PWARP;
    idType tid = i % PWARP;
    idType local_rid = rid % (blockDim.x / PWARP);
    idType j;

    __shared__ idType id_table[SHARED_N_P];
    __shared__ valType value_table[SHARED_N_P];
  
    idType soffset = local_rid * (TS_N_P);
  
    for (j = tid; j < TS_N_P; j += PWARP) {
        id_table[soffset + j] = -1;
        value_table[soffset + j] = 0;
    }
  
    if (rid >= bin_size) {
        return;
    }
    rid = d_permutation[rid + bin_offset];
  
    if (tid == 0) {
        d_nz[rid] = 0;
    }

    idType k;
    idType acol, bcol, hash, key, adr;
    idType offset = d_crpt[rid];
    idType old, index;
    valType aval, bval;

    for (j = d_arpt[rid] + tid; j < d_arpt[rid + 1]; j += PWARP) {
        // acol = ld_gbl_col(d_acol + j); // @OGUZ-BEWARE @OGUZ-TODO
        // aval = ld_gbl_val(d_aval + j); // @OGUZ-BEWARE @OGUZ-TODO
		acol = d_acol[j];
		aval = d_aval[j];
        for (k = d_brpt[acol]; k < d_brpt[acol + 1]; k++) {
            bcol = d_bcol[k];
            bval = d_bval[k];
            
            key = bcol;
            hash = (bcol * HASH_SCAL) & ((TS_N_P) - 1);
            adr = soffset + hash;
            while (1) {
                if (id_table[adr] == key) {
                    atomicAdd(value_table + adr, aval * bval);
                    break;
                }
                else if (id_table[adr] == -1) {
                    old = atomicCAS(id_table + adr, -1, key);
                    if (old == -1) {
                        atomicAdd(value_table + adr, aval * bval);
                        break;
                    }
                }
                else {
                    hash = (hash + 1) & ((TS_N_P) - 1);
                    adr = soffset + hash;
                }
            }
        }
    }

    __syncthreads();
    
    for (j = tid; j < (TS_N_P); j += PWARP) {
        if (id_table[soffset + j] != -1) {
            index = atomicAdd(d_nz + rid, 1);
            id_table[soffset + index] = id_table[soffset + j];
            value_table[soffset + index] = value_table[soffset + j];
        }
    }
    
    __syncthreads();
    
    idType nz = d_nz[rid];
    if (sort) {
        // Sorting for shared data
        idType count, target;
        for (j = tid; j < nz; j += PWARP) {
            target = id_table[soffset + j];
            count = 0;
            for (k = 0; k < nz; k++) {
                count += (unsigned int)(id_table[soffset + k] - target) >> 31;
            }
            d_ccol[offset + count] = id_table[soffset + j];
            d_cval[offset + count] = value_table[soffset + j];
        }
    }
    else {
        // No sorting
        for (j = tid; j < nz; j += PWARP) {
            d_ccol[offset + j] = id_table[soffset + j];
            d_cval[offset + j] = value_table[soffset + j];
        }
    }
}



template <class idType,
		  class valType,
		  int SH_ROW,
		  bool sort>
__global__
void
hash_numeric_tb(const idType *d_arpt,
				const idType *d_acolids,
				const valType *d_avalues,
				const idType* __restrict__ d_brpt,
				const idType* __restrict__ d_bcolids,
				const valType* __restrict__ d_bvalues,
				idType *d_crpt,
				idType *d_ccolids,
				valType *d_cvalues,
				const idType *d_permutation,
				idType *d_nz,
				idType bin_offset,
				idType bin_size)
{
    idType rid = blockIdx.x;
    idType tid = threadIdx.x & (nsp_warp_sz - 1);
    idType wid = threadIdx.x / nsp_warp_sz;
    idType wnum = blockDim.x / nsp_warp_sz;
    idType j;
    __shared__ idType id_table[SH_ROW];
    __shared__ valType value_table[SH_ROW];
  
    for (j = threadIdx.x; j < SH_ROW; j += blockDim.x) {
        id_table[j] = -1;
        value_table[j] = 0;
    }
  
    if (rid >= bin_size) {
        return;
    }

    rid = d_permutation[rid + bin_offset];

    if (threadIdx.x == 0) {
        d_nz[rid] = 0;
    }
    __syncthreads();

    idType acolids;
    idType k;
    idType bcolids, hash, key;
    idType offset = d_crpt[rid];
    idType old, index;
    valType avalues, bvalues;

    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum) {
        // acolids = ld_gbl_col(d_acolids + j); // @OGUZ-BEWARE @OGUZ-TODO
        // avalues = ld_gbl_val(d_avalues + j); // @OGUZ-BEWARE @OGUZ-TODO
		acolids = d_acolids[j];
		avalues = d_avalues[j];
        for (k = d_brpt[acolids] + tid; k < d_brpt[acolids + 1]; k += nsp_warp_sz) {
            bcolids = d_bcolids[k];
            bvalues = d_bvalues[k];
	
            key = bcolids;
            hash = (bcolids * HASH_SCAL) & (SH_ROW - 1);
            while (1) {
                if (id_table[hash] == key) {
                    atomicAdd(value_table + hash, avalues * bvalues);
                    break;
                }
                else if (id_table[hash] == -1) {
                    old = atomicCAS(id_table + hash, -1, key);
                    if (old == -1) {
                        atomicAdd(value_table + hash, avalues * bvalues);
                        break;
                    }
                }
                else {
                    hash = (hash + 1) & (SH_ROW - 1);
                }
            }
        }
    }
  
    __syncthreads();
    if (threadIdx.x < nsp_warp_sz) {
        for (j = tid; j < SH_ROW; j += nsp_warp_sz) {
            if (id_table[j] != -1) {
                index = atomicAdd(d_nz + rid, 1);
                id_table[index] = id_table[j];
                value_table[index] = value_table[j];
            }
        }
    }
    __syncthreads();
    idType nz = d_nz[rid];
    if (sort) {
        /* Sorting for shared data */
        idType count, target;
        for (j = threadIdx.x; j < nz; j += blockDim.x) {
            target = id_table[j];
            count = 0;
            for (k = 0; k < nz; k++) {
                count += (unsigned int)(id_table[k] - target) >> 31;
            }
            d_ccolids[offset + count] = id_table[j];
            d_cvalues[offset + count] = value_table[j];
        }
    }
    else {
        /* No Sorting */
        for (j = threadIdx.x; j < nz; j += blockDim.x) {
            d_ccolids[offset + j] = id_table[j];
            d_cvalues[offset + j] = value_table[j];
        }
    }
}



#ifdef ORIGINAL_HASH
template <class idType,
		  class valType,
		  bool sort>
__global__
void
hash_numeric_gl(const idType *d_arpt,
				const idType *d_acolids,
				const valType *d_avalues,
				const idType* __restrict__ d_brpt,
				const idType* __restrict__ d_bcolids,
				const valType* __restrict__ d_bvalues,
				idType *d_crpt,
				idType *d_ccolids,
				valType *d_cvalues,
				const idType *d_permutation,
				idType *d_nz,
				idType *d_id_table,
				valType *d_value_table,
				idType max_row_nz,
				idType bin_offset,
				idType M)
{
    idType rid = blockIdx.x;
    idType tid = threadIdx.x & (nsp_warp_sz - 1);
    idType wid = threadIdx.x / nsp_warp_sz;
    idType wnum = blockDim.x / nsp_warp_sz;
    idType j;
  
    if (rid >= M) {
        return;
    }

    idType doffset = rid * max_row_nz;

    rid = d_permutation[rid + bin_offset];
  
    if (threadIdx.x == 0) {
        d_nz[rid] = 0;
    }
    __syncthreads();

    idType acolids;
    idType k;
    idType bcolids, hash, key, adr;
    idType offset = d_crpt[rid];
    idType old, index;
    valType avalues, bvalues;

    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum) {
        // acolids = ld_gbl_col(d_acolids + j); // @OGUZ-BEWARE @OGUZ-TODO
        // avalues = ld_gbl_val(d_avalues + j); // @OGUZ-BEWARE @OGUZ-TODO
		acolids = d_acolids[j];
		avalues = d_avalues[j];
        for (k = d_brpt[acolids] + tid; k < d_brpt[acolids + 1]; k += nsp_warp_sz) {
            bcolids = d_bcolids[k];
            bvalues = d_bvalues[k];
      
            key = bcolids;
            hash = (bcolids * HASH_SCAL) % max_row_nz;
            adr = doffset + hash;
            while (1) {
                if (d_id_table[adr] == key) {
                    atomicAdd(d_value_table + adr, avalues * bvalues);
                    break;
                }
                else if (d_id_table[adr] == -1) {
                    old = atomicCAS(d_id_table + adr, -1, key);
                    if (old == -1) {
                        atomicAdd(d_value_table + adr, avalues * bvalues);
                        break;
                    }
                }
                else {
                    hash = (hash + 1) % max_row_nz;
                    adr = doffset + hash;
                }
            }
        }
    }
  
    __syncthreads();
    if (threadIdx.x < nsp_warp_sz) {
        for (j = tid; j < max_row_nz; j += nsp_warp_sz) {
            if (d_id_table[doffset + j] != -1) {
                index = atomicAdd(d_nz + rid, 1);
                d_id_table[doffset + index] = d_id_table[doffset + j];
                d_value_table[doffset + index] = d_value_table[doffset + j];
            }
        }
    }
    __syncthreads();
    idType nz = d_nz[rid];
    if (sort) {
        /* Sorting for shared data */
        idType count, target;
        for (j = threadIdx.x; j < nz; j += blockDim.x) {
            target = d_id_table[doffset + j];
            count = 0;
            for (k = 0; k < nz; k++) {
                count += (unsigned int)(d_id_table[doffset + k] - target) >> 31;
            }
            d_ccolids[offset + count] = d_id_table[doffset + j];
            d_cvalues[offset + count] = d_value_table[doffset + j];
        }
    }
    else {
        /* No sorting */
        for (j = threadIdx.x; j < nz; j += blockDim.x) {
            d_ccolids[offset + j] = d_id_table[doffset + j];
            d_cvalues[offset + j] = d_value_table[doffset + j];
        }
    }
}



#else
template <class idType,
		  class valType,
		  bool sort>
__global__
void
hash_numeric_gl(const idType *d_arpt,
				const idType *d_acolids,
				const valType *d_avalues,
				const idType* __restrict__ d_brpt,
				const idType* __restrict__ d_bcolids,
				const valType* __restrict__ d_bvalues,
				idType *d_crpt,
				idType *d_ccolids,
				valType *d_cvalues,
				const idType *d_permutation,
				idType *d_nz,
				idType *d_id_table,
				valType *d_value_table,
				idType max_row_nz,
				idType bin_offset,
				idType M)
{
    idType rid = blockIdx.x;
    idType tid = threadIdx.x & (nsp_warp_sz - 1);
    idType wid = threadIdx.x / nsp_warp_sz;
    idType wnum = blockDim.x / nsp_warp_sz;
    idType j;
    idType conc_row_num = gridDim.x;
    idType target;
  
    if (rid >= M) {
        return;
    }

    idType doffset = rid * max_row_nz;
    
    for (; rid < M; rid += conc_row_num) {
        target = d_permutation[rid + bin_offset];
  
        for (j = threadIdx.x; j < max_row_nz; j += blockDim.x) {
            d_id_table[doffset + j] = -1;
            d_value_table[doffset + j] = 0;
        }
        if (threadIdx.x == 0) {
            d_nz[target] = 0;
        }
        __syncthreads();

        idType acolids;
        idType k;
        idType bcolids, hash, key, adr;
        idType offset = d_crpt[target];
        idType old, index;
        valType avalues, bvalues;

        for (j = d_arpt[target] + wid; j < d_arpt[target + 1]; j += wnum) {
            // acolids = ld_gbl_col(d_acolids + j); // @OGUZ-BEWARE @OGUZ-TODO
            // avalues = ld_gbl_val(d_avalues + j); // @OGUZ-BEWARE @OGUZ-TODO
			acolids = d_acolids[j];
			avalues = d_avalues[j];
            for (k = d_brpt[acolids] + tid; k < d_brpt[acolids + 1]; k += nsp_warp_sz) {
                bcolids = d_bcolids[k];
                bvalues = d_bvalues[k];
      
                key = bcolids;
                hash = (bcolids * HASH_SCAL) % max_row_nz;
                adr = doffset + hash;
                while (1) {
                    if (d_id_table[adr] == key) {
                        atomicAdd(d_value_table + adr, avalues * bvalues);
                        break;
                    }
                    else if (d_id_table[adr] == -1) {
                        old = atomicCAS(d_id_table + adr, -1, key);
                        if (old == -1) {
                            atomicAdd(d_value_table + adr, avalues * bvalues);
                            break;
                        }
                    }
                    else {
                        hash = (hash + 1) % max_row_nz;
                        adr = doffset + hash;
                    }
                }
            }
        }
  
        __syncthreads();
        if (threadIdx.x < nsp_warp_sz) {
            for (j = tid; j < max_row_nz; j += nsp_warp_sz) {
                if (d_id_table[doffset + j] != -1) {
                    index = atomicAdd(d_nz + target, 1);
                    d_id_table[doffset + index] = d_id_table[doffset + j];
                    d_value_table[doffset + index] = d_value_table[doffset + j];
                }
            }
        }
        __syncthreads();
        idType nz = d_nz[target];
        if (sort) {
            /* Sorting for shared data */
            idType count, target;
            for (j = threadIdx.x; j < nz; j += blockDim.x) {
                target = d_id_table[doffset + j];
                count = 0;
                for (k = 0; k < nz; k++) {
                    count += (unsigned int)(d_id_table[doffset + k] - target) >> 31;
                }
                d_ccolids[offset + count] = d_id_table[doffset + j];
                d_cvalues[offset + count] = d_value_table[doffset + j];
            }
        }
        else {
            /* No sorting */
            for (j = threadIdx.x; j < nz; j += blockDim.x) {
                d_ccolids[offset + j] = d_id_table[doffset + j];
                d_cvalues[offset + j] = d_value_table[doffset + j];
            }
        }
    }
}
#endif


template <class idType,
		  class valType,
		  bool sort>
void
hash_numeric(CSR<idType, valType> a,
			 CSR<idType, valType> b,
			 CSR<idType, valType> &c,
			 BIN<idType, BIN_NUM> &bin)
{
    idType i;
    idType GS, BS;
    for (i = BIN_NUM - 1; i >= 0; i--)
	{
		// std::cout << "bin " << i << " size " << bin.bin_size[i] << std::endl;
        if (bin.bin_size[i] > 0)
		{
            switch (i) {
            case 0:
				// std::cout << "case 0 calling hash_numeric_pwarp" << std::endl;				
				BS = 512;
                GS = div_round_up(bin.bin_size[i] * PWARP, BS);
				hipLaunchKernelGGL(HIP_KERNEL_NAME(hash_numeric_pwarp<idType, valType, sort>), dim3(GS), dim3(BS), 0, bin.stream[i],
								   a.d_rpt, a.d_colids, a.d_values, b.d_rpt, b.d_colids, b.d_values,
								   c.d_rpt, c.d_colids, c.d_values, bin.d_permutation,
								   bin.d_count, bin.bin_offset[i], bin.bin_size[i]);
                break;
            case 1:
				// std::cout << "case 1 calling hash_numeric_tb-256" << std::endl;
                BS = 64;
                GS = bin.bin_size[i];
                hipLaunchKernelGGL(HIP_KERNEL_NAME(hash_numeric_tb<idType, valType, 256, sort>), dim3(GS), dim3(BS), 0, bin.stream[i],
								   a.d_rpt, a.d_colids, a.d_values, b.d_rpt, b.d_colids, b.d_values,
								   c.d_rpt, c.d_colids, c.d_values, bin.d_permutation,
								   bin.d_count, bin.bin_offset[i], bin.bin_size[i]);
                break;
            case 2:
				// std::cout << "case 2 calling hash_numeric_tb-512" << std::endl;
                BS = 128;
                GS = bin.bin_size[i];
                hipLaunchKernelGGL(HIP_KERNEL_NAME(hash_numeric_tb<idType, valType, 512, sort>), dim3(GS), dim3(BS), 0, bin.stream[i],
								   a.d_rpt, a.d_colids, a.d_values, b.d_rpt, b.d_colids, b.d_values,
								   c.d_rpt, c.d_colids, c.d_values, bin.d_permutation,
								   bin.d_count, bin.bin_offset[i], bin.bin_size[i]);
                break;
            case 3:
				// std::cout << "case 3 calling hash_numeric_tb-1024" << std::endl;
                BS = 256;
                GS = bin.bin_size[i];
                hipLaunchKernelGGL(HIP_KERNEL_NAME(hash_numeric_tb<idType, valType, 1024, sort>), dim3(GS), dim3(BS), 0, bin.stream[i],
								   a.d_rpt, a.d_colids, a.d_values, b.d_rpt, b.d_colids, b.d_values,
								   c.d_rpt, c.d_colids, c.d_values, bin.d_permutation,
								   bin.d_count, bin.bin_offset[i], bin.bin_size[i]);
                break;
            case 4:
				// std::cout << "case 4 calling hash_numeric_tb-2048" << std::endl;
                BS = 512;
                GS = bin.bin_size[i];
                hipLaunchKernelGGL(HIP_KERNEL_NAME(hash_numeric_tb<idType, valType, 2048, sort>), dim3(GS), dim3(BS), 0, bin.stream[i],
								   a.d_rpt, a.d_colids, a.d_values, b.d_rpt, b.d_colids, b.d_values,
								   c.d_rpt, c.d_colids, c.d_values, bin.d_permutation,
								   bin.d_count, bin.bin_offset[i], bin.bin_size[i]);
                break;
            case 5:
				// std::cout << "case 5 calling hash_numeric_tb-4096" << std::endl;
                BS = 1024;
                GS = bin.bin_size[i];
                hipLaunchKernelGGL(HIP_KERNEL_NAME(hash_numeric_tb<idType, valType, 4096, sort>), dim3(GS), dim3(BS), 0, bin.stream[i],
								   a.d_rpt, a.d_colids, a.d_values, b.d_rpt, b.d_colids, b.d_values,
								   c.d_rpt, c.d_colids, c.d_values, bin.d_permutation,
								   bin.d_count, bin.bin_offset[i], bin.bin_size[i]);
                break;
            case 6 :
                {
					// std::cout << "case 6 calling hash_numeric_gl" << std::endl;
#ifdef ORIGINAL_HASH
                    idType max_row_nz = bin.max_nz * 2;
                    idType table_size = max_row_nz * bin.bin_size[i];
#else
                    idType max_row_nz = 8192;
                    while (max_row_nz <= bin.max_nz) {
                        max_row_nz *= 2;
                    }
                    idType conc_row_num = min(56 * 2, bin.bin_size[i]);
                    idType table_size = max_row_nz * conc_row_num;
#endif
                    idType *d_id_table;
                    valType *d_value_table;
                    HIPCHECK(hipMalloc((void **)&(d_id_table), sizeof(idType) * table_size));
                    HIPCHECK(hipMalloc((void **)&(d_value_table), sizeof(valType) * table_size));
                    BS = 1024;
#ifdef ORIGINAL_HASH
                    GS = div_round_up(table_size, BS);
                    hipLaunchKernelGGL(init_id_table, dim3(GS), dim3(BS), 0, bin.stream[i], d_id_table, table_size);
                    hipLaunchKernelGGL(init_value_table, dim3(GS), dim3(BS), 0, bin.stream[i], d_value_table, table_size);
                    GS = bin.bin_size[i];
#else
                    GS = conc_row_num;
#endif
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(hash_numeric_gl<idType, valType, sort>), dim3(GS), dim3(BS), 0, bin.stream[i],
									   a.d_rpt, a.d_colids, a.d_values, b.d_rpt, b.d_colids, b.d_values,
									   c.d_rpt, c.d_colids, c.d_values, bin.d_permutation,
									   bin.d_count, d_id_table, d_value_table, max_row_nz, bin.bin_offset[i], bin.bin_size[i]);
                    hipFree(d_id_table);
                    hipFree(d_value_table);
                }
                break;
            }
        }
    }
    hipDeviceSynchronize();
}



template <bool sort,
		  class idType,
		  class valType>
void
SpGEMM_Hash(CSR<idType, valType> a,
			CSR<idType, valType> b,
			CSR<idType, valType> &c)
{
    hipEvent_t event[2];
    float msec;
    for (int i = 0; i < 2; i++) {
        hipEventCreate(&(event[i]));
    }

    BIN<idType, BIN_NUM> bin(a.nrow);

	// hipEventRecord(event[0], 0);
    c.nrow = a.nrow;
    c.ncolumn = b.ncolumn;
    c.device_malloc = true;
    hipMalloc((void **)&(c.d_rpt), sizeof(idType) * (c.nrow + 1));
	bin.set_max_bin(a.d_rpt, a.d_colids, b.d_rpt, a.nrow, TS_S_P, TS_S_T);
	// hipEventRecord(event[1], 0);
	// hipDeviceSynchronize();
    // hipEventElapsedTime(&msec, event[0], event[1]);
	// cout << "before HashSymbolic: " << msec << endl;
    
    // hipEventRecord(event[0], 0);
    hash_symbolic(a, b, c, bin);
    // hipEventRecord(event[1], 0);
    // hipDeviceSynchronize();
    // hipEventElapsedTime(&msec, event[0], event[1]);

	// std::cout << "symbolic finish c.nnz = " << c.nnz << std::endl;
    // cout << "HashSymbolic: " << msec << endl;

	// hipEventRecord(event[0], 0);
	hipMalloc((void **)&(c.d_colids), sizeof(idType) * (c.nnz));
    hipMalloc((void **)&(c.d_values), sizeof(valType) * (c.nnz));
    bin.set_min_bin(a.nrow, TS_N_P, TS_N_T);
	// hipEventRecord(event[1], 0);
	// hipDeviceSynchronize();
    // hipEventElapsedTime(&msec, event[0], event[1]);
	// cout << "in-between: " << msec << endl;

    hipEventRecord(event[0], 0);
    hash_numeric<idType, valType, sort>(a, b, c, bin);
    hipEventRecord(event[1], 0);
    hipDeviceSynchronize();
    hipEventElapsedTime(&msec, event[0], event[1]);

	// std::cout << "HashNumeric: " << msec << endl;
	// long long int flop_count;
	// get_spgemm_flop(a, b, flop_count);
	// printf("total flops %lld\n", flop_count);
	// float flops = (float)(flop_count) / 1000 / 1000 / msec;
    // printf("SpGEMM using CSR format (Hash, only numeric phase): %f[GFLOPS], %f[ms]\n", flops, msec);
}



template <class idType,
		  class valType>
void
SpGEMM_Hash(CSR<idType, valType> a,
			CSR<idType, valType> b,
			CSR<idType, valType> &c)
{
    SpGEMM_Hash<true, idType, valType>(a, b, c);
}



template <bool sort,
		  class idType,
		  class valType>
void
SpGEMM_Hash_Numeric(CSR<idType, valType> a,
					CSR<idType, valType> b,
					CSR<idType, valType> &c,
					float &msec)
{
    hipEvent_t event[2];
    // float msec;
    for (int i = 0; i < 2; i++) {
        hipEventCreate(&(event[i]));
    }

    BIN<idType, BIN_NUM> bin(a.nrow);

    bin.set_min_bin(c.d_rpt, a.nrow, TS_N_P, TS_N_T);

	hipEventRecord(event[0], 0);
    hash_numeric<idType, valType, sort>(a, b, c, bin);
	hipEventRecord(event[1], 0);
	hipDeviceSynchronize();
	hipEventElapsedTime(&msec, event[0], event[1]);

	// cout << "hash_numeric took " << msec << " msec\n";
}



template <class idType,
		  class valType>
void
SpGEMM_Hash_Numeric(CSR<idType, valType> a,
					CSR<idType, valType> b,
					CSR<idType, valType> &c,
					float &msec)
{
    SpGEMM_Hash_Numeric<true, idType, valType>(a, b, c, msec);
}



#endif
