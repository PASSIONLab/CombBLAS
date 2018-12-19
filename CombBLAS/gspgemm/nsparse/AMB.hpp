#include <iostream>
#include <string>
#include <cuda.h>

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include <nsparse.hpp>
#include <CSR.hpp>
#include <Plan.hpp>
#include <nsparse_asm.hpp>

using namespace std;

#define AT

template <class idType, class compIdType, class valType>
class AMB
{
public:
    AMB():M(1),N(1),chunk(warp),pad_M(chunk),SIGMA(1)
    {
    }
    ~AMB()
    {
    }
    void release_cpu_amb()
    {
    }
    void release_amb()
    {
        cudaFree(d_cs);
        cudaFree(d_cl);
        cudaFree(d_coffset);
        cudaFree(d_sellcs_col);
        cudaFree(d_sellcs_val);
        cudaFree(d_s_write_permutation);
        cudaFree(d_s_write_permutation_offset);
        
    }
    void evaluate_spmv(valType *d_x, valType *d_y, Plan<idType> &plan);
    void convert_amb_at(CSR<idType, valType> csr_mat, valType *d_x, valType *d_y, Plan<idType> &plan);
    void convert_from_csr(CSR<idType, valType> mat, Plan<idType> &plan, valType *d_x);
    void spmv(const valType *d_x, valType *d_y, const Plan<idType> &plan);
    
    idType *d_cs;
    compIdType *d_cl;
    compIdType *d_coffset;
    compIdType *d_sellcs_col;
    valType *d_sellcs_val;
    compIdType *d_s_write_permutation;
    compIdType *d_s_write_permutation_offset;
    
    idType block_size;
    idType nnz;
    idType M;
    idType N;
    idType pad_M;

    int chunk;
    int SIGMA;
    int group_num_col;
    idType nnz_max;
    idType c_size;
    idType seg_size;
    idType seg_num;
};

template <class T, class idType>
__global__ void zero_fill(T *d_array, idType size)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size) {
        return;
    }

    d_array[i] = 0;
  
}

template <class idType>
__global__ void init_permutation(idType *d_permutation, idType size)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size) {
        return;
    }
  
    d_permutation[i] = i;
}

template <class idType>
__global__ void set_segmented_nnz_num(idType *d_rpt, idType *d_col, idType *d_nnz_num,
                                      idType *d_group_seg, idType *d_offset,
                                      idType seg_size, idType seg_num,
                                      idType M, idType pad_M, int group_num_col)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (i >= M) {
        return;
    }

    idType width = d_rpt[i + 1] - d_rpt[i];

    idType g, j;
    idType col;

    idType offset = d_rpt[i];
    idType index;

    for (j = 0; j < width; j++) {
        index = offset + j;
        col = d_col[index];
        g = col / seg_size;
        d_offset[index] = d_nnz_num[g * pad_M + i];
        d_nnz_num[g * pad_M + i]++;
        d_group_seg[index] = g;
    }
}

template <class idType>
__global__ void init_segmented_rpt(idType *d_nnz_num, idType *d_seg_rpt, idType total_pad_row_num)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > total_pad_row_num) {
        return;
    }

    if (i == 0) {
        d_seg_rpt[i] = 0;
    }

    else {
        d_seg_rpt[i] = d_nnz_num[i - 1];
    }
}

template <class idType, class valType>
__global__ void set_segmented_col_val(idType *d_rpt, idType *d_col, valType *d_val,
                                      idType *d_seg_rpt, idType *d_seg_col, valType *d_seg_val,
                                      idType *d_group_seg, idType *d_offset,
                                      idType M, idType pad_M)
{
    idType i = blockIdx.x;

    if (i >= M) {
        return;
    }
  
    idType width = d_rpt[i + 1] - d_rpt[i];
  
    idType j = threadIdx.x;
    idType bs = blockDim.x;
    idType index;
    for (; j < width; j += bs) {
        index = d_rpt[i] + j;
        d_seg_col[d_seg_rpt[d_group_seg[index] * pad_M + i] + d_offset[index]] = d_col[index];
        d_seg_val[d_seg_rpt[d_group_seg[index] * pad_M + i] + d_offset[index]] = d_val[index];
    }
}

template <class idType, class valType>
void convert_segmented_csr(CSR<idType, valType> csr_mat,
                           idType *d_nnz_num, idType *d_seg_rpt, idType *d_seg_col, valType *d_seg_val,
                           idType seg_size, idType seg_num,
                           idType M, idType pad_M, idType group_num_col)
{
    size_t GS, BS;
  
    idType nz;
    idType total_pad_row_num;
    idType *d_group_seg, *d_offset;
  
    nz = csr_mat.nnz;
    total_pad_row_num = pad_M * group_num_col;
    
    checkCudaErrors(cudaMalloc((void **)&d_group_seg, sizeof(idType) * nz));
    checkCudaErrors(cudaMalloc((void **)&d_offset, sizeof(idType) * nz));
  
    BS = MAX_LOCAL_THREAD_NUM;
    GS = div_round_up(total_pad_row_num, BS);
    zero_fill<<<GS, BS>>>(d_nnz_num, total_pad_row_num);
  
    BS = warp;
    GS = div_round_up(M, BS);
    set_segmented_nnz_num<<<GS, BS>>>(csr_mat.d_rpt, csr_mat.d_colids, d_nnz_num, d_group_seg, d_offset, seg_size, seg_num, M, pad_M, group_num_col);
  
    /*Set segmented rpt*/
    GS = div_round_up((pad_M * group_num_col + 1), BS);
    init_segmented_rpt<<<GS, BS>>>(d_nnz_num, d_seg_rpt, pad_M * group_num_col);
    thrust::inclusive_scan(thrust::device, d_seg_rpt, d_seg_rpt + pad_M * group_num_col + 1, d_seg_rpt);
  
    /*Set segmented col and val*/
    GS = M;
    set_segmented_col_val<<<GS, BS>>>(csr_mat.d_rpt, csr_mat.d_colids, csr_mat.d_values, d_seg_rpt, d_seg_col, d_seg_val, d_group_seg, d_offset, M, pad_M);

    cudaThreadSynchronize();

    checkCudaErrors(cudaFree(d_offset));
    checkCudaErrors(cudaFree(d_group_seg));

}

template <class idType>
__global__ void set_d_check_nnz(idType *d_check_nnz, idType *d_nnz_num,
                                idType pad_M, int SIGMA, idType sigma_block_row)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pad_M) {
        return;
    }

    idType a = 1;
    if (d_nnz_num[blockIdx.y * pad_M + i] > 0) {
        atomicAdd(&(d_check_nnz[blockIdx.y * sigma_block_row + i / SIGMA]), a);
    }
}

template <class idType>
void set_check_nnz(idType *d_check_nnz, idType *d_nnz_num,
                   idType sigma_block, idType pad_M, int SIGMA, int group_num_col)
{
    size_t GS, BS;
    BS = MAX_LOCAL_THREAD_NUM;
    GS = div_round_up(sigma_block, BS);
  
    zero_fill<<<GS, BS>>>(d_check_nnz, sigma_block);

    GS = div_round_up(pad_M, BS);
    set_d_check_nnz<<<dim3(GS, group_num_col), dim3(BS, 1)>>>(d_check_nnz, d_nnz_num, pad_M, SIGMA, div_round_up(pad_M, SIGMA));
  
}

template <class idType>
__global__ void set_cl(idType *nnz_num, idType *cl, int chunk, idType c_size)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c_size) {
        return;
    }

    idType offset = chunk * i;
    idType max = 0;
    idType j, length;
    for (j = 0; j < chunk; j++) {
        length = nnz_num[offset + j];
        if (length > max) {
            max = length;
        }
    }
    cl[i] = max;
}

template <class idType>
__global__ void init_cs(idType *d_cl, idType *d_cs, idType c_size, int chunk)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c_size) {
        return;
    }

    if (i == 0) {
        d_cs[i] = 0;
    }
    else {
        d_cs[i] = d_cl[i - 1] * chunk;
    }

}

template <class idType>
void set_sellcs_chunk(idType *d_nnz_num, idType *d_cl, idType *d_cs, idType *elements_num, idType total_pad_row_num, int chunk)
{
    size_t GS, BS;
    idType r_size, c_size;
  
    c_size = total_pad_row_num / chunk;
  
    BS = MAX_LOCAL_THREAD_NUM;
    GS = div_round_up(c_size, BS);

    set_cl<<<GS, BS>>>(d_nnz_num, d_cl, chunk, c_size);
    init_cs<<<GS, BS>>>(d_cl, d_cs, c_size, chunk);
    thrust::inclusive_scan(thrust::device, d_cs, d_cs + c_size, d_cs);

    /*Get elements_num*/
    checkCudaErrors(cudaMemcpy(elements_num, d_cs + (c_size - 1), sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&r_size, d_cl + (c_size - 1), sizeof(int), cudaMemcpyDeviceToHost));

    *elements_num += r_size * chunk;
}

template <class idType, class valType>
__global__ void set_segmented_sellcs_col_val(idType *d_rpt, idType *d_col, valType *d_val, idType *d_nnz_num, idType *d_write_permutation, idType *d_sellcs_col, valType *d_sellcs_val, idType *d_cl, idType *d_cs, idType group_num_col, idType pad_M, idType chunk)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pad_M * group_num_col) {
        return;
    }

    idType bid = i / chunk;
    idType tid = i % chunk;
  
    idType j;
    idType width = d_cl[bid];
    idType nnz_width = d_nnz_num[i];
    
    for (j = 0; j < width; j++) {
        if (j < nnz_width) {
            d_sellcs_val[d_cs[bid] + tid + j * chunk] = d_val[d_rpt[d_write_permutation[i]] + j];
            d_sellcs_col[d_cs[bid] + tid + j * chunk] = d_col[d_rpt[d_write_permutation[i]] + j];

        }
        else {
            d_sellcs_val[d_cs[bid] + tid + j * chunk] = 0;
            d_sellcs_col[d_cs[bid] + tid + j * chunk] = d_col[d_rpt[d_write_permutation[bid * chunk]] + j];
        }
    }
}

template <class idType>
__global__ void get_c_size(idType *d_c_size, idType *d_full_cl, idType full_c_size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= full_c_size) {
        return;
    }

    if (d_full_cl[i] != 0) {
        atomicAdd(d_c_size, 1);
    }
}

template <class idType, class compIdType>
__global__ void set_ushort_col(compIdType *d_us_sellcs_col, idType *d_sellcs_col,
                               idType *d_cs, idType *d_cl, bool *d_is_empty,
                               int group_num_col, idType pad_M, int chunk, idType seg_size)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= group_num_col * pad_M) {
        return;
    }

    idType offset = i % chunk;
    idType chunk_id = i / chunk;
    idType width = d_cl[chunk_id];
    
    idType j;
    idType adr;
    for (j = 0; j < width; j++) {
        adr = d_cs[chunk_id] + j * chunk + offset;
        d_us_sellcs_col[adr] = (compIdType)(d_sellcs_col[adr] % seg_size);
    }

    if (offset == 0) {
        idType c_offset = 0;
        if (width > 0) {
            c_offset = (d_sellcs_col[d_cs[chunk_id]] / seg_size) << SCL_BORDER;
            d_is_empty[chunk_id] = false;
            d_cl[chunk_id] = (width - 1) | c_offset;
        }
        else {
            d_is_empty[chunk_id] = true;
            d_cl[chunk_id] = 0;
        }
    }
}

template <class idType>
__global__ void init_gcs(idType *d_gcs, bool *d_is_empty, idType chunk_num)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= chunk_num) {
        return;
    }
  
    if (i == 0) d_gcs[i] = 0;

    if (d_is_empty[i] == true) d_gcs[i + 1] = 0;
    else d_gcs[i + 1] = 1;
}

template <class idType>
void set_gcs(idType *d_gcs, bool *d_is_empty, idType chunk_num)
{
    size_t GS, BS;
    BS = 256;
    GS = div_round_up(chunk_num, BS);
    init_gcs<<<GS, BS>>>(d_gcs, d_is_empty, chunk_num);
    thrust::inclusive_scan(thrust::device, d_gcs, d_gcs + chunk_num + 1, d_gcs);

}

template <class idType>
__global__ void set_packed_cl_cs(idType *d_packed_cl, idType *d_packed_cs,
                                 idType *d_cl, idType *d_cs, idType *d_gcs,
                                 idType chunk_num)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= chunk_num) {
        return;
    }

    if (d_gcs[i + 1] - d_gcs[i] > 0) {
        d_packed_cl[d_gcs[i]] = d_cl[i];
        d_packed_cs[d_gcs[i]] = d_cs[i];
    }
}

template <class idType>
__global__ void update_write_permutation(idType *write_permutation, idType *nnz_num, idType total_pad_row_num, idType pad_M, idType M)
{  
    idType i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= total_pad_row_num) {
        return;
    }

    write_permutation[i] -= (i / pad_M) * pad_M;
    if (write_permutation[i] >= M) {
        write_permutation[i] = M - 1;
    } 
}

template <class idType>
__global__ void compress_write_permutation(idType *d_write_permutation,
                                           idType *d_full_write_permutation,
                                           idType *d_gcs,
                                           idType total_pad_row_num, int chunk)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_pad_row_num) {
        return;
    }

    idType chunk_id = i / chunk;
    if (d_gcs[chunk_id + 1] - d_gcs[chunk_id] > 0) {
        idType tid = i % chunk;
        d_write_permutation[d_gcs[chunk_id] * chunk + tid] = d_full_write_permutation[i];
    }
}

template <class idType, class compIdType>
__global__ void compress_s_write_permutation(compIdType *d_s_write_permutation,
                                             compIdType *d_s_write_permutation_offset,
                                             idType *d_write_permutation,
                                             idType c_size, int chunk)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c_size * chunk) {
        return;
    }

    idType chunk_id = i / chunk;
    d_s_write_permutation[i] = (compIdType)(d_write_permutation[i] % USHORT_MAX);
    if (i % chunk == 0) {
        d_s_write_permutation_offset[chunk_id] = (compIdType)(d_write_permutation[i] / USHORT_MAX);
    }
}

template <class idType, class compIdType>
__global__ void set_blocked_cl(compIdType *d_blocked_cl, compIdType *d_blocked_coffset,
                               const idType *d_packed_cl, const idType *d_packed_cs,
                               const compIdType *d_s_col_ell,
                               const idType c_size, const int chunk, const int block_size)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c_size * chunk) {
        return;
    }

    idType k;
    idType cl_width = (d_packed_cl[i / chunk] & SCL_BIT) + 1;
    idType max_width = 0;
    idType j = i % chunk;
    idType width = 0;
    idType offset = d_packed_cs[i / chunk];
    idType base = d_s_col_ell[offset + j];
    for (k = 1; k < cl_width; k++) {
        if (d_s_col_ell[offset + j + k * chunk] - base >= block_size) {
            base = d_s_col_ell[offset + j + k * chunk];
            width += block_size;
        }
    }
    width += block_size;
  
    idType shfl_width;
    shfl_width = __shfl_xor(width, 16);
    width = (width < shfl_width)? shfl_width : width;
    shfl_width = __shfl_xor(width, 8);
    width = (width < shfl_width)? shfl_width : width;
    shfl_width = __shfl_xor(width, 4);
    width = (width < shfl_width)? shfl_width : width;
    shfl_width = __shfl_xor(width, 2);
    width = (width < shfl_width)? shfl_width : width;
    shfl_width = __shfl_xor(width, 1);
    width = (width < shfl_width)? shfl_width : width;

    if (j == 0) {
        max_width = width;
        d_blocked_cl[i / chunk] = (max_width / block_size) - 1;
        d_blocked_coffset[i / chunk] = (d_packed_cl[i / chunk] >> SCL_BORDER);
    }
}

template <class idType, class compIdType>
__global__ void init_blocked_cs(idType *d_blocked_cs, const compIdType *d_blocked_cl,
                                const idType c_size, const int chunk, const int block_size)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c_size) {
        return;
    }

    if (i == 0) {
        d_blocked_cs[i] = 0;
    }
    else {
        d_blocked_cs[i] = ((d_blocked_cl[i - 1]) + 1) * chunk * block_size;
    }
}

template <class idType, class compIdType>
void set_blocked_cl_cs(compIdType *d_blocked_cl, compIdType *d_blocked_coffset, idType *d_blocked_cs,
                       const idType *d_packed_cl, const idType *d_packed_cs,
                       const compIdType *d_s_col_ell,
                       const idType c_size, const int chunk,
                       const int block_size, idType *c_nnz)
{
    size_t GS, BS;
  
    BS = 256;
    GS = div_round_up((c_size * chunk), BS);

    set_blocked_cl<<<GS, BS>>>(d_blocked_cl, d_blocked_coffset, d_packed_cl, d_packed_cs, d_s_col_ell, c_size, chunk, block_size);

    GS = div_round_up(c_size, BS);
    init_blocked_cs<<<GS, BS>>>(d_blocked_cs, d_blocked_cl, c_size, chunk, block_size);

    thrust::inclusive_scan(thrust::device, d_blocked_cs, d_blocked_cs + c_size, d_blocked_cs);

    compIdType r_size;
    checkCudaErrors(cudaMemcpy(c_nnz, d_blocked_cs + (c_size - 1), sizeof(idType), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&r_size, d_blocked_cl + (c_size - 1), sizeof(compIdType), cudaMemcpyDeviceToHost));

    *c_nnz += (idType)(r_size + 1) * block_size * chunk;

}

template <class idType, class compIdType, class valType>
__global__ void set_blocked_col_val(compIdType *d_bs_col_ell, valType *d_b_val_ell,
                                    compIdType *d_blocked_cl, idType *d_blocked_cs,
                                    idType *d_packed_cl, idType *d_packed_cs,
                                    compIdType *d_s_col_ell, valType *d_val_ell,
                                    idType c_size, int chunk, int block_size)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= c_size * chunk) {
        return;
    }

    idType chunk_id = i / chunk;
    idType tid = i % chunk;
    idType cl_width = (d_blocked_cl[chunk_id] + 1) * block_size;

    idType base, h, k, c;
    idType it = 0;
    for (k = 0; k < cl_width / block_size; k++) {
        if (it < ((d_packed_cl[chunk_id] & SCL_BIT) + 1)) {
            c = d_s_col_ell[d_packed_cs[chunk_id] + tid + it * chunk];
            base = c;
            d_bs_col_ell[d_blocked_cs[chunk_id] / block_size + tid + k * chunk] = base;
            for (h = 0; h < c - base; h++) {
                d_b_val_ell[d_blocked_cs[chunk_id] + tid + (k * block_size + h) * chunk] = 0;
            }
            d_b_val_ell[d_blocked_cs[chunk_id] + tid + (k * block_size + (c - base)) * chunk] = d_val_ell[d_packed_cs[chunk_id] + tid + it * chunk];
            it++;
            for (h = c - base + 1; h < block_size; h++) {
                if (it < ((d_packed_cl[chunk_id] & SCL_BIT) + 1)) {
                    if (d_s_col_ell[d_packed_cs[chunk_id] + tid + it * chunk] - base == h) {
                        d_b_val_ell[d_blocked_cs[chunk_id] + tid + (k * block_size + h) * chunk] = d_val_ell[d_packed_cs[chunk_id] + tid + it * chunk];
                        it++;
                    }
                    else {
                        d_b_val_ell[d_blocked_cs[chunk_id] + tid + (k * block_size + h) * chunk] = 0;
                    }
                }
                else {
                    d_b_val_ell[d_blocked_cs[chunk_id] + tid + (k * block_size + h) * chunk] = 0;
                }
            }
        }
        else {
            d_bs_col_ell[d_blocked_cs[chunk_id] / block_size + tid + k * chunk] = (d_s_col_ell[d_packed_cs[chunk_id] + tid + (d_packed_cl[chunk_id] & SCL_BIT) * chunk] / block_size) * block_size;
            for (h = 0; h < block_size; h++) {
                d_b_val_ell[d_blocked_cs[chunk_id] + tid + (k * block_size + h) * chunk] = 0;
            }
        }
    }
}

template <class idType, class compIdType, class valType>
__global__ void adjust_blocked_col_val(compIdType *d_bs_col_ell, valType *d_b_val_ell,
                                       compIdType *d_blocked_cl, idType *d_blocked_cs, compIdType *d_coffset,
                                       idType c_size, int chunk, int block_size, idType N, idType seg_size)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= c_size * chunk) {
        return;
    }

    idType chunk_id = i / chunk;
    idType tid = i % chunk;
    idType width = d_blocked_cl[chunk_id] + 1;

    idType it, k, c, h;
    for (k = 0; k < width; k++) {
        c = d_bs_col_ell[d_blocked_cs[chunk_id] / block_size + tid + k * chunk] + (d_coffset[chunk_id] * seg_size);
        if (c + block_size > N) {
            d_bs_col_ell[d_blocked_cs[chunk_id] / block_size + tid + k * chunk] = (N - (d_coffset[chunk_id] * seg_size) - block_size);
            h = block_size - (N - c);
            for (it = block_size - h - 1; it >= 0; --it) {
                d_b_val_ell[d_blocked_cs[chunk_id] + tid + (k * block_size + h + it) * chunk] = d_b_val_ell[d_blocked_cs[chunk_id] + tid + (k * block_size + it) * chunk];
            }
            for (it = 0; it < h; ++it) {
                d_b_val_ell[d_blocked_cs[chunk_id] + tid + (k * block_size + it) * chunk] = 0;
            }
        }
    }
}

template <class idType, class compIdType, class valType>
void AMB<idType, compIdType, valType>::evaluate_spmv(valType *d_x, valType *d_y, Plan<idType> &plan)
{
    int i, coe;
    float msec, ave_msec, best_msec;
    Plan<idType> plan_;
    cudaEvent_t event[2];
    valType *y = (valType *)malloc(sizeof(valType) * M);
    for (i = 0; i < 2; i++) {
        cudaEventCreate(&(event[i]));
    }

    best_msec = sfFLT_MAX;
    for (coe = 2; coe <= MAX_THREAD_BLOCK; coe *= 2) {
        plan_.thread_block = warp * coe;
        plan_.thread_grid = div_round_up(chunk * c_size, plan_.thread_block);

        ave_msec = 0;
        for (i = 0; i < TEST_NUM; i++) {
            cudaEventRecord(event[0], 0);
            spmv(d_x, d_y, plan_);
            cudaEventRecord(event[1], 0);
            cudaThreadSynchronize();
	
            checkCudaErrors(cudaMemcpy(y, d_y, sizeof(valType) * M, cudaMemcpyDeviceToHost));

            cudaEventElapsedTime(&msec, event[0], event[1]);
	
            if (i > 0) {
                ave_msec += msec;
            }
        }
        ave_msec /= TEST_NUM - 1;
        if (plan.min_msec > ave_msec) {
            plan.min_msec = ave_msec;
            plan.seg_size = (seg_size);
            plan.block_size = block_size;
            plan.thread_block = plan_.thread_block;
            plan.thread_grid = plan_.thread_grid;
        }
        if (best_msec > ave_msec) {
            best_msec = ave_msec;
        }
    }
    cudaEventDestroy(event[0]);
    cudaEventDestroy(event[1]);
  
}

template <class idType, class compIdType, class valType>
void AMB<idType, compIdType, valType>::convert_amb_at(CSR<idType, valType> csr_mat, valType *d_x, valType *d_y, Plan<idType> &plan)
{
    size_t GS, BS;
    idType i;
  
    idType nz, c_nnz;
    idType total_pad_row_num, full_c_size;

    // int SIGMA, chunk;
    idType start, end;
    // int block_size;
    idType sigma_block;

    idType *d_nnz_num;
    idType *d_seg_rpt;
    idType *d_seg_col;
    valType *d_seg_val;
    idType *d_full_write_permutation;
    idType *d_write_permutation;
    idType *d_full_cs, *d_full_cl;
    bool *d_is_empty;
    idType *d_packed_cl, *d_packed_cs;

    idType *check_nnz;
    idType *d_check_nnz;
    idType *d_nb_sellcs_col;
    compIdType *d_nbs_sellcs_col;
    valType *d_nb_sellcs_val;
  
    idType *d_c_size;
    idType *d_gcs;

    nz = csr_mat.nnz;
    total_pad_row_num = pad_M * group_num_col;
    full_c_size = total_pad_row_num / chunk;
    BS = MAX_LOCAL_THREAD_NUM;

    /* Step 1 : Convert the matrix into Segmented Format */
    /* Convert format from CSR to Segmented CSR */
    checkCudaErrors(cudaMalloc((void **)&d_nnz_num, sizeof(idType) * total_pad_row_num));
    checkCudaErrors(cudaMalloc((void **)&d_seg_rpt, sizeof(idType) * (total_pad_row_num + 1)));
    checkCudaErrors(cudaMalloc((void **)&d_seg_col, sizeof(idType) * nz));
    checkCudaErrors(cudaMalloc((void **)&d_seg_val, sizeof(valType) * nz));
  
    convert_segmented_csr(csr_mat, d_nnz_num, d_seg_rpt, d_seg_col, d_seg_val, seg_size, seg_num, M, pad_M, group_num_col);
  
    /* Step 2 : Segmented CSR => Segmented SELL-C-sigma */
    /* Set permutation */
    checkCudaErrors(cudaMalloc((void **)&(d_full_write_permutation), sizeof(idType) * total_pad_row_num));
    
    GS = div_round_up(total_pad_row_num, BS);
    init_permutation<<<GS, BS>>>(d_full_write_permutation, total_pad_row_num);
    
    sigma_block = div_round_up(pad_M, SIGMA) * group_num_col;
    check_nnz = (idType *)malloc(sizeof(idType) * sigma_block);
    checkCudaErrors(cudaMalloc((void **)&(d_check_nnz), sizeof(idType) * sigma_block));
    set_check_nnz(d_check_nnz, d_nnz_num, sigma_block, pad_M, SIGMA, group_num_col);
    checkCudaErrors(cudaMemcpy(check_nnz, d_check_nnz, sizeof(idType) * sigma_block, cudaMemcpyDeviceToHost));
    
    /* Sorting each sigma rows */
    if (SIGMA > 1) {
        thrust::device_ptr<idType> dev_nnz_num(d_nnz_num);
        thrust::device_ptr<idType> dev_full_write_permutation(d_full_write_permutation);
        for (i = 0; i < group_num_col; i++) {
            start = 0;
            end = 0;
            while (start < M) {
                end += SIGMA;
                if (end >= M) {
                    end = M;
                }
                if (check_nnz[i * div_round_up(pad_M, SIGMA) + start / SIGMA] > 0) {
                    thrust::stable_sort_by_key(dev_nnz_num + i * pad_M + start,
                                               dev_nnz_num + i * pad_M + end,
                                               dev_full_write_permutation + i * pad_M + start,
                                               thrust::greater<int>());
                }
                start += SIGMA;
            }
        }
    }

    /* Set chunk size */
    checkCudaErrors(cudaMalloc((void **)&d_full_cl, sizeof(idType) * full_c_size));
    checkCudaErrors(cudaMalloc((void **)&d_full_cs, sizeof(idType) * full_c_size));
    set_sellcs_chunk(d_nnz_num, d_full_cl, d_full_cs, &nnz, total_pad_row_num, chunk);
    
    /* Set sellcs_col and sellcs_val */
    checkCudaErrors(cudaMalloc((void **)&(d_nb_sellcs_col), sizeof(idType) * nnz));
    checkCudaErrors(cudaMalloc((void **)&(d_nb_sellcs_val), sizeof(valType) * nnz));
  
    GS = div_round_up(total_pad_row_num, BS);
    set_segmented_sellcs_col_val<<<GS, BS>>>(d_seg_rpt, d_seg_col, d_seg_val, d_nnz_num, d_full_write_permutation, d_nb_sellcs_col, d_nb_sellcs_val, d_full_cl, d_full_cs, group_num_col, pad_M, chunk);

    cudaFree(d_seg_rpt);
    cudaFree(d_seg_col);
    cudaFree(d_seg_val);

    /* Step 3 : Compress column indices (eg. 32bit => 16bit)*/
    /* Compression */
    c_size = 0;
    checkCudaErrors(cudaMalloc((void **)&d_c_size, sizeof(idType)));
    checkCudaErrors(cudaMemcpy(d_c_size, &(c_size), sizeof(idType), cudaMemcpyHostToDevice));
    get_c_size<<<div_round_up(full_c_size, BS), BS>>>(d_c_size, d_full_cl, full_c_size);
    checkCudaErrors(cudaMemcpy(&(c_size), d_c_size, sizeof(idType), cudaMemcpyDeviceToHost));
    cudaFree(d_c_size);
    
    checkCudaErrors(cudaMalloc((void **)&(d_nbs_sellcs_col), sizeof(compIdType) * nnz));
    checkCudaErrors(cudaMalloc((void **)&(d_is_empty), sizeof(bool) * full_c_size));
    checkCudaErrors(cudaMalloc((void **)&d_gcs, sizeof(idType) * (full_c_size + 1)));
  
    set_ushort_col<<<div_round_up(total_pad_row_num, BS), BS>>>(d_nbs_sellcs_col, d_nb_sellcs_col, d_full_cs, d_full_cl, d_is_empty, group_num_col, pad_M, chunk, seg_size);
    set_gcs(d_gcs, d_is_empty, full_c_size);

    checkCudaErrors(cudaMalloc((void **)&d_packed_cl, sizeof(idType) * c_size));
    checkCudaErrors(cudaMalloc((void **)&d_packed_cs, sizeof(idType) * c_size));

    set_packed_cl_cs<<<div_round_up(full_c_size, BS), BS>>>(d_packed_cl, d_packed_cs, d_full_cl, d_full_cs, d_gcs, full_c_size);

    cudaFree(d_full_cl);
    cudaFree(d_full_cs);
    cudaFree(d_nb_sellcs_col);
    
    /* Updating the write permutation */
    update_write_permutation<<<div_round_up(total_pad_row_num, BS), BS>>>(d_full_write_permutation, d_nnz_num, total_pad_row_num, pad_M, M);
    checkCudaErrors(cudaMalloc((void **)&d_write_permutation, sizeof(idType) * (c_size * chunk)));
    compress_write_permutation<<<div_round_up(total_pad_row_num, BS), BS>>>(d_write_permutation, d_full_write_permutation, d_gcs, total_pad_row_num, chunk);

    checkCudaErrors(cudaMalloc((void **)&(d_s_write_permutation), sizeof(compIdType) * (c_size * chunk)));
    checkCudaErrors(cudaMalloc((void **)&(d_s_write_permutation_offset), sizeof(compIdType) * c_size));
    compress_s_write_permutation<<<div_round_up((c_size * chunk), BS), BS>>>(d_s_write_permutation, d_s_write_permutation_offset, d_write_permutation, c_size, chunk);
  
    cudaFree(d_nnz_num);
    cudaFree(d_full_write_permutation);
    cudaFree(d_write_permutation);
    cudaFree(d_gcs);

    /* Step 4 : Blocking */
    int max_block = min(MAX_BLOCK_SIZE, N);
    max_block = min(max_block, seg_size);
    max_block = min(max_block, (N - seg_size * (seg_num - 1)));
    if (plan.isPlan == false) {
        for (block_size = 1; block_size <= max_block; block_size++) {
            if (block_size > 1) {
                checkCudaErrors(cudaFree(d_cl));
                checkCudaErrors(cudaFree(d_coffset));
                checkCudaErrors(cudaFree(d_cs));
                checkCudaErrors(cudaFree(d_sellcs_col));
                checkCudaErrors(cudaFree(d_sellcs_val));
            }
            checkCudaErrors(cudaMalloc((void **)&(d_cl), sizeof(compIdType) * c_size));
            checkCudaErrors(cudaMalloc((void **)&(d_coffset), sizeof(compIdType) * c_size));
            checkCudaErrors(cudaMalloc((void **)&(d_cs), sizeof(idType) * c_size));
    
            c_nnz = 0;
            set_blocked_cl_cs(d_cl, d_coffset, d_cs, d_packed_cl, d_packed_cs, d_nbs_sellcs_col, c_size, chunk, block_size, &c_nnz);

            nnz = c_nnz;
            checkCudaErrors(cudaMalloc((void **)&(d_sellcs_col), sizeof(compIdType) * c_nnz / block_size));
            checkCudaErrors(cudaMalloc((void **)&(d_sellcs_val), sizeof(valType) * c_nnz));
            set_blocked_col_val<<<div_round_up((c_size * chunk), BS), BS>>>(d_sellcs_col, d_sellcs_val, d_cl, d_cs, d_packed_cl, d_packed_cs, d_nbs_sellcs_col, d_nb_sellcs_val, c_size, chunk, block_size);
            adjust_blocked_col_val<<<div_round_up((c_size * chunk), BS), BS>>>(d_sellcs_col, d_sellcs_val, d_cl, d_cs, d_coffset, c_size, chunk, block_size, N, seg_size);
            cudaThreadSynchronize();
    
#ifdef AT
            evaluate_spmv(d_x, d_y, plan);
#else 
            unsigned long long int footprint = 0;
            footprint += (nnz / block_size) * sizeof(compIdType); // col
            footprint += (nnz) * sizeof(valType); // val
            footprint += (c_size) * sizeof(idType) * 2; // cs, cl
            footprint += (c_size) * chunk * sizeof(compIdType) + (c_size) * sizeof(compIdType); // permutation
            footprint += (c_size) * chunk * sizeof(valType) * 2; // output
            footprint += (M) * sizeof(valType) * 2; // input + output_init
            
            if (plan.memory_access > footprint) {
                plan.memory_access = footprint;
                plan.seg_size = (seg_size);
                plan.block_size = block_size;
            }
#endif
        }

        /* Set Best Block Size */
        block_size = plan.block_size;
        checkCudaErrors(cudaFree(d_cl));
        checkCudaErrors(cudaFree(d_coffset));
        checkCudaErrors(cudaFree(d_cs));
        checkCudaErrors(cudaFree(d_sellcs_col));
        checkCudaErrors(cudaFree(d_sellcs_val));
    }

    checkCudaErrors(cudaMalloc((void **)&(d_cl), sizeof(compIdType) * c_size));
    checkCudaErrors(cudaMalloc((void **)&(d_coffset), sizeof(compIdType) * c_size));
    checkCudaErrors(cudaMalloc((void **)&(d_cs), sizeof(idType) * c_size));

    c_nnz = 0;
    set_blocked_cl_cs(d_cl, d_coffset, d_cs, d_packed_cl, d_packed_cs, d_nbs_sellcs_col, c_size, chunk, block_size, &c_nnz);

    nnz = c_nnz;
    checkCudaErrors(cudaMalloc((void **)&(d_sellcs_col), sizeof(compIdType) * c_nnz / (block_size)));
    checkCudaErrors(cudaMalloc((void **)&(d_sellcs_val), sizeof(valType) * c_nnz));

    set_blocked_col_val<<<div_round_up((c_size * chunk), BS), BS>>>(d_sellcs_col, d_sellcs_val, d_cl, d_cs, d_packed_cl, d_packed_cs, d_nbs_sellcs_col, d_nb_sellcs_val, c_size, chunk, (block_size));
    adjust_blocked_col_val<<<div_round_up((c_size * chunk), BS), BS>>>(d_sellcs_col, d_sellcs_val, d_cl, d_cs, d_coffset, c_size, chunk, block_size, N, seg_size);
    
    cudaFree(d_packed_cl);
    cudaFree(d_packed_cs);
    cudaFree(d_nb_sellcs_val);
    cudaFree(d_nbs_sellcs_col);

}

template <class idType, class compIdType, class valType>
void AMB<idType, compIdType, valType>::convert_from_csr(CSR<idType, valType> mat, Plan<idType> &plan, valType *d_x)
{
    int i;
    idType max_div_size, seg_pattern;
    size_t *base_size;
    int seg_it;
  
    valType *d_y;
    cudaEvent_t event[2];
    for (i = 0; i < 2; i++) {
        cudaEventCreate(&(event[i]));
    }
    
    M = mat.nrow;
    N = mat.ncolumn;
    chunk = warp;
    pad_M = chunk * div_round_up(mat.nrow, chunk);
    SIGMA = SHORT_MAX;
  
    checkCudaErrors(cudaMalloc((void **)&d_y, sizeof(valType) * (M + warp)));
  
    if (plan.isPlan == true) {
        seg_size = plan.seg_size;
        seg_num = div_round_up(N, seg_size);
        plan.seg_num = seg_num;
        group_num_col = seg_num;
        block_size = plan.block_size;
    
        convert_amb_at(mat, d_x, d_y, plan);
        evaluate_spmv(d_x, d_y, plan);
    }
    else {
        max_div_size = 128 * 1024;
        seg_pattern = (N < max_div_size)? 5 : 1;
        base_size = (size_t *)malloc(sizeof(size_t) * seg_pattern);
        base_size[0] = 64 * 1024;
        if (N < max_div_size) {
            for (i = 1; i < seg_pattern; i++) {
                base_size[i] = i * 1024;
            }
        }
        if (N < 100) {
            for (i = 1; i < seg_pattern; i++) {
                base_size[i] = i;
            }
        }

        for (seg_it = 0; seg_it < seg_pattern; seg_it++) {
            if (seg_it > 0) {
                release_amb();
            }
            (seg_size) = base_size[seg_it];
            (seg_num) = div_round_up(N, seg_size);
      
            group_num_col = (seg_num);
  
            convert_amb_at(mat, d_x, d_y, plan);
        }
      
        plan.isPlan = true;
        /* Set Best AMB format */
        if ((seg_size != plan.seg_size)) {
            release_amb();
            (seg_size) = plan.seg_size;
            (seg_num) = div_round_up(N, (seg_size));
            block_size = plan.block_size;
            group_num_col = seg_num;
      
            convert_amb_at(mat, d_x, d_y, plan);
        }

#ifdef AT
#else
        evaluate_spmv(d_x, d_y, plan);
#endif

        plan.seg_num = (seg_num);
  
    }
    cudaFree(d_y);
}

template <class idType, class valType>
__global__ void kernel_spmv_init_ans(valType *d_ans, idType M)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) {
        return;
    }
    d_ans[i] = 0;
}

template <int block_size, class idType, class compIdType, class valType, class AddOperation, class MultOperation, class AAddOperation>
__global__ void kernel_spmv_amb_atomic(valType *ans,
                                       const valType *value, const compIdType *col,
                                       const compIdType* __restrict__ cl,
                                       const compIdType* __restrict__ coffset,
                                       const idType* __restrict__ cs,
                                       const valType* __restrict__ vector,
                                       const compIdType *d_permutation,
                                       const compIdType* __restrict__ d_permutation_offset,
                                       idType row_num,
                                       idType seg_size,
                                       AddOperation addop, MultOperation mulop, AAddOperation aaddop)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (i >= row_num) {
        return;
    }
    
    idType c_index = i >> warp_BIT;
    idType offset = ld_gbl_col(d_permutation + i) + d_permutation_offset[c_index] * USHORT_MAX;
    
    idType start = cs[c_index] + (threadIdx.x & (warp - 1));
    idType colstart = (cs[c_index] / block_size) + (threadIdx.x & (warp - 1));

    idType width = cl[c_index];
    idType c_offset = coffset[c_index] * seg_size;
  
    idType h;
    idType c = ld_gbl_col(col + colstart) + c_offset;
    valType answer = 0;
#pragma unroll
    for (int b = 0; b < block_size; ++b) {
        answer = addop(answer, mulop(ld_gbl_val(value + start), vector[c + b]));
        start += warp;
    }
    colstart += warp;

    for (h = 0; h < width; h++) {
        c = ld_gbl_col(col + colstart) + c_offset;
#pragma unroll
        for (int b = 0; b < block_size; ++b) {
            answer = addop(answer, mulop(ld_gbl_val(value + start), vector[c + b]));
            start += warp;
        }
        colstart += warp;
    }
    aaddop(ans + offset, answer);
}

template <int bs, class idType, class compIdType, class valType>
class call_kernel
{
public:
    inline static void f(const valType *d_x, valType *d_y, const Plan<idType> &plan, AMB<idType, compIdType, valType> *mat)
    {
        if (bs == mat->block_size) {
            kernel_spmv_amb_atomic<bs><<<plan.thread_grid, plan.thread_block>>>(d_y, mat->d_sellcs_val, mat->d_sellcs_col, mat->d_cl, mat->d_coffset, mat->d_cs, d_x, mat->d_s_write_permutation, mat->d_s_write_permutation_offset, mat->c_size * mat->chunk, mat->seg_size, Add<valType>(), Multiply<valType>(), AtomicAdd<valType>());
        }
        call_kernel<bs + 1, idType, compIdType, valType>::f(d_x, d_y, plan, mat);
    }
};

template <class idType, class compIdType, class valType>
class call_kernel<MAX_BLOCK_SIZE, idType, compIdType, valType>
{
public:
    inline static void f(const valType *d_x, valType *d_y, const Plan<idType> &plan, AMB<idType, compIdType, valType> *mat)
    {
        if (MAX_BLOCK_SIZE == mat->block_size) {
            kernel_spmv_amb_atomic<MAX_BLOCK_SIZE><<<plan.thread_grid, plan.thread_block>>>(d_y, mat->d_sellcs_val, mat->d_sellcs_col, mat->d_cl, mat->d_coffset, mat->d_cs, d_x, mat->d_s_write_permutation, mat->d_s_write_permutation_offset, mat->c_size * mat->chunk, mat->seg_size, Add<valType>(), Multiply<valType>(), AtomicAdd<valType>());
        }
    }
};

template <class idType, class compIdType, class valType>
void AMB<idType, compIdType, valType>::spmv(const valType *d_x, valType *d_y, const Plan<idType> &plan)
{
    kernel_spmv_init_ans<<<div_round_up(M, MAX_LOCAL_THREAD_NUM), MAX_LOCAL_THREAD_NUM>>>(d_y, M);
    call_kernel<1, idType, compIdType, valType>::f(d_x, d_y, plan, this);
    cudaThreadSynchronize();
}


