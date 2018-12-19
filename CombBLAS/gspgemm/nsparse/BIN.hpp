#ifndef BIN_H
#define BIN_H

template <class idType, int BIN_NUM>
class BIN
{
public:
    BIN(idType nrow):max_flop(0), max_nz(0)
    {
        stream = (cudaStream_t *)malloc(sizeof(cudaStream_t) * BIN_NUM);
        for (int i = 0; i < BIN_NUM; ++i) {
            cudaStreamCreate(&(stream[i]));
        }
        bin_size = (idType *)malloc(sizeof(idType) * BIN_NUM);
        bin_offset = (idType *)malloc(sizeof(idType) * BIN_NUM);
        cudaMalloc((void **)&(d_permutation), sizeof(idType) * nrow);
        cudaMalloc((void **)&(d_count), sizeof(idType) * (nrow + 1));
        cudaMalloc((void **)&(d_max), sizeof(idType));
        cudaMalloc((void **)&(d_bin_size), sizeof(idType) * BIN_NUM);
        cudaMalloc((void **)&(d_bin_offset), sizeof(idType) * BIN_NUM);
    }
    ~BIN()
    {
        cudaFree(d_count);
        cudaFree(d_permutation);
        cudaFree(d_max);
        cudaFree(d_bin_size);
        cudaFree(d_bin_offset);
        free(bin_size);
        free(bin_offset);
        for (int i = 0; i < BIN_NUM; i++) {
            cudaStreamDestroy(stream[i]);
        }
        free(stream);
    }
    
    void set_max_bin(idType *d_arpt, idType *d_acol, idType *d_brpt, idType M, int TS_S_P, int TS_S_T);
    void set_min_bin(idType M, int TS_N_P, int TS_N_T);
    void set_min_bin(idType *d_rpt, idType M, int TS_N_P, int TS_N_T);

    cudaStream_t *stream;
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
__global__ void set_flop_per_row(idType *d_arpt, idType *d_acol, const idType* __restrict__ d_brpt, idType *d_count, idType *d_max_flop, idType nrow)
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

template <class idType, int BIN_NUM>
__global__ void set_bin(idType *d_count, idType *d_bin_size, idType *d_max, idType nrow, idType min, idType mmin)
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
__global__ void init_row_perm(idType *d_permutation, idType M)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= M) {
        return;
    }
  
    d_permutation[i] = i;
}

template <class idType, int BIN_NUM>
__global__ void set_row_perm(idType *d_bin_size, idType *d_bin_offset,
                             idType *d_max_row_nz, idType *d_permutation,
                             idType M, idType min, idType mmin)
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

template <class idType, int BIN_NUM>
void BIN<idType, BIN_NUM>::set_max_bin(idType *d_arpt, idType *d_acol, idType *d_brpt, idType M, int TS_S_P, int TS_S_T)
{
    idType i;
    idType GS, BS;
  
    for (i = 0; i < BIN_NUM; i++) {
        bin_size[i] = 0;
        bin_offset[i] = 0;
    }
  
    cudaMemcpy(d_bin_size, bin_size, sizeof(idType) * BIN_NUM, cudaMemcpyHostToDevice);
    cudaMemcpy(d_max, &(max_flop), sizeof(idType), cudaMemcpyHostToDevice);
  
    BS = 1024;
    GS = div_round_up(M, BS);
    set_flop_per_row<idType><<<GS, BS>>>(d_arpt, d_acol, d_brpt, d_count, d_max, M);
    cudaMemcpy(&(max_flop), d_max, sizeof(idType), cudaMemcpyDeviceToHost);
  
    if (max_flop > TS_S_P) {
        set_bin<idType, BIN_NUM><<<GS, BS>>>(d_count, d_bin_size, d_max, M, TS_S_T, TS_S_P);
  
        cudaMemcpy(bin_size, d_bin_size, sizeof(idType) * BIN_NUM, cudaMemcpyDeviceToHost);
        cudaMemcpy(d_bin_size, bin_offset, sizeof(idType) * BIN_NUM, cudaMemcpyHostToDevice);

        for (i = 0; i < BIN_NUM - 1; i++) {
            bin_offset[i + 1] = bin_offset[i] + bin_size[i];
        }
        cudaMemcpy(d_bin_offset, bin_offset, sizeof(idType) * BIN_NUM, cudaMemcpyHostToDevice);

        set_row_perm<idType, BIN_NUM><<<GS, BS>>>(d_bin_size, d_bin_offset, d_count, d_permutation, M, TS_S_T, TS_S_P);
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
        init_row_perm<<<GS, BS>>>(d_permutation, M);
    }
}

template <class idType, int BIN_NUM>
void BIN<idType, BIN_NUM>::set_min_bin(idType M, int TS_N_P, int TS_N_T)
{
    idType i;
    idType GS, BS;
  
    for (i = 0; i < BIN_NUM; i++) {
        bin_size[i] = 0;
        bin_offset[i] = 0;
    }
  
    cudaMemcpy(d_bin_size, bin_size, sizeof(idType) * BIN_NUM, cudaMemcpyHostToDevice);
    cudaMemcpy(d_max, &(max_nz), sizeof(idType), cudaMemcpyHostToDevice);
  
    BS = 1024;
    GS = div_round_up(M, BS);
    set_bin<idType, BIN_NUM><<<GS, BS>>>(d_count, d_bin_size,
                                         d_max,
                                         M, TS_N_T, TS_N_P);
  
    cudaMemcpy(&(max_nz), d_max, sizeof(idType), cudaMemcpyDeviceToHost);
    if (max_nz > TS_N_P) {
        cudaMemcpy(bin_size, d_bin_size, sizeof(idType) * BIN_NUM, cudaMemcpyDeviceToHost);
        cudaMemcpy(d_bin_size, bin_offset, sizeof(idType) * BIN_NUM, cudaMemcpyHostToDevice);

        for (i = 0; i < BIN_NUM - 1; i++) {
            bin_offset[i + 1] = bin_offset[i] + bin_size[i];
        }
        cudaMemcpy(d_bin_offset, bin_offset, sizeof(idType) * BIN_NUM, cudaMemcpyHostToDevice);
  
        set_row_perm<idType, BIN_NUM><<<GS, BS>>>(d_bin_size, d_bin_offset, d_count, d_permutation, M, TS_N_T, TS_N_P);
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
        init_row_perm<<<GS, BS>>>(d_permutation, M);
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
  
    cudaMemcpy(d_bin_size, bin_size, sizeof(idType) * BIN_NUM, cudaMemcpyHostToDevice);
    cudaMemcpy(d_max, &(max_nz), sizeof(idType), cudaMemcpyHostToDevice);
  
    BS = 1024;
    GS = div_round_up(M, BS);
    set_nnz_per_row_from_rpt<<<GS, BS>>>(d_rpt, d_count, M);
    set_bin<idType, BIN_NUM><<<GS, BS>>>(d_count, d_bin_size,
                                         d_max,
                                         M, TS_N_T, TS_N_P);
    
    cudaMemcpy(&(max_nz), d_max, sizeof(idType), cudaMemcpyDeviceToHost);
    if (max_nz > TS_N_P) {
        cudaMemcpy(bin_size, d_bin_size, sizeof(idType) * BIN_NUM, cudaMemcpyDeviceToHost);
        cudaMemcpy(d_bin_size, bin_offset, sizeof(idType) * BIN_NUM, cudaMemcpyHostToDevice);

        for (i = 0; i < BIN_NUM - 1; i++) {
            bin_offset[i + 1] = bin_offset[i] + bin_size[i];
        }
        cudaMemcpy(d_bin_offset, bin_offset, sizeof(idType) * BIN_NUM, cudaMemcpyHostToDevice);
  
        set_row_perm<idType, BIN_NUM><<<GS, BS>>>(d_bin_size, d_bin_offset, d_count, d_permutation, M, TS_N_T, TS_N_P);
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
        init_row_perm<<<GS, BS>>>(d_permutation, M);
    }
}
#endif
