//////////////////////////////////////////////////////////////////////////
// < A CUDA/OpenCL General Sparse Matrix-Matrix Multiplication Program >
//
// < See paper:
// Weifeng Liu and Brian Vinter, "An Efficient GPU General Sparse
// Matrix-Matrix Multiplication for Irregular Data," Parallel and
// Distributed Processing Symposium, 2014 IEEE 28th International
// (IPDPS '14), pp.370-381, 19-23 May 2014
// for details. >
//////////////////////////////////////////////////////////////////////////

#ifndef BHSPARSE_CUDA_H
#define BHSPARSE_CUDA_H

#include "bhsparse_common.h"

class bhsparse_cuda
{
public:
    bhsparse_cuda();
    int initPlatform();
    int initData(int m, int k, int n,
             int nnzA, value_type *csrValA, index_type *csrRowPtrA, index_type *csrColIndA,
             int nnzB, value_type *csrValB, index_type *csrRowPtrB, index_type *csrColIndB,
             index_type *csrRowPtrC, index_type *csrRowPtrCt, index_type *queue_one);

    void setProfiling(bool profiling);
    int kernel_barrier();
    int warmup();
    int freePlatform();
    int free_mem();

    int compute_nnzCt();
    int create_Ct(int nnzCt);
    int create_C();

    int compute_nnzC_Ct_0(int num_threads, int num_blocks, int j, int counter, int position);
    int compute_nnzC_Ct_1(int num_threads, int num_blocks, int j, int counter, int position);
    int compute_nnzC_Ct_2heap_noncoalesced(int num_threads, int num_blocks, int j, int counter, int position);
    int compute_nnzC_Ct_bitonic(int num_threads, int num_blocks, int j, int position);
    int compute_nnzC_Ct_mergepath(int num_threads, int num_blocks, int j, int mergebuffer_size, int position, int *count_next, int mergepath_location);


    int copy_Ct_to_C_Single(int num_threads, int num_blocks, int local_size, int position);
    int copy_Ct_to_C_Loopless(int num_threads, int num_blocks, int j, int position);
    int copy_Ct_to_C_Loop(int num_threads, int num_blocks, int j, int position);

    int get_nnzC();
    int get_C(index_type *csrColIndC, value_type *csrValC);

	index_type* get_ColIndC(){return _d_csrColIndC;}
	index_type* get_RowPtrC(){return _d_csrRowPtrC;}
	value_type* get_ValC(){return _d_csrValC;}

private:
    bool _profiling;

    int _num_smxs;
    int _max_blocks_per_smx;

    int _m;
    int _k;
    int _n;

    // A
    int _nnzA;
    value_type *_d_csrValA;
    index_type *_d_csrRowPtrA;
    index_type *_d_csrColIndA;

    // B
    int _nnzB;
    value_type *_d_csrValB;
    index_type *_d_csrRowPtrB;
    index_type *_d_csrColIndB;

    // C
    int _nnzC;
    index_type *_h_csrRowPtrC;
    index_type *_d_csrRowPtrC;
    index_type *_d_csrColIndC;
    value_type *_d_csrValC;

    // Ct
    int _nnzCt;
    value_type *_d_csrValCt;
    index_type *_d_csrColIndCt;
    index_type *_d_csrRowPtrCt;
    index_type *_h_csrRowPtrCt;

    // QUEUE_ONEs
    index_type *_h_queue_one;
    index_type *_d_queue_one;
};

bhsparse_cuda::bhsparse_cuda()
{
}

int bhsparse_cuda::initPlatform()
{
    _profiling = false;

    int device_id = 0;
    cudaSetDevice(device_id);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);

    _num_smxs = deviceProp.multiProcessorCount;
    _max_blocks_per_smx = deviceProp.maxThreadsPerMultiProcessor / WARPSIZE_NV;

    //cout << "Device [" <<  device_id << "] " << deviceProp.name
    //     << " @ " << deviceProp.clockRate * 1e-3f << "MHz. "
    //     << _num_smxs << " SMXs." << endl;

    return 0;
}

int bhsparse_cuda::freePlatform()
{
    int err = 0;
    return err;
}

int bhsparse_cuda::free_mem()
{
    int err = 0;

    // A
    //cudaFree(_d_csrValA);
    //cudaFree(_d_csrRowPtrA);
    //cudaFree(_d_csrColIndA);

    // B
    //cudaFree(_d_csrValB);
    //cudaFree(_d_csrRowPtrB);
    //cudaFree(_d_csrColIndB);

    // C
	//Thesee are released by the user
    //cudaFree(_d_csrValC);
    //cudaFree(_d_csrRowPtrC);
    //cudaFree(_d_csrColIndC);

    // Ct
    cudaFree(_d_csrValCt);
    cudaFree(_d_csrRowPtrCt);
    cudaFree(_d_csrColIndCt);

    // QUEUE_ONEs
    cudaFree(_d_queue_one);

    return err;
}

//csrRowPtrC, csrRowPtrCt and queue_one are on host mem
int bhsparse_cuda::initData(int m, int k, int n,
                      int nnzA, value_type *csrValA, index_type *csrRowPtrA, index_type *csrColIndA,
                      int nnzB, value_type *csrValB, index_type *csrRowPtrB, index_type *csrColIndB,
                      index_type *csrRowPtrC, index_type *csrRowPtrCt, index_type *queue_one)
{
    int err = 0;

    _m = m;
    _k = k;
    _n = n;

    _nnzA = nnzA;
    _nnzB = nnzB;
    _nnzC = 0;
    _nnzCt = 0;

    // Matrix A
    _d_csrColIndA=csrColIndA;
    _d_csrRowPtrA=csrRowPtrA;
    _d_csrValA=csrValA;

    // Matrix B
    _d_csrColIndB=csrColIndB;
    _d_csrRowPtrB=csrRowPtrB;
    _d_csrValB=csrValB;

    // Matrix C
    _h_csrRowPtrC = csrRowPtrC;
    cudaMalloc((void **)&_d_csrRowPtrC, (_m+1) * sizeof(index_type));
    cudaMemset(_d_csrRowPtrC, 0, (_m+1) * sizeof(index_type));

    // Matrix Ct
    _h_csrRowPtrCt = csrRowPtrCt;
    cudaMalloc((void **)&_d_csrRowPtrCt, (_m+1) * sizeof(index_type));
    cudaMemset(_d_csrRowPtrCt, 0, (_m+1) * sizeof(index_type));

    // statistics - queue_one
    _h_queue_one = queue_one;
    cudaMalloc((void **)&_d_queue_one, TUPLE_QUEUE * _m * sizeof(index_type));
    cudaMemset(_d_queue_one, 0, TUPLE_QUEUE * _m * sizeof(index_type));

    return err;
}

void bhsparse_cuda::setProfiling(bool profiling)
{
    _profiling = profiling;
}

__global__ void
compute_nnzCt_cudakernel(const int*  d_csrRowPtrA,
                         const  int* __restrict__ d_csrColIndA,
                         const int*  d_csrRowPtrB,
                         int                    *d_csrRowPtrCt,
                         const int m)
{
    int global_id  = blockIdx.x * blockDim.x + threadIdx.x;
    int start, stop, index, strideB, row_size_Ct = 0;

    if (global_id < m)
    {
        start = d_csrRowPtrA[global_id];
        stop  = d_csrRowPtrA[global_id + 1];

        for (int i = start; i < stop; i++)
        {
            index = d_csrColIndA[i];
            strideB = d_csrRowPtrB[index + 1] - d_csrRowPtrB[index];
            row_size_Ct += strideB;
        }

        d_csrRowPtrCt[global_id] = row_size_Ct;
    }

    if (global_id == 0)
        d_csrRowPtrCt[m] = 0;
}

int bhsparse_cuda::kernel_barrier()
{
    return cudaDeviceSynchronize();
}

int bhsparse_cuda::compute_nnzCt()
{
    cudaError_t err = cudaSuccess;

    int num_threads = GROUPSIZE_256;
    int num_blocks  = int(ceil((double)_m / (double)num_threads));

    compute_nnzCt_cudakernel<<< num_blocks, num_threads >>>(_d_csrRowPtrA,
                                                            _d_csrColIndA,
                                                            _d_csrRowPtrB,
                                                            _d_csrRowPtrCt,
                                                            _m);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {  cout << "err = " << cudaGetErrorString(err) << endl; return -1; }

    cudaMemcpy(_h_csrRowPtrCt, _d_csrRowPtrCt, (_m + 1) * sizeof(index_type), cudaMemcpyDeviceToHost);

    return BHSPARSE_SUCCESS;
}

int bhsparse_cuda::create_Ct(int nnzCt)
{
    int err = 0;

    cudaMemcpy(_d_queue_one, _h_queue_one, TUPLE_QUEUE * _m * sizeof(index_type),   cudaMemcpyHostToDevice);

    _nnzCt = nnzCt;

    // create device mem of Ct
    cudaMalloc((void **)&_d_csrColIndCt, _nnzCt  * sizeof(index_type));
    cudaMalloc((void **)&_d_csrValCt,    _nnzCt  * sizeof(value_type));

    cudaMemset(_d_csrColIndCt, 0, _nnzCt * sizeof(index_type));
    cudaMemset(_d_csrValCt,    0, _nnzCt * sizeof(value_type));

    return err;
}

__inline__ __device__ void
siftDown(int   *s_key,
         value_type *s_val,
         const int    start,
         const int    stop,
         const int local_id,
         const int local_size)
{
    int root = start;
    int child, swap;

    int temp_swap_key;
    value_type temp_swap_val;

    while (root * 2 + 1 <= stop)
    {
        child = root * 2 + 1;
        swap = root;

        if (s_key[swap * local_size + local_id] < s_key[child * local_size + local_id])
            swap = child;

        if (child + 1 <= stop && s_key[swap * local_size + local_id] < s_key[(child + 1) * local_size + local_id])
            swap = child + 1;

        if (swap != root)
        {
            const int index1 = root * local_size + local_id;
            const int index2 = swap * local_size + local_id;

            //swap root and swap
            temp_swap_key = s_key[index1];
            s_key[index1] = s_key[index2];
            s_key[index2] = temp_swap_key;

            temp_swap_val = s_val[index1];
            s_val[index1] = s_val[index2];
            s_val[index2] = temp_swap_val;

            root = swap;
        }
        else
            return;
    }
}

__inline__ __device__ int
heapsort(int   *s_key,
         value_type *s_val,
         const int    segment_size,
         const int local_id,
         const int local_size)
{
    // heapsort - heapify max-heap
    int start = (segment_size - 1) / 2;
    int stop  = segment_size - 1;

    int index1, index2;

    while (start >= 0)
    {
        siftDown(s_key, s_val, start, stop, local_id, local_size);
        start--;
    }

    // inject root element to the end

    int temp_swap_key;
    value_type temp_swap_val;

    index1 = stop * local_size + local_id;

    temp_swap_key = s_key[local_id];
    s_key[local_id] = s_key[index1];
    s_key[index1] = temp_swap_key;

    temp_swap_val = s_val[local_id];
    s_val[local_id] = s_val[index1];
    s_val[index1] = temp_swap_val;

    stop--;

    siftDown(s_key, s_val, 0, stop, local_id, local_size);

    // this start is compressed list's start
    start = segment_size - 1;

    // heapsort - remove-max and compress
    while (stop >= 0)
    {
        index2 = stop * local_size + local_id;

        if (s_key[local_id] == s_key[start * local_size + local_id])
        {
            s_val[start * local_size + local_id] += s_val[local_id];

            s_key[local_id] = s_key[index2];
            s_val[local_id] = s_val[index2];
        }
        else
        {
            start--;

            index1 = start * local_size + local_id;

            if (stop == start)
            {
                temp_swap_key = s_key[local_id];
                s_key[local_id] = s_key[index2];
                s_key[index2] = temp_swap_key;

                temp_swap_val = s_val[local_id];
                s_val[local_id] = s_val[index2];
                s_val[index2] = temp_swap_val;
            }
            else
            {
                s_key[index1] = s_key[local_id];
                s_val[index1] = s_val[local_id];

                s_key[local_id] = s_key[index2];
                s_val[local_id] = s_val[index2];
            }
        }

        stop--;

        siftDown(s_key, s_val, 0, stop, local_id, local_size);
    }

    return start;
}

template<typename vT, int c_segmentsize>
__global__  void
ESC_2heap_noncoalesced(const int*     d_queue,
                       const int*     d_csrRowPtrA,
                       const  int*   __restrict__  d_csrColIndA,
                       const  vT*  __restrict__ d_csrValA,
                       const int*     d_csrRowPtrB,
                       const int*    __restrict__ d_csrColIndB,
                       const vT*  __restrict__ d_csrValB,
                       int*                       d_csrRowPtrC,
                       const int*     d_csrRowPtrCt,
                       int*                       d_csrColIndCt,
                       vT*                     d_csrValCt,
                       const int                  queue_size,
                       const int                  d_queue_offset)
{
    __shared__ int   s_key[c_segmentsize * WARPSIZE_NV_2HEAP];
    __shared__ vT s_val[c_segmentsize * WARPSIZE_NV_2HEAP];

    const int local_id   = threadIdx.x;
    const int group_id   = blockIdx.x;
    const int global_id  = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_size = blockDim.x;
    int index = 0;

    if (global_id < queue_size)
    {
        int i, counter = 0;
        int start_col_index_A, stop_col_index_A;
        int rowidB, start_col_index_B, stop_col_index_B;
        vT value_A;

        int rowidC = d_queue[TUPLE_QUEUE * (d_queue_offset + global_id)];

        start_col_index_A = d_csrRowPtrA[rowidC];
        stop_col_index_A  = d_csrRowPtrA[rowidC + 1];

        // i is both col index of A and row index of B
        for (i = start_col_index_A; i < stop_col_index_A; i++)
        {
            rowidB = d_csrColIndA[i];
            value_A  = d_csrValA[i];

            start_col_index_B = d_csrRowPtrB[rowidB];
            stop_col_index_B  = d_csrRowPtrB[rowidB + 1];

            for (int j = start_col_index_B; j < stop_col_index_B; j++)
            {
                index = counter * local_size + local_id;
                s_key[index] = d_csrColIndB[j];
                s_val[index] = d_csrValB[j] * value_A;

                counter++;
            }
        }

        // heapsort in each work-item
        int local_start = heapsort(s_key, s_val, counter, local_id, local_size);

        counter -= local_start;
        d_csrRowPtrC[rowidC] = counter;

        int base_index = d_queue[TUPLE_QUEUE * (d_queue_offset + group_id * local_size + local_id) + 1];;
        for (int i = 0; i < counter; i++)
        {
            d_csrColIndCt[base_index + i] = s_key[(local_start+i) * local_size + local_id];
            d_csrValCt[base_index + i] = s_val[(local_start+i) * local_size + local_id];
        }
    }
}

int bhsparse_cuda::compute_nnzC_Ct_2heap_noncoalesced(int num_threads, int num_blocks, int j, int counter, int position)
{
    cudaError_t err = cudaSuccess;

    switch (j)
    {
    case 2:
        ESC_2heap_noncoalesced<value_type, 2><<< num_blocks, num_threads >>>(_d_queue_one,
                                           _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                           _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                           _d_csrRowPtrC, _d_csrRowPtrCt, _d_csrColIndCt, _d_csrValCt,
                                           counter, position);
        break;
    case 3:
        ESC_2heap_noncoalesced<value_type, 3><<< num_blocks, num_threads >>>(_d_queue_one,
                                           _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                           _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                           _d_csrRowPtrC, _d_csrRowPtrCt, _d_csrColIndCt, _d_csrValCt,
                                           counter, position);
        break;
    case 4:
        ESC_2heap_noncoalesced<value_type, 4><<< num_blocks, num_threads >>>(_d_queue_one,
                                           _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                           _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                           _d_csrRowPtrC, _d_csrRowPtrCt, _d_csrColIndCt, _d_csrValCt,
                                           counter, position);
        break;
    case 5:
        ESC_2heap_noncoalesced<value_type, 5><<< num_blocks, num_threads >>>(_d_queue_one,
                                           _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                           _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                           _d_csrRowPtrC, _d_csrRowPtrCt, _d_csrColIndCt, _d_csrValCt,
                                           counter, position);
        break;
    case 6:
        ESC_2heap_noncoalesced<value_type, 6><<< num_blocks, num_threads >>>(_d_queue_one,
                                           _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                           _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                           _d_csrRowPtrC, _d_csrRowPtrCt, _d_csrColIndCt, _d_csrValCt,
                                           counter, position);
        break;
    case 7:
        ESC_2heap_noncoalesced<value_type, 7><<< num_blocks, num_threads >>>(_d_queue_one,
                                           _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                           _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                           _d_csrRowPtrC, _d_csrRowPtrCt, _d_csrColIndCt, _d_csrValCt,
                                           counter, position);
        break;
    case 8:
        ESC_2heap_noncoalesced<value_type, 8><<< num_blocks, num_threads >>>(_d_queue_one,
                                           _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                           _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                           _d_csrRowPtrC, _d_csrRowPtrCt, _d_csrColIndCt, _d_csrValCt,
                                           counter, position);
        break;
    case 9:
        ESC_2heap_noncoalesced<value_type, 9><<< num_blocks, num_threads >>>(_d_queue_one,
                                           _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                           _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                           _d_csrRowPtrC, _d_csrRowPtrCt, _d_csrColIndCt, _d_csrValCt,
                                           counter, position);
        break;
    case 10:
        ESC_2heap_noncoalesced<value_type, 10><<< num_blocks, num_threads >>>(_d_queue_one,
                                           _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                           _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                           _d_csrRowPtrC, _d_csrRowPtrCt, _d_csrColIndCt, _d_csrValCt,
                                           counter, position);
        break;
    case 11:
        ESC_2heap_noncoalesced<value_type, 11><<< num_blocks, num_threads >>>(_d_queue_one,
                                           _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                           _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                           _d_csrRowPtrC, _d_csrRowPtrCt, _d_csrColIndCt, _d_csrValCt,
                                           counter, position);
        break;
    case 12:
        ESC_2heap_noncoalesced<value_type, 12><<< num_blocks, num_threads >>>(_d_queue_one,
                                           _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                           _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                           _d_csrRowPtrC, _d_csrRowPtrCt, _d_csrColIndCt, _d_csrValCt,
                                           counter, position);
        break;
    case 13:
        ESC_2heap_noncoalesced<value_type, 13><<< num_blocks, num_threads >>>(_d_queue_one,
                                           _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                           _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                           _d_csrRowPtrC, _d_csrRowPtrCt, _d_csrColIndCt, _d_csrValCt,
                                           counter, position);
        break;
    case 14:
        ESC_2heap_noncoalesced<value_type, 14><<< num_blocks, num_threads >>>(_d_queue_one,
                                           _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                           _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                           _d_csrRowPtrC, _d_csrRowPtrCt, _d_csrColIndCt, _d_csrValCt,
                                           counter, position);
        break;
    case 15:
        ESC_2heap_noncoalesced<value_type, 15><<< num_blocks, num_threads >>>(_d_queue_one,
                                           _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                           _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                           _d_csrRowPtrC, _d_csrRowPtrCt, _d_csrColIndCt, _d_csrValCt,
                                           counter, position);
        break;
    case 16:
        ESC_2heap_noncoalesced<value_type, 16><<< num_blocks, num_threads >>>(_d_queue_one,
                                           _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                           _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                           _d_csrRowPtrC, _d_csrRowPtrCt, _d_csrColIndCt, _d_csrValCt,
                                           counter, position);
        break;
    case 17:
        ESC_2heap_noncoalesced<value_type, 17><<< num_blocks, num_threads >>>(_d_queue_one,
                                           _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                           _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                           _d_csrRowPtrC, _d_csrRowPtrCt, _d_csrColIndCt, _d_csrValCt,
                                           counter, position);
        break;
    case 18:
        ESC_2heap_noncoalesced<value_type, 18><<< num_blocks, num_threads >>>(_d_queue_one,
                                           _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                           _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                           _d_csrRowPtrC, _d_csrRowPtrCt, _d_csrColIndCt, _d_csrValCt,
                                           counter, position);
        break;
    case 19:
        ESC_2heap_noncoalesced<value_type, 19><<< num_blocks, num_threads >>>(_d_queue_one,
                                           _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                           _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                           _d_csrRowPtrC, _d_csrRowPtrCt, _d_csrColIndCt, _d_csrValCt,
                                           counter, position);
        break;
    case 20:
        ESC_2heap_noncoalesced<value_type, 20><<< num_blocks, num_threads >>>(_d_queue_one,
                                           _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                           _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                           _d_csrRowPtrC, _d_csrRowPtrCt, _d_csrColIndCt, _d_csrValCt,
                                           counter, position);
        break;
    case 21:
        ESC_2heap_noncoalesced<value_type, 21><<< num_blocks, num_threads >>>(_d_queue_one,
                                           _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                           _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                           _d_csrRowPtrC, _d_csrRowPtrCt, _d_csrColIndCt, _d_csrValCt,
                                           counter, position);
        break;
    case 22:
        ESC_2heap_noncoalesced<value_type, 22><<< num_blocks, num_threads >>>(_d_queue_one,
                                           _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                           _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                           _d_csrRowPtrC, _d_csrRowPtrCt, _d_csrColIndCt, _d_csrValCt,
                                           counter, position);
        break;
    case 23:
        ESC_2heap_noncoalesced<value_type, 23><<< num_blocks, num_threads >>>(_d_queue_one,
                                           _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                           _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                           _d_csrRowPtrC, _d_csrRowPtrCt, _d_csrColIndCt, _d_csrValCt,
                                           counter, position);
        break;
    case 24:
        ESC_2heap_noncoalesced<value_type, 24><<< num_blocks, num_threads >>>(_d_queue_one,
                                           _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                           _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                           _d_csrRowPtrC, _d_csrRowPtrCt, _d_csrColIndCt, _d_csrValCt,
                                           counter, position);
        break;
    case 25:
        ESC_2heap_noncoalesced<value_type, 25><<< num_blocks, num_threads >>>(_d_queue_one,
                                           _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                           _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                           _d_csrRowPtrC, _d_csrRowPtrCt, _d_csrColIndCt, _d_csrValCt,
                                           counter, position);
        break;
    case 26:
        ESC_2heap_noncoalesced<value_type, 26><<< num_blocks, num_threads >>>(_d_queue_one,
                                           _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                           _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                           _d_csrRowPtrC, _d_csrRowPtrCt, _d_csrColIndCt, _d_csrValCt,
                                           counter, position);
        break;
    case 27:
        ESC_2heap_noncoalesced<value_type, 27><<< num_blocks, num_threads >>>(_d_queue_one,
                                           _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                           _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                           _d_csrRowPtrC, _d_csrRowPtrCt, _d_csrColIndCt, _d_csrValCt,
                                           counter, position);
        break;
    case 28:
        ESC_2heap_noncoalesced<value_type, 28><<< num_blocks, num_threads >>>(_d_queue_one,
                                           _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                           _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                           _d_csrRowPtrC, _d_csrRowPtrCt, _d_csrColIndCt, _d_csrValCt,
                                           counter, position);
        break;
    case 29:
        ESC_2heap_noncoalesced<value_type, 29><<< num_blocks, num_threads >>>(_d_queue_one,
                                           _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                           _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                           _d_csrRowPtrC, _d_csrRowPtrCt, _d_csrColIndCt, _d_csrValCt,
                                           counter, position);
        break;
    case 30:
        ESC_2heap_noncoalesced<value_type, 30><<< num_blocks, num_threads >>>(_d_queue_one,
                                           _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                           _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                           _d_csrRowPtrC, _d_csrRowPtrCt, _d_csrColIndCt, _d_csrValCt,
                                           counter, position);
        break;
    case 31:
        ESC_2heap_noncoalesced<value_type, 31><<< num_blocks, num_threads >>>(_d_queue_one,
                                           _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                           _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                           _d_csrRowPtrC, _d_csrRowPtrCt, _d_csrColIndCt, _d_csrValCt,
                                           counter, position);
        break;
    case 32:
        ESC_2heap_noncoalesced<value_type, 32><<< num_blocks, num_threads >>>(_d_queue_one,
                                           _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                           _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                           _d_csrRowPtrC, _d_csrRowPtrCt, _d_csrColIndCt, _d_csrValCt,
                                           counter, position);
        break;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {  cout << "err = " << cudaGetErrorString(err) << endl; return -1; }

    return BHSPARSE_SUCCESS;
}

__inline__ __device__
void coex(int      *keyA,
          value_type    *valA,
          int      *keyB,
          value_type    *valB,
          const int dir)
{
    int t;
    value_type v;

    if ((*keyA > *keyB) == dir)
    {
        t = *keyA;
        *keyA = *keyB;
        *keyB = t;
        v = *valA;
        *valA = *valB;
        *valB = v;
    }
}

__inline__ __device__
void oddeven(int   *s_key,
                     value_type *s_val,
                     int    arrayLength)
{
    int dir = 1;

    for (int size = 2; size <= arrayLength; size <<= 1)
    {
        int stride = size >> 1;
        int offset = threadIdx.x & (stride - 1);

        {
            __syncthreads();
            int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            coex(&s_key[pos], &s_val[pos], &s_key[pos + stride], &s_val[pos + stride], dir);

            stride >>= 1;
        }

        for (; stride > 0; stride >>= 1)
        {
            __syncthreads();
            int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            if (offset >= stride)
                coex(&s_key[pos - stride], &s_val[pos - stride], &s_key[pos], &s_val[pos], dir);
        }
    }
}

template<typename T>
__inline__ __device__
T scan_32_shfl(T x, const int local_id)
{
    #pragma unroll
    for( int offset = 1 ; offset < WARPSIZE_NV ; offset <<= 1 )
    {
        T y = __shfl_up(x, offset);
        if(local_id >= offset)
            x += y;
    }
    return x;
}

template<typename T>
__inline__ __device__
void scan_single( volatile  T *s_scan,
              const int      local_id,
              const int      l)
{
    T old_val, new_val;
    if (!local_id)
    {
        old_val = s_scan[0];
        s_scan[0] = 0;
        for (int i = 1; i < l; i++)
        {
            new_val = s_scan[i];
            s_scan[i] = old_val + s_scan[i-1];
            old_val = new_val;
        }
    }
}

template<typename T>
__inline__ __device__
T scan_plus1_shfl(volatile  T *s_scan,
                  const int     local_id,
                  T r_in,
                  const int seg_num)
{
    // 3-stage method. scan-scan-propogate

    // shfl version
    const int lane_id = local_id % WARPSIZE_NV;
    const int seg_id = local_id / WARPSIZE_NV;

    // stage 1. thread bunch scan
    T r_scan = 0;

    //if (seg_id < seg_num)
    //{
        r_scan = scan_32_shfl<T>(r_in, lane_id);

        if (lane_id == WARPSIZE_NV - 1)
            s_scan[seg_id] = r_scan;

        r_scan = __shfl_up(r_scan, 1);
        r_scan = lane_id ? r_scan : 0;
    //}

    __syncthreads();

    // stage 2. one thread bunch scan
    r_in = (local_id < seg_num) ? s_scan[local_id] : 0;
    if (!seg_id)
        r_in = scan_32_shfl<T>(r_in, lane_id);

    if (local_id < seg_num)
        s_scan[local_id + 1] = r_in;

    // single thread in-place scan
    //scan_single<T>(s_scan, local_id, seg_num+1);

    __syncthreads();

    // stage 3. propogate (element-wise add) to all
    if (seg_id) // && seg_id < seg_num)
        r_scan += s_scan[seg_id];

    return r_scan;
}

template<typename sT, typename T>
__inline__ __device__
void scan_double_width_plus1_shfl(volatile  sT *s_scan,
                                  volatile  T *s_scan_shfl,
                                  const int     local_id,
                                  T r_in,
                                  T r_in_halfwidth,
                                  const int seg_num)
{
    // 3-stage method. scan-scan-propogate

    // shfl version
    const int lane_id = local_id % WARPSIZE_NV;
    const int seg_id = local_id / WARPSIZE_NV;

    // stage 1. thread bunch scan
    T r_scan = scan_32_shfl<T>(r_in, lane_id);
    T r_scan_halfwidth = scan_32_shfl<T>(r_in_halfwidth, lane_id);

    if (lane_id == WARPSIZE_NV - 1)
    {
        s_scan_shfl[seg_id] = r_scan;
        s_scan_shfl[seg_id + seg_num] = r_scan_halfwidth;
    }

    // inclusive to exclusive
    r_scan = __shfl_up(r_scan, 1);
    r_scan_halfwidth = __shfl_up(r_scan_halfwidth, 1);
    r_scan = lane_id ? r_scan : 0;
    r_scan_halfwidth = lane_id ? r_scan_halfwidth : 0;

    __syncthreads();

    // stage 2. one thread bunch scan
    r_in = (local_id < 2 * seg_num) ? s_scan_shfl[local_id] : 0;
    if (!seg_id)
        r_in = scan_32_shfl<T>(r_in, lane_id);

    if (local_id < 2 * seg_num)
        s_scan_shfl[local_id + 1] = r_in;

    // single thread in-place scan
    //scan_single<T>(s_scan_shfl, local_id, seg_num+1);

    __syncthreads();

    // stage 3. propogate (element-wise add) to all
    if (seg_id)
    {
        r_scan += s_scan_shfl[seg_id];
    }
    r_scan_halfwidth += s_scan_shfl[seg_id + seg_num];

    s_scan[local_id] = r_scan;
    s_scan[local_id + blockDim.x] = r_scan_halfwidth;
    if (!local_id)
        s_scan[2 * blockDim.x] = s_scan_shfl[2 * seg_num];

    return;
}

__inline__ __device__
void scan_32(volatile short *s_scan)
{
    int ai, bi;
    int baseai = 1 + 2 * threadIdx.x;
    int basebi = baseai + 1;
    short temp;

    if (threadIdx.x < 16)  { ai = baseai - 1;     bi = basebi - 1;     s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x < 8)  { ai = 2 * baseai - 1;  bi = 2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x < 4)  { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x < 2)  { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x == 0) { s_scan[31] += s_scan[15]; s_scan[32] = s_scan[31]; s_scan[31] = 0; temp = s_scan[15]; s_scan[15] = 0; s_scan[31] += temp; }
    if (threadIdx.x < 2)  { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 4)  { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 8)  { ai = 2 * baseai - 1;  bi = 2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 16)  { ai = baseai - 1;   bi = basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp; }
}

__inline__ __device__
void scan_64(volatile short *s_scan)
{
    int ai, bi;
    int baseai = 1 + 2 * threadIdx.x;
    int basebi = baseai + 1;
    short temp;

    if (threadIdx.x < 32) { ai = baseai - 1;     bi = basebi - 1;     s_scan[bi] += s_scan[ai]; }
    __syncthreads();
    if (threadIdx.x < 16) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x < 8)  { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x < 4)  { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x < 2)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x == 0) { s_scan[63] += s_scan[31]; s_scan[64] = s_scan[63]; s_scan[63] = 0; temp = s_scan[31]; s_scan[31] = 0; s_scan[63] += temp; }
    if (threadIdx.x < 2)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 4)  { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 8)  { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 16) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 32) { ai = baseai - 1;   bi = basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp; }
}

__inline__ __device__
void scan_128(volatile short *s_scan)
{
    int ai, bi;
    int baseai = 1 + 2 * threadIdx.x;
    int basebi = baseai + 1;
    short temp;

    if (threadIdx.x < 64) { ai = baseai - 1;     bi = basebi - 1;     s_scan[bi] += s_scan[ai]; }
    __syncthreads();
    if (threadIdx.x < 32) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    __syncthreads();
    if (threadIdx.x < 16) { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x < 8)  { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x < 4)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x < 2)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x == 0) { s_scan[127] += s_scan[63]; s_scan[128] = s_scan[127]; s_scan[127] = 0; temp = s_scan[63]; s_scan[63] = 0; s_scan[127] += temp; }
    if (threadIdx.x < 2)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 4)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 8)  { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 16) { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 32) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    __syncthreads();
    if (threadIdx.x < 64) { ai = baseai - 1;   bi = basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp; }
}

__inline__ __device__
void scan_256(volatile short *s_scan)
{
    int ai, bi;
    int baseai = 1 + 2 * threadIdx.x;
    int basebi = baseai + 1;
    short temp;

    if (threadIdx.x < 128) { ai = baseai - 1;     bi = basebi - 1;     s_scan[bi] += s_scan[ai]; }
    __syncthreads();
    if (threadIdx.x < 64) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    __syncthreads();
    if (threadIdx.x < 32) { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    __syncthreads();
    if (threadIdx.x < 16) { ai =  8 * baseai - 1;  bi =  8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x < 8)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x < 4)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x < 2)  { ai = 64 * baseai - 1;  bi = 64 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x == 0) { s_scan[255] += s_scan[127]; s_scan[256] = s_scan[255]; s_scan[255] = 0; temp = s_scan[127]; s_scan[127] = 0; s_scan[255] += temp; }
    if (threadIdx.x < 2)  { ai = 64 * baseai - 1;  bi = 64 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 4)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 8)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 16) { ai =  8 * baseai - 1;  bi =  8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 32) { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    __syncthreads();
    if (threadIdx.x < 64) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    __syncthreads();
    if (threadIdx.x < 128) { ai = baseai - 1;   bi = basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp; }
}

__inline__ __device__
void scan_512(volatile short *s_scan)
{
    int ai, bi;
    int baseai = 1 + 2 * threadIdx.x;
    int basebi = baseai + 1;
    short temp;

    if (threadIdx.x < 256) { ai = baseai - 1;     bi = basebi - 1;     s_scan[bi] += s_scan[ai]; }
    __syncthreads();
    if (threadIdx.x < 128) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    __syncthreads();
    if (threadIdx.x < 64)  { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    __syncthreads();
    if (threadIdx.x < 32) { ai =  8 * baseai - 1;  bi =  8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    __syncthreads();
    if (threadIdx.x < 16) { ai =  16 * baseai - 1;  bi =  16 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x < 8)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x < 4)  { ai = 64 * baseai - 1;  bi = 64 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x < 2)  { ai = 128 * baseai - 1;  bi = 128 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x == 0) { s_scan[511] += s_scan[255]; s_scan[512] = s_scan[511]; s_scan[511] = 0; temp = s_scan[255]; s_scan[255] = 0; s_scan[511] += temp; }
    if (threadIdx.x < 2)  { ai = 128 * baseai - 1;  bi = 128 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 4)  { ai = 64 * baseai - 1;  bi = 64 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 8)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 16) { ai =  16 * baseai - 1;  bi =  16 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 32) { ai =  8 * baseai - 1;  bi =  8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    __syncthreads();
    if (threadIdx.x < 64) { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    __syncthreads();
    if (threadIdx.x < 128) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    __syncthreads();
    if (threadIdx.x < 256) { ai = baseai - 1;   bi = basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp; }
}


__inline__ __device__
void compression_scan(volatile short *s_scan,
                      volatile int   *s_scan_shfl,
                      int            *s_key,
                      value_type     *s_val,
                      const int       local_counter,
                      const int       local_size,
                      const int       local_id,
                      const int       local_id_halfwidth)
{
    // compression - prefix sum
    bool duplicate = 1;
    bool duplicate_halfwidth = 1;

    // generate bool value in registers
    if (local_id < local_counter && local_id > 0)
        duplicate = (s_key[local_id] != s_key[local_id - 1]);
    if (local_id_halfwidth < local_counter)
        duplicate_halfwidth = (s_key[local_id_halfwidth] != s_key[local_id_halfwidth - 1]);

#if __CUDA_ARCH__ >= 300
    scan_double_width_plus1_shfl<short, int>(s_scan, s_scan_shfl, local_id,
                                             duplicate, duplicate_halfwidth, local_size/WARPSIZE_NV);
#else
    // copy bool values from register to local memory (s_scan)
    s_scan[local_id]                    = duplicate;
    s_scan[local_id_halfwidth]          = duplicate_halfwidth;
    __syncthreads();

    // in-place exclusive prefix-sum scan on s_scan
    switch (local_size)
    {
    case 16:
        scan_32(s_scan);
        break;
    case 32:
        scan_64(s_scan);
        break;
    case 64:
        scan_128(s_scan);
        break;
    case 128:
        scan_256(s_scan);
        break;
    case 256:
        scan_512(s_scan);
        break;
    }
#endif

    __syncthreads();

    // compute final position and final value in registers
    int   move_pointer;
    short final_position, final_position_halfwidth;
    int   final_key,      final_key_halfwidth;
    value_type final_value,    final_value_halfwidth;

    if (local_id < local_counter && duplicate == 1)
    {
        final_position = s_scan[local_id];
        final_key = s_key[local_id];
        final_value = s_val[local_id];
        move_pointer = local_id + 1;

        while (s_scan[move_pointer] == s_scan[move_pointer + 1])
        {
            final_value += s_val[move_pointer];
            move_pointer++;
        }
    }

    if (local_id_halfwidth < local_counter && duplicate_halfwidth == 1)
    {
        final_position_halfwidth = s_scan[local_id_halfwidth];
        final_key_halfwidth = s_key[local_id_halfwidth];
        final_value_halfwidth = s_val[local_id_halfwidth];
        move_pointer = local_id_halfwidth + 1;

        while (s_scan[move_pointer] == s_scan[move_pointer + 1] && move_pointer < 2 * local_size)
        {
            final_value_halfwidth += s_val[move_pointer];
            move_pointer++;
        }
    }
    __syncthreads();

    // write final_positions and final_values to s_key and s_val
    if (local_id < local_counter && duplicate == 1)
    {
        s_key[final_position] = final_key;
        s_val[final_position] = final_value;
    }
    if (local_id_halfwidth < local_counter && duplicate_halfwidth == 1)
    {
        s_key[final_position_halfwidth] = final_key_halfwidth;
        s_val[final_position_halfwidth] = final_value_halfwidth;
    }
}

template<typename vT, int c_scansize>
__global__
void ESC_bitonic_scan(const int*     d_queue,
                      const int*     d_csrRowPtrA,
                      const  int*   __restrict__  d_csrColIndA,
                      const  vT*   __restrict__   d_csrValA,
                      const int*     d_csrRowPtrB,
                      const int*     d_csrColIndB,
                      const vT*      d_csrValB,
                      int*           d_csrRowPtrC,
                      int*           d_csrColIndCt,
                      vT*            d_csrValCt,
                      const int      queue_offset,
                      const int      n)
{
    __shared__ int   s_key[2 * c_scansize];
    __shared__ vT s_val[2 * c_scansize];
    __shared__ short s_scan[2 * c_scansize + 1];
#if __CUDA_ARCH__ >= 300
    volatile __shared__ int s_scan_shfl[2 * c_scansize / WARPSIZE_NV + 1];
#else
    volatile __shared__ int *s_scan_shfl;
#endif

    int local_id = threadIdx.x;
    int group_id = blockIdx.x;
    int local_size = blockDim.x;
    int width = local_size * 2;

    int i, local_counter = 0;
    int strideB, local_offset, global_offset;
    int invalid_width;
    int local_id_halfwidth = local_id + local_size;

    int row_id_B; // index_type

    int row_id;// index_type
    row_id = d_queue[TUPLE_QUEUE * (queue_offset + group_id)];

    int start_col_index_A, stop_col_index_A;  // index_type
    int start_col_index_B, stop_col_index_B;  // index_type
    vT value_A;                            // value_type

    start_col_index_A = d_csrRowPtrA[row_id];
    stop_col_index_A  = d_csrRowPtrA[row_id + 1];

    // i is both col index of A and row index of B
    for (i = start_col_index_A; i < stop_col_index_A; i++)
    {
        row_id_B = d_csrColIndA[i];
        value_A  = d_csrValA[i];

        start_col_index_B = d_csrRowPtrB[row_id_B];
        stop_col_index_B  = d_csrRowPtrB[row_id_B + 1];

        strideB = stop_col_index_B - start_col_index_B;

        if (local_id < strideB)
        {
            local_offset = local_counter + local_id;
            global_offset = start_col_index_B + local_id;

            s_key[local_offset] = d_csrColIndB[global_offset];
            s_val[local_offset] = d_csrValB[global_offset] * value_A;
        }

        if (local_id_halfwidth < strideB)
        {
            local_offset = local_counter + local_id_halfwidth;
            global_offset = start_col_index_B + local_id_halfwidth;

            s_key[local_offset] = d_csrColIndB[global_offset];
            s_val[local_offset] = d_csrValB[global_offset] * value_A;
        }

        local_counter += strideB;
    }
    __syncthreads();

    invalid_width = width - local_counter;

    // to meet 2^N, set the rest elements to n (number of columns of C)
    if (local_id < invalid_width)
        s_key[local_counter + local_id] = n;
    //if (local_id_halfwidth < invalid_width)
    //    s_key[local_counter + local_id_halfwidth] = n;
    __syncthreads();

    // bitonic sort
    oddeven(s_key, s_val, width);
    __syncthreads();

    // compression - scan
    compression_scan(s_scan, s_scan_shfl, s_key, s_val, local_counter,
                     local_size, local_id, local_id_halfwidth);
    __syncthreads();

    local_counter = s_scan[width] - invalid_width;
    if (local_id == 0)
        d_csrRowPtrC[row_id] = local_counter;

    // write compressed lists to global mem
    int row_offset = d_queue[TUPLE_QUEUE * (queue_offset + group_id) + 1]; //d_csrRowPtrCt[row_id];

    if (local_id < local_counter)
    {
        global_offset = row_offset + local_id;

        d_csrColIndCt[global_offset] = s_key[local_id];
        d_csrValCt[global_offset] = s_val[local_id];
    }
    if (local_id_halfwidth < local_counter)
    {
        global_offset = row_offset + local_id_halfwidth;

        d_csrColIndCt[global_offset] = s_key[local_id_halfwidth];
        d_csrValCt[global_offset] = s_val[local_id_halfwidth];
    }
}

int bhsparse_cuda::compute_nnzC_Ct_bitonic(int num_threads, int num_blocks, int j, int position)
{
    cudaError_t err = cudaSuccess;

    switch (num_threads)
    {
    case 16:
        ESC_bitonic_scan<value_type, 16><<< num_blocks, num_threads >>>(_d_queue_one, _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                           _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                           _d_csrRowPtrC, _d_csrColIndCt, _d_csrValCt,
                                           position, _n);
        break;
    case 32:
        ESC_bitonic_scan<value_type, 32><<< num_blocks, num_threads >>>(_d_queue_one, _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                           _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                           _d_csrRowPtrC, _d_csrColIndCt, _d_csrValCt,
                                           position, _n);
        break;
    case 64:
        ESC_bitonic_scan<value_type, 64><<< num_blocks, num_threads >>>(_d_queue_one, _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                           _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                           _d_csrRowPtrC, _d_csrColIndCt, _d_csrValCt,
                                           position, _n);
        break;
    case 128:
        ESC_bitonic_scan<value_type, 128><<< num_blocks, num_threads >>>(_d_queue_one, _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                           _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                           _d_csrRowPtrC, _d_csrColIndCt, _d_csrValCt,
                                           position, _n);
        break;
    case 256:
        ESC_bitonic_scan<value_type, 256><<< num_blocks, num_threads >>>(_d_queue_one, _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                           _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                           _d_csrRowPtrC, _d_csrColIndCt, _d_csrValCt,
                                           position, _n);
        break;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {  cout << "err = " << cudaGetErrorString(err) << endl; return -1; }


    return BHSPARSE_SUCCESS;
}

__global__
void ESC_0_cudakernel(const int*     d_queue,
                      int*                       d_csrRowPtrC,
                      const int                  queue_size,
                      const int                  queue_offset)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_id < queue_size)
    {
        int row_id = d_queue[TUPLE_QUEUE * (queue_offset + global_id)];
        d_csrRowPtrC[row_id] = 0;
    }
}

__global__
void ESC_1_cudakernel(const int*     d_queue,
                      const int*     d_csrRowPtrA,
                      const  int*   __restrict__  d_csrColIndA,
                      const  value_type* __restrict__  d_csrValA,
                      const int*     d_csrRowPtrB,
                      const int*     d_csrColIndB,
                      const value_type*   d_csrValB,
                      int*                       d_csrRowPtrC,
                      const int*     d_csrRowPtrCt,
                      int*                       d_csrColIndCt,
                      value_type*                     d_csrValCt,
                      const int                  queue_size,
                      const int                  queue_offset)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_id < queue_size)
    {
        int row_id = d_queue[TUPLE_QUEUE * (queue_offset + global_id)];
        d_csrRowPtrC[row_id] = 1;

        int base_index = d_queue[TUPLE_QUEUE * (queue_offset + global_id) + 1]; //d_csrRowPtrCt[row_id];

        int col_index_A_start = d_csrRowPtrA[row_id];
        int col_index_A_stop = d_csrRowPtrA[row_id+1];

        for (int col_index_A = col_index_A_start; col_index_A < col_index_A_stop; col_index_A++)
        {
            int row_id_B = d_csrColIndA[col_index_A];
            int col_index_B = d_csrRowPtrB[row_id_B];

            if (col_index_B == d_csrRowPtrB[row_id_B+1])
                continue;

            value_type value_A  = d_csrValA[col_index_A];

            d_csrColIndCt[base_index] = d_csrColIndB[col_index_B];
            d_csrValCt[base_index] = d_csrValB[col_index_B] * value_A;

            break;
        }
    }
}

int bhsparse_cuda::compute_nnzC_Ct_0(int num_threads, int num_blocks, int j, int counter, int position)
{
    cudaError_t err = cudaSuccess;

    ESC_0_cudakernel<<< num_blocks, num_threads >>>(_d_queue_one, _d_csrRowPtrC, counter, position);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {  cout << "err = " << cudaGetErrorString(err) << endl; return -1; }

    return BHSPARSE_SUCCESS;
}

int bhsparse_cuda::compute_nnzC_Ct_1(int num_threads, int num_blocks, int j, int counter, int position)
{
    cudaError_t err = cudaSuccess;

    ESC_1_cudakernel<<< num_blocks, num_threads >>>(_d_queue_one,
                                       _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                       _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                       _d_csrRowPtrC, _d_csrRowPtrCt, _d_csrColIndCt, _d_csrValCt,
                                       counter, position);


    err = cudaGetLastError();
    if (err != cudaSuccess)
    {  cout << "err = " << cudaGetErrorString(err) << endl; return -1; }

    return BHSPARSE_SUCCESS;
}

__inline__ __device__
void binarysearch_sub(int         *s_key,
                  value_type  *s_val,
                  int          key_input,
                  value_type   val_input,
                  int          merged_size)
{
    int start = 0;
    int stop  = merged_size - 1;
    int median;
    int key_median;

    while (stop >= start)
    {
        median = (stop + start) / 2;
        key_median = s_key[median];

        if (key_input > key_median)
            start = median + 1;
        else if (key_input < key_median)
            stop = median - 1;
        else
        {
            // atomicAdd is not needed since duplicate is not existed in each input row
            s_val[median] -= val_input;
            break;
        }
    }
    //return start;
}

__inline__ __device__
void binarysearch(int         *s_key,
                  value_type  *s_val,
                  int          key_input,
                  value_type   val_input,
                  int          merged_size,
                  bool        *is_new_col)
{
    int start = 0;
    int stop  = merged_size - 1;
    int median;
    int key_median;

    while (stop >= start)
    {
        median = (stop + start) / 2;
        key_median = s_key[median];

        if (key_input > key_median)
            start = median + 1;
        else if (key_input < key_median)
            stop = median - 1;
        else
        {
            // atomicAdd is not needed since duplicate is not existed in each input row
            s_val[median] += val_input;
            *is_new_col = 0;
            break;
        }
    }
    //return start;
}

__inline__ __device__
void scan(volatile short *s_scan)
{
    switch (blockDim.x)
    {
    case 32:
        scan_32(s_scan);
        break;
    case 64:
        scan_64(s_scan);
        break;
    case 128:
        scan_128(s_scan);
        break;
    case 256:
        scan_256(s_scan);
        break;
    case 512:
        scan_512(s_scan);
        break;
    }
}

__inline__ __device__
bool comp(int a, int b)
{
    return a < b ? true : false;
}

__inline__ __device__
int y_pos(const int x_pos,
          const int b_length,
          const int offset)
{
    int pos = b_length - (x_pos + b_length - offset); //offset - x_pos
    return pos > b_length ? b_length : pos;
}

__inline__ __device__
int mergepath_partition(int  *a,
                        const int      aCount,
                        int  *b,
                        const int      bCount,
                        const int      diag)
{
    int begin = max(0, diag - bCount);
    int end   = min(diag, aCount);
    int mid;
    int key_a, key_b;
    bool pred;

    while(begin < end)
    {
        mid = (begin + end) >> 1;

        key_a = a[mid];
        key_b = b[diag - 1 - mid];

        pred = comp(key_a, key_b);

        if(pred)
            begin = mid + 1;
        else
            end = mid;
    }
    return begin;
}

__inline__ __device__
void mergepath_serialmerge(int   *s_key,
                           value_type *s_val,
                           int            aBegin,
                           const int      aEnd,
                           int            bBegin,
                           const int      bEnd,
                           int           *reg_key,
                           value_type         *reg_val,
                           const int      VT)
{
    int  key_a = s_key[aBegin];
    int  key_b = s_key[bBegin];
    bool p;

    for(int i = 0; i < VT; ++i)
    {
        p = (bBegin >= bEnd) || ((aBegin < aEnd) && !comp(key_b, key_a));

        reg_key[i] = p ? key_a : key_b;
        reg_val[i] = p ? s_val[aBegin] : s_val[bBegin];

        if(p)
            key_a = s_key[++aBegin];
        else
            key_b = s_key[++bBegin];
    }
}

__inline__ __device__
void readwrite_mergedlist(int   *d_csrColIndCt,
                          value_type *d_csrValCt,
                          int   *s_key_merged,
                          value_type *s_val_merged,
                          const int       merged_size,
                          const int       row_offset,
                          const bool      is_write)
{
    int stride, offset_local_id, global_offset;
    int loop = ceil((float)merged_size / (float)blockDim.x);

    for (int i = 0; i < loop; i++)
    {
        stride = i != loop - 1 ? blockDim.x : merged_size - i * blockDim.x;
        offset_local_id = i * blockDim.x + threadIdx.x;
        global_offset = row_offset + offset_local_id;

        if (threadIdx.x < stride)
        {
            if (is_write)
            {
                d_csrColIndCt[global_offset] = s_key_merged[offset_local_id];
                d_csrValCt[global_offset]    = s_val_merged[offset_local_id];
            }
            else
            {
                s_key_merged[offset_local_id] = d_csrColIndCt[global_offset];
                s_val_merged[offset_local_id] = d_csrValCt[global_offset];
            }
        }
    }
}

template<typename vT, int c_buffsize, int c_scansize>
__global__
void EM_mergepath(int            * d_queue,
                  const int      *  d_csrRowPtrA,
                  const  int      * __restrict__ d_csrColIndA,
                  const  vT    * __restrict__ d_csrValA,
                  const int      *  d_csrRowPtrB,
                  const int      *  d_csrColIndB,
                  const vT    *  d_csrValB,
                  int            *d_csrRowPtrC,
                  int            *d_csrColIndCt,
                  vT          *d_csrValCt,
                  const int                queue_offset)
{
    __shared__ int   s_key_merged[c_buffsize+1];
    __shared__ vT s_val_merged[c_buffsize+1];

#if __CUDA_ARCH__ >= 300
    int seg_num = c_scansize / WARPSIZE_NV;
    volatile __shared__ int s_scan[c_scansize / WARPSIZE_NV + 1];
#else
    volatile __shared__ short s_scan[c_scansize+1];
#endif

    //volatile __shared__ short s_a_border[c_scansize+1];
    //volatile __shared__ short s_b_border[c_scansize+1];

    const int queue_id = TUPLE_QUEUE * (queue_offset + blockIdx.x);

    // if merged size equals -1, kernel return since this row is done
    int merged_size = d_queue[queue_id + 2];

    const int local_id = threadIdx.x; //threadIdx.x;
    const int row_id = d_queue[queue_id];

    const int   local_size = blockDim.x;
    const float local_size_value_type = local_size;

    int reg_reuse1;

    int   col_Ct;      // index_type
    vT val_Ct;      // value_type
    vT val_A;       // value_type

    int start_col_index_A, stop_col_index_A;  // index_type
    int start_col_index_B, stop_col_index_B;  // index_type

    bool  is_new_col;
    bool  is_last;
    int   VT, diag, mp;
    int   reg_key[9];
    vT reg_val[9];

    start_col_index_A = d_csrRowPtrA[row_id];
    stop_col_index_A  = d_csrRowPtrA[row_id + 1];

    if (merged_size == 0)
    {
        is_last = false;

        // read the first set of current nnzCt row to merged list
        reg_reuse1 = d_csrColIndA[start_col_index_A];      // reg_reuse1 = row_id_B
        val_A    = d_csrValA[start_col_index_A];

        start_col_index_B = d_csrRowPtrB[reg_reuse1];      // reg_reuse1 = row_id_B
        stop_col_index_B  = d_csrRowPtrB[reg_reuse1 + 1];  // reg_reuse1 = row_id_B

        const int stride = stop_col_index_B - start_col_index_B;
        const int loop   = ceil((float)stride / local_size_value_type); //ceil((value_type)stride / (value_type)local_size);

        start_col_index_B += local_id;

        for (int k = 0; k < loop; k++)
        {
            reg_reuse1 = k != loop - 1 ? local_size : stride - k * local_size; // reg_reuse1 = input_size

            // if merged_size + reg_reuse1 > c_buffsize, write it to global mem and return
            if (merged_size + reg_reuse1 > c_buffsize)
            {
                // write a signal to some place, not equals -1 means next round is needed
                if (local_id == 0)
                {
                    d_queue[queue_id + 2] = merged_size;
                    d_queue[queue_id + 3] = start_col_index_A;
                    d_queue[queue_id + 4] = start_col_index_B;
                }

                // dump current data to global mem
                reg_reuse1 = d_queue[queue_id + 1];
                readwrite_mergedlist(d_csrColIndCt, d_csrValCt, s_key_merged, s_val_merged, merged_size, reg_reuse1, 1);

                return;
            }

            if (start_col_index_B < stop_col_index_B)
            {
                col_Ct = d_csrColIndB[start_col_index_B];
                val_Ct = d_csrValB[start_col_index_B] * val_A;

                s_key_merged[merged_size + local_id] = col_Ct;
                s_val_merged[merged_size + local_id] = val_Ct;
            }
            __syncthreads();
            merged_size += reg_reuse1;   // reg_reuse1 = input_size
            start_col_index_B += local_size;
        }

        start_col_index_A++;
    }
    else
    {
        is_last = true;
        start_col_index_A = d_queue[queue_id + 3];

        // load existing merged list
        reg_reuse1 = d_queue[queue_id + 5];
        readwrite_mergedlist(d_csrColIndCt, d_csrValCt, s_key_merged, s_val_merged, merged_size, reg_reuse1, 0);
    }
    __syncthreads();

    // merge the rest of sets of current nnzCt row to the merged list
    while (start_col_index_A < stop_col_index_A)
    {
        reg_reuse1 = d_csrColIndA[start_col_index_A];                      // reg_reuse1 = row_id_B
        val_A    = d_csrValA[start_col_index_A];

        start_col_index_B = is_last ? d_queue[queue_id + 4] : d_csrRowPtrB[reg_reuse1];      // reg_reuse1 = row_id_B
        is_last = false;
        stop_col_index_B  = d_csrRowPtrB[reg_reuse1 + 1];  // reg_reuse1 = row_id_B

        const int stride = stop_col_index_B - start_col_index_B;
        const int loop  = ceil((float)stride / local_size_value_type); //ceil((value_type)stride / (value_type)local_size);

        //int start_col_index_B_zeropoint = start_col_index_B;
        start_col_index_B += local_id;

        for (int k = 0; k < loop; k++)
        {
            __syncthreads();
            is_new_col = 0;

            if (start_col_index_B < stop_col_index_B)
            {
                col_Ct = d_csrColIndB[start_col_index_B];
                val_Ct = d_csrValB[start_col_index_B] * val_A;

                // binary search on existing sorted list
                // if the column is existed, add the value to the position
                // else, set scan value to 1, and wait for scan
                is_new_col = 1;
                binarysearch(s_key_merged, s_val_merged, col_Ct, val_Ct, merged_size, &is_new_col);
            }

#if __CUDA_ARCH__ >= 300
            //const int seg_num = (k == loop - 1) ?
            //            ceil((float)(stop_col_index_B - start_col_index_B_zeropoint) / (float)WARPSIZE_NV) :
            //            local_size / WARPSIZE_NV;
            //if (!local_id)
            //    printf("blockIdx = %d, seg_num = %d\n", blockIdx.x, seg_num);
            int r_scan = scan_plus1_shfl<int>(s_scan, local_id, is_new_col, seg_num);
            const int s_scan_sum = s_scan[seg_num];
#else
            s_scan[local_id] = is_new_col;
            __syncthreads();

            // scan with half-local_size work-items
            // s_scan[local_size] is the size of input non-duplicate array
            scan(s_scan);
            __syncthreads();
            const int s_scan_sum = s_scan[local_size];
#endif

            // if all elements are absorbed into merged list,
            // the following work in this inner-loop is not needed any more
            if (s_scan_sum == 0)
            {
                start_col_index_B += local_size;
                //start_col_index_B_zeropoint += local_size;
                continue;
            }

            // check if the total size is larger than the capicity of merged list
            if (merged_size + s_scan_sum > c_buffsize)
            {
                // roll back 'binary serach plus' in this round
                if (start_col_index_B < stop_col_index_B)
                {
                    binarysearch_sub(s_key_merged, s_val_merged, col_Ct, val_Ct, merged_size);
                }
                __syncthreads();

                // write a signal to some place, not equals -1 means next round is needed
                if (local_id == 0)
                {
                    d_queue[queue_id + 2] = merged_size;
                    d_queue[queue_id + 3] = start_col_index_A;
                    d_queue[queue_id + 4] = start_col_index_B;
                }

                // dump current data to global mem
                reg_reuse1 = d_queue[queue_id + 1]; //d_csrRowPtrCt[row_id];
                readwrite_mergedlist(d_csrColIndCt, d_csrValCt, s_key_merged, s_val_merged, merged_size, reg_reuse1, 1);

                return;
            }

            // write compact input to free place in merged list
            if(is_new_col)
            {
#if __CUDA_ARCH__ >= 300
                reg_reuse1 = merged_size + r_scan;
#else
                reg_reuse1 = merged_size + s_scan[local_id];
#endif
                s_key_merged[reg_reuse1] = col_Ct;
                s_val_merged[reg_reuse1] = val_Ct;
            }
            __syncthreads();

            // merge path partition
            VT = ceil((float)(merged_size + s_scan_sum) / local_size_value_type);

            diag = VT * local_id;
            mp = mergepath_partition(s_key_merged, merged_size, &s_key_merged[merged_size], s_scan_sum, diag);

            mergepath_serialmerge(s_key_merged, s_val_merged,
                                  mp, merged_size, merged_size + diag - mp, merged_size + s_scan_sum,
                                  reg_key, reg_val, VT);
            __syncthreads();

            for (int is = 0; is < VT; is++)
            {
                s_key_merged[diag + is] = reg_key[is];
                s_val_merged[diag + is] = reg_val[is];
            }
            __syncthreads();

            merged_size += s_scan_sum;
            start_col_index_B += local_size;
            //start_col_index_B_zeropoint += local_size;
        }

        start_col_index_A++;
    }
    __syncthreads();

    if (local_id == 0)
    {
        d_csrRowPtrC[row_id] = merged_size;
        d_queue[queue_id + 2] = -1;
    }

    // write merged list to global mem
    reg_reuse1 = d_queue[queue_id + 1]; //d_csrRowPtrCt[row_id];
    readwrite_mergedlist(d_csrColIndCt, d_csrValCt, s_key_merged, s_val_merged, merged_size, reg_reuse1, 1);
}

template<typename vT>
__device__
void mergepath_global_2level_liu( int          *s_a_key, //__global
                                vT               *s_a_val, //__global
                               const int                 a_length,
                                int              *s_b_key, //__global
                                vT               *s_b_val, //__global
                               const int                 b_length,
                               int                      *reg_key,
                               vT                       *reg_val,
                                 int             *s_key, //__local
                                 vT              *s_val, //__local
                                int              *d_temp_key, //__global
                                vT               *d_temp_val) //__global
{
    if (b_length == 0)
        return;
    if (s_a_key[a_length-1] < s_b_key[0])
        return;
    __shared__ int s_a_border[WARPSIZE_NV];

    int local_id = threadIdx.x; //get_local_id(0);
    int local_size = blockDim.x; //get_local_size(0);
    int delta_2level = local_size * 9;
    int loop_2level = ceil((float)(a_length + b_length) / (float)delta_2level);
    int a_border_2level_l, b_border_2level_l, a_border_2level_r, b_border_2level_r;
    for (int i = 0; i < loop_2level; i++)
    {
        // compute `big' borders
        int offset_2level = delta_2level * i;
        a_border_2level_l = i == 0 ? 0 : a_border_2level_r; //mergepath_partition_global_liu(s_a_key, a_length, s_b_key, b_length, offset_2level);
        b_border_2level_l = i == 0 ? 0 : b_border_2level_r; //y_pos(a_border_2level_l, b_length, offset_2level);
        int offset_2level_next = delta_2level * (i + 1);
        if (i == (loop_2level - 1)){
            a_border_2level_r = a_length;
            b_border_2level_r = b_length;
        }
        else
        {
           //s_a_border[local_id] = a_border_2level_r = local_id < WARPSIZE_NV ? mergepath_partition(s_a_key, a_length, s_b_key, b_length, offset_2level_next) : 0;
           if (local_id < WARPSIZE_NV)
               s_a_border[local_id] = a_border_2level_r = mergepath_partition(s_a_key, a_length, s_b_key, b_length, offset_2level_next);
           __syncthreads();
           a_border_2level_r = local_id < WARPSIZE_NV ? a_border_2level_r : s_a_border[local_id % WARPSIZE_NV];
           b_border_2level_r = y_pos(a_border_2level_r, b_length, offset_2level_next);
        }
        //__syncthreads();
        // load entries in the borders
        int a_size = a_border_2level_r - a_border_2level_l;
        int b_size = b_border_2level_r - b_border_2level_l;
        for (int j = local_id; j < a_size; j += local_size)
        {
            s_key[j] = s_a_key[a_border_2level_l + j];
            s_val[j] = s_a_val[a_border_2level_l + j];
        }
        for (int j = local_id; j < b_size; j += local_size)
        {
            s_key[a_size + j] = s_b_key[b_border_2level_l + j];
            s_val[a_size + j] = s_b_val[b_border_2level_l + j];
        }
        __syncthreads();
        // merge path in local mem
//        mergepath_liu(s_key, s_val, a_size,
//                      &s_key[a_size], &s_val[a_size], b_size,
//                      s_a_border, s_b_border, reg_key, reg_val);
//        __syncthreads();


        // merge path partition on l1

        int VT = ceil((a_size + b_size) / (float)local_size);
        int diag = VT * local_id;
        int mp = mergepath_partition(s_key, a_size,
                                 &s_key[a_size], b_size, diag);

        mergepath_serialmerge(s_key, s_val,
                              mp, a_size, a_size + diag - mp, a_size + b_size,
                              reg_key, reg_val, VT);
        __syncthreads();

        for (int is = 0; is < VT; is++)
        {
            s_key[diag + is] = reg_key[is];
            s_val[diag + is] = reg_val[is];
        }
        __syncthreads();








        // dump the merged part to device mem (temp)
        for (int j = local_id; j < a_size + b_size; j += local_size)
        {
            d_temp_key[offset_2level + j] = s_key[j];
            d_temp_val[offset_2level + j] = s_val[j];
        }
        __syncthreads();
    }
    // dump the temp data to the target place, both in device mem
    for (int j = local_id; j < a_length + b_length; j += local_size)
    {
        s_a_key[j] = d_temp_key[j];
        s_a_val[j] = d_temp_val[j];
    }
    __syncthreads();
}

template<typename vT, int c_buffsize, int c_scansize>
__global__
void EM_mergepath_global(int            * d_queue,
                         const int      *  d_csrRowPtrA,
                         const  int      * __restrict__ d_csrColIndA,
                         const  vT    * __restrict__ d_csrValA,
                         const int      *  d_csrRowPtrB,
                         const int      *  d_csrColIndB,
                         const vT    *  d_csrValB,
                         int            *d_csrRowPtrC,
                         int            *d_csrColIndCt,
                         vT          *d_csrValCt,
                         const int       queue_offset)
{
    __shared__ int   s_key_merged_l1[c_buffsize+1];
    __shared__ vT s_val_merged_l1[c_buffsize+1];

#if __CUDA_ARCH__ >= 300
    const int seg_num = c_scansize / WARPSIZE_NV;
    volatile __shared__ int s_scan[c_scansize / WARPSIZE_NV + 1];
#else
    volatile __shared__ short s_scan[c_scansize+1];
#endif

    int queue_id = TUPLE_QUEUE * (queue_offset + blockIdx.x);

    // if merged size equals -1, kernel return since this row is done
    int merged_size_l2 = d_queue[queue_id + 2];
    int merged_size_l1 = 0;

    int local_id = threadIdx.x; //threadIdx.x;
    int row_id = d_queue[queue_id];

    int   local_size = blockDim.x;
    float local_size_value_type = local_size;

    int stride, loop;
    int reg_reuse1;

    int   col_Ct;      // index_type
    vT val_Ct;      // vT
    vT val_A;       // vT

    int start_col_index_A, stop_col_index_A;  // index_type
    int start_col_index_B, stop_col_index_B;  // index_type

    int k, is;

    bool  is_new_col;
    bool  is_last;
    int   VT, diag, mp;
    int   reg_key[9];
    vT reg_val[9];

    start_col_index_A = d_csrRowPtrA[row_id];
    stop_col_index_A  = d_csrRowPtrA[row_id + 1];

    is_last = true;
    start_col_index_A = d_queue[queue_id + 3];

    // load existing merged list
    reg_reuse1 = d_queue[queue_id + 1];
    int   *d_key_merged = &d_csrColIndCt[reg_reuse1];
    vT *d_val_merged = &d_csrValCt[reg_reuse1];

    reg_reuse1 = d_queue[queue_id + 5];
    readwrite_mergedlist(d_csrColIndCt, d_csrValCt, d_key_merged, d_val_merged, merged_size_l2, reg_reuse1, 0);
    __syncthreads();

    // merge the rest of sets of current nnzCt row to the merged list
    while (start_col_index_A < stop_col_index_A)
    {
        reg_reuse1 = d_csrColIndA[start_col_index_A];                      // reg_reuse1 = row_id_B
        val_A    = d_csrValA[start_col_index_A];

        start_col_index_B = is_last ? d_queue[queue_id + 4] : d_csrRowPtrB[reg_reuse1];      // reg_reuse1 = row_id_B
        is_last = false;
        stop_col_index_B  = d_csrRowPtrB[reg_reuse1 + 1];  // reg_reuse1 = row_id_B

        stride = stop_col_index_B - start_col_index_B;
        loop  = ceil(stride / local_size_value_type); //ceil((value_type)stride / (value_type)local_size);

        start_col_index_B += local_id;

        for (k = 0; k < loop; k++)
        {
            __syncthreads();
            is_new_col = 0;

            if (start_col_index_B < stop_col_index_B)
            {
                col_Ct = d_csrColIndB[start_col_index_B];
                val_Ct = d_csrValB[start_col_index_B] * val_A;

                // binary search on existing sorted list
                // if the column is existed, add the value to the position
                // else, set scan value to 1, and wait for scan
                is_new_col = 1;

                // search on l2
                binarysearch(d_key_merged, d_val_merged, col_Ct, val_Ct, merged_size_l2, &is_new_col);

                // search on l1
                if (is_new_col == 1)
                    binarysearch(s_key_merged_l1, s_val_merged_l1, col_Ct, val_Ct, merged_size_l1, &is_new_col);
            }

#if __CUDA_ARCH__ >= 300
            int r_scan = scan_plus1_shfl<int>(s_scan, local_id, is_new_col, seg_num);
            const int s_scan_sum = s_scan[seg_num];
#else
            s_scan[local_id] = is_new_col;
            __syncthreads();

            // scan with half-local_size work-items
            // s_scan[local_size] is the size of input non-duplicate array
            scan(s_scan);
            __syncthreads();
            const int s_scan_sum = s_scan[local_size];
#endif

            // if all elements are absorbed into merged list,
            // the following work in this inner-loop is not needed any more
            if (s_scan_sum == 0)
            {
                start_col_index_B += local_size;
                continue;
            }

            // check if the total size is larger than the capicity of merged list
            if (merged_size_l1 + s_scan_sum > c_buffsize)
            {
                if (start_col_index_B < stop_col_index_B)
                {
                    // rollback on l2
                    binarysearch_sub(d_key_merged, d_val_merged, col_Ct, val_Ct, merged_size_l2);

                    // rollback on l1
                    binarysearch_sub(s_key_merged_l1, s_val_merged_l1, col_Ct, val_Ct, merged_size_l1);
                }
                __syncthreads();

                // write a signal to some place, not equals -1 means next round is needed
                if (local_id == 0)
                {
                    d_queue[queue_id + 2] = merged_size_l2 + merged_size_l1;
                    d_queue[queue_id + 3] = start_col_index_A;
                    d_queue[queue_id + 4] = start_col_index_B;
                }

                // dump l1 to global
                readwrite_mergedlist(d_key_merged, d_val_merged, s_key_merged_l1, s_val_merged_l1,
                                     merged_size_l1, merged_size_l2, 1);
                __syncthreads();

//                // merge l2 + l1 on global
//                VT = ceil((merged_size_l2 + merged_size_l1) / local_size_value_type);
//                diag = VT * local_id;
//                mp = mergepath_partition(d_key_merged, merged_size_l2,
//                                                &d_key_merged[merged_size_l2], merged_size_l1, diag);

//                mergepath_serialmerge(d_key_merged, d_val_merged,
//                                      mp, merged_size_l2, merged_size_l2 + diag - mp, merged_size_l2 + merged_size_l1,
//                                      reg_key, reg_val, VT);
//                __syncthreads();

//                for (is = 0; is < VT; is++)
//                {
//                    d_key_merged[diag + is] = reg_key[is];
//                    d_val_merged[diag + is] = reg_val[is];
//                }

                mergepath_global_2level_liu<vT>(d_key_merged, d_val_merged, merged_size_l2,
                                      &d_key_merged[merged_size_l2], &d_val_merged[merged_size_l2], merged_size_l1,
                                      reg_key, reg_val,
                                      s_key_merged_l1, s_val_merged_l1,
                                      &d_key_merged[merged_size_l2 + merged_size_l1],
                                      &d_val_merged[merged_size_l2 + merged_size_l1]);

                return;
            }

            // write compact input to free place in merged list
            if(is_new_col)
            {
#if __CUDA_ARCH__ >= 300
                reg_reuse1 = merged_size_l1 + r_scan;
#else
                reg_reuse1 = merged_size_l1 + s_scan[local_id];
#endif
                s_key_merged_l1[reg_reuse1] = col_Ct;
                s_val_merged_l1[reg_reuse1] = val_Ct;
            }
            __syncthreads();

            // merge path partition on l1

            VT = ceil((merged_size_l1 + s_scan_sum) / local_size_value_type);
            diag = VT * local_id;
            mp = mergepath_partition(s_key_merged_l1, merged_size_l1,
                                     &s_key_merged_l1[merged_size_l1], s_scan_sum, diag);

            mergepath_serialmerge(s_key_merged_l1, s_val_merged_l1,
                                  mp, merged_size_l1, merged_size_l1 + diag - mp, merged_size_l1 + s_scan_sum,
                                  reg_key, reg_val, VT);
            __syncthreads();

            for (is = 0; is < VT; is++)
            {
                s_key_merged_l1[diag + is] = reg_key[is];
                s_val_merged_l1[diag + is] = reg_val[is];
            }
            __syncthreads();

            merged_size_l1 += s_scan_sum;
            start_col_index_B += local_size;
        }

        start_col_index_A++;
    }
    __syncthreads();

    if (local_id == 0)
    {
        d_csrRowPtrC[row_id] = merged_size_l2 + merged_size_l1;
        d_queue[queue_id + 2] = -1;
    }

    // dump l1 to global
    readwrite_mergedlist(d_key_merged, d_val_merged, s_key_merged_l1, s_val_merged_l1,
                         merged_size_l1, merged_size_l2, 1);
    __syncthreads();

//    // merge l2 + l1 on global
//    VT = ceil((merged_size_l2 + merged_size_l1) / local_size_value_type);
//    diag = VT * local_id;
//    mp = mergepath_partition(d_key_merged, merged_size_l2,
//                                    &d_key_merged[merged_size_l2], merged_size_l1, diag);

//    mergepath_serialmerge(d_key_merged, d_val_merged,
//                          mp, merged_size_l2, merged_size_l2 + diag - mp, merged_size_l2 + merged_size_l1,
//                          reg_key, reg_val, VT);
//    __syncthreads();

//    for (is = 0; is < VT; is++)
//    {
//        d_key_merged[diag + is] = reg_key[is];
//        d_val_merged[diag + is] = reg_val[is];
//    }
    mergepath_global_2level_liu<vT>(d_key_merged, d_val_merged, merged_size_l2,
                      &d_key_merged[merged_size_l2], &d_val_merged[merged_size_l2], merged_size_l1,
                      reg_key, reg_val,
                      s_key_merged_l1, s_val_merged_l1,
                      &d_key_merged[merged_size_l2 + merged_size_l1],
                      &d_val_merged[merged_size_l2 + merged_size_l1]);
}

int bhsparse_cuda::compute_nnzC_Ct_mergepath(int num_threads, int num_blocks, int j,
                                             int mergebuffer_size, int position, int *count_next, int mergepath_location)
{
    cudaError_t err = cudaSuccess;

    if (mergepath_location == MERGEPATH_LOCAL)
    {
        //cout << "doing merge with num_threads = " << num_threads << endl;
        switch (mergebuffer_size)
        {
        case 256:
            EM_mergepath<value_type, 256, 64><<< num_blocks, num_threads >>>(_d_queue_one,
                                   _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                   _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                   _d_csrRowPtrC, _d_csrColIndCt, _d_csrValCt, position);
            break;
        case 512:
            EM_mergepath<value_type, 512, 128><<< num_blocks, num_threads >>>(_d_queue_one,
                                   _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                   _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                   _d_csrRowPtrC, _d_csrColIndCt, _d_csrValCt, position);
            break;
        case 1024:
            EM_mergepath<value_type, 1024, 256><<< num_blocks, num_threads >>>(_d_queue_one,
                                   _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                   _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                   _d_csrRowPtrC, _d_csrColIndCt, _d_csrValCt, position);
            break;
        case 2048:
            EM_mergepath<value_type, 2048, 256><<< num_blocks, num_threads >>>(_d_queue_one,
                                   _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                   _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                   _d_csrRowPtrC, _d_csrColIndCt, _d_csrValCt, position);
            break;
        case 2304:
            EM_mergepath<value_type, 2304, 256><<< num_blocks, num_threads >>>(_d_queue_one,
                                   _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                   _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                   _d_csrRowPtrC, _d_csrColIndCt, _d_csrValCt, position);
            break;
        }
    }
    else if (mergepath_location == MERGEPATH_GLOBAL)
    {
        //cout << "EM_mergepath_global is called." << endl;
        EM_mergepath_global<value_type, 2304, 256><<< num_blocks, num_threads >>>(_d_queue_one,
                                       _d_csrRowPtrA, _d_csrColIndA, _d_csrValA,
                                       _d_csrRowPtrB, _d_csrColIndB, _d_csrValB,
                                       _d_csrRowPtrC, _d_csrColIndCt, _d_csrValCt, position);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {  cout << "err = " << cudaGetErrorString(err) << endl; return -1; }

    // load d_queue back, check if there is still any row needs next level merge,
    cudaMemcpy(&_h_queue_one[TUPLE_QUEUE * position],
                               &_d_queue_one[TUPLE_QUEUE * position],
                               TUPLE_QUEUE * num_blocks * sizeof(int),   cudaMemcpyDeviceToHost);


    int temp_queue [6] = {0, 0, 0, 0, 0, 0};
    int counter = 0;
    int temp_num = 0;
    for (int i = position; i < position + num_blocks; i++)
    {

        // if yes, (1)malloc device mem, (2)upgrade mem address on pos1 and (3)use pos5 as last mem address
        if (_h_queue_one[TUPLE_QUEUE * i + 2] != -1)
        {
            temp_queue[0] = _h_queue_one[TUPLE_QUEUE * i]; // row id
            if (mergepath_location == MERGEPATH_LOCAL || mergepath_location == MERGEPATH_LOCAL_L2)
            {
                int accum = 0;
                switch (mergebuffer_size)
                {
                case 256:
                    accum = 512;
                    break;
                case 512:
                    accum = 1024;
                    break;
                case 1024:
                    accum = 2048;
                    break;
                case 2048:
                    accum = 2304;
                    break;
                case 2304:
                    accum = 2 * (2304 * 2);
                    break;
                }

                temp_queue[1] = _nnzCt + counter * accum; // new start address
            }
            else if (mergepath_location == MERGEPATH_GLOBAL)
                temp_queue[1] = _nnzCt + counter * (2 * (mergebuffer_size + 2304)); // new start address
            //temp_queue[1] = _nnzCt + counter * mergebuffer_size * 2; // new start address
            temp_queue[2] = _h_queue_one[TUPLE_QUEUE * i + 2]; // merged size
            temp_queue[3] = _h_queue_one[TUPLE_QUEUE * i + 3]; // i
            temp_queue[4] = _h_queue_one[TUPLE_QUEUE * i + 4]; // k
            temp_queue[5] = _h_queue_one[TUPLE_QUEUE * i + 1]; // old start address

            _h_queue_one[TUPLE_QUEUE * i]     = _h_queue_one[TUPLE_QUEUE * (position + counter)];     // row id
            _h_queue_one[TUPLE_QUEUE * i + 1] = _h_queue_one[TUPLE_QUEUE * (position + counter) + 1]; // new start address
            _h_queue_one[TUPLE_QUEUE * i + 2] = _h_queue_one[TUPLE_QUEUE * (position + counter) + 2]; // merged size
            _h_queue_one[TUPLE_QUEUE * i + 3] = _h_queue_one[TUPLE_QUEUE * (position + counter) + 3]; // i
            _h_queue_one[TUPLE_QUEUE * i + 4] = _h_queue_one[TUPLE_QUEUE * (position + counter) + 4]; // k
            _h_queue_one[TUPLE_QUEUE * i + 5] = _h_queue_one[TUPLE_QUEUE * (position + counter) + 5]; // old start address

            _h_queue_one[TUPLE_QUEUE * (position + counter)]     = temp_queue[0]; // row id
            _h_queue_one[TUPLE_QUEUE * (position + counter) + 1] = temp_queue[1]; // new start address
            _h_queue_one[TUPLE_QUEUE * (position + counter) + 2] = temp_queue[2]; // merged size
            _h_queue_one[TUPLE_QUEUE * (position + counter) + 3] = temp_queue[3]; // i
            _h_queue_one[TUPLE_QUEUE * (position + counter) + 4] = temp_queue[4]; // k
            _h_queue_one[TUPLE_QUEUE * (position + counter) + 5] = temp_queue[5]; // old start address

            counter++;
            temp_num += _h_queue_one[TUPLE_QUEUE * i + 2];
        }
    }

    if (counter > 0)
    {
        //int nnzCt_new = _nnzCt + counter * mergebuffer_size * 2;
        int nnzCt_new;
        if (mergepath_location == MERGEPATH_LOCAL || mergepath_location == MERGEPATH_LOCAL_L2)
        {
            int accum = 0;
            switch (mergebuffer_size)
            {
            case 256:
                accum = 512;
                break;
            case 512:
                accum = 1024;
                break;
            case 1024:
                accum = 2048;
                break;
            case 2048:
                accum = 2304;
                break;
            case 2304:
                accum = 2 * (2304 * 2);
                break;
            }

            nnzCt_new = _nnzCt + counter * accum; //_nnzCt + counter * mergebuffer_size * 2;
        }
        else if (mergepath_location == MERGEPATH_GLOBAL)
            nnzCt_new = _nnzCt + counter * (2 * (mergebuffer_size + 2304));
        //cout << "nnzCt_new = " << nnzCt_new << endl;

        // malloc new device memory

        index_type *d_csrColIndCt_new;
        //checkCudaErrors(cudaMalloc((void **)&d_csrColIndCt_new, nnzCt_new  * sizeof(index_type)));
        err = cudaMalloc((void **)&d_csrColIndCt_new, nnzCt_new  * sizeof(index_type));

        if (err != cudaSuccess)
        {
            //cout << "errb = " << cudaGetErrorString(err) << ". malloc extra memory." << endl;
            index_type *h_csrColIndCt = (index_type *)malloc(_nnzCt  * sizeof(index_type));
            // copy last device mem to a temp space on host
            cudaMemcpy(h_csrColIndCt, _d_csrColIndCt, _nnzCt  * sizeof(index_type), cudaMemcpyDeviceToHost);
            //cout << "err1c = " << cudaGetErrorString(err) << ". ." << endl;
            //err = cudaDeviceSynchronize();
            // free last device mem
            cudaFree(_d_csrColIndCt);
            //cout << "err2c = " << cudaGetErrorString(err) << ". ." << endl;
            //err = cudaDeviceSynchronize();

            cudaMalloc((void **)&d_csrColIndCt_new,    nnzCt_new  * sizeof(index_type));
            //cout << "err3c = " << cudaGetErrorString(err) << ". ." << endl;

            // copy data in the temp space on host to device
            cudaMemcpy(d_csrColIndCt_new, h_csrColIndCt, _nnzCt  * sizeof(index_type), cudaMemcpyHostToDevice);
            //cout << "err4c = " << cudaGetErrorString(err) << ". ." << endl;

            free(h_csrColIndCt);
        }
        else
        {
            cudaMemcpy(d_csrColIndCt_new, _d_csrColIndCt, _nnzCt  * sizeof(index_type),   cudaMemcpyDeviceToDevice);
            cudaFree(_d_csrColIndCt);
        }

        _d_csrColIndCt = d_csrColIndCt_new;

        value_type *d_csrValCt_new;
        //checkCudaErrors(cudaMalloc((void **)&d_csrValCt_new,    nnzCt_new  * sizeof(value_type)));
        err = cudaMalloc((void **)&d_csrValCt_new,    nnzCt_new  * sizeof(value_type));
        if (err != cudaSuccess)
        {
            //cout << "erra = " << cudaGetErrorString(err) << ". malloc extra memory." << endl;
            value_type *h_csrValCt = (value_type *)malloc(_nnzCt  * sizeof(value_type));
            // copy last device mem to a temp space on host
            cudaMemcpy(h_csrValCt, _d_csrValCt, _nnzCt  * sizeof(value_type), cudaMemcpyDeviceToHost);
            //cout << "err1v = " << cudaGetErrorString(err) << ". ." << endl;
            //err = cudaDeviceSynchronize();

            // free last device mem
            cudaFree(_d_csrValCt);
            //cout << "err2v = " << cudaGetErrorString(err) << ". ." << endl;
            //err = cudaDeviceSynchronize();

            cudaMalloc((void **)&d_csrValCt_new,    nnzCt_new  * sizeof(value_type));
            //cout << "err3v = " << cudaGetErrorString(err) << ". ." << endl;

            // copy data in the temp space on host to device
            cudaMemcpy(d_csrValCt_new, h_csrValCt, _nnzCt  * sizeof(value_type), cudaMemcpyHostToDevice);
            //cout << "err4v = " << cudaGetErrorString(err) << ". ." << endl;

            free(h_csrValCt);
        }
        else
        {
            // copy last device mem to current one, device to device copy
            cudaMemcpy(d_csrValCt_new,    _d_csrValCt,    _nnzCt  * sizeof(value_type),   cudaMemcpyDeviceToDevice);
            // free last device mem
            cudaFree(_d_csrValCt);
        }

        _d_csrValCt    = d_csrValCt_new;

        // rewrite d_queue
        cudaMemcpy(&_d_queue_one[TUPLE_QUEUE * position],
                                   &_h_queue_one[TUPLE_QUEUE * position],
                                   TUPLE_QUEUE * num_blocks * sizeof(int),   cudaMemcpyHostToDevice);

        //cout << "seems good." << endl;

        _nnzCt = nnzCt_new;
    }

    *count_next = counter;

    return BHSPARSE_SUCCESS;
}


int bhsparse_cuda::create_C()
{
    int err = 0;

    cudaMemcpy(_h_csrRowPtrC, _d_csrRowPtrC, (_m + 1) * sizeof(index_type), cudaMemcpyDeviceToHost);

    int old_val, new_val;
    old_val = _h_csrRowPtrC[0];
    _h_csrRowPtrC[0] = 0;
    for (int i = 1; i <= _m; i++)
    {
        new_val = _h_csrRowPtrC[i];
        _h_csrRowPtrC[i] = old_val + _h_csrRowPtrC[i-1];
        old_val = new_val;
    }

    _nnzC = _h_csrRowPtrC[_m];

    // create device mem of C
    cudaMalloc((void **)&_d_csrColIndC, _nnzC  * sizeof(index_type));
    cudaMalloc((void **)&_d_csrValC,    _nnzC  * sizeof(value_type));

    cudaMemset(_d_csrColIndC, 0, _nnzC * sizeof(index_type));
    cudaMemset(_d_csrValC,    0, _nnzC * sizeof(value_type));

    cudaMemcpy(_d_csrRowPtrC, _h_csrRowPtrC, (_m + 1) * sizeof(index_type), cudaMemcpyHostToDevice);

    return err;
}

__global__ void
copyCt2C_Single(const int*     d_csrRowPtrC,
                int*                       d_csrColIndC,
                value_type*                     d_csrValC,
                const int*     d_csrRowPtrCt,
                const int*     d_csrColIndCt,
                const value_type*   d_csrValCt,
                const int*     d_queue,
                const int                  size,
                const int                  d_queue_offset)
{
    int global_id  = blockIdx.x * blockDim.x + threadIdx.x;

    bool valid = (global_id < size);

    int row_id = valid ? d_queue[TUPLE_QUEUE * (d_queue_offset + global_id)] : 0;

    int Ct_base_start = valid ? d_queue[TUPLE_QUEUE * (d_queue_offset + global_id) + 1] : 0; //d_csrRowPtrCt[row_id] : 0;
    int C_base_start  = valid ? d_csrRowPtrC[row_id] : 0;

    int colC   = valid ? d_csrColIndCt[Ct_base_start] : 0;
    value_type valC = valid ? d_csrValCt[Ct_base_start] : 0.0f;

    if (valid)
    {
        d_csrColIndC[C_base_start] = colC;
        d_csrValC[C_base_start]    = valC;
    }
}

__global__ void
copyCt2C_Loopless(const int*     d_csrRowPtrC,
                  int*                       d_csrColIndC,
                  value_type*                     d_csrValC,
                  const int*     d_csrRowPtrCt,
                  const int*     d_csrColIndCt,
                  const value_type*   d_csrValCt,
                  const int*     d_queue,
                  const int                  d_queue_offset)

{
    int local_id   = threadIdx.x;
    int group_id   = blockIdx.x;

    int row_id = d_queue[TUPLE_QUEUE * (d_queue_offset + group_id)];

    int Ct_base_start = d_queue[TUPLE_QUEUE * (d_queue_offset + group_id) + 1] + local_id; //d_csrRowPtrCt[row_id] + local_id;
    int C_base_start  = d_csrRowPtrC[row_id]  + local_id;
    int C_base_stop   = d_csrRowPtrC[row_id + 1];

    if (C_base_start < C_base_stop)
    {
        d_csrColIndC[C_base_start] = d_csrColIndCt[Ct_base_start];
        d_csrValC[C_base_start]    = d_csrValCt[Ct_base_start];
    }
}

__global__ void
copyCt2C_Loop(const int*     d_csrRowPtrC,
              int*                       d_csrColIndC,
              value_type*                     d_csrValC,
              const int*     d_csrRowPtrCt,
              const int*     d_csrColIndCt,
              const value_type*   d_csrValCt,
              const int*     d_queue,
              const int                  d_queue_offset)
{
    int local_id   = threadIdx.x;
    int group_id   = blockIdx.x;
    int local_size = blockDim.x;

    int row_id = d_queue[TUPLE_QUEUE * (d_queue_offset + group_id)];

    int Ct_base_start = d_queue[TUPLE_QUEUE * (d_queue_offset + group_id) + 1]; //d_csrRowPtrCt[row_id];
    int C_base_start  = d_csrRowPtrC[row_id];
    int C_base_stop   = d_csrRowPtrC[row_id + 1];
    int stride        = C_base_stop - C_base_start;

    bool valid;

    int loop = ceil((float)stride / (float)local_size);

    C_base_start  += local_id;
    Ct_base_start += local_id;

    for (int i = 0; i < loop; i++)
    {
        valid = (C_base_start < C_base_stop);

        if (valid)
        {
            d_csrColIndC[C_base_start] = d_csrColIndCt[Ct_base_start];
            d_csrValC[C_base_start]    = d_csrValCt[Ct_base_start];
        }

        C_base_start += local_size;
        Ct_base_start += local_size;
    }
}

int bhsparse_cuda::copy_Ct_to_C_Single(int num_threads, int num_blocks, int local_size, int position)
{
    cudaError_t err = cudaSuccess;
	
    copyCt2C_Single<<< num_blocks, num_threads >>>(_d_csrRowPtrC, _d_csrColIndC, _d_csrValC,
                           _d_csrRowPtrCt, _d_csrColIndCt, _d_csrValCt,
                           _d_queue_one, local_size, position);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {  cout << "err = " << cudaGetErrorString(err) << endl; return -1; }
	
    return BHSPARSE_SUCCESS;
}

int bhsparse_cuda::copy_Ct_to_C_Loopless(int num_threads, int num_blocks, int j, int position)
{
    cudaError_t err = cudaSuccess;

    copyCt2C_Loopless<<< num_blocks, num_threads >>>(_d_csrRowPtrC, _d_csrColIndC, _d_csrValC,
                           _d_csrRowPtrCt, _d_csrColIndCt, _d_csrValCt,
                           _d_queue_one, position);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {  cout << "err = " << cudaGetErrorString(err) << endl; return -1; }

    return BHSPARSE_SUCCESS;
}

int bhsparse_cuda::copy_Ct_to_C_Loop(int num_threads, int num_blocks, int j, int position)
{
    cudaError_t err = cudaSuccess;

    copyCt2C_Loop<<< num_blocks, num_threads >>>(_d_csrRowPtrC, _d_csrColIndC, _d_csrValC,
                           _d_csrRowPtrCt, _d_csrColIndCt, _d_csrValCt,
                           _d_queue_one, position);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {  cout << "err = " << cudaGetErrorString(err) << endl; return -1; }

    return BHSPARSE_SUCCESS;
}

int bhsparse_cuda::get_nnzC()
{
    return _nnzC;
}

int bhsparse_cuda::get_C(index_type *csrColIndC, value_type *csrValC)
{
    int err = 0;

    cudaMemcpy(csrColIndC, _d_csrColIndC, _nnzC * sizeof(index_type),   cudaMemcpyDeviceToHost);
    cudaMemcpy(_h_csrRowPtrC, _d_csrRowPtrC, (_m + 1) * sizeof(index_type),   cudaMemcpyDeviceToHost);
    cudaMemcpy(csrValC, _d_csrValC, _nnzC * sizeof(value_type),   cudaMemcpyDeviceToHost);

    return err;
}

#endif // BHSPARSE_CUDA_H
