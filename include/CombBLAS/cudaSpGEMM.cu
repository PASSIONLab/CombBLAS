

#include "cudaSpGEMM.h"
#include <cstdint>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <algorithm>
#include "../GALATIC/include/CSR.cuh"
#include "../GALATIC/include/dCSR.cuh"

//#include "../GALATIC/source/device/Multiply.cuh"

template <typename NTO, typename IT, typename NT1, typename NT2>
__global__ void transformColumn_d(IT A_nzc, IT* A_Tran_CP,
    IT* A_Tran_IR,
    IT* A_Tran_JC,
    NT1* A_Tran_numx,
    IT* B_CP,
    IT* B_IR,
    IT* B_JC,
    NT2* B_numx,
    std::tuple<IT,IT,NTO> * tuplesC, IT* curptrC, IT B_nzc) {
        for(size_t i = blockIdx.x; i < B_nzc; i += gridDim.x) {
            size_t nnzcolB = B_CP[i+1] - B_CP[i];
                //if(j == 0) printf("BlockDim = %i, GridDim = %i", blockDim.x, gridDim.x);
                for(size_t j = threadIdx.x; j < A_nzc; j += blockDim.x) {
                bool made = false;
                size_t r = A_Tran_CP[j];
                uint ptr = curptrC[i];
                for (size_t k = 0; k < nnzcolB; ++k) {
                    
                    while (r < A_Tran_CP[j + 1] && B_IR[B_CP[i]+k] > A_Tran_IR[r]) { 
                        r++;
                    }
                    if (r >= A_Tran_CP[j + 1]) {
                            break;
                        }
                    if (B_IR[B_CP[i]+k] == A_Tran_IR[r]) {
                        NTO mrhs = A_Tran_numx[r] * B_numx[B_CP[i]+k];
                        if(true) {
                            if (made) {
                                std::get<2>(tuplesC[ptr]) = std::get<2>(tuplesC[ptr]) + mrhs;
                            } else {
                                made = true;
                                ptr = atomicAdd((unsigned long long*) &curptrC[i],(unsigned long long) 1);
                                //if (colptr_size_d[i] != ptr - curptrC[i]) printf("Potential conflict\n");
                                //__syncthreads();
                                //printf("Adding at ptr = %i\n", (int) ptr);
                               // colptr_size_d[i]++;
                                std::get<0>(tuplesC[ptr]) = A_Tran_JC[j];
                                //if (A_Tran_JC[j] < 0 || B_JC[i] < 0) {
                                //    printf("Somehow got a <0, %i, %i", (int) A_Tran_JC[j], (int) B_JC[i]);
                                //}
                                std::get<1>(tuplesC[ptr])= B_JC[i];
                                std::get<2>(tuplesC[ptr])  = mrhs;
                            }
                        }
                    }
                }
            }
        }
}
template < typename NTO, typename IT, typename NT1, typename NT2>
void transformColumn(IT A_nzc, IT* A_Tran_CP,
    IT* A_Tran_IR,
    IT* A_Tran_JC,
    NT1* A_Tran_numx,
    IT* B_CP,
    IT* B_IR,
    IT* B_JC,
    NT2* B_numx,
     std::tuple<IT,IT,NTO> * tuplesC_d, IT* curptrC, IT B_nzc) {
        int blks = std::min(65535,(int) B_nzc);
        transformColumn_d<<<blks,256>>>(A_nzc, A_Tran_CP,
    A_Tran_IR,
    A_Tran_JC,
     A_Tran_numx,
    B_CP,
B_IR,
    B_JC,
     B_numx,
    tuplesC_d, curptrC, B_nzc);
}

template void transformColumn< double, int64_t, double, double>(
   int64_t A_nzc, int64_t* A_Tran_CP,
    int64_t* A_Tran_IR,
    int64_t* A_Tran_JC,
    double* A_Tran_numx,
    int64_t* B_CP,
    int64_t* B_IR,
    int64_t* B_JC,
    double* B_numx,
    std::tuple<int64_t,int64_t,double> * tuplesC_d, int64_t* curptrC, int64_t B_nzc);

template <typename Arith_SR, typename NTO, typename NT1, typename NT2, typename IT>
__host__  CSR<NTO> LocalGalaticSPGEMM
(CSR<NT1> input_A_CPU,
CSR<NT2> input_B_CPU,
 bool clearA, bool clearB, Arith_SR semiring, IT * aux = nullptr) {
 /*   dCSR<NT1> input_A_GPU;
dCSR<NT2> input_B_GPU;

dCSR<NTO> result_mat_GPU;
convert(input_A_GPU, input_A_CPU);
convert(input_B_GPU, input_B_CPU);

// load data into semiring struct. For this one, we don't need to do anything,
// but you still need to pass it in for generality. The cost is trivial.


// Setup execution options, we'll skip the details for now.

const int Threads = 256;
const int BlocksPerMP = 1;
const int NNZPerThread = 2;
const int InputElementsPerThreads = 2;
const int RetainElementsPerThreads = 1;
const int MaxChunksToMerge = 16;
const int MaxChunksGeneralizedMerge = 256; // MAX: 865
const int MergePathOptions = 8;


GPUMatrixMatrixMultiplyTraits  DefaultTraits(Threads, BlocksPerMP, NNZPerThread,
                                             InputElementsPerThreads, RetainElementsPerThreads,
                                             MaxChunksToMerge, MaxChunksGeneralizedMerge, MergePathOptions);

const bool Debug_Mode = true;
DefaultTraits.preferLoadBalancing = true;
ExecutionStats stats;
stats.measure_all = false;

// Actually perform the matrix multiplicaiton
//ACSpGEMM::Multiply<Arith_SR>(input_A_GPU, input_B_GPU, result_mat_GPU, DefaultTraits, stats, Debug_Mode, semiring);

CSR<NTO> result_mat_CPU;
// load results  onto CPU.
convert(result_mat_CPU, result_mat_GPU);
return result_mat_CPU;*/
 }

template CSR<double> LocalGalaticSPGEMM<Arith_SR, double, double, double, int64_t>
(CSR<double> input_A_CPU,
CSR<double> input_B_CPU,
 bool clearA, bool clearB, Arith_SR semiring, int64_t * aux = nullptr);
