#include "cudaSpGEMM.h"
#include <device_launch_parameters.h>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>

template <typename NTO, typename IT, typename NT1, typename NT2>
__global__ void transformColumn_d(IT A_nzc, size_t i, size_t nnzcolB, uint* curptr, IT* A_Tran_CP,
    IT* A_Tran_IR,
    IT* A_Tran_JC,
    NT1* A_Tran_numx,
    IT* B_CP,
    IT* B_IR,
    IT* B_JC,
    NT2* B_numx,
    std::tuple<IT,IT,NTO> * tuplesC) {
        for(size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < A_nzc; j += gridDim.x * blockDim.x) {
                bool made = false;
                size_t r = A_Tran_CP[j];
                for (size_t k = 0; k < nnzcolB; ++k) {
                    uint ptr = *curptr;
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
                                ptr = atomicAdd(curptr, 1u);
                                __syncthreads();
                                std::get<0>(tuplesC[ptr]) = A_Tran_JC[j];
                                std::get<1>(tuplesC[ptr]) = B_JC[i];
                                std::get<2>(tuplesC[ptr]) = mrhs;
                            }
                        }
                    }
                }
            }
}
template < typename NTO, typename IT, typename NT1, typename NT2>
void transformColumn(IT A_nzc, size_t i, size_t nnzcolB, uint* curptr, IT* A_Tran_CP,
    IT* A_Tran_IR,
    IT* A_Tran_JC,
    NT1* A_Tran_numx,
    IT* B_CP,
    IT* B_IR,
    IT* B_JC,
    NT2* B_numx,
    std::tuple<IT,IT,NTO> * tuplesC) {

        transformColumn_d<<<1,1>>>(A_nzc, i, nnzcolB, curptr, A_Tran_CP, A_Tran_IR, A_Tran_JC, A_Tran_numx, B_CP, B_IR, B_JC, B_numx, tuplesC);
}

template void transformColumn< double, int64_t, double, double>(
    int64_t A_nzc, size_t i, size_t nnzcolB, uint* curptr, int64_t* A_Tran_CP,
    int64_t* A_Tran_IR,
    int64_t* A_Tran_JC,
    double* A_Tran_numx,
    int64_t* B_CP,
    int64_t* B_IR,
    int64_t* B_JC,
    double* B_numx,
    std::tuple<int64_t,int64_t,double> * tuplesC);