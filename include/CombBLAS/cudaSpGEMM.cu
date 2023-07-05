#include "cudaSpGEMM.h"
#include <device_launch_parameters.h>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>

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
        for(size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < B_nzc; i += gridDim.x * blockDim.x) {
            size_t nnzcolB = B_CP[i+1] - B_CP[i];
                //if(j == 0) printf("BlockDim = %i, GridDim = %i", blockDim.x, gridDim.x);
                IT ptr = curptrC[i];
                for(size_t j = 0; j < A_nzc; ++j) {
                bool made = false;
                size_t r = A_Tran_CP[j];
                
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
                                std::get<2>(tuplesC[ptr - 1]) = std::get<2>(tuplesC[ptr - 1]) + mrhs;
                            } else {
                                made = true;
                                //if (colptr_size_d[i] != ptr - curptrC[i]) printf("Potential conflict\n");
                                //__syncthreads();
                                //printf("Adding at ptr = %i\n", (int) ptr);
                               // colptr_size_d[i]++;
                                std::get<0>(tuplesC[ptr]) = A_Tran_JC[j];
                                //if (A_Tran_JC[j] < 0 || B_JC[i] < 0) {
                                //    printf("Somehow got a <0, %i, %i", (int) A_Tran_JC[j], (int) B_JC[i]);
                                //}
                                std::get<1>(tuplesC[ptr])= B_JC[i];
                                std::get<2>(tuplesC[ptr++])  = mrhs;
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
        int blks = (B_nzc + 256 - 1) / 256;
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
