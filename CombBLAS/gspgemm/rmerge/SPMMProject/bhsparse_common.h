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

#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <iostream>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>
//#include <helper_functions.h>
//#include <helper_cuda.h>

#include "bhsparse_cudatimer.h"

#define BHSPARSE_SUCCESS 0

using namespace std;

typedef int     index_type;
typedef double   value_type;

#define NUM_PLATFORMS     9

#define NAIVE             0
#define BHSPARSE_CUDA     1
#define BHSPARSE_OPENCL   2

#define WARPSIZE_NV  32
#define WARPSIZE_AMD 64

#define NUM_BANKS 32

#define WARPSIZE_NV_2HEAP 32

#define TUPLE_QUEUE 6
#define MERGELIST_INITSIZE 256

#define MERGEPATH_LOCAL     0
#define MERGEPATH_LOCAL_L2  1
#define MERGEPATH_GLOBAL    2

#define GROUPSIZE_32   32
#define GROUPSIZE_128  128
#define GROUPSIZE_256  256
#define GROUPSIZE_512  512
#define GROUPSIZE_1024 1024

#define NUM_SEGMENTS 128

#endif // COMMON_H
