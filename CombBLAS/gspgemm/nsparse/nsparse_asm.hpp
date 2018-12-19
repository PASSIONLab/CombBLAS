/*
 * Inline PTX
 */
#ifndef NSPARSE_ASM_H
#define NSPARSE_ASM_H

__device__ __inline__ float ld_gbl_val(const float *val)
{
    float return_value;
    asm("ld.global.cv.f32 %0, [%1];" : "=f"(return_value) : "l"(val));
    return return_value;
}

__device__ __inline__ double ld_gbl_val(const double *val)
{
    double return_value;
    asm("ld.global.cv.f64 %0, [%1];" : "=d"(return_value) : "l"(val));
    return return_value;
}

__device__ __inline__ int ld_gbl_col(const int *col)
{
    int return_value;
    asm("ld.global.cv.s32 %0, [%1];" : "=r"(return_value) : "l"(col));
    return return_value;
}

__device__ __inline__ short ld_gbl_col(const short *col)
{
    short return_value;
    asm("ld.global.cv.u16 %0, [%1];" : "=h"(return_value) : "l"(col));
    return return_value;
}

__device__ __inline__ unsigned short ld_gbl_col(const unsigned short *col)
{
    unsigned short return_value;
    asm("ld.global.cv.u16 %0, [%1];" : "=h"(return_value) : "l"(col));
    return return_value;
}

__device__ __inline__ void st_gbl_val(const float *ptr, float val)
{
    asm("st.global.cs.f32 [%0], %1;" :: "l"(ptr) , "f"(val));

}

__device__ __inline__ void st_gbl_val(const double *ptr, double val)
{
    asm("st.global.cs.f64 [%0], %1;" :: "l"(ptr) , "d"(val));
}

/*
 * Multiply and Add
 */
/* template <class T>
class Add
{
public:
    __device__ __inline__ T operator()(T a, T b)
    {
        return a + b;
    }
}; */

template <class T>
class Multiply
{
public:
    __device__ __inline__ T operator()(T a, T b)
    {
        return a * b;
    }
};

template <class T>
class AtomicAdd
{
public:
    __device__ T operator()(T* a, T v);
};

template <>
__device__ __inline__ float AtomicAdd<float>::operator()(float *a, float v)
{
    return atomicAdd(a, v);
}

template <>
__device__ __inline__ double AtomicAdd<double>::operator()(double *a, double v)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)
    return atomicAdd(a, v);
#else
    unsigned long long int *a_ull = (unsigned long long int *)(a);
    unsigned long long int old = *a_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(a_ull, assumed, __double_as_longlong(v + __longlong_as_double(assumed)));
    } while (assumed != old);
    return old;
#endif
}

#endif
