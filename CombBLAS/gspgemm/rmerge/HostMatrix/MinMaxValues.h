#pragma once
#include "General/WinOnlyStatic.h"
#include "HostMatrix/Intrinsics.h"
#include <climits>
#include <cfloat>
//#include <limits.h>

//Somehow this does not work with Cuda
//template<typename T> static __device__ __host__ T MinVal(){return numeric_limits<T>::min();}
//template<typename T> static __device__ __host__ T MaxVal(){return numeric_limits<T>::max();}

typedef unsigned char byte;

template<typename T> static __device__ __host__ T MinVal(){return 0;}
template<typename T> static __device__ __host__ T MaxVal(){return 1;}

template <> WinOnlyStatic inline __device__ __host__  bool MinVal<bool>(){return false;}
template <> WinOnlyStatic inline __device__ __host__  bool MaxVal<bool>(){return true;}
template <> WinOnlyStatic inline __device__ __host__  char MinVal<char>(){return -128;}
template <> WinOnlyStatic inline __device__ __host__  char MaxVal<char>(){return 127;}
template <> WinOnlyStatic inline __device__ __host__  byte MinVal<byte>(){return 0;}
template <> WinOnlyStatic inline __device__ __host__  byte MaxVal<byte>(){return 255;}
template <> WinOnlyStatic inline __device__ __host__  short MinVal<short>(){return -32768;}
template <> WinOnlyStatic inline __device__ __host__  short MaxVal<short>(){return 32767;}
template <> WinOnlyStatic inline __device__ __host__  ushort MinVal<ushort>(){return 0;}
template <> WinOnlyStatic inline __device__ __host__  ushort MaxVal<ushort>(){return 0xffff;}
template <> WinOnlyStatic inline __device__ __host__  int MinVal<int>(){return (-2147483647 - 1);}
template <> WinOnlyStatic inline __device__ __host__  int MaxVal<int>(){return 2147483647;}
template <> WinOnlyStatic inline __device__ __host__  uint MinVal<uint>(){return 0;}
template <> WinOnlyStatic inline __device__ __host__  uint MaxVal<uint>(){return 0xffffffff;}
template <> WinOnlyStatic inline __device__ __host__  float MinVal<float>(){return -3.402823466e+38F;}
template <> WinOnlyStatic inline __device__ __host__  float MaxVal<float>(){return 3.402823466e+38F;}
template <> WinOnlyStatic inline __device__ __host__  double MinVal<double>(){return -1.7976931348623158e+308;}
template <> WinOnlyStatic inline __device__ __host__  double MaxVal<double>(){return 1.7976931348623158e+308;}
template <> WinOnlyStatic inline __device__ __host__  int64 MinVal<int64>(){return LLONG_MIN;}
template <> WinOnlyStatic inline __device__ __host__  int64 MaxVal<int64>(){return LLONG_MAX;}
template <> WinOnlyStatic inline __device__ __host__  uint64 MinVal<uint64>(){return 0;}
template <> WinOnlyStatic inline __device__ __host__  uint64 MaxVal<uint64>(){return ULLONG_MAX;}
