#pragma once
#include "DeviceMatrix/SparseDeviceMatrixCSR.h"

template<typename T>
SparseDeviceMatrixCSR<T> MulCUSP(SparseDeviceMatrixCSR<T> A, SparseDeviceMatrixCSR<T> B);
