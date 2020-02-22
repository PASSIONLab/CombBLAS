#pragma once

//This file defines functors for SpGEMM. These functors are used by generic functions to measure the performance.

#include "cuda_runtime_api.h"

#include "HostMatrix/MulMKL.h"
/* #include "MulCusp.h" */
/* #include "MulCusparse.h" */
/* #include "LoadMatrixCusp.h" */
#include "HostMatrix/SparseHostMatrixCSR.h"
#include "DeviceMatrix/SparseDeviceMatrixCSR.h"
#include "HostMatrix/SparseHostMatrixCSROperations.h"
#include "DeviceMatrix/SparseDeviceMatrixCSROperations.h"
#include "DeviceMatrix/SpmmWarp.h"
#include "DeviceMatrix/SpmmWarpN.h"
#include "DeviceMatrix/SpmmWarpNSorted.h"

#include "DeviceMatrix/DeviceTransfers.h"
#include "General/WallTime.h"

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <vector>


class RMerge1Functor{
	int mergeFactor;
	bool adjustLast;
public:
	RMerge1Functor(int mergeFactor=16, bool adjustLast=true):mergeFactor(mergeFactor),adjustLast(adjustLast){}
	template<typename T>
	SparseDeviceMatrixCSR<T> operator()(SparseDeviceMatrixCSR<T> A, SparseDeviceMatrixCSR<T> B){
		return MulRMerge1(A,B,mergeFactor,adjustLast);
	}

	template<typename T>
	SparseHostMatrixCSR<T> operator()(SparseHostMatrixCSR<T> A, SparseHostMatrixCSR<T> B){
		return ToHost(MulRMerge1(ToDevice(A),ToDevice(B),mergeFactor,adjustLast));
	}
};

class RMerge11Functor{
	int mergeFactor;	
public:
	RMerge11Functor(int mergeFactor=256):mergeFactor(mergeFactor){}
	template<typename T>
	SparseDeviceMatrixCSR<T> operator()(SparseDeviceMatrixCSR<T> A, SparseDeviceMatrixCSR<T> B){
		return MulRMerge11<T>(A,B,mergeFactor);
	}

	template<typename T>
	SparseHostMatrixCSR<T> operator()(SparseHostMatrixCSR<T> A, SparseHostMatrixCSR<T> B){
		return ToHost(MulRMerge11<T>(ToDevice(A),ToDevice(B),mergeFactor));
	}
};

class RMerge12Functor{
public:	
	template<typename T>
	SparseDeviceMatrixCSR<T> operator()(SparseDeviceMatrixCSR<T> A, SparseDeviceMatrixCSR<T> B){
		return MulRMerge2<T>(A,B,1);
	}

	template<typename T>
	SparseHostMatrixCSR<T> operator()(SparseHostMatrixCSR<T> A, SparseHostMatrixCSR<T> B){
		return ToHost(MulRMerge2<T>(ToDevice(A),ToDevice(B),1));
	}
};

class RMerge2Functor {
	int computeCapability;
public:

	RMerge2Functor() {
		computeCapability = CudaComputeCapabilityMajor();
	}

	template<typename T>
	SparseDeviceMatrixCSR<T> operator()(SparseDeviceMatrixCSR<T> A, SparseDeviceMatrixCSR<T> B) {
		return MulRMerge2<T>(A, B,StreamCount(),false,computeCapability);
	}

	template<typename T>
	SparseHostMatrixCSR<T> operator()(SparseHostMatrixCSR<T> A, SparseHostMatrixCSR<T> B) {
		return ToHost(operator()(ToDevice(A), ToDevice(B)));
	}
};

class RMerge2PascalFunctor {
public:
	template<typename T>
	SparseDeviceMatrixCSR<T> operator()(SparseDeviceMatrixCSR<T> A, SparseDeviceMatrixCSR<T> B) {
		return MulRMerge2<T>(A, B);
	}

	template<typename T>
	SparseHostMatrixCSR<T> operator()(SparseHostMatrixCSR<T> A, SparseHostMatrixCSR<T> B) {
		return ToHost(operator()(ToDevice(A), ToDevice(B)));
	}
};

class RMerge2KeplerFunctor {
public:
	template<typename T>
	SparseDeviceMatrixCSR<T> operator()(SparseDeviceMatrixCSR<T> A, SparseDeviceMatrixCSR<T> B) {
		return MulRMerge2<T>(A, B, StreamCount(), false, 3);
	}

	template<typename T>
	SparseHostMatrixCSR<T> operator()(SparseHostMatrixCSR<T> A, SparseHostMatrixCSR<T> B) {
		return ToHost(operator()(ToDevice(A), ToDevice(B)));
	}
};

SparseDeviceMatrixCSR<double> Mul_bhsparse(SparseDeviceMatrixCSR<double> A, SparseDeviceMatrixCSR<double> B);
SparseDeviceMatrixCSR<float> Mul_bhsparse(SparseDeviceMatrixCSR<float> A, SparseDeviceMatrixCSR<float> B);

class BhsparseFunctor{
public:
	
	template<typename T>
	SparseDeviceMatrixCSR<T> operator()(SparseDeviceMatrixCSR<T> A, SparseDeviceMatrixCSR<T> B){
		return Mul_bhsparse(A,B);
	}

	template<typename T>
	SparseHostMatrixCSR<T> operator()(SparseHostMatrixCSR<T> A, SparseHostMatrixCSR<T> B){
		return ToHost(Mul_bhsparse(ToDevice(A),ToDevice(B)));
	}
};

/*
class CUSPMulFunctor{	
public:
	template<typename T>
	SparseDeviceMatrixCSR<T> operator()(SparseDeviceMatrixCSR<T> A, SparseDeviceMatrixCSR<T> B){
		return MulCUSP(A,B);
	}

	template<typename T>
	SparseHostMatrixCSR<T> operator()(SparseHostMatrixCSR<T> A, SparseHostMatrixCSR<T> B){
		return ToHost(MulCUSP(ToDevice(A),ToDevice(B)));
	}
};


class CUSPARSEMulFunctor{	
public:	
	template<typename T>
	SparseDeviceMatrixCSR<T> operator()(SparseDeviceMatrixCSR<T> A, SparseDeviceMatrixCSR<T> B){
		return MulCusparse(A,B);
	}

	template<typename T>
	SparseHostMatrixCSR<T> operator()(SparseHostMatrixCSR<T> A, SparseHostMatrixCSR<T> B){
		return ToHost(MulCusparse(ToDevice(A),ToDevice(B)));
	}
};
*/

class MKLMulZeroBasedFunctor{	
public:
	template<typename T>
	SparseHostMatrixCSR<T> operator()(SparseHostMatrixCSR<T> A, SparseHostMatrixCSR<T> B){
		return MulMKL(A,B);
	}

	template<typename T>
	SparseDeviceMatrixCSR<T> operator()(SparseDeviceMatrixCSR<T> A, SparseDeviceMatrixCSR<T> B){
		//MKL only supports one-based matrices for SPMM
		//Perform the conversion to one-based on the GPU because it is faster
		ToOneBased(A);
		ToOneBased(B);
		SparseHostMatrixCSR<T> A_=ToHost(A);
		SparseHostMatrixCSR<T> B_=ToHost(B);
		ToZeroBased(A);
		ToZeroBased(B);
		SparseHostMatrixCSR<T> C_=MulMKLOneBased(A_,B_);
		SparseDeviceMatrixCSR<T> C=ToDevice(C_);
		ToZeroBased(C);
		return C;
		//return ToDevice(MulMKL(ToHost(A),ToHost(B)));
	}
};

class MKLMulOneBasedFunctor{	
public:
	template<typename T>
	SparseHostMatrixCSR<T> operator()(SparseHostMatrixCSR<T> A, SparseHostMatrixCSR<T> B){
		return MulMKLOneBased(A,B);
	}

	template<typename T>
	SparseDeviceMatrixCSR<T> operator()(SparseDeviceMatrixCSR<T> A, SparseDeviceMatrixCSR<T> B){		
		SparseHostMatrixCSR<T> A_=ToHost(A);
		SparseHostMatrixCSR<T> B_=ToHost(B);
		SparseHostMatrixCSR<T> C_=MulMKLOneBased(A_,B_);
		SparseDeviceMatrixCSR<T> C=ToDevice(C_);
		return C;
	}
};
