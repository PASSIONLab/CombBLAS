#include "cuda_runtime_api.h"

//This file instantiates the code for RMerge1, RMerge1.1 and RMerge2

#define IGNORE_MKL

#include "DeviceMatrix/SpmmWarp.h"
#include "DeviceMatrix/SpmmWarpN.h"
#include "DeviceMatrix/SpmmWarpNSorted.h"
#include "DeviceMatrix/Sort.h"

void InstantiateMulRMerge(){		
	MulRMerge1(SparseDeviceMatrixCSR<double>(),SparseDeviceMatrixCSR<double>());
	MulRMerge11(SparseDeviceMatrixCSR<double>(),SparseDeviceMatrixCSR<double>());
	MulRMerge2(SparseDeviceMatrixCSR<double>(),SparseDeviceMatrixCSR<double>());		

	MulRMerge1(SparseDeviceMatrixCSR<float>(),SparseDeviceMatrixCSR<float>());
	MulRMerge11(SparseDeviceMatrixCSR<float>(),SparseDeviceMatrixCSR<float>());
	MulRMerge2(SparseDeviceMatrixCSR<float>(),SparseDeviceMatrixCSR<float>());		
	
	MaxTmpSize(SparseDeviceMatrixCSR<double>(),SparseDeviceMatrixCSR<double>());
	SpmmEstimateTmpSize(DeviceVector<uint>(),SparseDeviceMatrixCSR<double>(),SparseDeviceMatrixCSR<double>());
	SpmmEstimateTmpSize(DeviceVector<uint>(),SparseDeviceMatrixCSR<float>(),SparseDeviceMatrixCSR<float>());

	double sum=Sum(DeviceVector<double>());
	float sumF=Sum(DeviceVector<float>());
	uint sumU=Sum(DeviceVector<uint>());
	ComponentWiseInit(DeviceVector<uchar>(),uchar(9));
	ComponentWiseAddUpConstant(DeviceVector<uint>(),uint(1));
	ComponentWiseSubUpConstant(DeviceVector<uint>(),uint(1));
}
