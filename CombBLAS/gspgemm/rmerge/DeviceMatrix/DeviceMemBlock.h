#pragma once
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#include "DeviceMatrix/CudaCheckError.h"
#include "HostMatrix/int64.h"
#include "HostMatrix/Verify.h"

template<typename T>
class DeviceMemBlock{
	T* data;
public:
	DeviceMemBlock():data(0){}
	explicit DeviceMemBlock(int64 n){
		data=0;
		CudaCheckErrorImportant();
		cudaError_t e1=cudaMalloc(&data,n*sizeof(T));
		if(e1!=0){
			cudaGetLastError();//Needed to reset an error
			throw std::runtime_error("cudaMalloc failed. Out of GPU memory?");
		}
	}
	explicit DeviceMemBlock(T* data):data(data){
	}

	~DeviceMemBlock(){
		//Here we cannot throw exceptions, because this might be called during an exception
		cudaFree(data);
		data=0;
	}
	T* Pointer(){return (T*)data;}
};

	
