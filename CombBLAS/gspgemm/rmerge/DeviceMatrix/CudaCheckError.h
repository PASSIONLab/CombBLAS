#pragma once

#include "cuda_runtime_api.h"

#include <exception>
#include <stdexcept>
#include <sstream>
#include <string>


static void CudaCheckError(cudaError error){
//#ifndef __CUDACC__
	if(error!=cudaSuccess)
	{
		//throw gcnew System::Exception("cudaError "+error);
		std::stringstream ss; ss << "CUDA error: " << cudaGetErrorString(error);
		std::string e=ss.str();
		throw std::runtime_error(e.c_str());
	}
//#endif
}

static void CudaCheckErrorImportant(){
	//cudaError error=cudaDeviceSynchronize();
	CudaCheckError(cudaGetLastError());
}


//Checks for the last error, resets to no error and throws if error
static void CudaCheckError(){
	//cudaError error=cudaDeviceSynchronize();
	CudaCheckError(cudaGetLastError());
}

static void CudaCheckErrorSync(){
	cudaError error=cudaDeviceSynchronize();
	CudaCheckError(cudaGetLastError());
}

static void CudaCheckErrorTmp(){
	//cudaError error=cudaDeviceSynchronize();
	//CudaCheckError(cudaGetLastError());
}

