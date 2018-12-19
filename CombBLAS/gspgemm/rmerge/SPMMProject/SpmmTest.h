#pragma once

#include "cuda_runtime_api.h"

#include "LoadMatrixCusp.h"
#include "HostMatrix/SparseHostMatrixCSR.h"
#include "DeviceMatrix/SparseDeviceMatrixCSR.h"
#include "HostMatrix/SparseHostMatrixCSROperations.h"
#include "DeviceMatrix/SparseDeviceMatrixCSROperations.h"

#include "DeviceMatrix/DeviceTransfers.h"
#include "General/WallTime.h"

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <vector>


template<typename T>
static void CheckResult(SparseHostMatrixCSR<T> A, SparseHostMatrixCSR<T> B, SparseHostMatrixCSR<T> C, SparseHostMatrixCSR<T> C__, double maxError, double& relError){
	bool equalStructure=EqualStructure(C,C__);
	if(!equalStructure)
	{
		HostVector<uint> rowStartsC=C.RowStarts();
		HostVector<uint> rowStartsC_=C__.RowStarts();
		std::cout<<std::endl;
		//Check the row starts
		DeviceVector<uint> tmpSizes(A.Height());SpmmEstimateTmpSize(tmpSizes,ToDevice(A),ToDevice(B));
		for(int i=0;i<C.Height();i++){
			if((rowStartsC[i]!=rowStartsC_[i] || C.RowLength(i)!=C__.RowLength(i)))
				std::cout<<i<<" "<<A.RowLength(i)<<" "<<rowStartsC[i]<<" "<<rowStartsC_[i]<<" "<<C.RowLength(i)<<" "<<C__.RowLength(i)<<" "<<tmpSizes[i]<<std::endl;		
		}
		//Check the collumn indices
		HostVector<uint> colIndicesC=C.ColIndices();		
		HostVector<uint> colIndicesC_=C__.ColIndices();
		for(int i=0;i<colIndicesC.Length32();i++){
			if(colIndicesC[i]!=colIndicesC_[i])
				std::cout<<i<<" "<<colIndicesC[i]<<" "<<colIndicesC_[i]<<std::endl;		
		}		
	}
	Verify(equalStructure,"Structure differs");
	
	double diff=DistanceMax(C.Values(),C__.Values());
	relError=diff/NormMax(C.Values());
	if(relError>=maxError){
		HostVector<T> valuesC=C.Values();
		HostVector<T> valuesC_=C__.Values();		
		std::cout<<std::endl;
		//Check the values
		for(int r=0;r<C.Height();r++){
			SparseHostVector<T> r1=C.Row(r);
			SparseHostVector<T> r2=C__.Row(r);
			std::cout<<"row: "<<r<<"\tnnz: "<<A.Row(r).NonZeroCount()<<std::endl;
			for(int i=0;i<r1.NonZeroCount();i++){
				if(r1.Value(i)!=r2.Value(i))
					std::cout<<"row: "<<r<<"\tnnz: "<<A.Row(r).NonZeroCount()<<"\tindex: "<<i<<"\tcolumn: "<<r1.Index(i)<<"\t"<<r1.Value(i)<<"\t"<<r2.Value(i)<<std::endl;
			}
		}
	}
	Verify(relError<maxError,"Values differ");
}

//Measure performance for SpGEMM on the GPU
template<typename T, typename MulFunctor>
static double MeasureSPMM_Device(SparseHostMatrixCSR<T> A, SparseHostMatrixCSR<T> B, SparseHostMatrixCSR<T> C, int repetitions, double maxError, double& relError, MulFunctor mulFunctor){
	SparseDeviceMatrixCSR<T> A_ = ToDevice(A);
	SparseDeviceMatrixCSR<T> B_ = ToDevice(B);
	SparseDeviceMatrixCSR<T> C_ = mulFunctor(A_,B_);
	CheckResult(A,B,C,ToHost(C_),maxError,relError);
	C_= SparseDeviceMatrixCSR<T>();//release memory

	cudaDeviceSynchronize();
	CudaCheckError();
	ExMI::WallTime t;
	for (int k = 0; k < repetitions; k++) {			
		C_= SparseDeviceMatrixCSR<T>();//release memory
		C_ = mulFunctor(A_,B_);
	}
	cudaDeviceSynchronize();
	double time = t.Seconds()/repetitions;
	CudaCheckError();
	CheckResult(A,B,C,ToHost(C_),maxError,relError);
	return time;
}

//Measure performance for SpGEMM on the CPU
template<typename T, typename MulFunctor>
static double MeasureSPMM_Host(SparseHostMatrixCSR<T> A, SparseHostMatrixCSR<T> B, SparseHostMatrixCSR<T> C, int repetitions, double maxError, double& relError, MulFunctor mulFunctor, bool zeroBased=true){
	if(!zeroBased){
		A=Clone(A);
		B=Clone(B);
		ToOneBased(A);
		ToOneBased(B);
	}
	SparseHostMatrixCSR<T> C_ = mulFunctor(A,B);
	C_= SparseHostMatrixCSR<T>();//release memory
	cudaDeviceSynchronize();
	CudaCheckError();
	ExMI::WallTime t;
	for (int k = 0; k < repetitions; k++){
		C_= SparseHostMatrixCSR<T>();//release memory
		C_ = mulFunctor(A,B);
	}	
	double time = t.Seconds()/repetitions;
	if(!zeroBased)
		ToZeroBased(C_);
	Verify(EqualStructure(C,C_),"Structure differs");
	double diff=DistanceMax(C.Values(),C_.Values());
	relError=diff/NormMax(C.Values());
	Verify(relError<maxError,"Values differ");
	return time;
}

//Computes the FLOPS
//Also the compression of the first multiplication
template<typename T> 
static int64 FlopsOfGalerkin(SparseHostMatrixCSR<T> A, std::vector<SparseHostMatrixCSR<T> > up, double& compression) {
	std::vector<SparseHostMatrixCSR<T> > down;
	for (int j = 0; j < up.size(); j++)
		down.push_back(Transpose(up[j]));
	int64 count=0;
	SparseHostMatrixCSR<T> current=A;
	for (int j = 0; j < up.size(); j++){
		count+=FlopsOfSPMM(current,up[j]);
		SparseHostMatrixCSR<T> tmp=Mul(current,up[j]);

		if(j==0){
			int64 mulCount=count/2;
			compression=double(mulCount)/double(tmp.NonZeroCount());
		}

		count+=FlopsOfSPMM(down[j],tmp);
		current=Mul(down[j],tmp);
	}
	return count;
}

//Measure performance for the Galerkin pyramid
template<typename T, typename MulFunctor> 
static double MeasureGalerkinDevice(SparseHostMatrixCSR<T> A, std::vector<SparseHostMatrixCSR<T>> up, int repetitions, double maxError, double& relError, MulFunctor mulfunctor) {
		std::vector<SparseDeviceMatrixCSR<T>> down_, up_;

		for (int j = 0; j < up.size(); j++) {
			down_.push_back(ToDevice(Transpose(up[j])));
			up_.push_back(ToDevice(up[j]));
		}
		
		SparseDeviceMatrixCSR<T> A_ = ToDevice(A);
		{//Perform computations once and check results
			SparseDeviceMatrixCSR<T> current=A_;
			for (int j = 0; j < up.size(); j++) {
				SparseHostMatrixCSR<T> tmp_=ToHost(MulRMerge1(current,up_[j]));
				SparseDeviceMatrixCSR<T> tmp=mulfunctor(current,up_[j]);
				
				CheckResult(ToHost(current),ToHost(up_[j]),tmp_,ToHost(tmp),maxError,relError);

				SparseHostMatrixCSR<T> current_=ToHost(MulRMerge1(down_[j],tmp));
				current=mulfunctor(down_[j],tmp);
				CheckResult(ToHost(down_[j]),ToHost(tmp),current_,ToHost(current),maxError,relError);				
			}
		}		
		
		cudaDeviceSynchronize();
		ExMI::WallTime t;
		for (int i = 0; i < repetitions; i++) {		
			SparseDeviceMatrixCSR<T> current=A_;
			for (int j = 0; j < up.size(); j++) {
				SparseDeviceMatrixCSR<T> tmp=mulfunctor(current,up_[j]);
				current=mulfunctor(down_[j],tmp);
			}
		}
		cudaDeviceSynchronize();
		return t.Seconds()/repetitions;
	}

template<typename T>
class SpmmTest{
public:
	static void MaxError(double& e){
		e=0.000000001;
	}
	static void MaxError(float& e){
		e=0.00001f;
	}
	static T MaxError(){T e;MaxError(e);return e;}		
};
	

template<typename T> 
static void ToOneBased(SparseDeviceMatrixCSR<T>& A){
	ComponentWiseAddUpConstant(A.RowStarts(),uint(1));
	ComponentWiseAddUpConstant(A.ColIndices(),uint(1));
}

template<typename T> 
static void ToZeroBased(SparseDeviceMatrixCSR<T>& A){
	ComponentWiseSubUpConstant(A.RowStarts(),uint(1));
	ComponentWiseSubUpConstant(A.ColIndices(),uint(1));
}