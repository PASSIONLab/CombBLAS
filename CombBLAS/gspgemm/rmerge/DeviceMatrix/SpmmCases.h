#pragma once
#include "DeviceMatrix/SparseDeviceMatrixCSR.h"
#include "DeviceMatrix/Scan.h"
#include "DeviceMatrix/DeviceReductions.h"
#include "HostMatrix/Intrinsics.h"
#include "HostMatrix/MinMaxValues.h"
#include "HostMatrix/IO.h"
#include "DeviceMatrix/ldg.h"
#include "DeviceMatrix/WarpReduction.h"
#include "DeviceMatrix/SparseDeviceMatrixCSROperations.h"
#include "DeviceMatrix/Sort.h"
#include "DeviceMatrix/Range.h"

//Computes the number of cases using the sorted array.
//The cumulative counts are computed
template<int BlockSize>
__global__ void __cdecl CumulativeCaseCountsKernel(CVector<uint> dst, CVector<uchar> sortedCases){
	int i=blockIdx.x*BlockSize+threadIdx.x;
	if(i>=sortedCases.Length32())
		return;

	uint case_i=sortedCases[i];
	if(i==0){//Special case for the first (and the last)
		for(int c=0;c<case_i;c++)
			dst[c]=0;

		int case_last=sortedCases[sortedCases.Length32()-1];
		int caseCount=dst.Length32();
		for(int c=case_last;c<caseCount;c++)
			dst[c]=sortedCases.Length32();
	}
	else{		
		int case_prev=sortedCases[i-1];
		if(case_i==case_prev)
			return;
		//We have a jump
		dst[case_prev]=i;
		//If the jump is over a few, they need to be set to i as well
		for(int c=case_prev+1;c<case_i;c++)
			dst[c]=i;
	}
}

//Computes cumulative case counts using the sorted array
//dst[i] will contain the number of cases <= i
// @OGUZ-EDIT
__inline__
void __cdecl CumulativeCaseCounts(DeviceVector<uint> dst, DeviceVector<uchar> sortedCases)
#ifdef __CUDACC__
{	
	const int BlockSize=256;
	dim3 blockDim(BlockSize,1,1);	
	dim3 gridDim(DivUp(sortedCases.Length32(),BlockSize),1,1);
	CumulativeCaseCountsKernel<BlockSize> <<< gridDim, blockDim, 0>>>(dst.GetC(),sortedCases.GetC());
}
#else
;
#endif

//Computes the case counts using the sorted array of cases.
static HostVector<uint> RowsPerCase(int caseCount, DeviceVector<uchar> sortedCases){
	DeviceVector<uint> tmp(caseCount);
	CumulativeCaseCounts(tmp,sortedCases);
	
	HostVector<uint> a=ToHost(tmp);
	HostVector<uint> rowsPerCase(caseCount);
	rowsPerCase[0]=a[0];
	for(int i=1;i<caseCount;i++)
		rowsPerCase[i]=a[i]-a[i-1];	
	return rowsPerCase;
}

//Computes the case counts using the sorted array of cases.
static void RowsPerCase(HostVector<uint> rowsPerCase, HostVector<uint> cumulativeRowsPerCase, int caseCount, DeviceVector<uchar> sortedCases){
	DeviceVector<uint> tmp(caseCount);
	CumulativeCaseCounts(tmp,sortedCases);
	Copy(tmp,cumulativeRowsPerCase);		
	rowsPerCase[0]=cumulativeRowsPerCase[0];
	for(int i=1;i<caseCount;i++)
		rowsPerCase[i]=cumulativeRowsPerCase[i]-cumulativeRowsPerCase[i-1];
}

//Counts the number of uchars having a certain value
static uint CountCases(DeviceVector<uchar> cases, uchar c){
	DeviceVector<uint> tmp(cases.Length());
	ComponentWise(tmp,cases,ElementFunctors::Threshold<uint,uchar>(c,c,1,0));
	return Sum(tmp);
}

//This function tests the functions above
static void TestRowsPerCase(uint caseCount, DeviceVector<uchar> sortedCases){
	CudaCheckError();
	HostVector<uint> rowsPerCase=RowsPerCase(caseCount,sortedCases);
	CudaCheckError();
	HostVector<uint> rowsPerCase2(caseCount);
	for(uint i=0;i<caseCount;i++){
		rowsPerCase2[i]=CountCases(sortedCases,i);
		CudaCheckError();
	}
	std::cout<<std::endl;
	for(uint i=0;i<caseCount;i++){		
		std::cout<<"case "<<i<<" "<<rowsPerCase[i]<<" "<<rowsPerCase2[i]<<std::endl;
		if(rowsPerCase[i]!=rowsPerCase2[i])
			throw std::runtime_error("Bad case count");
	}
}