#pragma once
#include "DeviceMatrix/DeviceVector.h"

//Sparse matrix matrix multiplication by row merging
template<int BlockDim, typename T>
__global__ void __cdecl RangeKernel(CVector<T> dst, T offset, T scale){	
	int i=threadIdx.x+blockIdx.x*BlockDim;
	if(i>=dst.Length())
		return;
	dst[i]=offset+scale*T(i);	
}

//Each thread maintains four rows, therefore the warp size is a quarter of the MergeFactor
template<typename T>
void __cdecl Range(DeviceVector<T> dst, T offset=T(0), T scale=T(1))
#ifdef __CUDACC__
{
	if(dst.Length()==0)return;
	const int BlockDim=512;
	dim3 blockDim(BlockDim,1,1);
	dim3 gridDim(DivUp(dst.Length32(),(int)blockDim.x),1,1);
	RangeKernel<BlockDim> <<< gridDim, blockDim, 0>>>(dst.GetC(),offset,scale);
}
#else
;
#endif

