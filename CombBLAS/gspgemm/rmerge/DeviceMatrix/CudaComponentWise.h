#pragma once

#include "HostMatrix/Intrinsics.h"
#include "HostMatrix/VectorOperators.h"
#include "HostMatrix/CVector.h"

//The functions perform component wise operations.
//StrideIsOne means that all vectors have stride 1
//AddUp means that the result is added to the destination instead of replacing it.


#ifdef __CUDACC__
template<int ThreadCountX, int ThreadCountY, typename T, typename ElementFunctor>
__global__ void __cdecl CudaComponentWiseInline2DKernel(CMatrix<T> m, ElementFunctor functor){
	int x=blockIdx.x*ThreadCountX+threadIdx.x;
	int y=blockIdx.y*ThreadCountY+threadIdx.y;
	if(x>=m.DimX() || y>=m.DimY())
		return;
	functor(m(x,y),m(x,y));	
}
#endif
template<typename T, typename ElementFunctor>
void __cdecl CudaComponentWiseInline(DeviceMatrix<T> m, ElementFunctor functor)
#ifdef __CUDACC__
{
	const int ThreadCountX=32;
	const int ThreadCountY=32;
	dim3 gridDim(DivUp(m.DimX(),ThreadCountX),DivUp(m.DimY(),ThreadCountY),1);
	dim3 blockDim(ThreadCountX,ThreadCountY,1);
	CudaComponentWiseInline2DKernel<ThreadCountX, ThreadCountY><<<gridDim, blockDim,0>>>(m.GetC(),functor);
}
#else
;
#endif

#ifdef __CUDACC__
template<int ThreadCountX, int ThreadCountY, typename T, typename ElementFunctor>
__global__ void __cdecl CudaComponentWiseInline3DKernel(CCube<T> cube, ElementFunctor functor){
	int x=blockIdx.x*ThreadCountX+threadIdx.x;
	int y=blockIdx.y*ThreadCountY+threadIdx.y;
	if(x>=cube.DimX() || y>=cube.DimY())
		return;
	T* p=cube.RowPointerZ(x,y);
	for(int z=0;z<cube.DimZ();z++){
		functor(*p,*p);
		p+=cube.SliceStride();
	}
}
#endif
template<typename T, typename ElementFunctor>
void __cdecl CudaComponentWiseInline(DeviceCube<T> cube, ElementFunctor functor)
#ifdef __CUDACC__
{
	const int ThreadCountX=64;
	const int ThreadCountY=8;
	dim3 gridDim(DivUp(cube.DimX(),ThreadCountX),DivUp(cube.DimY(),ThreadCountY),1);
	dim3 blockDim(ThreadCountX,ThreadCountY,1);
	CudaComponentWiseInline3DKernel<ThreadCountX, ThreadCountY><<<gridDim, blockDim,0>>>(cube.GetC(),functor);
}
#else
;
#endif

#ifdef __CUDACC__
template<bool StrideIsOne, int BlockSize, int PerThread, typename DST, typename ElementFunctor>
__global__ void __cdecl CudaComponentWiseInlineKernel(DST* y, uint yStride, uint n,ElementFunctor f){
	uint start=blockIdx.x*(BlockSize*PerThread);
	uint end=Min_rmerge(n,start+(BlockSize*PerThread));
	for(uint i=start+threadIdx.x;i<end;i+=BlockSize)
		f(y[i*(StrideIsOne?1:yStride)],y[i*(StrideIsOne?1:yStride)]);
}

template<uint BlockSize, uint PerThread, typename DST, typename ElementFunctor>
void __cdecl CudaComponentWiseInline(CVector<DST> y, ElementFunctor functor){	
	uint n=(uint)y.Length();
	if(y.Stride()==1)
		CudaComponentWiseInlineKernel<true,BlockSize,PerThread> <<<dim3(DivUp(n,BlockSize*PerThread),1,1),dim3(BlockSize,1,1), 0>>>(y.Data(),y.Stride(),n,functor);
	else
		CudaComponentWiseInlineKernel<false,BlockSize,PerThread> <<<dim3(DivUp(n,BlockSize*PerThread),1,1),dim3(BlockSize,1,1), 0>>>(y.Data(),y.Stride(),n,functor);		
}
#endif

template<typename DST, typename ElementFunctor>
void __cdecl CudaComponentWiseInline(CVector<DST> y, ElementFunctor functor)
#ifdef __CUDACC__
{
	if(y.Length()==0)return;
	if(y.Length()<6500000)
		CudaComponentWiseInline<256,2>(y,functor);
	else
		CudaComponentWiseInline<256,256>(y,functor);
}
#else
;
#endif

#ifdef __CUDACC__

template<int BlockSize, int PerThread, typename DST, typename SRC, typename ElementFunctor>
__global__ void __cdecl CudaComponentWiseKernel(CVector<DST> y, CVector<SRC> x, ElementFunctor f){
	uint start=blockIdx.x*(BlockSize*PerThread);
	uint end=Min_rmerge(uint(x.Length()),start+(BlockSize*PerThread));
	for(uint i=start+threadIdx.x;i<end;i+=BlockSize)
		f(y[i],x[i]);
}

template<int BlockSize, int PerThread, typename DST, typename SRC, typename ElementFunctor>
__global__ void __cdecl CudaComponentWiseAddUpKernel(CVector<DST> y, CVector<SRC> x, ElementFunctor f){
	uint start=blockIdx.x*(BlockSize*PerThread);
	uint end=Min_rmerge(uint(x.Length()),start+(BlockSize*PerThread));
	for(uint i=start+threadIdx.x;i<end;i+=BlockSize){
		DST tmp=y[i];
		f(tmp,x[i]);
		y[i]+=tmp;
	}
}

template<bool StrideIsOne, bool AddUp, int BlockSize, int PerThread, typename DST, typename A, typename B, typename ElementFunctor>
__global__ void __cdecl CudaBinaryComponentWiseKernel(DST* y, uint yStride, A* a, uint aStride, B*b, uint bStride, uint n, ElementFunctor f){
	uint start=blockIdx.x*(BlockSize*PerThread);
	uint end=Min_rmerge(n,start+(BlockSize*PerThread));
	for(uint i=start+threadIdx.x;i<end;i+=BlockSize){
		if(AddUp){
			DST tmp=y[i*(StrideIsOne?1:yStride)];
			DST old=tmp;
			f(tmp,a[i*(StrideIsOne?1:aStride)],b[i*(StrideIsOne?1:bStride)]);
			y[i*(StrideIsOne?1:yStride)]=old+tmp;
		}
		else
			f(y[i*(StrideIsOne?1:yStride)],a[i*(StrideIsOne?1:aStride)],b[i*(StrideIsOne?1:bStride)]);
	}
}



template<uint BlockSize, uint PerThread, typename DST, typename SRC, typename ElementFunctor>
void __cdecl CudaComponentWise(CVector<DST> y, CVector<SRC> x, ElementFunctor functor){
	uint n=uint(x.Length());
	CudaComponentWiseKernel<BlockSize,PerThread><<<dim3(DivUp(n,BlockSize*PerThread),1,1),dim3(BlockSize,1,1), 0>>>(y,x,functor);
}

template<typename DST, typename SRC, typename ElementFunctor>
void __cdecl CudaComponentWise(CVector<DST> y, CVector<SRC> x, ElementFunctor functor){
	if(y.Length()==0)return;
	if(y.Length()<6500000)
		CudaComponentWise<256,2>(y,x,functor);
	else
		CudaComponentWise<256,256>(y,x,functor);
}


template<uint BlockSize, uint PerThread, typename DST, typename SRC, typename ElementFunctor>
void __cdecl CudaComponentWiseAddUp(CVector<DST> y, CVector<SRC> x, ElementFunctor functor){
	uint n=uint(x.Length());
	CudaComponentWiseAddUpKernel<BlockSize,PerThread><<<dim3(DivUp(n,BlockSize*PerThread),1,1),dim3(BlockSize,1,1), 0>>>(y,x,functor);
}

template<typename DST, typename SRC, typename ElementFunctor>
void __cdecl CudaComponentWiseAddUp(CVector<DST> y, CVector<SRC> x, ElementFunctor functor){
	if(y.Length()==0)return;
	if(y.Length()<6500000)
		CudaComponentWiseAddUp<256,2>(y,x,functor);
	else
		CudaComponentWiseAddUp<256,256>(y,x,functor);
}

template<bool AddUp, uint BlockSize, uint PerThread, typename DST, typename A, typename B, typename ElementFunctor>
void __cdecl CudaBinaryComponentWise(CVector<DST> y, CVector<A> a, CVector<B> b, ElementFunctor functor){
	uint n=(uint)y.Length();
	if(y.Stride()==1 && a.Stride()==1 && b.Stride()==1)
		CudaBinaryComponentWiseKernel<true,AddUp,BlockSize,PerThread><<<dim3(DivUp(n,BlockSize*PerThread),1,1),dim3(BlockSize,1,1), 0>>>(y.Data(),y.Stride(), a.Data(),a.Stride(),b.Data(),b.Stride(),n,functor);
	else
		CudaBinaryComponentWiseKernel<false,AddUp,BlockSize,PerThread><<<dim3(DivUp(n,BlockSize*PerThread),1,1),dim3(BlockSize,1,1), 0>>>(y.Data(),y.Stride(), a.Data(),a.Stride(),b.Data(),b.Stride(),n,functor);
}

template<bool AddUp, typename DST, typename A, typename B, typename ElementFunctor>
void __cdecl CudaBinaryComponentWise(CVector<DST> y, CVector<A> a, CVector<B> b, ElementFunctor functor){
	if(y.Length()==0)return;
	if(y.Length()<6500000)
		CudaBinaryComponentWise<AddUp,256,2>(y,a,b,functor);
	else
		CudaBinaryComponentWise<AddUp,256,256>(y,a,b,functor);
}


template<int ThreadCountX, int ThreadCountY, typename DST, typename SRC, typename ElementFunctor>
__global__ void __cdecl CudaComponentWise3DKernel(CCube<DST> dst, CCube<SRC> src, ElementFunctor functor){
	int x=blockIdx.x*ThreadCountX+threadIdx.x;
	int y=blockIdx.y*ThreadCountY+threadIdx.y;
	if(x>=dst.DimX() || y>=dst.DimY())
		return;
	DST* pDst=dst.RowPointerZ(x,y);
	SRC* pSrc=src.RowPointerZ(x,y);
	for(int z=0;z<dst.DimZ();z++){
		functor(*pDst,*pSrc);
		pDst+=dst.SliceStride();
		pSrc+=src.SliceStride();
	}
}

template<typename DST, typename SRC, typename ElementFunctor>
void __cdecl CudaComponentWise(DeviceCube<DST> dst, DeviceCube<SRC> src, ElementFunctor functor){	
	Verify(dst.Size()==src.Size(),FileAndLine);
	const int ThreadCountX=64;
	const int ThreadCountY=8;
	dim3 gridDim(DivUp(src.DimX(),ThreadCountX),DivUp(src.DimY(),ThreadCountY),1);
	dim3 blockDim(ThreadCountX,ThreadCountY,1);
	CudaComponentWise3DKernel<ThreadCountX, ThreadCountY><<<gridDim, blockDim,0>>>(dst.GetC(),src.GetC(),functor);
}

#else

template<typename DST, typename SRC, typename ElementFunctor>
void __cdecl CudaComponentWise(DeviceCube<DST> dst, DeviceCube<SRC> src, ElementFunctor functor);

template<typename DST, typename SRC, typename ElementFunctor>
void __cdecl CudaComponentWise(CVector<DST> y, CVector<SRC> x, ElementFunctor functor);

template<typename DST, typename SRC, typename ElementFunctor>
void __cdecl CudaComponentWiseAddUp(CVector<DST> y, CVector<SRC> x, ElementFunctor functor);

template<bool AddUp, typename DST, typename A, typename B, typename ElementFunctor>
void __cdecl CudaBinaryComponentWise(CVector<DST> y, CVector<A> a, CVector<B> b, ElementFunctor functor);

#endif
