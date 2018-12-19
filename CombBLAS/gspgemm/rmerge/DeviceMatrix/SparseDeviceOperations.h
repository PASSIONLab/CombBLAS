#pragma once
#include "DeviceMatrix/DeviceFourD.h"
#include "HostMatrix/SparseHostVector.h"
#include "DeviceMatrix/SparseDeviceVector.h"
#include "DeviceMatrix/CudaSparseDeviceOps.h"


template<typename T>
SparseDeviceVector<T> ToDevice(SparseHostVector<T> x){return SparseDeviceVector<T>(ToDevice(x.Values()),ToDevice(x.Indices()),x.Length());}


template<typename T> 
static void ExtractSparse(DeviceVector<T> sparse, DeviceVector<T> dense, DeviceVector<uint> indices){
	Verify(sparse.Length()==indices.Length(),"4s34r34r");
	CudaExtractSparse(sparse.GetC(),dense.GetC(),indices.GetC());
}

template<typename T> 
static void ExtractRows(DeviceMatrix<T> dst, DeviceMatrix<T> src, DeviceVector<uint> rowIndices){
	Verify(dst.Height()==rowIndices.Length() && dst.Width()==src.Width(),FileAndLine);
	CudaExtractRows(dst.GetC(),src.GetC(),rowIndices.GetC());
}

template<typename T> 
static void InjectRows(DeviceMatrix<T> dst, DeviceMatrix<T> src, DeviceVector<uint> rowIndices){
	Verify(src.Height()==rowIndices.Length() && dst.Width()==src.Width(),FileAndLine);
	CudaInjectRows(dst.GetC(),src.GetC(),rowIndices.GetC());
}

template<typename A, typename B> 
static void AddUpSparse(DeviceVector<A> dense, DeviceVector<B> sparse, DeviceVector<uint> indices){
	Verify(sparse.Length()==indices.Length(),"3434yy33");
	CudaInjectSparse<true>(dense.GetC(),sparse.GetC(),indices.GetC());	
}

template<typename T> 
static void InjectSparse(DeviceVector<T> dense, DeviceVector<T> sparse, DeviceVector<uint> indices){
	Verify(indices.Length()==sparse.Length(),"2s23r32r3");
	CudaInjectSparse<false>(dense.GetC(),sparse.GetC(),indices.GetC());
}

template<typename T> static void ExtractSparse(DeviceVector<T> dst, DeviceMatrix<T> src, DeviceVector<unsigned int> indices){ExtractSparse(dst,src.GetSimple(),indices);}
template<typename T> static void ExtractSparse(DeviceVector<T> dst, DeviceCube<T> src, DeviceVector<unsigned int> indices){ExtractSparse(dst,src.GetSimple(),indices);}
template<typename T> static void ExtractSparse(DeviceVector<T> dst, DeviceFourD<T> src, DeviceVector<unsigned int> indices){ExtractSparse(dst,src.GetSimple(),indices);}

template<typename T> static void Extract(SparseDeviceVector<T> dst, DeviceVector<T> src){ExtractSparse(dst.Values(),src,dst.Indices());}
template<typename T> static void Extract(SparseDeviceMatrix<T> dst, DeviceMatrix<T> src){Extract(dst.GetSimple(),src.GetSimple());}
template<typename T> static void Extract(SparseDeviceCube<T> dst, DeviceCube<T> src){Extract(dst.GetSimple(),src.GetSimple());}
template<typename T> static void Extract(SparseDeviceFourD<T> dst, DeviceFourD<T> src){Extract(dst.GetSimple(),src.GetSimple());}


template<typename A, typename B> static void AddUpSparse(DeviceMatrix<A> dst, DeviceVector<B> src, DeviceVector<unsigned int> indices){AddUpSparse(dst.GetSimple(),src,indices);}
template<typename A, typename B> static void AddUpSparse(DeviceCube<A> dst, DeviceVector<B> src, DeviceVector<unsigned int> indices){AddUpSparse(dst.GetSimple(),src,indices);}
template<typename A, typename B> static void AddUpSparse(DeviceFourD<A> dst, DeviceVector<B> src, DeviceVector<unsigned int> indices){AddUpSparse(dst.GetSimple(),src,indices);}


template<typename T> static void Inject(DeviceVector<T> dst, SparseDeviceVector<T> src){InjectSparse(dst,src.Values(),src.Indices());}
template<typename T> static void Inject(DeviceMatrix<T> dst, SparseDeviceMatrix<T> src){Inject(dst.GetSimple(),src.GetSimple());}
template<typename T> static void Inject(DeviceCube<T> dst, SparseDeviceCube<T> src){Inject(dst.GetSimple(),src.GetSimple());}
template<typename T> static void Inject(DeviceFourD<T> dst, SparseDeviceFourD<T> src){Inject(dst.GetSimple(),src.GetSimple());}

template<typename A, typename B> static void ComponentWiseAddUp(DeviceVector<A> dst, SparseDeviceVector<B> src){AddUpSparse(dst,src.Values(),src.Indices());}
template<typename A, typename B>  static void ComponentWiseAddUp(DeviceMatrix<A> dst, SparseDeviceMatrix<B> src){AddUp(dst.GetSimple(),src.GetSimple());}
template<typename A, typename B>  static void ComponentWiseAddUp(DeviceCube<A> dst, SparseDeviceCube<B> src){AddUp(dst.GetSimple(),src.GetSimple());}
template<typename A, typename B>  static void ComponentWiseAddUp(DeviceFourD<A> dst, SparseDeviceFourD<B> src){AddUp(dst.GetSimple(),src.GetSimple());}

template<typename A, typename B, typename S> 
static void ComponentWiseAddUpScaled(DeviceVector<A> dst, SparseDeviceVector<B> src, S scale){
	//CudaAddUpSparse(dst,src.Values(),src.Indices());
	throw std::runtime_error("");
	//CudaCall(AddUpScaledSparseVectorFunctor<A,B,S>(dst.Data(),src.Values().Data(),src.Indices().Data(),src.Indices().Length(),scale));
}

template<typename A, typename B, typename S> static void ComponentWiseAddUpScaled(DeviceMatrix<A> dst, SparseDeviceMatrix<B> src, S scale){ComponentWiseAddUpScaled(dst.GetSimple(),src.GetSimple(),scale);}
template<typename A, typename B, typename S> static void ComponentWiseAddUpScaled(DeviceCube<A> dst, SparseDeviceCube<B> src, S scale){ComponentWiseAddUpScaled(dst.GetSimple(),src.GetSimple(),scale);}
template<typename A, typename B, typename S> static void ComponentWiseAddUpScaled(DeviceFourD<A> dst, SparseDeviceFourD<B> src, S scale){ComponentWiseAddUpScaled(dst.GetSimple(),src.GetSimple(),scale);}
