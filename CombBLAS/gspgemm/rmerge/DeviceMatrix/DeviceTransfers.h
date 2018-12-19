#pragma once
#include "cuda_runtime_api.h"
#include "HostMatrix/HostVector.h"
#include "HostMatrix/HostMatrix.h"
#include "HostMatrix/HostCube.h"
#include "HostMatrix/HostFourD.h"
#include "HostMatrix/VectorOperators.h"

#include "DeviceMatrix/DeviceVector.h"
#include "DeviceMatrix/DeviceMatrix.h"
#include "DeviceMatrix/DeviceCube.h"
#include "DeviceMatrix/DeviceFourD.h"

#include "DeviceMatrix/CudaCheckError.h"

#include <exception>

template<typename T> 
void Copy(CVector<T> src, CVector<T> dst, cudaMemcpyKind kind){
	Verify(src.Size()==dst.Size(),"Size mismatch. 8s098ds1");
	if(src.Length()==0)return;
	if(kind==cudaMemcpyHostToDevice || kind==cudaMemcpyDeviceToHost)
		CudaCheckErrorImportant();
	if(src.IsSimple()&&dst.IsSimple())
		cudaMemcpy(dst.Data(),src.Data(),dst.DimX()*sizeof(T),kind);	
	else
		cudaMemcpy2D(dst.Data(),dst.Stride()*sizeof(T),src.Data(),src.Stride()*sizeof(T),sizeof(T),dst.Length(),kind);
	if(kind==cudaMemcpyHostToDevice || kind==cudaMemcpyDeviceToHost)
		CudaCheckErrorImportant();
}

template<typename T> void Copy(HostVector<T> src, DeviceVector<T> dst){Verify(src.Size()==dst.Size(),"Size mismatch 3344321");Copy(src.GetC(),dst.GetC(),cudaMemcpyHostToDevice);}
template<typename T> void Copy(DeviceVector<T> src, HostVector<T> dst){Verify(src.Size()==dst.Size(),"Size mismatch 090933");Copy(src.GetC(),dst.GetC(),cudaMemcpyDeviceToHost);}
template<typename T> void Copy(DeviceVector<T> src, DeviceVector<T> dst){Verify(src.Size()==dst.Size(),"Size mismatch 91898ed");Copy(src.GetC(),dst.GetC(),cudaMemcpyDeviceToDevice);}

template<typename T> 
void Copy(CMatrix<T> src, CMatrix<T> dst, cudaMemcpyKind kind){
	Verify(src.Size()==dst.Size(),"932090ffefs1");
	if(src.DimX()==0 || src.DimY()==0)return;
	if(src.IsSimple() && dst.IsSimple()){
		Copy(src.GetSimple(),dst.GetSimple(),kind);
		return;
	}
	if(kind==cudaMemcpyHostToDevice || kind==cudaMemcpyDeviceToHost)CudaCheckError();
	cudaMemcpy2D(dst.Data(),dst.Stride()*sizeof(T),src.Data(),src.Stride()*sizeof(T),dst.DimX()*sizeof(T),dst.Height(),kind);
	if(kind==cudaMemcpyHostToDevice || kind==cudaMemcpyDeviceToHost)CudaCheckError();
}

template<typename T> void Copy(HostMatrix<T> src, DeviceMatrix<T> dst){Verify(src.Size()==dst.Size(),"098109809");Copy(src.GetC(),dst.GetC(),cudaMemcpyHostToDevice);}
template<typename T> void Copy(DeviceMatrix<T> src, HostMatrix<T> dst){Verify(src.Size()==dst.Size(),"098109d809");Copy(src.GetC(),dst.GetC(),cudaMemcpyDeviceToHost);}
template<typename T> void Copy(DeviceMatrix<T> src, DeviceMatrix<T> dst){Verify(src.Size()==dst.Size(),"0f98109809");Copy(src.GetC(),dst.GetC(),cudaMemcpyDeviceToDevice);}


template<typename T>
cudaPitchedPtr Make_cudaPitchedPtr(CCube<T> cube){
	cudaPitchedPtr p;
	p.ptr=cube.Data();
	p.pitch=cube.RowStride()*sizeof(T);
	p.xsize=cube.DimX();
	//p.ysize=cube.DimY();
	int ratio=cube.SliceStride()/cube.RowStride();
	Verify(ratio*cube.RowStride()==cube.SliceStride(),FileAndLine);//sliceStride should be a multiple of rowStride
	p.ysize=ratio;	
	return p;
}

//static cudaPitchedPtr Make_cudaPitchedPtr( void *ptr,size_t  pitch, size_t  xsize,size_t  ysize){
//	cudaPitchedPtr p;
//	p.ptr=ptr;
//	p.pitch=pitch;
//	p.xsize=xsize;
//	p.ysize=ysize;
//	return p;
//}

template<typename T> 
void Copy(CCube<T> src, CCube<T> dst, cudaMemcpyKind kind){
	Verify(src.Size()==dst.Size(),"dc4c334");
	if(src.DimX()==0 || src.DimY()==0 || src.DimZ()==0)return;
	if(src.IsSimple() && dst.IsSimple()){
		Copy(src.GetSimple(),dst.GetSimple(),kind);
		return;
	}

	cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = Make_cudaPitchedPtr(src);
	copyParams.dstPtr   = Make_cudaPitchedPtr(dst);   
	cudaExtent extent;
	extent.width=dst.DimX()*sizeof(T);
	extent.height=dst.DimY();
	extent.depth=dst.DimZ();
    copyParams.extent   = extent;
    copyParams.kind     = kind;
    cudaMemcpy3D(&copyParams);

	//for(int z=0;z<src.DimZ();z++)
	//	Copy(src.SliceZ(z),dst.SliceZ(z),kind);
}

template<typename T> void Copy(HostCube<T> src, DeviceCube<T> dst){Verify(src.Size()==dst.Size(),FileAndLine);Copy(src.GetC(),dst.GetC(),cudaMemcpyHostToDevice);}
template<typename T> void Copy(DeviceCube<T> src, HostCube<T> dst){Verify(src.Size()==dst.Size(),FileAndLine);Copy(src.GetC(),dst.GetC(),cudaMemcpyDeviceToHost);}
template<typename T> void Copy(DeviceCube<T> src, DeviceCube<T> dst){Verify(src.Size()==dst.Size(),FileAndLine);Copy(src.GetC(),dst.GetC(),cudaMemcpyDeviceToDevice);}

template<typename T> void Copy(HostFourD<T> src, DeviceFourD<T> dst){Verify(src.Size()==dst.Size(),FileAndLine);Copy(src.GetSimple(),dst.GetSimple());}
template<typename T> void Copy(DeviceFourD<T> src, HostFourD<T> dst){Verify(src.Size()==dst.Size(),FileAndLine);Copy(src.GetSimple(),dst.GetSimple());}
template<typename T> void Copy(DeviceFourD<T> src, DeviceFourD<T> dst){Verify(src.Size()==dst.Size(),FileAndLine);Copy(src.GetSimple(),dst.GetSimple());}

template<typename T> HostVector<T> ToHost(DeviceVector<T> x){HostVector<T> y(x.Size());Copy(x,y);return y;}
template<typename T> DeviceVector<T> ToDevice(HostVector<T> x){DeviceVector<T> y(x.Size());Copy(x,y);return y;}
template<typename T> HostMatrix<T> ToHost(DeviceMatrix<T> x){HostMatrix<T> y(x.Size());Copy(x,y);return y;}
template<typename T> DeviceMatrix<T> ToDevice(HostMatrix<T> x){DeviceMatrix<T> y(x.Size());Copy(x,y);return y;}
template<typename T> HostCube<T> ToHost(DeviceCube<T> x){HostCube<T> y(x.Size());Copy(x,y);return y;}
template<typename T> DeviceCube<T> ToDevice(HostCube<T> x){DeviceCube<T> y(x.Size());Copy(x,y);return y;}
template<typename T> HostFourD<T> ToHost(DeviceFourD<T> x){HostFourD<T> y(x.Size());Copy(x,y);return y;}
template<typename T> DeviceFourD<T> ToDevice(HostFourD<T> x){DeviceFourD<T> y(x.Size());Copy(x,y);return y;}

template<typename T> DeviceVector<T> Clone(DeviceVector<T> x){DeviceVector<T> y(x.Size());Copy(x,y);return y;}
template<typename T> DeviceMatrix<T> Clone(DeviceMatrix<T> x){DeviceMatrix<T> y(x.Size());Copy(x,y);return y;}
template<typename T> DeviceCube<T> Clone(DeviceCube<T> x){DeviceCube<T> y(x.Size());Copy(x,y);return y;}
template<typename T> DeviceFourD<T> Clone(DeviceFourD<T> x){DeviceFourD<T> y(x.Size());Copy(x,y);return y;}

template<typename OStream, typename T>
static OStream& operator<<(OStream& o, DeviceVector<T> x){
	HostVector<T> tmp=ToHost(x);
	for(int i=0;i<(int)tmp.Length();i++)
		o<<tmp[i]<<" ";
	return o;
}

template<typename T>
static void Crop(DeviceCube<T> dst, DeviceCube<T> src, Int3 start){
	Copy(src.SubCube(start,dst.Size()),dst);
}

template<typename OSTREAM>
static void ReportDeviceMem(OSTREAM& o){
	size_t freeMem;
	size_t totalMem;
	cudaMemGetInfo(&freeMem,&totalMem);
	size_t usedMem=totalMem-freeMem;
	o<<"GPU Memory: used "<<usedMem/(1024*1024)<<"MB, free "<<freeMem/(1024*1024)<<"MB, total "<<totalMem/(1024*1024)<<"MB\n";
}