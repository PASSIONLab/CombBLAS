#pragma once

#include "DeviceMatrix/DeviceVector.h"
#include "DeviceMatrix/DeviceMatrix.h"
#include "DeviceMatrix/DeviceCube.h"
#include "DeviceMatrix/DeviceFourD.h"
#include "DeviceMatrix/CudaComponentWise.h"
#include "HostMatrix/Verify.h"

//Provides template functions for ComponentWise operations.
//Operations for Matrix, Cube and FourD are sent through the Vector implementation

template<typename T, typename EF> void ComponentWiseInline(DeviceVector<T> x, EF f){
	if(x.Length()==0)return;
	CudaComponentWiseInline(x.GetC(),f);
}
template<typename T, typename EF> void ComponentWiseInline(DeviceMatrix<T> x, EF f){CudaComponentWiseInline(x,f);}
template<typename T, typename EF> void ComponentWiseInline(DeviceCube<T> cube, EF f){CudaComponentWiseInline(cube,f);}
template<typename T, typename EF> void ComponentWiseInline(DeviceFourD<T> x, EF f){ComponentWiseInline(x.GetSimple(),f);}

template<typename DST, typename SRC, typename EF>
void ComponentWise(DeviceVector<DST> dst, DeviceVector<SRC> src, EF f){
	Verify(dst.Length()==src.Length(),"Size mismatch 98129");	
	if(dst.Length()==0)return;
	CudaComponentWise(dst.GetC(),src.GetC(),f);
}

template<typename DST, typename SRC, typename EF> void ComponentWise(DeviceMatrix<DST> dst, DeviceMatrix<SRC> src, EF f){ComponentWise(dst.GetSimple(),src.GetSimple(),f);}
template<typename DST, typename SRC, typename EF> void ComponentWise(DeviceCube<DST> dst, DeviceCube<SRC> src, EF f){
	CudaComponentWise(dst,src,f);
}
template<typename DST, typename SRC, typename EF> void ComponentWise(DeviceFourD<DST> dst, DeviceFourD<SRC> src, EF f){ComponentWise(dst.GetSimple(),src.GetSimple(),f);}

template<typename T,typename EF> DeviceVector<T> ComponentWise(DeviceVector<T> x,EF f){DeviceVector<T> y(x.Size());ComponentWise(y.GetSimple(),x.GetSimple(),f);return y;}
template<typename T,typename EF> DeviceMatrix<T> ComponentWise(DeviceMatrix<T> x,EF f){DeviceMatrix<T> y(x.Size());ComponentWise(y.GetSimple(),x.GetSimple(),f);return y;}
template<typename T,typename EF> DeviceCube<T> ComponentWise(DeviceCube<T> x,EF f){DeviceCube<T> y(x.Size());ComponentWise(y.GetSimple(),x.GetSimple(),f);return y;}
template<typename T,typename EF> DeviceFourD<T> ComponentWise(DeviceFourD<T> x,EF f){DeviceFourD<T> y(x.Size());ComponentWise(y.GetSimple(),x.GetSimple(),f);return y;}

template<typename DST, typename SRC, typename EF>
void ComponentWiseAddUp(DeviceVector<DST> dst, DeviceVector<SRC> src, EF f){	
	Verify(dst.Length()==src.Length(),"092809sd8f09w8");
	if(dst.Length()==0)return;
	CudaComponentWiseAddUp(dst.GetC(),src.GetC(),f);
}
template<typename DST, typename SRC, typename EF> void ComponentWiseAddUp(DeviceMatrix<DST> dst, DeviceMatrix<SRC> src, EF f){ComponentWiseAddUp(dst.GetSimple(),src.GetSimple(),f);}
template<typename DST, typename SRC, typename EF> void ComponentWiseAddUp(DeviceCube<DST> dst, DeviceCube<SRC> src, EF f){ComponentWiseAddUp(dst.GetSimple(),src.GetSimple(),f);}
template<typename DST, typename SRC, typename EF> void ComponentWiseAddUp(DeviceFourD<DST> dst, DeviceFourD<SRC> src, EF f){ComponentWiseAddUp(dst.GetSimple(),src.GetSimple(),f);}

template<typename DST, typename A, typename B, typename EF>
void BinaryComponentWise(DeviceVector<DST> dst, DeviceVector<A> a, DeviceVector<B> b, EF f){
	Verify(dst.Length()==a.Length()&&dst.Length()==b.Length(),FileAndLine);		
	if(dst.Length()==0)return;
	CudaBinaryComponentWise<false>(dst.GetC(),a.GetC(),b.GetC(),f);
}

template<typename DST, typename A, typename B, typename EF> void BinaryComponentWise(DeviceMatrix<DST> dst, DeviceMatrix<A> a, DeviceMatrix<B> b, EF f){BinaryComponentWise(dst.GetSimple(),a.GetSimple(),b.GetSimple(),f);}
template<typename DST, typename A, typename B, typename EF> void BinaryComponentWise(DeviceCube<DST> dst, DeviceCube<A> a, DeviceCube<B> b, EF f){BinaryComponentWise(dst.GetSimple(),a.GetSimple(),b.GetSimple(),f);}
template<typename DST, typename A, typename B, typename EF> void BinaryComponentWise(DeviceFourD<DST> dst, DeviceFourD<A> a, DeviceFourD<B> b, EF f){BinaryComponentWise(dst.GetSimple(),a.GetSimple(),b.GetSimple(),f);}

template<typename DST, typename A, typename B, typename EF>
void BinaryComponentWiseAddUp(DeviceVector<DST> dst, DeviceVector<A> a, DeviceVector<B> b, EF f){	
	Verify(dst.Length()==a.Length()&&dst.Length()==b.Length(),"4dssdf23");
	if(dst.Length()==0)return;
	CudaBinaryComponentWise<true>(dst.GetC(),a.GetC(),b.GetC(),f);
}

template<typename DST, typename A, typename B, typename EF> void BinaryComponentWiseAddUp(DeviceMatrix<DST> dst, DeviceMatrix<A> a, DeviceMatrix<B> b, EF f){BinaryComponentWiseAddUp(dst.GetSimple(),a.GetSimple(),b.GetSimple(),f);}
template<typename DST, typename A, typename B, typename EF> void BinaryComponentWiseAddUp(DeviceCube<DST> dst, DeviceCube<A> a, DeviceCube<B> b, EF f){BinaryComponentWiseAddUp(dst.GetSimple(),a.GetSimple(),b.GetSimple(),f);}
template<typename DST, typename A, typename B, typename EF> void BinaryComponentWiseAddUp(DeviceFourD<DST> dst, DeviceFourD<A> a, DeviceFourD<B> b, EF f){BinaryComponentWiseAddUp(dst.GetSimple(),a.GetSimple(),b.GetSimple(),f);}
