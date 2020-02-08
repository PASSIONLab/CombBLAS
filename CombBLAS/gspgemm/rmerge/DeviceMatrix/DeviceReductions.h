#pragma once
#include "DeviceMatrix/DeviceVector.h"
#include "DeviceMatrix/CudaReductions.h"
#include "DeviceMatrix/DeviceTransfers.h"
#include "HostMatrix/ReduceFunctors.h"
//#include "HostMatrix/MinMaxValues.h"
#include "HostMatrix/ComponentWiseNames.h"
#include "DeviceMatrix/DeviceComponentWise.h"
//#include <thrust/detail/static_assert.h>
#include <thrust/transform_reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

class SDHSDKGfhrtd{
public:
	__device__ __host__ uint operator()(uchar a){return a;}
};


#ifdef __CUDACC__
#ifdef INSTANTIATE_0
void __cdecl Sum(uint& sum, DeviceVector<uchar> x){
	Verify(x.IsSimple(),"ve987n59756");
	thrust::device_ptr<uchar> xBegin(x.Data());
	thrust::device_ptr<uchar> xEnd(x.Data()+x.Length());
	sum=thrust::transform_reduce(xBegin,xEnd,SDHSDKGfhrtd(),uint(0),thrust::plus<uint>());
}
#endif
#else
void __cdecl Sum(uint& sum, DeviceVector<uchar> x);
#endif

template<typename DST, typename T> static void Sum(DeviceVector<DST> sum, DeviceVector<T> x){
	Verify(sum.Length()==1,"1434111");
	Verify(x.Stride()==1,"r3f34");
	CudaReduceTransformed(sum.Data(),x.Data(),(int)x.Length(),ReduceFunctors::AddFunctor(),ElementFunctors::Identity(),DST(0));
}

template<typename T> static T Sum(DeviceVector<T> x){DeviceVector<T> s(1);Sum(s,x);return ToHost(s)[0];}
template<typename T> static T Sum(DeviceMatrix<T> x){return Sum(x.GetSimple());}
template<typename T> static T Sum(DeviceCube<T> x){return Sum(x.GetSimple());}
template<typename T> static T Sum(DeviceFourD<T> x){return Sum(x.GetSimple());}

template<typename DST, typename T> static void SumSquared(DeviceVector<DST> sum, DeviceVector<T> x){
	Verify(sum.Length()==1,"43t08089");
	Verify(x.Stride()==1,"f52992r");
	CudaReduceTransformed(sum.Data(),x.Data(),(int)x.Length(), ReduceFunctors::AddFunctor(),ElementFunctors::Square(),DST(0));
}
template<typename DST, typename T> static void SumSquared(DeviceVector<DST> sum, DeviceMatrix<T> x){SumSquared(sum,x.GetSimple());}
template<typename DST, typename T> static void SumSquared(DeviceVector<DST> sum, DeviceCube<T> x){SumSquared(sum,x.GetSimple());}
template<typename DST, typename T> static void SumSquared(DeviceVector<DST> sum, DeviceFourD<T> x){SumSquared(sum,x.GetSimple());}

template<typename T> static T SumSquared(DeviceVector<T> x){DeviceVector<T> sum(1);SumSquared(sum,x);return ToHost(sum)[0];}
template<typename T> static T SumSquared(DeviceMatrix<T> x){return SumSquared(x.GetSimple());}
template<typename T> static T SumSquared(DeviceCube<T> x){return SumSquared(x.GetSimple());}
template<typename T> static T SumSquared(DeviceFourD<T> x){return SumSquared(x.GetSimple());}



template<typename T> T Norm2(DeviceVector<T> a){return sqrt(SumSquared(a));}

//*****************

template<typename T> 
static void Max_rmerge(DeviceVector<T> result, DeviceVector<T> x){
	Verify(result.Length()==1,"44343ff3");
	Verify(x.Stride()==1,"2f24");
	CudaReduceTransformed(result.Data(),x.Data(),(int)x.Length(), ReduceFunctors::MaxFunctor(),ElementFunctors::Identity(),x[0]);
}
template<typename T> static void Max_rmerge(DeviceVector<T> sum, DeviceMatrix<T> x){Max_rmerge(sum,x.GetSimple());}
template<typename T> static void Max_rmerge(DeviceVector<T> sum, DeviceCube<T> x){Max_rmerge(sum,x.GetSimple());}
template<typename T> static void Max_rmerge(DeviceVector<T> sum, DeviceFourD<T> x){Max_rmerge(sum,x.GetSimple());}

template<typename T> static T Max_rmerge(DeviceVector<T> x){DeviceVector<T> sum(1);Max_rmerge(sum,x);return ToHost(sum)[0];}
template<typename T> static T Max_rmerge(DeviceMatrix<T> x){return Max_rmerge(x.GetSimple());}
template<typename T> static T Max_rmerge(DeviceCube<T> x){return Max_rmerge(x.GetSimple());}
template<typename T> static T Max_rmerge(DeviceFourD<T> x){return Max_rmerge(x.GetSimple());}

//*************************************

template<typename T> static void Min_rmerge(DeviceVector<T> result, DeviceVector<T> x){
	Verify(result.Length()==1,"43x33xr3rdxvv");
	Verify(x.Stride()==1,"x5t45t4se");
	CudaReduceTransformed(result.Data(),x.Data(),(int)x.Length(), ReduceFunctors::MinFunctor(),ElementFunctors::Identity(),x[0]);
}
template<typename T> static void Min_rmerge(DeviceVector<T> sum, DeviceMatrix<T> x){Min_rmerge(sum,x.GetSimple());}
template<typename T> static void Min_rmerge(DeviceVector<T> sum, DeviceCube<T> x){Min_rmerge(sum,x.GetSimple());}
template<typename T> static void Min_rmerge(DeviceVector<T> sum, DeviceFourD<T> x){Min_rmerge(sum,x.GetSimple());}

template<typename T> static T Min_rmerge(DeviceVector<T> x){DeviceVector<T> sum(1);ComponentWiseInit(sum,T(0));Min_rmerge(sum,x);return ToHost(sum)[0];}
template<typename T> static T Min_rmerge(DeviceMatrix<T> x){return Min_rmerge(x.GetSimple());}
template<typename T> static T Min_rmerge(DeviceCube<T> x){return Min_rmerge(x.GetSimple());}
template<typename T> static T Min_rmerge(DeviceFourD<T> x){return Min_rmerge(x.GetSimple());}



//*************************************
template<typename DST, typename T> static void NormMax(DeviceVector<DST> result, DeviceVector<T> x){
	Verify(result.Length()==1,"4wxrx32r");
	Verify(x.Stride()==1,"344x2t3");
	CudaReduceTransformed(result.Data(),x.Data(),(int)x.Length(), ReduceFunctors::MaxFunctor(),ElementFunctors::Absolute(),DST(0));
}
template<typename DST, typename T> static void NormMax(DeviceVector<DST> sum, DeviceMatrix<T> x){NormMax(sum,x.GetSimple());}
template<typename DST, typename T> static void NormMax(DeviceVector<DST> sum, DeviceCube<T> x){NormMax(sum,x.GetSimple());}
template<typename DST, typename T> static void NormMax(DeviceVector<DST> sum, DeviceFourD<T> x){NormMax(sum,x.GetSimple());}

template<typename T> static T NormMax(DeviceVector<T> x){DeviceVector<T> sum(1);NormMax(sum,x);return ToHost(sum)[0];}
template<typename T> static T NormMax(DeviceMatrix<T> x){return NormMax(x.GetSimple());}
template<typename T> static T NormMax(DeviceCube<T> x){return NormMax(x.GetSimple());}
template<typename T> static T NormMax(DeviceFourD<T> x){return NormMax(x.GetSimple());}

template<typename DST, typename A, typename B> static void Dot(DeviceVector<DST> dst, DeviceVector<A> a, DeviceVector<B> b){
	Verify(dst.Length()==1,"3435t45");
	Verify(a.Length()==b.Length(),"34345325");
	if(a.Stride()==1 && b.Stride()==1)
		CudaReduceCombined(dst.Data(),a.Data(),b.Data(),(int)a.Length(),ReduceFunctors::AddFunctor(),BinaryFunctors::Mul(),DST(0));
	else
		CudaReduceCombined(dst.Data(),a.Data(),a.Stride(),b.Data(),b.Stride(),(int)a.Length(),ReduceFunctors::AddFunctor(),BinaryFunctors::Mul(),DST(0));
}

template<typename DST, typename A, typename B> static void Dot(DST& dst, DeviceVector<A> a, DeviceVector<B> b){
	Verify(a.Length()==b.Length(),"3434f3c");
	DeviceVector<DST> tmp(1);
	Dot(tmp,a,b);
	HostVector<DST> bla=ToHost(tmp);
	dst=bla[0];
}

template<typename DST, typename A, typename B> static void Dot(DeviceVector<DST> result, DeviceMatrix<A> a, DeviceMatrix<B> b){Dot(result,a.GetSimple(),b.GetSimple());}
template<typename DST, typename A, typename B> static void Dot(DeviceVector<DST> result, DeviceCube<A> a, DeviceCube<B> b){Dot(result,a.GetSimple(),b.GetSimple());}
template<typename DST, typename A, typename B> static void Dot(DeviceVector<DST> result, DeviceFourD<A> a, DeviceFourD<B> b){Dot(result,a.GetSimple(),b.GetSimple());}


template<typename A, typename B> static A Dot(DeviceVector<A> a, DeviceVector<B> b){DeviceVector<A> sum(1);Dot(sum,a,b);return ToHost(sum)[0];}
template<typename A, typename B> static A Dot(DeviceMatrix<A> a, DeviceMatrix<B> b){return Dot(a.GetSimple(),b.GetSimple());}
template<typename A, typename B> static A Dot(DeviceCube<A> a, DeviceCube<B> b){return Dot(a.GetSimple(),b.GetSimple());}
template<typename A, typename B> static A Dot(DeviceFourD<A> a, DeviceFourD<B> b){return Dot(a.GetSimple(),b.GetSimple());}


//*************************************
template<typename T> static bool Equal(DeviceVector<T> a, DeviceVector<T> b){
	if(a.Length()!=b.Length())
		return false;
	Verify(a.IsSimple() && b.IsSimple(),"xww54t4wtxw");
	DeviceVector<bool> dst(1);
	CudaReduceCombined(dst.Data(),a.Data(),b.Data(),(int)a.Length(),ReduceFunctors::AndFunctor(),BinaryFunctors::Equal(),true);
	HostVector<bool> tmp=ToHost(dst);
	return tmp[0];
}

template<typename T> static bool Equal(DeviceMatrix<T> a, DeviceMatrix<T> b){if(a.Size()!=b.Size())return false;return Equal(a.GetSimple(),b.GetSimple());}
template<typename T> static bool Equal(DeviceCube<T> a, DeviceCube<T> b){if(a.Size()!=b.Size())return false;return Equal(a.GetSimple(),b.GetSimple());}
template<typename T> static bool Equal(DeviceFourD<T> a, DeviceFourD<T> b){if(a.Size()!=b.Size())return false;return Equal(a.GetSimple(),b.GetSimple());}

template<typename DST, typename A, typename B> static void DistanceSquared(DeviceVector<DST> dst, DeviceVector<A> a, DeviceVector<B> b){
	Verify(dst.Length()==1,"322xr2");
	Verify(a.Length()==b.Length(),"5z35665");
	if(a.Stride()==1 && b.Stride()==1)
		CudaReduceCombined(dst.Data(),a.Data(),b.Data(),(int)a.Length(),ReduceFunctors::AddFunctor(),BinaryFunctors::SquaredDifference(),DST(0));
	else
		CudaReduceCombined(dst.Data(),a.Data(),a.Stride(),b.Data(),b.Stride(),(int)a.Length(),ReduceFunctors::AddFunctor(),BinaryFunctors::SquaredDifference(),DST(0));
}
template<typename DST, typename A, typename B> static void DistanceSquared(DeviceVector<DST> dst, DeviceMatrix<A> a, DeviceMatrix<B> b){DistanceSquared(dst,a.GetSimple(),b.GetSimple());}
template<typename DST, typename A, typename B> static void DistanceSquared(DeviceVector<DST> dst, DeviceCube<A> a, DeviceCube<B> b){DistanceSquared(dst,a.GetSimple(),b.GetSimple());}
template<typename DST, typename A, typename B> static void DistanceSquared(DeviceVector<DST> dst, DeviceFourD<A> a, DeviceFourD<B> b){DistanceSquared(dst,a.GetSimple(),b.GetSimple());}

template<typename A, typename B> static A DistanceSquared(DeviceVector<A> a, DeviceVector<B> b){DeviceVector<A> sum(1);DistanceSquared(sum,a,b);return ToHost(sum)[0];}
template<typename A, typename B> static A DistanceSquared(DeviceMatrix<A> a, DeviceMatrix<B> b){return DistanceSquared(a.GetSimple(),b.GetSimple());}
template<typename A, typename B> static A DistanceSquared(DeviceCube<A> a, DeviceCube<B> b){return DistanceSquared(a.GetSimple(),b.GetSimple());}
template<typename A, typename B> static A DistanceSquared(DeviceFourD<A> a, DeviceFourD<B> b){return DistanceSquared(a.GetSimple(),b.GetSimple());}


#if defined (__CUDACC__) && defined (INSTANTIATE_0)

#include <thrust/inner_product.h>

class AbsDiff_vncn4x979487{
public:
	template<typename T>
	__device__ __host__ T operator()(T a, T b){if(a>b)return a-b;else return b-a;}
};

class AbsDiff_vncn4x979487_float{
public:
	__device__ __host__ float operator()(float a, float b){if(a>b)return a-b;else return b-a;}
};

class AbsDiff_vncn4x979487_uint{
public:
	__device__ __host__ uint operator()(uint a, uint b){if(a>b)return a-b;else return b-a;}
};

class AbsDiff_vncn4x979487_double{
public:
	__device__ __host__ double operator()(double a, double b){if(a>b)return a-b;else return b-a;}
};

uint __cdecl DistanceMax(DeviceVector<uint> a, DeviceVector<uint> b){	
	thrust::device_ptr<uint> aBegin(a.Data());
	thrust::device_ptr<uint> aEnd(a.Data()+a.Length());
	thrust::device_ptr<uint> bBegin(b.Data());
	uint result=thrust::inner_product(aBegin,aEnd,bBegin,uint(0),thrust::maximum<uint>(),AbsDiff_vncn4x979487_uint());
	return result;
}

double __cdecl DistanceMax(DeviceVector<double> a, DeviceVector<double> b){	
	thrust::device_ptr<double> aBegin(a.Data());
	thrust::device_ptr<double> aEnd(a.Data()+a.Length());
	thrust::device_ptr<double> bBegin(b.Data());
	double result=thrust::inner_product(aBegin,aEnd,bBegin,double(0),thrust::maximum<double>(),AbsDiff_vncn4x979487_double());
	return result;
}

float __cdecl DistanceMax(DeviceVector<float> a, DeviceVector<float> b){	
	thrust::device_ptr<float> aBegin(a.Data());
	thrust::device_ptr<float> aEnd(a.Data()+a.Length());
	thrust::device_ptr<float> bBegin(b.Data());
	float result=thrust::inner_product(aBegin,aEnd,bBegin,float(0),thrust::maximum<float>(),AbsDiff_vncn4x979487_float());
	return result;
}

uint __cdecl DistanceSum(DeviceVector<uint> a, DeviceVector<uint> b){	
	thrust::device_ptr<uint> aBegin(a.Data());
	thrust::device_ptr<uint> aEnd(a.Data()+a.Length());
	thrust::device_ptr<uint> bBegin(b.Data());
	uint result=thrust::inner_product(aBegin,aEnd,bBegin,uint(0),thrust::plus<uint>(),AbsDiff_vncn4x979487_uint());
	return result;
}

#else
uint __cdecl DistanceMax(DeviceVector<uint> a, DeviceVector<uint> b);
uint __cdecl DistanceSum(DeviceVector<uint> a, DeviceVector<uint> b);
double __cdecl DistanceMax(DeviceVector<double> a, DeviceVector<double> b);
float __cdecl DistanceMax(DeviceVector<float> a, DeviceVector<float> b);
#endif


template<typename T> static T Mean(DeviceVector<T> x){return Sum(x)/T(x.Length());}
template<typename T> static T Mean(DeviceMatrix<T> x){return Sum(x)/T(x.DimX()*x.DimY());}
template<typename T> static T Mean(DeviceCube<T> x){return Sum(x)/T(x.DimX()*x.DimY()*x.DimZ());}

template<typename T>
static T Variance(DeviceVector<T> x){
	T sum=Sum(x);
	T sumSquared=SumSquared(x);
	T count((T)x.Length());
	return (sumSquared-sum*sum/count)/count;
}

template<typename T>
static T Variance(DeviceMatrix<T> x){
	T sum=Sum(x);
	T sumSquared=SumSquared(x);
	T count((T)(x.DimX()*x.DimY()));
	return (sumSquared-sum*sum/count)/count;
}

template<typename T> static T Stddev(DeviceVector<T> x){return sqrt(Variance(x));}
template<typename T> static T Stddev(DeviceMatrix<T> x){return sqrt(Variance(x));}
