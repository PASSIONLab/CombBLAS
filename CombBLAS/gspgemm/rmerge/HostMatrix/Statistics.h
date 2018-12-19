#pragma once

#include "HostMatrix/HostVector.h"
#include "HostMatrix/HostMatrix.h"
#include "HostMatrix/HostCube.h"
#include "HostMatrix/Reductions.h"
#include <algorithm>

template<typename T>
static T Median(HostVector<T> x){
	HostVector<T> y=Clone(x);
	int n=x.Length32()/2;
	std::nth_element(y.Data(),y.Data()+n,y.Data()+y.Length());
	return y[n];
}

template<typename T>
static T MedianAbsoluteDeviation(HostVector<T> x){
	T median=Median(x);
	HostVector<T> y(x.Length());
	for(int64 i=0;i<x.Length();i++)
		y[i]=x[i]>=median?x[i]-median:median-x[i];
	return Median(y);
}

template<typename Dst, typename T> static void Mean(Dst& mean, HostVector<T> x){
	Dst sum;
	Sum(sum,x);
	mean=sum/Dst(x.Length());
}

template<typename Dst, typename T> static void Mean(Dst& mean, HostMatrix<T> x){
	Dst sum;
	Sum(sum,x);
	mean=sum/Dst(x.DimX()*int64(x.DimY()));
}

template<typename Dst, typename T> static void Mean(Dst& mean, HostCube<T> x){
	Dst sum;
	Sum(sum,x);
	int64 count=int64(x.DimX())*int64(x.DimY())*int64(x.DimZ());
	mean=sum/Dst(count);
}
template<typename Dst, typename T> static void Mean(Dst& mean, HostFourD<T> x){
	Dst sum;
	Sum(sum,x);
	int64 count=int64(x.DimX())*int64(x.DimY())*int64(x.DimZ())*int64(x.DimT());
	mean=sum/Dst(count);
}

template<typename T> static T Mean(HostVector<T> x){T mean;Mean(mean,x);return mean;}
template<typename T> static T Mean(HostMatrix<T> x){T mean;Mean(mean,x);return mean;}
template<typename T> static T Mean(HostCube<T> x){T mean;Mean(mean,x);return mean;}
template<typename T> static T Mean(HostFourD<T> x){T mean;Mean(mean,x);return mean;}

template<typename T>
static T Variance(HostVector<T> x){
	T sum=Sum(x);
	T sumSquared=SumSquared(x);
	T count((T)x.Length());
	return (sumSquared-sum*sum/count)/count;
}

template<typename T>
static T Variance(HostMatrix<T> x){
	T sum=Sum(x);
	T sumSquared=SumSquared(x);
	T count((T)(x.DimX()*x.DimY()));
	return (sumSquared-sum*sum/count)/count;
}


//template<typename T> static T Stddev(HostVector<T> x){return sqrt(Variance(x));}
template<typename T> static T Stddev(HostMatrix<T> x){return sqrt(Variance(x));}
