#pragma once

#include "HostMatrix/Intrinsics.h"
#include "HostMatrix/HostVector.h"
#include "HostMatrix/HostMatrix.h"
#include "HostMatrix/HostCube.h"
#include "HostMatrix/HostFourD.h"
#include "HostMatrix/ReduceFunctors.h"
#include "HostMatrix/ElementFunctors.h"
#include <omp.h>


template<typename Dst, typename T, typename ReduceFunctor>
static void Reduce(Dst& dst, HostVector<T> src, ReduceFunctor reduceFunctor){	
	if(src.Length()<1025){
		Dst sum(src[0]);
		for(int64 i=1;i<src.Length();i++)
			reduceFunctor(sum,src[i]);
		dst=sum;
	}
	else{
		int64 blockSize=1024;
		HostVector<Dst> tmp(DivUp(src.Length(),blockSize));
		#pragma omp parallel for
		for(int64 i=0;i<tmp.Length();i++){
			int64 start=i*blockSize;
			int64 end=Min_rmerge(src.Length(),start+blockSize);
			Dst sum(src[start]);
			for(int64 t=start+1;t<end;t++)
				reduceFunctor(sum,src[t]);
			tmp[i]=sum;
		}
		Reduce(dst,tmp,reduceFunctor);
	}
}

template<typename Dst, typename T, typename ReduceFunctor, typename Transform>
static void ReduceTransformed(Dst& dst, CVector<T> src, ReduceFunctor reduceFunctor, Transform transform){	
	//TODO: Parallelize for large vectors
	Dst sum;	
	transform(sum,src[0]);
	Dst tmp;
	for(int64 i=1;i<src.Length();i++){		
		transform(tmp,src[i]);
		reduceFunctor(sum,tmp);
	}
	dst=sum;
}
/*
template<typename Dst, typename T, typename ReduceFunctor, typename Transform>
static void ReduceTransformed(Dst& dst, CVector<T> src, ReduceFunctor reduceFunctor, Transform transform){
	transform(dst,src[0]);
	#pragma omp parallel
	{
		Dst sum=Dst(0);
		#pragma omp parallel for
		for(int64 i=1;i<src.Length();i++){
			Dst tmp;
			transform(tmp,src[i]);
			sum=reduceFunctor(sum,tmp);
		}
		#pragma omp critical 
		{
			dst=reduceFunctor(dst,sum);
		}
	}
}
*/


template<typename Dst, typename T, typename ReduceFunctor, typename Transform>
static void ReduceTransformed(Dst& dst, HostVector<T> src, ReduceFunctor reduceFunctor, Transform transform){
	ReduceTransformed(dst,src.GetC(),reduceFunctor,transform);
}

template<typename Dst, typename T, typename ReduceFunctor, typename Transform>
static void ReduceTransformed(Dst& dst, HostMatrix<T> src, ReduceFunctor reduceFunctor, Transform transform){
	HostVector<Dst> tmp(src.DimY());
	#pragma omp parallel for
	for(int i=0;i<src.DimY();i++)
		ReduceTransformed(tmp[i],src.Row(i),reduceFunctor,transform);
	ReduceTransformed(dst,tmp,reduceFunctor,ElementFunctors::Identity());
}

template<typename Dst, typename T, typename ReduceFunctor, typename Transform>
static void ReduceTransformed(Dst& dst, HostCube<T> src, ReduceFunctor reduceFunctor, Transform transform){
	HostVector<Dst> tmp(src.DimZ());
	#pragma omp parallel for
	for(int i=0;i<src.DimZ();i++)
		ReduceTransformed(tmp[i],src.SliceZ(i),reduceFunctor,transform);
	ReduceTransformed(dst,tmp,reduceFunctor,ElementFunctors::Identity());
}

template<typename Dst, typename T, typename ReduceFunctor, typename Transform>
static void ReduceTransformed(Dst& dst, HostFourD<T> src, ReduceFunctor reduceFunctor, Transform transform){
	if(src.DimT()==1)
		ReduceTransformed(dst,src.CubeT(0),reduceFunctor,transform);
	else{
		HostVector<Dst> tmp(src.DimT());
		#pragma omp parallel for
		for(int i=0;i<src.DimT();i++)
			ReduceTransformed(tmp[i],src.CubeT(i),reduceFunctor,transform);
		ReduceTransformed(dst,tmp,reduceFunctor,ElementFunctors::Identity());
	}
}

template<typename Dst, typename A, typename B, typename ReduceFunctor, typename Combine>
static void ReduceCombined(Dst& dst, CVector<A> a, CVector<B> b, ReduceFunctor reduceFunctor, Combine combine){
	Verify(a.Length()==b.Length(),"Length mismatch. 3232dsds");
	Dst sum;
	combine(sum,a[0],b[0]);
	//TODO: Parallelize for large vectors
	for(int64 i=1;i<a.Length();i++){
		Dst tmp;
		combine(tmp,a[i],b[i]);
		reduceFunctor(sum,sum,tmp);
	}
	dst=sum;
}

template<typename Dst, typename A, typename B, typename ReduceFunctor, typename Combine>
static void ReduceCombined(Dst& dst, HostVector<A> a, HostVector<B> b, ReduceFunctor reduceFunctor, Combine combine){
	ReduceCombined(dst,a.GetC(),b.GetC(),reduceFunctor,combine);
}
template<typename Dst, typename A, typename B, typename ReduceFunctor, typename Combine>
static void ReduceCombined(Dst& dst, HostMatrix<A> a, HostMatrix<B> b, ReduceFunctor reduceFunctor, Combine combine){
	Verify(a.Size()==b.Size(),"ReduceCombined 2D");
	HostVector<Dst> tmp(a.DimY());
	#pragma omp parallel for
	for(int i=0;i<a.DimY();i++)
		ReduceCombined(tmp[i],a.Row(i),b.Row(i),reduceFunctor,combine);
	ReduceTransformed(dst,tmp,reduceFunctor,ElementFunctors::Identity());
}

template<typename Dst, typename A, typename B, typename ReduceFunctor, typename Combine>
static void ReduceCombined(Dst& dst, HostCube<A> a, HostCube<B> b, ReduceFunctor reduceFunctor, Combine combine){
	Verify(a.Size()==b.Size(),"ReduceCombined 3D");
	HostVector<Dst> tmp(a.DimZ());
	#pragma omp parallel for
	for(int i=0;i<a.DimZ();i++)
		ReduceCombined(tmp[i],a.SliceZ(i),b.SliceZ(i),reduceFunctor,combine);
	ReduceTransformed(dst,tmp,reduceFunctor,ElementFunctors::Identity());
}

template<typename Dst, typename A, typename B, typename ReduceFunctor, typename Combine>
static void ReduceCombined(Dst& dst, HostFourD<A> a, HostFourD<B> b, ReduceFunctor reduceFunctor, Combine combine){
	Verify(a.Size()==b.Size(), "ReduceCombined 4D");
	HostVector<Dst> tmp(a.DimT());
	#pragma omp parallel for
	for(int i=0;i<a.DimT();i++)
		ReduceCombined(tmp[i],a.CubeT(i),b.CubeT(i),reduceFunctor,combine);
	ReduceTransformed(dst,tmp,reduceFunctor,ElementFunctors::Identity());
}

template<typename DST, typename T> static void Sum(DST& sum, HostVector<T> x){Reduce(sum,x,ReduceFunctors::AddFunctor());}
template<typename DST, typename T> static void Sum(DST& sum, HostMatrix<T> x){ReduceTransformed(sum,x,ReduceFunctors::AddFunctor(), ElementFunctors::Identity());}
template<typename DST, typename T> static void Sum(DST& sum, HostCube<T> x){ReduceTransformed(sum,x,ReduceFunctors::AddFunctor(), ElementFunctors::Identity());}
template<typename DST, typename T> static void Sum(DST& sum, HostFourD<T> x){ReduceTransformed(sum,x,ReduceFunctors::AddFunctor(), ElementFunctors::Identity());}
template<typename T> static T Sum(HostVector<T> x){T sum;Sum(sum,x);return sum;}
template<typename T> static T Sum(HostMatrix<T> x){T sum;Sum(sum,x);return sum;}
template<typename T> static T Sum(HostCube<T> x){T sum;Sum(sum,x);return sum;}
template<typename T> static T Sum(HostFourD<T> x){T sum;Sum(sum,x);return sum;}


template<typename T> static T Min_rmerge(HostVector<T> x){T dst;ReduceTransformed(dst,x,ReduceFunctors::MinFunctor(),ElementFunctors::Identity());return dst;}
template<typename T> static T Min_rmerge(HostMatrix<T> x){T dst;ReduceTransformed(dst,x,ReduceFunctors::MinFunctor(),ElementFunctors::Identity());return dst;}
template<typename T> static T Min_rmerge(HostCube<T> x){T dst;ReduceTransformed(dst,x,ReduceFunctors::MinFunctor(),ElementFunctors::Identity());return dst;}
template<typename T> static T Min_rmerge(HostFourD<T> x){T dst;ReduceTransformed(dst,x,ReduceFunctors::MinFunctor(),ElementFunctors::Identity());return dst;}

template<typename T> static T Max_rmerge(HostVector<T> x){T dst;ReduceTransformed(dst,x,ReduceFunctors::MaxFunctor(),ElementFunctors::Identity());return dst;}
template<typename T> static T Max_rmerge(HostMatrix<T> x){T dst;ReduceTransformed(dst,x,ReduceFunctors::MaxFunctor(),ElementFunctors::Identity());return dst;}
template<typename T> static T Max_rmerge(HostCube<T> x){T dst;ReduceTransformed(dst,x,ReduceFunctors::MaxFunctor(),ElementFunctors::Identity());return dst;}
template<typename T> static T Max_rmerge(HostFourD<T> x){T dst;ReduceTransformed(dst,x,ReduceFunctors::MaxFunctor(),ElementFunctors::Identity());return dst;}

template<typename Dst, typename T> static void SumSquared(Dst& sum, HostVector<T> x){ReduceTransformed(sum,x,ReduceFunctors::AddFunctor(),ElementFunctors::Square());}
template<typename Dst, typename T> static void SumSquared(Dst& sum, HostMatrix<T> x){ReduceTransformed(sum,x,ReduceFunctors::AddFunctor(),ElementFunctors::Square());}
template<typename Dst, typename T> static void SumSquared(Dst& sum, HostCube<T> x){ReduceTransformed(sum,x,ReduceFunctors::AddFunctor(),ElementFunctors::Square());}
template<typename Dst, typename T> static void SumSquared(Dst& sum, HostFourD<T> x){ReduceTransformed(sum,x,ReduceFunctors::AddFunctor(),ElementFunctors::Square());}
template<typename T> static T SumSquared(HostVector<T> x){T sum;SumSquared(sum,x);return sum;}
template<typename T> static T SumSquared(HostMatrix<T> x){T sum;SumSquared(sum,x);return sum;}
template<typename T> static T SumSquared(HostCube<T> x){T sum;SumSquared(sum,x);return sum;}
template<typename T> static T SumSquared(HostFourD<T> x){T sum;SumSquared(sum,x);return sum;}

template<typename Dst, typename T> static void SumHuber(Dst& sum, HostVector<T> x){ReduceTransformed(sum,x,ReduceFunctors::AddFunctor(),ElementFunctors::Huber());}
template<typename T> static T SumHuber(HostVector<T> x){T sum;SumHuber(sum,x);return sum;}

template<typename T> static bool Equal(HostVector<T> a, HostVector<T> b){if(a.Size()!=b.Size())return false;bool dst;ReduceCombined(dst,a,b,ReduceFunctors::MinFunctor(),BinaryFunctors::Equal());return dst;}
template<typename T> static bool Equal(HostMatrix<T> a, HostMatrix<T> b){if(a.Size()!=b.Size())return false;bool dst;ReduceCombined(dst,a,b,ReduceFunctors::MinFunctor(),BinaryFunctors::Equal());return dst;}
template<typename T> static bool Equal(HostCube<T> a, HostCube<T> b){if(a.Size()!=b.Size())return false;bool dst;ReduceCombined(dst,a,b,ReduceFunctors::MinFunctor(),BinaryFunctors::Equal());return dst;}
template<typename T> static bool Equal(HostFourD<T> a, HostFourD<T> b){if(a.Size()!=b.Size())return false;bool dst;ReduceCombined(dst,a,b,ReduceFunctors::MinFunctor(),BinaryFunctors::Equal());return dst;}


template<typename Dst, typename A, typename B> static void Dot(Dst& dst, HostVector<A> a, HostVector<B> b){ReduceCombined(dst,a,b,ReduceFunctors::AddFunctor(),BinaryFunctors::Mul());}
template<typename Dst, typename A, typename B> static void Dot(Dst& dst, HostMatrix<A> a, HostMatrix<B> b){ReduceCombined(dst,a,b,ReduceFunctors::AddFunctor(),BinaryFunctors::Mul());}
template<typename Dst, typename A, typename B> static void Dot(Dst& dst, HostCube<A> a, HostCube<B> b){ReduceCombined(dst,a,b,ReduceFunctors::AddFunctor(),BinaryFunctors::Mul());}
template<typename Dst, typename A, typename B> static void Dot(Dst& dst, HostFourD<A> a, HostFourD<B> b){ReduceCombined(dst,a,b,ReduceFunctors::AddFunctor(),BinaryFunctors::Mul());}
template<typename T> static T Dot(HostVector<T> a, HostVector<T> b){T dst;Dot(dst,a,b);return dst;}
template<typename T> static T Dot(HostMatrix<T> a, HostMatrix<T> b){T dst;Dot(dst,a,b);return dst;}
template<typename T> static T Dot(HostCube<T> a, HostCube<T> b){T dst;Dot(dst,a,b);return dst;}
template<typename T> static T Dot(HostFourD<T> a, HostFourD<T> b){T dst;Dot(dst,a,b);return dst;}

template<typename Dst, typename TA, typename TB> static void DistanceSquared(Dst& dst, HostVector<TA> a, HostVector<TB> b){	ReduceCombined(dst,a,b,ReduceFunctors::AddFunctor(),BinaryFunctors::SquaredDifference());}
template<typename Dst, typename T> static void DistanceSquared(Dst& dst, HostMatrix<T> a, HostMatrix<T> b){	ReduceCombined(dst,a,b,ReduceFunctors::AddFunctor(),BinaryFunctors::SquaredDifference());}
template<typename Dst, typename T> static void DistanceSquared(Dst& dst, HostCube<T> a, HostCube<T> b){	ReduceCombined(dst,a,b,ReduceFunctors::AddFunctor(),BinaryFunctors::SquaredDifference());}
template<typename Dst, typename T> static void DistanceSquared(Dst& dst, HostFourD<T> a, HostFourD<T> b){ReduceCombined(dst,a,b,ReduceFunctors::AddFunctor(),BinaryFunctors::SquaredDifference());}
template<typename T> static T DistanceSquared(HostVector<T> a, HostVector<T> b){T sum;DistanceSquared(sum,a,b);return sum;}
template<typename T> static T DistanceSquared(HostMatrix<T> a, HostMatrix<T> b){T sum;DistanceSquared(sum,a,b);return sum;}
template<typename T> static T DistanceSquared(HostCube<T> a, HostCube<T> b){T sum;DistanceSquared(sum,a,b);return sum;}
template<typename T> static T DistanceSquared(HostFourD<T> a, HostFourD<T> b){T sum;DistanceSquared(sum,a,b);return sum;}

template<typename T> static T DistanceMax(HostVector<T> a, HostVector<T> b){T d;ReduceCombined(d,a,b,ReduceFunctors::MaxFunctor(),BinaryFunctors::DistanceMax());return d;}
template<typename T> static T DistanceMax(HostMatrix<T> a, HostMatrix<T> b){T d;ReduceCombined(d,a,b,ReduceFunctors::MaxFunctor(),BinaryFunctors::DistanceMax());return d;}
template<typename T> static T DistanceMax(HostCube<T> a, HostCube<T> b){T d;ReduceCombined(d,a,b,ReduceFunctors::MaxFunctor(),BinaryFunctors::DistanceMax());return d;}
template<typename T> static T DistanceMax(HostFourD<T> a, HostFourD<T> b){T d;ReduceCombined(d,a,b,ReduceFunctors::MaxFunctor(),BinaryFunctors::DistanceMax());return d;}

template<typename T> T NormMax(HostVector<T> x){T dst;ReduceTransformed(dst,x,ReduceFunctors::MaxFunctor(),ElementFunctors::Absolute());return dst;}
template<typename T> T NormMax(HostMatrix<T> x){T dst;ReduceTransformed(dst,x,ReduceFunctors::MaxFunctor(),ElementFunctors::Absolute());return dst;}
template<typename T> T NormMax(HostCube<T> x){T dst;ReduceTransformed(dst,x,ReduceFunctors::MaxFunctor(),ElementFunctors::Absolute());return dst;}
template<typename T> T NormMax(HostFourD<T> x){T dst;ReduceTransformed(dst,x,ReduceFunctors::MaxFunctor(),ElementFunctors::Absolute());return dst;}

template<typename T> T Norm2(HostVector<T> a){return sqrt(SumSquared(a));}
