#pragma once

#include "HostMatrix/HostVector.h"
#include "HostMatrix/HostMatrix.h"
#include "HostMatrix/HostCube.h"
#include "HostMatrix/HostFourD.h"
#include <omp.h>

template<typename DST, typename ElementFunctor>
void ComponentWiseInline(DST* dst, int64 length, ElementFunctor functor){
	#pragma omp parallel for
	for(int64 i=0;i<length;i++)
		functor(dst[i],dst[i]);			
}

template<typename DST, typename ElementFunctor>
void ComponentWiseInline(DST* dst, int64 length, int stride, ElementFunctor functor){
	#pragma omp parallel for
	for(int64 i=0;i<length;i++)
		functor(dst[i*stride],dst[i*stride]);
}
template<typename DST, typename ElementFunctor>
void ComponentWiseInline(HostVector<DST> dst, ElementFunctor functor){
	if(dst.Stride()==1)
		ComponentWiseInline(dst.Data(),dst.Length(),functor);
	else
		ComponentWiseInline(dst.Data(),dst.Length(),dst.Stride(),functor);
}
template<typename T, typename EF> void ComponentWiseInline(HostMatrix<T> dst, EF functor){
	if(dst.IsSimple())
		ComponentWiseInline(dst.GetSimple(),functor);
	else
		for(int y=0;y<dst.DimY();y++)
			ComponentWiseInline(dst.Row(y),functor);
}
template<typename T, typename EF> void ComponentWiseInline(HostCube<T> dst, EF functor){
	if(dst.IsSimple())
		ComponentWiseInline(dst.GetSimple(),functor);
	else
		for(int z=0;z<dst.DimZ();z++)
			ComponentWiseInline(dst.SliceZ(z),functor);
}
template<typename T, typename EF> void ComponentWiseInline(HostFourD<T> dst, EF functor){ComponentWiseInline(dst.GetSimple(),functor);}

template<typename DST, typename SRC, typename ElementFunctor>
void ComponentWise(HostVector<DST> dst, HostVector<SRC> src, ElementFunctor functor){
	Verify(dst.Length()==src.Length(),"Size mismatch. 23898f12122");
	DST* pDst=dst.Data();
	SRC* pSrc=src.Data();
	int dstStride=dst.Stride();
	int srcStride=src.Stride();
	int64 n=dst.Length();

	if(dst.Stride()==1 && src.Stride()==1){
		#pragma omp parallel for
		for(int64 i=0;i<n;i++)
			functor(pDst[i],pSrc[i]);
	}
	else{
		#pragma omp parallel for
		for(int64 i=0;i<n;i++)			
			functor(pDst[i*dstStride],pSrc[i*srcStride]);
	}
}

template<typename DST,typename SRC,typename EF> void ComponentWise(HostMatrix<DST> dst, HostMatrix<SRC> src, EF f){
	Verify(dst.Size()==src.Size(),"tf3c6947536");
	if(dst.IsSimple()&&src.IsSimple())
		ComponentWise(dst.GetSimple(),src.GetSimple(),f);
	else{
		#pragma omp parallel for
		for(int y=0;y<dst.DimY();y++)
			ComponentWise(dst.Row(y),src.Row(y),f);
	}
}
template<typename DST,typename SRC,typename EF> void ComponentWise(HostCube<DST> dst, HostCube<SRC> src, EF f){
	Verify(dst.Size()==src.Size(),"a0w6nvc26t");
	if(dst.IsSimple()&&src.IsSimple())
		ComponentWise(dst.GetSimple(),src.GetSimple(),f);
	else{
		#pragma omp parallel for
		for(int z=0;z<dst.DimZ();z++)
			ComponentWise(dst.SliceZ(z),src.SliceZ(z),f);
	}
}
template<typename DST,typename SRC,typename EF> void ComponentWise(HostFourD<DST> dst, HostFourD<SRC> src, EF f){
	Verify(dst.Size()==src.Size(),"5v6b396t9386");
	if(dst.IsSimple()&&src.IsSimple())
		ComponentWise(dst.GetSimple(),src.GetSimple(),f);
	else{
		#pragma omp parallel for
		for(int t=0;t<dst.DimT();t++)
			ComponentWise(dst.CubeT(t),src.CubeT(t),f);
	}
}

template<typename DST, typename SRC, typename ElementFunctor>
void ComponentWiseAddUp(HostVector<DST> dst, HostVector<SRC> src, ElementFunctor functor){
	Verify(dst.Length()==src.Length(),"Size mismatch. 98w09au0u");
	DST* pDst=dst.Data();
	SRC* pSrc=src.Data();
	int dstStride=dst.Stride();
	int srcStride=src.Stride();
	int64 n=dst.Length();

	if(dst.Stride()==1 && src.Stride()==1){
		#pragma omp parallel for
		for(int64 i=0;i<n;i++){
			DST tmp=pDst[i];
			functor(tmp,pSrc[i]);
			pDst[i]+=tmp;
		}
	}
	else{
		#pragma omp parallel for
		for(int64 i=0;i<n;i++){
			DST tmp=pDst[i*dstStride];
			functor(tmp,pSrc[i*srcStride]);
			pDst[i*dstStride]+=tmp;
		}
	}
}

template<typename DST,typename SRC,typename EF> void ComponentWiseAddUp(HostMatrix<DST> dst, HostMatrix<SRC> src, EF f){ComponentWiseAddUp(dst.GetSimple(),src.GetSimple(),f);}
template<typename DST,typename SRC,typename EF> void ComponentWiseAddUp(HostCube<DST> dst, HostCube<SRC> src, EF f){ComponentWiseAddUp(dst.GetSimple(),src.GetSimple(),f);}
template<typename DST,typename SRC,typename EF> void ComponentWiseAddUp(HostFourD<DST> dst, HostFourD<SRC> src, EF f){ComponentWiseAddUp(dst.GetSimple(),src.GetSimple(),f);}

template<typename T,typename EF> HostVector<T> ComponentWise(HostVector<T> x,EF f){HostVector<T> y(x.Size());ComponentWise(y.GetSimple(),x.GetSimple(),f);return y;}
template<typename T,typename EF> HostMatrix<T> ComponentWise(HostMatrix<T> x,EF f){HostMatrix<T> y(x.Size());ComponentWise(y.GetSimple(),x.GetSimple(),f);return y;}
template<typename T,typename EF> HostCube<T> ComponentWise(HostCube<T> x,EF f){HostCube<T> y(x.Size());ComponentWise(y.GetSimple(),x.GetSimple(),f);return y;}
template<typename T,typename EF> HostFourD<T> ComponentWise(HostFourD<T> x,EF f){HostFourD<T> y(x.Size());ComponentWise(y.GetSimple(),x.GetSimple(),f);return y;}

template<typename DST, typename A, typename B, typename ElementFunctor>
void BinaryComponentWise(HostVector<DST> dst, HostVector<A> a, HostVector<B> b, ElementFunctor functor){
	Verify(dst.Length()==a.Length() && a.Length()==b.Length(),"Size mismatch. mv287098275");

	DST* pDst=dst.Data();
	A* pA=a.Data();
	B* pB=b.Data();
	int dstStride=dst.Stride();
	int aStride=a.Stride();
	int bStride=b.Stride();
	int64 n=dst.Length();

	if(dst.Stride()==1 && a.Stride()==1 && b.Stride()==1){
		#pragma omp parallel for
		for(int64 i=0;i<n;i++)
			functor(pDst[i],pA[i],pB[i]);
	}
	else{
		#pragma omp parallel for
		for(int64 i=0;i<n;i++)
			functor(pDst[i*dstStride],pA[i*aStride],pB[i*bStride]);
	}
}

template<typename DST, typename A, typename B, typename EF>
void BinaryComponentWise(HostMatrix<DST> dst, HostMatrix<A> a, HostMatrix<B> b, EF functor){BinaryComponentWise(dst.GetSimple(),a.GetSimple(),b.GetSimple(),functor);}
template<typename DST, typename A, typename B, typename EF>
void BinaryComponentWise(HostCube<DST> dst, HostCube<A> a, HostCube<B> b, EF functor){BinaryComponentWise(dst.GetSimple(),a.GetSimple(),b.GetSimple(),functor);}
template<typename DST, typename A, typename B, typename EF>
void BinaryComponentWise(HostFourD<DST> dst, HostFourD<A> a, HostFourD<B> b, EF functor){BinaryComponentWise(dst.GetSimple(),a.GetSimple(),b.GetSimple(),functor);}


template<typename DST, typename A, typename B, typename ElementFunctor>
void BinaryComponentWiseAddUp(HostVector<DST> dst, HostVector<A> a, HostVector<B> b, ElementFunctor functor){
	Verify(dst.Length()==a.Length() && a.Length()==b.Length(),FileAndLine);
	DST* pDst=dst.Data();
	A* pA=a.Data();
	B* pB=b.Data();
	int dstStride=dst.Stride();
	int aStride=a.Stride();
	int bStride=b.Stride();
	int64 n=dst.Length();
	if(dst.Stride()==1 && a.Stride()==1 && b.Stride()==1){
		#pragma omp parallel for
		for(int64 i=0;i<n;i++){
			DST tmp=pDst[i];
			functor(tmp,pA[i],pB[i]);
			pDst[i]+=tmp;
		}
	}
	else{
		#pragma omp parallel for
		for(int64 i=0;i<n;i++){
			DST tmp=pDst[i*dstStride];
			functor(tmp,pA[i*aStride],pB[i*bStride]);
			pDst[i*dstStride]+=tmp;
		}
	}
}

template<typename DST, typename A, typename B, typename EF>
void BinaryComponentWiseAddUp(HostMatrix<DST> dst, HostMatrix<A> a, HostMatrix<B> b, EF functor){BinaryComponentWiseAddUp(dst.GetSimple(),a.GetSimple(),b.GetSimple(),functor);}
template<typename DST, typename A, typename B, typename EF>
void BinaryComponentWiseAddUp(HostCube<DST> dst, HostCube<A> a, HostCube<B> b, EF functor){BinaryComponentWiseAddUp(dst.GetSimple(),a.GetSimple(),b.GetSimple(),functor);}
template<typename DST, typename A, typename B, typename EF>
void BinaryComponentWiseAddUp(HostFourD<DST> dst, HostFourD<A> a, HostFourD<B> b, EF functor){BinaryComponentWiseAddUp(dst.GetSimple(),a.GetSimple(),b.GetSimple(),functor);}
