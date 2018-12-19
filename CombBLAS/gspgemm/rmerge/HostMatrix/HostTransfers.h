#pragma once

#include "HostMatrix/HostVector.h"
#include "HostMatrix/HostMatrix.h"
#include "HostMatrix/HostCube.h"
#include "HostMatrix/HostFourD.h"
#include "HostMatrix/VectorOperators.h"

#include <exception>
#include <memory.h>
#include <omp.h>


namespace Hidden34wd672{//This namespace is not meant to be used directly
	//Copy simple requires that both are simple
template<typename T> 
static void CopySimple(HostVector<T> src, HostVector<T> dst){	
	T* pSrc=src.Data();
	T* pDst=dst.Data();
	#pragma omp parallel for
	for(int64 i=0;i<dst.Length();i++)
		pDst[i]=pSrc[i];
}

//template specialization for special types
static void CopySimple(HostVector<char> src, HostVector<char> dst){memcpy(dst.Data(),src.Data(),dst.Length32()*sizeof(char));}
static void CopySimple(HostVector<uchar> src, HostVector<uchar> dst){memcpy(dst.Data(),src.Data(),dst.Length32()*sizeof(uchar));}
static void CopySimple(HostVector<short> src, HostVector<short> dst){memcpy(dst.Data(),src.Data(),dst.Length32()*sizeof(short));}
static void CopySimple(HostVector<ushort> src, HostVector<ushort> dst){memcpy(dst.Data(),src.Data(),dst.Length32()*sizeof(ushort));}
static void CopySimple(HostVector<int> src, HostVector<int> dst){memcpy(dst.Data(),src.Data(),dst.Length32()*sizeof(int));}
static void CopySimple(HostVector<uint> src, HostVector<uint> dst){memcpy(dst.Data(),src.Data(),dst.Length32()*sizeof(uint));}
static void CopySimple(HostVector<float> src, HostVector<float> dst){memcpy(dst.Data(),src.Data(),dst.Length32()*sizeof(float));}
static void CopySimple(HostVector<double> src, HostVector<double> dst){memcpy(dst.Data(),src.Data(),dst.Length32()*sizeof(double));}
}

template<typename T> 
static void Copy(HostVector<T> src, HostVector<T> dst){
	if(src.IsSimple() && dst.IsSimple()){
		Hidden34wd672::CopySimple(src,dst);
		return;
	}
	Verify(src.Size()==dst.Size(),"Size mismatch. 4buuu5u");
	#pragma omp parallel for
	for(int64 i=0;i<src.Length();i++)
		dst[i]=src[i];
}

template<typename T> 
static void Copy(HostMatrix<T> src, HostMatrix<T> dst){
	Verify(src.Size()==dst.Size(),"Size mismatch. vb4u6v6u6v");
	if(src.IsSimple() && dst.IsSimple()){
		Hidden34wd672::CopySimple(src.GetSimple(),dst.GetSimple());
		return;
	}
	#pragma omp parallel for
	for(int y=0;y<src.DimY();y++)
		for(int x=0;x<src.DimX();x++)
			dst(x,y)=src(x,y);

}
template<typename T> 
static void Copy(HostCube<T> src, HostCube<T> dst){
	Verify(src.Size()==dst.Size(),"Size mismatch. nv037t037");
	if(src.IsSimple() && dst.IsSimple()){
		Hidden34wd672::CopySimple(src.GetSimple(),dst.GetSimple());
		return;
	}
	#pragma omp parallel for
	for(int z=0;z<src.DimZ();z++)
		for(int y=0;y<src.DimY();y++)
			for(int x=0;x<src.DimX();x++)
				dst(x,y,z)=src(x,y,z);
}

template<typename T> 
static void Copy(HostFourD<T> src, HostFourD<T> dst){
	Verify(src.Size()==dst.Size(),"Size mismatch. vmv08326");
	if(src.IsSimple() && dst.IsSimple()){
		Hidden34wd672::CopySimple(src.GetSimple(),dst.GetSimple());
		return;
	}
	#pragma omp parallel for
	for(int t=0;t<src.DimT();t++)
		Copy(src.CubeT(t),dst.CubeT(t));
		
}

template<typename T> HostVector<T> Clone(HostVector<T> x){HostVector<T> y(x.Size());Copy(x,y);return y;}
template<typename T> HostMatrix<T> Clone(HostMatrix<T> x){HostMatrix<T> y(x.Size());Copy(x,y);return y;}
template<typename T> HostCube<T> Clone(HostCube<T> x){HostCube<T> y(x.Size());Copy(x,y);return y;}
template<typename T> HostFourD<T> Clone(HostFourD<T> x){HostFourD<T> y(x.Size());Copy(x,y);return y;}