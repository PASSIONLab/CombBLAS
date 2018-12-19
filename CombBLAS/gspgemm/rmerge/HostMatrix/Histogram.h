#pragma once
#include "HostMatrix/HostFourD.h"
#include <algorithm>

namespace ExMI
{

//fast version
template <typename DST, typename T>
void Histogram(DST* dst, int dstLength, T* src, int64 srcLength, double min, double max){
	Verify(max > min, "Maximum has to exceed minimum for histogram creation");
	Verify(dstLength > 0, "Number of bins has to be greater 0 for histogram creation");
	double inv_step = double(dstLength)/(max-min);
	int n=dstLength;
	for(int i=0;i<n;i++)
		dst[i]=0;
	for(int64 i=0;i<srcLength;i++){
		int bin = int((double(src[i])-min)*inv_step);
		bin = std::max<int>(0, std::min<int>(bin, n-1));
		dst[bin] += DST(1);		
	}
}

template <typename DST, typename T>
void Histogram(HostVector<DST>& dst, HostVector<T> src, double min, double max){
	Verify(dst.IsSimple(),"dfkuhdlfkh");	
	if(!src.IsSimple()){
		HostVector<T> tmp(src.Length());
		#pragma omp parallel for
		for(int64 i=0;i<src.Length();i++)
			tmp[i]=src[i];
		src=tmp;
		
	}
	Histogram(dst.Pointer(),dst.Length32(),src.Pointer(),src.Length(),min,max);
}

template <typename DST, typename T>
void Histogram(HostVector<DST>& dst, HostMatrix<T> img, double min, double max){
	if(dst.IsSimple() && img.IsSimple()){
		Histogram(dst.Pointer(),dst.Length32(),img.Pointer(),img.DimX()*img.DimY(),min,max);
		return;
	}
	Verify(max > min, "Maximum has to exceed minimum for histogram creation");
	Verify(dst.Length() > 0, "Number of bins has to be greater 0 for histogram creation");
	int n=dst.Length32();
	double inv_step = double(n) / (max-min);
	ComponentWiseInit(dst, 0);
	for(int y = 0; y < img.DimY(); ++y){
		for(int x = 0; x < img.DimX(); ++x){
			int bin = int((double(img(x,y))-min) * inv_step);
			bin = std::max<int>(0, std::min<int>(bin, n-1));
			dst[bin] += DST(1);
		}
	}
}

template <typename DST, typename T>
void Histogram(HostVector<DST>& dst, HostCube<T> img, double min, double max){
	HostMatrix<DST> histos(dst.Length32(), img.DimZ());
	#pragma omp parallel for
	for(int z = 0; z < img.DimZ(); ++z)
		Histogram(histos.Row(z), img.SliceZ(z), min, max);

	#pragma omp parallel for
	for(int x = 0; x < histos.DimX(); ++x)
		dst[x] = Sum(histos.Column(x));
	
}

template <typename DST, typename T>
void Histogram(HostVector<DST>& dst, HostFourD<T> img, double min, double max){
	if(img.DimT()==1)
		Histogram(dst,img.CubeT(0),min,max);
	else
	{
		HostMatrix<DST> histos(dst.Length32(), img.DimT());
		#pragma omp parallel for
		for(int t = 0; t < img.DimT(); ++t)
			Histogram(histos.Row(t), img.CubeT(t), min, max);
		#pragma omp parallel for
		for(int x = 0; x < histos.DimX(); ++x)
			dst[x] = Sum(histos.Column(x));	
	}
}

}