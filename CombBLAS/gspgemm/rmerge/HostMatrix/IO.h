#pragma once

#include "HostMatrix/HostFourD.h"

template<typename OStream, typename T>
static OStream& operator << (OStream& o, HostVector<T> x){
	for(int i=0;i<x.Length();i++){
		T t=x[i];
		o<<t;
		if(i<x.Length()-1)
			o<<" ";
	}
	return o;
}


template<typename OStream, typename T>
OStream& operator << (OStream& o, HostMatrix<T> m){
	for(int y=0;y<m.DimY();y++)
		o<<m.Row(y)<<"\n";
	return o;
}

template<typename OStream, typename T>
OStream& operator << (OStream& o, HostCube<T> a){
	for(int z=0;z<a.DimZ();z++)
		o<<"Slice "<<z<<"\n"<<a.SliceZ(z)<<"\n";
	return o;
}

template<typename OStream, typename T>
OStream& operator << (OStream& o, HostFourD<T> a){
	for(int t=0;t<a.DimT();t++)
		o<<"Cube "<<t<<"\n"<<a.CubeT(t)<<"\n";
	return o;
}
