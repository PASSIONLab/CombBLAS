#pragma once

#include "DeviceMatrix/DeviceVector.h"
#include "DeviceMatrix/DeviceCube.h"
#include "HostMatrix/CFourD.h"
#include "HostMatrix/Verify.h"

template<typename T>
class DeviceFourD{
	TrackedObject memBlock;
	CFourD<T> fourD;

	void Init(int dimX, int dimY,int dimZ, int dimT){
		DeviceVector<T> tmp((int64)dimX*(int64)dimY*(int64)dimZ*(int64)dimT);
		memBlock=tmp.GetMemBlock();
		fourD=CFourD<T>(tmp.Pointer(),dimX,dimY,dimZ,dimT);
	}

public:
	typedef T Element;
	typedef DeviceVector<T> Vector;
	void operator=(const DeviceFourD& rhs){memBlock=rhs.memBlock;fourD=rhs.fourD;}
	DeviceFourD(){}
	DeviceFourD(DeviceVector<T> data, Int4 size):memBlock(data.GetMemBlock()),fourD(data.Data(),size){
		Verify(data.IsSimple() && size.x*size.y*size.z*size.w==data.Length(),FileAndLine);
	}
	DeviceFourD(Int3 size, int dimT){Init(size.x,size.y,size.z,dimT);}
	DeviceFourD(int dimX, int dimY,int dimZ, int dimT){Init(dimX,dimY,dimZ,dimT);}
	DeviceFourD(Int4 size){Init(size.x,size.y,size.z,size.w);}
	DeviceFourD(const DeviceFourD& rhs):memBlock(rhs.memBlock),fourD(rhs.fourD){}
	DeviceFourD(CFourD<T> fourD, TrackedObject memBlock):memBlock(memBlock),fourD(fourD){}
	int DimX()const{return fourD.DimX();}
	int DimY()const{return fourD.DimY();}
	int DimZ()const{return fourD.DimZ();}
	int DimT()const{return fourD.DimT();}
	Int4 Size()const{Int4 s;s.x=DimX();s.y=DimY();s.z=DimZ();s.w=DimT();return s;}
	bool IsSimple()const{return fourD.IsSimple();}
	DeviceVector<T> GetSimple(){
		if(!IsSimple())
			throw std::runtime_error("!Simple");
		return DeviceVector<T>(memBlock,fourD.GetSimple());
	}
	CFourD<T> GetC(){return fourD;}
	T* Pointer(){return fourD.Data();}
	T* Data(){return fourD.Data();}
	DeviceCube<T> CubeT(int t){return DeviceCube<T>(memBlock,fourD.CubeT(t));}
	DeviceFourD<T> SubFourD(int startT, int dimT){
		if(startT<0||startT+dimT>fourD.DimT())
			throw std::runtime_error("HostFourD<T>::Sub() out of bounds");
		return DeviceFourD<T>(fourD.SubFourD(startT,dimT),memBlock);
	}	
};