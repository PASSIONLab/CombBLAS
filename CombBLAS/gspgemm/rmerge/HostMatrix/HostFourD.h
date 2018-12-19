#pragma once

#include "HostMatrix/CFourD.h"
#include "HostMatrix/HostCube.h"

template<typename T>
class HostFourD{
public:	
	CFourD<T> fourD;
	TrackedObject memBlock;
public:
	typedef T Element;
	typedef HostVector<T> Vector;
	void operator=(const HostFourD& rhs){memBlock=rhs.memBlock;fourD=rhs.fourD;}
	HostFourD(){}
	HostFourD(int dimX, int dimY,int dimZ, int dimT):fourD(new T[dimX*(int64)dimY*(int64)dimZ*(int64)dimT],dimX,dimY,dimZ,dimT),memBlock(new HostMemBlock<T>(fourD.Data())){}
	explicit HostFourD(Int4 size):fourD(new T[size.x*(int64)size.y*(int64)size.z*(int64)size.w],size.x,size.y,size.z,size.w),memBlock(new HostMemBlock<T>(fourD.Data())){}
	explicit HostFourD(Int3 size, int t):fourD(new T[size.x*(int64)size.y*(int64)size.z*(int64)t],size.x,size.y,size.z,t),memBlock(new HostMemBlock<T>(fourD.Data())){}
	explicit HostFourD(HostCube<T> cube):fourD(cube.GetC()),memBlock(cube.MemBlock()){}
	HostFourD(const HostFourD& rhs):memBlock(rhs.memBlock),fourD(rhs.fourD){}
	HostFourD(CFourD<T> fourD, TrackedObject memBlock):fourD(fourD),memBlock(memBlock){}		
	HostFourD(HostVector<T> data, Int4 size):memBlock(data.GetMemBlock()),fourD(data.Data(),size){
		Verify(data.IsSimple() && (int64)size.x*(int64)size.y*(int64)size.z*(int64)size.w==data.Length(),"HostFourD: Size mismatch.");
	}

	T& operator()(int x, int y, int z, int t){return fourD(x,y,z,t);}	
	T& operator()(Int4 pos){return fourD(pos);}	
	int DimX()const{return fourD.DimX();}
	int DimY()const{return fourD.DimY();}
	int DimZ()const{return fourD.DimZ();}
	int DimT()const{return fourD.DimT();}
	Int4 Size()const{return fourD.Size();}
	HostCube<T> CubeT(int t){return HostCube<T>(fourD.CubeT(t),memBlock);}
	HostVector<T> RowT(int x, int y, int z){return HostVector<T>(fourD.RowT(x,y,z),memBlock);}
	CFourD<T> GetC(){return fourD;}
	bool IsSimple()const{return fourD.IsSimple();}
	HostVector<T> GetSimple(){
		Verify(IsSimple(),"FourD not Simple");
		return HostVector<T>(fourD.GetSimple(),memBlock);
	}
	T* Pointer(){return fourD.Data();}

	HostFourD<T> SubFourD(Int3 start, Int3 size){
		Verify(start.x>=0&&start.y>=0&&start.z>=0, "Out of bounds. 3n02gskchgkjw45o87");
		Verify(start.x+size.x<=fourD.DimX()&&start.y+size.y<=fourD.DimY()&&start.z+size.z<=fourD.DimZ(),"Out of bounds. 3c67tn30302");
		return HostFourD<T>(fourD.SubFourD(start,size),memBlock);
	}

	HostFourD<T> SubFourD(int startT, int dimT){
		if(startT<0||startT+dimT>fourD.DimT())
			throw std::runtime_error("HostFourD<T>::Sub() out of bounds");
		return HostFourD<T>(fourD.SubFourD(startT,dimT),memBlock);
	}	
};
