#pragma once

#include "DeviceMatrix/DeviceVector.h"
#include "DeviceMatrix/DeviceMatrix.h"
#include "DeviceMatrix/DeviceMemBlock.h"
#include "HostMatrix/CCube.h"

template<typename T>
class DeviceCube{
public:
	TrackedObject memBlock;
	CCube<T> cube;
	void Init(int dimX, int dimY,int dimZ){
		DeviceVector<T> tmp((int64)dimX*(int64)dimY*(int64)dimZ);
		memBlock=tmp.GetMemBlock();
		cube=CCube<T>(tmp.Pointer(),dimX,dimY,dimZ);
	}
public:
	typedef T Element;
	typedef DeviceVector<T> Vector;
	void operator=(const DeviceCube& rhs){memBlock=rhs.memBlock;cube=rhs.cube;}
	DeviceCube(){}
	DeviceCube(DeviceVector<T> data, Int3 size):memBlock(data.GetMemBlock()),cube(data.Data(),size.x,size.y,size.z){
		Verify(data.IsSimple() && (int64)size.x*(int64)size.y*(int64)size.z==data.Length(),FileAndLine);
	}
	DeviceCube(int dimX, int dimY,int dimZ){Init(dimX,dimY,dimZ);}
	DeviceCube(Int3 size){Init(size.x,size.y,size.z);}
	DeviceCube(const DeviceCube& rhs):cube(rhs.cube),memBlock(rhs.memBlock){}
	DeviceCube(TrackedObject memBlock,CCube<T> cube):memBlock(memBlock),cube(cube){}
	int64 Index(int x,int y,int z)const{return cube.Index(x,y,z);}
	int DimX()const{return cube.DimX();}
	int DimY()const{return cube.DimY();}
	int DimZ()const{return cube.DimZ();}
	Int3 Size()const{return cube.Size();}
	int RowStride() const {return cube.RowStride();}
	int SliceStride() const {return cube.SliceStride();}
	bool IsSimple()const{return cube.IsSimple();}
	DeviceVector<T> GetSimple(){
		if(!IsSimple())
			throw std::runtime_error("!Simple");
		return DeviceVector<T>(memBlock,cube.GetSimple());
	}
	void Clear(){*this=DeviceCube<T>();}
	CCube<T> GetC(){return cube;}
	T* Pointer(){return cube.Data();}
	T* Data(){return cube.Data();}
	DeviceCube<T> SubCube(int startZ, int dimZ){return DeviceCube<T>(memBlock,cube.SubCube(0, 0, startZ,cube.DimX(),cube.DimY(),dimZ));}
	DeviceCube<T> SubCube(int startX, int startY, int startZ, int dimX, int dimY,int dimZ){return DeviceCube<T>(memBlock,cube.SubCube(startX, startY, startZ,dimX,dimY,dimZ));}
	DeviceCube<T> SubCube(Int3 start, Int3 size){return DeviceCube<T>(memBlock,cube.SubCube(start,size));}
	DeviceMatrix<T> SliceZ(int z){return DeviceMatrix<T>(cube.SliceZ(z),memBlock);}
	DeviceVector<T> RowX(int y, int z){return DeviceVector<T>(memBlock,cube.RowX(y,z));}
	DeviceVector<T> RowY(int x, int z){return DeviceVector<T>(memBlock,cube.RowY(x,z));}
	DeviceVector<T> RowZ(int x, int y){return DeviceVector<T>(memBlock,cube.RowZ(x,y));}
	void Set(Int3 pos, T value){cudaMemcpy(cube.Data()+cube.Index(pos),&value,sizeof(T),cudaMemcpyHostToDevice);}
};