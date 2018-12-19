#pragma once

#include "HostMatrix/TrackedObject.h"
#include "HostMatrix/CCube.h"
#include "HostMatrix/HostMatrix.h"
#include "HostMatrix/HostVector.h"
#include "HostMatrix/HostMemBlock.h"

template<typename T>
class HostCube{
public:
	CCube<T> cube;
	TrackedObject memBlock;
public:
	typedef T Element;
	typedef HostVector<T> Vector;
	void operator=(const HostCube& rhs){memBlock=rhs.memBlock;cube=rhs.cube;}
	HostCube(){}
	HostCube(HostVector<T> data, Int3 size):memBlock(data.GetMemBlock()),cube(data.Data(),size.x,size.y,size.z){
		Verify(data.IsSimple() && (int64)size.x*(int64)size.y*(int64)size.z==data.Length(),"HostCube: Size mismatch.");
	}
	HostCube(int dimX, int dimY,int dimZ):cube(new T[dimX*(int64)dimY*(int64)dimZ],dimX,dimY,dimZ),memBlock(new HostMemBlock<T>(cube.Data())){}
	explicit HostCube(Int3 size):cube(new T[size.x*(int64)size.y*(int64)size.z],size.x,size.y,size.z),memBlock(new HostMemBlock<T>(cube.Data())){}
	HostCube(const HostCube& rhs):cube(rhs.cube),memBlock(rhs.memBlock){}
	HostCube(CCube<T> cube, TrackedObject memBlock):cube(cube),memBlock(memBlock){}
	void Clear(){*this=HostCube<T>();}
	TrackedObject MemBlock(){return memBlock;}
	int64 Index(int x,int y,int z)const{return cube.Index(x,y,z);}
	int64 Index(Int3 pos)const{return cube.Index(pos);}
	const T& operator()(int x, int y, int z)const{return cube(x,y,z);}	
	const T& operator()(Int3 pos)const{return cube(pos);}
	T& operator()(int x, int y, int z){return cube(x,y,z);}	
	T& operator()(Int3 pos){return cube(pos);}
	int DimX()const{return cube.DimX();}
	int DimY()const{return cube.DimY();}
	int DimZ()const{return cube.DimZ();}
	int RowStride() const {return cube.RowStride();}
	int SliceStride() const {return cube.SliceStride();}
	bool IsEmpty()const{return DimX()==0 && DimY()==0 && DimZ()==0;}
	Int3 Size()const{return cube.Size();}
	HostMatrix<T> SliceY(int y){return HostMatrix<T>(cube.SliceY(y),memBlock);}
	HostMatrix<T> SliceZ(int z){return HostMatrix<T>(cube.SliceZ(z),memBlock);}
	T* RowPointerX(int y, int z){return cube.RowPointerX(y,z);}
	T* RowPointerY(int x, int z){return cube.RowPointerY(x,z);}
	T* RowPointerZ(int x, int y){return cube.RowPointerZ(x,y);}
	HostVector<T> RowX(int y, int z){return HostVector<T>(cube.RowX(y,z),memBlock);}
	HostVector<T> RowY(int x, int z){return HostVector<T>(cube.RowY(x,z),memBlock);}
	HostVector<T> RowZ(int x, int y){return HostVector<T>(cube.RowZ(x,y),memBlock);}
	CCube<T> GetC(){return cube;}
	bool IsSimple()const{return cube.IsSimple();}
	HostVector<T> GetSimple(){Verify(IsSimple(),"c2870t27t087t");return HostVector<T>(cube.GetSimple(),memBlock);}
	T* Pointer(){return cube.Data();}
	HostCube<T> SubCube(int startX, int startY, int startZ, int dimX, int dimY, int dimZ){
		Verify(startX>=0 && startY>=0 && startZ>=0 && startX+dimX<=DimX() && startY+dimY<=DimY()&& startZ+dimZ<=DimZ(),"Out of bounds. cw06nt9b6t986t");
		return HostCube<T>(cube.SubCube(startX,startY,startZ,dimX,dimY,dimZ),memBlock);
	}
	HostCube<T> SubCube(Int3 start, Int3 size){
		Verify(start.x>=0 && start.y>=0 && start.z>=0 && start.x+size.x<=DimX() && start.y+size.y<=DimY()&& start.z+size.z<=DimZ(),"Out of bounds. 23nfkj3kjh");
		return SubCube(start.x,start.y,start.z,size.x,size.y,size.z);
	}
	HostCube<T> SubCube(int startZ, int sizeZ){
		Verify(startZ>=0 && startZ+sizeZ<=DimZ(),"20909ds89");
		return HostCube<T>(cube.SubCube(startZ,sizeZ),memBlock);
	}	
};