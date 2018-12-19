#pragma once

#include "HostMatrix/VectorTypes.h"
#include "HostMatrix/CMatrix.h"
#include "HostMatrix/devicehost.h"

template<typename T>
class CCube{
	T* data;
	int dimX;
	int dimY;
	int dimZ;
	int rowStride;//in elements
	int sliceStride;//in elements, should be a multiple of rowStride

public:

	int64 __device__ __host__ Index(int x, int y, int z)const{return z*int64(sliceStride)+y*rowStride+x;}
	int64 __device__ __host__ Index(Int3 pos)const{return pos.z*int64(sliceStride)+pos.y*rowStride+pos.x;}
	__device__ __host__ CCube():data(0),dimX(0),dimY(0),dimZ(0),rowStride(0),sliceStride(0){}
	//__device__ __host__ CCube(const CCube& other):data(other.data),dimX(other.dimX),dimY(other.dimY),dimZ(other.dimZ),rowStride(other.rowStride),sliceStride(other.sliceStride){}
	__device__ __host__ CCube(T* data, int dimX, int dimY, int dimZ):data(data),dimX(dimX),dimY(dimY),dimZ(dimZ),rowStride(dimX),sliceStride(dimX*(int64)dimY){}
	__device__ __host__ CCube(T* data, Int3 size):data(data),dimX(size.x),dimY(size.y),dimZ(size.z),rowStride(dimX),sliceStride(dimX*(int64)dimY){}
	__device__ __host__ CCube(T* data, int dimX, int dimY, int dimZ, int rowStride, int sliceStride):data(data),dimX(dimX),dimY(dimY),dimZ(dimZ),rowStride(rowStride),sliceStride(sliceStride){}	
	__device__ __host__ T& operator()(int x, int y, int z){return data[Index(x,y,z)];}
	__device__ __host__ const T& operator()(int x, int y, int z)const{return data[Index(x,y,z)];}
	__device__ __host__ T& operator()(Int3 pos){return data[Index(pos.x,pos.y,pos.z)];}
	__device__ __host__ const T& operator()(Int3 pos)const{return data[Index(pos.x,pos.y,pos.z)];}
	//The cube is simple if the voxels are continuous in memory
	__device__ __host__ bool IsSimple()const{return dimY==1&&dimZ==1 || dimZ==1&&rowStride==dimX || rowStride==dimX&&sliceStride==dimX*dimY;}
	__device__ __host__ CVector<T> GetSimple(){return CVector<T>(data,int64(dimX)*int64(dimY)*int64(dimZ));}
	__device__ __host__ T* Data(){return data;}
	__device__ __host__ const T* Data()const{return data;}
	__device__ __host__ int DimX() const {return dimX;}
	__device__ __host__ int DimY() const {return dimY;}
	__device__ __host__ int DimZ() const {return dimZ;}
	__device__ __host__ Int3 Size() const {Int3 size;size.x=dimX;size.y=dimY;size.z=dimZ;return size;}
	__device__ __host__ int RowStride() const {return rowStride;}
	__device__ __host__ int SliceStride() const {return sliceStride;}
	__device__ __host__ T* RowPointerX(int y, int z){return data+y*(int64)rowStride+z*int64(sliceStride);}
	__device__ __host__ T* RowPointerY(int x, int z){return data+x+z*int64(sliceStride);}
	__device__ __host__ T* RowPointerZ(int x, int y){return data+x+y*rowStride;}
	__device__ __host__ CVector<T> RowX(int y, int z){return CVector<T>(RowPointerX(y,z),dimX,1);}
	__device__ __host__ CVector<T> RowY(int x, int z){return CVector<T>(RowPointerY(x,z),dimY,rowStride);}
	__device__ __host__ CVector<T> RowZ(int x, int y){return CVector<T>(RowPointerZ(x,y),dimZ,sliceStride);}
	__device__ __host__ CMatrix<T> SliceY(int y){return CMatrix<T>(data+y*rowStride,dimX,dimZ,sliceStride);}
	__device__ __host__ CMatrix<T> SliceZ(int z){return CMatrix<T>(data+(int64)z*(int64)sliceStride,dimX,dimY,rowStride);}
	__device__ __host__ CCube<T> SubCube(int startX, int startY, int startZ, int sizeX, int sizeY, int sizeZ){return CCube<T>(data+startZ*int64(sliceStride)+startY*(int64)rowStride+startX,sizeX,sizeY,sizeZ,rowStride,sliceStride);}
	__device__ __host__ CCube<T> SubCube(Int3 start, Int3 size){return CCube<T>(data+(int64)start.z*int64(sliceStride)+(int64)start.y*(int64)rowStride+start.x,size.x,size.y,size.z,rowStride,sliceStride);}
	__device__ __host__ CCube<T> SubCube(int startZ, int sizeZ){return CCube<T>(data+(int64)startZ*int64(sliceStride),dimX,dimY,sizeZ,rowStride,sliceStride);}
};