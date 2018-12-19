#pragma once

#include "HostMatrix/VectorTypes.h"
#include "HostMatrix/devicehost.h"
#include "HostMatrix/CCube.h"

template<typename T>
class CFourD{
	T* data;
	int dimX;
	int dimY;
	int dimZ;
	int dimT;
	int rowStride;
	int sliceStride;
	int64 cubeStride;

public:
	__device__ __host__ CFourD():data(0),dimX(0),dimY(0),dimZ(0),dimT(0){}
	//__device__ __host__ CFourD(const CFourD& other):data(other.data),dimX(other.dimX),dimY(other.dimY),dimZ(other.dimZ),dimT(other.dimT),rowStride(other.rowStride),sliceStride(other.sliceStride),cubeStride(other.cubeStride){}
	__device__ __host__ CFourD(T* data, int dimX, int dimY, int dimZ, int dimT):data(data),dimX(dimX),dimY(dimY),dimZ(dimZ),dimT(dimT),rowStride(dimX),sliceStride(dimX*dimY),cubeStride(dimX*dimY*dimZ){}
	__device__ __host__ CFourD(T* data, Int4 size):data(data),dimX(size.x),dimY(size.y),dimZ(size.z),dimT(size.w),rowStride(dimX),sliceStride(dimX*dimY),cubeStride(dimX*dimY*dimZ){}
	__device__ __host__ CFourD(T* data, int dimX, int dimY, int dimZ, int dimT, int rowStride, int sliceStride, int64 cubeStride):data(data),dimX(dimX),dimY(dimY),dimZ(dimZ),dimT(dimT),rowStride(rowStride),sliceStride(sliceStride),cubeStride(cubeStride){}	
	__device__ __host__ explicit CFourD(CCube<T> cube):data(cube.Data()),dimX(cube.DimX()),dimY(cube.DimY()),dimZ(cube.DimZ()),dimT(1),rowStride(cube.RowStride()),sliceStride(cube.SliceStride()),cubeStride(sliceStride*int64(cube.DimZ())){}		
	__device__ __host__ int64 Index(int x, int y, int z, int t)const{return t*(int64)cubeStride+z*(int64)sliceStride+y*rowStride+x;}
	__device__ __host__ T& operator()(int x, int y, int z, int t){return data[Index(x,y,z,t)];}
	__device__ __host__ T& operator()(Int4 pos){return data[Index(pos.x,pos.y,pos.z,pos.w)];}
	__device__ __host__ const T& operator()(int x, int y, int z, int t)const{return data[Index(x,y,z,t)];}
	__device__ __host__ bool IsSimple()const{return rowStride==dimX && sliceStride==dimX*dimY && cubeStride==dimX*dimY*dimZ;}
	__device__ __host__ CVector<T> GetSimple(){return CVector<T>(data,int64(dimX)*int64(dimY)*int64(dimZ)*int64(dimT));}
	__device__ __host__ T* Data(){return data;}
	__device__ __host__ int DimX() const {return dimX;}
	__device__ __host__ int DimY() const {return dimY;}
	__device__ __host__ int DimZ() const {return dimZ;}
	__device__ __host__ int DimT() const {return dimT;}
	__device__ __host__ Int4 Size() const {Int4 size;size.x=dimX;size.y=dimY;size.z=dimZ;size.w=dimT;return size;}
	__device__ __host__ int RowStride() const {return rowStride;}
	__device__ __host__ int SliceStride() const {return sliceStride;}
	__device__ __host__ int CubeStride() const {return cubeStride;}
	__device__ __host__ T* RowPointerX(int y, int z, int t){return data+y*rowStride+z*sliceStride+t*cubeStride;}
	__device__ __host__ CCube<T> CubeT(int t){return CCube<T>(data+t*cubeStride,dimX,dimY,dimZ,rowStride,sliceStride);}
	__device__ __host__ T* RowPointerT(int x, int y, int z){return data+x+y*rowStride+z*sliceStride;}
	__device__ __host__ CVector<T> RowT(int x, int y, int z){return CVector<T>(data+x+y*rowStride+z*sliceStride,dimT,(int)cubeStride);}
	__device__ __host__ CFourD<T> SubFourD(int startT, int dimT){return CFourD<T>(data+startT*(int64)cubeStride,dimX,dimY,dimZ,dimT,rowStride,sliceStride,cubeStride);}
	__device__ __host__ CFourD<T> SubFourD(Int3 start, Int3 size){return CFourD<T>(data+Index(start.x,start.y,start.z,0),size.x,size.y,size.z,dimT,rowStride,sliceStride,cubeStride);} 
};