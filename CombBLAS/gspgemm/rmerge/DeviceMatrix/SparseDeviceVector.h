#pragma once

#include "DeviceMatrix/DeviceVector.h"

//Contains a value and index for each non-zero element.
//The index is a one dimensional index, also for 2D, 3D, 4D data sets

template<typename T>
class SparseDeviceVector{
public:
	DeviceVector<T> values;
	DeviceVector<unsigned int> indices;
	int64 dimX;
public:
	typedef T Element;	
	SparseDeviceVector(){}
	SparseDeviceVector(int dimX):dimX(dimX){}
	SparseDeviceVector(const SparseDeviceVector& rhs):values(rhs.values),indices(rhs.indices),dimX(dimX){}
	SparseDeviceVector(DeviceVector<T> values,DeviceVector<unsigned int> indices,int64 dimX)
		:values(values),indices(indices),dimX(dimX){}
	DeviceVector<T> Values(){return values;}
	DeviceVector<unsigned int> Indices(){return indices;}
	int64 NonZeros()const{return indices.Length();}
	int64 DimX()const{return dimX;}
	int64 Size()const{return dimX;}
	int64 Length()const{return dimX;}
};

template<typename T>
class SparseDeviceMatrix{
public:
	DeviceVector<T> values;
	DeviceVector<unsigned int> indices;
	int dimX;
	int dimY;
public:
	typedef T Element;
	typedef DeviceVector<T> Vector;
	unsigned int Index(int x, int y){return y*dimX+x;}
	SparseDeviceMatrix(){}
	SparseDeviceMatrix(int dimX, int dimY):dimX(dimX),dimY(dimY){}
	SparseDeviceMatrix(const SparseDeviceMatrix& rhs):values(rhs.values),indices(rhs.indices),dimX(dimX),dimY(dimY){}
	SparseDeviceMatrix(SparseDeviceVector<T> simple,Int2 size):values(simple.values()),indices(simple.indices()),dimX(size.x),dimY(size.y){}
	SparseDeviceMatrix(DeviceVector<T> values,DeviceVector<unsigned int> indices,int dimX,int dimY):values(values),indices(indices),dimX(dimX),dimY(dimY){}
	SparseDeviceMatrix(DeviceVector<T> values,DeviceVector<unsigned int> indices,Int2 size):values(values),indices(indices),dimX(size.x),dimY(size.y){}
	DeviceVector<T> Values(){return values;}
	DeviceVector<unsigned int> Indices(){return indices;}
	int DimX()const{return dimX;}
	int DimY()const{return dimY;}
	Int2 Size()const{Int2 size;size.x=dimX;size.y=dimY;return size;}
	SparseDeviceVector<T> GetSimple(){return SparseDeviceVector<T>(values,indices,dimX*dimY);}
};

template<typename T>
class SparseDeviceCube{
public:
	DeviceVector<T> values;
	DeviceVector<unsigned int> indices;
	int dimX;
	int dimY;
	int dimZ;
public:
	typedef T Element;
	typedef DeviceVector<T> Vector;
	unsigned int Index(int x, int y, int z){return z*dimX*dimY+y*dimX+x;}
	SparseDeviceCube(){}
	SparseDeviceCube(int dimX, int dimY,int dimZ):dimX(dimX),dimY(dimY),dimZ(dimZ){}
	SparseDeviceCube(const SparseDeviceCube& rhs):values(rhs.values),indices(rhs.indices),dimX(dimX),dimY(dimY),dimZ(dimZ){}
	SparseDeviceCube(SparseDeviceVector<T> simple,Int3 size):values(simple.values()),indices(simple.indices()),dimX(size.x),dimY(size.y),dimZ(size.z){}
	SparseDeviceCube(DeviceVector<T> values,DeviceVector<unsigned int> indices,int dimX,int dimY,int dimZ):values(values),indices(indices),dimX(dimX),dimY(dimY),dimZ(dimZ){}
	SparseDeviceCube(DeviceVector<T> values,DeviceVector<unsigned int> indices,Int3 size):values(values),indices(indices),dimX(size.x),dimY(size.y),dimZ(size.z){}
	DeviceVector<T> Values(){return values;}
	DeviceVector<unsigned int> Indices(){return indices;}
	int DimX()const{return dimX;}
	int DimY()const{return dimY;}
	int DimZ()const{return dimZ;}
	Int3 Size()const{Int3 size;size.x=dimX;size.y=dimY;size.z=dimZ;return size;}
	SparseDeviceVector<T> GetSimple(){return SparseDeviceVector<T>(values,indices,dimX*dimY*dimZ);}
};


template<typename T>
class SparseDeviceFourD{
public:
	DeviceVector<T> values;
	DeviceVector<unsigned int> indices;
	int dimX;
	int dimY;
	int dimZ;
	int dimT;
public:
	typedef T Element;
	typedef DeviceVector<T> Vector;
	unsigned int Index(int x, int y, int z, int t){return t*dimX*dimY*dimZ+z*dimX*dimY+y*dimX+x;}
	SparseDeviceFourD(){}
	SparseDeviceFourD(int dimX, int dimY, int dimZ, int dimT):dimX(dimX),dimY(dimY),dimZ(dimZ),dimT(dimT){}
	SparseDeviceFourD(const SparseDeviceFourD& rhs):values(rhs.values),indices(rhs.indices),dimX(dimX),dimY(dimY),dimZ(dimZ),dimT(dimT){}
	SparseDeviceFourD(SparseDeviceVector<T> simple,Int4 size):values(simple.values()),indices(simple.indices()),dimX(size.x),dimY(size.y),dimZ(size.z),dimT(size.w){}
	SparseDeviceFourD(DeviceVector<T> values,DeviceVector<unsigned int> indices,Int4 size):values(values),indices(indices),dimX(size.x),dimY(size.y),dimZ(size.z),dimT(size.w){}
	DeviceVector<T> Values(){return values;}
	DeviceVector<unsigned int> Indices(){return indices;}
	int DimX()const{return dimX;}
	int DimY()const{return dimY;}
	int DimZ()const{return dimZ;}
	int DimT()const{return dimT;}
	Int4 Size()const{Int4 size;size.x=dimX;size.y=dimY;size.z=dimZ;size.w=dimT;return size;}
	SparseDeviceVector<T> GetSimple(){return SparseDeviceVector<T>(values,indices,dimX*dimY*dimZ*dimT);}
};