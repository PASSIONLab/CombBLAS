#pragma once

#include "HostMatrix/HostVector.h"
#include "HostMatrix/CSparseVector.h"
typedef unsigned int uint;

//Contains a value and index for each non-zero element.
//The index is a one dimensional index, also for 2D, 3D, 4D data sets

template<typename T>
class SparseHostVector{
public:
	HostVector<T> values;
	HostVector<uint> indices;
	int64 dimX;
public:
	typedef T Element;	
	SparseHostVector(){}
	SparseHostVector(int64 dimX):dimX(dimX){}
	SparseHostVector(const SparseHostVector& rhs):values(rhs.values),indices(rhs.indices),dimX(rhs.dimX){}
	SparseHostVector(HostVector<T> values,HostVector<unsigned int> indices,int64 dimX):values(values),indices(indices),dimX(dimX){}
	SparseHostVector(int dimX, int nonZeros):dimX(dimX),values(nonZeros),indices(nonZeros){}
	HostVector<T> Values(){return values;}
	HostVector<uint> Indices(){return indices;}
	CSparseVector<T> GetC(){return CSparseVector<T>(values.Data(),indices.Data(),(int)dimX,(int)values.Length());}
	int64 NonZeroCount()const{return indices.Length();}
	int64 Length()const{return dimX;}
	int64 DimX()const{return dimX;}
	int64 Size()const{return dimX;}
	const T& Value(int i)const{return values[i];}
	T& Value(int i){return values[i];}
	uint& Index(int i){return indices[i];}
	const uint& Index(int i)const{return indices[i];}
};

template<typename T>
class SparseHostMatrix{
public:
	HostVector<T> values;
	HostVector<uint> indices;
	int dimX;
	int dimY;
public:
	typedef T Element;
	typedef HostVector<T> Vector;
	unsigned int Index(int x, int y){return y*dimX+x;}
	SparseHostMatrix(){}
	SparseHostMatrix(int dimX, int dimY):dimX(dimX),dimY(dimY){}
	SparseHostMatrix(const SparseHostMatrix& rhs):values(rhs.values),indices(rhs.indices),dimX(dimX),dimY(dimY){}
	SparseHostMatrix(SparseHostVector<T> simple,Int2 size):values(simple.values()),indices(simple.indices()),dimX(size.x),dimY(size.y){}
	SparseHostMatrix(HostVector<T> values,HostVector<uint> indices,int dimX,int dimY):values(values),indices(indices),dimX(dimX),dimY(dimY){}
	SparseHostMatrix(HostVector<T> values,HostVector<uint> indices,Int2 size):values(values),indices(indices),dimX(size.x),dimY(size.y){}
	HostVector<T> Values(){return values;}
	HostVector<uint> Indices(){return indices;}
	int DimX()const{return dimX;}
	int DimY()const{return dimY;}
	Int2 Size()const{Int2 size;size.x=dimX;size.y=dimY;return size;}
	SparseHostVector<T> GetSimple(){return SparseHostVector<T>(values,indices,dimX*dimY);}
};

template<typename T>
class SparseHostCube{
public:
	HostVector<T> values;
	HostVector<unsigned int> indices;
	int dimX;
	int dimY;
	int dimZ;
public:
	typedef T Element;
	typedef HostVector<T> Vector;
	unsigned int Index(int x, int y, int z){return z*dimX*dimY+y*dimX+x;}
	SparseHostCube(){}
	SparseHostCube(int dimX, int dimY,int dimZ):dimX(dimX),dimY(dimY),dimZ(dimZ){}
	SparseHostCube(const SparseHostCube& rhs):values(rhs.values),indices(rhs.indices),dimX(dimX),dimY(dimY),dimZ(dimZ){}
	SparseHostCube(SparseHostVector<T> simple,Int3 size):values(simple.Values()),indices(simple.Indices()),dimX(size.x),dimY(size.y),dimZ(size.z){}
	SparseHostCube(HostVector<T> values,HostVector<unsigned int> indices,int dimX,int dimY,int dimZ):values(values),indices(indices),dimX(dimX),dimY(dimY),dimZ(dimZ){}
	SparseHostCube(HostVector<T> values,HostVector<unsigned int> indices,Int3 size):values(values),indices(indices),dimX(size.x),dimY(size.y),dimZ(size.z){}
	HostVector<T> Values(){return values;}
	HostVector<unsigned int> Indices(){return indices;}
	int DimX()const{return dimX;}
	int DimY()const{return dimY;}
	int DimZ()const{return dimZ;}
	Int3 Size()const{Int3 size;size.x=dimX;size.y=dimY;size.z=dimZ;return size;}
	SparseHostVector<T> GetSimple(){return SparseHostVector<T>(values,indices,dimX*dimY*dimZ);}
};


template<typename T>
class SparseHostFourD{
public:
	HostVector<T> values;
	HostVector<unsigned int> indices;
	int dimX;
	int dimY;
	int dimZ;
	int dimT;
public:
	typedef T Element;
	typedef HostVector<T> Vector;
	unsigned int Index(int x, int y, int z, int t){return t*dimX*dimY*dimZ+z*dimX*dimY+y*dimX+x;}
	SparseHostFourD(){}
	SparseHostFourD(int dimX, int dimY, int dimZ, int dimT):dimX(dimX),dimY(dimY),dimZ(dimZ),dimT(dimT){}
	SparseHostFourD(const SparseHostFourD& rhs):values(rhs.values),indices(rhs.indices),dimX(dimX),dimY(dimY),dimZ(dimZ),dimT(dimT){}
	SparseHostFourD(SparseHostVector<T> simple,Int4 size):values(simple.values()),indices(simple.indices()),dimX(size.x),dimY(size.y),dimZ(size.z),dimT(size.w){}
	SparseHostFourD(HostVector<T> values,HostVector<unsigned int> indices,Int4 size):values(values),indices(indices),dimX(size.x),dimY(size.y),dimZ(size.z),dimT(size.w){}
	HostVector<T> Values(){return values;}
	HostVector<unsigned int> Indices(){return indices;}
	int DimX()const{return dimX;}
	int DimY()const{return dimY;}
	int DimZ()const{return dimZ;}
	int DimT()const{return dimT;}
	Int4 Size()const{Int4 size;size.x=dimX;size.y=dimY;size.z=dimZ;size.w=dimT;return size;}
	SparseHostVector<T> GetSimple(){return SparseHostVector<T>(values,indices,dimX*dimY*dimZ*dimT);}
};
