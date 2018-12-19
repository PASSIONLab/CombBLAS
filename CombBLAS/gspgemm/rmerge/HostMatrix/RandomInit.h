#pragma once

#include "HostMatrix/HostVector.h"
#include "HostMatrix/HostMatrix.h"
#include "HostMatrix/HostCube.h"
#include "HostMatrix/HostFourD.h"
#include "HostMatrix/Intrinsics.h"
#include "General/RandomGenerator.h"
#include "stdlib.h"

template<typename T>
static void RandomValue(T& t){
	int r=rand();
	double d=double(r)*(1.0/double(RAND_MAX));
	t=T(d);
}

static void RandomValue(int& t){
	int upper=rand()%32768;
	int lower=rand()%32768;
	t=upper*32768+lower;
}

static void RandomValue(uint& t){
	uint upper=rand()%65536;
	uint lower=rand()%65536;
	t=upper*65536+lower;
}

static void RandomValue(short& t){
	int r=rand();
	t=short(r%10);
}

static void RandomValue(unsigned short& t){
	int r=rand();
	t= (unsigned short)(r%10);
}


static void RandomValue(unsigned char& t){
	int r=rand();
	t=(unsigned char)(r%10);
}

template<typename T>
static T Rand(){
	T t;
	RandomValue(t);
	return t;
}

template<typename T>
static void RandomInit(HostVector<T> m, int seed=0){
	RandomGenerator rng(seed);
	for(int64 x=0;x<m.Length();x++)
		m[x]=rng.Rand<T>();
}

template<typename T>
static void RandomInit(HostMatrix<T> m,int seed=0){
	RandomGenerator rng(seed);
	for(int y=0;y<m.DimY();y++)
		for(int x=0;x<m.DimX();x++)
			m(x,y)=rng.Rand<T>();
}

template<typename T>
static void RandomInit(HostCube<T> m,int seed=0){
	RandomGenerator rng(seed);
	for(int z=0;z<m.DimZ();z++)
		for(int y=0;y<m.DimY();y++)
			for(int x=0;x<m.DimX();x++)
				m(x,y,z)=rng.Rand<T>();
}

template<typename T>
static void RandomInit(HostFourD<T> m,int seed=0){
	RandomGenerator rng(seed);
	for(int t=0;t<m.DimT();t++)
		for(int z=0;z<m.DimZ();z++)
			for(int y=0;y<m.DimY();y++)
				for(int x=0;x<m.DimX();x++)
					m(x,y,z,t)=rng.Rand<T>();
}

template<typename T>
static HostVector<T> RandomVector(int n, int seed=0){
	HostVector<T> x(n);
	RandomInit(x,seed);
	return x;
}
template<typename T>
static HostMatrix<T> RandomMatrix(int dimX, int dimY, int seed=0){
	HostMatrix<T> A(dimX,dimY);
	RandomInit(A,seed);
	return A;
}

template<typename T>
static HostCube<T> RandomCube(int dimX, int dimY, int dimZ, int seed=0){
	HostCube<T> A(dimX,dimY,dimZ);
	RandomInit(A,seed);
	return A;
}
