#pragma once

#include <exception>
#include <stdexcept>
#include "HostMatrix/TrackedObject.h"
#include "HostMatrix/HostMemBlock.h"
#include "HostMatrix/Verify.h"
#include "HostMatrix/CVector.h"
#include "HostMatrix/StrideIter.h"

template<typename T>
class HostVector{
public:
	CVector<T> v;
	TrackedObject memBlock;
	
public:
	typedef T Element;
	typedef HostVector<T> Vector;
	void operator=(const HostVector& rhs){v=rhs.v;memBlock=rhs.memBlock;}
	HostVector(){}
	HostVector(int64 length):v(new T[length],length,1),memBlock(new HostMemBlock<T>(v.Data())){
		Verify(v.Data()!=0,"Allocation not possible");
	}
	HostVector(T* data, int64 length, TrackedObject memBlock=TrackedObject()):v(data,length),memBlock(memBlock){}
	HostVector(const HostVector& rhs):v(rhs.v),memBlock(rhs.memBlock){}
	HostVector(CVector<T> v,TrackedObject memBlock)//Order has been changed on 6/20/2012
		:v(v),memBlock(memBlock){}
	void Clear(){*this=HostVector<T>();}
	T* Pointer(){return v.Data();}
	CVector<T> GetC(){return v;}
	bool IsSimple()const{return v.IsSimple();}
	HostVector<T> GetSimple(){
		if(!IsSimple())
			throw std::runtime_error("!Simple");
		return *this;
	}	
	int64 Size(){return v.Size();}
	T& operator[](int64 i){return v[i];}
	const T& operator[](int64 i)const{return v[i];}
	void Set(int64 i, T x){v[i]=x;}
	int64 Length()const{return v.Length();}
	int Length32()const{return (int)v.Length();}
	int64 DimX()const{return v.Length();}
	int Stride()const{return v.Stride();}
	const T* Data()const{return v.Data();}
	StrideIter<T> begin(){return v.begin();}
	StrideIter<T> end(){return v.end();}
	TrackedObject GetMemBlock(){return memBlock;}
	T* Data(){return v.Data();}
	HostVector SubVector(int64 start, int64 length){
		Verify(start>=0&&start+length<=v.Length(),FileAndLine);
		return HostVector(v.SubVector(start,length),memBlock);
	}
};
