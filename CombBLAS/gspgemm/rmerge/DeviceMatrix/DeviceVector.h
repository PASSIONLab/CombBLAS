#pragma once

#include "cuda_runtime_api.h"
#include "DeviceMatrix/DeviceMemBlock.h"
#include "HostMatrix/CVector.h"
#include "HostMatrix/Verify.h"
#include "HostMatrix/TrackedObject.h"

template<typename T>
class DeviceVector{
public:
	TrackedObject memBlock;
	CVector<T> v;	

public:
	typedef T Element;
	typedef DeviceVector<T> Vector;	
	void operator=(const DeviceVector& rhs){v=rhs.v;memBlock=rhs.memBlock;}
	DeviceVector(){}
	explicit DeviceVector(int64 length){
		std::shared_ptr<DeviceMemBlock<T> > p(new DeviceMemBlock<T>(length));
		memBlock=TrackedObject(p);
		v=CVector<T>(p->Pointer(),length,1);
	}		
	//Takes ownership. Releases with cudaFree
	explicit DeviceVector(T* data, int64 length){
		std::shared_ptr<DeviceMemBlock<T> > p(new DeviceMemBlock<T>(data));
		memBlock=TrackedObject(p);
		v=CVector<T>(p->Pointer(),length,1);
	}
	DeviceVector(const DeviceVector& rhs):v(rhs.v),memBlock(rhs.memBlock){}
	DeviceVector(TrackedObject memBlock, T* data, int64 length,int stride=1):memBlock(memBlock),v(data,length,stride){}
	DeviceVector(TrackedObject memBlock, CVector<T> v):memBlock(memBlock),v(v){}
	void Clear(){*this=DeviceVector<T>();}
	TrackedObject GetMemBlock(){return memBlock;}
	CVector<T> GetCVector(){return v;}
	CVector<T> GetC(){return v;}
	bool IsSimple(){return v.IsSimple();}
	DeviceVector<T> GetSimple(){
		if(!IsSimple())
			throw std::runtime_error("!Simple");
		return *this;
	}
	int64 Length()const{return v.Length();}
	int Length32()const{return int(v.Length());}
	int64 DimX()const{return v.Length();}
	int64 Size()const{return v.Size();}
	int Stride()const{return v.Stride();}
	const T* Data()const{return v.Data();}
	T* Data(){return v.Data();}
	T* Pointer(){return v.Data();}
	DeviceVector SubVector(int64 start, int64 length){
		if(start<0 || length<0 || start+length>v.Length())
			throw std::runtime_error("DeviceVector SubVector(int start, int length)");
		return DeviceVector(memBlock,v.SubVector(start,length));
	}
	T operator[](int64 i)const{T tmp;cudaMemcpy(&tmp,v.Data()+v.Index(i),sizeof(T),cudaMemcpyDeviceToHost);return tmp;}
	T operator()(int64 i)const{T tmp;cudaMemcpy(&tmp,v.Data()+v.Index(i),sizeof(T),cudaMemcpyDeviceToHost);return tmp;}
	void Set(int64 i, T value){cudaMemcpy(v.Data()+v.Index(i),&value,sizeof(T),cudaMemcpyHostToDevice);}
};