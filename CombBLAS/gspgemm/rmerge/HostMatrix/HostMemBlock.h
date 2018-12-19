#pragma once

#include "HostMatrix/int64.h"

//Allocates and deallocates an array in host memory.
template<typename T>
class HostMemBlock{
	T* data;
public:
	HostMemBlock():data(0){}
	explicit HostMemBlock(int64 length):data(new T[length]){}
	explicit HostMemBlock(T* data):data(data){}		
	~HostMemBlock(){delete [] data;data=0;}
	T* Pointer(){return data;}
};
