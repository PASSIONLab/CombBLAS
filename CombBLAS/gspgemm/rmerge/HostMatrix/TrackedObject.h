#pragma once

#include <memory>

class TrackedObject{
	std::shared_ptr<void> ptr;
public:
	TrackedObject(){}

	explicit TrackedObject(std::shared_ptr<void> ptr):ptr(ptr){}

	//This c'tor takes over ownership
	template<typename T>
	explicit TrackedObject(T* theObject):ptr(std::shared_ptr<T>(theObject)){}
	void* Pointer(){
		return ptr.get();
	}	
};
