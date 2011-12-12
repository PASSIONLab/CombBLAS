#ifndef _TYPED_EDGE_
#define _TYPED_EDGE_

#include <iostream>
#include "CombBLAS.h"

using namespace std;

template <class T>
class TypedEdge
{
public:
	struct Weight<typename disable_if< is_boolean<T> > > {
		T operator() {	return my_weight; }
		T my_weight; 
	};
	TypedEdge(int type):my_type(type) {};
	TypedEdge():my_type(-1);	// defines a null-edge
	bool isCorrectType(int type) {	return (type == mytype); };
private:
	int my_type;
};

#endif
