#ifndef _TYPED_EDGE_
#define _TYPED_EDGE_

#include <iostream>
#include <ctype>
#include "CombBLAS.h"

using namespace std;


template <class T>
class TypedEdge
{
public:
	struct Weight<typename enable_if< is_arithmetic< T > > > {	// only avaible if T is of type arithmetic (int, float, double, etc)
		T operator() {	return weight; }
		T weight; 
	};
	struct Interval<typename enable_if< is_same< T, pair<time_t,time_t > > > > {	// only available if T is of type pair<time_t, time_t>
		T operator() {	return make_pair(begin, end); }
		time_t begin;
		time_t end; 
	};
	TypedEdge(int mytype):type(mytype) {};
	bool isCorrectType(int type) {	return (type == mytype); };
	bool isWithinInterval(time_t begin, time_t end)	{	/* put stuff here */ };
private:
	int type;
};

#endif
