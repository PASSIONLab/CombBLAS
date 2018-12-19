#pragma once

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>

class RandomGenerator{
	boost::random::mt19937 generator;	
	double RandomDouble(double minVal, double maxVal){
		boost::random::uniform_real_distribution<> dist(minVal, maxVal);
		return dist(generator);
	}	
	
public:
		
	explicit RandomGenerator(int seed=0){
		generator.seed(seed);
	}

	template<typename T>
	T Rand(T minVal, T maxVal){
		return T(RandomDouble(minVal,maxVal));		
	}

	int Rand(int minVal, int maxVal){
		boost::random::uniform_int_distribution<> dist(minVal, maxVal);
		return dist(generator);
	}

	template<typename T>
	void Rand(T& t){t=T(RandomDouble(0.0,1.0));}

	void Rand(int& t){t=int(RandomDouble(0.0,10.0));}
	void Rand(uchar& t){t=uchar(RandomDouble(0.0,255.9));}
	void Rand(char& t){t=char(RandomDouble(-100,100));}
	void Rand(short& t){t=short(RandomDouble(-1000,1000));}
	void Rand(uint& t){t=uint(RandomDouble(0,1000));}
	void Rand(ushort& t){t=ushort(RandomDouble(0,1000));}

	template<typename T>
	T Rand(){T t;Rand(t);return t;}

};