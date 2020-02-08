#pragma once

#include <cmath>
#include "HostMatrix/int64.h"
#include "HostMatrix/devicehost.h"
#include "HostMatrix/VectorTypes.h"
#include <float.h>

typedef unsigned char byte;
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;

template<typename T> static __device__ __host__ T LogSigmoid(T x){
    if (x < -20.0)
        return x;
    return T(-log(1.0 + exp(-x)));
}

static __device__ __host__ bool IsFinite(float a){
    #if defined(_MSC_VER)
    return _finite(a)!=0;//for MSVC
    #else
    return __isinf(a)==0;//For gcc
    #endif
}

static __device__ __host__ bool IsFinite(double a){
    #if defined(_MSC_VER)
    return _finite(a)!=0;//for MSVC
    #else
    return __isinf(a)==0;//For gcc
    #endif
}

static __device__ __host__ bool IsNAN(float a){return !IsFinite(a);}
static __device__ __host__ bool IsNAN(double a){return !IsFinite(a);}

#ifdef __CUDACC__

//static __device__ float NegLogit(float x){
//	if (x > 20.0f)
//		return x;
//	return logf(1.0f + expf(x));
//}

//static __device__ double NegLogit(double x){
//	return (double)NegLogit(float(x));
//}

#endif

template<typename A, typename B, typename C>
static __device__ __host__ void MulAdd(A& a, const B& b, const C& c){
	a+=b*c;
}


template<typename T> static __device__ __host__ T NegLogit(T x){
    if (x > 20.0)
        return x;
    return T(log(T(1) + exp(x)));
}

template<typename T> static __device__ __host__ T Sigmoid(T d){
    if (d < -20.0)
        return T(0.0);
    return T(T(1)/(T(1)+exp(-d)));
}

template<typename T> static __device__ __host__ T SigmoidDeriv(T d){
	T tmp=Sigmoid(d);
	return tmp*(T(1.0)-tmp);
}

template<typename T> __device__ __host__ T Abs_rmerge(T x){return x>=T(0)?x:-x;}
static __device__ __host__ unsigned int Abs_rmerge(unsigned int x){return x;}
static __device__ __host__ unsigned short Abs_rmerge(unsigned short x){return x;}
static __device__ __host__ unsigned char Abs_rmerge(unsigned char x){return x;}

template<typename T> __device__ __host__ T Square(T x){return x*x;}

template<typename T> __device__ __host__ T MyMin(T a, T b){return a<b?a:b;}
template<typename T> __device__ __host__ T Min_rmerge(T a, T b){
	#if defined(__CUDACC__)
	return min(a,b);
	#else
	return a<b?a:b;
	#endif
}
template<typename T> __device__ __host__ T Max_rmerge(T a, T b){return a>b?a:b;}


//returns nominator/denominator rounded up.
static __device__ __host__ int DivUp(int nominator, int denominator){
    int roundDown=nominator/denominator;
    if(nominator%denominator != 0)
        roundDown++;
    return roundDown;
}

//returns nominator/denominator rounded up.
static __device__ __host__ int DivUp(unsigned int nominator, unsigned int denominator){
    unsigned int roundDown=nominator/denominator;
    if(nominator%denominator != 0)
        roundDown++;
    return roundDown;
}

static __device__ __host__ int64 DivUp(int64 nominator, int64 denominator){
    int64 roundDown=nominator/denominator;
    if(nominator%denominator != 0)
        roundDown++;
    return roundDown;
}

static Int3 DivUp(Int3 size,int by){
    return Int3(DivUp(size.x,by),DivUp(size.y,by),DivUp(size.z,by));
}

static __device__ __host__ bool InBounds(Int2 p, Int2 bounds){
    return p.x>=0 && p.y>=0 && p.x<bounds.x && p.y<bounds.y;
}

static __device__ __host__ bool InBounds(Int3 p, Int3 bounds){
    return p.x>=0 && p.y>=0 && p.z>=0 && p.x<bounds.x && p.y<bounds.y && p.z<bounds.z;
}

static __device__ __host__ int Round(double a){
    return (int)floor(a+0.5);
}
static __device__ __host__ Int3 Round(Float3 a){
    return Int3((int)floor(a.x+0.5f),(int)floor(a.y+0.5f),(int)floor(a.z+0.5f));
}

static __device__ __host__ Int2 Round(Double2 a){
    return Int2((int)floor(a.x+0.5),(int)floor(a.y+0.5));
}

static __device__ __host__ Int3 Round(Double3 a){
    return Int3((int)floor(a.x+0.5),(int)floor(a.y+0.5),(int)floor(a.z+0.5));
}

//This is a dangerous thing
template<typename T>
static void SwapEndianT(T& t){
    char* p=(char*)&t;
    for(int i=0;i<sizeof(t)/2;i++){
        char tmp=p[i];
        p[i]=p[sizeof(t)-i-1];
        p[sizeof(t)-i-1]=tmp;
    }
}

//Make only few overloads because it is wrong for many types.
static void SwapEndian(char& x){}
static void SwapEndian(uchar& x){}
static void SwapEndian(short& x){SwapEndianT(x);}
static void SwapEndian(ushort& x){SwapEndianT(x);}
static void SwapEndian(int& x){SwapEndianT(x);}
static void SwapEndian(uint& x){SwapEndianT(x);}
static void SwapEndian(float& x){SwapEndianT(x);}
static void SwapEndian(double& x){SwapEndianT(x);}
static void SwapEndian(UChar3& x){SwapEndianT(x);}
static void SwapEndian(Double3& x){SwapEndian(x.x);SwapEndian(x.y);SwapEndian(x.z);}

template<typename T>
static T AbsDifference(T a, T b){
    if(a>b)
        return a-b;
    else
        return b-a;
}

//It is better to use AddUpScaled than operator notation (e.g. y+=x*scale) because this can be implemented more
//efficiently for vectors.
template<typename T1, typename T2, typename S>
static  __device__ __host__ void AddUpScaled(T1& sum, const T2& plus, const S& scale){
	sum+=plus*scale;
}
template<typename T, typename S>
static  __device__ __host__ void AddScaled(T& sum, const T& a, const T& b, const S& scale_b){
	sum=a+b*scale_b;
}


//Might not be perfect...
static double RelativeError(double a, double b){
    if (a == b)
        return 0.0;
	return abs(a-b)/Max_rmerge(abs(a),abs(b));
}

template<typename T>
static __device__ __host__ T Huber(T src){
	if (Abs_rmerge(src) <= 1.28)
		return T(0.5)*src*src;
	else
		return T(1.28)*Abs_rmerge(src)-T(0.8192); // 0.8192=0.5*1.28^2
}

template<typename T>
static __device__ __host__ T HuberDeriv(T src){
	if (Abs_rmerge(src) <= 1.28)
		return src;
	else
		return src<0?T(-1.28):T(1.28);
}
