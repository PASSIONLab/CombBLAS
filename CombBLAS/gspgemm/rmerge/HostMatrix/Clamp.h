#pragma once

#include "HostMatrix/devicehost.h"
#include <cmath>
#include "float.h"
#include "HostMatrix/Intrinsics.h"
//#include "HostMatrix/MinMaxValues.h"
#include "HostMatrix/VectorTypes.h"
#include "General/WinOnlyStatic.h"

typedef unsigned char byte;

//http://en.wikipedia.org/wiki/Limits.h

//Clamping is fun.
//Clamp can convert from many types to many others (if implemented).
//It will for example clamp to [0...255] when clamping from float to byte.
//Clamp can also be used to go from Float4 to Float3 (by removing w) 
//or from Float3 to Float4 (by setting w = 0).

//Clamp with border given
template <typename T>
static __device__ __host__  T Clamp(T a, T low, T high){
    return Max_rmerge(low,Min_rmerge(high,a));
}

//Default implementation
template <typename Dst, typename Src>
static __device__ __host__ void Clamp(Dst& dst, Src x){
    dst=Dst(x);
}

//complete_basic means that the basic types char,uchar,short,ushort,int,uint,float,double are supported.

//To bool
static __device__ __host__  void Clamp(bool& y, char x){y=(x!=0);}
static __device__ __host__  void Clamp(bool& y, uchar x){y=x>0;}
static __device__ __host__  void Clamp(bool& y,short x){y=(x!=0);}
static __device__ __host__  void Clamp(bool& y,ushort x){y=(x!=0);}
static __device__ __host__  void Clamp(bool& y,int x){y=(x!=0);}
static __device__ __host__  void Clamp(bool& y,uint x){y=(x!=0);}
static __device__ __host__  void Clamp(bool& y,float x){y=(x!=0);}
static __device__ __host__  void Clamp(bool& y,double x){y=(x!=0);}

//To char (complete_basic)
static __device__ __host__  void Clamp(char& y,char x){y=x;}
static __device__ __host__  void Clamp(char& y,byte x){y=(char)Min_rmerge<byte>(127,x);}
static __device__ __host__  void Clamp(char& y,short x){y=(char)Max_rmerge<short>(-128,Min_rmerge<short>(127,x));}
static __device__ __host__  void Clamp(char& y,ushort x){y=(char)Min_rmerge<ushort>(127,x);}
static __device__ __host__  void Clamp(char& y,int x){y=(char)Max_rmerge<int>(-128,Min_rmerge<int>(127,x));}
static __device__ __host__  void Clamp(char& y,uint x){y=(char)Min_rmerge<uint>(127,x);}
static __device__ __host__  void Clamp(char& y,float x){y=(char)Max_rmerge<float>(-128.0f,Min_rmerge<float>(127.0f,x+0.5f));}
static __device__ __host__  void Clamp(char& y,double x){y=(char)Max_rmerge<double>(-128.0,Min_rmerge<double>(127.0,x+0.5));}

//To byte (complete_basic)
static __device__ __host__  void Clamp(byte& y,char x){y=(byte)Max_rmerge<char>(0,x);}
static __device__ __host__  void Clamp(byte& y,byte x){y=x;}
static __device__ __host__  void Clamp(byte& y,short x){y=(byte)Max_rmerge<short>(0,Min_rmerge<short>(255,x));}
static __device__ __host__  void Clamp(byte& y,ushort x){y=(byte)Min_rmerge<ushort>(255,x);}
static __device__ __host__  void Clamp(byte& y,int x){y=(byte)Max_rmerge(0,Min_rmerge(255,x));}
static __device__ __host__  void Clamp(byte& y,uint x){y=(byte)Min_rmerge<uint>(255,x);}
static __device__ __host__  void Clamp(byte& y,float x){y=(byte)Max_rmerge(0.0f,Min_rmerge(255.0f,x+0.5f));}
static __device__ __host__  void Clamp(byte& y,double x){y=(byte)Max_rmerge(0.0,Min_rmerge(255.0,x+0.5));}

//To short (complete_basic)
static __device__ __host__  void Clamp(short& y,char x){y=x;}
static __device__ __host__  void Clamp(short& y,byte x){y=x;}
static __device__ __host__  void Clamp(short& y,short x){y=x;}
static __device__ __host__  void Clamp(short& y,ushort x){y=(short)Min_rmerge<ushort>(32767,x);}
static __device__ __host__  void Clamp(short& y,int x){y=(short)Max_rmerge<int>(-32768,Min_rmerge<int>(32767,x));}
static __device__ __host__  void Clamp(short& y,uint x){y=(short)Min_rmerge<uint>(32767,x);}
static __device__ __host__  void Clamp(short& y,float x){y=(short)Max_rmerge<float>(-32768.0f,Min_rmerge<float>(32767.0f,x+0.5f));}
static __device__ __host__  void Clamp(short& y,double x){y=(short)Max_rmerge<double>(-32768.0,Min_rmerge<double>(32767.0,x+0.5));}

//To ushort (complete_basic)
static __device__ __host__  void Clamp(ushort& y,char x){y=x;}
static __device__ __host__  void Clamp(ushort& y,byte x){y=x;}
static __device__ __host__  void Clamp(ushort& y,short x){y=Max_rmerge<short>(0,x);}
static __device__ __host__  void Clamp(ushort& y,ushort x){y=x;}
static __device__ __host__  void Clamp(ushort& y,int x){y=(ushort)Max_rmerge<int>(-0,Min_rmerge<int>(65535,x));}
static __device__ __host__  void Clamp(ushort& y,uint x){y=(ushort)Min_rmerge<uint>(65535,x);}
static __device__ __host__  void Clamp(ushort& y,float x){y=(ushort)Max_rmerge<float>(-0.0,Min_rmerge<float>(65535.0f,x+0.5f));}
static __device__ __host__  void Clamp(ushort& y,double x){y=(ushort)Max_rmerge<double>(-0.0,Min_rmerge<double>(65535.0,x+0.5));}

//To int (complete_basic)
static __device__ __host__  void Clamp(int& y,char x){y=x;}
static __device__ __host__  void Clamp(int& y,byte x){y=x;}
static __device__ __host__  void Clamp(int& y,short x){y=x;}
static __device__ __host__  void Clamp(int& y,ushort x){y=x;}
static __device__ __host__  void Clamp(int& y,int x){y=x;}
static __device__ __host__  void Clamp(int& y,uint x){y=(int)Min_rmerge<uint>(2147483647,x);}
static __device__ __host__  void Clamp(int& y,float x){y=(int)Max_rmerge<float>(-2147483648.0f,Min_rmerge<float>(2147483647.0f,x+0.5f));}
static __device__ __host__  void Clamp(int& y,double x){y=(int)Max_rmerge<double>(-2147483648.0,Min_rmerge<double>(2147483647.0,x+0.5));}

//To uint (complete_basic)
static __device__ __host__  void Clamp(uint& y,char x){y=Max_rmerge<char>(0,x);}
static __device__ __host__  void Clamp(uint& y,byte x){y=x;}
static __device__ __host__  void Clamp(uint& y,short x){y=Max_rmerge<short>(0,x);}
static __device__ __host__  void Clamp(uint& y,ushort x){y=x;}
static __device__ __host__  void Clamp(uint& y,int x){y=Max_rmerge<int>(0,x);}
static __device__ __host__  void Clamp(uint& y,uint x){y=x;}
static __device__ __host__  void Clamp(uint& y,float x){y=(uint)Max_rmerge<float>(-0.0,Min_rmerge<float>(float(0xffffffff),x+0.5f));}
static __device__ __host__  void Clamp(uint& y,double x){y=(uint)Max_rmerge<double>(-0.0,Min_rmerge<double>(double(0xffffffff),x+0.5));}

//To float
static __device__ __host__  void Clamp(float& y, int x){y=(float)Max_rmerge<double>(-FLT_MAX,Min_rmerge<double>(FLT_MAX,x));}
static __device__ __host__  void Clamp(float& y, uint x){y=(float)Max_rmerge<double>(-FLT_MAX,Min_rmerge<double>(FLT_MAX,x));}
static __device__ __host__  void Clamp(float& y, double x){y=(float)Max_rmerge<double>(-FLT_MAX,Min_rmerge<double>(FLT_MAX,x));}

//To double: default implementation is sufficient

//Vector to scalar: use only x
template <typename Dst, typename Src> static __device__ __host__ void Clamp(Dst& dst, Vec2<Src> src){Clamp(dst,src.x);}
template <typename Dst, typename Src> static __device__ __host__ void Clamp(Dst& dst, Vec3<Src> src){Clamp(dst,src.x);}
template <typename Dst, typename Src> static __device__ __host__ void Clamp(Dst& dst, Vec4<Src> src){Clamp(dst,src.x);}

static __device__ __host__ Float4 Clamp(Float4 a, float low, float high){
	a.x=Clamp<float>(a.x,low,high);
	a.y=Clamp<float>(a.y,low,high);
	a.z=Clamp<float>(a.z,low,high);
	a.w=Clamp<float>(a.w,low,high);
	return a;
}

//Scalar to vector
template <typename Dst, typename Src>
static __device__ __host__ void Clamp(Vec2<Dst>& dst, Src src){
	Clamp(dst.x,src);
	Clamp(dst.y,src);
}

template <typename Dst, typename Src>
static __device__ __host__ void Clamp(Vec3<Dst>& dst, Src src){
	Clamp(dst.x,src);
	Clamp(dst.y,src);
	Clamp(dst.z,src);
}

template <typename Dst, typename Src>
static __device__ __host__ void Clamp(Vec4<Dst>& dst, Src src){
	Clamp(dst.x,src);
	Clamp(dst.y,src);
	Clamp(dst.z,src);
	Clamp(dst.w,src);
}

//Vector to vector
template <typename Dst, typename Src>
static __device__ __host__ void Clamp(Vec2<Dst>& dst, Vec2<Src> src){
	Clamp(dst.x,src.x);
	Clamp(dst.y,src.y);
}

template <typename Dst, typename Src>
static __device__ __host__ void Clamp(Vec2<Dst>& dst, Vec3<Src> src){
	Clamp(dst.x,src.x);
	Clamp(dst.y,src.y);
}

template <typename Dst, typename Src>
static __device__ __host__ void Clamp(Vec3<Dst>& dst, Vec3<Src> src){
	Clamp(dst.x,src.x);
	Clamp(dst.y,src.y);
	Clamp(dst.z,src.z);
}

template <typename Dst, typename Src>
static __device__ __host__ void Clamp(Vec3<Dst>& dst, Vec4<Src> src){
	Clamp(dst.x,src.x);
	Clamp(dst.y,src.y);
	Clamp(dst.z,src.z);
}

template <typename Dst, typename Src>
static __device__ __host__ void Clamp(Vec4<Dst>& dst, Vec4<Src> src){
	Clamp(dst.x,src.x);
	Clamp(dst.y,src.y);
	Clamp(dst.z,src.z);
	Clamp(dst.w,src.w);
}

template <typename Dst, typename Src>
static __device__ __host__ void Clamp(Vec4<Dst>& dst, Vec3<Src> src){
	Clamp(dst.x,src.x);
	Clamp(dst.y,src.y);
	Clamp(dst.z,src.z);
	Clamp(dst.w,0);
}

template <typename Dst, typename Src>
static __device__ __host__ Dst Clamp (Src x){
	Dst dst;
	Clamp(dst,x);
    return dst;
}


class ClampFunctor{	
public:
	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		Clamp(dst,src);
	}
};

//To apply a scale and offset and then clamp
template<typename T>
class AffineClampFunctor{	
	T scale;
	T offset;
public:
	AffineClampFunctor(T scale ,T offset):scale(scale), offset(offset){}
	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		dst=Clamp<Dst>(Clamp<T>(src)*scale+offset);
	}
};
