#pragma once

#include "HostMatrix/devicehost.h"

template<typename T>
class Vec1{
public:
	T x;	
	__device__ __host__ Vec1(){}
	explicit __device__ __host__ Vec1(T init):x(init){}
	__device__ __host__ Vec1(const Vec1& rhs):x(rhs.x){}
	__device__ __host__ Vec1& operator=(const Vec1& rhs){x=rhs.x;return *this;}
};

template<typename T>
class Vec2{
public:
	T x;
	T y;
	__device__ __host__ Vec2(){}
	explicit __device__ __host__ Vec2(T init):x(init),y(init){}
	__device__ __host__ Vec2(T x, T y):x(x),y(y){}	
	__device__ __host__ Vec2(const Vec2& rhs):x(rhs.x),y(rhs.y){}
	__device__ __host__ Vec2& operator=(const Vec2& rhs){x=rhs.x;y=rhs.y;return *this;}
};

template<typename T>
class Vec3{
public:
	T x;
	T y;
	T z;
	__device__ __host__ Vec3(){}
	explicit __device__ __host__ Vec3(T init):x(init),y(init),z(init){}
	__device__ __host__ Vec3(T x, T y, T z):x(x),y(y),z(z){}
	__device__ __host__ Vec3(const Vec3& rhs):x(rhs.x),y(rhs.y),z(rhs.z){}//TODO: Why is this needed?
	__device__ __host__ Vec3& operator=(const Vec3& rhs){x=rhs.x;y=rhs.y;z=rhs.z;return *this;}//TODO: Why is this needed?
};

template<typename T>
class Vec4{
public:
	T x;
	T y;
	T z;
	T w;
	__device__ __host__ Vec4(){}
	explicit __device__ __host__ Vec4(T init):x(init),y(init),z(init),w(init){}
	__device__ __host__ Vec4(T x, T	y, T z, T w):x(x),y(y),z(z),w(w){}	
	__device__ __host__ Vec4(const Vec3<T>& a, T w):x(a.x),y(a.y),z(a.z),w(w){}
	__device__ __host__ Vec4(const Vec4& rhs):x(rhs.x),y(rhs.y),z(rhs.z),w(rhs.w){}
	__device__ __host__ Vec4& operator=(const Vec4& rhs){x=rhs.x;y=rhs.y;z=rhs.z;w=rhs.w;return *this;}
};


typedef Vec2<bool> Bool2;
typedef Vec2<char> Char2;
typedef Vec2<unsigned char> UChar2;
typedef Vec2<short> Short2;
typedef Vec2<unsigned short> UShort2;
typedef Vec2<int> Int2;
typedef Vec2<unsigned int> UInt2;
typedef Vec2<float> Float2;
typedef Vec2<double> Double2;

typedef Vec3<bool> Bool3;
typedef Vec3<char> Char3;
typedef Vec3<unsigned char> UChar3;
typedef Vec3<short> Short3;
typedef Vec3<unsigned short> UShort3;
typedef Vec3<int> Int3;
typedef Vec3<unsigned int> UInt3;
typedef Vec3<float> Float3;
typedef Vec3<double> Double3;


typedef Vec4<bool> Bool4;
typedef Vec4<char> Char4;
typedef Vec4<unsigned char> UChar4;
typedef Vec4<short> Short4;
typedef Vec4<unsigned short> UShort4;
typedef Vec4<int> Int4;
typedef Vec4<unsigned int> UInt4;
typedef Vec4<float> Float4;
typedef Vec4<double> Double4;
