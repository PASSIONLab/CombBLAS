#pragma once

#include "HostMatrix/Intrinsics.h"
#include "HostMatrix/VectorTypes.h"
#include "HostMatrix/devicehost.h"
#include <cmath>

typedef unsigned char byte;

//Unary
template<typename T> __device__ __host__ static Vec1<T> operator-(const Vec1<T>& a){return Vec1<T>(-a.x);}
template<typename T> __device__ __host__ static Vec2<T> operator-(const Vec2<T>& a){return Vec2<T>(-a.x,-a.y);}
template<typename T> __device__ __host__ static Vec3<T> operator-(const Vec3<T>& a){return Vec3<T>(-a.x,-a.y,-a.z);}
template<typename T> __device__ __host__ static Vec4<T> operator-(const Vec4<T>& a){return Vec4<T>(-a.x,-a.y,-a.z,-a.w);}

//Binary
template<typename T> __device__ __host__ static Vec1<T> operator+(const Vec1<T>& a, const Vec1<T>& b){return Vec1<T>(a.x+b.x);}
template<typename T> __device__ __host__ static Vec2<T> operator+(const Vec2<T>& a, const Vec2<T>& b){return Vec2<T>(a.x+b.x,a.y+b.y);}
template<typename T> __device__ __host__ static Vec3<T> operator+(const Vec3<T>& a, const Vec3<T>& b){return Vec3<T>(a.x+b.x,a.y+b.y,a.z+b.z);}
template<typename T> __device__ __host__ static Vec4<T> operator+(const Vec4<T>& a, const Vec4<T>& b){return Vec4<T>(a.x+b.x,a.y+b.y,a.z+b.z,a.w+b.w);}

//TODO: pass const references everywhere
template<typename T> __device__ __host__ static Vec1<T> operator-(const Vec1<T>& a, const Vec1<T>& b){return Vec1<T>(a.x-b.x);}
template<typename T> __device__ __host__ static Vec2<T> operator-(const Vec2<T>& a, const Vec2<T>& b){return Vec2<T>(a.x-b.x,a.y-b.y);}
template<typename T> __device__ __host__ static Vec3<T> operator-(const Vec3<T>& a, const Vec3<T>& b){return Vec3<T>(a.x-b.x,a.y-b.y,a.z-b.z);}
template<typename T> __device__ __host__ static Vec4<T> operator-(const Vec4<T>& a, const Vec4<T>& b){return Vec4<T>(a.x-b.x,a.y-b.y,a.z-b.z,a.w-b.w);}

template<typename T> __device__ __host__ static void operator+=(Vec1<T>& dst, const Vec1<T>& src){dst.x+=src.x;}
template<typename T> __device__ __host__ static void operator+=(Vec2<T>& dst, const Vec2<T>& src){dst.x+=src.x;dst.y+=src.y;}
template<typename T> __device__ __host__ static void operator+=(Vec3<T>& dst, const Vec3<T>& src){dst.x+=src.x;dst.y+=src.y;dst.z+=src.z;}
template<typename T> __device__ __host__ static void operator+=(Vec4<T>& dst, const Vec4<T>& src){dst.x+=src.x;dst.y+=src.y;dst.z+=src.z;dst.w+=src.w;}
__device__ __host__ static void operator+=(Vec4<float>& dst, Vec4<unsigned char> src){dst.x+=src.x;dst.y+=src.y;dst.z+=src.z;dst.w+=src.w;}

template<typename T> __device__ __host__ static void operator-=(Vec1<T>& dst, const Vec1<T>& src){dst.x-=src.x;}
template<typename T> __device__ __host__ static void operator-=(Vec2<T>& dst, const Vec2<T>& src){dst.x-=src.x;dst.y-=src.y;}
template<typename T> __device__ __host__ static void operator-=(Vec3<T>& dst, const Vec3<T>& src){dst.x-=src.x;dst.y-=src.y;dst.z-=src.z;}
template<typename T> __device__ __host__ static void operator-=(Vec4<T>& dst, const Vec4<T>& src){dst.x-=src.x;dst.y-=src.y;dst.z-=src.z;dst.w-=src.w;}

template<typename T> __device__ __host__ static void operator*=(Vec1<T>& dst, T src){dst.x*=src;}
template<typename T> __device__ __host__ static void operator*=(Vec2<T>& dst, T src){dst.x*=src;dst.y*=src;}
template<typename T> __device__ __host__ static void operator*=(Vec3<T>& dst, T src){dst.x*=src;dst.y*=src;dst.z*=src;}
template<typename T> __device__ __host__ static void operator*=(Vec4<T>& dst, T src){dst.x*=src;dst.y*=src;dst.z*=src;dst.w*=src;}

template<typename T, typename S> __device__ __host__ static Vec1<T> operator*(S a, const Vec1<T>& b){Vec1<T> dst;dst.x=a*b.x;return dst;}
template<typename T, typename S> __device__ __host__ static Vec2<T> operator*(S a, const Vec2<T>& b){Vec2<T> dst;dst.x=a*b.x;dst.y=a*b.y;return dst;}
template<typename T, typename S> __device__ __host__ static Vec3<T> operator*(S a, const Vec3<T>& b){Vec3<T> dst;dst.x=a*b.x;dst.y=a*b.y;dst.z=a*b.z;return dst;}
template<typename T, typename S> __device__ __host__ static Vec4<T> operator*(S a, const Vec4<T>& b){Vec4<T> dst;dst.x=a*b.x;dst.y=a*b.y;dst.z=a*b.z;dst.w=a*b.w;return dst;}

template<typename T, typename S> __device__ __host__ static Vec1<T> operator*(const Vec1<T>& a, S b){return b*a;}
template<typename T, typename S> __device__ __host__ static Vec2<T> operator*(const Vec2<T>& a, S b){return b*a;}
template<typename T, typename S> __device__ __host__ static Vec3<T> operator*(const Vec3<T>& a, S b){return b*a;}
template<typename T, typename S> __device__ __host__ static Vec4<T> operator*(const Vec4<T>& a, S b){return b*a;}


template<typename T> __device__ __host__ static Vec1<T> operator/(const Vec1<T>& a, T b){Vec1<T> dst;dst.x=a.x/b;return dst;}
template<typename T> __device__ __host__ static Vec2<T> operator/(const Vec2<T>& a, T b){Vec2<T> dst;dst.x=a.x/b;dst.y=a.y/b;return dst;}
template<typename T> __device__ __host__ static Vec3<T> operator/(const Vec3<T>& a, T b){Vec3<T> dst;dst.x=a.x/b;dst.y=a.y/b;dst.z=a.z/b;return dst;}
template<typename T> __device__ __host__ static Vec4<T> operator/(const Vec4<T>& a, T b){Vec4<T> dst;dst.x=a.x/b;dst.y=a.y/b;dst.z=a.z/b;dst.w=a.w/b;return dst;}

template<typename T> __device__ __host__ static Vec1<T> operator%(const Vec1<T>& a, T b){Vec2<T> dst;dst.x=a.x%b;return dst;}
template<typename T> __device__ __host__ static Vec2<T> operator%(const Vec2<T>& a, T b){Vec2<T> dst;dst.x=a.x%b;dst.y=a.y%b;return dst;}
template<typename T> __device__ __host__ static Vec3<T> operator%(const Vec3<T>& a, T b){Vec3<T> dst;dst.x=a.x%b;dst.y=a.y%b;dst.z=a.z%b;return dst;}
template<typename T> __device__ __host__ static Vec4<T> operator%(const Vec4<T>& a, T b){Vec4<T> dst;dst.x=a.x%b;dst.y=a.y%b;dst.z=a.z%b;dst.w=a.w%b;return dst;}

template<typename T> __device__ __host__ static bool operator==(const Vec1<T>& a, const Vec1<T>& b){return a.x==b.x;}
template<typename T> __device__ __host__ static bool operator==(const Vec2<T>& a, const Vec2<T>& b){return a.x==b.x&&a.y==b.y;}
template<typename T> __device__ __host__ static bool operator==(const Vec3<T>& a, const Vec3<T>& b){return a.x==b.x&&a.y==b.y&&a.z==b.z;}
template<typename T> __device__ __host__ static bool operator==(const Vec4<T>& a, const Vec4<T>& b){return a.x==b.x&&a.y==b.y&&a.z==b.z&&a.w==b.w;}

template<typename T> __device__ __host__ static bool operator!=(const Vec1<T>& a, const Vec1<T>& b){return !(a==b);}
template<typename T> __device__ __host__ static bool operator!=(const Vec2<T>& a, const Vec2<T>& b){return !(a==b);}
template<typename T> __device__ __host__ static bool operator!=(const Vec3<T>& a, const Vec3<T>& b){return !(a==b);}
template<typename T> __device__ __host__ static bool operator!=(const Vec4<T>& a, const Vec4<T>& b){return !(a==b);}

template<typename T> __device__ __host__ static bool operator<(const Vec1<T>& a, const Vec1<T>& b){return a.x<b.x;}
template<typename T> __device__ __host__ static bool operator<(const Vec2<T>& a, const Vec2<T>& b){return a.x<b.x&&a.y<b.y;}
template<typename T> __device__ __host__ static bool operator<(const Vec3<T>& a, const Vec3<T>& b){return a.x<b.x&&a.y<b.y&&a.z<b.z;}
template<typename T> __device__ __host__ static bool operator<(const Vec4<T>& a, const Vec4<T>& b){return a.x<b.x&&a.y<b.y&&a.z<b.z&&a.w<b.w;}

template<typename T> __device__ __host__ static bool operator>(const Vec1<T>& a, const Vec1<T>& b){return a.x>b.x;}
template<typename T> __device__ __host__ static bool operator>(const Vec2<T>& a, const Vec2<T>& b){return a.x>b.x&&a.y>b.y;}
template<typename T> __device__ __host__ static bool operator>(const Vec3<T>& a, const Vec3<T>& b){return a.x>b.x&&a.y>b.y&&a.z>b.z;}
template<typename T> __device__ __host__ static bool operator>(const Vec4<T>& a, const Vec4<T>& b){return a.x>b.x&&a.y>b.y&&a.z>b.z&&a.w>b.w;}

template<typename T> __device__ __host__ static bool operator<=(const Vec1<T>& a, const Vec1<T>& b){return a.x<=b.x;}
template<typename T> __device__ __host__ static bool operator<=(const Vec2<T>& a, const Vec2<T>& b){return a.x<=b.x&&a.y<=b.y;}
template<typename T> __device__ __host__ static bool operator<=(const Vec3<T>& a, const Vec3<T>& b){return a.x<=b.x&&a.y<=b.y&&a.z<=b.z;}
template<typename T> __device__ __host__ static bool operator<=(const Vec4<T>& a, const Vec4<T>& b){return a.x<=b.x&&a.y<=b.y&&a.z<=b.z&&a.w<=b.w;}

template<typename T> __device__ __host__ static bool operator>=(const Vec1<T>& a, const Vec1<T>& b){return a.x>=b.x;}
template<typename T> __device__ __host__ static bool operator>=(const Vec2<T>& a, const Vec2<T>& b){return a.x>=b.x&&a.y>=b.y;}
template<typename T> __device__ __host__ static bool operator>=(const Vec3<T>& a, const Vec3<T>& b){return a.x>=b.x&&a.y>=b.y&&a.z>=b.z;}
template<typename T> __device__ __host__ static bool operator>=(const Vec4<T>& a, const Vec4<T>& b){return a.x>=b.x&&a.y>=b.y&&a.z>=b.z&&a.w>=b.w;}

//reductions, dot products
template<typename T> __device__ __host__ static T operator*(const Vec1<T>& a, const Vec1<T>& b){return a.x*b.x;}
template<typename T> __device__ __host__ static T operator*(const Vec2<T>& a, const Vec2<T>& b){return a.x*b.x+a.y*b.y;}
template<typename T> __device__ __host__ static T operator*(const Vec3<T>& a, const Vec3<T>& b){return a.x*b.x+a.y*b.y+a.z*b.z;}
template<typename T> __device__ __host__ static T operator*(const Vec4<T>& a, const Vec4<T>& b){return a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w;}

template<typename OStream, typename T> OStream& operator << (OStream& o, const Vec2<T>& x){o<<x.x<<" "<<x.y;return o;}
template<typename OStream, typename T> OStream& operator << (OStream& o, const Vec3<T>& x){o<<x.x<<" "<<x.y<<" "<<x.z;return o;}
template<typename OStream, typename T> OStream& operator << (OStream& o, const Vec4<T>& x){o<<x.x<<" "<<x.y<<" "<<x.z<<" "<<x.w;return o;}

template<typename IStream, typename T> IStream& operator >> (IStream& i, Vec2<T>& x){i>>x.x>>x.y;return i;}
template<typename IStream, typename T> IStream& operator >> (IStream& i, Vec3<T>& x){i>>x.x>>x.y>>x.z;return i;}
template<typename IStream, typename T> IStream& operator >> (IStream& i, Vec4<T>& x){i>>x.x>>x.y>>x.z>>x.w;return i;}

template<typename T> __device__ __host__ static T Dot(const Vec1<T>& a, const Vec1<T>& b){return a.x*b.x;}
template<typename T> __device__ __host__ static T Dot(const Vec2<T>& a, const Vec2<T>& b){return a.x*b.x+a.y*b.y;}
template<typename T> __device__ __host__ static T Dot(const Vec3<T>& a, const Vec3<T>& b){return a.x*b.x+a.y*b.y+a.z*b.z;}
template<typename T> __device__ __host__ static T Dot(const Vec4<T>& a, const Vec4<T>& b){return a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w;}

template<typename T> __device__ __host__ static T Sum(const Vec1<T>& a){return a.x;}
template<typename T> __device__ __host__ static T Sum(const Vec2<T>& a){return a.x+a.y;}
template<typename T> __device__ __host__ static T Sum(const Vec3<T>& a){return a.x+a.y+a.z;}
template<typename T> __device__ __host__ static T Sum(const Vec4<T>& a){return a.x+a.y+a.z+a.w;}

template<typename T> __device__ __host__ static T SumSquared(const Vec1<T>& a){return a.x*a.x;}
template<typename T> __device__ __host__ static T SumSquared(const Vec2<T>& a){return a.x*a.x+a.y*a.y;}
template<typename T> __device__ __host__ static T SumSquared(const Vec3<T>& a){return a.x*a.x+a.y*a.y+a.z*a.z;}
template<typename T> __device__ __host__ static T SumSquared(const Vec4<T>& a){return a.x*a.x+a.y*a.y+a.z*a.z+a.w*a.w;}

template<typename T> __device__ __host__ static T Norm2(const Vec1<T>& a){return a.x;}
template<typename T> __device__ __host__ static T Norm2(const Vec2<T>& a){return sqrt(SumSquared(a));}
template<typename T> __device__ __host__ static T Norm2(const Vec3<T>& a){return sqrt(SumSquared(a));}
template<typename T> __device__ __host__ static T Norm2(const Vec4<T>& a){return sqrt(SumSquared(a));}

template<typename T> __device__ __host__ static T NormMax(const Vec1<T>& a){return Abs_rmerge(a.x);}
template<typename T> __device__ __host__ static T NormMax(const Vec2<T>& a){return Max_rmerge(Abs_rmerge(a.x),Abs_rmerge(a.y));}
template<typename T> __device__ __host__ static T NormMax_rmerge(const Vec3<T>& a){return Max_rmerge(Max_rmerge(Abs_rmerge(a.x),Abs_rmerge(a.y)),Abs_rmerge(a.z));}
template<typename T> __device__ __host__ static T NormMax_rmerge(const Vec4<T>& a){return Max_rmerge(Max_rmerge(Max_rmerge(Abs_rmerge(a.x),Abs_rmerge(a.y)),Abs_rmerge(a.z)),Abs_rmerge(a.w));}

template<typename T> __device__ __host__ static T Distance(const Vec1<T>& a, const Vec1<T>& b){return sqrt(SumSquared(a-b));}
template<typename T> __device__ __host__ static T Distance(const Vec2<T>& a, const Vec2<T>& b){return sqrt(SumSquared(a-b));}
template<typename T> __device__ __host__ static T Distance(const Vec3<T>& a, const Vec3<T>& b){return sqrt(SumSquared(a-b));}
template<typename T> __device__ __host__ static T Distance(const Vec4<T>& a, const Vec4<T>& b){return sqrt(SumSquared(a-b));}

template<typename T, typename S> __device__ __host__ static void AddUpScaled(Vec1<T>& sum, const Vec1<T>& plus, const S& scale){sum.x+=plus.x*scale;}
template<typename T, typename S> __device__ __host__ static void AddUpScaled(Vec2<T>& sum, const Vec2<T>& plus, const S& scale){sum.x+=plus.x*scale;sum.y+=plus.y*scale;}
template<typename T, typename S> __device__ __host__ static void AddUpScaled(Vec3<T>& sum, const Vec3<T>& plus, const S& scale){sum.x+=plus.x*scale;sum.y+=plus.y*scale;sum.z+=plus.z*scale;}
template<typename T, typename S> __device__ __host__ static void AddUpScaled(Vec4<T>& sum, const Vec4<T>& plus, const S& scale){sum.x+=plus.x*scale;sum.y+=plus.y*scale;sum.z+=plus.z*scale;sum.w+=plus.w*scale;}

template<typename T, typename S> __device__ __host__ static void AddScaled(Vec1<T>& sum, const Vec1<T>& a, const Vec1<T>& b, const S& scale_b){sum.x=a.x+b.x*scale_b;}
template<typename T, typename S> __device__ __host__ static void AddScaled(Vec2<T>& sum, const Vec2<T>& a, const Vec2<T>& b, const S& scale_b){sum.x=a.x+b.x*scale_b;sum.y=a.y+b.y*scale_b;}
template<typename T, typename S> __device__ __host__ static void AddScaled(Vec3<T>& sum, const Vec3<T>& a, const Vec3<T>& b, const S& scale_b){sum.x=a.x+b.x*scale_b;sum.y=a.y+b.y*scale_b;sum.z=a.z+b.z*scale_b;}
template<typename T, typename S> __device__ __host__ static void AddScaled(Vec4<T>& sum, const Vec4<T>& a, const Vec4<T>& b, const S& scale_b){sum.x=a.x+b.x*scale_b;sum.y=a.y+b.y*scale_b;sum.z=a.z+b.z*scale_b;sum.w=a.w+b.w*scale_b;}

// - linear interpolation between a and b, based on value t in [0, 1] range
template<typename T, typename S> __device__ __host__ static T Lerp(const T& a, const T& b, const S& s){return a + s*(b-a);}
template<typename T, typename S> __device__ __host__ static Vec2<T> Lerp(const Vec2<T>& a, const Vec2<T>& b, const S& s){return a + s*(b-a);}
template<typename T, typename S> __device__ __host__ static Vec3<T> Lerp(const Vec3<T>& a, const Vec3<T>& b, const S& s){return a + s*(b-a);}
template<typename T, typename S> __device__ __host__ static Vec4<T> Lerp(const Vec4<T>& a, const Vec4<T>& b, const S& s){return a + s*(b-a);}

template<typename T> __device__ __host__ static Vec1<T> Min_rmerge(const Vec1<T>& a, const Vec1<T>& b){return Vec1<T>(Min_rmerge(a.x,b.x));}
template<typename T> __device__ __host__ static Vec2<T> Min_rmerge(const Vec2<T>& a, const Vec2<T>& b){return Vec2<T>(Min_rmerge(a.x,b.x),Min_rmerge(a.y,b.y));}
template<typename T> __device__ __host__ static Vec3<T> Min_rmerge(const Vec3<T>& a, const Vec3<T>& b){return Vec3<T>(Min_rmerge(a.x,b.x),Min_rmerge(a.y,b.y),Min_rmerge(a.z,b.z));}
template<typename T> __device__ __host__ static Vec4<T> Min_rmerge(const Vec4<T>& a, const Vec4<T>& b){return Vec4<T>(Min_rmerge(a.x,b.x),Min_rmerge(a.y,b.y),Min_rmerge(a.z,b.z),Min_rmerge(a.w,b.w));}

template<typename T> __device__ __host__ static Vec1<T> Max_rmerge(const Vec1<T>& a, const Vec1<T>& b){return Vec1<T>(Max_rmerge(a.x,b.x));}
template<typename T> __device__ __host__ static Vec2<T> Max_rmerge(const Vec2<T>& a, const Vec2<T>& b){return Vec2<T>(Max_rmerge(a.x,b.x),Max_rmerge(a.y,b.y));}
template<typename T> __device__ __host__ static Vec3<T> Max_rmerge(const Vec3<T>& a, const Vec3<T>& b){return Vec3<T>(Max_rmerge(a.x,b.x),Max_rmerge(a.y,b.y),Max_rmerge(a.z,b.z));}
template<typename T> __device__ __host__ static Vec4<T> Max_rmerge(const Vec4<T>& a, const Vec4<T>& b){return Vec4<T>(Max_rmerge(a.x,b.x),Max_rmerge(a.y,b.y),Max_rmerge(a.z,b.z),Max_rmerge(a.w,b.w));}

template<typename T> __device__ __host__ static T Min_rmerge(const Vec1<T>& a){return a.x;}
template<typename T> __device__ __host__ static T Min_rmerge(const Vec2<T>& a){return Min_rmerge(a.x,a.y);}
template<typename T> __device__ __host__ static T Min_rmerge(const Vec3<T>& a){return Min_rmerge(a.x,Min_rmerge(a.y,a.z));}
template<typename T> __device__ __host__ static T Min_rmerge(const Vec4<T>& a){return Min_rmerge(a.x,Min_rmerge(a.y,Min_rmerge(a.z,a.w)));}

template<typename T> __device__ __host__ static T Max_rmerge(const Vec1<T>& a){return a.x;}
template<typename T> __device__ __host__ static T Max_rmerge(const Vec2<T>& a){return Max_rmerge(a.x,a.y);}
template<typename T> __device__ __host__ static T Max_rmerge(const Vec3<T>& a){return Max_rmerge(a.x,Max_rmerge(a.y,a.z));}
template<typename T> __device__ __host__ static T Max_rmerge(const Vec4<T>& a){return Max_rmerge(a.x,Max_rmerge(a.y,Max_rmerge(a.z,a.w)));}

__device__ __host__ static double Distance(Int2 a, Int2 b){return sqrt(double(SumSquared(a-b)));}
__device__ __host__ static double Distance(Int3 a, Int3 b){return sqrt(double(SumSquared(a-b)));}
__device__ __host__ static double Distance(Int4 a, Int4 b){return sqrt(double(SumSquared(a-b)));}

template<typename T> __device__ __host__ static Vec3<T> Cross(const Vec3<T>& a, const Vec3<T>& b){return Vec3<T>(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);}

template<typename T> __device__ __host__ static Vec3<T> Normalize(const Vec3<T>& a){
	T norm=Norm2(a);
	if(norm==0)
		return a;
	return a*(T(1)/norm);
}

//Requires Float4 in [0,1]
static __device__ __host__ Float4 BlendFrontToBack(Float4 current, Float4 back){
	back.x *= back.w;
	back.y *= back.w;
	back.z *= back.w;
	return current + back*(1.0f - current.w);
}
