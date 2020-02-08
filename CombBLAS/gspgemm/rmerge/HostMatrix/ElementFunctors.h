#pragma once

#include "HostMatrix/Intrinsics.h"

namespace ElementFunctors{

//*****************************************************************************************************************
//Functors with interface: void operator()(Dst& dst, Src src)
//However the dst should only be assigned to. It should not be used in the calculations.

template<typename T>
class Init{
	T init;
public:
	Init(T init):init(init){}
	T Value(){return init;}
	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		dst=init;
	}
};

template<typename T>
class AddUpConstant{
	T plus;
public:
	AddUpConstant(T plus):plus(plus){}
	T Value(){return plus;}
	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		dst=src+plus;
	}
};

template<typename T>
class SubUpConstant{
	T minus;
public:
	SubUpConstant(T minus):minus(minus){}
	T Value(){return minus;}
	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		dst=src-minus;
	}
};

class SwapEndian{
public:
		template<typename T>
	__device__ __host__ void operator()(T& dst, T src){
		T tmp=src;
		::SwapEndian(tmp);
		dst=tmp;
	}
};

template<typename T>
class Constant{
	T t;
public:
	Constant(T t):t(t){}

	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		dst=t;
	}
};

class Copy{
	char dummy;
public:
	template<typename Src, typename Dst>
	__device__ __host__ void operator()(Dst& dst, Src src){
		dst=src;
	}
};

class Identity{
	char dummy;
public:
	template<typename Src, typename Dst>
	__device__ __host__ void operator()(Dst& dst, Src src){
		dst=src;
	}
};

class Negate{
	char dummy;
public:
	template<typename Src, typename Dst>
	__device__ __host__ void operator()(Dst& dst, Src src){
		dst=!src;
	}
};

class Negative{
	char dummy;
public:
	template<typename Src, typename Dst>
	__device__ __host__ void operator()(Dst& dst, Src src){
		dst=-src;
	}
	
	//This is to avoid a warning for uint
	__device__ __host__ void operator()(uint& dst, uint src){dst=src;}
	__device__ __host__ void operator()(ushort& dst, ushort src){dst=src;}
	__device__ __host__ void operator()(uchar& dst, uchar src){dst=src;}

};

template<typename T>
class AndWithConstant{
	T t;
public:
	AndWithConstant(T t):t(t){}
	template<typename Src, typename Dst>
	__device__ __host__ void operator()(Dst& dst, Src src){
		dst=src&t;
	}
};

template<typename T>
class OrWithConstant{
	T t;
public:
	OrWithConstant(T t):t(t){}
	template<typename Src, typename Dst>
	__device__ __host__ void operator()(Dst& dst, Src src){
		dst=src|t;
	}
};

class AndEqual{
	char dummy;
public:	
	template<typename Src, typename Dst>
	__device__ __host__ void operator()(Dst& dst, Src src){
		dst&=src;
	}
};

class OrEqual{
	char dummy;
public:	
	template<typename Src, typename Dst>
	__device__ __host__ void operator()(Dst& dst, Src src){
		dst|=src;
	}
};

class SubUp{
	char dummy;
public:
	template<typename Src, typename Dst>
	__device__ __host__ void operator()(Dst& dst, Src src){
		dst-=src;
	}
};

class Convert{
	char dummy;
public:
	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		dst=src;
	}
};

template<typename T>
class ClampBelow{
	T threshold;
public:
	ClampBelow(T threshold):threshold(threshold){}
	T Threshold(){return threshold;}
	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		if(src<threshold)
			dst=Dst(threshold);
		else
			dst=src;
	}
};

template<typename T>
class ClampAbove{
	T threshold;
public:
	ClampAbove(T threshold):threshold(threshold){}
	T Threshold(){return threshold;}
	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		if(src>threshold)
			dst=Dst(threshold);
		else
			dst=src;
	}
};

class ClampPos{
	char dummy;
public:
	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		if(src>Src(0))
			dst=Dst(0);
		else
			dst=src;
	}
};

class ClampNeg{
	char dummy;
public:
	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		if(src<0)
			dst=Dst(0);
		else
			dst=src;
	}
};

class IsNegative{
	char dummy;
public:
	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		if(src<=Src(0))
			dst=Dst(1);
		else
			dst=Dst(0);
	}
};

template<typename T>
class Scale{	
public:
	T scale;
	Scale(T scale):scale(scale){}
	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		dst=src*scale;
	}
	T Value(){return scale;}
};

template<typename T>
class AddUpScaled{	
public:
	T scale;
	AddUpScaled(T scale):scale(scale){}
	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		MulAdd(dst,src,scale);
		//dst+=src*scale;
	}
	T Value(){return scale;}
};

template<typename T>
class ScaleWithThresh{	
public:
	T scaleLess, scaleMore, thresh;
	ScaleWithThresh(T scaleLess, T scaleMore, T thresh):scaleLess(scaleLess),scaleMore(scaleMore),thresh(thresh){}
	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		if(src <= thresh)
			dst=src*scaleLess;
		else
			dst=src*scaleMore;
	}
	T Value(){return scaleLess;}
};


template<typename T>
class DivByConst{	
public:
	T div;
	DivByConst(T div):div(div){}
	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		dst=src/div;
	}
	T Value(){return div;}
};

template<typename T>
class DivUpByConst{	
public:
	T div;
	DivUpByConst(T div):div(div){}
	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		dst=DivUp(src,div);
	}
	T Value(){return div;}
};

class Exp{
	char dummy;
public:
	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		dst=exp(src);
	}
};

class Log{
	char dummy;
public:
	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		dst=log(src);
	}
};




template<typename T>
class LogPositive{
	T defaultVal;
public:
	explicit LogPositive(T defaultVal=T(0)):defaultVal(defaultVal){}
	
	//template<typename T>
	__device__ __host__ void operator()(T& dst, T src){
		if(src>0)
			dst=log(src);
		else
			dst=defaultVal;
	}	
};

class Sigmoid{
	char dummy;
public:
	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		dst=::Sigmoid(src);
	}	
};

class SigmoidDeriv{
	char dummy;
public:
	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		dst=::SigmoidDeriv(src);
	}	
};

class Sign{
	char dummy;
public:
	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		dst=src>Src(0)?Dst(1):Dst(-1);
	}	
};

class LogSigmoid{
	char dummy;
public:
	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		dst=::LogSigmoid(src);
	}	
};

class NegLogit{
	char dummy;
public:
	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		dst=::NegLogit(src);
	}	
};



class One{
	char dummy;
public:
	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		dst=Dst(1);
	}
};
class Sin{
	char dummy;
public:
	template<typename Src, typename Dst>
	__device__ __host__ void operator()(Dst& dst, Src src){
		dst=sin(src);
	}
};

class NegSin{
	char dummy;
public:
	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		dst=-sin(src);
	}
};

class Cos{
	char dummy;
public:
	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		dst=cos(src);
	}
};

class NegCos{
	char dummy;
public:
	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		dst=-cos(src);
	}
};

//Changes elements with a certain values
template<typename T>
class ChangeValueTmp{
	T a;
	T b;	
public:
	ChangeValueTmp(T a,T b):a(a),b(b){}

	__device__ __host__ void operator()(T& dst, T src){
		dst=(src==a)?b:src;
	}
};


//Changes elements with a certain values
template<typename T>
class ChangeValue{
	T which;
	T newValue;	
public:
	ChangeValue(T which,T newValue):which(which),newValue(newValue){}

	__device__ __host__ void operator()(T& dst, T src){
		dst=(src==which)?newValue:src;
	}
};


template<typename T>
class IsBelow{
	T threshold;
public:
	IsBelow(T threshold):threshold(threshold){}
	T Threshold(){return threshold;}
	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		if(src<threshold)
			dst=Dst(1);
		else
			dst=Dst(0);
	}
};


template<typename T>
class IsAbove{
	T threshold;
public:
	IsAbove(T threshold):threshold(threshold){}
	T Threshold(){return threshold;}
	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		if(src>threshold)
			dst=Dst(1);
		else
			dst=Dst(0);
	}
};

template<typename DST, typename Thresh=float>
class Threshold{
	Thresh lower;
	Thresh upper;
	DST positive;
	DST negative;

public:
	Threshold(Thresh lower,Thresh upper, DST positive=1,DST negative=0):lower(lower),upper(upper),positive(positive),negative(negative){}

	template<typename SRC>
	__device__ __host__ void operator()(DST& dst, SRC src){
		DST result=negative;
		if(src>=lower && src<=upper)
			result=positive;
		dst=result;
	}
};

//Only sets the positive guys
template<typename DST>
class ThresholdPositive{
	double lower;
	double upper;
	DST newVal;

public:
	ThresholdPositive(double lower,double upper, DST newVal=1):lower(lower),upper(upper),newVal(newVal){}

	template<typename SRC>
	__device__ __host__ void operator()(DST& dst, SRC src){
		if(src>=lower && src<=upper)
			dst=newVal;
	}
};

//Only sets the negative guys, i.e. those outside the threshold range
template<typename DST>
class ThresholdNegative{
	double lower;
	double upper;
	DST newVal;

public:
	ThresholdNegative(double lower,double upper, DST newVal=1):lower(lower),upper(upper),newVal(newVal){}

	template<typename SRC>
	__device__ __host__ void operator()(DST& dst, SRC src){
		if(!(src>=lower && src<=upper))
			dst=newVal;
	}
};

class Sqrt{
	char dummy;
public:
	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		dst=sqrt(src);
	}
};

class Square{
	char dummy;
public:
	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		dst=::Square(src);
	}
};

class Absolute{
	char dummy;
public:
	template<typename T>
	__device__ __host__ void operator()(T& dst, T src){
		dst=Abs_rmerge(src);
	}
};

class Invert{
	char dummy;
public:
	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		dst=Src(1)/src;
	}
};

class DerivOfInvert{
	char dummy;
public:
	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		Src tmp=src*src;
		dst=-Src(1)/tmp;
	}
};

class InvertNonZeros{
	char dummy;
public:
	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		dst=src!=0?Src(1)/src:src;
	}
};

class Huber{
	char dummy;
public:
	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		dst=::Huber(src);
	}
};


class HuberDeriv{
	char dummy;
public:
	template<typename Dst, typename Src>
	__device__ __host__ void operator()(Dst& dst, Src src){
		dst=::HuberDeriv(src);
	}
};



}



namespace BinaryFunctors{

class And{
	char dummy;
public:
	template<typename Dst, typename A, typename B>
	__device__ __host__ void operator()(Dst& dst, A a, B b){
		dst=a&&b;
	}
};

class Or{
	char dummy;
public:
	template<typename Dst, typename A, typename B>
	__device__ __host__ void operator()(Dst& dst, A a, B b){
		dst=a||b;
	}
};

class Add{
	char dummy;
public:
	template<typename Dst, typename A, typename B>
	__device__ __host__ void operator()(Dst& dst, A a, B b){
		dst=a+b;
	}
};

template<typename Scale>
class AddScaled{
	Scale scale;
public:
	__device__ __host__  AddScaled(){}
	explicit __device__ __host__  AddScaled(Scale scale):scale(scale){}
	template<typename Dst, typename A, typename B>
	__device__ __host__ void operator()(Dst& dst, A a, B b){
		dst=a+b*scale;
	}
};

class Subtract{
	char dummy;
public:	
	template<typename Dst, typename A, typename B>
	__device__ __host__ void operator()(Dst& dst, A a, B b){
		dst=a-b;
	}
};

class Mul{
	char dummy;
public:
	template<typename Dst, typename A, typename B>
	__device__ __host__ void operator()(Dst& dst, A a, B b){
		dst=a*b;
	}
};

class Div{
	char dummy;
public:	
	template<typename Dst, typename A, typename B>
	__device__ __host__ void operator()(Dst& dst, A a, B b){
		dst=a/b;
	}
};

class DivNonZeros{
	char dummy;
public:	
	template<typename Dst, typename A, typename B>
	__device__ __host__ void operator()(Dst& dst, A a, B b){
		if(b!=B(0))
			dst=a/b;
	}
};

class Equal{
	char dummy;
public:	
	template<typename Dst, typename A, typename B>
	__device__ __host__ void operator()(Dst& dst, A a, B b){
		dst=(a==b);
	}
};

class Max_rmerge{
	char dummy;
public:	
	template<typename Dst, typename A, typename B>
	__device__ __host__ void operator()(Dst& dst, A a, B b){
		dst=a>=b?a:b;
	}
};

class Min_rmerge{
	char dummy;
public:	
	template<typename Dst, typename A, typename B>
	__device__ __host__ void operator()(Dst& dst, A a, B b){
		dst=a<=b?a:b;
	}
};

class DistanceMax{
	char dummy;
public:	
	template<typename Dst, typename A, typename B>
	__device__ __host__ void operator()(Dst& dst, A a, B b){
		if(a>b)
			dst=a-b;
		else
			dst=b-a;
	}
};


class SquaredDifference{
	char dummy;
public:	
	template<typename Dst, typename A, typename B>
	__device__ __host__ void operator()(Dst& dst, A a, B b){
		Dst tmp=a-b;
		dst=tmp*tmp;
	}
};

template<typename T, typename NewVal>
class ThresholdMasked{
	uchar which;
	T lower;
	T upper;
	NewVal positive;
	NewVal negative;

public:
	ThresholdMasked(uchar which, T lower, T upper,NewVal positive=1, NewVal negative=0):which(which),lower(lower),upper(upper),positive(positive),negative(negative){}
	template<typename DST, typename Mask, typename Val>
	__device__ __host__ void operator()(DST& dst, Mask m, Val val){
		if(m==which)
			dst=(val>=lower && val<=upper)?positive:negative;
	}
};

template<typename T, typename NewVal>
class ThresholdPositiveMasked{
	unsigned char which;
	T lower;
	T upper;
	NewVal positive;

public:
	ThresholdPositiveMasked(unsigned char which, T lower, T upper,NewVal positive=1):which(which),lower(lower),upper(upper),positive(positive){}
	template<typename DST, typename Mask, typename Val>
	__device__ __host__ void operator()(DST& dst, Mask m, Val val){
		if(m==which && val>=lower && val<=upper)
			dst=positive;
	}
};


}
