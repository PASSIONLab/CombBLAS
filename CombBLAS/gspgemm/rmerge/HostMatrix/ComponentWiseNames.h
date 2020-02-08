#pragma once

//Give some better names for many operations
//This header requires some definitions for ComponentWiseInline, ComponentWise and BinaryComponentWise

#include "HostMatrix/Intrinsics.h"
#include "HostMatrix/ElementFunctors.h"

//Inline
template<typename T> void ComponentWiseInit(T x, typename T::Element init){ComponentWiseInline(x,ElementFunctors::Init<typename T::Element>(init));}
template<typename T, typename C> void ComponentWiseScale(T x, C scale){ComponentWiseInline(x,ElementFunctors::Scale<C>(scale));}
template<typename T> void ComponentWiseInv(T x){ComponentWiseInline(x,ElementFunctors::Invert());}
template<typename T, typename C> void ComponentWiseAddUpConstant(T x, C c){ComponentWiseInline(x,ElementFunctors::AddUpConstant<C>(c));}
template<typename T, typename C> void ComponentWiseSubUpConstant(T x, C c){ComponentWiseInline(x,ElementFunctors::SubUpConstant<C>(c));}
template<typename T> void ComponentWiseLog(T x){ComponentWiseInline(x,ElementFunctors::Log());}

//Src and Dst operand
template<typename DST, typename SRC> void ComponentWiseCopy(SRC src, DST dst){ComponentWise(dst,src,ElementFunctors::Copy());}
template<typename DST, typename SRC> void ComponentWiseConvert(SRC src, DST dst){ComponentWise(dst,src,ElementFunctors::Convert());}
template<typename DST, typename SRC, typename C> void ComponentWiseScale(DST dst, SRC src, C scale){ComponentWise(dst,src,ElementFunctors::Scale<C>(scale));}
template<typename T> void ComponentWiseAbs(T dst, T src){ComponentWise(dst,src,ElementFunctors::Absolute());}
template<typename T> void ComponentWiseSin(T dst, T src){ComponentWise(dst,src,ElementFunctors::Sin());}
template<typename T> void ComponentWiseCos(T dst, T src){ComponentWise(dst,src,ElementFunctors::Cos());}
template<typename T> void ComponentWiseExp(T dst, T src){ComponentWise(dst,src,ElementFunctors::Exp());}
template<typename T> void ComponentWiseLog(T dst, T src){ComponentWise(dst,src,ElementFunctors::Log());}
template<typename T> void ComponentWiseSqrt(T dst, T src){ComponentWise(dst,src,ElementFunctors::Sqrt());}
template<typename T> void ComponentWiseSquare(T dst, T src){ComponentWise(dst,src,ElementFunctors::Square());}
template<typename T> void ComponentWiseInv(T dst, T src){ComponentWise(dst,src,ElementFunctors::Invert());}
template<typename T> T ComponentWiseSquare(T src){T dst(src.Size());ComponentWise(dst,src,ElementFunctors::Square());return dst;}
template<typename T> void ComponentWiseSigmoid(T dst, T src){ComponentWise(dst,src,ElementFunctors::Sigmoid());}
template<typename T> void ComponentWiseLogSigmoid(T dst, T src){ComponentWise(dst,src,ElementFunctors::LogSigmoid());}
template<typename DST, typename SRC> void ComponentWiseAddUp(DST dst, SRC src){ComponentWiseAddUp(dst,src,ElementFunctors::Identity());}
template<typename DST, typename SRC, typename S> void ComponentWiseAddUpScaled(DST dst, SRC src, S scale){ComponentWise(dst,src,ElementFunctors::AddUpScaled<S>(scale));}
template<typename DST, typename SRC, typename S> void ComponentWiseAddUpScaledWithThresh(DST dst, SRC src, S scaleLess, S scaleMore, S thresh){ComponentWiseAddUp(dst,src,ElementFunctors::ScaleWithThresh<S>(scaleLess,scaleMore,thresh));}


//Two inputs, one output
template<typename DST, typename A, typename B> void ComponentWiseAdd(DST dst, A a, B b){BinaryComponentWise(dst,a,b,BinaryFunctors::Add());}
template<typename DST, typename A, typename B> void ComponentWiseSubtract(DST dst, A a, B b){BinaryComponentWise(dst,a,b,BinaryFunctors::Subtract());}
template<typename DST, typename A, typename B> void ComponentWiseMul(DST dst, A a, B b){BinaryComponentWise(dst,a,b,BinaryFunctors::Mul());}
template<typename A, typename B> A ComponentWiseMul(A a, B b){A c(a.Size());BinaryComponentWise(c,a,b,BinaryFunctors::Mul());return c;}
template<typename DST, typename A, typename B> void ComponentWiseAddUpMul(DST dst, A a, B b){BinaryComponentWiseAddUp(dst,a,b,BinaryFunctors::Mul());}
template<typename DST, typename A, typename B, typename Scale> void ComponentWiseAddScaled(DST dst, A a, B b, Scale scale){BinaryComponentWise(dst,a,b,BinaryFunctors::AddScaled<Scale>(scale));}
template<typename DST, typename A, typename B> void ComponentWiseMax(DST dst, A a, B b){BinaryComponentWise(dst,a,b,BinaryFunctors::Max_rmerge());}
template<typename DST, typename A, typename B> void ComponentWiseMin(DST dst, A a, B b){BinaryComponentWise(dst,a,b,BinaryFunctors::Min_rmerge());}
