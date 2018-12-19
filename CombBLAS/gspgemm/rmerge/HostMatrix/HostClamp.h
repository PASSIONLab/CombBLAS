#pragma once

#include "HostMatrix/HostComponentWise.h"
#include "HostMatrix/HostVector.h"
#include "HostMatrix/HostMatrix.h"
#include "HostMatrix/HostCube.h"
#include "HostMatrix/HostFourD.h"
#include "HostMatrix/Clamp.h"


template<typename DST,typename T> void Clamp(HostVector<DST> dst, HostVector<T> a){ComponentWise(dst,a,ClampFunctor());}
template<typename DST,typename T> void Clamp(HostMatrix<DST> dst, HostMatrix<T> a){ComponentWise(dst,a,ClampFunctor());}
template<typename DST,typename T> void Clamp(HostCube<DST> dst, HostCube<T> a){ComponentWise(dst,a,ClampFunctor());}
template<typename DST,typename T> void Clamp(HostFourD<DST> dst, HostFourD<T> a){ComponentWise(dst,a,ClampFunctor());}

template<typename DST,typename T> HostVector<DST> Clamp(HostVector<T> a){HostVector<DST> b(a.Size());ComponentWise(b,a,ClampFunctor());return b;}
template<typename DST,typename T> HostMatrix<DST> Clamp(HostMatrix<T> a){HostMatrix<DST> b(a.Size());ComponentWise(b,a,ClampFunctor());return b;}
template<typename DST,typename T> HostCube<DST> Clamp(HostCube<T> a){HostCube<DST> b(a.Size());ComponentWise(b,a,ClampFunctor());return b;}
template<typename DST,typename T> HostFourD<DST> Clamp(HostFourD<T> a){HostFourD<DST> b(a.Size());ComponentWise(b,a,ClampFunctor());return b;}