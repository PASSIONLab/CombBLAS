#pragma once
#include "HostMatrix/SparseHostVector.h"
#include "HostMatrix/MinMaxValues.h"
template<typename T>
static int OverlapCount(const SparseHostVector<T>& a, const SparseHostVector<T>& b)
{
	int count(0);
	int posA=0;
	int posB=0;
	uint aCurrent,bCurrent;
	if(posA<a.NonZeroCount())
		aCurrent=a.Index(0);
	else
		aCurrent=MaxVal<uint>();
	if(posB<b.NonZeroCount())
		bCurrent=b.Index(0);
	else
		bCurrent=MaxVal<uint>();
	uint minIndex=Min_rmerge(aCurrent,bCurrent);
	
	while(minIndex!=MaxVal<uint>()){
		//Advance the one with the smaller index or both		
		if(aCurrent==minIndex){
			posA++;
			if(posA<a.NonZeroCount())
				aCurrent=a.Index(posA);
			else
				aCurrent=MaxVal<uint>();			
		}
		if(bCurrent==minIndex){
			posB++;
			if(posB<b.NonZeroCount())
				bCurrent=b.Index(posB);
			else
				bCurrent=MaxVal<uint>();		
		}
		minIndex=Min_rmerge(aCurrent,bCurrent);
		count++;
	}
	return count;
}

//dst must be preallocated.
//Will fill dst.Values and dst.Indices
template<typename T>
static void Add(SparseHostVector<T> dst, const SparseHostVector<T>& a, const SparseHostVector<T>& b)
{
	int dstPos=0;
	int posA=0;
	int posB=0;
	uint aCurrent,bCurrent;
	if(posA<a.NonZeroCount())
		aCurrent=a.Index(0);
	else
		aCurrent=MaxVal<uint>();
	if(posB<b.NonZeroCount())
		bCurrent=b.Index(0);
	else
		bCurrent=MaxVal<uint>();
	uint minIndex=Min_rmerge(aCurrent,bCurrent);
	while(minIndex!=MaxVal<uint>()){
		T dstVal=0;		

		//Advance the one with the smaller index or both
		if(aCurrent==minIndex){
			dstVal+=a.Value(posA);
			posA++;
			if(posA<a.NonZeroCount())
				aCurrent=a.Index(posA);
			else
				aCurrent=MaxVal<uint>();			
		}
		if(bCurrent==minIndex){
			dstVal+=b.Value(posB);
			posB++;
			if(posB<b.NonZeroCount())
				bCurrent=b.Index(posB);
			else
				bCurrent=MaxVal<uint>();		
		}
		dst.Value(dstPos)=dstVal;
		dst.Index(dstPos)=minIndex;
		dstPos++;
		minIndex=Min_rmerge(aCurrent,bCurrent);		
	}
	Verify(dstPos==dst.NonZeroCount(),FileAndLine);
}

template<typename T>
static __device__ __host__ T Dot(const CSparseVector<T>& a, const CSparseVector<T>& b)
{
	T sum(0);
	int posB=0;
	for(int posA=0;posA<a.NonZeroCount();posA++){
		while(posB<b.NonZeroCount() && b.Index(posB)<a.Index(posA))
			posB++;
		if(posB==b.NonZeroCount())
			break;
		if(a.Index(posA)==b.Index(posB))
			sum+=a.Value(posA)*b.Value(posB);
	}
	return sum;
}

template<typename T>
static T Dot(SparseHostVector<T> a, SparseHostVector<T> b){
	return Dot(a.GetC(),b.GetC());
}

template<typename T>
static void Copy(SparseHostVector<T> x, SparseHostVector<T> y){
	Verify(x.Length()==y.Length(),FileAndLine);
	Verify(x.NonZeroCount()==y.NonZeroCount(),FileAndLine);
	for(int i=0;i<x.NonZeroCount();i++){
		y.Index(i)=x.Index(i);
		y.Value(i)=x.Value(i);
	}
}


template<typename T>
static void Copy(SparseHostVector<T> x, HostVector<T> y){
	Verify(x.Length()==y.Length(),FileAndLine);
	for(int i=0;i<y.DimX();i++)
		y[i]=0;
	HostVector<T> values=x.Values();
	HostVector<unsigned int> indices=x.Indices();
	for(int i=0;i<x.NonZeroCount();i++)
		y[indices[i]]=values[i];
}

template<typename Ty, typename Ta, typename Tb>
static void Dot(Ty& sum, HostVector<Ta>& a, SparseHostVector<Tb>& b){
	sum=Ty(0);
	HostVector<Tb> values=b.Values();
	HostVector<unsigned int> indices=b.Indices();
	for(int i=0;i<b.NonZeroCount();i++)
		sum+=a[indices[i]]*values[i];
}

template<typename Ty, typename Ta, typename Tb>
static void Dot(Ty& sum, SparseHostVector<Ta>& a, HostVector<Tb>& b){
	Dot(sum,b,a);
}

template<typename T>
static T Dot(HostVector<T>& a, SparseHostVector<T>& b){
	T sum(0);
	Dot(sum,a,b);
	return sum;
}

template<typename T>
static T Dot(SparseHostVector<T>& a, HostVector<T>& b){
	return Dot(b,a);
}

template<typename T>
static SparseHostMatrixCSR<T> ExtractRows(SparseHostMatrixCSR<T> A, HostVector<int> indices){
	HostVector<uint> rowLengths(indices.Length());
	for(int i=0;i<indices.Length32();i++)
		rowLengths[i]=A.RowLength(indices[i]);
	HostVector<uint> rowStarts=Scan(rowLengths);
	int total=rowStarts[rowStarts.Length()-1];
	SparseHostMatrixCSR<T> B(A.Width(),indices.Length32(),HostVector<T>(total),HostVector<uint>(total),rowStarts);
	for(int i=0;i<indices.Length32();i++)
		Copy(A.Row(indices[i]),B.Row(i));
	return B;
}


//Extract nonzeros
template<typename T> SparseHostVector<T> ToSparse(HostVector<T> v){	
	Verify(v.IsSimple(),"sasaahggfgy1");
	int count=0;
	for(int x=0;x<v.DimX();x++){
		if(v[x]!=T(0)){
			count++;
		}
	}
	HostVector<T> values(count);
	HostVector<unsigned int> indices(count);

	int pos=0;
	for(int x=0;x<v.DimX();x++){
		T value=v[x];
		if(value!=T(0)){
			values[pos]=value;
			indices[pos]=x;
			pos++;
		}
	}
	return SparseHostVector<T>(values,indices,v.DimX());
}
