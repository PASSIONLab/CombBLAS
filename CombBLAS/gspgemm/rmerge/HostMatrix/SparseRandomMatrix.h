#pragma once
#include "HostMatrix/RandomInit.h"
#include "HostMatrix/SparseHostMatrixCSR.h"
#include "HostMatrix/VectorTypes.h"
#include <map>

//sorts by y first, then by x
class ComparerYFirstThenX{
public:
	bool operator()(const UInt2& a, const UInt2& b)const{
		if(a.y<b.y)
			return true;
		if(a.y>b.y)
			return false;
		if(a.x<b.x)
			return true;
		return false;			
	}
};

template<typename T>
static SparseHostMatrixCSR<T> ToCSR(int dimX, int dimY, const std::map<UInt2, T, ComparerYFirstThenX>& nonZeros){
	int nonZeroCount=(int)nonZeros.size();
	HostVector<T> values(nonZeroCount);
	HostVector<uint> colIndices(nonZeroCount);
	HostVector<uint> rowCounts(dimY);
	ComponentWiseInit(rowCounts,0);

	int64 pos=0;
	typename std::map<UInt2, T, ComparerYFirstThenX>::const_iterator iter;		
	for(iter=nonZeros.begin();iter!=nonZeros.end();++iter){
		UInt2 current=iter->first;
		T val=iter->second;
		uint posY=current.y;
		uint posX=current.x;
		rowCounts[posY]++;
		values[pos]=val;
		colIndices[pos]=posX;
		pos++;
	}
	HostVector<uint> rowStarts=Scan(rowCounts);
	return SparseHostMatrixCSR<T>(dimX,dimY,values,colIndices,rowStarts);
}

//Creates a matrix with a fixed number of elements per row.
template<typename T>
static SparseHostMatrixCSR<T> RandomMatrixCSRFixed(int dimX, int dimY, int nonZerosPerRow){
	Verify(nonZerosPerRow>0,FileAndLine);
	Verify(nonZerosPerRow<=dimX,FileAndLine);
	int nonZeroCount=dimY*nonZerosPerRow;
	srand(0);
	HostVector<uint> rowCounts(dimY);
	ComponentWiseInit(rowCounts,nonZerosPerRow);
	HostVector<uint> rowStarts=Scan(rowCounts);
	HostVector<T> values(nonZeroCount);
	HostVector<uint> colIndices(nonZeroCount);
	SparseHostMatrixCSR<T> A(dimX,dimY,values,colIndices,rowStarts);

	#pragma omp parallel for
	for(int y=0;y<dimY;y++){
		T* rowValues;uint* rowIndices;int rowLength;
		A.GetRow(y,rowValues,rowIndices,rowLength);
		//Search for nonZerosPerRow unique indices
		int found=0;
		while(found<nonZerosPerRow){
			uint x=Rand<uint>()%dimX;
			//check if we have this
			bool already=false;
			for(int i=0;i<found;i++)
				if(rowIndices[i]==x)
					already=true;
			if(already)
				continue;
			rowIndices[found]=x;
			found++;
		}
		std::sort(rowIndices,rowIndices+rowLength);
		for(int i=0;i<rowLength;i++)
			rowValues[i]=Rand<T>();
	}	
	return A;
}

//Creates a matrix with a fixed number of elements per row.
template<typename T>
static SparseHostMatrixCSR<T> RandomBandedMatrixCSR(int widthHeight, int bandRadius){
	HostVector<uint> rowLengths(widthHeight);
	for(int i=0;i<widthHeight;i++){
		int start=Max_rmerge(0,i-bandRadius);
		int end=Min_rmerge(widthHeight,i+bandRadius+1);
		rowLengths[i]=end-start;
	}
	int nnz=Sum(rowLengths);
	HostVector<uint> rowStarts(widthHeight+1);
	Scan(rowStarts,rowLengths);
	SparseHostMatrixCSR<T> A(widthHeight,widthHeight,HostVector<T>(nnz),HostVector<uint>(nnz),rowStarts);
	for(int i=0;i<widthHeight;i++){
		int start=Max_rmerge(0,i-bandRadius);
		int end=Min_rmerge(widthHeight,i+bandRadius+1);
		T* rowValues;uint* rowIndices;int rowLength;
		A.GetRow(i,rowValues,rowIndices,rowLength);
		int pos=0;
		for(int u=start;u<end;u++){
			rowValues[pos]=1.22454;
			rowIndices[pos]=u;
			pos++;
		}
	}	
	return A;
}

static int ToIndex(int x, int y, int z, int n){
	return x+y*n+z*n*n;
}

template<typename T>
static SparseHostMatrixCSR<T> Poisson3DMatrixCSR(int n){	
	int widthHeight=n*n*n;
	HostVector<uint> rowCounts(widthHeight);
	//First compute the number of elements per row
	#pragma omp parallel for
	for(int x=0;x<n;x++){
		for(int y=0;y<n;y++){
			for(int z=0;z<n;z++){				
				int self=ToIndex(x,y,z,n);
				uint count=1;				
				if(x>0)count++;
				if(x<n-1)count++;
				if(y>0)count++;
				if(y<n-1)count++;
				if(z>0)count++;
				if(z<n-1)count++;
				rowCounts[self]=count;
			}
		}
	}
	
	HostVector<uint> rowStarts(rowCounts.Length()+1);
	Scan(rowStarts,rowCounts);
	uint nonZeroCount=rowStarts[rowStarts.Length()-1];
	SparseHostMatrixCSR<T> A(widthHeight,widthHeight,HostVector<T>(nonZeroCount),HostVector<uint>(nonZeroCount),rowStarts);
	//Now fill the matrix
	#pragma omp parallel for
	for(int x=0;x<n;x++){
		for(int y=0;y<n;y++){
			for(int z=0;z<n;z++){
				int self=ToIndex(x,y,z,n);
				T* rowValues;uint*rowIndices;int rowLength;
				A.GetRowPointer(self,rowValues,rowIndices,rowLength);
				uint pos=0;
				//Insert up to 7 values. Luckily they are ordered already...
				if(z>0){rowValues[pos]=T(-1);rowIndices[pos]=ToIndex(x,y,z-1,n);pos++;}
				if(y>0){rowValues[pos]=T(-1);rowIndices[pos]=ToIndex(x,y-1,z,n);pos++;}
				if(x>0){rowValues[pos]=T(-1);rowIndices[pos]=ToIndex(x-1,y,z,n);pos++;}
				rowValues[pos]=T(6);rowIndices[pos]=self;pos++;
				if(x<n-1){rowValues[pos]=T(-1);rowIndices[pos]=ToIndex(x+1,y,z,n);pos++;}								
				if(y<n-1){rowValues[pos]=T(-1);rowIndices[pos]=ToIndex(x,y+1,z,n);pos++;}			
				if(z<n-1){rowValues[pos]=T(-1);rowIndices[pos]=ToIndex(x,y,z+1,n);pos++;}
			}
		}
	}
	return A;
}


template<typename T>
static SparseHostMatrixCSR<T> RandomMatrixCSR(int dimX, int dimY, int nonZeroCount){
	Verify((int64)nonZeroCount<=((int64)dimX*(int64)dimY)/2,FileAndLine);//Otherwise this becomes too slow	
	std::map<UInt2, T, ComparerYFirstThenX> nonZeros;
	srand(0);

	while(nonZeros.size()<nonZeroCount){
		uint x;RandomValue(x);x=x%dimX;
		uint y;RandomValue(y);y=y%dimY;
		if(x>=(uint)dimX || y>=(uint)dimY)
			throw std::runtime_error("");
		UInt2 key(x,y);

		if(nonZeros.find(key)==nonZeros.end()){
			T t;RandomValue(t);
			nonZeros[key]=t;
		}
	}
	return ToCSR(dimX,dimY,nonZeros);
}

template<typename T>
static SparseHostMatrixCSR<T> RandomSPDMatrixCSR(int n, int nonZeroCount){
	Verify(nonZeroCount<=n*n/2,FileAndLine);//Otherwise this becomes too slow
	Verify(nonZeroCount>=n,FileAndLine);
	std::map<UInt2, T, ComparerYFirstThenX> nonZeros;
	HostVector<T> rowSums(n);
	ComponentWiseInit(rowSums,T(0));
	srand(0);
	while(nonZeros.size()<nonZeroCount-n){
		unsigned int x=Rand<unsigned int>()%n;
		unsigned int y=Rand<unsigned int>()%n;
		if(x==y)
			continue;
		if(x>=(unsigned int)n || y>=(unsigned int)n)
			throw std::runtime_error("");
		
		T value=-1;
		if(nonZeros.find(UInt2(x,y))==nonZeros.end()){
			nonZeros[UInt2(x,y)]=value;
			nonZeros[UInt2(y,x)]=value;
			rowSums[y]+=value;
			rowSums[x]+=value;
		}
	}
	for(int i=0;i<n;i++)
		nonZeros[UInt2(i,i)]=1+-rowSums[i];
	return ToCSR(n,n,nonZeros);
}
