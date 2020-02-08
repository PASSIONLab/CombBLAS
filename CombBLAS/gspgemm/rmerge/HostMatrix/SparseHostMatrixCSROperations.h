#pragma once
#include "HostMatrix/SparseHostMatrixCSR.h"
#include "HostMatrix/SparseHostMatrixCSRCSC.h"
#include "HostMatrix/HostMatrix.h"
#include "HostMatrix/Intrinsics.h"
#include "HostMatrix/Reductions.h"
#include "HostMatrix/Scan.h"
#include "HostMatrix/HostTransfers.h"
#include "HostMatrix/SparseHostVectorOperations.h"
#include "HostMatrix/SparseHostMatrixCSRBLASOperations.h"
#include "HostMatrix/HostComponentWise.h"
#include "HostMatrix/ComponentWiseNames.h"
#include <vector>
#include <queue>
#include <set>
#include <map>
#include <algorithm>
#include <omp.h>
#include "General/ClosingFile.h"
#include "General/Serialize.h"

template<typename T>
static void Save(SparseHostMatrixCSR<T> A, std::string fileName){
	Verify(fileName.size()>4&&fileName.substr(fileName.size()-4,4)==".csr","File must end with .csr");
	ClosingFile file(fileName,"wb");
	ExMI::Serialize(file,(int)55673);//Control number
	ExMI::Serialize(file,(int)sizeof(T));
	ExMI::Serialize(file,(int)A.Width());
	ExMI::Serialize(file,(int)A.Height());
	ExMI::Serialize(file,(int64)A.NonZeroCount());
	ExMI::Serialize(file,A.Values().Pointer(),A.Values().Length());
	ExMI::Serialize(file,A.ColIndices().Pointer(),A.ColIndices().Length());
	ExMI::Serialize(file,A.RowStarts().Pointer(),A.RowStarts().Length());
}

template<typename T>
static SparseHostMatrixCSR<T> LoadSparseHostMatrixCSR(std::string fileName){
	Verify(fileName.size()>4&&fileName.substr(fileName.size()-4,4)==".csr","File must end with .csr");
	ClosingFile file(fileName,"rb");
	Verify(ExMI::Deserialize<int>(file)==55673,"Control number missing");//Control number
	Verify(ExMI::Deserialize<int>(file)==sizeof(T),"Wrong data type");
	int width=ExMI::Deserialize<int>(file);
	int height=ExMI::Deserialize<int>(file);	
	int64 nnz=ExMI::Deserialize<int64>(file);

	HostVector<T> values(nnz);
	ExMI::Deserialize(file,values.Pointer(),values.Length());
	HostVector<uint> colIndices(nnz);
	ExMI::Deserialize(file,colIndices.Pointer(),colIndices.Length());

	HostVector<uint> rowStarts(height+1);
	ExMI::Deserialize(file,rowStarts.Pointer(),rowStarts.Length());
	return SparseHostMatrixCSR<T>(width,height,values,colIndices,rowStarts);
}

template<typename T>
static void JacobiIteration(HostMatrix<T> Y, SparseHostMatrixCSR<T> A, HostVector<T> p, HostMatrix<T> X, HostMatrix<T> F){
	Verify(Y.Height()==A.Height() && X.Width()==Y.Width() && X.Height()==A.Width(),"voiu3o3uzu5zoi43uz5");
	#pragma omp parallel for
	for(int r=0;r<Y.Height();r++){		
		CSparseVector<T> rowA=A.GetRow(r);
		for(int i=0;i<Y.Width();i++){
			T Ax(0);
			for(int t=0;t<rowA.NonZeroCount();t++){
				uint j=rowA.Index(t);
				Ax+=rowA.Value(t)*X(i,j);
			}
			T negativeResiduum=Ax-F(i,r);
			T result=X(i,r)+p[r]*negativeResiduum;
			Y(i,r)=result;
		}
	}
}

template<typename T>
static HostVector<uint> RowLengths(SparseHostMatrixCSR<T> A){
	HostVector<uint> rowStarts=A.RowStarts();
	HostVector<uint> rowLengths(rowStarts.Length()-1);
	BinaryComponentWise(rowLengths,rowStarts.SubVector(1,A.Height()),rowStarts.SubVector(0,A.Height()),BinaryFunctors::Subtract());
	return rowLengths;
}

template<typename DST,typename SRC> SparseHostMatrixCSR<DST> Clamp(SparseHostMatrixCSR<SRC> A){
	return SparseHostMatrixCSR<DST>(A.Width(),A.Height(),Clamp<DST>(A.Values()),Clone(A.ColIndices()),Clone(A.RowStarts()));
}

template<typename T>
HostVector<T> Diag(SparseHostMatrixCSR<T> A){
	Verify(A.Width()==A.Height(),FileAndLine);
	HostVector<T> D(A.Height());
	#pragma omp parallel for
	for(int y=0;y<A.Height();y++){
		T* rowValues;
		unsigned int* rowIndices;
		int rowLength;
		A.GetRowPointer(y,rowValues,rowIndices,rowLength);
		T d=T(0);
		for(int i=0;i<rowLength;i++){
			if(rowIndices[i]==y){
				d=rowValues[i];
			}
		}
		D[y]=d;
	}
	return D;
}

template<typename T>
static T DiagonalDominance(SparseHostMatrixCSR<T> A){
	Verify(A.Width()==A.Height(),FileAndLine);
	HostVector<T> differences(A.Height());
	#pragma omp parallel for
	for(int y=0;y<A.Height();y++){
		T* rowValues;
		unsigned int* rowIndices;
		int rowLength;
		A.GetRow(y,rowValues,rowIndices,rowLength);
		T d(0);
		T sumOffDiagonal(0);
		for(int i=0;i<rowLength;i++){
			if(rowIndices[i]==y)
				d=Abs_rmerge(rowValues[i]);
			else
				sumOffDiagonal+=Abs_rmerge(rowValues[i]);

		}
		differences[y]=d-sumOffDiagonal;
	}
	return Min_rmerge(differences);
}

template<typename T>
static SparseHostMatrixCSR<T> Clone(SparseHostMatrixCSR<T> A){
	return SparseHostMatrixCSR<T>(A.Width(),A.Height(),Clone(A.Values()),Clone(A.ColIndices()),Clone(A.RowStarts()));
}

template<typename T>
static void Copy(SparseHostMatrixCSR<T> src, HostMatrix<T> dst){
	Verify(src.Width()==dst.Width() && src.Height()==dst.Height(),FileAndLine);
	ComponentWiseInit(dst,T(0));
	#pragma omp parallel for
	for(int r=0;r<src.Height();r++){
		T* rowValues;uint* rowIndices;int rowLength;
		src.GetRowPointer(r,rowValues,rowIndices,rowLength);
		for(int i=0;i<rowLength;i++)
			dst(rowIndices[i],r)=rowValues[i];
	}
}

template<typename T>
static HostVector<T> ToDense(SparseHostVector<T> x){
	HostVector<T> y(x.Length());
	Copy(x,y);
	return y;
}

template<typename T>
static HostMatrix<T> ToDense(SparseHostMatrixCSR<T> A){
	HostMatrix<T> B(A.Width(),A.Height());
	Copy(A,B);
	return B;
}

template<typename Ty, typename TA, typename Tx>
static void Mul(HostVector<Ty> y, SparseHostMatrixCSR<TA> A, HostVector<Tx> x){
	Verify(y.Length()==A.Height() && x.Length()==A.Width(),FileAndLine);
	int height=A.Height();
	#pragma omp parallel for
	for(int i=0;i<height;i++){
		TA* rowValues;unsigned int* rowIndices;int rowLength;
		A.GetRowPointer(i,rowValues,rowIndices,rowLength);
		Ty sum(0);
		for(int c=0;c<rowLength;c++)
			MulAdd(sum,rowValues[c],x[rowIndices[c]]);
			//sum+=rowValues[c]*x[rowIndices[c]];
		y[i]=sum;
	}
}
template<typename T> static HostVector<T> Mul(SparseHostMatrixCSR<T> A, HostVector<T> x){HostVector<T> y(A.Height());Mul(y,A,x);return y;}
template<typename T> static HostVector<T> operator*(SparseHostMatrixCSR<T> A, HostVector<T> x){return Mul(A,x);}

template<typename Ty, typename TA, typename Tx>
static void Mul(HostMatrix<Ty>& Y, SparseHostMatrixCSR<TA>& A, HostMatrix<Tx>& X){
	Verify(Y.Height()==A.Height() && X.Width()==Y.Width() && X.Height()==A.Width(),FileAndLine);
	int rhs=Y.Width();
	#pragma omp parallel for
	for(int r=0;r<A.Height();r++){
		TA* rowValues;uint* rowIndices;int rowLength;
		A.GetRowPointer(r,rowValues,rowIndices,rowLength);
		for(int w=0;w<rhs;w++)
			Y(w,r)=Ty(0);
		for(int c=0;c<rowLength;c++){
			TA value=rowValues[c];
			uint index=rowIndices[c];
			for(int w=0;w<rhs;w++)
				Y(w,r)+=value*X(w,index);
		}
	}
}
template<typename T> static HostMatrix<T> Mul(SparseHostMatrixCSR<T> A, HostMatrix<T> X){HostMatrix<T> Y(X.Width(),A.Height());Mul(Y,A,X);return Y;}
template<typename T> static HostMatrix<T> operator*(SparseHostMatrixCSR<T> A, HostMatrix<T> X){return Mul(A,X);}

template<typename Ty, typename TA, typename Tx>
static void Mul(HostVector<Ty> y, SparseHostMatrixCSRCSC<TA> A, HostVector<Tx> x){
	Mul(y,A.GetA(),x);
}


template<typename T>
static SparseHostMatrixCSR<T> ToSparse(int width, int height, HostVector<std::vector<T> > values,HostVector<std::vector<unsigned int> > colIndices){
	HostVector<unsigned int> rowStarts(height+1);
	int count=0;
	for(int i=0;i<height;i++){
		rowStarts[i]=count;
		count+=(int)values[i].size();
	}
	rowStarts[height]=count;

	SparseHostMatrixCSR<T> W(width,height,HostVector<T>(count),HostVector<unsigned int>(count),rowStarts);
	#pragma omp parallel for
	for(int i=0;i<height;i++){
		T* rowValues;unsigned int* rowIndices;int rowLength;
		W.GetRow(i,rowValues,rowIndices,rowLength);
		const std::vector<T>& vt=values[i];
		const std::vector<unsigned int>& it=colIndices[i];
		for(int t=0;t<rowLength;t++){
			rowValues[t]=vt[t];
			rowIndices[t]=it[t];
		}
	}
	return W;
}

//Each row has its values and column indices.
//Requires that each row is sorted by indices
template<typename T>
static SparseHostMatrixCSR<T> ToSparse(int width, int height, HostVector<std::vector<std::pair<T,unsigned int> > > rows){
	Verify(height==rows.Length(),FileAndLine);
	HostVector<unsigned int> rowStarts(height+1);
	int count=0;
	for(int i=0;i<height;i++){
		rowStarts[i]=count;
		count+=(int)rows[i].size();
	}
	rowStarts[height]=count;

	SparseHostMatrixCSR<T> W(width,height,HostVector<T>(count),HostVector<unsigned int>(count),rowStarts);
	#pragma omp parallel for
	for(int i=0;i<height;i++){
		T* rowValues;unsigned int* rowIndices;int rowLength;
		W.GetRow(i,rowValues,rowIndices,rowLength);
		const std::vector<std::pair<T,unsigned int> >& row=rows[i];
		for(int t=0;t<rowLength;t++){
			rowValues[t]=row[t].first;
			rowIndices[t]=row[t].second;
		}
	}
	return W;
}

template<typename T>
static SparseHostMatrixCSR<T> ToSparseCSR(HostMatrix<T> A){
	HostVector<uint> rowLengths(A.Height());	
	//#pragma omp parallel for
	for(int y=0;y<A.Height();y++){
		int nonZeros=0;
		for(int x=0;x<A.Width();x++)
			if(A(x,y)!=T(0.0))
				nonZeros++;
		rowLengths[y]=nonZeros;		
	}
	uint count=Sum(rowLengths);
	HostVector<uint> rowStarts=Scan(rowLengths);	

	SparseHostMatrixCSR<T> W(A.Width(),A.Height(),HostVector<T>(count),HostVector<uint>(count),rowStarts);
	//#pragma omp parallel for
	for(int y=0;y<A.Height();y++){
		SparseHostVector<T> row=W.Row(y);		
		int pos=0;
		for(int x=0;x<A.Width();x++){
			if(A(x,y)!=T(0.0)){
				row.Values().Set(pos,A(x,y));
				row.Indices().Set(pos,x);
				pos++;
			}
		}
	}
	return W;
}

template<typename T>
static bool EqualStructure(SparseHostMatrixCSR<T> A, SparseHostMatrixCSR<T> B){
	if(A.Width()!=B.Width())
		return false;
	if(A.Height()!=B.Height())
		return false;
	if(!Equal(A.RowStarts(),B.RowStarts()))
		return false;
	if(!Equal(A.ColIndices(),B.ColIndices()))
		return false;
	return true;
}

template<typename T>
static bool Equal(SparseHostMatrixCSR<T> A, SparseHostMatrixCSR<T> B){
	if(A.Width()!=B.Width())return false;
	if(A.Height()!=B.Height())return false;
	if(!Equal(A.RowStarts(),B.RowStarts()))return false;
	if(!Equal(A.Values(),B.Values()))return false;
	if(!Equal(A.ColIndices(),B.ColIndices()))return false;	
	return true;
}

template<typename T>
static SparseHostMatrixCSR<T> TransposeSlow(SparseHostMatrixCSR<T> A){
	HostVector<std::vector<T> > values(A.Width());
	HostVector<std::vector<unsigned int> > collumnIndices(A.Width());	
	for(int i=0;i<A.Height();i++)
	{
		T* rowValues;unsigned int* rowIndices;int rowLength;
		A.GetRow(i,rowValues,rowIndices,rowLength);
		for(int t=0;t<rowLength;t++){
			int j=rowIndices[t];
			T aij=rowValues[t];
			values[j].push_back(aij);
			collumnIndices[j].push_back(i);
		}
	}
	return ToSparse(A.Height(),A.Width(),values,collumnIndices);
}

template<typename T>
static SparseHostMatrixCSR<T> Transpose(SparseHostMatrixCSR<T> A){
	HostVector<T> values(A.Values().Length());
	HostVector<uint> indices(A.Values().Length());
	//count the elements per column
	HostVector<uint> colCounts(A.Width());
	ComponentWiseInit(colCounts,(uint)0);
	HostVector<uint> t=A.ColIndices();
	for(int64 i=0;i<t.Length();i++)
		colCounts[t[i]]++;
	HostVector<uint> colStarts(A.Width()+1);
	colStarts[0]=0;
	for(int64 i=1;i<colStarts.Length();i++)
		colStarts[i]=colStarts[i-1]+colCounts[i-1];

	//Now fill in the values and indices
	HostVector<uint> colPositions=Clone(colStarts);
	for(int i=0;i<A.Height();i++)
	{
		T* rowValues;uint* rowIndices;int rowLength;
		A.GetRowPointer(i,rowValues,rowIndices,rowLength);
		for(int t=0;t<rowLength;t++){
			int j=rowIndices[t];
			T aij=rowValues[t];
			uint& pos=colPositions[j];
			values[pos]=aij;
			indices[pos]=i;
			pos++;
		}
	}
	return SparseHostMatrixCSR<T>(A.Height(),A.Width(),values,indices,colStarts);
}

template<typename T>
void VerifyConsistency(SparseHostMatrixCSR<T> A){
	HostVector<uint> rowStarts=A.RowStarts();
	Verify(rowStarts.Length()==A.Height()+1,FileAndLine);	
	Verify(rowStarts[0]==0,FileAndLine);
	Verify(rowStarts[rowStarts.Length()-1]==A.Values().Length(),FileAndLine);
	unsigned int last=0;
	for(int i=1;i<rowStarts.Length();i++){
		unsigned int rowStart=rowStarts[i];
		Verify(rowStart>=last,FileAndLine);
		last=rowStart;
	}
	for(int i=0;i<A.Height();i++)
	{
		T* rowValues;unsigned int* rowIndices;int rowLength;
		A.GetRowPointer(i,rowValues,rowIndices,rowLength);
		for(int t=0;t<rowLength;t++){
			int j=rowIndices[t];
			Verify(j>=0 && j<A.Width(),FileAndLine);
		}
	}
}

//dst=a+b*bFactor
//Requires that a and b are sorted by indices.
//dst will also be sorted
template<typename T>
static std::vector< std::pair<T,unsigned int> > AddScaled(const std::vector<std::pair<T,unsigned int> >& a, CSparseVector<T> b, T bFactor){
	//typedef std::pair<T,unsigned int> VK;
	//dst.clear();
	//dst.reserve(a.size()+b.NonZeroCount());
	std::vector< std::pair<T,unsigned int> > dst;

	int aPos=0;
	int bPos=0;

	//While both have values 
	while(aPos<a.size() && bPos<b.NonZeroCount()){
		unsigned int aj=a[aPos].second;
		unsigned int bj=b.Index(bPos);
		if(aj==bj){
			dst.push_back(std::pair<T,unsigned int>(a[aPos].first+b.Value(bPos)*bFactor,aj));
			aPos++;
			bPos++;
		}
		else if(aj<bj){
			dst.push_back(a[aPos]);
			aPos++;
		}
		else{
			dst.push_back(std::pair<T,unsigned int>(b.Value(bPos)*bFactor,bj));
			bPos++;
		}
	}
	while(aPos<(int)a.size()){
		dst.push_back(a[aPos]);
		aPos++;
	}
	while(bPos<b.NonZeroCount()){
		dst.push_back(std::pair<T,unsigned int>(b.Value(bPos)*bFactor,b.Index(bPos)));
		bPos++;
	}
	return dst;
}

template<typename T>
static SparseHostMatrixCSR<T> CreateIdentityCSR(int n, T init=T(1)){
	HostVector<T> values(n);
	HostVector<uint> colIndices(n);
	HostVector<uint> rowStarts(n+1);
	for(int i=0;i<n;i++){
		values[i]=init;
		colIndices[i]=(uint)i;
		rowStarts[i]=(uint)i;
	}
	rowStarts[n]=n;
	return SparseHostMatrixCSR<T>(n,n,values,colIndices,rowStarts);
}

template<typename T>
static SparseHostMatrixCSR<T> Add(SparseHostMatrixCSR<T> A, SparseHostMatrixCSR<T> B){
	Verify(A.Height()==B.Height(),FileAndLine);
	Verify(A.Width()==B.Width(),FileAndLine);
	HostVector<uint> rowLengths(A.Height());
	#pragma omp parallel for
	for(int i=0;i<A.Height();i++)
		rowLengths[i]=OverlapCount(A.Row(i),B.Row(i));
	HostVector<uint> rowStarts=Scan(rowLengths);
	int nonZeroCount=rowStarts[rowStarts.Length()-1];
	SparseHostMatrixCSR<T> C(A.Width(),A.Height(),HostVector<T>(nonZeroCount),HostVector<uint>(nonZeroCount),rowStarts);
	#pragma omp parallel for
	for(int i=0;i<A.Height();i++)
		Add(C.Row(i),A.Row(i),B.Row(i));
	return C;
}

template<typename T>
static SparseHostMatrixCSR<T> MulSerial(SparseHostMatrixCSR<T> A, SparseHostMatrixCSR<T> B){
	typedef std::pair<T,unsigned int> VK;
	//Let result be C
	//Row i of C is the linear combination of some rows of B. Which rows and which coefficients is determined by row i of A.
	Verify(A.Width()==B.Height(),FileAndLine);
	HostVector<std::vector<VK> > dstRows(A.Height());
	//#pragma omp parallel for
	std::vector<VK> dstRow;
	std::vector<VK> tmp;
	dstRow.reserve(1024);
	tmp.reserve(1024);
	for(int i=0;i<A.Height();i++){
		CSparseVector<T> Arow=A.GetRow(i);
		tmp.clear();
		dstRow.clear();
		int cap=(int)tmp.capacity();
		for(int t=0;t<Arow.NonZeroCount();t++){
			unsigned int j=Arow.Index(t);
			T aij=Arow.Value(t);
			CSparseVector<T> Brow=B.GetRow(j);
			if(Brow.NonZeroCount()>0){
				//dstRow.swap(tmp);
				dstRow=AddScaled(dstRow,Brow,aij);
			}
		}
		dstRows[i]=dstRow;//TODO: avoid this copy
	}
	return ToSparse(B.Width(),A.Height(),dstRows);
}

template<typename T>
static std::vector< std::pair<T,unsigned int> > MulOld(CSparseVector<T> x, SparseHostMatrixCSR<T>& A){
	std::vector< std::pair<T,unsigned int> > y;
	for(int t=0;t<x.NonZeroCount();t++){
		unsigned int j=x.Index(t);
		T aij=x.Value(t);
		CSparseVector<T> row=A.GetRowC(j);
		if(row.NonZeroCount()>0){
			y=AddScaled(y,row,aij);
		}
	}
	return y;
}

class PairCompareSecond{
public:
	template<typename T>
	bool operator()(const std::pair<T,unsigned int>& a, const std::pair<T,unsigned int>& b){return a.second<b.second;}
};

template<typename OSTREAM, typename T, typename Sequence, typename Compare>
void operator<<(OSTREAM& o, std::priority_queue<T,Sequence,Compare> queue){
	while(!queue.empty()){
		o<<queue.top().second<<" ";
		queue.pop();
	}
	o<<"\n";
}

template<typename OSTREAM, typename T>
void operator<<(OSTREAM& o, CSparseVector<T> x){
	for(int t=0;t<x.NonZeroCount();t++)	
		o<<x.Index(t)<<" ";	
	o<<"\n";
}

template<typename T>
static int MulSparseTmpCount(const uint* xIndices, int xNonZeros, SparseHostMatrixCSR<T>& M){
	int n=0;
	for(int i=0;i<xNonZeros;i++){
		int r=xIndices[i];
		T* rowValues;uint*rowIndices;int rowLength;
		M.GetRowPointer(r,rowValues,rowIndices,rowLength);
		n+=rowLength;
	}
	return n;
}

//Computes results size for vector*sparseMatrix
template<typename T>
static int MulNonZeroCount(const uint* xIndices, int xNonZeros, SparseHostMatrixCSR<T>& M){
	std::set<uint> indices;
	for(int i=0;i<xNonZeros;i++){
		int r=xIndices[i];
		T* rowValues;uint*rowIndices;int rowLength;
		M.GetRow(r,rowValues,rowIndices,rowLength);
		for(int j=0;j<rowLength;j++)
			indices.insert(rowIndices[j]);
	}
	return int(indices.size());
}

//vector * sparse matrix
//with known result size
template<typename T>
static bool Mul(T* dstValues, uint*dstIndices, uint dstCount, T* xValues, uint* xIndices, uint xCount, SparseHostMatrixCSR<T>& M){
	//typedef std::map<uint,T> Elements;
	std::map<uint,T> elements;
	for(uint i=0;i<xCount;i++){
		uint r=xIndices[i];
		T xValue=xValues[i];
		T* rowValues;uint*rowIndices;int rowLength;
		M.GetRow(r,rowValues,rowIndices,rowLength);
		for(int j=0;j<rowLength;j++){
			uint index=rowIndices[j];
			typename std::map<uint,T>::iterator iter=elements.find(index);
			if(iter==elements.end())
				elements[index]=xValue*rowValues[j];
			else
				iter->second+=xValue*rowValues[j];
		}
	}
	//now convert std::map to the sparse format
	if(elements.size()!=dstCount)
		return false;
	int pos=0;
	for(typename std::map<uint,T>::iterator iter=elements.begin();iter!=elements.end();iter++){
		dstIndices[pos]=iter->first;
		dstValues[pos]=iter->second;		
		pos++;
	}
	return true;
}

template<typename T>
void MulOld(std::vector< std::pair<T,unsigned int> >& y, T* xValues, unsigned int* xIndices, int xNonZeros, SparseHostMatrixCSR<T>& M){
	typedef std::pair<T,unsigned int> VK;
	int totalSize=0;
	for(int t=0;t<xNonZeros;t++){
		unsigned int j=xIndices[t];
		totalSize+=M.RowLength(j);
	}

	std::vector<VK> all(totalSize);
	int pos=0;
	for(int t=0;t<xNonZeros;t++){
		unsigned int j=xIndices[t];
		T aij=xValues[t];
		CSparseVector<T> row=M.GetRowC(j);
		for(int k=0;k<row.NonZeroCount();k++){
			all[pos]=VK(aij*row.Value(k),row.Index(k));
			pos++;
		}
	}
	y.clear();
	if(all.size()==0)
		return;
	std::sort(all.begin(),all.end(),PairCompareSecond());
	y.reserve(all.size());
	y.push_back(all[0]);
	for(int i=1;i<(int)all.size();i++){
		VK o=all[i];
		if(y.back().second==o.second)
			y.back().first+=o.first;
		else{
			y.push_back(o);
		}
	}
}

template<typename T>
static SparseHostMatrixCSR<T> Mul(SparseHostMatrixCSR<T> A, SparseHostMatrixCSR<T> B){
	//Row i of C is the linear combination of some rows of B. Which rows and which coefficients is determined by row i of A.
	Verify(A.Width()==B.Height(),FileAndLine);

	//First determine the memory consumption of dst
	HostVector<uint> dstCounts(A.Height());//Number of nonZeros of each dst row
	//HostVector<uint> tmpCounts(A.Height());
	#pragma omp parallel for
	for(int r=0;r<A.Height();r++){
		T* rowValues;uint*rowIndices;int rowLength;
		A.GetRowPointer(r,rowValues,rowIndices,rowLength);
		dstCounts[r]=MulNonZeroCount(rowIndices,rowLength,B);
		//tmpCounts[r]=MulSparseTmpCount(rowIndices,rowLength,B);		
	}

	//int64 sumTmp;Sum(sumTmp,tmpCounts);
	//std::cout<<"Sum of tmp counts: "<<sumTmp<<"\n";
	//std::cout<<"Max tmp count: "<<Max(tmpCounts)<<"\n";
	//std::cout<<"Max row length: "<<Max(dstCounts)<<"\n";
	//double sum;Sum(sum,tmpCounts);
	//std::cout<<"Mean tmp count: "<<sum/double(A.Height())<<"\n";
	//Sum(sum,dstCounts);
	//std::cout<<"Mean row length: "<<sum/double(A.Height())<<"\n";

	//Now allocate dst
	HostVector<uint> dstRowStarts(A.Height()+1);
	dstRowStarts[0]=0;
	for(int i=1;i<=A.Height();i++)
		dstRowStarts[i]=dstRowStarts[i-1]+dstCounts[i-1];
	uint nonZeros=dstRowStarts[A.Height()];
	HostVector<T> dstValues(nonZeros);
	HostVector<uint> dstColIndices(nonZeros);
	SparseHostMatrixCSR<T> dst(B.Width(),A.Height(),dstValues,dstColIndices,dstRowStarts);

	//now fill dst
	#pragma omp parallel for
	for(int r=0;r<A.Height();r++){
		T* rowValues;uint*rowIndices;int rowLength;
		dst.GetRowPointer(r,rowValues,rowIndices,rowLength);
		T* aRowValues;uint*aRowIndices;int aRowLength;
		A.GetRowPointer(r,aRowValues,aRowIndices,aRowLength);
		Mul(rowValues,rowIndices,rowLength,aRowValues,aRowIndices,aRowLength,B);
	}
	return dst;
}

template<typename T>
static void Mul_mmT(SparseHostMatrixCSR<T> dst, SparseHostMatrixCSR<T> A, SparseHostMatrixCSR<T> B){
	Verify(dst.Height()==A.Height(),FileAndLine);
	Verify(dst.Width()==B.Height(),FileAndLine);
	Verify(A.Width()==B.Width(),FileAndLine);

	//#pragma omp parallel for
	for(int r=0;r<dst.Height();r++){
		SparseHostVector<T> dstRow=dst.Row(r);		
		SparseHostVector<T> aRow=A.Row(r);
		for(int i=0;i<dstRow.NonZeroCount();i++){
			int c=dstRow.Index(i);
			SparseHostVector<T> bRow=B.Row(c);
			T sum=Dot(aRow,bRow);
			dstRow.Value(i)=sum;
		}
	}
}

template<typename T>
static SparseHostMatrixCSR<T> MulOld(SparseHostMatrixCSR<T> A, SparseHostMatrixCSR<T> B){
	//typedef std::pair<T,unsigned int> VK;
	//Let result be C
	//Row i of C is the linear combination of some rows of B. Which rows and which coefficients is determined by row i of A.
	Verify(A.Width()==B.Height(),FileAndLine);
	HostVector<std::vector<std::pair<T,unsigned int> > > dstRows(A.Height());
	
	int n=A.Height();
	T* Avalues=A.Values().Data();
	unsigned int* ArowStarts=A.RowStarts().Data();
	unsigned int* AcolIndices=A.ColIndices().Data();

	#pragma omp parallel for
	for(int i=0;i<n;i++){
		unsigned int rowStart=ArowStarts[i];
		T* rowValues=Avalues+rowStart;
		unsigned int* rowIndices=AcolIndices+rowStart;
		unsigned int rowLength=ArowStarts[i+1]-rowStart;
		MulOld(dstRows[i],rowValues,rowIndices,rowLength,B);
	}
	return ToSparse(B.Width(),A.Height(),dstRows);
}

template<typename T, typename T1, typename T2, typename Scale>
static void RankOneUpdate(SparseHostMatrixCSR<T> A, HostVector<T1> x, HostVector<T2> y, Scale scale){
	Verify(A.Height()==x.Length() && A.Width()==y.Length(),FileAndLine);
	int64 n=A.Height();
	T* Avalues=A.Values().Data();
	unsigned int* rowStarts=A.RowStarts().Data();
	unsigned int* colIndices=A.ColIndices().Data();

	#pragma omp parallel for
	for(int64 r=0;r<n;r++){
		unsigned int rowStart=rowStarts[r];
		T* rowValues=Avalues+rowStart;
		unsigned int* rowIndices=colIndices+rowStart;
		unsigned int rowLength=rowStarts[r+1]-rowStart;
		for(unsigned int j=0;j<rowLength;j++){
			unsigned int c=rowIndices[j];
			rowValues[j]+=x[r]*y[c]*scale;
		}
	}
}
template<typename T, typename T1, typename T2>
static void RankOneUpdate(SparseHostMatrixCSR<T> A, HostVector<T1> x, HostVector<T2> y){RankOneUpdate(A,x,y,T(1));}
