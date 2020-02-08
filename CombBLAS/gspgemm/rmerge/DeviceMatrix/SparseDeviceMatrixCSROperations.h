#pragma once
#include "DeviceMatrix/SparseDeviceMatrixCSR.h"
#include "HostMatrix/SparseHostMatrixCSR.h"
#include "DeviceMatrix/SparseDeviceOperations.h"
#include "HostMatrix/HostMatrix.h"
#include "DeviceMatrix/WarpReduction.h"
#include "DeviceMatrix/DeviceTransfers.h"
#include "DeviceMatrix/DeviceReductions.h"
#include "DeviceMatrix/SparseDeviceMatrixCSRCSC.h"
#include "DeviceMatrix/SparseDeviceMatrixCOO.h"
#include "DeviceMatrix/Sort.h"
#include "HostMatrix/IO.h"
#include "DeviceMatrix/Scan.h"

template<typename T>
static void RowLengths(DeviceVector<uint> rowLengths, SparseDeviceMatrixCSR<T> A){	
	Verify(rowLengths.Length()==A.Height(),FileAndLine);	
	DeviceVector<uint> rowStarts=A.RowStarts();	
	BinaryComponentWise(rowLengths,rowStarts.SubVector(1,A.Height()),rowStarts.SubVector(0,A.Height()),BinaryFunctors::Subtract());	
}

template<typename T>
static DeviceVector<uint> RowLengths(SparseDeviceMatrixCSR<T> A){
	DeviceVector<uint> rowLengths(A.Height());
	RowLengths(rowLengths,A);
	return rowLengths;
}



template<typename T>
static uint MaxRowLength(SparseDeviceMatrixCSR<T> A){
	DeviceVector<uint> rowStarts=A.RowStarts();
	DeviceVector<uint> a=rowStarts.SubVector(0,rowStarts.Length()-1);
	DeviceVector<uint> b=rowStarts.SubVector(1,rowStarts.Length()-1);
	uint result=DistanceMax(a,b);	
	return result;
}

template<typename T>
static double MeanRowLength(SparseDeviceMatrixCSR<T> A){
	DeviceVector<uint> rowStarts=A.RowStarts();
	DeviceVector<uint> a=rowStarts.SubVector(0,rowStarts.Length()-1);
	DeviceVector<uint> b=rowStarts.SubVector(1,rowStarts.Length()-1);
	uint sum=DistanceSum(a,b);	
	return double(sum)/double(A.Height());
}


//Computes the max possible size of the tmp memory of A*B
//Overestimates the size, therefore wastes memory
template<int WarpSize, typename T>
__global__ void __cdecl SpmmEstimateTmpSizeKernel(uint* dstRowLengths, CSparseMatrixCSR<T> A, CSparseMatrixCSR<T> B){
	int r=threadIdx.x+(gridDim.x*blockIdx.y+blockIdx.x)*WarpSize;
	if(r<A.Height())
	{
		//const T* __restrict__ rowValues;
		//const uint* __restrict__ rowIndices;
		T* rowValues;
		uint* rowIndices;
		int rowLength=0;
		A.GetRow(r,rowValues,rowIndices,rowLength);
		uint result=0;
		for(int i=0;i<rowLength;i++)
			result+=B.RowLength(rowIndices[i]);
		dstRowLengths[r]=result;
	}
}

template<typename T>
void __cdecl SpmmEstimateTmpSize(DeviceVector<uint> dstRowLengths, SparseDeviceMatrixCSR<T> A, SparseDeviceMatrixCSR<T> B)
#ifdef __CUDACC__
{
	dim3 blockDim(512);
	dim3 gridDim(DivUp(A.Height(),(int)blockDim.x),1,1);
	SpmmEstimateTmpSizeKernel<512> <<< gridDim, blockDim, 0>>>(dstRowLengths.Data(),A.GetC(),B.GetC());
}
#else
;
#endif

template<typename T>
static double MaxTmpSize(SparseDeviceMatrixCSR<T> A, SparseDeviceMatrixCSR<T> B){
	DeviceVector<uint> tmp(A.Height());
	SpmmEstimateTmpSize(tmp,A,B);
	return Max_rmerge(tmp);
}

template<typename T>
static double MeanTmpSize(SparseDeviceMatrixCSR<T> A, SparseDeviceMatrixCSR<T> B){
	DeviceVector<uint> tmp(A.Height());
	SpmmEstimateTmpSize(tmp,A,B);
	return Sum(tmp)/double(A.Height());
}

template<typename T>
static SparseDeviceMatrixCSR<T> Clone(SparseDeviceMatrixCSR<T> A){
	return SparseDeviceMatrixCSR<T>(A.Width(),A.Height(),Clone(A.Values()),Clone(A.ColIndices()),Clone(A.RowStarts()));
}

template<typename T>
static SparseDeviceMatrixCSR<T> ToDevice(SparseHostMatrixCSR<T> A){
	return SparseDeviceMatrixCSR<T>(
		A.Width(),A.Height(),ToDevice(A.Values()),ToDevice(A.ColIndices()),ToDevice(A.RowStarts())
		);
}

template<typename T>
static SparseHostMatrixCSR<T> ToHost(SparseDeviceMatrixCSR<T> A){
	return SparseHostMatrixCSR<T>(
		A.Width(),A.Height(),ToHost(A.Values()),ToHost(A.ColIndices()),ToHost(A.RowStarts())
		);
}

template<typename DST,typename SRC> 
static SparseDeviceMatrixCSR<DST> Clamp(SparseDeviceMatrixCSR<SRC> A){
	return SparseDeviceMatrixCSR<DST>(A.Width(),A.Height(),Clamp<DST>(A.Values()),Clone(A.ColIndices()),Clone(A.RowStarts()));
}


#if defined (__CUDACC__) && defined (INSTANTIATE_0)
template<int BlockSize>
__global__ void __cdecl CudaPrepareTransposeKernel(uint* rowIndices, uint* range, int nonZeroCount, uint* rowStarts, int height){	
	int r=blockIdx.x*BlockSize+threadIdx.x;
	if(r>=height)
		return;
	uint rowStart=rowStarts[r];
	uint rowEnd=rowStarts[r+1];	
	for(int i=rowStart;i<rowEnd;i++){
		rowIndices[i]=r;
		range[i]=i;
	}	
}

//Go from row starts (0,3,4,7) to expanded row indices (0,0,0,1,2,2,2)
void __cdecl PrepareTranspose(DeviceVector<uint> rowIndices, DeviceVector<uint> range,DeviceVector<uint> rowStarts){
	const int BlockSize=1024;
	dim3 blockDim(BlockSize,1,1);
	int height=rowStarts.Length32()-1;
	dim3 gridDim(DivUp(height,BlockSize),1,1);
	CudaPrepareTransposeKernel<BlockSize> <<<gridDim, blockDim, 0>>>(
		rowIndices.Data(),range.Data(),rowIndices.Length32(),rowStarts.Data(),height);
}


template<int BlockSize>
__global__ void __cdecl CudaToRowStartsKernel(uint* rowStarts, int height, uint* rowIndices, int nonZeroCount){
	int i=BlockSize*blockIdx.x+threadIdx.x;
	//search for jumps
	if(i>=nonZeroCount)
		return;
	uint mine=rowIndices[i];
	uint next=height;
	if(i<nonZeroCount-1)
		next=rowIndices[i+1];	
	if(next>mine){//compute and set row end, i.e. row start of next
		uint rowEnd=i+1;
		rowStarts[mine+1]=rowEnd;
	}
}

//Requires sorted row indices
void __cdecl ToRowStarts(DeviceVector<uint> rowStarts, DeviceVector<uint> rowIndices){
	ComponentWiseInit(rowStarts,uint(0));
	const int BlockSize=512;
	dim3 blockDim(BlockSize,1,1);
	int blocks=DivUp(rowIndices.Length32(),BlockSize);
	dim3 gridDim(blocks,1,1);
	CudaToRowStartsKernel<BlockSize> <<<gridDim, blockDim, 0>>>(
		rowStarts.Data(),rowStarts.Length32()-1,rowIndices.Data(),rowIndices.Length32());
	rowStarts.Set(rowStarts.Length()-1,(uint)rowIndices.Length());
	ScanMaxInclusive(rowStarts);
}
#else
void __cdecl PrepareTranspose(DeviceVector<uint> rowIndices, DeviceVector<uint> range, DeviceVector<uint> rowStarts);

void __cdecl ToRowStarts(DeviceVector<uint> rowStarts, DeviceVector<uint> rowIndices);
#endif

template<typename T>
static SparseDeviceMatrixCSR<T> Transpose(SparseDeviceMatrixCSR<T> A){
	DeviceVector<uint> rowIndicesA(A.NonZeroCount());
	DeviceVector<uint> permutation(A.NonZeroCount());
	//std::cout<<"A.RowStarts(): "<<ToHost(A.RowStarts())<<std::endl;
	PrepareTranspose(rowIndicesA,permutation,A.RowStarts());
	//std::cout<<"range: "<<ToHost(permutation)<<std::endl;
	//std::cout<<"row Indices A: "<<ToHost(rowIndicesA)<<std::endl;
	DeviceVector<uint> rowIndices=Clone(A.ColIndices());
	StableSortByKey(permutation,rowIndices);
	//std::cout<<"permutation: "<<ToHost(permutation)<<std::endl;
	//std::cout<<"row Indices: "<<ToHost(rowIndices)<<std::endl;
	DeviceVector<uint> rowStarts(A.Width()+1);
	ToRowStarts(rowStarts,rowIndices);
	//std::cout<<"row Starts:  "<<ToHost(rowStarts)<<std::endl;	
	CudaCheckErrorTmp();
	DeviceVector<uint> colIndices(A.NonZeroCount());
	ExtractSparse(colIndices,rowIndicesA,permutation);
	//std::cout<<"col Indices: "<<ToHost(colIndices)<<std::endl;
	CudaCheckErrorTmp();
	DeviceVector<T> values(A.NonZeroCount());
	ExtractSparse(values,A.Values(),permutation);
	//std::cout<<"values: "<<ToHost(values)<<std::endl;
	CudaCheckErrorTmp();

	return SparseDeviceMatrixCSR<T>(A.Height(),A.Width(),values,colIndices,rowStarts);
}

#ifdef __CUDACC__

//A warp computes one row
template<int WarpSize, int BlockDimY, typename T>
__global__ void __cdecl CudaSparseRankOneUpdateKernel(CSparseMatrixCSR<T> dst, CVector<T> x, CVector<T> y, T scale){
	int r=blockIdx.x*BlockDimY+threadIdx.y;
	if(r>=dst.Height())
		return;
	T* rowValues;uint* rowIndices;int rowLength;
	dst.GetRow(r,rowValues,rowIndices,rowLength);
	for(int i=threadIdx.x;i<rowLength;i+=WarpSize){
		uint c=rowIndices[i];
		rowValues[i]+=x[r]*y[c]*scale;
	}
}
template<typename T>
void __cdecl CudaSparseRankOneUpdate(CSparseMatrixCSR<T> dst, CVector<T> x, CVector<T> y, T scale){
	const int WarpSize=8;
	const int BlockSizeY=64;
	int blocks=DivUp(dst.Height(),BlockSizeY);
	if(blocks>64000)
		throw std::runtime_error("CudaSparseRankOneUpdate: to large blockSize");
	dim3 gridDim(blocks,1,1);
	dim3 blockDim(WarpSize,BlockSizeY,1);
	CudaSparseRankOneUpdateKernel<WarpSize,BlockSizeY> <<< gridDim, blockDim, 0>>>(dst,x,y,scale);
}

#else
template<typename T>
void __cdecl CudaSparseRankOneUpdate(CSparseMatrixCSR<T> dst, CVector<T> x, CVector<T> y, T scale);

#endif

//Updates only the nonzero elements.
template<typename T>
static void RankOneUpdate(SparseDeviceMatrixCSR<T> dst, DeviceVector<T> x, DeviceVector<T> y, T scale){
	Verify(x.Length()==dst.Height() && y.Length()==dst.Width(),"5454tf");
	CudaSparseRankOneUpdate(dst.GetC(),x.GetC(),y.GetC(),scale);
}

#ifdef __CUDACC__


#define CUDA_SPAM_WARPSIZE 8
#define CUDA_SPAM_WARPSPERBLOCK (256/CUDA_SPAM_WARPSIZE)
#define CUDA_SPAM_GRIDSIZE_X (1024)

//A warp computes one output
template<bool SimpleX, typename Ty, typename TA, typename Tx>
__global__ void __cdecl CudaMulSparseMatrixCSRVectorKernel(int width, int height, CVector<Ty> y, TA* A_values, unsigned int* A_colIndices, unsigned int* A_rowStarts, CVector<Tx> x){
	int r=(blockIdx.x+CUDA_SPAM_GRIDSIZE_X*blockIdx.y)*CUDA_SPAM_WARPSPERBLOCK+threadIdx.y;
	if(r>=height)
		return;
	unsigned int rowStart=A_rowStarts[r];
	unsigned int rowLength=A_rowStarts[r+1]-rowStart;
	TA* rowValues=A_values+rowStart;
	unsigned int* rowIndices=A_colIndices+rowStart;
	__shared__ Ty allShared[CUDA_SPAM_WARPSIZE*CUDA_SPAM_WARPSPERBLOCK];
	Ty* shared=&allShared[0]+threadIdx.y*CUDA_SPAM_WARPSIZE;	
	int thread=threadIdx.x;
	shared[thread]=Ty(0);
	Tx* px=x.Data();
	for(int i=thread;i<rowLength;i+=CUDA_SPAM_WARPSIZE){
		uint j=rowIndices[i];
		shared[thread]+=rowValues[i]*(SimpleX?px[j]:x[j]);
	}
	BlockReduce<CUDA_SPAM_WARPSIZE>(shared,shared,thread,ReduceFunctors::AddFunctor());
	if(thread==0)
		y[r]=shared[0];
}

#define CUDA_SPAM_BLOCKSIZE 512
template<typename Ty, typename TA, typename Tx>
__global__ void __cdecl CudaMulSparseMatrixCSRVectorKernelSlow(int width, int height, Ty* py, TA* A_values, unsigned int* A_colIndices, unsigned int* A_rowStarts, Tx* px){
	int block=blockIdx.x;
	int r=block*CUDA_SPAM_BLOCKSIZE+threadIdx.x;
	if(r>=height)
		return;	
	unsigned int rowStart=A_rowStarts[r];
	unsigned int rowLength=A_rowStarts[r+1]-rowStart;
	TA* rowValues=A_values+rowStart;
	unsigned int* rowIndices=A_colIndices+rowStart;

	Ty sum(0);
	for(int i=0;i<rowLength;i++)
		sum+=rowValues[i]*px[rowIndices[i]];
	py[r]=sum;
}

//****************************************
template<typename Ty, typename TA, typename Tx>
void __cdecl CudaMulSparseMatrixCSRVector(int width, int height, CVector<Ty> y, TA* A_values, unsigned int* A_colIndices, unsigned int* A_rowStarts, CVector<Tx> x){
	int blocks=DivUp(height,CUDA_SPAM_WARPSPERBLOCK);
	dim3 gridDim(CUDA_SPAM_GRIDSIZE_X,DivUp(blocks,CUDA_SPAM_GRIDSIZE_X),1);
	if(gridDim.y==1)
		gridDim.x=blocks;
	if(blocks>65535*CUDA_SPAM_WARPSPERBLOCK*CUDA_SPAM_GRIDSIZE_X)
		throw std::runtime_error("CudaMulSparseMatrixCSRVector");
	dim3 blockDim(CUDA_SPAM_WARPSIZE,CUDA_SPAM_WARPSPERBLOCK,1);
	if(x.IsSimple())
		CudaMulSparseMatrixCSRVectorKernel<true> <<< gridDim, blockDim, 0>>>(width,height,y,A_values,A_colIndices,A_rowStarts,x);
	else
		CudaMulSparseMatrixCSRVectorKernel<false> <<< gridDim, blockDim, 0>>>(width,height,y,A_values,A_colIndices,A_rowStarts,x);
}

#else
template<typename Ty, typename TA, typename Tx>
void __cdecl CudaMulSparseMatrixCSRVector(int width, int height, CVector<Ty> y, TA* A_values, unsigned int* A_colIndices, unsigned int* A_rowStarts, CVector<Tx> x);

#endif


template<typename Ty, typename TA, typename Tx>
static void Mul(DeviceVector<Ty> y, SparseDeviceMatrixCSR<TA> A, DeviceVector<Tx> x){
	Verify(y.Length()==A.Height() && x.Length()==A.Width(),"11e1e4ffw");
	CudaMulSparseMatrixCSRVector(A.Width(),A.Height(),y.GetC(),A.Values().Data(),A.ColIndices().Data(),A.RowStarts().Data(),x.GetC());	
}

template<typename Ty, typename TA, typename Tx>
static void Mul(DeviceVector<Ty> y, SparseDeviceMatrixCSRCSC<TA> A, DeviceVector<Tx> x){
	Mul(y,A.GetA(),x);
}


template<typename T> static DeviceVector<T> Mul(SparseDeviceMatrixCSR<T> A, DeviceVector<T> x){DeviceVector<T> y(A.Height());Mul(y,A,x);return y;}
template<typename T> static DeviceVector<T> operator*(SparseDeviceMatrixCSR<T> A, DeviceVector<T> x){return Mul(A,x);}


#ifdef __CUDACC__

//Y=A*X or Y+=A*X
template<bool AddUp,uint BlockSizeX, uint BlockSizeY, uint GridSizeX, typename TY, typename TA, typename TX>
__global__ void __cdecl CudaMulSparseMatrixCSR_MatrixKernel(CMatrix<TY> Y, CSparseMatrixCSR<TA> A, CMatrix<TX> X){
	uint block=blockIdx.y*GridSizeX+blockIdx.x;
	uint r=block*BlockSizeY+threadIdx.y;
	if(r>=Y.Height())
		return;
	CSparseVector<TA> rowA=A.GetRow(r);
	//Each thread batch (blockDim.x) computes one row of Y
	for(int i=threadIdx.x;i<Y.Width();i+=BlockSizeX){
		TY result(0);
		for(int t=0;t<rowA.NonZeroCount();t++){
			uint j=rowA.Index(t);
			result+=rowA.Value(t)*X(i,j);
		}
		if(AddUp)
			Y(i,r)+=result;
		else
			Y(i,r)=result;
	}
}

template<typename TY, typename TA, typename TX>
void __cdecl CudaMulSparseMatrixCSR_Matrix(CMatrix<TY> Y, CSparseMatrixCSR<TA> A, CMatrix<TX> X){
	const uint BlockSizeX=16;
	const uint BlockSizeY=16;
	const uint GridSizeX=2048;
	uint blocks=DivUp((uint)Y.Height(),BlockSizeY);
	dim3 gridDim(GridSizeX,DivUp(blocks,GridSizeX),1);
	dim3 blockDim(BlockSizeX,BlockSizeY,1);
	CudaMulSparseMatrixCSR_MatrixKernel<false,BlockSizeX,BlockSizeY,GridSizeX,TY,TA,TX> <<<gridDim, blockDim,0>>>(Y,A,X);
}

#else
template<typename TY, typename TA, typename TX>
void __cdecl CudaMulSparseMatrixCSR_Matrix(CMatrix<TY> Y, CSparseMatrixCSR<TA> A, CMatrix<TX> X);
#endif

template<typename T>
static void Mul(DeviceMatrix<T> Y, SparseDeviceMatrixCSR<T> A, DeviceMatrix<T> X){
	Verify(Y.Height()==A.Height() && X.Width()==Y.Width() && X.Height()==A.Width(),"Size mismatch 129ds0vuxj");
	CudaMulSparseMatrixCSR_Matrix(Y.GetC(),A.GetC(),X.GetC());
}

template<typename T>
static void Mul(DeviceMatrix<T> Y, SparseDeviceMatrixCSRCSC<T> A, DeviceMatrix<T> X){
	Mul(Y,A.GetA(),X);	
}

template<typename T>
static DeviceMatrix<T> operator*(SparseDeviceMatrixCSR<T> A, DeviceMatrix<T> X){
	DeviceMatrix<T> Y(X.Width(),A.Height());
	Mul(Y,A,X);
	return Y;
}


//Jacobi iteration
#ifdef __CUDACC__

//P=-damping*invDiagA
template<uint BlockSizeX, uint BlockSizeY, uint GridSizeX, typename TY, typename TA, typename TX, typename TF>
__global__ void __cdecl CudaJacobiIterationKernel(CMatrix<TY> Y, CSparseMatrixCSR<TA> A, CVector<TA> p, CMatrix<TX> X, CMatrix<TF> F){
	uint block=blockIdx.y*GridSizeX+blockIdx.x;
	uint r=block*BlockSizeY+threadIdx.y;
	if(r>=Y.Height())
		return;
	CSparseVector<TA> rowA=A.GetRow(r);
	//Each thread batch (blockDim.x) computes one row of Y
	for(int i=threadIdx.x;i<Y.Width();i+=BlockSizeX){
		TY Ax(0);
		for(int t=0;t<rowA.NonZeroCount();t++){
			uint j=rowA.Index(t);
			Ax+=rowA.Value(t)*X(i,j);
		}
		TY negativeResiduum=Ax-F(i,r);
		TY result=X(i,r)+p[r]*negativeResiduum;
		Y(i,r)=result;
	}
}

template<typename TY, typename TA, typename TX, typename TF>
void __cdecl CudaJacobiIteration(CMatrix<TY> Y, CSparseMatrixCSR<TA> A, CVector<TA> P, CMatrix<TX> X, CMatrix<TF> F){
	const uint BlockSizeX=16;
	const uint BlockSizeY=16;
	const uint GridSizeX=2048;
	uint blocks=DivUp((uint)Y.Height(),BlockSizeY);
	dim3 gridDim(GridSizeX,DivUp(blocks,GridSizeX),1);
	dim3 blockDim(BlockSizeX,BlockSizeY,1);
	CudaJacobiIterationKernel<BlockSizeX,BlockSizeY,GridSizeX,TY,TA,TX,TF> <<<gridDim, blockDim,0>>>(Y,A,P,X,F);
}

#else
template<typename TY, typename TA, typename TX, typename TF>
void __cdecl CudaJacobiIteration(CMatrix<TY> Y, CSparseMatrixCSR<TA> A, CVector<TA> P, CMatrix<TX> X, CMatrix<TF> F);
#endif

template<typename T>
static void JacobiIteration(DeviceMatrix<T> Y, SparseDeviceMatrixCSR<T> A, DeviceVector<T> p, DeviceMatrix<T> X, DeviceMatrix<T> F){
	Verify(Y.Height()==A.Height() && X.Width()==Y.Width() && X.Height()==A.Width(),"fv9729873792");
	CudaJacobiIteration(Y.GetC(),A.GetC(),p.GetC(),X.GetC(),F.GetC());
}
//End of Jacobi Iteration

#ifdef __CUDACC__

//Y+=scale*A
template<int BlockSizeX, int BlockSizeY, typename TY, typename TA, typename Scale>
__global__ void __cdecl CudaAddScaledMatrix_SparseMatrixCSRKernel(CMatrix<TY> Y, CSparseMatrixCSR<TA> A, Scale scale){
	int r=BlockSizeX*blockIdx.y*32768+blockIdx.x*BlockSizeY+threadIdx.y;
	if(r>=A.Height())
		return;
	TA* rowValues;uint* rowIndices;int rowLength;
	A.GetRow(r,rowValues,rowIndices,rowLength);

	for(int t=threadIdx.x;t<rowLength;t+=BlockSizeX){
		uint j=rowIndices[t];
		Y(j,r)+=rowValues[t]*scale;
	}	
}

template<typename TY, typename TA, typename Scale>
void __cdecl CudaAddScaledMatrix_SparseMatrixCSR(CMatrix<TY> Y, CSparseMatrixCSR<TA> A, Scale scale){
	const int BlockSizeX=4;
	const int BlockSizeY=256;
	int64 blocks=DivUp(A.Height(),BlockSizeY);
	if(blocks>32768*2)
		throw "";
	dim3 gridDim((int)blocks,1,1);
	dim3 blockDim(BlockSizeX,BlockSizeY,1);
	CudaAddScaledMatrix_SparseMatrixCSRKernel<BlockSizeX,BlockSizeY> <<< gridDim, blockDim, 0>>>(Y,A,scale);

}

#else
template<typename TY, typename TA, typename Scale>
void __cdecl CudaAddScaledMatrix_SparseMatrixCSR(CMatrix<TY> Y, CSparseMatrixCSR<TA> A, Scale scale);
#endif

template<typename TY, typename TA, typename Scale>
void ComponentWiseAddUpScaled(DeviceMatrix<TY> Y, SparseDeviceMatrixCSR<TA> A, Scale scale){
	Verify(Y.Size()==A.Size(),"Size mismatch 42fwed2");
	CudaAddScaledMatrix_SparseMatrixCSR(Y.GetC(),A.GetC(),scale);
}

template<typename TY, typename TA>
void ComponentWiseAddUp(DeviceMatrix<TY> Y, SparseDeviceMatrixCSR<TA> A){
	Verify(Y.Size()==A.Size(),"Size mismatch. 208udsu98u");
	CudaAddScaledMatrix_SparseMatrixCSR(Y.GetC(),A.GetC(),TY(1));
}
template<typename T>
static DeviceMatrix<T> ToDense(SparseDeviceMatrixCSR<T> A){
	DeviceMatrix<T> D(A.Size());
	ComponentWiseInit(D,T(0));
	ComponentWiseAddUp(D,A);
	return D;
}

template<typename T>
static bool EqualStructure(SparseDeviceMatrixCSR<T> A, SparseDeviceMatrixCSR<T> B){
	if(A.Width()!=B.Width())return false;
	if(A.Height()!=B.Height())return false;
	if(!Equal(A.RowStarts(),B.RowStarts()))return false;
	if(!Equal(A.ColIndices(),B.ColIndices()))return false;
	return true;
}

// Diag
#ifdef __CUDACC__

template<int BlockSize, typename TY, typename TA>
__global__ void __cdecl CudaDiag_SparseMatrixCSRKernel(CVector<TY> Y, CSparseMatrixCSR<TA> A) {
	int r = BlockSize*blockIdx.x+threadIdx.x;
	if(r>=A.Height())
		return;
	TA* rowValues;
	unsigned int* rowIndices;
	int rowLength;
	A.GetRow(r,rowValues,rowIndices,rowLength);
	Y[r]=TY(0);
	for(int i=0;i<rowLength;i++){
		if(rowIndices[i]==r){
			Y[r]=rowValues[i];
			break;
		}
	}
}

template<typename TY, typename TA>
void __cdecl CudaDiag_SparseMatrixCSR(CVector<TY> Y, CSparseMatrixCSR<TA> A) {
	const int BlockSize=256;
	int64 numBlocks = DivUp(A.Height(),BlockSize);
	if(numBlocks>32768*2)
		throw "";
	dim3 gridDim((int)numBlocks,1,1);
	dim3 blockDim(BlockSize,1,1);
	CudaDiag_SparseMatrixCSRKernel<BlockSize><<<gridDim,blockDim>>>(Y,A);
}

#else
template<typename TY, typename TA>
void __cdecl CudaDiag_SparseMatrixCSR(CVector<TY> Y, CSparseMatrixCSR<TA> A);
#endif

template<typename T>
static DeviceVector<T> Diag(SparseDeviceMatrixCSR<T> A) {
	Verify(A.Width()==A.Height(),FileAndLine);
	DeviceVector<T> D(A.Height());
	CudaDiag_SparseMatrixCSR(D.GetC(),A.GetC());
	return D;
}
