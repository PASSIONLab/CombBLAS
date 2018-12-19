#pragma once

#include "HostMatrix/CMatrix.h"
#include "HostMatrix/Verify.h"
#include "DeviceMatrix/DeviceVector.h"

template<typename T>
class DeviceMatrix{
	TrackedObject memBlock;
	CMatrix<T> m;
	void Init(int dimX, int dimY){
		DeviceVector<T> tmp(int64(dimX)*int64(dimY));
		memBlock=tmp.GetMemBlock();
		m=CMatrix<T>(tmp.Pointer(),dimX,dimY);
	}
public:
	typedef T Element;
	typedef DeviceVector<T> Vector;
	DeviceMatrix(){}
	void operator=(const DeviceMatrix& rhs){memBlock=rhs.memBlock;m=rhs.m;}
	DeviceMatrix(int dimX,int dimY){Init(dimX,dimY);}
	DeviceMatrix(Int2 size){Init(size.x,size.y);}
	DeviceMatrix(DeviceVector<T> data, Int2 size):memBlock(data.GetMemBlock()),m(data.Data(),size){
		Verify(data.IsSimple() && (int64)size.x*(int64)size.y==data.Length(),FileAndLine);
	}	
	DeviceMatrix(const DeviceMatrix& rhs):m(rhs.m),memBlock(rhs.memBlock){}
	DeviceMatrix(CMatrix<T> m, TrackedObject memBlock):memBlock(memBlock),m(m){}
	void Clear(){*this=DeviceMatrix<T>();}
	int Height()const{return m.Height();}
	int Width()const{return m.Width();}
	int DimX()const{return m.DimX();}
	int DimY()const{return m.DimY();}
	Int2 Size()const{return m.Size();}
	int RowStride()const{return m.RowStride();}
	int Stride()const{return m.Stride();}
	T* Data(){return m.Data();}
	bool IsSimple()const{return m.IsSimple();}
	DeviceVector<T> GetSimple(){
		if(!IsSimple())
			throw std::runtime_error("!Simple");
		return DeviceVector<T>(memBlock,m.GetSimple());
	}
	DeviceVector<T> Row(int r){Verify(r>=0 && r<DimY(),"231r1r1");return DeviceVector<T>(memBlock,m.Row(r));}
	DeviceVector<T> Column(int c){Verify(c>=0 && c<DimX(),"22r2ewq");return DeviceVector<T>(memBlock,m.Column(c));}
	DeviceVector<T> Diagonal(){return DeviceVector<T>(memBlock,m.Diagonal());}
	CMatrix<T> GetCMatrix(){return m;}
	CMatrix<T> GetC(){return m;}
	DeviceMatrix SubMatrix(int startY, int dimY){Verify(startY>=0&&dimY>=0&&startY+dimY<=DimY(),"453dg54d");return DeviceMatrix (m.SubMatrix(startY,dimY),memBlock);}
	DeviceMatrix SubMatrix(int x, int y, int dimX, int dimY){
		Verify(x>=0&&y>=0&&x+dimX<=DimX()&&y+dimY<=DimY(),FileAndLine);
		return DeviceMatrix(m.SubMatrix(x,y,dimX,dimY),memBlock);
	}
	T operator()(int x, int y)const{T tmp;cudaMemcpy(&tmp,m.Data()+m.Index(x,y),sizeof(T),cudaMemcpyDeviceToHost);return tmp;}
	void Set(int x, int y, T value){cudaMemcpy(m.Data()+m.Index(x,y),&value,sizeof(T),cudaMemcpyHostToDevice);}
};