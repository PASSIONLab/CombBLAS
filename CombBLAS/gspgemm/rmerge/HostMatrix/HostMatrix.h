#pragma once

#include "HostMatrix/CMatrix.h"
#include "HostMatrix/HostVector.h"
#include "HostMatrix/Verify.h"

template<typename T>
class HostMatrix{
public:	
	CMatrix<T> m;
	TrackedObject memBlock;
public:
	typedef T Element;
	typedef HostVector<T> Vector;
	void operator=(const HostMatrix& rhs){memBlock=rhs.memBlock;m=rhs.m;}
	HostMatrix():m(0,0,0){}
	HostMatrix(HostVector<T> data, int dimX, int dimY):m(data.Data(),dimX,dimY),memBlock(data.GetMemBlock()){
		Verify(data.IsSimple() && (int64)dimX*(int64)dimY==data.Length(),"08230980987");
	}
	HostMatrix(HostVector<T> data, Int2 size):memBlock(data.GetMemBlock()),m(data.Data(),size.x,size.y){
		Verify(data.IsSimple() && (int64)size.x*(int64)size.y==data.Length(),"v0943098098");
	}

	HostMatrix(int dimX,int dimY):m(new T[dimX*(int64)dimY],dimX,dimY),memBlock(new HostMemBlock<T>(m.Data())){}
	HostMatrix(Int2 size):m(new T[(int64)size.x*(int64)size.y],size.x,size.y),memBlock(new HostMemBlock<T>(m.Data())){}
	HostMatrix(const HostMatrix& rhs):m(rhs.m),memBlock(rhs.memBlock){}
	HostMatrix(CMatrix<T> m, TrackedObject memBlock):m(m),memBlock(memBlock){}		
	bool IsSimple()const{return m.IsSimple();}
	HostVector<T> GetSimple(){
		Verify(IsSimple(),"!Simple");
		return HostVector<T>(m.GetSimple(),memBlock);	
	}
	T* Data(){return m.Data();}
	void Clear(){*this=HostMatrix<T>();}
	bool IsEmpty()const{return DimX()==0 && DimY()==0;}
	const T& operator()(int x, int y)const{return m(x,y);}
	T& operator()(int x, int y){return m(x,y);}
	T& operator()(Int2 pos){return m(pos);}
	int Height()const{return m.Height();}
	int Width()const{return m.Width();}
	int Stride()const{return m.Stride();}
	int RowStride()const{return m.Stride();}
	int DimX()const{return m.DimX();}
	int DimY()const{return m.DimY();}
	Int2 Size()const{return m.Size();}
	HostVector<T> Row(int r){Verify(r>=0 && r<DimY(),"Out of bounds 1h1ohoih");return HostVector<T>(m.Row(r),memBlock);}
	T* RowPointer(int r){Verify(r>=0 && r<DimY(),"wt32tsxydgs");return m.RowPointer(r);}
	T* RowPointerX(int y){Verify(y>=0 && y<DimY(),"498574978");return m.RowPointer(y);}
	T* RowPointerY(int x){Verify(x>=0 && x<DimX(),"v5n3876836487");return m.RowPointerY(x);}
	HostVector<T> Column(int c){Verify(c>=0 && c<DimX(),"09238dd112d");return HostVector<T>(m.Column(c),memBlock);}
	HostVector<T> Diagonal(){return HostVector<T>(m.Diagonal(),memBlock);}
	HostMatrix SubMatrix(int startY, int dimY){Verify(startY>=0&&dimY>=0&&startY+dimY<=DimY(),"e4tdehddfdsfgd");return HostMatrix(m.SubMatrix(startY,dimY),memBlock);}
	HostMatrix SubMatrix(int x, int y, int dimX, int dimY){Verify(x>=0&&x+dimX<=DimX()&&y>=0&&y+dimY<=DimY(),"aouiuiuiuie"); return HostMatrix(m.SubMatrix(x,y,dimX,dimY),memBlock);}
	CMatrix<T> GetC(){return m;}
	T* Pointer(){return m.Data();}
};
