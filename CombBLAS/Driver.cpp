#include <iostream>
#include <fstream>
#include "SpDCCols.h"

using namespace std;

int main()
{
	ifstream inputA("matrixA.txt");
	ifstream inputB("matrixB.txt");
	if(!(inputA.is_open() && inputB.is_open()))
	{
		cerr << "One of the input files doesn't exist\n";
		exit(-1);
	}

	int m,n,nnz;
	inputA >> m >> n >> nnz;

	SpTuples<int,double> tuplesA(nnz,m,n);
	inputA >> tuplesA;
	tuplesA.SortColBased();
	
	SpDCCols<int,double> dcolsA(tuplesA, false, NULL);

	inputB >> m >> n >> nnz;

	SpTuples<int,double> tuplesB(nnz,m,n);
	inputB >> tuplesB;
	tuplesB.SortColBased();
	
	SpDCCols<int,double> dcolsB(tuplesB, false, NULL);		

	
}

