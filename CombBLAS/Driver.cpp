#include <iostream>
#include <fstream>

#include "SpTuples.h"
#include "SpDCCols.h"

using namespace std;

int main()
{
	ifstream inputa("matrixA.txt");
	ifstream inputb("matrixB.txt");
	if(!(inputa.is_open() && inputb.is_open()))
	{
		cerr << "One of the input files doesn't exist\n";
		exit(-1);
	}
	
	SpDCCols<int,double> A;
	inputa >> A;
	A.PrintInfo();
	
	SpDCCols<int,double> B;
	inputb >> B;		
	B.PrintInfo();

	// arguments (in this order): nnz, n, m, nzc
	SpDCCols<int,double> C(0, A.getnrow(), B.getncol(), 0);
	C.PrintInfo();

	typedef PlusTimesSRing<double, double> PT;	
	C.SpGEMM <PT> (A, B, false, false);	// C = A*B
	C.PrintInfo();
}
