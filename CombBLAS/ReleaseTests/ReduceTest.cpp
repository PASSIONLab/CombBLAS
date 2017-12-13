/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.6 -------------------------------------------------*/
/* date: 11/15/2016 --------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc, Adam Lugowski ------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2016, The Regents of the University of California
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */

#include <mpi.h>
#include <sys/time.h> 
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include "CombBLAS/CombBLAS.h"

using namespace std;
using namespace combblas;

// Simple helper class for declarations: Just the numerical type is templated 
// The index type and the sequential matrix type stays the same for the whole code
// In this case, they are "int" and "SpDCCols"
template <class NT>
class PSpMat 
{ 
public: 
	typedef SpDCCols < int, NT > DCCols;
	typedef SpParMat < int, NT, DCCols > MPI_DCCols;
};

int main(int argc, char* argv[])
{
	int nprocs, myrank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

	if(argc < 4)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./ReduceTest <MatrixA> <SumColumns> <SumRows>" << endl;
			cout << "<Matrix>,<SumColumns>,<SumRows> are absolute addresses, and files should be in triples format" << endl;
		}
		MPI_Finalize(); 
		return -1;
	}				
	{
		string Aname(argv[1]);		
		string Bname(argv[2]);
		string Cname(argv[3]);

		ifstream inputB(Bname.c_str());
		ifstream inputC(Cname.c_str());
		MPI_Barrier(MPI_COMM_WORLD);

		shared_ptr<CommGrid> fullWorld;
		fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );
		
		PSpMat<double>::MPI_DCCols A(fullWorld);
		FullyDistVec<int,double> colsums(A.getcommgrid());
		FullyDistVec<int,double> rowsums(A.getcommgrid());

		A.ReadDistribute(Aname, 0);
		colsums.ReadDistribute(inputB, 0);
		rowsums.ReadDistribute(inputC, 0);
		
        FullyDistVec< int, double > rowsums_control(fullWorld);
        FullyDistVec< int, double > colsums_control(fullWorld);
		A.Reduce(rowsums_control, Row, std::plus<double>() , 0.0);
		A.Reduce(colsums_control, Column, std::plus<double>() , 0.0);
		
		if (rowsums_control == rowsums && colsums_control == colsums)
		{
			SpParHelper::Print("Reduction via summation working correctly\n");	
		}
		else
		{
			SpParHelper::Print("ERROR in Reduce via summation, go fix it!\n");	
		}

		inputB.clear();
		inputB.close();
		inputC.clear();
		inputC.close();
	}
	MPI_Finalize();
	return 0;
}


