/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.5 -------------------------------------------------*/
/* date: 10/09/2015 ---------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc, Adam Lugowski ------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2015, The Regents of the University of California
 
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

#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include "CombBLAS/CombBLAS.h"

using namespace std;
using namespace combblas;

int main(int argc, char* argv[])
{
	int nprocs, myrank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

	if(argc < 3)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./IteratorTest <BASEADDRESS> <Matrix>" << endl;
			cout << "Input file <Matrix> should be under <BASEADDRESS> in triples format" << endl;
		}
		MPI_Finalize(); 
		return -1;
	}				
	{
		string directory(argv[1]);		
		string name(argv[2]);
		name = directory+"/"+name;

		typedef SpParMat <int, double, SpDCCols<int,double> > PARMAT;
		shared_ptr<CommGrid> fullWorld;
		fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );

		PARMAT A(fullWorld);
		A.ReadDistribute(name, 0);	// read it from file

		int count = 0;	
		int total = 0;
	
		for(SpDCCols<int,double>::SpColIter colit = A.seq().begcol(); colit != A.seq().endcol(); ++colit)	// iterate over columns
		{
			for(SpDCCols<int,double>::SpColIter::NzIter nzit = A.seq().begnz(colit); nzit != A.seq().endnz(colit); ++nzit)
			{	
				// cout << nzit.rowid() << '\t' << colit.colid() << '\t' << nzit.value() << '\n';	
				count++;
			}
		}	
		MPI_Allreduce( &count, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		
		if(total == A.getnnz())
			SpParHelper::Print( "Iteration passed soft test\n");
		else
			SpParHelper::Print( "Iteration failed !!!\n") ;
		
	}
	MPI_Finalize();
	return 0;
}

