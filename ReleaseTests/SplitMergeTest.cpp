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
#define NSPLITS 5

int main(int argc, char* argv[])
{
	int nprocs, myrank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

    {
        typedef SpParMat < int64_t, int, SpDCCols<int32_t,int> > PSpMat_s32p64_Int;
        
        double initiator[4] = {.57, .19, .19, .05};
        DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>();
        DEL->GenGraph500Data(initiator, 12, 8, true, true );	// generate packed edges
        SpParHelper::Print("Generated renamed edge lists\n");
        
        // conversion from distributed edge list, keeps self-loops, sums duplicates
        PSpMat_s32p64_Int G(*DEL, false);
        delete DEL;	// free memory before symmetricizing
        SpParHelper::Print("Created Sparse Matrix (with int32 local indices and values)\n");
        
        SpDCCols<int32_t,int> G_perproc = G.seq();
        G_perproc.PrintInfo();
        
        SpDCCols<int32_t,int> G_perproc_copy = G_perproc;
        vector< SpDCCols<int32_t,int> > splits(NSPLITS);
        G_perproc.ColSplit(NSPLITS, splits);
        
        SpDCCols<int32_t,int> G_perproc_reconstructed;
        G_perproc_reconstructed.ColConcatenate(splits);
        
        G_perproc_reconstructed.PrintInfo();
        
        if(G_perproc_copy == G_perproc_reconstructed)
        {
            cout << "ColSplit/ColConcatenate works" << endl;
        }
        else
        {
            cout << "ERROR" << endl;
        }
    }
	MPI_Finalize();
	return 0;
}
