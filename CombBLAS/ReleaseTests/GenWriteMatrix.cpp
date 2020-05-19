/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.6 -------------------------------------------------*/
/* date: 6/15/2017 ---------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc  --------------------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2017, The Regents of the University of California
 
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

#include "CombBLAS/CombBLAS.h"
#include <mpi.h>
#include <sys/time.h> 
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>

using namespace std;
using namespace combblas;


template <class T>
bool from_string(T & t, const string& s, std::ios_base& (*f)(std::ios_base&))
{
        istringstream iss(s);
        return !(iss >> f >> t).fail();
}


template <typename PARMAT>
void Symmetricize(PARMAT & A)
{
	// boolean addition is practically a "logical or"
	// therefore this doesn't destruct any links
	PARMAT AT = A;
	AT.Transpose();
	A += AT;
}


int main(int argc, char* argv[])
{
	int nprocs, myrank;
#ifdef _OPENMP
    	int provided, flag, claimed;
    	MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided );
    	MPI_Is_thread_main( &flag );
    	if (!flag)
        	SpParHelper::Print("This thread called init_thread but Is_thread_main gave false\n");
    	MPI_Query_thread( &claimed );
    	if (claimed != provided)
        	SpParHelper::Print("Query thread gave different thread level than requested\n");
#else
	MPI_Init(&argc, &argv);
#endif
    
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
	if(argc < 4)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./genwritemat <Scale> <Edgefactor> <Symmetricize> <outputname>" << endl;
			cout << "Example: ./genwritemat 25 16 1 scale25_ef16_symmetric.mtx" << endl;
		}
		MPI_Finalize();
		return -1;
	}		
	{
        unsigned scale = static_cast<unsigned>(atoi(argv[1]));
        unsigned edgefactor = static_cast<unsigned>(atoi(argv[2]));
        int symmetric = static_cast<unsigned>(atoi(argv[3]));


        double initiator[4] = {.57, .19, .19, .05};

        double t01 = MPI_Wtime();
        double t02;
        DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>();
        DEL->GenGraph500Data(initiator, scale, edgefactor, true, true );    // generate packed edges
        SpParHelper::Print("Generated renamed edge lists\n");
        t02 = MPI_Wtime();
        ostringstream tinfo;
        tinfo << "Generation took " << t02-t01 << " seconds" << endl;
        SpParHelper::Print(tinfo.str());
        

        typedef SpParMat < int64_t, int, SpDCCols<int32_t,int> > PSpMat_s32p64_Int;    // use 32-bits for local matrices, but parallel semantics are 64-bits
        PSpMat_s32p64_Int G(*DEL, false);         // conversion from distributed edge list, keeps self-loops, sums duplicates
        delete DEL;    // free memory before symmetricizing
        SpParHelper::Print("Created Sparse Matrix (with int32 local indices and values)\n");
    
        int64_t removed  = G.RemoveLoops();
        ostringstream loopinfo;
        loopinfo << "Removed " << removed << " loops" << endl;
        SpParHelper::Print(loopinfo.str());
        G.PrintInfo();
        
        if(symmetric)
        {
            Symmetricize(G);
            SpParHelper::Print("Symmetricized\n");
        }
        
        float balance = G.LoadImbalance();
        ostringstream outs;
        outs << "Load balance: " << balance << endl;
        SpParHelper::Print(outs.str());
        
        G.ParallelWriteMM(string(argv[4]), true);   // write one-based
	}
	MPI_Finalize();
	return 0;
}

