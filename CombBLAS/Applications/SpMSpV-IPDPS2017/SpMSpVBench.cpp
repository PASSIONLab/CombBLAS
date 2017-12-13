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

#include "CombBLAS/CombBLAS.h"
#include <mpi.h>
#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>

using namespace combblas;

#ifdef _OPENMP
int cblas_splits = omp_get_max_threads();
#else
int cblas_splits = 1;
#endif

double cblas_alltoalltime;
double cblas_allgathertime;
double cblas_mergeconttime;
double cblas_transvectime;
double cblas_localspmvtime;


#define EDGEFACTOR 16
int ITERS;
using namespace std;


typedef SelectMaxSRing<bool, int32_t> SR;
typedef SpParMat < int64_t, bool, SpDCCols<int64_t,bool> > PSpMat_Bool;
typedef SpParMat < int64_t, bool, SpDCCols<int32_t,bool> > PSpMat_s32p64;	// sequentially use 32-bits for local matrices, but parallel semantics are 64-bits
typedef SpParMat < int64_t, int, SpDCCols<int32_t,int> > PSpMat_s32p64_Int;	// similarly mixed, but holds integers as upposed to booleans
typedef SpParMat < int64_t, int64_t, SpDCCols<int64_t,int64_t> > PSpMat_Int64;
typedef SpParMat < int64_t, bool, SpCCols<int64_t,bool> > Par_CSC_Bool;


template <typename PARMAT>
void Symmetricize(PARMAT & A)
{
    // boolean addition is practically a "logical or"
    // therefore this doesn't destruct any links
    PARMAT AT = A;
    AT.Transpose();
    A += AT;
}



struct SelectMinSR
{
    typedef int64_t T_promote;
    static T_promote id(){ return -1; };
    static bool returnedSAID() { return false; }
    //static MPI_Op mpi_op() { return MPI_MIN; };
    
    static T_promote add(const T_promote & arg1, const T_promote & arg2)
    {
        return std::min(arg1, arg2);
    }
    
    static T_promote multiply(const bool & arg1, const T_promote & arg2)
    {
        return arg2;
    }
    
    static void axpy(bool a, const T_promote & x, T_promote & y)
    {
        y = std::min(y, x);
    }
};




void BFS_CSC(PSpMat_s32p64 Aeff, int64_t source, FullyDistVec<int64_t, int64_t> degrees)
{
    
    int nthreads=1;
#ifdef THREADED
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
#endif

    Par_CSC_Bool ABoolCSC (Aeff);
    PreAllocatedSPA<int64_t> SPA(ABoolCSC.seq(), nthreads*4);
    
    double tspmvall=0, tall=0;
    int iterall = 0;
    int visitedE = 0;
    int visitedV = 0;
    
    
    for(int i=0; i<ITERS; ++i)
    {
        // FullyDistVec ( shared_ptr<CommGrid> grid, IT globallen, NT initval);
        FullyDistVec<int64_t, int64_t> parents ( Aeff.getcommgrid(), Aeff.getncol(), (int64_t) -1);	// identity is -1
        
        // FullyDistSpVec ( shared_ptr<CommGrid> grid, IT glen);
        FullyDistSpVec<int64_t, int64_t> fringe(Aeff.getcommgrid(), Aeff.getncol());	// numerical values are stored 0-based
        
        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();
        
        fringe.SetElement(source, source);
        int iterations = 0;
        
        while(fringe.getnnz() > 0)
        {
            int64_t xnnz = fringe.getnnz();
            fringe.setNumToInd();
            double tstart = MPI_Wtime();
            SpMV<SelectMinSR>(ABoolCSC, fringe, fringe, false, SPA);
            double tspmv = MPI_Wtime()-tstart;
            tspmvall += tspmv;
            int64_t ynnz = fringe.getnnz();
            fringe = EWiseMult(fringe, parents, true, (int64_t) -1);	// clean-up vertices that already has parents

            ostringstream outs1;
            outs1 << "iteration: " << iterations << " xnnz: "<< xnnz << " ynnz: " << ynnz << " SpMSpV time: " << tspmv << endl;
            SpParHelper::Print(outs1.str());
            
            parents.Set(fringe);
            iterations++;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        double t2 = MPI_Wtime();
        tall += (t2 - t1);
        
        FullyDistSpVec<int64_t, int64_t> parentsp = parents.Find(bind2nd(greater<int64_t>(), -1));
        parentsp.Apply(myset<int64_t>(1));
        
        // we use degrees on the directed graph, so that we don't count the reverse edges in the teps score
        int64_t nedges = EWiseMult(parentsp, degrees, false, (int64_t) 0).Reduce(plus<int64_t>(), (int64_t) 0);
        
        visitedE += nedges;
        visitedV += parentsp.Reduce(plus<int64_t>(), (int64_t) 0);
        iterall += iterations;
    }
    
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    if(myrank == 0)
    {
        cout << "\nOverall stats:"  << endl;
        cout << "  starting vertex: " << source << endl;
        cout << "  Avg number iterations: " << iterall/ITERS << endl;
        cout << "  Avg number of vertices found: " << visitedV/ITERS << endl;
        cout << "  Avg Number of edges traversed: " << visitedE/ITERS << endl;
        cout << "  Avg SpMSpV time: " << tspmvall/ITERS << endl;
        cout << "  Avg Total time: " << tall/ITERS << endl;
    }

}


void BFS_DCSC(PSpMat_s32p64 Aeff1, int64_t source, FullyDistVec<int64_t, int64_t> degrees)
{
    
    PSpMat_Bool Aeff = Aeff1;
    OptBuf<int32_t, int64_t> optbuf;	// let indices be 32-bits
    //Aeff.OptimizeForGraph500(optbuf);
    int nthreads=1;
#ifdef THREADED
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
        cblas_splits = nthreads*4;
    }
    Aeff.ActivateThreading(cblas_splits);
#endif
    
    double tspmvall=0, tall=0;
    int iterall = 0;
    int visitedE = 0;
    int visitedV = 0;
    
    
    for(int i=0; i<ITERS; ++i)
    {
        // FullyDistVec ( shared_ptr<CommGrid> grid, IT globallen, NT initval);
        FullyDistVec<int64_t, int64_t> parents ( Aeff.getcommgrid(), Aeff.getncol(), (int64_t) -1);	// identity is -1
        
        // FullyDistSpVec ( shared_ptr<CommGrid> grid, IT glen);
        FullyDistSpVec<int64_t, int64_t> fringe(Aeff.getcommgrid(), Aeff.getncol());	// numerical values are stored 0-based
        
        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();
        
        fringe.SetElement(source, source);
        int iterations = 0;
        
        while(fringe.getnnz() > 0)
        {
            int64_t xnnz = fringe.getnnz();
            fringe.setNumToInd();
            double tstart = MPI_Wtime();
            SpMV<SelectMinSR>(Aeff, fringe, fringe, false);
            double tspmv = MPI_Wtime()-tstart;
            tspmvall += tspmv;
            int64_t ynnz = fringe.getnnz();
            fringe = EWiseMult(fringe, parents, true, (int64_t) -1);	// clean-up vertices that already has parents
            
            ostringstream outs1;
            outs1 << "iteration: " << iterations << " xnnz: "<< xnnz << " ynnz: " << ynnz << " SpMSpV time: " << tspmv << endl;
            SpParHelper::Print(outs1.str());
            

            parents.Set(fringe);
            iterations++;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        double t2 = MPI_Wtime();
        tall += (t2 - t1);
        
        FullyDistSpVec<int64_t, int64_t> parentsp = parents.Find(bind2nd(greater<int64_t>(), -1));
        parentsp.Apply(myset<int64_t>(1));
        
        // we use degrees on the directed graph, so that we don't count the reverse edges in the teps score
        int64_t nedges = EWiseMult(parentsp, degrees, false, (int64_t) 0).Reduce(plus<int64_t>(), (int64_t) 0);
        
        visitedE += nedges;
        visitedV += parentsp.Reduce(plus<int64_t>(), (int64_t) 0);
        iterall += iterations;
    }
    
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    if(myrank == 0)
    {
        cout << "\nOverall stats:" << endl;
        cout << "  starting vertex: " << source << endl;
        cout << "  Avg number iterations: " << iterall/ITERS << endl;
        cout << "  Avg number of vertices found: " << visitedV/ITERS << endl;
        cout << "  Avg Number of edges traversed: " << visitedE/ITERS << endl;
        cout << "  Avg SpMSpV time: " << tspmvall/ITERS << endl;
        cout << "  Avg Total time: " << tall/ITERS << endl;
    }

    
}




void BFS_CSC_Split(PSpMat_s32p64 Aeff, int64_t source, FullyDistVec<int64_t, int64_t> degrees)
{
    int nthreads=1;
    
    
    Par_CSC_Bool ABoolCSC(Aeff);

    

    
#ifdef THREADED
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
        cblas_splits = nthreads*4;
    }
    ABoolCSC.ActivateThreading(cblas_splits);
#endif
    
    double tspmvall=0, tall=0;
    int iterall = 0;
    int visitedE = 0;
    int visitedV = 0;
    
    
    for(int i=0; i<ITERS; ++i)
    {
        // FullyDistVec ( shared_ptr<CommGrid> grid, IT globallen, NT initval);
        FullyDistVec<int64_t, int64_t> parents ( Aeff.getcommgrid(), Aeff.getncol(), (int64_t) -1);	// identity is -1
        
        // FullyDistSpVec ( shared_ptr<CommGrid> grid, IT glen);
        FullyDistSpVec<int64_t, int64_t> fringe(Aeff.getcommgrid(), Aeff.getncol());	// numerical values are stored 0-based
        
        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();
        
        fringe.SetElement(source, source);
        int iterations = 0;
        
        while(fringe.getnnz() > 0)
        {
            int64_t xnnz = fringe.getnnz();
            fringe.setNumToInd();
            double tstart = MPI_Wtime();
            SpMV<SelectMinSR>(ABoolCSC, fringe, fringe, false);
            double tspmv = MPI_Wtime()-tstart;
            tspmvall += tspmv;
            int64_t ynnz = fringe.getnnz();
            fringe = EWiseMult(fringe, parents, true, (int64_t) -1);	// clean-up vertices that already has parents
            ostringstream outs1;
            outs1 << "iteration: " << iterations << " xnnz: "<< xnnz << " ynnz: " << ynnz << " SpMSpV time: " << tspmv << endl;
            SpParHelper::Print(outs1.str());
            
            parents.Set(fringe);
            iterations++;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        double t2 = MPI_Wtime();
        tall += (t2-t1);
        
        FullyDistSpVec<int64_t, int64_t> parentsp = parents.Find(bind2nd(greater<int64_t>(), -1));
        parentsp.Apply(myset<int64_t>(1));
        
        // we use degrees on the directed graph, so that we don't count the reverse edges in the teps score
        int64_t nedges = EWiseMult(parentsp, degrees, false, (int64_t) 0).Reduce(plus<int64_t>(), (int64_t) 0);
        
        visitedE += nedges;
        visitedV += parentsp.Reduce(plus<int64_t>(), (int64_t) 0);
        iterall += iterations;
    }
    
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    if(myrank == 0)
    {
        cout << "\nOverall stats:" << endl;
        cout << "  starting vertex: " << source << endl;
        cout << "  Avg number iterations: " << iterall/ITERS << endl;
        cout << "  Avg number of vertices found: " << visitedV/ITERS << endl;
        cout << "  Avg Number of edges traversed: " << visitedE/ITERS << endl;
        cout << "  Avg SpMSpV time: " << tspmvall/ITERS << endl;
        cout << "  Avg Total time: " << tall/ITERS << endl;
        

    }
    
    
}




int main(int argc, char* argv[])
{
    int nprocs, myrank;

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (provided < MPI_THREAD_SERIALIZED)
    {
        printf("ERROR: The MPI library does not have MPI_THREAD_SERIALIZED support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    if(argc < 3)
    {
        if(myrank == 0)
        {
            cout << "Usage: ./SpMSpVBench <-input|-rmat|-er> <scale|filename> " << endl;
            cout << "    optional parameters:" << endl;
            cout << "       -source \"source of BFS\" (default: 0) " << endl;
            cout << "       -iter \"number of BFS iterations\" (default: 1)" << endl;
            cout << "Example with a user supplied matrix:" << endl;
            cout << "    mpirun -np 4 ./SpMSpVBench -input a.mtx -source 2" << endl;
            cout << "Example with a user supplied matrix (pre-permute the input matrix for load balance):" << endl;
            cout << "    mpirun -np 4 ./SpMSpVBench -input a.mtx -permute" << endl;
            cout << "Example with RMAT matrix: mpirun -np 4 ./SpMSpVBench -rmat 18" << endl;
            cout << "Example with an Erdos-Renyi matrix: mpirun -np 4 ./SpMSpVBench -er 18" << endl;
        }
        MPI_Finalize();
        return -1;
    }
    {
        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );
        int nthreads=1;
        
#ifdef THREADED
#pragma omp parallel
        {
            nthreads = omp_get_num_threads();
        }
#endif
        
        
        // Declare objects
        PSpMat_Bool A(fullWorld);
        PSpMat_s32p64 Aeff(fullWorld);
        FullyDistVec<int64_t, int64_t> degrees(fullWorld);	// degrees of vertices (including multi-edges and self-loops)
        FullyDistVec<int64_t, int64_t> nonisov(fullWorld);	// id's of non-isolated (connected) vertices
        unsigned scale;
        bool scramble = false;
        int source = 0;
        ITERS = 1;
        bool randpermute = false;
        bool symm = false;
        int maxthreads = nthreads;
        int minthreads = nthreads;
        string filename(argv[2]);
        for (int i = 1; i < argc; i++)
        {
            if (strcmp(argv[i],"-permute")==0)
            {
                if(myrank == 0) cout << "Randomly permute the matrix " << endl;
                randpermute = true;
            }
            if (strcmp(argv[i],"-source")==0)
            {
                source = atoi(argv[i + 1]);
                if(myrank == 0) cout << "Source vertex: " << source << endl;
            }
            if (strcmp(argv[i],"-iter")==0)
            {
                ITERS = atoi(argv[i + 1]);
                if(myrank == 0) cout << "Number of iterations: " << ITERS << endl;
            }
        }
        
        
        
        if(string(argv[1]) == string("-input")) // input option
        {
            
            //A.ReadDistribute(string(argv[2]), 0);	// read it from file
            A.ParallelReadMM(filename, true, maximum<bool>());
            SpParHelper::Print("Read input\n");
            
            PSpMat_Int64 * G = new PSpMat_Int64(A);
            G->Reduce(degrees, Row, plus<int64_t>(), static_cast<int64_t>(0));	// identity is 0
            delete G;
            
            Symmetricize(A);
           
            FullyDistVec<int64_t, int64_t> * ColSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid());
            A.Reduce(*ColSums, Column, plus<int64_t>(), static_cast<int64_t>(0)); 	// plus<int64_t> matches the type of the output vector
            nonisov = ColSums->FindInds(bind2nd(greater<int64_t>(), 0));	// only the indices of non-isolated vertices
            delete ColSums;
            
            if(randpermute)
                nonisov.RandPerm();
            A(nonisov, nonisov, true);	// in-place permute to save memory
            degrees = degrees(nonisov);	// fix the degrees array too
            FullyDistVec<int64_t, int64_t> newsource = nonisov.FindInds(bind2nd(equal_to<int64_t>(), source));
            source = newsource.GetElement(0);
            degrees = degrees(nonisov);	// fix the source vertex too
            Aeff = PSpMat_s32p64(A);
            A.FreeMemory();
        }
        else if(string(argv[1]) == string("-rmat"))
        {
            unsigned scale;
            scale = static_cast<unsigned>(atoi(argv[2]));
            double initiator[4] = {.57, .19, .19, .05};
            DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>();
            DEL->GenGraph500Data(initiator, scale, EDGEFACTOR, true, false );
            MPI_Barrier(MPI_COMM_WORLD);
            
            A = PSpMat_Bool(*DEL, false);
            delete DEL;
            Symmetricize(A);
            Aeff = PSpMat_s32p64(A);
            Aeff.Reduce(degrees, Row, plus<int64_t>(), static_cast<int64_t>(0));	// identity is 0
            A.FreeMemory();
            

        }
        else if(string(argv[1]) == string("-er"))
        {
            unsigned scale;
            scale = static_cast<unsigned>(atoi(argv[2]));
            double initiator[4] = {.25, .25, .25, .25};
            DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>();
            DEL->GenGraph500Data(initiator, scale, EDGEFACTOR, true, false );
            MPI_Barrier(MPI_COMM_WORLD);
            
            A = PSpMat_Bool(*DEL, false);
            delete DEL;
            Symmetricize(A);
            Aeff = PSpMat_s32p64(A);
            Aeff.Reduce(degrees, Row, plus<int64_t>(), static_cast<int64_t>(0));	// identity is 0
            A.FreeMemory();
        }
        else
        {
            SpParHelper::Print("Unknown input option\n");
            MPI_Finalize();
            return -1;
        }
        
        
        
        
        Aeff.PrintInfo();
        float balance = Aeff.LoadImbalance();
        ostringstream outs;
        outs << "Load balance: " << balance << endl;
        SpParHelper::Print(outs.str());
        
        MPI_Barrier(MPI_COMM_WORLD);
      
        
        SpParHelper::Print("-------------------------------------------------\n");
        SpParHelper::Print ("BFS With CSC matrix and SpMSpV-bucket algorithm \n");
        SpParHelper::Print("-------------------------------------------------\n");
        BFS_CSC(Aeff, source, degrees);
        SpParHelper::Print("-------------------------------------------------\n");
        SpParHelper::Print ("BFS With Split CSC matrix and SpMSpV-heapsort algorithm \n");
        SpParHelper::Print("-------------------------------------------------\n");
        BFS_CSC_Split(Aeff, source, degrees);
        SpParHelper::Print("-------------------------------------------------\n");
        SpParHelper::Print ("BFS With DCSC matric and SpMSpV-SPA algorithm \n");
        SpParHelper::Print("-------------------------------------------------\n");
        BFS_DCSC(Aeff, source, degrees);
        
    }
    MPI_Finalize();
    return 0;
    
}
