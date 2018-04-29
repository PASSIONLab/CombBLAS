#define DETERMINISTIC 1

#ifdef THREADED
#ifndef _OPENMP
#define _OPENMP // should be defined before any COMBBLAS header is included
#endif
#include <omp.h>
#endif

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



typedef SpParMat < int64_t, bool, SpDCCols<int64_t,bool> > Par_DCSC_Bool;
typedef SpParMat < int64_t, int64_t, SpDCCols<int64_t, int64_t> > Par_DCSC_int64_t;
typedef SpParMat < int64_t, double, SpDCCols<int64_t, double> > Par_DCSC_Double;



int main(int argc, char* argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (provided < MPI_THREAD_SERIALIZED)
    {
        printf("ERROR: The MPI library does not have MPI_THREAD_SERIALIZED support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    if(argc < 2)
    {
        if(myrank == 0)
        {
            cout << "Usage: ./rcm <filename>" << endl;
            
        }
        MPI_Finalize();
        return -1;
    }
    {
        Par_DCSC_Bool * ABool;
        ostringstream tinfo;
        
        ABool = new Par_DCSC_Bool();
        string filename(argv[1]);
        tinfo.str("");
        tinfo << "**** Reading input matrix: " << filename << " ******* " << endl;
        SpParHelper::Print(tinfo.str());
        double t01 = MPI_Wtime();
        ABool->ParallelReadMM(filename, true, maximum<bool>());
        double t02 = MPI_Wtime();
        tinfo.str("");
        tinfo << "Reader took " << t02-t01 << " seconds" << endl;
        SpParHelper::Print(tinfo.str());
        

        if(ABool->getnrow() == ABool->getncol())
        {
            FullyDistVec<int64_t, int64_t> p( ABool->getcommgrid());
            p.iota(ABool->getnrow(), 0);
            p.RandPerm();
            (*ABool)(p,p,true);// in-place permute to save memory
            SpParHelper::Print("Applied symmetric permutation.\n");
        }
        else
        {
            SpParHelper::Print("Rectangular matrix: Can not apply symmetric permutation.\n");
        }

        filename += "_permuted";
        ABool->SaveGathered(filename);
	tinfo.str("");
        tinfo << "**** Saved to output matrix: " << filename << " ******* " << endl;
        SpParHelper::Print(tinfo.str());
	delete ABool;
        
    }
    MPI_Finalize();
    return 0;
}

