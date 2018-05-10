//#define DETERMINISTIC 1

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


#define EDGEFACTOR 16
#define RAND_PERMUTE 1

#ifdef DETERMINISTIC
MTRand GlobalMT(1);
#else
MTRand GlobalMT;	// generate random numbers with Mersenne Twister
#endif

double cblas_alltoalltime;
double cblas_allgathertime;
double cblas_localspmvtime;
double cblas_mergeconttime;
double cblas_transvectime;



using namespace std;
using namespace combblas;



template <typename PARMAT>
void Symmetricize(PARMAT & A)
{
    // boolean addition is practically a "logical or"
    // therefore this doesn't destruct any links
    PARMAT AT = A;
    AT.Transpose();
    AT.RemoveLoops(); // not needed for boolean matrices, but no harm in keeping it
    A += AT;
}




typedef SpParMat < int64_t, bool, SpDCCols<int64_t,bool> > Par_DCSC_Bool;
typedef SpParMat < int64_t, int64_t, SpDCCols<int64_t, int64_t> > Par_DCSC_int64_t;
typedef SpParMat < int64_t, double, SpDCCols<int64_t, double> > Par_DCSC_Double;
typedef SpParMat < int64_t, bool, SpCCols<int64_t,bool> > Par_CSC_Bool;




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
            cout << "Usage: ./rcm <rmat|er|input> <scale|filename> <splitPerThread>" << endl;
            cout << "Example: mpirun -np 4 ./rcm rmat 20" << endl;
            cout << "Example: mpirun -np 4 ./rcm er 20" << endl;
            cout << "Example: mpirun -np 4 ./rcm input a.mtx" << endl;
            
        }
        MPI_Finalize();
        return -1;
    }
    {
        Par_DCSC_Bool * ABool;
        Par_DCSC_Bool AAT;
        bool unsym=false;
        ostringstream tinfo;
        
        if(string(argv[1]) == string("input")) // input option
        {
            ABool = new Par_DCSC_Bool();
            string filename(argv[2]);
            tinfo.str("");
            tinfo << "**** Reading input matrix: " << filename << " ******* " << endl;
            
            
            SpParHelper::Print(tinfo.str());
            double t01 = MPI_Wtime();
            ABool->ParallelReadMM(filename, false, maximum<bool>());
            double t02 = MPI_Wtime();
            int64_t bw = ABool->Bandwidth();
            int64_t pf = ABool->Profile();
            tinfo.str("");
            tinfo << "Reader took " << t02-t01 << " seconds" << endl;
            tinfo << "Bandwidth before random permutation " << bw << endl;
            tinfo << "Profile before random permutation " << pf << endl;
            SpParHelper::Print(tinfo.str());
            


            
        }
        else if(string(argv[1]) == string("rmat"))
        {
            unsigned scale;
            scale = static_cast<unsigned>(atoi(argv[2]));
            double initiator[4] = {.57, .19, .19, .05};
            DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>();
            DEL->GenGraph500Data(initiator, scale, EDGEFACTOR, true, false );
            MPI_Barrier(MPI_COMM_WORLD);
            
            ABool = new Par_DCSC_Bool(*DEL, false);
            Symmetricize(*ABool);
            delete DEL;
        }
        else if(string(argv[1]) == string("er"))
        {
            unsigned scale;
            scale = static_cast<unsigned>(atoi(argv[2]));
            double initiator[4] = {.25, .25, .25, .25};
            DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>();
            DEL->GenGraph500Data(initiator, scale, EDGEFACTOR, true, false );
            MPI_Barrier(MPI_COMM_WORLD);
            
            ABool = new Par_DCSC_Bool(*DEL, false);
            Symmetricize(*ABool);
            delete DEL;
        }
        else
        {
            SpParHelper::Print("Unknown input option\n");
            MPI_Finalize();
            return -1;
        }
        
        
        
        
        MPI_Comm com = ABool->getcommgrid()->GetWorld();
        double gtime = MPI_Wtime();
        SpParHelper::GatherMatrix(com, ABool->seq(), (int)0);
        if(myrank==0)
        {
            cout << "gathertime " << MPI_Wtime() - gtime << endl;
        }
        
        
        
        
        
       
        delete ABool;

        
    }
    MPI_Finalize();
    return 0;
}

