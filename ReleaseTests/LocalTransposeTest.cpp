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

#define EDGEFACTOR 16

int main(int argc, char* argv[])
{
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

    if(argc < 2)
    {
        if(myrank == 0)
        {
            cout << "Usage: ./TransposeTest scale" << endl;
        }
        MPI_Finalize();
        return -1;
    }
    {
        
        unsigned scale = scale = static_cast<unsigned>(atoi(argv[1]));
        double initiator[4] = {.57, .19, .19, .05};

        double t01 = MPI_Wtime();
        double t02;
        
        DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>();
        DEL->GenGraph500Data(initiator, scale, EDGEFACTOR, true, true );    // generate packed edges
        SpParHelper::Print("Generated renamed edge lists\n");
        t02 = MPI_Wtime();
        ostringstream tinfo;
        tinfo << "Generation took " << t02-t01 << " seconds" << endl;
        SpParHelper::Print(tinfo.str());
    

        // Start Kernel #1
        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        // conversion from distributed edge list, keeps self-loops, sums duplicates
        SpParMat<int64_t, double, SpDCCols<int64_t,double>> A(*DEL, false);
        delete DEL;    // free memory before symmetricizing
        SpParHelper::Print("Created Sparse Matrix\n");

	
	SpDCCols<int64_t,double> localA = A.seq(); 

	double ftb = MPI_Wtime();
	auto localAT  = localA.TransposeConst();
	double fte = MPI_Wtime();
	ostringstream ftrinfo;
        ftrinfo << "Fast transpose took " << fte-ftb << " seconds" << endl;
        SpParHelper::Print(ftrinfo.str());

	
	localA.Transpose();       
 
        if (localA == localAT)
        {
            SpParHelper::Print("Transpose working correctly\n");
        }
        else
        {
            SpParHelper::Print("ERROR in matrix transpose!\n");
        }
    }
    MPI_Finalize();
    return 0;
}

