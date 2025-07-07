#include <mpi.h>
#include <sys/time.h>
#include <iostream>
#include <functional>
#include <sstream>
#include "CombBLAS/CombBLAS.h"

using namespace std;
using namespace combblas;

#define EDGEFACTOR 16

int main(int argc, char *argv[]) {
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (argc < 2) {
        if (myrank == 0) {
            cout << "Usage: ./TransposeTest scale" << endl;
        }
        MPI_Finalize();
        return -1;
    } {
        unsigned scale = static_cast<unsigned>(atoi(argv[1]));
        double initiator[4] = {.57, .19, .19, .05};

        double t01 = MPI_Wtime();
        double t02;

        DistEdgeList<int64_t> *DEL = new DistEdgeList<int64_t>();
        DEL->GenGraph500Data(initiator, scale, EDGEFACTOR, true, true); // generate packed edges
        SpParHelper::Print("Generated renamed edge lists\n");
        t02 = MPI_Wtime();
        ostringstream tinfo;
        tinfo << "Generation took " << t02 - t01 << " seconds" << endl;
        SpParHelper::Print(tinfo.str());


        // Start Kernel #1
        MPI_Barrier(MPI_COMM_WORLD);

        // conversion from distributed edge list, keeps self-loops, sums duplicates
        SpParMat<int64_t, double, SpDCCols<int64_t, double> > A(*DEL, false);
        delete DEL; // free memory before symmetricizing
        SpParHelper::Print("Created Sparse Matrix\n");


        SpDCCols<int64_t, double> localA = A.seq();

        // SpDCCols TransposeConst API
        {
            double ftb = MPI_Wtime();
            auto localAT = localA.TransposeConst();
            double fte = MPI_Wtime();
            ostringstream ftrinfo;
            ftrinfo << "SpDCCols TransposeConst took " << fte - ftb << " seconds" << endl;
            SpParHelper::Print(ftrinfo.str());
            localA.DeprecatedTranspose();
            if (localA == localAT) {
                SpParHelper::Print("SpDCCols TransposeConst working correctly\n");
            } else {
                SpParHelper::Print("ERROR!!!\n");
                localA.PrintInfo();
                localAT.PrintInfo();
            }
        }
        // SpDCCols Transpose API
        {
            SpDCCols<int64_t, double> tmpA(localA);
            double ftb = MPI_Wtime();
            localA.Transpose();
            double fte = MPI_Wtime();
            ostringstream ftrinfo;
            ftrinfo << "SpDCCols Transpose took " << fte - ftb << " seconds" << endl;
            SpParHelper::Print(ftrinfo.str());
            tmpA.DeprecatedTranspose();
            if (localA == tmpA) {
                SpParHelper::Print("SpDCCols Transpose working correctly\n");
            } else {
                SpParHelper::Print("ERROR!!!\n");
                localA.PrintInfo();
            }
        }
        // SpDCCols TransposeConstPtr API
        {
            double ftb = MPI_Wtime();
            auto localAT = localA.TransposeConstPtr();
            double fte = MPI_Wtime();
            ostringstream ftrinfo;
            ftrinfo << "SpDCCols TransposeConst took " << fte - ftb << " seconds" << endl;
            SpParHelper::Print(ftrinfo.str());
            localA.DeprecatedTranspose();
            if (localA == *localAT) {
                SpParHelper::Print("SpDCCols TransposeConst working correctly\n");
            } else {
                SpParHelper::Print("ERROR!!!\n");
                localA.PrintInfo();
                localAT->PrintInfo();
            }
            delete localAT;
        }

    }
    MPI_Finalize();
    return 0;
}
