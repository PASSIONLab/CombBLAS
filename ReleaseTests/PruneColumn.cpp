#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include "CombBLAS/CombBLAS.h"

using namespace combblas;

int main(int argc, char *argv[])
{
    int myrank, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    {
        std::shared_ptr<CommGrid> fullWorld;
        fullWorld.reset(new CommGrid(MPI_COMM_WORLD, 0, 0));

        FullyDistVec<int, int> ri(fullWorld, 5, -1); FullyDistVec<int, int> ci(fullWorld, 5, -1);

        ri.SetElement(0, 1); ri.SetElement(1, 5); ri.SetElement(2, 9); ri.SetElement(3, 6); ri.SetElement(4, 10);
        ci.SetElement(0, 1); ci.SetElement(1, 2); ci.SetElement(2, 2); ci.SetElement(3, 4); ci.SetElement(4, 4);

        SpParMat<int, int, SpDCCols<int, int>> A(13, 5, ri, ci, 1);
        A.ParallelWriteMM("A.mm", false);

        FullyDistSpVec<int, int> ciprune(A.getcommgrid(), 5);
        ciprune.SetElement(2, 2); ciprune.SetElement(3, 0); ciprune.SetElement(4, 2);
        ciprune.DebugPrint();

        A.PruneColumnByIndex(ciprune);

        A.ParallelWriteMM("B.mm", false);
    }


    MPI_Finalize();
    return 0;
}
