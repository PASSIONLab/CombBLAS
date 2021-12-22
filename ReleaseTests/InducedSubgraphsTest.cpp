#include <mpi.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include "CombBLAS/CombBLAS.h"

int main(int argc, char *argv[])
{
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (argc < 3) {
        if (!myrank)
            std::cerr << "Usage: ./Subgraphs2ProcsTest <MatrixA> <VectorAssignments>" << std::endl;
        MPI_Finalize();
        return -1;
    }
    {
        if (!myrank) std::cerr << "processor grid: (" << std::sqrt(nprocs) << " x " << std::sqrt(nprocs) << ")" << std::endl;

        std::shared_ptr<combblas::CommGrid> fullWorld;
        fullWorld.reset(new combblas::CommGrid(MPI_COMM_WORLD, 0, 0));

        combblas::SpParMat<int, double, combblas::SpCCols<int, double> > A(fullWorld);
        combblas::FullyDistVec<int, int> assignments(A.getcommgrid());

        A.ParallelReadMM(std::string(argv[1]), true, combblas::maximum<double>());
        assignments.ParallelRead(std::string(argv[2]), true, combblas::maximum<int>());

        std::vector<int> local_idx_map;

        combblas::SpCCols<int, double> locmat = A.InducedSubgraphs2Procs(assignments, local_idx_map);

        for (auto colit = locmat.begcol(); colit != locmat.endcol(); ++colit) {
            for (auto nzit = locmat.begnz(colit); nzit != locmat.endnz(colit); ++nzit) {
                std::cout << myrank << ": " << local_idx_map[nzit.rowid()]+1 << "\t" << local_idx_map[colit.colid()]+1 << "\t" << nzit.value() << std::endl;
            }
        }
        std::cout << std::endl;

    }

    MPI_Finalize();
    return 0;
}
