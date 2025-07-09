#include <mpi.h>
#include <iostream>
#include <algorithm>
#include <functional>
#include <string>
#include <vector>
#include "CombBLAS/CombBLAS.h"

using namespace combblas;

template <class IT>
struct KTipsSR
{
    static IT id() { return static_cast<IT>(0); }
    static bool returnedSAID() { return false; }
    static MPI_Op mpi_op() { return MPI_LOR; }
    static IT add(const IT& arg1, const IT& arg2) { return (arg1 || arg2); }
    static IT multiply(const IT& arg1, const IT& arg2) { return (arg1 && arg2); }
    static void axpy(IT a, const IT& x, IT& y) { y = add(y, multiply(a, x)); }
};

template <class IT, class NT, class DER>
FullyDistVec<IT,IT> LastNzRowIdxPerCol(const SpParMat<IT,NT,DER>& A)
{
    std::shared_ptr<CommGrid> grid = A.getcommgrid();
    int myrank = grid->GetRank();
    int myproccol = grid->GetRankInProcRow();
    int myprocrow = grid->GetRankInProcCol();

    MPI_Comm ColWorld = grid->GetColWorld();

    IT total_rows = A.getnrow();
    IT total_cols = A.getncol();

    int procrows = grid->GetGridRows();
    int proccols = grid->GetGridCols();

    IT rows_perproc = total_rows / procrows;
    IT cols_perproc = total_cols / proccols;

    IT row_offset = myprocrow * rows_perproc;
    IT col_offset = myproccol * cols_perproc;

    DER *spSeq = A.seqptr();

    IT localcols = spSeq->getncol();
    std::vector<IT> local_colidx(localcols, static_cast<IT>(-1));

    for (auto colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)
    {
        auto nzit = spSeq->begnz(colit);
        if (nzit != spSeq->endnz(colit))
            local_colidx[colit.colid()] = nzit.rowid() + row_offset;
    }

    MPI_Allreduce(MPI_IN_PLACE, local_colidx.data(), static_cast<int>(localcols), MPIType<IT>(), MPI_MAX, ColWorld);

    std::vector<IT> fillarr;

    if (!myprocrow)
        for (auto itr = local_colidx.begin(); itr != local_colidx.end(); ++itr)
            fillarr.push_back(*itr);

    return FullyDistVec<IT,IT>(fillarr, grid);
}

template <class IT, class NT, class DER>
SpParMat<IT,NT,DER> FrontierMat(const SpParMat<IT,NT,DER>& A, const FullyDistSpVec<IT,IT>& sources, const NT& initval)
{
    FullyDistVec<IT,IT> ri = sources.FindInds([](int arg1) { return true; });
    FullyDistVec<IT,IT> ci(A.getcommgrid());
    ci.iota(sources.getnnz(), static_cast<IT>(0));
    return SpParMat<IT,NT,DER>(A.getnrow(), sources.getnnz(), ri, ci, initval, false);
}

int main(int argc, char *argv[])
{
    int myrank, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc != 3)
    {
        if (!myrank)
            std::cerr << "Usage: ./KTipsTest <Matrix> <l>" << std::endl;
        MPI_Finalize();
        return -1;
    }

    {
        int l = atoi(argv[2]);

        std::shared_ptr<CommGrid> fullWorld;
        fullWorld.reset(new CommGrid(MPI_COMM_WORLD, 0, 0));

        SpParMat<int,int, SpDCCols<int,int>> A(fullWorld);

        A.ParallelReadMM(std::string(argv[1]), false, maximum<int>());

        FullyDistVec<int,int> D = A.Reduce(Column, std::plus<int>(), static_cast<int>(0));
        FullyDistSpVec<int,int> R = D.Find([](int val){return val == 1;});

        SpParMat<int,int, SpDCCols<int,int>> F0 = FrontierMat(A, R, static_cast<int>(1));

        SpParMat<int,int, SpDCCols<int,int>> F1 = PSpGEMM<KTipsSR<int>>(A, F0);
        SpParMat<int,int, SpDCCols<int,int>> V = F0;
        V += F1;

        FullyDistVec<int,int> TipSources(A.getcommgrid(), F0.getncol(), static_cast<int>(-1));
        FullyDistVec<int,int> TipDests(A.getcommgrid(), F0.getncol(), static_cast<int>(-1));

        for (int k = 1; k <= l; ++k)
        {
            SpParMat<int,int, SpDCCols<int,int>> F2 = PSpGEMM<KTipsSR<int>>(A, F1);
            F2.SetDifference(V);
            V += F2;

            FullyDistVec<int,int> Ns = F2.Reduce(Column, std::plus<int>(), static_cast<int>(0));

            FullyDistSpVec<int,int> Tc = Ns.Find([](int val){return val >= 2;});
            FullyDistSpVec<int,int> Td = Ns.Find([](int val){return val != 1;});

            FullyDistVec<int,int> C0 = LastNzRowIdxPerCol(F0);
            FullyDistVec<int,int> C1 = LastNzRowIdxPerCol(F1);

            FullyDistSpVec<int,int> kSources = C0.GGet(Tc, [](const int arg1, const int arg2) { return arg2; }, static_cast<int>(-1));
            FullyDistSpVec<int,int> kDests = C1.GGet(Tc, [](const int arg1, const int arg2) { return arg2; }, static_cast<int>(-1));

            TipSources.Set(kSources);
            TipDests.Set(kDests);

            F1.PruneColumnByIndex(Td);
            F2.PruneColumnByIndex(Td);

            F0 = F1;
            F1 = F2;
        }

        TipSources.DebugPrint();
        TipDests.DebugPrint();
    }

    MPI_Finalize();
    return 0;
}
