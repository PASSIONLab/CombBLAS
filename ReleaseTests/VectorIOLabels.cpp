// Round-trip test for FullyDistVec<IT, std::array<char, MAXVERTNAME>> I/O.
//
// CombBLAS has two incompatible vector file formats:
//   (1) MatrixMarket-style:    "<nrows> <ncols> <nnz>" header + "<row> <col> <val>" lines.
//                              Used by ReadDistribute / SaveGathered.
//   (2) Native vector format:  "<glen> <gnnz>" header + "<idx> <val>" lines.
//                              Used by ParallelRead / ParallelWrite.
//
// The input fixture is in format (1), so Step 1 uses ReadDistribute as a
// format-conversion step. Steps 2-7 then exercise format (2) round-trips twice: once
// via the handler-based API, once via the native `if constexpr` fast path for
// std::array<char, MAXVERTNAME> labels.

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include <algorithm>
#include "CombBLAS/CombBLAS.h"

using namespace std;
using namespace combblas;

typedef int64_t IT;
typedef array<char, MAXVERTNAME> LBL;

// Copied from ReleaseTests/ParIOTest.cpp
class StdArrayReadSaveHandler
{
public:
    LBL getNoNum(IT index) { return LBL(); }

    template <typename c, typename t>
    LBL read(std::basic_istream<c,t>& is, IT index)
    {
        LBL strarray{};
        string str;
        is >> str;
        std::copy(str.begin(), str.end(), strarray.begin());
        if (str.length() < MAXVERTNAME) strarray[str.length()] = '\0';
        return strarray;
    }

    template <typename c, typename t>
    void save(std::basic_ostream<c,t>& os, const LBL& strarray, IT index)
    {
        auto locnull = find(strarray.begin(), strarray.end(), '\0');
        string str(strarray.begin(), locnull);
        os << str;
    }
};

int main(int argc, char* argv[])
{
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (argc < 2)
    {
        if (myrank == 0)
        {
            cout << "Usage: ./VectorIOLabels <input-vector-file>" << endl;
            cout << "  <input-vector-file>: vector in MatrixMarket-style format" << endl;
            cout << "  Header: numrows numcols nnz; lines: row col value" << endl;
        }
        MPI_Finalize();
        return -1;
    }

    string inputFile(argv[1]);
    string tempFile = "temp_label_vec_io.txt";
    bool allPass = true;

    {
    shared_ptr<CommGrid> fullWorld;
    fullWorld.reset(new CommGrid(MPI_COMM_WORLD, 0, 0));

    // Step 1: Read input file via ReadDistribute
    FullyDistVec<IT, LBL> v1(fullWorld);
    {
        ifstream infile(inputFile.c_str());
        if (infile.fail())
        {
            if (myrank == 0) cout << "Failed to open " << inputFile << endl;
            MPI_Finalize();
            return -1;
        }
        v1.ReadDistribute(infile, 0, StdArrayReadSaveHandler());
        infile.close();
    }
    if (myrank == 0) cout << "[Step 1] ReadDistribute complete. v1 length = " << v1.TotalLength() << endl;

    // Step 2: ParallelWrite v1 to temp file
    v1.ParallelWrite(tempFile, 1, StdArrayReadSaveHandler(), true);
    if (myrank == 0) cout << "[Step 2] ParallelWrite complete to " << tempFile << endl;

    // Step 3: ParallelRead temp file into v2 (exercises the new if-constexpr path)
    FullyDistVec<IT, LBL> v2(fullWorld);
    v2.ParallelRead(tempFile, 1, [](LBL a, LBL b) { return a; });
    if (myrank == 0) cout << "[Step 3] ParallelRead complete. v2 length = " << v2.TotalLength() << endl;

    // Step 4: Compare v1 vs v2
    if (v1 == v2)
    {
        if (myrank == 0) cout << "[Step 4] PASS — v1 matches v2" << endl;
    }
    else
    {
        if (myrank == 0) cout << "[Step 4] FAIL — v1 differs from v2" << endl;
        allPass = false;
    }

    // Step 5: ParallelWrite v2 without handler (native-write path)
    string tempFile2 = "temp_label_vec_io_native.txt";
    v2.ParallelWrite(tempFile2, 1, true);
    if (myrank == 0) cout << "[Step 5] ParallelWrite (no handler) complete to " << tempFile2 << endl;

    // Step 6: ParallelRead tempFile2 into v3
    FullyDistVec<IT, LBL> v3(fullWorld);
    v3.ParallelRead(tempFile2, 1, [](LBL a, LBL b) { return a; });
    if (myrank == 0) cout << "[Step 6] ParallelRead complete. v3 length = " << v3.TotalLength() << endl;

    // Step 7: Compare v2 vs v3
    if (v2 == v3)
    {
        if (myrank == 0) cout << "[Step 7] PASS — v2 matches v3 (native round-trip)" << endl;
    }
    else
    {
        if (myrank == 0) cout << "[Step 7] FAIL — v2 differs from v3" << endl;
        allPass = false;
    }
    }

    MPI_Finalize();
    return allPass ? 0 : 1;
}
