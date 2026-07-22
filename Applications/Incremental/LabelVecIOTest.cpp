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
            cout << "Usage: ./label-vec-io-test <input-vector-file>" << endl;
            cout << "  <input-vector-file>: vector in MatrixMarket-style format" << endl;
            cout << "  Header: numrows numcols nnz; lines: row col value" << endl;
        }
        MPI_Finalize();
        return -1;
    }

    string inputFile(argv[1]);
    string tempFile = "temp_label_vec_io.txt";

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
    }
    }

    MPI_Finalize();
    return 0;
}
