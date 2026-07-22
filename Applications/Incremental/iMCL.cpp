/*
 * Authors: Md Taufique Hussain
 * */

#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/CommGrid3D.h"
#include "CombBLAS/ParFriends.h"
#include "../CC.h"
#include "../WriteMCLClusters.h"
#include "IncClust.h"

using namespace std;
using namespace combblas;

#ifdef _OPENMP
int cblas_splits = omp_get_max_threads();
#else
int cblas_splits = 1;
#endif

typedef int64_t IT;
typedef double NT;
typedef SpDCCols<int64_t, double> DER;
typedef PlusTimesSRing<double, double> PTFF;
typedef PlusTimesSRing<int64_t, int64_t> PTII;
typedef PlusTimesSRing<bool, double> PTBOOLNT;
typedef PlusTimesSRing<double, bool> PTNTBOOL;
typedef std::array<char, MAXVERTNAME> LBL;

void runInit(const string& triples, const string& stateDir, int perProcessMem);

void runUpdate(const string& triplesNew, const string& triplesMixed,
               const string& stateDir, int perProcessMem,
               int summaryThreshold, int selectivePruneThreshold);

int main(int argc, char* argv[])
{
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    string mode = "";
    string triples = "";
    string triplesNew = "";
    string triplesMixed = "";
    string stateDir = "";
    int perProcessMem = 4;
    int summaryThreshold = 70;
    int selectivePruneThreshold = 10;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--mode") == 0)                           mode = string(argv[i+1]);
        else if (strcmp(argv[i], "--triples") == 0)                   triples = string(argv[i+1]);
        else if (strcmp(argv[i], "--triples-new") == 0)               triplesNew = string(argv[i+1]);
        else if (strcmp(argv[i], "--triples-mixed") == 0)             triplesMixed = string(argv[i+1]);
        else if (strcmp(argv[i], "--state") == 0)                     stateDir = string(argv[i+1]);
        else if (strcmp(argv[i], "--per-process-mem") == 0)           perProcessMem = atoi(argv[i+1]);
        else if (strcmp(argv[i], "--summary-threshold") == 0)         summaryThreshold = atoi(argv[i+1]);
        else if (strcmp(argv[i], "--selective-prune-threshold") == 0) selectivePruneThreshold = atoi(argv[i+1]);
    }

    bool valid = true;
    if (mode == "init") {
        if (triples == "" || stateDir == "") {
            if (myrank == 0) cout << "Usage: imcl --mode init --triples <file> --state <dir>" << endl;
            valid = false;
        }
    } else if (mode == "update") {
        if (triplesNew == "" || triplesMixed == "" || stateDir == "") {
            if (myrank == 0) cout << "Usage: imcl --mode update --triples-new <file> --triples-mixed <file> --state <dir>" << endl;
            valid = false;
        }
    } else {
        if (myrank == 0) cout << "Usage: imcl --mode <init|update> [options]" << endl;
        valid = false;
    }

    if (!valid) { MPI_Finalize(); return -1; }

    if (mode == "init")        runInit(triples, stateDir, perProcessMem);
    else if (mode == "update") runUpdate(triplesNew, triplesMixed, stateDir, perProcessMem, summaryThreshold, selectivePruneThreshold);

    MPI_Finalize();
    return 0;
}

void runInit(const string& triples, const string& stateDir, int perProcessMem)
{
    // TODO
}

void runUpdate(const string& triplesNew, const string& triplesMixed,
               const string& stateDir, int perProcessMem,
               int summaryThreshold, int selectivePruneThreshold)
{
    // TODO
}
