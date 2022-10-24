
#include <mpi.h>
#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include <cstdlib>
#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/CommGrid3D.h"
#include "CombBLAS/ParFriends.h"
#include "../CC.h"
#include "../WriteMCLClusters.h"

using namespace std;
using namespace combblas;

#ifdef _OPENMP
int cblas_splits = omp_get_max_threads();
#else
int cblas_splits = 1;
#endif

int main(int argc, char* argv[])
{
    int nprocs, myrank, nthreads = 1;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
#ifdef THREADED
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
#endif
    if(myrank == 0)
    {
        cout << "Process Grid (p x p x t): " << sqrt(nprocs) << " x " << sqrt(nprocs) << " x " << nthreads << endl;
    }
    if(argc < 7)
    {
        if(myrank == 0)
        {
            cout << "Usage: ./inc -I <mm|triples> -M <MATRIX_FILENAME> -N <NUMBER OF SPLITS>\n";
            cout << "-I <INPUT FILE TYPE> (mm: matrix market, triples: (vtx1, vtx2, edge_weight) triples, default: mm)\n";
            cout << "-M <MATRIX FILE NAME>\n";
            cout << "-base <BASE OF MATRIX MARKET> (default:1)\n";
            cout << "-N <NUMBER OF SPLITS>\n";
        }
        MPI_Finalize();
        return -1;
    }
    else{
        string Mname = "";
        int nSplit = 2;
        int base = 1;
        bool isMatrixMarket = true;
        
        for (int i = 1; i < argc; i++)
        {
            if (strcmp(argv[i],"-I")==0)
            {
                string mfiletype = string(argv[i+1]);
                if(mfiletype == "triples") isMatrixMarket = false;
            }
            else if (strcmp(argv[i],"-M")==0)
            {
                Mname = string(argv[i+1]);
                if(myrank == 0) printf("Matrix filename: %s\n", Mname.c_str());
            }
            else if (strcmp(argv[i],"-base")==0)
            {
                base = atoi(argv[i + 1]);
                if(myrank == 0) printf("Base of MM (1 or 0):%d\n", base);
            }
            else if (strcmp(argv[i],"-N")==0)
            {
                nSplit = atoi(argv[i+1]);
                if(myrank == 0) printf("Number of splits: %d\n", nSplit);
            }
        }

        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );

        if(myrank == 0) printf("Running TestvtxShuffle\n");

        typedef int64_t IT;
        typedef double NT;
        typedef SpDCCols < int64_t, double > DER;
        typedef PlusTimesSRing<double, double> PTFF;
        typedef PlusTimesSRing<bool, double> PTBOOLNT;
        typedef PlusTimesSRing<double, bool> PTNTBOOL;
        
        double t0, t1, t2, t3, t4, t5;

        SpParMat<IT, NT, DER> M(fullWorld);

        if(isMatrixMarket)
            M.ParallelReadMM(Mname, base, maximum<double>());
        else
            M.ReadGeneralizedTuples(Mname,  maximum<double>());
        M.PrintInfo();

        std::string incFileName = Mname + std::string(".") + std::to_string(nSplit) + std::string(".inc-opt");

        for(int s = 1; s < nSplit; s++){
            std::string ncPrefix = Mname + std::string(".") + std::to_string(nSplit) + std::string(".inc-opt-nc");
            std::string nmPrefix = Mname + std::string(".") + std::to_string(nSplit) + std::string(".inc-opt-nm");
            std::string ymPrefix = Mname + std::string(".") + std::to_string(nSplit) + std::string(".inc-opt-ym");
            std::string prevSuffix = std::string(".prevVtx") + std::string(".") + std::to_string(s);
            std::string nextSuffix = std::string(".newVtx") + std::string(".") + std::to_string(s);

            FullyDistVec<IT, IT> ncPrev(fullWorld);
            FullyDistVec<IT, IT> ncNext(fullWorld);
            FullyDistVec<IT, IT> nmPrev(fullWorld);
            FullyDistVec<IT, IT> nmNext(fullWorld);
            FullyDistVec<IT, IT> ymPrev(fullWorld);
            FullyDistVec<IT, IT> ymNext(fullWorld);

            ncPrev.ParallelRead( ncPrefix+prevSuffix, base, maximum<double>() );
            ncNext.ParallelRead( ncPrefix+nextSuffix, base, maximum<double>() );
            nmPrev.ParallelRead( nmPrefix+prevSuffix, base, maximum<double>() );
            nmNext.ParallelRead( nmPrefix+nextSuffix, base, maximum<double>() );
            ymPrev.ParallelRead( ymPrefix+prevSuffix, base, maximum<double>() );
            ymNext.ParallelRead( ymPrefix+nextSuffix, base, maximum<double>() );

            if(myrank == 0){
                fprintf(stderr, "Split: %d\n", s);
            }

            if(myrank == 0){
                fprintf(stderr, "\tComparing ncPrev and nmPrev\n");
            }
            if (ncPrev == nmPrev){
                if(myrank == 0){
                    fprintf(stderr, "\t\tEqual\n");
                }
            }
            else{
                if(myrank == 0){
                    fprintf(stderr, "\t\tNot Equal\n");
                }
            }

            if(myrank == 0){
                fprintf(stderr, "\tComparing ncNext and nmNext\n");
            }
            if (ncNext == nmNext){
                if(myrank == 0){
                    fprintf(stderr, "\t\tEqual\n");
                }
            }
            else{
                if(myrank == 0){
                    fprintf(stderr, "\t\tNot Equal\n");
                }
            }
            
            ymPrev.sort();
            nmPrev.sort();

            if(myrank == 0){
                fprintf(stderr, "\tComparing ymPrev and nmPrev\n");
            }
            if (ymPrev == nmPrev){
                if(myrank == 0){
                    fprintf(stderr, "\t\tEqual\n");
                }
            }
            else{
                if(myrank == 0){
                    fprintf(stderr, "\t\tNot Equal\n");
                }
            }

            ymNext.sort();
            nmNext.sort();

            if(myrank == 0){
                fprintf(stderr, "\tComparing ymNext and nmNext\n");
            }
            if (ymNext == nmNext){
                if(myrank == 0){
                    fprintf(stderr, "\t\tEqual\n");
                }
            }
            else{
                if(myrank == 0){
                    fprintf(stderr, "\t\tNot Equal\n");
                }
            }
        }
    }
    MPI_Finalize();
    return 0;
}
