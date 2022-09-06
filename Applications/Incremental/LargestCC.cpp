/*
 * Dumps the largest connected component of a given graph to a file with the 
 * given filename as prefix and .cc as suffix
 * */

#include <mpi.h>
#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include <cstdlib>
#include "../CC.h"
#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/CommGrid3D.h"
#include "CombBLAS/SpParMat3D.h"
#include "CombBLAS/ParFriends.h"
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
            cout << "Usage: ./lcc -I <mm|triples> -M <MATRIX_FILENAME> -C <CC_FILENAME>\n";
            cout << "-I <INPUT FILE TYPE> (mm: matrix market, triples: (vtx1, vtx2, edge_weight) triples, default: mm)\n";
            cout << "-M <MATRIX FILE NAME>\n";
            cout << "-C <CONNECTED COMPONENT FILE NAME>\n";
            cout << "-base <BASE OF MATRIX MARKET> (default:1)\n";
        }
        MPI_Finalize();
        return -1;
    }
    else{
        string Mname = "";
        string Cname = "";
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
                if(myrank == 0) printf("Matrix filename: %s\n",Mname.c_str());
            }
            else if (strcmp(argv[i],"-C")==0)
            {
                Cname = string(argv[i+1]);
                if(myrank == 0) printf("Connected component filename: %s\n", Cname.c_str());
            }
            else if (strcmp(argv[i],"-base")==0)
            {
                base = atoi(argv[i + 1]);
                if(myrank == 0) printf("\nBase of MM (1 or 0):%d",base);
            }
        }

        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );

        typedef PlusTimesSRing<double, double> PTFF;
        typedef PlusTimesSRing<bool, double> PTBOOLNT;
        typedef PlusTimesSRing<double, bool> PTNTBOOL;
        typedef int64_t IT;
        typedef double NT;
        typedef SpDCCols < int64_t, double > DER;
        
        double t0, t1;

        SpParMat<IT, NT, DER> M(fullWorld);
        FullyDistVec<IT, IT> ccLabels(fullWorld);
        FullyDistVec<IT, IT> ccLabelsVusVus(fullWorld);

        if(isMatrixMarket)
            M.ParallelReadMM(Mname, base, maximum<double>());
        else
            M.ReadGeneralizedTuples(Mname,  maximum<double>());
        M.PrintInfo();
        ccLabels.ParallelRead(Cname, base, maximum<IT>());
        IT nCC = ccLabels.Reduce(maximum<IT>(), (IT) 0 ) ;

        SpParMat<IT, NT, DER> A(M); // Create a copy of given matrix

        // A is a directed graph
        // symmetricize A
        SpParMat<IT,NT,DER> AT = A;
        AT.Transpose();
        A += AT;

        ccLabelsVusVus = CC(A, nCC);
        //ccLabels = CC(A, nCC);

        if(ccLabelsVusVus == ccLabels){
            if(myrank == 0) fprintf(stderr, "Both vectors equal\n");
        }
        else{
            if(myrank == 0) fprintf(stderr, "Both vectors not equal\n");
        }

        if(myrank == 0) fprintf(stderr, "Number of connected components: %lld\n", nCC);
        std::vector<IT> ccLabelsLocal = ccLabels.GetLocVec();
        IT* ccSizesLocal = new IT[nCC];
        IT* ccSizesGlobal = new IT[nCC];
        memset(ccSizesLocal, 0, sizeof(IT) * nCC);
        memset(ccSizesGlobal, 0, sizeof(IT) * nCC);
#ifdef THREADED
#pragma omp parallel for
#endif
        for(IT i = 0; i < nCC; i++){
            ccSizesLocal[i] = std::count_if( ccLabelsLocal.begin(), ccLabelsLocal.end(), bind2nd(equal_to<IT>(), i));
        }
        MPI_Allreduce(ccSizesLocal, ccSizesGlobal, (int)nCC, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

        IT largestCC = 0;
        IT largestCCSize = ccSizesGlobal[largestCC];
        if(myrank == 0){
            fprintf(stderr, "LargestCC: %lld, LargestCCSize: %lld\n", largestCC, largestCCSize);
        }
        for(IT i = 1; i < nCC; i++){
            if (ccSizesGlobal[i] > largestCCSize){
                largestCC = i;
                largestCCSize = ccSizesGlobal[i];
                if(myrank == 0){
                    fprintf(stderr, "LargestCC: %lld, LargestCCSize: %lld\n", largestCC, largestCCSize);
                }
            }
        }

        delete ccSizesLocal;
        delete ccSizesGlobal;

        if(myrank == 0) printf("Largest connected component is %dth component, size %d\n",largestCC, largestCCSize);
        FullyDistVec<IT,IT> isov = ccLabels.FindInds(bind2nd(equal_to<IT>(), largestCC));
        isov.ParallelWrite(Mname+std::string(".isov2"), 0);
        SpParMat<IT, NT, DER> MCC = M.SubsRef_SR<PTNTBOOL,PTBOOLNT>(isov, isov, false);
        MCC.PrintInfo();
        MCC.ParallelWriteMM(Mname+std::string(".lcc"), true);
    }

    MPI_Finalize();
    return 0;
}
