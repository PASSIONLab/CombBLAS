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
#include "IncClust.h"

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
    if(argc < 5)
    {
        if(myrank == 0)
        {
            cout << "Usage: ./prep-data -I <mm|triples> -M <matrix> -old <0.9> -prefix <directory path + file prefix>\n";
            cout << "-I <INPUT FILE TYPE> (mm: matrix market, triples: (vtx1, vtx2, edge_weight) triples, default: mm)\n";
            cout << "-base <BASE OF MATRIX MARKET> (default:1)\n";
            cout << "-M <MATRIX FILE NAME>\n";
            cout << "-old <PERCENTAGE OF OLD VERTICES>\n";
            cout << "-prefix <PREFIX OF FILES TO BE SAVED>\n";
        }
        MPI_Finalize();
        return -1;
    }
    else{
        string Mname = "";
        int nSplit = 2;
        int base = 1;
        bool isMatrixMarket = true;
        float pctOld = 0.9;
        float pctNew = 0.1;
        string prefix = "";
        
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
            //else if (strcmp(argv[i],"-nsplit")==0)
            //{
                //nSplit = atoi(argv[i+1]);
                //if(myrank == 0) printf("Number of splits: %d\n", nSplit);
            //}
            else if (strcmp(argv[i],"-old")==0)
            {
                pctOld = atof(argv[i+1]);
                pctNew = 1.0 - pctOld;
            }
            else if (strcmp(argv[i],"-prefix")==0)
            {
                prefix = string(argv[i+1]);
            }
        }

        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );

        typedef int64_t IT;
        typedef double NT;
        typedef SpDCCols < int64_t, double > DER;
        typedef PlusTimesSRing<double, double> PTFF;
        typedef PlusTimesSRing<bool, double> PTBOOLNT;
        typedef PlusTimesSRing<double, bool> PTNTBOOL;
        typedef std::array<char, MAXVERTNAME> LBL;
        
        double t0, t1, t2, t3, t4, t5;

        SpParMat<IT, NT, DER> M(fullWorld);

        if(isMatrixMarket)
            M.ParallelReadMM(Mname, base, maximum<double>());
        else
            M.ReadGeneralizedTuples(Mname,  maximum<double>());
        M.PrintInfo();

        IT nTotal = M.getnrow();
        IT nOld = (IT)(nTotal * pctOld);
        IT nNew = nTotal - nOld;
        
        FullyDistVec<IT, IT> pVtxMap( fullWorld );
        FullyDistVec<IT, IT> nVtxMap( fullWorld );
        pVtxMap.iota(nOld, 0); // Intialize with consecutive numbers
        nVtxMap.iota(nNew, nOld); // Initialize with consecutive numbers

        const std::vector<IT> pVtxMapLoc = pVtxMap.GetLocVec();
        std::vector<LBL> pVtxLblLoc(pVtxMapLoc.size());
        const std::vector<IT> nVtxMapLoc = nVtxMap.GetLocVec();
        std::vector<LBL> nVtxLblLoc(nVtxMapLoc.size());

        for(int i = 0; i < pVtxMapLoc.size(); i++){
            std::string labelStr = std::to_string(pVtxMapLoc[i]); 
            for ( IT j = 0; (j < labelStr.length()) && (j < MAXVERTNAME); j++){
                pVtxLblLoc[i][j] = labelStr[j]; 
            }
        }
        for(int i = 0; i < nVtxMapLoc.size(); i++){
            std::string labelStr = std::to_string(nVtxMapLoc[i]); 
            for ( IT j = 0; (j < labelStr.length()) && (j < MAXVERTNAME); j++){
                nVtxLblLoc[i][j] = labelStr[j]; 
            }
        }

        FullyDistVec<IT, LBL> pVtxLbl(pVtxLblLoc, fullWorld);
        FullyDistVec<IT, LBL> nVtxLbl(nVtxLblLoc, fullWorld);

        IT pLocLen = pVtxLbl.LocArrSize(); 
        IT nLocLen = nVtxLbl.LocArrSize();
               
        IT minLocLen = std::min(pLocLen, nLocLen);
        IT maxLocLen = std::max(pLocLen, nLocLen);

        // Boolean flags for each element to keep track of which elements have been swapped between prev and new
        std::vector<bool> pLocFlag(pLocLen, true);
        std::vector<bool> nLocFlag(nLocLen, true);

        // Initialize two uniform random number generators
        // one is a real generator in a range of [0.0-1.0] to do coin toss to decide whether a swap will happen or not
        // another is an integer generator to randomly pick positions to swap,
        std::mt19937 rng;
        rng.seed(myrank);
        std::uniform_real_distribution<float> urdist(0, 1.0);
        std::uniform_int_distribution<IT> uidist(0, std::numeric_limits<IT>::max()); 

        // MTH: Enable multi-threading?
        for (IT i = 0; i < minLocLen; i++){ // Run as many attempts as minimum of the candidate array lengths
            if(urdist(rng) < double(maxLocLen)/(maxLocLen + minLocLen)){ // If the picked random real number is less than the ratio of new and previous vertices
                // Randomly select an index from the previous vertex list
                IT idxPrev = uidist(rng) % pLocLen; 

                // If the selected index is already swapped then cyclicly probe the indices after that 
                // until an index is found which has not been swapped
                while(pLocFlag[idxPrev] == false) idxPrev = (idxPrev + 1) % pLocLen;
                
                // Mark the index as swapped
                pLocFlag[idxPrev] = false;

                // Randomly select an index from the new vertex list
                IT idxNew = uidist(rng) % nLocLen;

                // If the selected index is already swapped then cyclicly probe the indices after that 
                // until an index is found which has not been swapped
                while(nLocFlag[idxNew] == false) idxNew = (idxNew + 1) % nLocLen;

                // Mark the index as swapped
                nLocFlag[idxNew] = false;

                IT pVtxRM = pVtxMap.GetLocalElement(idxPrev);
                IT nVtxRM = nVtxMap.GetLocalElement(idxNew);
                pVtxMap.SetLocalElement(idxPrev, nVtxRM);
                nVtxMap.SetLocalElement(idxNew, pVtxRM);

                LBL pLbl = pVtxLbl.GetLocalElement(idxPrev);
                LBL nLbl = nVtxLbl.GetLocalElement(idxNew);
                pVtxLbl.SetLocalElement(idxPrev, nLbl);
                nVtxLbl.SetLocalElement(idxNew, pLbl);
            }
        }
        
        // Global permutation after local shuffle will result in true shuffle
        // Use same seed for all permutations
        pVtxMap.RandPerm(31415929535);
        nVtxMap.RandPerm(31415929535);
        pVtxLbl.RandPerm(31415929535);
        nVtxLbl.RandPerm(31415929535);
        
        SpParMat<IT, NT, DER> M11(fullWorld);
        SpParMat<IT, NT, DER> M12(fullWorld);
        SpParMat<IT, NT, DER> M21(fullWorld);
        SpParMat<IT, NT, DER> M22(fullWorld);

        t0 = MPI_Wtime();
        M11 = M.SubsRef_SR < PTNTBOOL, PTBOOLNT> (pVtxMap, pVtxMap, false);
        t1 = MPI_Wtime();
        if(myrank == 0) printf("Time to extract M11: %lf\n", t1-t0);
        t0 = MPI_Wtime();
        M11.ParallelWriteMM(prefix + std::string(".m11.mtx"), base);
        t1 = MPI_Wtime();
        if(myrank == 0) printf("Time to write M11: %lf\n", t1-t0);

        t0 = MPI_Wtime();
        M12 = M.SubsRef_SR < PTNTBOOL, PTBOOLNT> (pVtxMap, nVtxMap, false);
        t1 = MPI_Wtime();
        if(myrank == 0) printf("Time to extract M12: %lf\n", t1-t0);
        t0 = MPI_Wtime();
        M12.ParallelWriteMM(prefix + std::string(".m12.mtx"), base);
        t1 = MPI_Wtime();
        if(myrank == 0) printf("Time to write M12: %lf\n", t1-t0);

        t0 = MPI_Wtime();
        M21 = M.SubsRef_SR < PTNTBOOL, PTBOOLNT> (nVtxMap, pVtxMap, false);
        t1 = MPI_Wtime();
        if(myrank == 0) printf("Time to extract M21: %lf\n", t1-t0);
        t0 = MPI_Wtime();
        M21.ParallelWriteMM(prefix + std::string(".m21.mtx"), base);
        t1 = MPI_Wtime();
        if(myrank == 0) printf("Time to write M21: %lf\n", t1-t0);

        t0 = MPI_Wtime();
        M22 = M.SubsRef_SR < PTNTBOOL, PTBOOLNT> (nVtxMap, nVtxMap, false);
        t1 = MPI_Wtime();
        if(myrank == 0) printf("Time to extract M22: %lf\n", t1-t0);
        t0 = MPI_Wtime();
        M22.ParallelWriteMM(prefix + std::string(".m22.mtx"), base);
        t1 = MPI_Wtime();
        if(myrank == 0) printf("Time to write M22: %lf\n", t1-t0);

        t0 = MPI_Wtime();
        pVtxMap.ParallelWrite(prefix + std::string(".m11.lbl"), base);
        t1 = MPI_Wtime();
        if(myrank == 0) printf("Time to write M11 vertex label: %lf\n", t1-t0);

        t0 = MPI_Wtime();
        nVtxMap.ParallelWrite(prefix + std::string(".m22.lbl"), base);
        t1 = MPI_Wtime();
        if(myrank == 0) printf("Time to write M22 vertex label: %lf\n", t1-t0);
    }
    MPI_Finalize();
    return 0;
}
