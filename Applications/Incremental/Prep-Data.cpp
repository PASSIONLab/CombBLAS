
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
        string outPrefix = "";

        HipMCLParam incParam;
        InitParam(incParam);
        
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
            else if (strcmp(argv[i],"-out-prefix")==0)
            {
                outPrefix = string(argv[i+1]);
                if(myrank == 0) printf("Output file prefix: %s\n", outPrefix.c_str());
            }
        }

        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );

        if(myrank == 0) printf("Preparing incremental data\n");

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
            M.ParallelReadMM(Mname, base, maximum<NT>());
        else
            M.ReadGeneralizedTuples(Mname,  maximum<NT>());
        M.PrintInfo();
        
        std::mt19937 rng;
        rng.seed(myrank);
        std::uniform_int_distribution<IT> udist(0, 9999);

        IT gnRow = M.getnrow();
        IT nRowPerProc = gnRow / nprocs;
        IT lRowStart = myrank * nRowPerProc;
        IT lRowEnd = (myrank == nprocs - 1) ? gnRow : (myrank + 1) * nRowPerProc;

        std::vector < std::vector < IT > > lvList(nSplit);
        std::vector < std::vector < LBL > > lvListLabels(nSplit); // MAXVERTNAME is 64, defined in SpDefs
                                                                                           
        for (IT r = lRowStart; r < lRowEnd; r++) {
            IT randomNum = udist(rng);
            IT s = randomNum % nSplit;
            lvList[s].push_back(r);
            
            // Convert the integer vertex id to label as string
            std::string labelStr = std::to_string(r); 
            // Make a std::array of char with the label
            LBL labelArr = {};
            for ( IT i = 0; (i < labelStr.length()) && (i < MAXVERTNAME); i++){
                labelArr[i] = labelStr[i]; 
            }
            lvListLabels[s].push_back( labelArr );
        }

        std::vector < FullyDistVec<IT,IT>* > dvList;
        std::vector < FullyDistVec<IT, std::array<char, MAXVERTNAME> >* > dvListLabels;
        for (int s = 0; s < nSplit; s++){
            dvList.push_back(new FullyDistVec<IT, IT>(lvList[s], fullWorld));
            dvListLabels.push_back(new FullyDistVec<IT, std::array<char, MAXVERTNAME> >(lvListLabels[s], fullWorld));
        }

        SpParMat<IT, NT, DER> M11(fullWorld);
        SpParMat<IT, NT, DER> M12(fullWorld);
        SpParMat<IT, NT, DER> M21(fullWorld);
        SpParMat<IT, NT, DER> M22(fullWorld);

        std::string outFileName = Mname + std::string(".") + std::to_string(nSplit) + std::string(".inc-v1");

        FullyDistVec<IT, IT> prevVertices(*(dvList[0])); // Create a distributed vector to keep track of the vertices being considered at each incremental step
        FullyDistVec<IT, LBL> prevVerticesLabels(*(dvListLabels[0])); // Create a distributed vector to keep track of the vertex labels being considered at each incremental step

        for(int s = 1; s < nSplit; s++){
            MPI_Barrier(MPI_COMM_WORLD);
            if(myrank == 0) printf("[Start] Split: %d\n", s);

            FullyDistVec<IT, IT> newVertices(*(dvList[s]));
            FullyDistVec<IT, LBL> newVerticesLabels(*(dvListLabels[s]));

            for(int it=0; it<50 && s==2; it++){

            if(myrank == 0) printf("It: %d\n", it);
            if(myrank == 0) printf("[Start] Subgraph extraction\n");
            if(s == 1){
                M11.FreeMemory();
                t0 = MPI_Wtime();
                M11 = M.SubsRef_SR <PTNTBOOL, PTBOOLNT> (prevVertices, prevVertices, false);
                t1 = MPI_Wtime();
                if(myrank == 0) printf("Time to extract M11: %lf\n", t1 - t0);
                M11.PrintInfo();
                outFileName = outPrefix + std::string(".") + std::to_string(nSplit) + std::string(".") + std::to_string(0) + std::string(".m11.") + std::string("mtx");
                //M11.ParallelWriteMM(outFileName, base);
                outFileName = outPrefix + std::string(".") + std::to_string(nSplit) + std::string(".") + std::to_string(0) + std::string(".m11.") + std::string("lbl");
                //prevVertices.ParallelWrite(outFileName, base);
            }

            M12.FreeMemory();
            t0 = MPI_Wtime();
            M12 = M.SubsRef_SR <PTNTBOOL, PTBOOLNT> (prevVertices, newVertices, false);
            t1 = MPI_Wtime();
            if(myrank == 0) printf("Time to extract M12: %lf\n", t1 - t0);
            M12.PrintInfo();
            outFileName = outPrefix + std::string(".") + std::to_string(nSplit) + std::string(".") + std::to_string(s) + std::string(".m12.") + std::string("mtx");
            //M12.ParallelWriteMM(outFileName, base);

            M21.FreeMemory();
            t0 = MPI_Wtime();
            M21 = M.SubsRef_SR <PTNTBOOL, PTBOOLNT> (newVertices, prevVertices, false);
            t1 = MPI_Wtime();
            if(myrank == 0) printf("Time to extract M21: %lf\n", t1 - t0);
            M21.PrintInfo();
            outFileName = outPrefix + std::string(".") + std::to_string(nSplit) + std::string(".") + std::to_string(s) + std::string(".m21.") + std::string("mtx");
            //M21.ParallelWriteMM(outFileName, base);
            
            M22.FreeMemory();
            t0 = MPI_Wtime();
            M22 = M.SubsRef_SR <PTNTBOOL, PTBOOLNT> (newVertices, newVertices, false); // Get subgraph induced by newly added vertices in current step
            t1 = MPI_Wtime();
            if(myrank == 0) printf("Time to extract M22: %lf\n", t1 - t0);
            M22.PrintInfo();
            outFileName = outPrefix + std::string(".") + std::to_string(nSplit) + std::string(".") + std::to_string(s) + std::string(".m22.") + std::string("mtx");
            //M22.ParallelWriteMM(outFileName, base);
            outFileName = outPrefix + std::string(".") + std::to_string(nSplit) + std::string(".") + std::to_string(s) + std::string(".m22.") + std::string("lbl");
            //newVertices.ParallelWrite(outFileName, base);
            if(myrank == 0) printf("[End] Subgraph extraction\n");
            
            }
            
            std::vector<FullyDistVec<IT, IT>> toConcatenate(2); 
            toConcatenate[0] = prevVertices;
            toConcatenate[1] = newVertices;

            std::vector<FullyDistVec<IT, LBL>> toConcatenateLabels(2); 
            toConcatenateLabels[0] = prevVerticesLabels;
            toConcatenateLabels[1] = newVerticesLabels;
            

            prevVertices = Concatenate(toConcatenate);
            prevVerticesLabels = Concatenate(toConcatenateLabels);
            if(myrank == 0) printf("[End] Split: %d\n***\n", s);
        }

        for(IT s = 0; s < dvList.size(); s++){
            delete dvList[s];
            delete dvListLabels[s];
        }

    }
    MPI_Finalize();
    return 0;
}

