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
    if(argc < 13)
    {
        if(myrank == 0)
        {
            cout << "Usage: ./inc-pipeline -I <mm|triples> -M <MATRIX_FILENAME> -num-split <NUMBER OF SPLITS>\n";
            cout << "-I <INPUT FILE TYPE> (mm: matrix market, triples: (vtx1, vtx2, edge_weight) triples, default: mm)\n";
            cout << "-M <MATRIX FILE NAME>\n";
            cout << "-base <BASE OF MATRIX MARKET> (default:1)\n";
            cout << "-num-split <NUMBER OF SPLITS>\n";
            cout << "-out-prefix <OUTPUT PREFIX>\n";
            cout << "-alg <INCREMENTAL ALGORITHM>\n";
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

        string M11name = "";
        string M12name = "";
        string M21name = "";
        string M22name = "";
        string MINCname = "";
        string MSOname = "";
        string L11name = "";
        string L22name = "";
        string LOname = "";
        string COname = "";
        string CNOname = "";
        string ALGname = "";

        HipMCLParam incParam;
        InitParam(incParam);

        string outFileName = "";
        
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
            else if (strcmp(argv[i],"-num-split")==0)
            {
                nSplit = atoi(argv[i+1]);
                if(myrank == 0) printf("Number of splits: %d\n", nSplit);
            }
            else if (strcmp(argv[i],"-out-prefix")==0)
            {
                outPrefix = string(argv[i+1]);
                if(myrank == 0) printf("Output file prefix: %s\n", outPrefix.c_str());
            }
            else if (strcmp(argv[i],"-alg")==0)
            {
                ALGname = string(argv[i+1]);
                if(myrank == 0) printf("Incremental algorithm name: %s\n", ALGname.c_str());
            }
            else if(strcmp(argv[i],"-per-process-mem")==0){
                incParam.perProcessMem = atoi(argv[i+1]);
                if(myrank == 0) printf("Per process memory: %d GB\n", incParam.perProcessMem);
            }
        }

        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );

        if(myrank == 0) printf("Incremental pipeline simulation\n");

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
        SpParMat<IT, NT, DER> MS11(fullWorld);
        FullyDistVec<IT, LBL> L11(fullWorld); 
        FullyDistVec<IT, LBL> L22(fullWorld); 
        FullyDistVec<IT, IT> L11Num(fullWorld); // Temporary. To solve the inability of reading vector of text values
        FullyDistVec<IT, IT> L22Num(fullWorld); // Same

        FullyDistVec<IT, IT> prevVertices(*(dvList[0])); // Create a distributed vector to keep track of the vertices being considered at each incremental step
        FullyDistVec<IT, LBL> prevVerticesLabels(*(dvListLabels[0])); // Create a distributed vector to keep track of the vertex labels being considered at each incremental step

        for(int s = 1; s < nSplit; s++){
            MPI_Barrier(MPI_COMM_WORLD);
            if(myrank == 0) printf("[Start] Split: %d\n", s);

            FullyDistVec<IT, IT> newVertices(*(dvList[s]));
            FullyDistVec<IT, LBL> newVerticesLabels(*(dvListLabels[s]));

            IT totalLength = prevVertices.TotalLength() + newVertices.TotalLength();
            L22 = newVerticesLabels;
            L22Num = newVertices;
            L11 = prevVerticesLabels;
            L11Num = prevVertices;
            
            if(s == 1){
                M11name = outPrefix + std::string(".") + std::to_string(nSplit) + std::string(".") + std::to_string(s-1) + std::string(".m11.") + std::string("mtx");
                L11name = outPrefix + std::string(".") + std::to_string(nSplit) + std::string(".") + std::to_string(s-1) + std::string(".m11.") + std::string("lbl");
            }
            else{
                M11name = outPrefix + std::string(".") + std::to_string(nSplit) + std::string(".") + std::to_string(s-1) + std::string(".minc.") + std::string("mtx") + std::string(".") + ALGname;
                L11name = outPrefix + std::string(".") + std::to_string(nSplit) + std::string(".") + std::to_string(s-1) + std::string(".minc.") + std::string("lbl");
            }
            M12name = outPrefix + std::string(".") + std::to_string(nSplit) + std::string(".") + std::to_string(s) + std::string(".m12.") + std::string("mtx");
            M21name = outPrefix + std::string(".") + std::to_string(nSplit) + std::string(".") + std::to_string(s) + std::string(".m21.") + std::string("mtx");
            M22name = outPrefix + std::string(".") + std::to_string(nSplit) + std::string(".") + std::to_string(s) + std::string(".m22.") + std::string("mtx");
            L22name = outPrefix + std::string(".") + std::to_string(nSplit) + std::string(".") + std::to_string(s) + std::string(".m22.") + std::string("lbl");
            MINCname = outPrefix + std::string(".") + std::to_string(nSplit) + std::string(".") + std::to_string(s) + std::string(".minc.") + std::string("mtx") + std::string(".") + ALGname;
            MSOname = MINCname + std::string(".summary");
            LOname = outPrefix + std::string(".") + std::to_string(nSplit) + std::string(".") + std::to_string(s) + std::string(".minc.") + std::string("lbl");
            COname = outPrefix + std::string(".") + std::to_string(nSplit) + std::string(".") + std::to_string(s) + std::string(".") + ALGname;
            CNOname = outPrefix + std::string(".") + std::to_string(nSplit) + std::string(".") + std::to_string(s) + std::string(".m22.") + ALGname;

            SpParMat<IT, NT, DER> Minc(fullWorld);
            SpParMat<IT, NT, DER> MSO(fullWorld);
            FullyDistVec<IT, LBL> LOLbl(fullWorld, totalLength, LBL{});
            FullyDistVec<IT, IT> LONum(fullWorld, totalLength, 0); 
            FullyDistVec<IT, IT> permMap(fullWorld, totalLength, 0);
            FullyDistVec<IT, IT> isOld(fullWorld, totalLength, 0);
            FullyDistVec<IT, IT> CO(fullWorld, totalLength, 0); // Cluster assignment of new vertices
            FullyDistVec<IT, IT> CO22(fullWorld, M22.getnrow(), 0); // Cluster assignment of new vertices
            SpParMat<IT, NT, DER> MS22(fullWorld); // Summarized new subgraph

            //if( (s >= startSplit) && (s < endSplit))
            //for(int it=0; (it<50) && (s == 2); it++)
            {
                //if(myrank == 0) printf("It: %d\n", it);
                if(myrank == 0) printf("[Start] Subgraph extraction\n");
                //prevVertices.DebugPrint();
                //newVertices.DebugPrint();
                if(s == 1){
                    M11.FreeMemory();
                    t0 = MPI_Wtime();
                    M11 = M.SubsRef_SR <PTNTBOOL, PTBOOLNT> (prevVertices, prevVertices, false);
                    t1 = MPI_Wtime();
                    if(myrank == 0) printf("Time to extract M11: %lf\n", t1 - t0);
                    M11.PrintInfo();
                    M11.ParallelWriteMM(M11name, base);
                    prevVertices.ParallelWrite(L11name, base);
                }

                M12.FreeMemory();
                t0 = MPI_Wtime();
                M12 = M.SubsRef_SR <PTNTBOOL, PTBOOLNT> (prevVertices, newVertices, false);
                t1 = MPI_Wtime();
                if(myrank == 0) printf("Time to extract M12: %lf\n", t1 - t0);
                M12.PrintInfo();
                M12.ParallelWriteMM(M12name, base);

                M21.FreeMemory();
                t0 = MPI_Wtime();
                M21 = M.SubsRef_SR <PTNTBOOL, PTBOOLNT> (newVertices, prevVertices, false);
                t1 = MPI_Wtime();
                if(myrank == 0) printf("Time to extract M21: %lf\n", t1 - t0);
                M21.PrintInfo();
                M21.ParallelWriteMM(M21name, base);
                
                M22.FreeMemory();
                t0 = MPI_Wtime();
                M22 = M.SubsRef_SR <PTNTBOOL, PTBOOLNT> (newVertices, newVertices, false); // Get subgraph induced by newly added vertices in current step
                t1 = MPI_Wtime();
                if(myrank == 0) printf("Time to extract M22: %lf\n", t1 - t0);
                M22.PrintInfo();
                M22.ParallelWriteMM(M22name, base);
                newVertices.ParallelWrite(L22name, base);
                if(myrank == 0) printf("[End] Subgraph extraction\n");

                if(ALGname == "full"){
                    // If full graph clustering then do not make any modification to anything before constructing incremental graph
                    // Not even shuffle the vertex order
                    incParam.shuffleVertexOrder = false;
                    SpParHelper::Print("[Start] Preparing Minc\n");
                    t0 = MPI_Wtime();
                    PrepIncMat(M11, M12, M21, M22, L11, L22, Minc, LOLbl, isOld, permMap, incParam);
                    t1 = MPI_Wtime();
                    if(myrank == 0) printf("Time to prepare Minc: %lf\n", t1 - t0);
                    Minc.PrintInfo();
                    SpParHelper::Print("[End] Preparing Minc\n");
                    Minc.ParallelWriteMM(MINCname, base);
                    
                    M11 = Minc; // For full case, Minc would be M11 in next iteration

                    // Prepare parameters for clustering M22
                    incParam.summaryIter = 5;
                    incParam.maxIter = std::numeric_limits<int>::max(); // Arbitrary large number as maximum number of iterations. Run as many iterations as needed to converge;
                    
                    // Not necessary for full graph clustering, but still doing it for the sake of completeness
                    SpParHelper::Print("[Start] Clustering M22\n");
                    t0 = MPI_Wtime();
                    HipMCL(M22, incParam, CO22, MS22);
                    t1 = MPI_Wtime();
                    if(myrank == 0) printf("Time to find clusters in M22: %lf\n", t1 - t0);
                    MS22.PrintInfo();
                    SpParHelper::Print("[End] Clustering M22\n");

                    SpParHelper::Print("[Start] Clustering Minc\n");
                    t0 = MPI_Wtime();
                    IncrementalMCL(Minc, incParam, CO, MSO, isOld, Minc);
                    t1 = MPI_Wtime();
                    if(myrank == 0) printf("Time to find clusters in Minc: %lf\n", t1 - t0);
                    MSO.PrintInfo();
                    SpParHelper::Print("[End] Clustering Minc\n");
                    
                }
                else if (ALGname == "baseline"){
                    // Remove inter cluster edges from full matrix to be considered to be a summary matrix

                    // Summarize M22
                    incParam.summaryIter = 5; // Keep 5th iteration MCL state as summary
                    incParam.maxIter = std::numeric_limits<int>::max(); // Arbitrary large number as maximum number of iterations. Run as many iterations as needed to converge;
                    SpParHelper::Print("[Start] Clustering M22\n");
                    t0 = MPI_Wtime();
                    HipMCL(M22, incParam, CO22, MS22);
                    t1 = MPI_Wtime();
                    if(myrank == 0) printf("Time to find clusters in M22: %lf\n", t1 - t0);
                    MS22.PrintInfo();
                    SpParHelper::Print("[End] Clustering M22\n");

                    MS22.FreeMemory(); // Discard whatever summary has been saved
                    MS22 = M.SubsRef_SR <PTNTBOOL, PTBOOLNT> (newVertices, newVertices, false); // Save full version of M22 as MS22
                    RemoveInterClusterEdges(MS22, CO22); // Remove inter cluster edges to save as summary

                    // Prepare incremental matrix
                    incParam.normalizedAssign = false; // Because baseline means only inter cluster edges to be removed, all other edges should be kept untouched
                    incParam.shuffleVertexOrder = false; // Vertex order does not need to be shuffled as well
                    SpParHelper::Print("[Start] Preparing Minc\n");
                    t0 = MPI_Wtime();
                    PrepIncMat(M11, M12, M21, MS22, L11, L22, Minc, LOLbl, isOld, permMap, incParam); // Use summary of M22 in Minc preparation
                    t1 = MPI_Wtime();
                    if(myrank == 0) printf("Time to prepare Minc: %lf\n", t1 - t0);
                    Minc.PrintInfo();
                    SpParHelper::Print("[End] Preparing Minc\n");
                    Minc.ParallelWriteMM(MINCname, base);

                    // Prepare parameters for clustering Minc
                    incParam.summaryIter = 5; // Keep 5th iteration MCL state as summary
                    incParam.maxIter = std::numeric_limits<int>::max(); // Run as many iterations as needed for MCL to converge;
                    SpParHelper::Print("[Start] Clustering Minc\n");
                    t0 = MPI_Wtime();
                    IncrementalMCL(Minc, incParam, CO, MSO, isOld, Minc);
                    t1 = MPI_Wtime();
                    if(myrank == 0) printf("Time to find clusters in Minc: %lf\n", t1 - t0);
                    MSO.PrintInfo();
                    SpParHelper::Print("[End] Clustering Minc\n");

                    // For baseline case do not save summary here
                }
                else if (ALGname == "v1"){
                    // Maintain Markov state of 5th iteration as the summary for next incremental step

                    // Summarize M22
                    incParam.summaryIter = 5; // Keep 5th iteration MCL state as summary
                    incParam.maxIter = -1; // Run as many iterations as needed to get the summary 
                    SpParHelper::Print("[Start] Clustering M22\n");
                    t0 = MPI_Wtime();
                    HipMCL(M22, incParam, CO22, MS22);
                    t1 = MPI_Wtime();
                    if(myrank == 0) printf("Time to find clusters in M22: %lf\n", t1 - t0);
                    MS22.PrintInfo();
                    SpParHelper::Print("[End] Clustering M22\n");

                    // Prepare incremental matrix
                    incParam.shuffleVertexOrder = true; // Shuffle vertex order - to better load balance
                    incParam.normalizedAssign = true; // Normalize edge weights in accordance to the ratio of new and old vertices
                    SpParHelper::Print("[Start] Preparing Minc\n");
                    t0 = MPI_Wtime();
                    PrepIncMat(M11, M12, M21, MS22, L11, L22, Minc, LOLbl, isOld, permMap, incParam); // Use summary of M22 in Minc preparation
                    t1 = MPI_Wtime();
                    if(myrank == 0) printf("Time to prepare Minc: %lf\n", t1 - t0);
                    Minc.PrintInfo();
                    SpParHelper::Print("[End] Preparing Minc\n");
                    Minc.ParallelWriteMM(MINCname, base);

                    // Prepare parameters for clustering Minc
                    incParam.summaryIter = 5; // Keep 5th iteration MCL state as summary
                    incParam.maxIter = std::numeric_limits<int>::max(); // Run as many iterations as needed for MCL to converge;
                    SpParHelper::Print("[Start] Clustering Minc\n");
                    t0 = MPI_Wtime();
                    IncrementalMCL(Minc, incParam, CO, MSO, isOld, Minc);
                    t1 = MPI_Wtime();
                    if(myrank == 0) printf("Time to find clusters in Minc: %lf\n", t1 - t0);
                    MSO.PrintInfo();
                    SpParHelper::Print("[End] Clustering Minc\n");

                    M11 = MSO; // For v1 case, summary of Minc would be M11 in next iteration
                }
                else if (ALGname == "v2"){
                    // Maintain Markov state of the iteration when nnz drops to 70% of starting nnz as the summary for next incremental step

                    // Summarize M22
                    // summaryIter and summaryThresholdNNZ should be defined together
                    incParam.summaryIter = 5; 
                    incParam.summaryThresholdNNZ = 0.7; 
                    incParam.maxIter = -1; // Run as many iterations as needed to get the summary 
                    SpParHelper::Print("[Start] Clustering M22\n");
                    t0 = MPI_Wtime();
                    HipMCL(M22, incParam, CO22, MS22);
                    t1 = MPI_Wtime();
                    if(myrank == 0) printf("Time to find clusters in M22: %lf\n", t1 - t0);
                    MS22.PrintInfo();
                    SpParHelper::Print("[End] Clustering M22\n");

                    // Prepare incremental matrix
                    incParam.shuffleVertexOrder = true; // Shuffle vertex order - to better load balance
                    incParam.normalizedAssign = true; // Normalize edge weights in accordance to the ratio of new and old vertices
                    SpParHelper::Print("[Start] Preparing Minc\n");
                    t0 = MPI_Wtime();
                    PrepIncMat(M11, M12, M21, MS22, L11, L22, Minc, LOLbl, isOld, permMap, incParam); // Use summary of M22 in Minc preparation
                    t1 = MPI_Wtime();
                    if(myrank == 0) printf("Time to prepare Minc: %lf\n", t1 - t0);
                    Minc.PrintInfo();
                    SpParHelper::Print("[End] Preparing Minc\n");
                    Minc.ParallelWriteMM(MINCname, base);

                    // Prepare parameters for clustering Minc
                    // summaryIter and summaryThresholdNNZ should be defined together
                    incParam.summaryIter = 10; 
                    incParam.summaryThresholdNNZ = 0.7; 
                    incParam.maxIter = std::numeric_limits<int>::max(); // Run as many iterations as needed for MCL to converge;
                    SpParHelper::Print("[Start] Clustering Minc\n");
                    t0 = MPI_Wtime();
                    IncrementalMCL(Minc, incParam, CO, MSO, isOld, Minc);
                    t1 = MPI_Wtime();
                    if(myrank == 0) printf("Time to find clusters in Minc: %lf\n", t1 - t0);
                    MSO.PrintInfo();
                    SpParHelper::Print("[End] Clustering Minc\n");

                    M11 = MSO; // For v2 case, summary of Minc would be M11 in next iteration
                }
                else if (ALGname == "v3"){
                    // Maintain Markov state of the iteration when nnz drops to 70% of starting nnz as the summary for next incremental step
                    // Plus, use selective prunning at each MCL expansion (remove prevent old-old and new-new edge creation)

                    // Summarize M22
                    // summaryIter and summaryThresholdNNZ should be defined together
                    incParam.summaryIter = 5; 
                    incParam.summaryThresholdNNZ = 0.7; 
                    incParam.maxIter = -1; // Run as many iterations as needed to get the summary 
                    SpParHelper::Print("[Start] Clustering M22\n");
                    t0 = MPI_Wtime();
                    HipMCL(M22, incParam, CO22, MS22);
                    t1 = MPI_Wtime();
                    if(myrank == 0) printf("Time to find clusters in M22: %lf\n", t1 - t0);
                    MS22.PrintInfo();
                    SpParHelper::Print("[End] Clustering M22\n");

                    // Prepare incremental matrix
                    incParam.shuffleVertexOrder = true; // Shuffle vertex order - to better load balance
                    incParam.normalizedAssign = true; // Normalize edge weights in accordance to the ratio of new and old vertices
                    SpParHelper::Print("[Start] Preparing Minc\n");
                    t0 = MPI_Wtime();
                    PrepIncMat(M11, M12, M21, MS22, L11, L22, Minc, LOLbl, isOld, permMap, incParam); // Use summary of M22 in Minc preparation
                    t1 = MPI_Wtime();
                    if(myrank == 0) printf("Time to prepare Minc: %lf\n", t1 - t0);
                    Minc.PrintInfo();
                    SpParHelper::Print("[End] Preparing Minc\n");
                    Minc.ParallelWriteMM(MINCname, base);

                    // Prepare parameters for clustering Minc
                    // summaryIter and summaryThresholdNNZ should be defined together
                    incParam.summaryIter = 10; 
                    incParam.summaryThresholdNNZ = 0.7; 
                    incParam.selectivePruneThreshold = 0.1;
                    SpParMat<IT, NT, DER> SelectivePruneMask(Minc);
                    incParam.maxIter = std::numeric_limits<int>::max(); // Run as many iterations as needed for MCL to converge;
                    SpParHelper::Print("[Start] Clustering Minc\n");
                    t0 = MPI_Wtime();
                    IncrementalMCL(Minc, incParam, CO, MSO, isOld, SelectivePruneMask);
                    t1 = MPI_Wtime();
                    if(myrank == 0) printf("Time to find clusters in Minc: %lf\n", t1 - t0);
                    MSO.PrintInfo();
                    SpParHelper::Print("[End] Clustering Minc\n");

                    M11 = MSO; // For v3 case, summary of Minc would be M11 in next iteration
                }
                else if ((ALGname == "v31") || (ALGname == "v32") || (ALGname == "v33")){
                    // This version is a variation of v3
                    // These variations runs full HipMCL upto certain portion of the incremental pipeline (40%, 60% or 80% depending on v31, v32 or v33)
                    // Then after the certain point starts clustering incrementally  by applying the techniques of v3
                    int cutOff = int(nSplit * 0.4); 
                    if(ALGname == "v31") cutOff = int(nSplit * 0.4); 
                    if(ALGname == "v32") cutOff = int(nSplit * 0.6); 
                    if(ALGname == "v33") cutOff = int(nSplit * 0.8); 
                    
                    if(s <= cutOff){
                        incParam.shuffleVertexOrder = true;
                        SpParHelper::Print("[Start] Preparing Minc\n");
                        t0 = MPI_Wtime();
                        PrepIncMat(M11, M12, M21, M22, L11, L22, Minc, LOLbl, isOld, permMap, incParam);
                        t1 = MPI_Wtime();
                        if(myrank == 0) printf("Time to prepare Minc: %lf\n", t1 - t0);
                        Minc.PrintInfo();
                        SpParHelper::Print("[End] Preparing Minc\n");
                        Minc.ParallelWriteMM(MINCname, base);
                        
                        M11 = Minc; // For full case, Minc would be M11 in next iteration

                        // Prepare parameters for clustering M22
                        incParam.summaryIter = 5;
                        incParam.maxIter = -1; // As many iterations as it takes to get summary
                        SpParHelper::Print("[Start] Clustering M22\n");
                        t0 = MPI_Wtime();
                        HipMCL(M22, incParam, CO22, MS22);
                        t1 = MPI_Wtime();
                        if(myrank == 0) printf("Time to find clusters in M22: %lf\n", t1 - t0);
                        MS22.PrintInfo();
                        SpParHelper::Print("[End] Clustering M22\n");

                        incParam.summaryIter = 5;
                        incParam.maxIter = std::numeric_limits<int>::max(); // Run as many iterations as needed for MCL to converge;
                        SpParHelper::Print("[Start] Clustering Minc\n");
                        t0 = MPI_Wtime();
                        IncrementalMCL(Minc, incParam, CO, MSO, isOld, Minc);
                        t1 = MPI_Wtime();
                        if(myrank == 0) printf("Time to find clusters in Minc: %lf\n", t1 - t0);
                        MSO.PrintInfo();
                        SpParHelper::Print("[End] Clustering Minc\n");
                    }
                    else{
                        incParam.summaryIter = 5; 
                        incParam.summaryThresholdNNZ = 0.7; 
                        incParam.maxIter = -1; // Run as many iterations as needed to get the summary 
                        SpParHelper::Print("[Start] Clustering M22\n");
                        t0 = MPI_Wtime();
                        HipMCL(M22, incParam, CO22, MS22);
                        t1 = MPI_Wtime();
                        if(myrank == 0) printf("Time to find clusters in M22: %lf\n", t1 - t0);
                        MS22.PrintInfo();
                        SpParHelper::Print("[End] Clustering M22\n");

                        // Prepare incremental matrix
                        incParam.shuffleVertexOrder = true; // Shuffle vertex order - to better load balance
                        incParam.normalizedAssign = true; // Normalize edge weights in accordance to the ratio of new and old vertices
                        SpParHelper::Print("[Start] Preparing Minc\n");
                        t0 = MPI_Wtime();
                        PrepIncMat(M11, M12, M21, MS22, L11, L22, Minc, LOLbl, isOld, permMap, incParam); // Use summary of M22 in Minc preparation
                        t1 = MPI_Wtime();
                        if(myrank == 0) printf("Time to prepare Minc: %lf\n", t1 - t0);
                        Minc.PrintInfo();
                        SpParHelper::Print("[End] Preparing Minc\n");
                        Minc.ParallelWriteMM(MINCname, base);

                        // Prepare parameters for clustering Minc
                        // summaryIter and summaryThresholdNNZ should be defined together
                        incParam.summaryIter = 10; 
                        incParam.summaryThresholdNNZ = 0.7; 
                        incParam.selectivePruneThreshold = 0.1;
                        SpParMat<IT, NT, DER> SelectivePruneMask(Minc);
                        incParam.maxIter = std::numeric_limits<int>::max(); // Run as many iterations as needed for MCL to converge;
                        SpParHelper::Print("[Start] Clustering Minc\n");
                        t0 = MPI_Wtime();
                        IncrementalMCL(Minc, incParam, CO, MSO, isOld, SelectivePruneMask);
                        t1 = MPI_Wtime();
                        if(myrank == 0) printf("Time to find clusters in Minc: %lf\n", t1 - t0);
                        MSO.PrintInfo();
                        SpParHelper::Print("[End] Clustering Minc\n");

                        M11 = MSO; // For v3 case, summary of Minc would be M11 in next iteration

                    }
                }
                WriteMCLClusters(CNOname, CO22, L22);
                WriteMCLClusters(COname, CO, LOLbl); // Write output clusters before undoing vertex shuffle

                IT aLocLen = LOLbl.LocArrSize();
                for (IT i = 0; i < aLocLen; i++){
                    std::string strLbl(LOLbl.GetLocalElement(i).data());
                    IT numLbl = atoi(strLbl.c_str());
                    LONum.SetLocalElement(i, numLbl);
                }
                
                if(ALGname == "baseline"){
                    // Do it before everything moves back to original order
                    MSO.FreeMemory();
                    MSO = M.SubsRef_SR <PTNTBOOL, PTBOOLNT> (LONum, LONum, false); // Save full version of M22 as MS22
                    RemoveInterClusterEdges(MSO, CO); // Remove inter cluster edges to save as summary
                    M11 = MSO;
                }

                if(incParam.shuffleVertexOrder){
                    // Reorder the vertex order of whatever will be used as M11 in next iteration
                    FullyDistVec<IT,IT> iota(fullWorld);
                    iota.iota(totalLength, 0);
                    SpParMat<IT, NT, DER> P = SpParMat<IT,NT,DER>(totalLength, totalLength, iota, iota, 1.0, false); // Identity
                    (P)(permMap,iota,true); // Row permute matrix that caused vertex order to be shuffled
                    SpParMat<IT, NT, DER> Q(P);
                    Q.Transpose(); // Column permute matrix
                    
                    FullyDistSpVec<IT, IT> iotaSp(iota); 
                    FullyDistSpVec<IT, IT> revPermMapSp = SpMV<PTFF>(Q, iotaSp, false);
                    FullyDistVec<IT, IT> revPermMap(revPermMapSp);

                    M11(revPermMap, revPermMap, true);
                    MSO(revPermMap, revPermMap, true);

                    FullyDistSpVec<IT, IT> LONumSp(LONum); 
                    FullyDistSpVec<IT, IT> origLONumSp = SpMV<PTFF>(Q, LONumSp, false);
                    LONum = FullyDistVec<IT, IT>(origLONumSp);
                }

                MSO.ParallelWriteMM(MSOname, base);
                LONum.ParallelWrite(LOname, base);
            
            }
        
            const std::vector<IT> LONumLoc = LONum.GetLocVec();
            std::vector<LBL> LOLblLoc(LONumLoc.size());

            for(int i = 0; i < LONumLoc.size(); i++){
                std::string labelStr = std::to_string(LONumLoc[i]); 
                for ( IT j = 0; (j < labelStr.length()) && (j < MAXVERTNAME); j++){
                    LOLblLoc[i][j] = labelStr[j]; 
                }
            }
            
            prevVertices = LONum;
            prevVerticesLabels = FullyDistVec<IT, LBL>(LOLblLoc, fullWorld);
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

