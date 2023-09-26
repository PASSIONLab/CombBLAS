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
typedef SpDCCols < int64_t, double > DER;
typedef PlusTimesSRing<double, double> PTFF;
typedef PlusTimesSRing<int64_t, int64_t> PTII;
typedef PlusTimesSRing<bool, double> PTBOOLNT;
typedef PlusTimesSRing<double, bool> PTNTBOOL;
typedef std::array<char, MAXVERTNAME> LBL;


void convNumToLbl(FullyDistVec<IT, IT>& distNum, FullyDistVec<IT, LBL>& distLbl){
    const std::vector<IT> locNum = distNum.GetLocVec();

    for(int i = 0; i < locNum.size(); i++){
        std::string labelStr = std::to_string(locNum[i]); 
        LBL labelArr{};
        for ( IT j = 0; (j < labelStr.length()) && (j < MAXVERTNAME); j++){
            labelArr[j] = labelStr[j]; 
        }
        distLbl.SetLocalElement(i, labelArr);
    }
}

void convLblToNum(FullyDistVec<IT, LBL>& distLbl, FullyDistVec<IT, IT>& distNum){
    const std::vector<LBL> locLbl = distLbl.GetLocVec();

    for(int i = 0; i < locLbl.size(); i++){
        std::string labelStr(locLbl[i].data());
        IT labelNum = atoi(labelStr.c_str());
        distNum.SetLocalElement(i, labelNum);
    }
}

int calcExtraMemoryRequirement(SpParMat<IT, NT, DER>&A){
    int nprocs, myrank, nthreads = 1;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    IT nnz = A.getnnz();
    IT n = A.getnrow();
    IT matrixExtraMemory = nnz * (sizeof(IT)*2+sizeof(NT)) * 6; // 6 equivalent copies at various places
    IT vectorExtraMemory = n * sizeof(IT) * 13; // 13 equivalent copies at various places
    IT extraMemory = matrixExtraMemory + vectorExtraMemory;
    double perProcExtraMem = double(matrixExtraMemory + vectorExtraMemory) / (double)nprocs;
    double perProcExtraMemGB = double(perProcExtraMem) / (1024 * 1024 * 1024);
    
    return ceil( perProcExtraMemGB ); // Per process extra memory in GB
}

void reversePermutation(SpParMat<IT, NT, DER>&M, FullyDistVec<IT, LBL>& lbl, FullyDistVec<IT, IT>& permMap){
    shared_ptr<CommGrid> fullWorld;
    fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );
    IT totalLength = lbl.TotalLength();

    FullyDistVec<IT, IT> num(fullWorld, totalLength, 0);
    convLblToNum(lbl, num);

    // Reorder the vertex order of whatever will be used as M11 in next iteration
    FullyDistVec<IT,IT> iota(fullWorld);
    iota.iota(totalLength, 0);
    SpParMat<IT, NT, DER> P = SpParMat<IT,NT,DER>(totalLength, totalLength, iota, iota, 1.0, false); // Identity
    //SpParMat<IT, IT, DER> P = SpParMat<IT,IT,DER>(totalLength, totalLength, iota, iota, 1, false); // Identity
    (P)(permMap,iota,true); // Row permute matrix that caused vertex order to be shuffled
    SpParMat<IT, NT, DER> Q(P);
    //SpParMat<IT, IT, DER> Q(P);
    Q.Transpose(); // Column permute matrix because the permutation was symmetric
    
    FullyDistSpVec<IT, IT> iotaSp(iota); 
    FullyDistSpVec<IT, IT> revPermMapSp = SpMV<PTFF>(Q, iotaSp, false);
    //FullyDistSpVec<IT, IT> revPermMapSp = SpMV<PTII>(Q, iotaSp, false);
    FullyDistVec<IT, IT> revPermMap(revPermMapSp);

    M(revPermMap, revPermMap, true);

    FullyDistSpVec<IT, IT> numSp(num); 
    FullyDistSpVec<IT, IT> origNumSp = SpMV<PTFF>(Q, numSp, false);
    //FullyDistSpVec<IT, IT> origNumSp = SpMV<PTII>(Q, numSp, false);
    num = FullyDistVec<IT, IT>(origNumSp);
    convNumToLbl(num, lbl);
}

int main(int argc, char* argv[])
{
    int nprocs, myrank, nthreads = 1;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    if(myrank == 0) printf("Incremental pipeline simulation\n");
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
    if(argc < 9)
    {
        if(myrank == 0)
        {
            cout << "Usage: ./inc-pipeline --input-dir <DIRECTORY OF INPUT MATRICES> --infile-prefix <PREFIX OF INPUT MATRICES> --num-split <NUMBER OF SPLITS> --output-dir <DIRECTORY OF OUTPUT MATRICES>\n";
            cout << "--input-dir <DIRECTORY OF INPUT MATRICES>\n";
            cout << "--infile-prefix <PREFIX OF INPUT FILES>\n";
            cout << "--num-split <NUMBER OF SPLITS>\n";
            cout << "--output-dir <DIRECTORY OF OUTPUT MATRICES>\n";
            cout << "--incremental-start <STEP FROM WHICH INCREMENTAL KICKS IN>\n";
            cout << "--summary-threshold <PERCENTAGE OF STARTING NNZ WHEN SUMMARY IS SAVED>\n";
            cout << "--selective-prune-threshold <SELECTIVE PRUNE THRESHOLD>\n";
            cout << "--hipmcl-before-incremental <0 | 1>\n";
        }
        MPI_Finalize();
        return -1;
    }
    else{
        int base = 1; // Keep 1 all the time, because in this application we will only handle 1 base matrix market files
        string inputDir = "";
        string infilePrefix = "";
        string outputDir = "";
        string outPrefix = "";
        int nSplit = 2;
        int incStartStep = 2;
        int summaryThreshold = 70; // 70%
        int normalizedAssign = 1;
        int shuffleVertexOrder = 1;
        int perProcessMem = 1;
        int selectivePruneThreshold = 10; //10%
        int hipmclBeforeIncremental = 0; 

        string M11name = ""; string M12name = ""; string M21name = ""; string M22name = ""; string L11name = ""; string L22name = "";

        
        for (int i = 1; i < argc; i++)
        {
            if (strcmp(argv[i],"--input-dir")==0){
                inputDir = string(argv[i+1]);
                if(myrank == 0) printf("Input directory: %s\n", inputDir.c_str());
            }
            else if (strcmp(argv[i],"--infile-prefix")==0){
                infilePrefix = string(argv[i+1]);
                if(myrank == 0) printf("Input file prefix: %s\n", infilePrefix.c_str());
                infilePrefix = string(argv[i+1]);
            }
            else if (strcmp(argv[i],"--output-dir")==0){
                outputDir = string(argv[i+1]);
                if(myrank == 0) printf("Output directory: %s\n", outputDir.c_str());
            }
            else if (strcmp(argv[i],"--num-split")==0)
            {
                nSplit = atoi(argv[i+1]);
                if(myrank == 0) printf("Number of splits: %d\n", nSplit);
            }
            else if (strcmp(argv[i],"--incremental-start")==0)
            {
                incStartStep = atoi(argv[i+1]);
                if(myrank == 0) printf("Incremental clustering kick-in step: %d\n", incStartStep);
            }
            else if (strcmp(argv[i],"--selective-prune-threshold")==0)
            {
                selectivePruneThreshold = atoi(argv[i+1]);
                if(myrank == 0) printf("Selective prune threshold: %d\n", selectivePruneThreshold);
            }
            else if (strcmp(argv[i],"--summary-threshold")==0)
            {
                summaryThreshold = atoi(argv[i+1]);
                if(myrank == 0) printf("Summary threshold: %d\n", summaryThreshold);
            }
            else if(strcmp(argv[i],"--per-process-mem")==0){
                perProcessMem = atoi(argv[i+1]);
                if(myrank == 0) printf("Per process memory: %d GB\n", perProcessMem);
            }
            else if(strcmp(argv[i],"--hipmcl-before-incremental")==0){
                hipmclBeforeIncremental = atoi(argv[i+1]);
                if(myrank == 0) printf("Run HipMCL before incremental kicks in: %d\n", hipmclBeforeIncremental);
            }
        }

        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );
        
        double t0, t1, t2, t3, t4, t5;
        
        SpParMat<IT, NT, DER> M11(fullWorld);
        SpParMat<IT, NT, DER> M12(fullWorld);
        SpParMat<IT, NT, DER> M21(fullWorld);
        SpParMat<IT, NT, DER> M22(fullWorld);
        FullyDistVec<IT, LBL> L11(fullWorld); 
        FullyDistVec<IT, LBL> L22(fullWorld); 
        FullyDistVec<IT, IT> L11Num(fullWorld); // Temporary. To solve the inability of reading vector of text values
        FullyDistVec<IT, IT> L22Num(fullWorld); // Same

        for(int s = 1; s < nSplit; s++){
            /*
             * Initialize parameters at the start of each incremental step simulation
             * */
            HipMCLParam incParam; 
            InitParam(incParam);

            MPI_Barrier(MPI_COMM_WORLD);
            if(myrank == 0) printf("[Start] Split: %d\n", s);
            
            if (s == 1){
                M11name = inputDir + infilePrefix + std::string(".") + std::to_string(nSplit) + std::string(".") + std::to_string(s-1) + std::string(".m11.") + std::string("mtx");
                L11name = inputDir + infilePrefix + std::string(".") + std::to_string(nSplit) + std::string(".") + std::to_string(s-1) + std::string(".m11.") + std::string("lbl");
            }
            else{
                M11name = "";
                L11name = "";
            }
            M12name = inputDir + infilePrefix + std::string(".") + std::to_string(nSplit) + std::string(".") + std::to_string(s) + std::string(".m12.") + std::string("mtx");
            M21name = inputDir + infilePrefix + std::string(".") + std::to_string(nSplit) + std::string(".") + std::to_string(s) + std::string(".m21.") + std::string("mtx");
            M22name = inputDir + infilePrefix + std::string(".") + std::to_string(nSplit) + std::string(".") + std::to_string(s) + std::string(".m22.") + std::string("mtx");
            L22name = inputDir + infilePrefix + std::string(".") + std::to_string(nSplit) + std::string(".") + std::to_string(s) + std::string(".m22.") + std::string("lbl");

            if(myrank == 0) printf("[Start] Subgraph extraction\n");
            
            if(M11name != ""){
                M11.FreeMemory();
                t0 = MPI_Wtime();
                M11.ParallelReadMM( M11name, base, maximum<NT>() );
                t1 = MPI_Wtime();
                if(myrank == 0) printf("Time to extract M11: %lf\n", t1 - t0);
                M11.PrintInfo();
            }

            if (L11name != ""){
                t0 = MPI_Wtime();
                L11Num.ParallelRead( L11name, base, maximum<IT>() );
                L11 = FullyDistVec<IT, LBL>(fullWorld, L11Num.TotalLength(), LBL{}); 
                convNumToLbl(L11Num, L11);
                t1 = MPI_Wtime();
                if(myrank == 0) printf("Time to read L11: %lf\n", t1 - t0);
            }

            M12.FreeMemory();
            t0 = MPI_Wtime();
            M12.ParallelReadMM(M12name, base, maximum<NT>());
            t1 = MPI_Wtime();
            if(myrank == 0) printf("Time to extract M12: %lf\n", t1 - t0);
            M12.PrintInfo();

            M21.FreeMemory();
            t0 = MPI_Wtime();
            M21.ParallelReadMM(M21name, base, maximum<NT>());
            t1 = MPI_Wtime();
            if(myrank == 0) printf("Time to extract M21: %lf\n", t1 - t0);
            M21.PrintInfo();
            
            M22.FreeMemory();
            t0 = MPI_Wtime();
            M22.ParallelReadMM(M22name, base, maximum<NT>());
            t1 = MPI_Wtime();
            if(myrank == 0) printf("Time to extract M22: %lf\n", t1 - t0);
            M22.PrintInfo();

            t0 = MPI_Wtime();
            L22Num.ParallelRead( L22name, base, maximum<IT>() );
            L22 = FullyDistVec<IT, LBL>(fullWorld, L22Num.TotalLength(), LBL{}); 
            convNumToLbl(L22Num, L22);
            t1 = MPI_Wtime();
            if(myrank == 0) printf("Time to read L22: %lf\n", t1 - t0);

            if(myrank == 0) printf("[End] Subgraph extraction\n");
            
            IT totalLength = L11.TotalLength() + L22.TotalLength();

            /* 
             * Find clusters in the subgraph induced by new nodes
             * and summarize M22 in the process
             * */
            if (s >= incStartStep){
                FullyDistVec<IT, IT> CO22(fullWorld, M22.getnrow(), 0); // Cluster assignment of new vertices
                SpParMat<IT, NT, DER> MS22(fullWorld); // Summarized new subgraph

                // Find clusters in M22
                // Summarize M22 into MS22
                incParam.summaryIter = 5; // Keep exactly 5th iteration MCL state as summary
                //incParam.maxIter = std::numeric_limits<int>::max(); // Arbitrary large number as maximum number of iterations. Run as many iterations as needed to converge;
                incParam.maxIter = -1; 

                SpParHelper::Print("[Start] Clustering M22\n");

                t0 = MPI_Wtime();
                HipMCL(M22, incParam, CO22, MS22);
                t1 = MPI_Wtime();
                if(myrank == 0) printf("Time to find clusters in M22: %lf\n", t1 - t0);

                MS22.PrintInfo();
                M22 = MS22; // Replace M22 with MS22 as that will be used to prepare Minc

                SpParHelper::Print("[End] Clustering M22\n");

                MS22.FreeMemory();
            }
            else{
                // Let it pass. Do not do anything.
            }

            /*
             * Prepare incremental matrix
             * */
            SpParMat<IT, NT, DER> Minc(fullWorld);
            FullyDistVec<IT, LBL> LOLbl(fullWorld, totalLength, LBL{});
            FullyDistVec<IT, IT> permMap(fullWorld, totalLength, 0);
            FullyDistVec<IT, IT> isOld(fullWorld, totalLength, 0);
            
            incParam.shuffleVertexOrder = true; // Always shuffle vertices in incremental pipeline
            incParam.maxIter = 100; 
            if (s < incStartStep){
                // Treat each incremental step as full graph clustering
                // Hence, do not make any modification to edge weights
                incParam.normalizedAssign = false; 
                incParam.shuffleVertexOrder = true; // Always shuffle vertices in incremental pipeline
            }
            else{
                // Normalize nnz before preparing Minc
                incParam.normalizedAssign = true; 
                incParam.shuffleVertexOrder = true; // Always shuffle vertices in incremental pipeline
            }
            SpParHelper::Print("[Start] Preparing Minc\n");
            t0 = MPI_Wtime();
            PrepIncMat(M11, M12, M21, M22, L11, L22, Minc, LOLbl, isOld, permMap, incParam); 
            t1 = MPI_Wtime();
            if(myrank == 0) printf("Time to prepare Minc: %lf\n", t1 - t0);
            Minc.PrintInfo();
            SpParHelper::Print("[End] Preparing Minc\n");

            M11.FreeMemory();
            M12.FreeMemory();
            M21.FreeMemory();
            M22.FreeMemory();

            // Consider extra memory requirement to store the intermediate copies needed for incremental
            int perProcessExtraMem = calcExtraMemoryRequirement(Minc);
            if(myrank == 0) cout << "Per proc extra mem: " << perProcessExtraMem  << " GB" << endl;
            incParam.perProcessMem = perProcessMem - perProcessExtraMem;

            if (s < incStartStep-1){
                // This exact copy will be used as M11 for next incremental step
                M11 = Minc; 
            }
            else{
                // Do nothing at this moment.
                // Appropriate M11 for next incremental step will be figured out later.
            }

            string COname = outputDir + std::to_string(s) + std::string(".") + std::string("inc");
            SpParMat<IT, NT, DER> MSO(fullWorld);
            FullyDistVec<IT, IT> CO(fullWorld, totalLength, 0); // Cluster assignment of new vertices

            if (s >= incStartStep){
                incParam.summaryIter = 10; 
                incParam.summaryThresholdNNZ = double(summaryThreshold)/100; 
                incParam.selectivePruneThreshold = double(selectivePruneThreshold)/100;
                //incParam.maxIter = std::numeric_limits<int>::max(); // Run as many iterations as needed for MCL to converge;

                SpParMat<IT, NT, DER> SelectivePruneMask(Minc);

                SpParHelper::Print("[Start] Clustering Minc\n");

                t0 = MPI_Wtime();
                IncrementalMCL(Minc, incParam, CO, MSO, isOld, SelectivePruneMask);
                t1 = MPI_Wtime();
                if(myrank == 0) printf("Time to find clusters in Minc: %lf\n", t1 - t0);

                MSO.PrintInfo();
                M11 = MSO; // Use the obtained summary as M11 for the next incremental step

                SpParHelper::Print("[End] Clustering Minc\n");
                if(myrank == 0) printf("Writing clusters to file: %s\n", COname.c_str());
                WriteMCLClusters(COname, CO, LOLbl); // Write output clusters before undoing vertex shuffle
            }
            else{
                if ( (s == incStartStep-1) || (hipmclBeforeIncremental == 1) ){
                    incParam.summaryIter = 5; // Save summary exactly at 5th step
                    incParam.summaryThresholdNNZ = -1.0; // Effectively turn off saving summary based on nnz 
                    incParam.selectivePruneThreshold = -1.0; // Effectively turn of selective prunning

                    SpParHelper::Print("[Start] Clustering Minc\n");

                    t0 = MPI_Wtime();
                    //IncrementalMCL(Minc, incParam, CO, MSO, isOld, Minc); // Passing Minc as selective prune mask to avoid compilation issue. No effect as selective prun is turned off in incParam
                    HipMCL(Minc, incParam, CO, MSO);
                    t1 = MPI_Wtime();
                    if(myrank == 0) printf("Time to find clusters in Minc: %lf\n", t1 - t0);

                    MSO.PrintInfo();

                    SpParHelper::Print("[End] Clustering Minc\n");

                    if(myrank == 0) printf("Writing clusters to file: %s\n", COname.c_str());
                    WriteMCLClusters(COname, CO, LOLbl); // Write output clusters before undoing vertex shuffle
                }
                if (s == incStartStep -1){
                    M11 = MSO; // Use the obtained summary as M11 for the next incremental step
                }
            }
            Minc.FreeMemory(); // Not needed anymore
            MSO.FreeMemory(); // Not needed anymore

            if(incParam.shuffleVertexOrder){
                reversePermutation(M11, LOLbl, permMap);
            }

            L11 = LOLbl;


        
            if(myrank == 0) printf("[End] Split: %d\n***\n", s);
        }

    }
    MPI_Finalize();
    return 0;
}

