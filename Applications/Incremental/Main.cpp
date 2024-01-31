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
#ifdef THREADED
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
#endif
    if(myrank == 0)
    {
        cout << "Process Grid (p x p x t): " << sqrt(nprocs) << " x " << sqrt(nprocs) << " x " << nthreads << endl;
        cout << "argc " << argc << endl;
    }
    if(argc < 25)
    {
        if(myrank == 0)
        {
            cout << "Usage: ./inc\n";
            cout << "--M11 <M11 MATRIX FILE NAME> (required)\n";
            cout << "--M12 <M12 MATRIX FILE NAME> (required)\n";
            cout << "--M21 <M21 MATRIX FILE NAME> (required)\n";
            cout << "--M22 <M22 MATRIX FILE NAME> (required)\n";
            cout << "--L11 <M11 VERTEX LABEL> (required)\n";
            cout << "--L22 <M22 VERTEX LABEL> (required)\n";
            cout << "--summary-threshold <PERCENTAGE OF STARTING NNZ WHEN SUMMARY IS SAVED> (required)\n";
            cout << "--selective-prune-threshold <SELECTIVE PRUNE THRESHOLD> (required)\n";
            cout << "--inc-mat-out <FILENAME TO STORE INCREMENTAL MATRIX>\n";
            cout << "--summary-out <FILENAME TO STORE SUMMARY MATRIX> (required)\n";
            cout << "--cluster-out <FILENAME TO STORE OUTPUT CLUSTERS> (required)\n";
            cout << "--label-out <FILENAME TO STORE VERTEX LABELS OF SUMMARY GRAPH> (required)\n";
            cout << "--alg <ALGORITHM TO USE> (full | inc) (required)\n";
        }
    }
    else{
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
        int base = 1;
        bool isMatrixMarket = true;
        int summaryThreshold = 70; // 70%
        int perProcessMem = 1;
        int selectivePruneThreshold = 10; //10%

        HipMCLParam incParam;
        InitParam(incParam);
        
        for (int i = 1; i < argc; i++)
        {
            if (strcmp(argv[i],"--M11")==0)
            {
                M11name = string(argv[i+1]);
                if(myrank == 0) printf("M11 Matrix filename: %s\n", M11name.c_str());
            }
            else if (strcmp(argv[i],"--M12")==0)
            {
                M12name = string(argv[i+1]);
                if(myrank == 0) printf("M12 Matrix filename: %s\n", M12name.c_str());
            }
            else if (strcmp(argv[i],"--M21")==0)
            {
                M21name = string(argv[i+1]);
                if(myrank == 0) printf("M21 Matrix filename: %s\n", M21name.c_str());
            }
            else if (strcmp(argv[i],"--M22")==0)
            {
                M22name = string(argv[i+1]);
                if(myrank == 0) printf("M22 Matrix filename: %s\n", M22name.c_str());
            }
            else if (strcmp(argv[i],"--L11")==0)
            {
                L11name = string(argv[i+1]);
                if(myrank == 0) printf("M11 vertex label filename: %s\n", L11name.c_str());
            }
            else if (strcmp(argv[i],"--L22")==0)
            {
                L22name = string(argv[i+1]);
                if(myrank == 0) printf("M22 vertex label filename: %s\n", L22name.c_str());
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
            else if (strcmp(argv[i],"--inc-mat-out")==0)
            {
                MINCname = string(argv[i+1]);
                if(myrank == 0) printf("Incremental matrix filename: %s\n", MINCname.c_str());
            }
            else if (strcmp(argv[i],"--summary-out")==0)
            {
                MSOname = string(argv[i+1]);
                if(myrank == 0) printf("Output Summary Matrix filename: %s\n", MSOname.c_str());
            }
            else if (strcmp(argv[i],"--label-out")==0)
            {
                LOname = string(argv[i+1]);
                if(myrank == 0) printf("Output vertex label filename: %s\n", LOname.c_str());
            }
            else if (strcmp(argv[i],"--cluster-out")==0)
            {
                COname = string(argv[i+1]);
                if(myrank == 0) printf("Output cluster filename: %s\n", COname.c_str());
            }
            else if (strcmp(argv[i],"--cluster-new-out")==0)
            {
                CNOname = string(argv[i+1]);
                if(myrank == 0) printf("Output cluster(new) filename: %s\n", CNOname.c_str());
            }
            else if (strcmp(argv[i],"--alg")==0)
            {
                ALGname = string(argv[i+1]);
                if(myrank == 0) printf("Incremental algorithm name: %s\n", ALGname.c_str());
            }
            else if(strcmp(argv[i],"--per-process-mem")==0){
                perProcessMem = atoi(argv[i+1]);
                if(myrank == 0) printf("Per process memory: %d GB\n", perProcessMem);
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

        if(isMatrixMarket) {
            M11.ParallelReadMM(M11name, base, maximum<NT>());
            M12.ParallelReadMM(M12name, base, maximum<NT>());
            M21.ParallelReadMM(M21name, base, maximum<NT>());
            M22.ParallelReadMM(M22name, base, maximum<NT>());
            L11Num.ParallelRead(L11name, base, maximum<IT>());
            L11 = FullyDistVec<IT, LBL>(fullWorld, L11Num.TotalLength(), LBL{}); 
            convNumToLbl(L11Num, L11);
            L22Num.ParallelRead(L22name, base, maximum<IT>());
            L22 = FullyDistVec<IT, LBL>(fullWorld, L22Num.TotalLength(), LBL{}); 
            convNumToLbl(L22Num, L22);
        }
        else{
            //Handle it later
            //M.ReadGeneralizedTuples(Mname,  maximum<double>());
        }

        SpParHelper::Print("M11 info:\n");
        M11.PrintInfo();
        SpParHelper::Print("---\n");
        SpParHelper::Print("M12 info:\n");
        M12.PrintInfo();
        SpParHelper::Print("---\n");
        SpParHelper::Print("M21 info:\n");
        M21.PrintInfo();
        SpParHelper::Print("---\n");
        SpParHelper::Print("M22 info:\n");
        M22.PrintInfo();
        SpParHelper::Print("---\n");
        
        IT totalLength = L11.TotalLength()+L22.TotalLength();

        SpParMat<IT, NT, DER> Minc(fullWorld);
        SpParMat<IT, NT, DER> MSO(fullWorld);
        FullyDistVec<IT, LBL> LOLbl(fullWorld, totalLength, LBL{});
        FullyDistVec<IT, IT> permMap(fullWorld, totalLength, 0);
        FullyDistVec<IT, IT> isOld(fullWorld, totalLength, 0);
        FullyDistVec<IT, IT> CO(fullWorld, totalLength, 0); // Cluster assignment of new vertices
        FullyDistVec<IT, IT> CO22(fullWorld, M22.getnrow(), 0); // Cluster assignment of new vertices
        SpParMat<IT, NT, DER> MS22(fullWorld); // Summarized new subgraph
        
        if(ALGname != "full"){
            // Find clusters in M22
            // Summarize M22 into MS22
            incParam.summaryIter = 5; // Keep exactly 5th iteration MCL state as summary
            incParam.maxIter = -1; 
            incParam.perProcessMem = perProcessMem;
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

        // If full graph clustering then do not make any modification to anything before constructing incremental graph
        
        incParam.shuffleVertexOrder = true; // Always shuffle vertices in incremental pipeline
        if (ALGname == "full"){
            // Treat each incremental step as full graph clustering
            // Hence, do not make any modification to edge weights
            incParam.normalizedAssign = false; 
            incParam.shuffleVertexOrder = false; 
        }
        else{
            // Normalize nnz before preparing Minc
            incParam.normalizedAssign = true; 
            incParam.shuffleVertexOrder = true; 
        }

        SpParHelper::Print("[Start] Preparing Minc\n");
        t0 = MPI_Wtime();
        PrepIncMat(M11, M12, M21, M22, L11, L22, Minc, LOLbl, isOld, permMap, incParam);
        t1 = MPI_Wtime();
        if(myrank == 0) printf("Time to prepare Minc: %lf\n", t1 - t0);
        Minc.PrintInfo();
        float loadImbalance = Minc.LoadImbalance();
        if(myrank == 0) printf("Minc LoadImbalance: %lf\n", loadImbalance);
        SpParHelper::Print("[End] Preparing Minc\n");
        if(MINCname != "") Minc.ParallelWriteMM(MINCname, base);

        M11.FreeMemory();
        M12.FreeMemory();
        M21.FreeMemory();
        M22.FreeMemory();

        // Consider extra memory requirement to store the intermediate copies needed for incremental
        int perProcessExtraMem = calcExtraMemoryRequirement(Minc);
        if(myrank == 0) cout << "Per proc extra mem: " << perProcessExtraMem  << " GB" << endl;
        incParam.perProcessMem = perProcessMem - perProcessExtraMem;

        incParam.summaryIter = 10; 
        incParam.summaryThresholdNNZ = double(summaryThreshold)/100; 
        incParam.selectivePruneThreshold = double(selectivePruneThreshold)/100;
        //incParam.maxIter = std::numeric_limits<int>::max(); // Run as many iterations as needed for MCL to converge;
        incParam.maxIter = 100; 

        SpParMat<IT, NT, DER> SelectivePruneMask(Minc);
        
        SpParHelper::Print("[Start] Clustering Minc\n");

        t0 = MPI_Wtime();
        IncrementalMCL(Minc, incParam, CO, MSO, isOld, SelectivePruneMask);
        t1 = MPI_Wtime();
        if(myrank == 0) printf("Time to find clusters in Minc: %lf\n", t1 - t0);

        Minc.FreeMemory(); // Not needed anymore
        MSO.PrintInfo();

        SpParHelper::Print("[End] Clustering Minc\n");
        if(myrank == 0) printf("Writing clusters to file: %s\n", COname.c_str());
        if(COname != "") WriteMCLClusters(COname, CO, LOLbl); // Write output clusters before undoing vertex shuffle

        if(incParam.shuffleVertexOrder){
            reversePermutation(MSO, LOLbl, permMap);
        }

        FullyDistVec<IT, IT> LONum(fullWorld, totalLength, 0);
        convLblToNum(LOLbl, LONum);
        if(MSOname != "") MSO.ParallelWriteMM(MSOname, base);
        if(LOname != "") LONum.ParallelWrite(LOname, base);

        MSO.FreeMemory(); // Not needed anymore
    }
    MPI_Finalize();
    return 0;
}

