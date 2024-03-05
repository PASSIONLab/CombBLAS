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
    if(argc < 11)
    {
        if(myrank == 0)
        {
            cout << "Usage: ./full -M <MATRIX_FILENAME>\n";
            cout << "-M <MATRIX FILENAME> (required)\n";
            cout << "-base <0|1> (default 1)\n";
            cout << "-summary-out <FILENAME TO STORE SUMMARY MATRIX> (required)\n";
            cout << "-cluster-out <FILENAME TO STORE OUTPUT CLUSTERS> (required)\n";
            cout << "-label <FILENAME THAT CONTAINS VERTEX LABELS OF SUMMARY GRAPH> (required)\n";
        }
    }
    else{
        string Mname = "";
        string Lname = "";
        string MSOname = "";
        string COname = "";
        int base = 1;
        bool isMatrixMarket = true;

        HipMCLParam incParam;
        InitParam(incParam);
        incParam.shuffleVertexOrder = true;
        
        for (int i = 1; i < argc; i++)
        {
            if (strcmp(argv[i],"-I")==0)
            {
                string mfiletype = string(argv[i+1]);
                if(mfiletype == "triples") isMatrixMarket = false;
            }
            else if (strcmp(argv[i],"-base")==0)
            {
                base = atoi(argv[i + 1]);
                if(myrank == 0) printf("Base of MM (1 or 0):%d\n", base);
            }
            else if (strcmp(argv[i],"-M")==0)
            {
                Mname = string(argv[i+1]);
                if(myrank == 0) printf("Matrix filename: %s\n", Mname.c_str());
            }
            else if (strcmp(argv[i],"-label")==0)
            {
                Lname = string(argv[i+1]);
                if(myrank == 0) printf("Label filename: %s\n", Lname.c_str());
            }
            else if (strcmp(argv[i],"-summary-out")==0)
            {
                MSOname = string(argv[i+1]);
                if(myrank == 0) printf("Output Summary Matrix filename: %s\n", MSOname.c_str());
            }
            else if (strcmp(argv[i],"-cluster-out")==0)
            {
                COname = string(argv[i+1]);
                if(myrank == 0) printf("Output cluster filename: %s\n", COname.c_str());
            }
            else if(strcmp(argv[i],"-per-process-mem")==0){
                incParam.perProcessMem = atoi(argv[i+1]);
                if(myrank == 0) printf("Per process memory: %d GB\n", incParam.perProcessMem);
            }
        }

        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );
        
        double t0, t1, t2, t3, t4, t5;

        SpParMat<IT, NT, DER> M(fullWorld);

        if(isMatrixMarket) {
            M.ParallelReadMM(Mname, base, maximum<NT>());
        }
        else{
            //Handle it later
            //M.ReadGeneralizedTuples(Mname,  maximum<double>());
        }

        FullyDistVec<IT, IT> LNum(fullWorld); // Temporary. To solve the inability of reading vector of text values
        LNum.ParallelRead(Lname, base, maximum<IT>());
        FullyDistVec<IT, LBL> Lbl = FullyDistVec<IT, LBL>(fullWorld, LNum.TotalLength(), LBL{}); 
        convNumToLbl(LNum, Lbl);

        SpParMat<IT, NT, DER> MSO(fullWorld);
        FullyDistVec<IT, IT> CO(fullWorld, Lbl.TotalLength(), 0); // Cluster assignment of new vertices

        incParam.summaryIter = 10; 
        incParam.summaryThresholdNNZ = 0.7; 
        incParam.selectivePruneThreshold = -1.0;
        HipMCL(M, incParam, CO, MSO);

        WriteMCLClusters(COname, CO, Lbl); 
        MSO.ParallelWriteMM(MSOname, base);

    }
    MPI_Finalize();
    return 0;
}

