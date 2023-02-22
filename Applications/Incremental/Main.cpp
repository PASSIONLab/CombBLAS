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
    if(argc < 27)
    {
        if(myrank == 0)
        {
            cout << "Usage: ./inc -I <mm|triples> -M <MATRIX_FILENAME> -N <NUMBER OF SPLITS>\n";
            cout << "-I <INPUT FILE TYPE> (mm: matrix market, triples: (vtx1, vtx2, edge_weight) triples, default: mm)\n";
            cout << "-base <BASE OF MATRIX MARKET> (default:1)\n";
            cout << "-M11 <M11 MATRIX FILE NAME> (required)\n";
            cout << "-M12 <M12 MATRIX FILE NAME> (required)\n";
            cout << "-M21 <M21 MATRIX FILE NAME> (required)\n";
            cout << "-M22 <M22 MATRIX FILE NAME> (required)\n";
            cout << "-summary-in <SUMMARY MATRIX FILE NAME> (required)\n";
            cout << "-L11 <M11 VERTEX LABEL> (required)\n";
            cout << "-L22 <M22 VERTEX LABEL> (required)\n";
            cout << "-summary-out <FILENAME TO STORE SUMMARY GRAPH>\n";
            cout << "-cluster-out <FILENAME TO STORE OUTPUT CLUSTERS>\n";
            cout << "-label-out <FILENAME TO STORE VERTEX LABELS OF SUMMARY GRAPH>\n";
            cout << "-inc <ALGORITHM TO USE> (full, base, v1, v2)\n";
        }
        MPI_Finalize();
        return -1;
    }
    else{
        string M11name = "";
        string M12name = "";
        string M21name = "";
        string M22name = "";
        string MS11name = "";
        string MSOname = "";
        string L11name = "";
        string L22name = "";
        string LOname = "";
        string COname = "";
        string ALGname = "";
        int base = 1;
        bool isMatrixMarket = true;

        HipMCLParam incParam;
        InitParam(incParam);
        
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
            else if (strcmp(argv[i],"-M11")==0)
            {
                M11name = string(argv[i+1]);
                if(myrank == 0) printf("M11 Matrix filename: %s\n", M11name.c_str());
            }
            else if (strcmp(argv[i],"-M12")==0)
            {
                M12name = string(argv[i+1]);
                if(myrank == 0) printf("M12 Matrix filename: %s\n", M12name.c_str());
            }
            else if (strcmp(argv[i],"-M21")==0)
            {
                M21name = string(argv[i+1]);
                if(myrank == 0) printf("M21 Matrix filename: %s\n", M21name.c_str());
            }
            else if (strcmp(argv[i],"-M22")==0)
            {
                M22name = string(argv[i+1]);
                if(myrank == 0) printf("M22 Matrix filename: %s\n", M22name.c_str());
            }
            else if (strcmp(argv[i],"-summary-in")==0)
            {
                MS11name = string(argv[i+1]);
                if(myrank == 0) printf("M11 Summary Matrix filename: %s\n", MS11name.c_str());
            }
            else if (strcmp(argv[i],"-summary-out")==0)
            {
                MSOname = string(argv[i+1]);
                if(myrank == 0) printf("Output Summary Matrix filename: %s\n", MSOname.c_str());
            }
            else if (strcmp(argv[i],"-L11")==0)
            {
                L11name = string(argv[i+1]);
                if(myrank == 0) printf("M11 vertex label filename: %s\n", L11name.c_str());
            }
            else if (strcmp(argv[i],"-L22")==0)
            {
                L22name = string(argv[i+1]);
                if(myrank == 0) printf("M22 vertex label filename: %s\n", L22name.c_str());
            }
            else if (strcmp(argv[i],"-label-out")==0)
            {
                LOname = string(argv[i+1]);
                if(myrank == 0) printf("Output vertex label filename: %s\n", LOname.c_str());
            }
            else if (strcmp(argv[i],"-cluster-out")==0)
            {
                COname = string(argv[i+1]);
                if(myrank == 0) printf("Output cluster filename: %s\n", COname.c_str());
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

        typedef int64_t IT;
        typedef double NT;
        typedef SpDCCols < int64_t, double > DER;
        typedef PlusTimesSRing<double, double> PTFF;
        typedef PlusTimesSRing<bool, double> PTBOOLNT;
        typedef PlusTimesSRing<double, bool> PTNTBOOL;
        typedef std::array<char, MAXVERTNAME> LBL;
        
        double t0, t1, t2, t3, t4, t5;

        SpParMat<IT, NT, DER> M11(fullWorld);
        SpParMat<IT, NT, DER> M12(fullWorld);
        SpParMat<IT, NT, DER> M21(fullWorld);
        SpParMat<IT, NT, DER> M22(fullWorld);
        SpParMat<IT, NT, DER> MS11(fullWorld);
        FullyDistVec<IT, IT> L11(fullWorld); // Ideally vector of LBL. But vector of IT for now, because CombBLAS not supporting label read and write
        FullyDistVec<IT, IT> L22(fullWorld); // Same

        if(isMatrixMarket) {
            M11.ParallelReadMM(M11name, base, maximum<NT>());
            M11.PrintInfo();
            M12.ParallelReadMM(M12name, base, maximum<NT>());
            M12.PrintInfo();
            M21.ParallelReadMM(M21name, base, maximum<NT>());
            M21.PrintInfo();
            M22.ParallelReadMM(M22name, base, maximum<NT>());
            M22.PrintInfo();
            MS11.ParallelReadMM(MS11name, base, maximum<NT>());
            MS11.PrintInfo();
            L11.ParallelRead(L11name, base, maximum<IT>());
            //L11 -= FullyDistVec<IT,IT>(fullWorld, L11.TotalLength(), 1);
            L22.ParallelRead(L11name, base, maximum<IT>());
            //L22 -= FullyDistVec<IT,IT>(fullWorld, L22.TotalLength(), 1);
        }
        else{
            // Do it later
            //M.ReadGeneralizedTuples(Mname,  maximum<double>());
        }
        
        IT totalLength = L11.TotalLength()+L22.TotalLength();

        ////Test provided label vectors
        //std::vector< FullyDistVec<IT,IT> > toConcatenate;
        //toConcatenate.push_back(FullyDistVec<IT,IT>(L11));
        //toConcatenate.push_back(FullyDistVec<IT,IT>(L22));
        //FullyDistVec<IT, IT> Lall = Concatenate(toConcatenate);
        //Lall.sort();
        //FullyDistVec<IT, IT> litmus1(fullWorld);
        //litmus1.iota(totalLength, 0);
        //FullyDistVec<IT, IT> litmus2(fullWorld);
        //litmus2.iota(totalLength, 1);
        //if(Lall == litmus1){
            //if(myrank == 0) printf("Litmus 1 passed\n");
        //}
        //else if(Lall == litmus2){
            //if(myrank == 0) printf("Litmus 2 passed\n");
        //}
        //else{
            //if(myrank == 0) printf("Something fishy\n");
        //}
        //auto x = Lall.MinElement();
        //if(myrank == 0) printf("x: %ld, %d\n", std::get<0>(x), std::get<1>(x));

        const std::vector<IT> L11Loc = L11.GetLocVec();
        std::vector<LBL> L11LblLoc(L11Loc.size());
        const std::vector<IT> L22Loc = L22.GetLocVec();
        std::vector<LBL> L22LblLoc(L22Loc.size());

        for(int i = 0; i < L11Loc.size(); i++){
            std::string labelStr = std::to_string(L11Loc[i]); 
            for ( IT j = 0; (j < labelStr.length()) && (j < MAXVERTNAME); j++){
                L11LblLoc[i][j] = labelStr[j]; 
            }
        }
        for(int i = 0; i < L22Loc.size(); i++){
            std::string labelStr = std::to_string(L22Loc[i]); 
            for ( IT j = 0; (j < labelStr.length()) && (j < MAXVERTNAME); j++){
                L22LblLoc[i][j] = labelStr[j]; 
            }
        }

        FullyDistVec<IT, LBL> L11Lbl(L11LblLoc, fullWorld);
        FullyDistVec<IT, LBL> L22Lbl(L22LblLoc, fullWorld);
        
        // Hard coded parameters for now
        incParam.summaryIter = 0;
        incParam.summaryThresholdNNZ = 0.7;
        incParam.maxIter = 10000000; // Arbitrary large number as maximum number of iterations. Run as many iterations as needed to converge;

        FullyDistVec<IT, IT> CO22(fullWorld, M22.getnrow(), 0); // Cluster assignment of each vertex 
        SpParMat<IT, NT, DER> MS22(fullWorld); // Summarized graph
        if(myrank == 0) printf("Clustering new\n");
        t0 = MPI_Wtime();
        HipMCL(M22, incParam, CO22, MS22);
        t1 = MPI_Wtime();
        if(myrank == 0) printf("Time to find clusters: %lf\n", t1 - t0);

        SpParMat<IT, NT, DER> MSO(fullWorld);
        FullyDistVec<IT, LBL> LOLbl(fullWorld, L11.TotalLength()+L22.TotalLength(), LBL{});
        FullyDistVec<IT, IT> CO(fullWorld, L11.TotalLength()+L22.TotalLength(), 0);

        if(myrank == 0) printf("Clustering all\n");
        t0 = MPI_Wtime();
        IncClust(MS11, M12, M21, MS22, L11Lbl, L22Lbl, LOLbl, CO, MSO, 1, incParam);
        t1 = MPI_Wtime();
        if(myrank == 0) printf("Time to find clusters: %lf\n", t1 - t0);

        FullyDistVec<IT, IT> LO(fullWorld, L11.TotalLength()+L22.TotalLength(), 0); 
        IT aLocLen = LOLbl.LocArrSize();
        for (IT i = 0; i < aLocLen; i++){
            std::string strLbl(LOLbl.GetLocalElement(i).data());
            IT numLbl = atoi(strLbl.c_str());
            LO.SetLocalElement(i, numLbl);
        }
        
        MSO.ParallelWriteMM(MSOname, base);
        WriteMCLClusters(COname, CO, LOLbl);
        LO.ParallelWrite(LOname, base);
    }
    MPI_Finalize();
    return 0;
}

