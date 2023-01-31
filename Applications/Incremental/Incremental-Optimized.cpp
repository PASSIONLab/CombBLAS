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
#include "IncClust.h"

using namespace std;
using namespace combblas;


#ifdef _OPENMP
int cblas_splits = omp_get_max_threads();
#else
int cblas_splits = 1;
#endif

// Given an adjacency matrix, and cluster assignment vector, removes inter cluster edges
template <class IT, class NT, class DER>
void RemoveInterClusterEdges(SpParMat<IT, NT, DER>& M, FullyDistVec<IT, IT>& C){
    FullyDistVec<IT, NT> Ctemp(C); // To convert value types from IT to NT. Because DimApply and PruneColumn requires that.
    SpParMat<IT, NT, DER> Mask(M);
    Mask.DimApply(Row, Ctemp, [](NT mv, NT vv){return vv;});
    Mask.PruneColumn(Ctemp, [](NT mv, NT vv){return static_cast<NT>(vv == mv);}, true);

    //Mask.PrintInfo();
    M.SetDifference(Mask);
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
            else if(strcmp(argv[i],"--per-process-mem")==0){
                //incParam.perProcessMem = atoi(argv[i+1]);
                //if(myrank == 0) printf("Per process memory: %d GB\n", incParam.perProcessMem);
            }
        }

        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );

        if(myrank == 0) printf("Running Incremental-Optimized\n");

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

        /**********/
        //FullyDistVec<IT, NT> colsums = M.Reduce(Column, plus<NT>(), 0.0);
        ////colsums.DebugPrint();
        //colsums.Apply(safemultinv<NT>());
        ////colsums.DebugPrint();
        //M.DimApply(Column, colsums, multiplies<NT>());    // scale each "Column" with the given vector

        ////// sums of squares of columns
        //FullyDistVec<IT, NT> colssqs = M.Reduce(Column, plus<NT>(), 0.0, bind2nd(exponentiate(), 2));
        //colssqs.DebugPrint();

        ////M.ParallelWriteMM("blah-1.txt", 1);
        ////// Matrix entries are non-negative, so max() can use zero as identity
        //FullyDistVec<IT, NT> colmaxs = M.Reduce(Column, maximum<NT>(), 0.0);
        //colmaxs.DebugPrint();
        ////colmaxs -= colssqs;
        
        ////// multiplu by number of nonzeros in each column
        ////FullyDistVec<IT, NT> nnzPerColumn = A.Reduce(Column, plus<NT>(), 0.0, [](NT val){return 1.0;});
        ////colmaxs.EWiseApply(nnzPerColumn, multiplies<NT>());
        /**********/
        
        //std::mt19937 rng;
        //rng.seed(myrank);
        //std::uniform_int_distribution<int64_t> udist(0, 9999);

        //IT gnRow = M.getnrow();
        //IT nRowPerProc = gnRow / nprocs;
        //IT lRowStart = myrank * nRowPerProc;
        //IT lRowEnd = (myrank == nprocs - 1) ? gnRow : (myrank + 1) * nRowPerProc;

        //std::vector < std::vector < IT > > lvList(nSplit);
        //std::vector < std::vector < std::array<char, MAXVERTNAME> > > lvListLabels(nSplit); // MAXVERTNAME is 64, defined in SpDefs
                                                                                           
        //for (IT r = lRowStart; r < lRowEnd; r++) {
            //IT randomNum = udist(rng);
            //IT s = randomNum % nSplit;
            //lvList[s].push_back(r);
            
            //// Convert the integer vertex id to label as string
            //std::string labelStr = std::to_string(r); 
            //// Make a std::array of char with the label
            //std::array<char, MAXVERTNAME> labelArr = {};
            //for ( IT i = 0; (i < labelStr.length()) && (i < MAXVERTNAME); i++){
                //labelArr[i] = labelStr[i]; 
            //}
            //lvListLabels[s].push_back( labelArr );
        //}

        //std::vector < FullyDistVec<IT,IT>* > dvList;
        //std::vector < FullyDistVec<IT, std::array<char, MAXVERTNAME> >* > dvListLabels;
        //for (int s = 0; s < nSplit; s++){
            //dvList.push_back(new FullyDistVec<IT, IT>(lvList[s], fullWorld));
            //dvListLabels.push_back(new FullyDistVec<IT, std::array<char, MAXVERTNAME> >(lvListLabels[s], fullWorld));
        //}

        //SpParMat<IT, NT, DER> M11(fullWorld);
        //SpParMat<IT, NT, DER> M12(fullWorld);
        //SpParMat<IT, NT, DER> M21(fullWorld);
        //SpParMat<IT, NT, DER> M22(fullWorld);

        ////std::string incFileName = Mname + std::string(".") + std::to_string(nSplit) + std::string(".inc-opt-unshuffled");
        //std::string incFileName = Mname + std::string(".") + std::to_string(nSplit) + std::string(".inc-opt-shuffled");

        //FullyDistVec<IT, IT> prevVertices(*(dvList[0])); // Create a distributed vector to keep track of the vertices being considered at each incremental step
        //FullyDistVec<IT, std::array<char, MAXVERTNAME>> prevVerticesLabels(*(dvListLabels[0])); // Create a distributed vector to keep track of the vertex labels being considered at each incremental step
                                                                                                ////
        //M11 = M.SubsRef_SR < PTNTBOOL, PTBOOLNT> (prevVertices, prevVertices, false);

        //std::vector< FullyDistVec<IT, IT> > toConcatenate(2, FullyDistVec<IT, IT>(fullWorld));
        //std::vector< FullyDistVec<IT, std::array<char, MAXVERTNAME> > > toConcatenateLabels(2, FullyDistVec<IT, std::array<char, MAXVERTNAME> >(fullWorld));

        //for(int s = 1; s < nSplit; s++){
            //MPI_Barrier(MPI_COMM_WORLD);
            //if(myrank == 0) printf("***\n", s);
            //if(myrank == 0) printf("[Start] Processing split: %d\n", s);

            //FullyDistVec<IT, IT> newVertices(*(dvList[s]));
            //FullyDistVec<IT, std::array<char, MAXVERTNAME> > newVerticesLabels(*(dvListLabels[s]));

            //t0 = MPI_Wtime();
            //M11 = M.SubsRef_SR <PTNTBOOL, PTBOOLNT> (prevVertices, prevVertices, false);
            //t1 = MPI_Wtime();
            //if(myrank == 0) printf("[Step: %d]\tTime to extract M11: %lf\n", s, t1 - t0);
            //M11.PrintInfo();

            //t0 = MPI_Wtime();
            //M12 = M.SubsRef_SR <PTNTBOOL, PTBOOLNT> (prevVertices, newVertices, false);
            //t1 = MPI_Wtime();
            //if(myrank == 0) printf("[Step: %d]\tTime to extract M12: %lf\n", s, t1 - t0);
            //M12.PrintInfo();

            //t0 = MPI_Wtime();
            //M21 = M.SubsRef_SR <PTNTBOOL, PTBOOLNT> (newVertices, prevVertices, false);
            //t1 = MPI_Wtime();
            //if(myrank == 0) printf("[Step: %d]\tTime to extract M21: %lf\n", s, t1 - t0);
            //M21.PrintInfo();
            
            //t0 = MPI_Wtime();
            //M22 = M.SubsRef_SR <PTNTBOOL, PTBOOLNT> (newVertices, newVertices, false); // Get subgraph induced by newly added vertices in current step
            //t1 = MPI_Wtime();
            //if(myrank == 0) printf("[Step: %d]\tTime to extract M22: %lf\n", s, t1 - t0);
            //M22.PrintInfo();

            //IT totVertices = prevVertices.TotalLength() + newVertices.TotalLength();
            //SpParMat<IT, NT, DER> Mstar(fullWorld); // Summarized graph
            //FullyDistVec<IT, IT> clustAsn(fullWorld, totVertices, 0); // Cluster assignment of each vertex 
            //FullyDistVec<IT, std::array<char, MAXVERTNAME> > allVerticesLabels(fullWorld, totVertices, std::array<char, MAXVERTNAME>{}); // Merged and shuffled vertex labels after being shuffled during the incremental clustering process

            //IncClust(M11, M12, M21, M22, prevVerticesLabels, newVerticesLabels, allVerticesLabels, clustAsn, Mstar, 1);

            //// Store clustAsgn to appropriate place
            
            //// Prepare for next incremental step
            //FullyDistVec<IT, IT> allVertices(fullWorld, totVertices, 0); 
            //IT aLocLen = allVerticesLabels.LocArrSize();
            //for (IT i = 0; i < aLocLen; i++){
                //std::string strLbl(allVerticesLabels.GetLocalElement(i).data());
                //IT numLbl = atoi(strLbl.c_str());
                //allVertices.SetLocalElement(i, numLbl);
            //}

            //prevVertices = FullyDistVec<IT, IT > (allVertices);
            //prevVerticesLabels = FullyDistVec<IT, std::array<char, MAXVERTNAME> > (allVerticesLabels);
            //if(myrank == 0) printf("[End] Split: %d\n", s);
        //}

        //for(IT s = 0; s < dvList.size(); s++){
            //delete dvList[s];
            //delete dvListLabels[s];
        //}

    }
    MPI_Finalize();
    return 0;
}

