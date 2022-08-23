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

// Simple helper class for declarations: Just the numerical type is templated
// The index type and the sequential matrix type stays the same for the whole code
// In this case, they are "int" and "SpDCCols"
template <class NT>
class PSpMat
{
public:
    typedef SpDCCols < int64_t, NT > DCCols;
    typedef SpParMat < int64_t, NT, DCCols > MPI_DCCols;
};

int main(int argc, char* argv[])
{
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

    if(argc < 2){
        if(myrank == 0)
        {
            cout << "Usage: ./<Binary> <MatrixM> <MatrixC>" << endl;
        }
        MPI_Finalize();
        return -1;
    }
    else {
        double vm_usage, resident_set;
        string Mname(argv[1]);
        //string Cname(argv[2]);
        if(myrank == 0){
            fprintf(stderr, "Graph: %s\n", argv[1]);
            //fprintf(stderr, "Cluster assignment: %s\n", argv[2]);
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
        FullyDistVec<IT, NT> vtxLabels(fullWorld);
		//ifstream vecinpC(argv[2]);

        t0 = MPI_Wtime();
        M.ParallelReadMM(Mname, true, maximum<double>());
        //vtxLabels = M.ReadGeneralizedTuples(Mname,  maximum<NT>());

		//C.ReadDistribute(vecinpC, 0);
        t1 = MPI_Wtime();
        if(myrank == 0) fprintf(stderr, "Time taken to read files: %lf\n", t1-t0);
        
        /* 
         * Dump a submatrix corresponding to the largest connected component to a file 
         * */
        SpParMat<IT, NT, DER> A(M); // Create a copy of given matrix

        // A is a directed graph
        // symmetricize A
        SpParMat<IT,NT,DER> AT = A;
        AT.Transpose();
        A += AT;

        IT nCC;
        FullyDistVec<IT, IT> ccLabels = CC(A, nCC);
        if(myrank == 0) printf("Number of connected component %d\n", nCC);

        std::vector<IT> ccLabelsLocal = ccLabels.GetLocVec();
        IT* ccSizesLocal = new IT[nCC];
        IT* ccSizesGlobal = new IT[nCC];
        memset(ccSizesLocal, 0, sizeof(IT) * nCC);
        memset(ccSizesGlobal, 0, sizeof(IT) * nCC);
#pragma omp for
        for(IT i = 0; i < nCC; i++){
            ccSizesLocal[i] = std::count_if( ccLabelsLocal.begin(), ccLabelsLocal.end(), bind2nd(equal_to<IT>(), i));
        }
        MPI_Allreduce(ccSizesLocal, ccSizesGlobal, (int)nCC, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

        IT largestCC = 0;
        IT largestCCSize = ccSizesGlobal[largestCC];
        for(IT i = 1; i < nCC; i++){
            if (ccSizesGlobal[i] > largestCCSize){
                largestCC = i;
                largestCCSize = ccSizesGlobal[i];
            }
        }

        if(myrank == 0) printf("Largest connected component is %dth component, size %d\n",largestCC, largestCCSize);
        FullyDistVec<IT,IT> isov = ccLabels.FindInds(bind2nd(equal_to<IT>(), largestCC));
        SpParMat<IT, NT, DER> MCC = A.SubsRef_SR<PTNTBOOL,PTBOOLNT>(isov, isov, false);
        MCC.PrintInfo();
        MCC.ParallelWriteMM(Mname+std::string(".cc"), true);
    }
    MPI_Finalize();
    return 0;
}
