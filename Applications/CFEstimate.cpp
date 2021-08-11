#include <mpi.h>
#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/CommGrid3D.h"
#include "CombBLAS/SpParMat3D.h"
#include "CombBLAS/ParFriends.h"

using namespace std;
using namespace combblas;

#define EPS 0.0001

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
    double t0, t1, t2, t3;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    //printf("%d/%d Starting \n", myrank, nprocs);

    if(argc < 1){
        if(myrank == 0)
        {
            cout << "Usage: ./<Binary> " << endl;
        }
        MPI_Finalize();
        return -1;
    }
    else {
        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );
        
        SpParMat<int64_t, double, SpDCCols < int64_t, double >> M(fullWorld);
        std::vector<SpTuples <int64_t, double>*> vec;
        
        for(int i = 0; i < 64; i++){
            std::string filename = std::string("/global/cscratch1/sd/taufique/eukarya_int_3d_64l/");
            filename = filename + std::string("r") + std::to_string(myrank);
            //filename = filename + std::string("_p") + std::to_string(p);
            filename = filename + std::string("_s") + std::to_string(i);
            t0 = MPI_Wtime();
            M.ParallelReadMM(filename, true, maximum<double>());
            t1 = MPI_Wtime();
            printf("File read: %s\n", filename.c_str());
            printf("Time taken to read: %lf\n", t1-t0);
            t0 = MPI_Wtime();
            vec.push_back(new SpTuples<int64_t, double>( *(M.seqptr()) ));
            t1 = MPI_Wtime();
            printf("Time taken to copy: %lf\n", t1-t0);
            printf("Matrix dimension: (nrows: %d, ncols: %d, nnz: %d)\n\n", vec[vec.size()-1]->getnrow(), vec[vec.size()-1]->getncol(), vec[vec.size()-1]->getnnz());
        }
        
        typedef PlusTimesSRing<double, double> PTFF;
        SpTuples<int64_t, double>* res;
        printf("Before merge\n");

        t0 = MPI_Wtime();
        res = MultiwayMerge<PTFF, int64_t, double>(vec, vec[0]->getnrow(), vec[0]->getncol(), false);
        t1 = MPI_Wtime();
        printf("Time taken for MultiwayMerge: %lf\n", t1-t0);
        delete res;

        t0 = MPI_Wtime();
        res = MultiwayMergeHash<PTFF, int64_t, double>(vec, vec[0]->getnrow(), vec[0]->getncol(), false, true);
        t1 = MPI_Wtime();
        printf("Time taken for MultiwayMergeHash: %lf\n", t1-t0);
        delete res;

        t0 = MPI_Wtime();
        res = MultiwayMergeHashSliding<PTFF, int64_t, double>(vec, vec[0]->getnrow(), vec[0]->getncol(), false, true, 512);
        t1 = MPI_Wtime();
        printf("Time taken for MultiwayMergeHashSliding: %lf\n", t1-t0);
        delete res;

        printf("After merge\n");
    }
    MPI_Finalize();
    return 0;
}
