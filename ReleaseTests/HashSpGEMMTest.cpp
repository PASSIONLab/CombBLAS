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
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

    if(argc < 4){
        if(myrank == 0)
        {
            cout << "Usage: ./<Binary> <MatrixA> <MatrixB> <MatrixCC>" << endl;
        }
        MPI_Finalize();
        return -1;
    }
    else {
        string Aname(argv[1]);
        string Bname(argv[2]);
        string CCname(argv[3]);
        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );
        
        double t0, t1;
        
        typedef PlusTimesSRing<double, double> PTFF;

        SpParMat<int64_t, double, SpDCCols < int64_t, double >> A2D(fullWorld);
        SpParMat<int64_t, double, SpDCCols < int64_t, double >> B2D(fullWorld);
        SpParMat<int64_t, double, SpDCCols < int64_t, double >> CC2D(fullWorld);

        A2D.ParallelReadMM(Aname, true, maximum<double>());
        B2D.ParallelReadMM(Bname, true, maximum<double>());
        CC2D.ParallelReadMM(CCname, true, maximum<double>());

        SpDCCols < int64_t, double >* Alocal = A2D.seqptr();
        SpDCCols < int64_t, double >* Blocal = B2D.seqptr();
        SpDCCols < int64_t, double >* CClocal = CC2D.seqptr();

        //Clocal = LocalSpGEMMHash<PTFF, double>(*Alocal, *Blocal, false, false, true);
        SpTuples<int64_t, double>* Clocal_tuples = LocalSpGEMM<PTFF, double>(*Alocal, *Blocal, false, false);
        SpDCCols < int64_t, double >* Clocal = new SpDCCols < int64_t, double >(*Clocal_tuples, false);

        if(*Clocal == *CClocal){
            if(myrank == 0) fprintf(stderr, "Correct\n");
        }
        else{
            if(myrank == 0) fprintf(stderr, "Not correct\n");
        }
        
    }
    MPI_Finalize();
    return 0;
}
