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

#ifdef TIMING
double cblas_alltoalltime;
double cblas_allgathertime;
#endif

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
            cout << "Usage: ./<Binary> <MatrixA> " << endl;
        }
        MPI_Finalize();
        return -1;
    }
    else {
        string Aname(argv[1]);

        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );
        
        SpParMat<int64_t, double, SpDCCols < int64_t, double >> M(fullWorld);
        typedef PlusTimesSRing<double, double> PTFF;

        M.ParallelReadMM(Aname, true, maximum<double>());
        FullyDistVec<int64_t, int64_t> p( M.getcommgrid() );
        p.iota(M.getnrow(), 0);
        p.RandPerm();
        (M)(p,p,true);// in-place permute to save memory

        //M.ReadGeneralizedTuples(Aname, maximum<double>());
        
        SpParMat<int64_t, double, SpDCCols < int64_t, double >> A(M);
        //SpParMat3D<int64_t, double, SpDCCols < int64_t, double >> A3D(A, 4, true, false);
        SpParMat<int64_t, double, SpDCCols < int64_t, double >> B(M);
        //SpParMat3D<int64_t, double, SpDCCols < int64_t, double >> B3D(B, 4, false, false);

        typedef PlusTimesSRing<double, double> PTFF;
        typedef int64_t IT;
        typedef double NT;
        typedef SpDCCols < int64_t, double > DER;

        double Abcasttime = 0;
        double Abcasttime_prev;
        double Bbcasttime = 0;
        double Bbcasttime_prev;
        double t0, t1;

        int dummy, stages;
        std::shared_ptr<CommGrid> GridC = ProductGrid((A.getcommgrid()).get(), (B.getcommgrid()).get(), stages, dummy, dummy);
        
        //int buffsize = 1024 * 1024 * (512 / sizeof(IT));
        //if(myrank == 0) fprintf(stderr, "Memory to be allocated %d\n", buffsize);
        //IT * sendbuf = new IT[buffsize];
        //if(myrank == 0) fprintf(stderr, "Memory allocated\n");

        for(int phases = 1; phases <= 256; phases = phases * 2){
            if(myrank == 0) fprintf(stderr, "Running with phase: %d\n", phases);
            for(int it = 0; it < 3; it++){
                Abcasttime = 0;
                Bbcasttime = 0;
                
                std::vector< DER > PiecesOfB;
                DER CopyB = *(B.seqptr()); // we allow alias matrices as input because of this local copy
                
                CopyB.ColSplit(phases, PiecesOfB); // CopyB's memory is destroyed at this point
                MPI_Barrier(GridC->GetWorld());
                
                IT ** ARecvSizes = SpHelper::allocate2D<IT>(DER::esscount, stages);
                IT ** BRecvSizes = SpHelper::allocate2D<IT>(DER::esscount, stages);
                
                SpParHelper::GetSetSizes( *(A.seqptr()), ARecvSizes, (A.getcommgrid())->GetRowWorld());
                
                // Remotely fetched matrices are stored as pointers
                DER * ARecv;
                DER * BRecv;
                
                int Aself = (A.getcommgrid())->GetRankInProcRow();
                int Bself = (B.getcommgrid())->GetRankInProcCol();

                //int chunksize = buffsize / phases;
                //if(myrank == 0) fprintf(stderr, "chunksize: %d\n", chunksize);

                for(int p = 0; p < phases; ++p)
                {
                    SpParHelper::GetSetSizes( PiecesOfB[p], BRecvSizes, (B.getcommgrid())->GetColWorld());
                    for(int i = 0; i < stages; ++i)
                    {
                        //t0 = MPI_Wtime();
                        //MPI_Bcast(sendbuf+(chunksize*p), chunksize, MPIType<IT>(), i, GridC->GetColWorld());
                        //t1 = MPI_Wtime();
                        //Bbcasttime += (t1-t0);
                        
                        std::vector<IT> ess;

                        if(i == Aself)  ARecv = A.seqptr();	// shallow-copy
                        else {
                            ess.resize(DER::esscount);
                            for(int j=0; j< DER::esscount; ++j)
                                ess[j] = ARecvSizes[j][i];		// essentials of the ith matrix in this row
                            ARecv = new DER();				// first, create the object
                        }
                        MPI_Barrier(A.getcommgrid()->GetWorld());                       
                        t0=MPI_Wtime();
                        //SpParHelper::BCastMatrix(GridC->GetRowWorld(), *ARecv, ess, i);	// then, receive its elements
                        MPI_Barrier(A.getcommgrid()->GetWorld());                       
                        t1=MPI_Wtime();
                        Abcasttime += (t1-t0);

                        ess.clear();

                        if(i == Bself)  BRecv = &(PiecesOfB[p]);	// shallow-copy
                        else {
                            ess.resize(DER::esscount);
                            for(int j=0; j< DER::esscount; ++j)
                                ess[j] = BRecvSizes[j][i];
                            BRecv = new DER();
                        }
                        MPI_Barrier(A.getcommgrid()->GetWorld());                       
                        t0=MPI_Wtime();
                        SpParHelper::BCastMatrix(GridC->GetColWorld(), *BRecv, ess, i);	// then, receive its elements
                        MPI_Barrier(A.getcommgrid()->GetWorld());                       
                        t1=MPI_Wtime();
                        Bbcasttime += (t1-t0);

                        //if(i != Aself){
                            //if(ARecv != NULL) delete ARecv;
                        //}
                        if(i != Bself) {
                            if(BRecv != NULL) delete BRecv;
                        }
                        
                    }   // all stages executed
                    
                }
                
                
                SpHelper::deallocate2D(ARecvSizes, DER::esscount);
                SpHelper::deallocate2D(BRecvSizes, DER::esscount);
                if(myrank == 0){
                    fprintf(stderr, "Iteration : %d - Abcasttime: %lf\n", it, Abcasttime);
                    fprintf(stderr, "Iteration : %d - Bbcasttime: %lf\n", it, Bbcasttime);
                }
            }
            if(myrank == 0) fprintf(stderr, "\n\n++++++++++++++++++++++++++++++++++++++++++++\n\n\n\n");
        }
    }
    MPI_Finalize();
    return 0;
}
