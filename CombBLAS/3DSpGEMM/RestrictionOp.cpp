#define DETERMINISTIC
#include "../CombBLAS.h"
#include <mpi.h>
#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>
#ifdef THREADED
	#ifndef _OPENMP
	#define _OPENMP
	#endif
	#include <omp.h>
#endif
#include "Glue.h"
#include "CCGrid.h"
#include "Reductions.h"
#include "Multiplier.h"
#include "SplitMatDist.h"



using namespace std;

double comm_bcast;
double comm_reduce;
double comp_summa;
double comp_reduce;
double comp_result;
double comp_reduce_layer;
double comp_split;
double comp_trans;
double comm_split;


//#include "TwitterEdge.h"






typedef SpParMat < int64_t, double, SpDCCols<int32_t, double > > PSpMat_Twitter;
typedef SpParMat < int64_t, bool, SpDCCols<int64_t, bool > > PSpMat_Bool;


template <typename PARMAT>
void Symmetricize(PARMAT & A)
{
    PARMAT AT = A;
    AT.Transpose();
    A += AT;
}



#ifdef DETERMINISTIC
MTRand GlobalMT(1);
#else
MTRand GlobalMT;	// generate random numbers with Mersenne Twister 
#endif




template <typename T1, typename T2>
struct Select2ndMinSR
{
    static T2 id(){ return T2(); };
    static bool returnedSAID() { return false; }
    static MPI_Op mpi_op() { return MPI_MIN; };
    
    static T2 add(const T2 & arg1, const T2 & arg2)
    {
        return std::min(arg1, arg2);
    }
    
    static T2 multiply(const T1 & arg1, const T2 & arg2)
    {
        return arg2;
    }
    
    static void axpy(const T1 a, const T2 & x, T2 & y)
    {
        y = add(y, multiply(a, x));
    }
};

// MIS
// assumption: A is symmetric with zero diagonal
// Warning: will fall into infine loop with the presense of isolated self loops

template <typename ONT, typename IT, typename INT, typename DER>
FullyDistSpVec<IT, ONT> MIS(SpParMat < IT, INT, DER> A)
{
    IT nvert = A.getncol();
    
    //# the final result set. S[i] exists and is 1 if vertex i is in the MIS
    FullyDistSpVec<IT, ONT> mis ( A.getcommgrid(), nvert);
    
    
    //# the candidate set. initially all vertices are candidates.
    //# If cand[i] exists, then i is a candidate. The value cand[i] is i's random number for this iteration.
    FullyDistVec<IT, double> dcand( A.getcommgrid(), nvert, 1.0);
    FullyDistSpVec<IT, double> cand (dcand);
    FullyDistSpVec<IT, double> min_neighbor_r ( A.getcommgrid(), nvert);
    
    FullyDistSpVec<IT, ONT> new_S_members ( A.getcommgrid(), nvert);
    FullyDistSpVec<IT, ONT> new_S_neighbors ( A.getcommgrid(), nvert);
    
    while (cand.getnnz() > 0)
    {
        
        //# label each vertex in cand with a random value
        cand.Apply([](const double & ignore){return (double) GlobalMT.rand();});
        
        //# find the smallest random value among a vertex's neighbors
        SpMV<Select2ndMinSR<INT, double>>(A, cand, min_neighbor_r, false);
        
        //# The vertices to be added to S this iteration are those whose random value is
        //# smaller than those of all its neighbors:
        // **** if cand has isolated vertices, they will be included in new_S_members ******
        new_S_members = EWiseApply<ONT>(min_neighbor_r, cand,
                                        [](double x, double y){return (ONT)1;},
                                        [](double x, double y){return y<x;},
                                        true, false, 2.0,  2.0, true);
        
        new_S_members.DebugPrint();
        //# new_S_members are no longer candidates, so remove them from cand
        cand = EWiseApply<double>(cand, new_S_members,
                               [](double x, ONT y){return x;},
                               [](double x, ONT y){return true;},
                               false, true, 0.0, (ONT) 0, false);
        
        //# find neighbors of new_S_members
        SpMV<Select2ndMinSR<INT, ONT>>(A, new_S_members, new_S_neighbors, false);
        
        
        //# remove neighbors of new_S_members from cand, because they cannot be part of the MIS anymore
        cand = EWiseApply<double>(cand, new_S_neighbors,
                               [](double x, ONT y){return x;},
                               [](double x, ONT y){return true;},
                               false, true, 0.0, (ONT) 0, false);
        
        //# add new_S_members to mis
        mis = EWiseApply<ONT>(mis, new_S_members,
                            [](ONT x, ONT y){return x;},
                            [](ONT x, ONT y){return true;},
                            true, true, (ONT) 1, (ONT) 1, true);
    }
    
    return mis;
}



// second hop MIS (i.e., MIS on A^2)
template <typename ONT, typename IT, typename INT, typename DER>
FullyDistSpVec<IT, ONT> MIS2(SpParMat < IT, INT, DER> A)
{
    IT nvert = A.getncol();
    
    //# the final result set. S[i] exists and is 1 if vertex i is in the MIS
    FullyDistSpVec<IT, ONT> mis ( A.getcommgrid(), nvert);
    
    
    //# the candidate set. initially all vertices are candidates.
    //# If cand[i] exists, then i is a candidate. The value cand[i] is i's random number for this iteration.
    FullyDistVec<IT, double> dcand( A.getcommgrid(), nvert, 1.0);
    FullyDistSpVec<IT, double> cand (dcand);
    FullyDistSpVec<IT, double> min_neighbor_r ( A.getcommgrid(), nvert);
    FullyDistSpVec<IT, double> min_neighbor2_r ( A.getcommgrid(), nvert);
    
    FullyDistSpVec<IT, ONT> new_S_members ( A.getcommgrid(), nvert);
    FullyDistSpVec<IT, ONT> new_S_neighbors ( A.getcommgrid(), nvert);
    FullyDistSpVec<IT, ONT> new_S_neighbors2 ( A.getcommgrid(), nvert);
    
    while (cand.getnnz() > 0)
    {
        
        //# label each vertex in cand with a random value
        cand.Apply([](const double & ignore){return (double) GlobalMT.rand();});
        
        //# find the smallest random value among a vertex's neighbors
        SpMV<Select2ndMinSR<INT, double>>(A, cand, min_neighbor_r, false);
        SpMV<Select2ndMinSR<INT, double>>(A, min_neighbor_r, min_neighbor2_r, false);
        
        
        //# The vertices to be added to S this iteration are those whose random value is
        //# smaller than those of all its neighbors:
        // **** if cand has isolated vertices, they will be included in new_S_members ******
        new_S_members = EWiseApply<ONT>(min_neighbor2_r, cand,
                                        [](double x, double y){return (ONT)1;},
                                        [](double x, double y){return y<=x;}, // equality is for back edges since we are operating on A^2
                                        true, false, 2.0,  2.0, true);
        
        //# new_S_members are no longer candidates, so remove them from cand
        cand = EWiseApply<double>(cand, new_S_members,
                                  [](double x, ONT y){return x;},
                                  [](double x, ONT y){return true;},
                                  false, true, 0.0, (ONT) 0, false);
        
        //# find 2-hop neighbors of new_S_members
        SpMV<Select2ndMinSR<INT, ONT>>(A, new_S_members, new_S_neighbors, false);
        SpMV<Select2ndMinSR<INT, ONT>>(A, new_S_neighbors, new_S_neighbors2, false);
        
        
        //# remove neighbors of new_S_members from cand, because they cannot be part of the MIS anymore
        cand = EWiseApply<double>(cand, new_S_neighbors2,
                                  [](double x, ONT y){return x;},
                                  [](double x, ONT y){return true;},
                                  false, true, 0.0, (ONT) 0, false);
        
        //# add new_S_members to mis
        mis = EWiseApply<ONT>(mis, new_S_members,
                              [](ONT x, ONT y){return x;},
                              [](ONT x, ONT y){return true;},
                              true, true, (ONT) 1, (ONT) 1, true);
    }
    
    return mis;
}


template <typename ONT, typename IT, typename LIT, typename NT>
SpParMat < IT, ONT, SpDCCols < LIT, ONT >> RestrictionOp( SpParMat < IT, NT, SpDCCols < LIT, NT >> A)
{
    
    SpParMat < IT, bool, SpDCCols < LIT, bool >> B = SpParMat < IT, bool, SpDCCols < LIT, bool >> (A);
    B.RemoveLoops();
    SpParMat < IT, bool, SpDCCols < LIT, bool >> BT = B;
    BT.Transpose();
    B += BT;
    
    // ------------ compute MIS-2 ----------------------------
    FullyDistSpVec<IT, IT> mis2 (B.getcommgrid(), B.getncol());
    mis2 = MIS2<IT>(B);
    mis2.DebugPrint();
    // ------------ neighbors of mis-2 including themselves ----
    FullyDistVec<IT, IT> ri = mis2.FindInds([](IT x){return true;});
    FullyDistVec<IT, IT> ci(A.getcommgrid());
    ci.iota(mis2.getnnz(), (IT)0);
    SpParMat < IT, ONT, SpDCCols < LIT, ONT >> M(A.getnrow(), ci.TotalLength(), ri, ci, (ONT) 1, false);
    SpParMat < IT, ONT, SpDCCols < LIT, ONT >> R = PSpGEMM<PlusTimesSRing<bool, ONT>>(B,M);
    R += M;

    return R;
    
}



int main(int argc, char *argv[])
{
    int provided;
    //MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
    
    
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (provided < MPI_THREAD_SERIALIZED)
    {
        printf("ERROR: The MPI library does not have MPI_THREAD_SERIALIZED support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    
    
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    
    if(argc < 8)
    {
        if(myrank == 0)
        {
            printf("Usage (random): ./mpipspgemm <GridRows> <GridCols> <Layers> <Type> <Scale> <EDGEFACTOR> <algo>\n");
            printf("Usage (input): ./mpipspgemm <GridRows> <GridCols> <Layers> <Type=input> <matA> <matB> <algo>\n"); //TODO:<Scale>  not meaningful here. Need to remove it.  Still there because current scripts execute without error.
            printf("Example: ./mpipspgemm 4 4 2 ER 19 16 outer\n");
            printf("Example: ./mpipspgemm 4 4 2 Input matA.mtx matB.mtx threaded\n");
            printf("Type ER: Erdos-Renyi\n");
            printf("Type SSCA: R-MAT with SSCA benchmark parameters\n");
            printf("Type G500: R-MAT with Graph500 benchmark parameters\n");
            printf("algo: outer | column \n");
        }
        return -1;
    }
    
    
    unsigned GRROWS = (unsigned) atoi(argv[1]);
    unsigned GRCOLS = (unsigned) atoi(argv[2]);
    unsigned C_FACTOR = (unsigned) atoi(argv[3]);
    CCGrid CMG(C_FACTOR, GRCOLS);
    int nthreads;
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
    
    
    if(GRROWS != GRCOLS)
    {
        SpParHelper::Print("This version of the Combinatorial BLAS only works on a square logical processor grid\n");
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    int layer_length = GRROWS*GRCOLS;
    if(layer_length * C_FACTOR != nprocs)
    {
        SpParHelper::Print("The product of <GridRows> <GridCols> <Replicas> does not match the number of processes\n");
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    
    {
        //SpDCCols<int64_t, double> splitA, splitB;
        //SpDCCols<int64_t, double> *splitC;
        //string type;
        //shared_ptr<CommGrid> layerGrid;
        //layerGrid.reset( new CommGrid(CMG.layerWorld, 0, 0) );
        //FullyDistVec<int64_t, int64_t> p(layerGrid); // permutation vector defined on layers
        
        double initiator[4] = {.25, .25, .25, .25};	// creating erdos-renyi
        
        unsigned scale = (unsigned) atoi(argv[5]);
        unsigned EDGEFACTOR = (unsigned) atoi(argv[6]);
        
        SpDCCols<int64_t, double> *A = GenMat<int64_t,double>(CMG, scale, EDGEFACTOR, initiator, false, true);
        SpParMat < int64_t, double, SpDCCols<int64_t,double> > BB = SpParMat < int64_t, double, SpDCCols<int64_t,double> >(A, CMG.layerWorld);
        SpParMat < int64_t, double, SpDCCols<int64_t,double> > R;
        
        if(CMG.layer_grid == 0)
        {
            R = RestrictionOp<int64_t>(BB);
        }
        //SplitMat(CMG, A, splitA);
        //SplitMat(CMG, B, splitB);
        if(myrank == 0) cout << "RMATs Generated and replicated along layers : time " << endl;
        
    
        
    }
    
    MPI_Finalize();
    return 0;
}



int main1(int argc, char* argv[])
{
    
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    if(argc < 2)
    {
        if(myrank == 0)
        {
            cout << "Usage: ./FilteredMIS <Scale>" << endl;
        }
        MPI_Finalize();
        return -1;
    }
    {
        // Declare objects
        PSpMat_Twitter A;
        FullyDistVec<int64_t, int64_t> indegrees;	// in-degrees of vertices (including multi-edges and self-loops)
        FullyDistVec<int64_t, int64_t> oudegrees;	// out-degrees of vertices (including multi-edges and self-loops)
        FullyDistVec<int64_t, int64_t> degrees;	// combined degrees of vertices (including multi-edges and self-loops)
        PSpMat_Bool * ABool;
        
        SpParHelper::Print("Using synthetic data, which we ALWAYS permute for load balance\n");
        SpParHelper::Print("We only balance the original input, we don't repermute after each filter change\n");
        SpParHelper::Print("BFS is run on UNDIRECTED graph, hence hitting CCs, and TEPS is bidirectional\n");
        
        double initiator[4] = {.25, .25, .25, .25};	// creating erdos-renyi
        double t01 = MPI_Wtime();
        DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>();
        
        unsigned scale = static_cast<unsigned>(atoi(argv[1]));
        ostringstream outs;
        outs << "Forcing scale to : " << scale << endl;
        SpParHelper::Print(outs.str());
        
        // parameters: (double initiator[4], int log_numverts, int edgefactor, bool scramble, bool packed)
        DEL->GenGraph500Data(initiator, scale, 16, true, false );	// generate packed edges
        SpParHelper::Print("Generated renamed edge lists\n");
        
        ABool = new PSpMat_Bool(*DEL, false);
        Symmetricize(*ABool);
        ABool->RemoveLoops();
        delete DEL;	// free memory
        
        //typename Dist<int64_t, int32_t, double>::MAT B = Dist<int64_t, int32_t, double>::MAT(*ABool);
        SpParMat < int64_t, double, SpDCCols<int64_t,double> > B = SpParMat < int64_t, double, SpDCCols<int64_t,double> >(*ABool);
        

        SpParMat < int64_t, double, SpDCCols<int64_t,double> > R = RestrictionOp<int64_t>(B);
        
        R.PrintInfo();
        
        
    }
    MPI_Finalize();
    return 0;
}



