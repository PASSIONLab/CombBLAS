#include <mpi.h>
#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>
#include <stdint.h>
#include <cmath>
#include "CombBLAS/CombBLAS.h"
#include "Glue.h"
#include "CCGrid.h"
#include "Reductions.h"
#include "Multiplier.h"
#include "SplitMatDist.h"
#include "RestrictionOp.h"


using namespace std;
using namespace combblas;

double comm_bcast;
double comm_reduce;
double comp_summa;
double comp_reduce;
double comp_result;
double comp_reduce_layer;
double comp_split;
double comp_trans;
double comm_split;

#define ITERS 5

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
    
    if(argc < 6)
    {
        if(myrank == 0)
        {
            printf("Usage (random): ./mpipspgemm <GridRows> <GridCols> <Layers> <Type> <Scale> <EDGEFACTOR> \n");
            printf("Usage (input): ./mpipspgemm <GridRows> <GridCols> <Layers> <Type=input> <matA> \n"); //TODO:<Scale>  not meaningful here. Need to remove it.  Still there because current scripts execute without error.
            printf("Example: ./RestrictionOp 4 4 2 ER 19 16 \n");
            printf("Example: ./RestrictionOp 4 4 2 Input matA.mtx\n");
            printf("Type ER: Erdos-Renyi\n");
            printf("Type SSCA: R-MAT with SSCA benchmark parameters\n");
            printf("Type G500: R-MAT with Graph500 benchmark parameters\n");
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
        SpDCCols<int64_t, double> *A;
        
        string type;
        shared_ptr<CommGrid> layerGrid;
        layerGrid.reset( new CommGrid(CMG.layerWorld, 0, 0) );
        FullyDistVec<int64_t, int64_t> p(layerGrid); // permutation vector defined on layers
        
        if(string(argv[4]) == string("input")) // input option
        {
            string fileA(argv[5]);
            
            double t01 = MPI_Wtime();
            A = ReadMat<double>(fileA, CMG, true, p);
            
            
            if(myrank == 0) cout << "Input matrix read : time " << MPI_Wtime() - t01 << endl;
        }
        else
        {
            unsigned scale = (unsigned) atoi(argv[5]);
            unsigned EDGEFACTOR = (unsigned) atoi(argv[6]);
            double initiator[4];
            if(string(argv[4]) == string("ER"))
            {
                initiator[0] = .25;
                initiator[1] = .25;
                initiator[2] = .25;
                initiator[3] = .25;
            }
            else if(string(argv[4]) == string("G500"))
            {
                initiator[0] = .57;
                initiator[1] = .19;
                initiator[2] = .19;
                initiator[3] = .05;
                EDGEFACTOR  = 16;
            }
            else if(string(argv[4]) == string("SSCA"))
            {
                initiator[0] = .6;
                initiator[1] = .4/3;
                initiator[2] = .4/3;
                initiator[3] = .4/3;
                EDGEFACTOR  = 8;
            }
            else {
                if(myrank == 0)
                    printf("The initiator parameter - %s - is not recognized.\n", argv[5]);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            
            
            double t01 = MPI_Wtime();
            A = GenMat<int64_t,double>(CMG, scale, EDGEFACTOR, initiator, true);
            
            if(myrank == 0) cout << "RMATs Generated : time " << MPI_Wtime() - t01 << endl;
            
        }
        
        
        
        SpDCCols<int64_t, double>* R;
        SpDCCols<int64_t, double>* RT;
        
        if(myrank == 0) cout << "Computing restriction matrix \n";
        double t01 = MPI_Wtime();
        RestrictionOp( CMG, A, R, RT);
        
        if(myrank == 0) cout << "Restriction Op computed : time " << MPI_Wtime() - t01 << endl;
        SpDCCols<int64_t, double> *B = new SpDCCols<int64_t, double>(*A); // just a deep copy of A
        SpDCCols<int64_t, double> splitA, splitB, splitR, splitRT;
        SpDCCols<int64_t, double> *splitRTA;
        SpDCCols<int64_t, double> *splitRTAR;
        SpDCCols<int64_t, double> *splitC;
        
        
        SplitMat(CMG, A, splitA, true);
        SplitMat(CMG, B, splitB, false);
        SplitMat(CMG, R, splitR, true);
        SplitMat(CMG, RT, splitRT, false);
        
        if(myrank == 0)
        {
           	printf("\n Processor Grid (row x col x layers x threads): %dx%dx%dx%d \n", CMG.GridRows, CMG.GridCols, CMG.GridLayers, nthreads);
            printf(" prow pcol layer thread comm_bcast   comm_scatter comp_summa comp_merge  comp_scatter  comp_result     other      total\n");
        }
        SpParHelper::Print("Computing A square\n");
        
        splitC = multiply(splitB, splitA, CMG, false, true); // A^2
        delete splitC;
        splitC = multiply(splitB, splitA, CMG, false, true); // A^2
        
        SpParHelper::Print("Computing RTA\n");
        splitRTA = multiply(splitRT, splitA, CMG, false, true);
        delete splitRTA;
        splitRTA = multiply(splitRT, splitA, CMG, false, true);
        
        SpParHelper::Print("Computing RTAR\n");
        splitRTAR = multiply(*splitRTA, splitR, CMG, false, true);
        delete splitRTAR;
        splitRTAR = multiply(*splitRTA, splitR, CMG, false, true);
        
        // count nnz
        int64_t nnzA=0, nnzR=0, nnzC=0, nnzRTA=0, nnzRTAR=0;
        int64_t localnnzA = splitA.getnnz();
        int64_t localnnzR = splitR.getnnz();
        int64_t localnnzC = splitC->getnnz();
        int64_t localnnzRTA = splitRTA->getnnz();
        int64_t localnnzRTAR = splitRTAR->getnnz();
        MPI_Allreduce( &localnnzA, &nnzA, 1, MPIType<int64_t>(), MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce( &localnnzR, &nnzR, 1, MPIType<int64_t>(), MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce( &localnnzC, &nnzC, 1, MPIType<int64_t>(), MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce( &localnnzRTA, &nnzRTA, 1, MPIType<int64_t>(), MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce( &localnnzRTAR, &nnzRTAR, 1, MPIType<int64_t>(), MPI_SUM, MPI_COMM_WORLD);
        if(myrank == 0)
        {
            cout << "----------------------------\n";
            cout << " nnz(A)= " << nnzA << endl;
            cout << " nnz(R)= " << nnzR << endl;
            cout << " nnz(A^2)= " << nnzC << endl;
            cout << " nnz(RTA)= " << nnzRTA << endl;
            cout << " nnz(RTAR)= " << nnzRTAR << endl;
            cout << "----------------------------\n";
        }

        
        
        delete splitC;
        delete splitRTA;
        delete splitRTAR;
        
    }
    
    
    
    
    
    
    
    MPI_Finalize();
    return 0;
}


