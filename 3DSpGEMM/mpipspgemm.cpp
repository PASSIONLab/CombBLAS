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
            printf("Usage (input): ./mpipspgemm <GridRows> <GridCols> <Layers> <Type=input> <matA> <matB> <algo>\n");
            printf("Example: ./mpipspgemm 4 4 2 ER 19 16 outer\n");
            printf("Example: ./mpipspgemm 4 4 2 Input matA.mtx matB.mtx column\n");
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
        SpDCCols<int64_t, double> splitA, splitB;
        SpDCCols<int64_t, double> *splitC;
        string type;
        shared_ptr<CommGrid> layerGrid;
        layerGrid.reset( new CommGrid(CMG.layerWorld, 0, 0) );
        FullyDistVec<int64_t, int64_t> p(layerGrid); // permutation vector defined on layers
        
        if(string(argv[4]) == string("input")) // input option
        {
            string fileA(argv[5]);
            string fileB(argv[6]);
            
            double t01 = MPI_Wtime();
            SpDCCols<int64_t, double> *A = ReadMat<double>(fileA, CMG, true, p);
            SpDCCols<int64_t, double> *B = ReadMat<double>(fileB, CMG, true, p);
            SplitMat(CMG, A, splitA, false);
            SplitMat(CMG, B, splitB, true); //row-split
            if(myrank == 0) cout << "Matrices read and replicated along layers : time " << MPI_Wtime() - t01 << endl;
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
            SpDCCols<int64_t, double> *A = GenMat<int64_t,double>(CMG, scale, EDGEFACTOR, initiator, true);
            SpDCCols<int64_t, double> *B = GenMat<int64_t,double>(CMG, scale, EDGEFACTOR, initiator, true);
            
            SplitMat(CMG, A, splitA, false);
            SplitMat(CMG, B, splitB, true); //row-split
            if(myrank == 0) cout << "RMATs Generated and replicated along layers : time " << MPI_Wtime() - t01 << endl;
            
        }
        
        int64_t  globalnnzA=0, globalnnzB=0;
        int64_t localnnzA = splitA.getnnz();
        int64_t localnnzB = splitB.getnnz();
        MPI_Allreduce( &localnnzA, &globalnnzA, 1, MPIType<int64_t>(), MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce( &localnnzB, &globalnnzB, 1, MPIType<int64_t>(), MPI_SUM, MPI_COMM_WORLD);
        if(myrank == 0) cout << "After split: nnzA= " << globalnnzA << " & nnzB= " << globalnnzB;

        
        
        type = string(argv[7]);
        if(myrank == 0)
        {
           	printf("\n Processor Grid (row x col x layers x threads): %dx%dx%dx%d \n", CMG.GridRows, CMG.GridCols, CMG.GridLayers, nthreads);
            printf(" prow pcol layer thread comm_bcast   comm_scatter comp_summa comp_merge  comp_scatter  comp_result     other      total\n");
        }
        if(type == string("outer"))
        {
            splitB.Transpose(); //locally transpose for outer product
            for(int k=0; k<ITERS; k++)
            {
                splitC = multiply(splitA, splitB, CMG, true, false); // outer product
                delete splitC;
            }
        }
        
        else // default column-threaded
        {
            for(int k=0; ITERS>0 && k<ITERS-1; k++)
            {
                splitC = multiply(splitA, splitB, CMG, false, true);
                delete splitC;
            }
            splitC = multiply(splitA, splitB, CMG, false, true);
            int64_t  nnzC=0;
            int64_t localnnzC = splitC->getnnz();
            MPI_Allreduce( &localnnzC, &nnzC, 1, MPIType<int64_t>(), MPI_SUM, MPI_COMM_WORLD);
            if(myrank == 0) cout << "\n After multiplication: nnzC= " << nnzC << endl << endl;
            delete splitC;
            
        }
    }
    
    MPI_Finalize();
    return 0;
}


