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
#include "../CombBLAS.h"
#include "Glue.h"
#include "CCGrid.h"
#include "Reductions.h"
#include "Multiplier.h"
#include "SplitMatDist.h"
#include "RestrictionOp.h"


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
        SpDCCols<int32_t, double> splitA, splitB;
        SpDCCols<int32_t, double> *splitC;
        string type;
        shared_ptr<CommGrid> layerGrid;
        layerGrid.reset( new CommGrid(CMG.layerWorld, 0, 0) );
        FullyDistVec<int32_t, int32_t> p(layerGrid); // permutation vector defined on layers
        
        if(string(argv[4]) == string("input")) // input option
        {
            string fileA(argv[5]);
            string fileB(argv[6]);
            
            SpDCCols<int32_t, double> *A = ReadMat<double>(fileA, CMG, false, true, p);
            SpDCCols<int32_t, double> *B = ReadMat<double>(fileB, CMG, true, true, p);
            
            SplitMat(CMG, A, splitA);
            SplitMat(CMG, B, splitB);
            if(myrank == 0) printf("RMATs Generated and replicated along layers\n");
        }
        else
        {
            
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
            }
            else if(string(argv[4]) == string("SSCA"))
            {
                initiator[0] = .6;
                initiator[1] = .4/3;
                initiator[2] = .4/3;
                initiator[3] = .4/3;
            }
            else {
                if(myrank == 0)
                    printf("The initiator parameter - %s - is not recognized. Using default ER\n", argv[5]);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            
            unsigned scale = (unsigned) atoi(argv[5]);
            unsigned EDGEFACTOR = (unsigned) atoi(argv[6]);
            
            SpDCCols<int32_t, double> *A = GenMat<int32_t,double>(CMG, scale, EDGEFACTOR, initiator, false, true);
            SpDCCols<int32_t, double> *B = GenMat<int32_t,double>(CMG, scale, EDGEFACTOR, initiator, true, true);
            
            SplitMat(CMG, A, splitA);
            SplitMat(CMG, B, splitB);
            if(myrank == 0) printf("RMATs Generated and replicated along layers\n");
            
        }
        
        
        type = string(argv[7]);
        if(myrank == 0)
            printf("\n Processor Grid (row x col x layers x threads): %dx%dx%dx%d \n", CMG.GridRows, CMG.GridCols, CMG.GridLayers, nthreads);
        
        if(type == string("outer"))
        {
            for(int k=0; k<ITERS; k++)
            {
                splitC = multiply(splitA, splitB, CMG, true, false); // outer product
                delete splitC;
            }
            
        }
        
        else // default column-threaded
        {
            splitB.Transpose();
            for(int k=0; k<ITERS; k++)
            {
                splitC = multiply(splitA, splitB, CMG, false, true);
                delete splitC;
            }
            
        }
    }
    
	MPI_Finalize();
	return 0;
}
	

