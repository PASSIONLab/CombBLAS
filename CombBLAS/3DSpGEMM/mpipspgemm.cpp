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
#include "GenRmatDist.h"
#include "ReadMatDist.h"

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

#define ITERS 2

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
			printf("Usage: ./mpipspgemm <Scale> <GridRows> <GridCols> <Layers> <Type> <EDGEFACTOR> <algo>\n");
			printf("Example: ./mpipspgemm 19 4 4 2 ER 16 outer\n");
			printf("Type ER: Erdos-Renyi\n");
			printf("Type SSCA: R-MAT with SSCA benchmark parameters\n");
			printf("Type G500: R-MAT with Graph500 benchmark parameters\n");
            printf("algo: outer | column | threaded | all\n");
		}
		return -1;         
	}

	unsigned scale = (unsigned) atoi(argv[1]);
	unsigned GRROWS = (unsigned) atoi(argv[2]);
	unsigned GRCOLS = (unsigned) atoi(argv[3]);
	unsigned C_FACTOR = (unsigned) atoi(argv[4]);
    CCGrid CMG(C_FACTOR, GRCOLS);
    
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
    
	
	
    SpDCCols<int64_t, double> splitA, splitB;
    string type;
	
    if(string(argv[5]) == string("input")) // input option
    {
        string fileA = string(argv[6]);
        string fileB = string(argv[7]);
        Reader(fileA, CMG, splitA, false);
        Reader(fileB, CMG, splitB, true);
        type = string(argv[8]);
    }
    else
    {
        
        double initiator[4];
        if(string(argv[5]) == string("ER"))
        {
            initiator[0] = .25;
            initiator[1] = .25;
            initiator[2] = .25;
            initiator[3] = .25;
        }
        else if(string(argv[5]) == string("SSCA"))
        {
            initiator[0] = .57;
            initiator[1] = .19;
            initiator[2] = .19;
            initiator[3] = .05;
        }
        else if(string(argv[5]) == string("G500"))
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
        
        unsigned EDGEFACTOR = (unsigned) atoi(argv[6]);
        Generator(scale, EDGEFACTOR, initiator, CMG, splitA, false);
        Generator(scale, EDGEFACTOR, initiator, CMG, splitB, true); // also transpose before split
        if(myrank == 0) printf("RMATs Generated and replicated along layers\n");
        type = string(argv[7]);
    }
	
	

    
    
   
    if(type == string("outer"))
    {
        for(int k=0; k<ITERS; k++)
            multiply_exp(splitA, splitB, CMG, true, false); // outer product
    }
    else if(type == string("column"))
    {
        splitB.Transpose(); // locally "untranspose" [ABAB: check correctness]
        for(int k=0; k<ITERS; k++)
            multiply_exp(splitA, splitB, CMG, false, false);
    }
    else if(type == string("all"))
    {
        for(int k=0; k<ITERS; k++)
            multiply_exp(splitA, splitB, CMG, true, false); // outer product
            
        splitB.Transpose();
        for(int k=0; k<ITERS; k++)
            multiply_exp(splitA, splitB, CMG, false, false);
        for(int k=0; k<ITERS; k++)
            multiply_exp(splitA, splitB, CMG, false, true);
    }
    else // default threaded
    {
        splitB.Transpose();
        for(int k=0; k<ITERS; k++)
            multiply_exp(splitA, splitB, CMG, false, true);
    }
    
	MPI_Finalize();
	return 0;
}
	

