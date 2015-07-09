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

#define ITERS 1

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

	if(argc != 8)
	{
        cout << argc << endl;
		if(myrank == 0)
		{
            printf("Usage (input): ./mpipspgemm <GridRows> <GridCols> <Layers> <matA> <matB> <matC> <algo>\n");
            printf("Example: ./mpipspgemm 4 4 2 matA.mtx matB.mtx matB.mtx threaded\n");
            printf("algo: outer | column | threaded | all\n");
		}
		return -1;
	}

	unsigned GRROWS = (unsigned) atoi(argv[1]);
	unsigned GRCOLS = (unsigned) atoi(argv[2]);
	unsigned C_FACTOR = (unsigned) atoi(argv[3]);
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
    
	
    
    SpDCCols<int64_t, double> splitA, splitB, controlC;
    SpDCCols<int64_t, double> *splitC;
    string type;
    
    
    string fileA(argv[4]);
    string fileB(argv[5]);
    string fileC(argv[6]);
    Reader(fileA, CMG, splitA, false);
    Reader(fileB, CMG, splitB, true);
    Reader(fileC, CMG, controlC, false);
    //splitA.PrintInfo();
    //controlC.PrintInfo();
    type = string(argv[7]);
    
    if(type == string("outer"))
    {
        for(int k=0; k<ITERS; k++)
        {
            splitC = multiply_exp(splitA, splitB, CMG, true, false); // outer product
            if (controlC == *splitC)
                SpParHelper::Print("Outer product multiplication working correctly\n");
            else
                SpParHelper::Print("ERROR in Outer product multiplication, go fix it!\n");
            //splitC->PrintInfo();
            delete splitC;
        }
        
    }
    else if(type == string("column"))
    {
        splitB.Transpose(); // locally "untranspose" [ABAB: check correctness]
        for(int k=0; k<ITERS; k++)
        {
            splitC = multiply_exp(splitA, splitB, CMG, false, false);
            if (controlC == *splitC)
                SpParHelper::Print("Col-heap multiplication working correctly\n");
            else
                SpParHelper::Print("ERROR in Col-heap multiplication, go fix it!\n");

            delete splitC;
        }
        
    }
    else // default threaded
    {
        splitB.Transpose();
        for(int k=0; k<ITERS; k++)
        {
            splitC = multiply_exp(splitA, splitB, CMG, false, true);
            if (controlC == *splitC)
                SpParHelper::Print("Col-heap-threaded multiplication working correctly\n");
            else
                SpParHelper::Print("ERROR in Col-heap-threaded multiplication, go fix it!\n");
            delete splitC;
        }
    }
    
	MPI_Finalize();
	return 0;
}
	

