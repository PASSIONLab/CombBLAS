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
            printf("algo: outer | column | threaded \n");
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
    
	
    
    SpDCCols<int32_t, double> splitA, splitB, controlC;
    SpDCCols<int32_t, double> *splitC;
    string type;
    
    
    string fileA(argv[4]);
    string fileB(argv[5]);
    string fileC(argv[6]);
    
    {
        shared_ptr<CommGrid> layerGrid;
        layerGrid.reset( new CommGrid(CMG.layerWorld, 0, 0) );
        FullyDistVec<int32_t, int32_t> p(layerGrid); // permutation vector defined on layers
        
        double t01 = MPI_Wtime();
        
        SpDCCols<int32_t, double> *A = ReadMat<double>(fileA, CMG, true, p);
        SpDCCols<int32_t, double> *B = ReadMat<double>(fileB, CMG, true, p);
        SpDCCols<int32_t, double> *C = ReadMat<double>(fileC, CMG, true, p);
        
        SplitMat(CMG, A, splitA, false);
        SplitMat(CMG, B, splitB, true);
        SplitMat(CMG, C, controlC, false);
        
        if(myrank == 0) cout << "Matrices read and replicated along layers : time " << MPI_Wtime() - t01 << endl;

        type = string(argv[7]);
        
        if(type == string("outer"))
        {
            for(int k=0; k<ITERS; k++)
            {
                splitB.Transpose(); // locally "transpose" [ABAB: check correctness]
                splitC = multiply(splitA, splitB, CMG, true, false); // outer product
                if (controlC == *splitC)
                    SpParHelper::Print("Outer product multiplication working correctly\n");
                else
                    SpParHelper::Print("ERROR in Outer product multiplication, go fix it!\n");
                delete splitC;
            }
            
        }
        else if(type == string("column"))
        {
            
            for(int k=0; k<ITERS; k++)
            {
                splitC = multiply(splitA, splitB, CMG, false, false);
                if (controlC == *splitC)
                    SpParHelper::Print("Col-heap multiplication working correctly\n");
                else
                    SpParHelper::Print("ERROR in Col-heap multiplication, go fix it!\n");
                
                delete splitC;
            }
            
        }
        else // default threaded
        {
            for(int k=0; k<ITERS; k++)
            {
                splitC = multiply(splitA, splitB, CMG, false, true);
                if (controlC == *splitC)
                    SpParHelper::Print("Col-heap-threaded multiplication working correctly\n");
                else
                    SpParHelper::Print("ERROR in Col-heap-threaded multiplication, go fix it!\n");
                delete splitC;
            }
        }
    }

	MPI_Finalize();
	return 0;
}
	

