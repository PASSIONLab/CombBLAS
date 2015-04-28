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
#include "Glue.h"
#include "../CombBLAS.h"

using namespace std;

extern void Split_GetEssensials(void * full, void ** part1, void ** part2, SpDCCol_Essentials * sess1, SpDCCol_Essentials * sess2);
extern void * VoidRMat(unsigned scale, unsigned EDGEFACTOR, double initiator[4], int layergrid, int rankinlayer, void ** part1, void ** part2, bool trans);
extern int SUMMALayer(void * A1, void * A2, void * B1, void * B2, void ** C, CCGrid * cmg);
extern void * ReduceAll(void ** C, CCGrid * CMG, int totalcount);
extern void DeleteMatrix(void ** A);
extern int64_t GetNNZ(void * A);


double comm_bcast;
double comm_reduce;
double comp_summa;
double comp_reduce;

#define N 100
#define REPLICAS 1

int main(int argc, char *argv[])
{
    int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
    
	int THREADS = MPI::COMM_WORLD.Get_size();
	int MYTHREAD = MPI::COMM_WORLD.Get_rank();

	if(argc < 7)
	{
		if(MYTHREAD == 0)
		{
			printf("Usage: ./mpipspgemm <Scale> <GridRows> <GridCols> <Replicas> <Type> <EDGEFACTOR>\n");
			printf("Example: ./mpipspgemm 19 4 4 2 ER 16\n");
			printf("Type ER: Erdos-Renyi\n");
			printf("Type SSCA: R-MAT with SSCA benchmark parameters\n");
			printf("Type G500: R-MAT with Graph500 benchmark parameters\n");
		}
		return -1;         
	}

	unsigned scale = (unsigned) atoi(argv[1]);
	unsigned GRROWS = (unsigned) atoi(argv[2]);
	unsigned GRCOLS = (unsigned) atoi(argv[3]);
	unsigned C_FACTOR = (unsigned) atoi(argv[4]);
	unsigned EDGEFACTOR = (unsigned) atoi(argv[6]);
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
		if(MYTHREAD == 0)
			printf("The initiator parameter - %s - is not recognized\n", argv[5]);
	}


	if(GRROWS != GRCOLS)
	{
		if(MYTHREAD == 0)
			printf("This version of the Combinatorial BLAS only works on a square logical processor grid\n");
		return -1;
	}

	int layer_length = GRROWS*GRCOLS;
	if(layer_length * C_FACTOR != THREADS)
	{
		if(MYTHREAD == 0)
			printf("The product of <GridRows> <GridCols> <Replicas> does not match the number of threads\n");
		return -1;
	}

	/* pack along fibers, so that reductions are fast */
	CCGrid * CMG = (CCGrid *)malloc(sizeof(CCGrid));
	
    
	//for(int i=0; i< 2; ++i)
	{
        /*
		if (i == 1) 
		{
			// Now run the benchmark in summa grid if possible
			int sqrtc = static_cast<int>(sqrt(C_FACTOR));
			if(sqrtc * sqrtc == C_FACTOR)
			{
				C_FACTOR = 1; 
				GRROWS *= sqrtc;
				GRCOLS *= sqrtc;
			}
			else
			{
				if(MYTHREAD == 0)
					printf("The second run is not possible\n");
                MPI::Finalize();
				return 0;
			}
		}*/
		comm_bcast = 0, comm_reduce = 0, comp_summa = 0, comp_reduce = 0;
		
		CMG->layer_grid = MYTHREAD % C_FACTOR;	/* RANKINFIBER = layer_grid, indexed from 1 to C_FACTOR */
		CMG->rankinlayer = MYTHREAD / C_FACTOR;		/* indexed from 1 to layer_length */
		CMG->RANKINCOL = CMG->rankinlayer / GRCOLS;   	/* RANKINCOL = MYPROCROW */
		CMG->RANKINROW = CMG->rankinlayer % GRCOLS;		/* RANKINROW = MYPROCCOL */
		CMG->GRROWS = GRROWS;
		CMG->GRCOLS = GRCOLS;

		#ifdef DEBUG
		printf("Rank %d maps to layer %d (rankinlayer: %d), row %d, and col %d\n", MYTHREAD, CMG->layer_grid, CMG->rankinlayer, CMG->RANKINCOL, CMG->RANKINROW);
		#endif

		void *A1, *A2, *B1, *B2;
		VoidRMat(scale, EDGEFACTOR, initiator, CMG->layer_grid, CMG->rankinlayer, &A1, &A2, false);
		VoidRMat(scale, EDGEFACTOR, initiator, CMG->layer_grid, CMG->rankinlayer, &B1, &B2, true); // also transpose before split

		if(MYTHREAD == 0) printf("RMATs Generated and replicated along layers\n");

		void * mergedC;
		void ** C;

		MPI::COMM_WORLD.Barrier();
		double time_beg = MPI_Wtime();	

		int eachphase = SUMMALayer(A1, A2, B1, B2, &C, CMG);  

		MPI::COMM_WORLD.Barrier();
		double time_mid = MPI_Wtime();
	
		// MergeAll C's [there are 2 * eachphase of them on each processor]	
		mergedC = ReduceAll(C, CMG, 2*eachphase);
		MPI::Intracomm layerWorld = MPI::COMM_WORLD.Split(CMG->layer_grid, CMG->rankinlayer);

        int64_t local_nnz = GetNNZ(mergedC);
        int64_t global_nnz = 0;
        
#ifdef PARALLELREDUCE
        MPI_Reduce(&local_nnz, &global_nnz, 1, MPIType<int64_t>(), MPI_SUM, 0, MPI_COMM_WORLD);
        if(MYTHREAD == 0)
        {
            cout << "Global nonzeros in C is " << global_nnz << endl;
        }
#else
        if(CMG->layer_grid == 0)
        {
            MPI_Reduce(&local_nnz, &global_nnz, 1, MPI::LONG_LONG, MPI::SUM, layerWorld);
            if(layerWorld.Get_rank() == 0)
            {
                cout << "Global nonzeros in C is " << global_nnz << endl;
            }
        }

#endif
        
        
				
		DeleteMatrix(&mergedC);

		MPI::COMM_WORLD.Barrier();
		double time_end = MPI_Wtime();
        double time_total = time_end-time_beg;

		if(MYTHREAD == 0)
		{
			cout << "SUMMA Layer took " << time_mid-time_beg << " seconds" << endl;
			cout << "Reduce took " << time_end-time_mid << " seconds" << endl;
			//cout << "Total first run " << time_total << " seconds" << endl;
			//printf("comm_bcast = %f, comm_reduce = %f, comp_summa = %f, comp_reduce = %f\n", comm_bcast, comm_reduce, comp_summa, comp_reduce);
            double time_other = time_total - (comm_bcast + comm_reduce + comp_summa + comp_reduce);
            printf("\n Processor Grid: %dx%dx%d \n", GRROWS, GRCOLS, C_FACTOR);
            printf(" -------------------------------------------------------------------\n");
            printf(" comm_bcast   comm_reduce comp_summa comp_reduce    other      total\n");
            printf(" -------------------------------------------------------------------\n");
            printf("%10lf %12lf %10lf %12lf %10lf %10lf\n\n", comm_bcast, comm_reduce, comp_summa, comp_reduce, time_other, time_total);
		}
        
        /*
		comm_bcast = 0, comm_reduce = 0, comp_summa = 0, comp_reduce = 0;	// reset

		MPI::COMM_WORLD.Barrier();
		time_beg = MPI_Wtime();
		for(int i=0; i< 10; ++i)
		{
			eachphase = SUMMALayer(A1, A2, B1, B2, &C, CMG);  
			mergedC = ReduceAll(C, CMG, 2*eachphase);
			DeleteMatrix(&mergedC);
		}

		MPI::COMM_WORLD.Barrier();
		time_end = MPI_Wtime();

		if(MYTHREAD == 0)
		{
			// Since 0th processor contributes to all the gathering/merging/bcast/etc operations, we use his timings for now
			cout << "Ten further iterations took " << time_end-time_beg << " seconds" << endl;
			printf("comm_bcast = %f, comm_reduce = %f, comp_summa = %f, comp_reduce = %f\n", comm_bcast, comm_reduce, comp_summa, comp_reduce);
		}
        */
	}
	MPI::Finalize();
	return 0;
}
	

