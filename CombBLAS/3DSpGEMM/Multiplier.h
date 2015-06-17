#ifndef _MULTIPLIER_H_
#define _MULTIPLIER_H_

#include "../CombBLAS.h"
#include "CCGrid.h"
#include "SUMMALayer.h"


template <typename IT, typename NT>
void multiply_exp(SpDCCols<IT, NT> & splitA, SpDCCols<IT, NT> & splitB, CCGrid & CMG, bool isBT, bool threaded)
{
    comm_bcast = 0, comm_reduce = 0, comp_summa = 0, comp_reduce = 0, comp_result =0, comp_reduce_layer=0;
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    vector< SpTuples<IT,NT>* > unreducedC;
    
    MPI_Barrier(MPI_COMM_WORLD);
    double time_beg = MPI_Wtime();
    
    SUMMALayer(splitA, splitB, unreducedC, CMG, isBT, threaded);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double time_mid = MPI_Wtime();
    
    SpDCCols<IT,NT> * mergedC;
    if(threaded)
         mergedC = ReduceAll_threaded(unreducedC, CMG);
    else
         mergedC = ReduceAll(unreducedC, CMG);

    MPI_Barrier(MPI_COMM_WORLD);
    double time_end = MPI_Wtime();
    double time_total = time_end-time_beg;
    
    int64_t local_nnz = mergedC->getnnz();
    int64_t global_nnz = 0;
    
#ifdef PARALLELREDUCE
    MPI_Reduce(&local_nnz, &global_nnz, 1, MPIType<int64_t>(), MPI_SUM, 0, MPI_COMM_WORLD);
    if(myrank == 0)
    {
        cout << "Global nonzeros in C is " << global_nnz << endl;
    }
#else
    if(CMG.layer_grid == 0)
    {
        MPI_Reduce(&local_nnz, &global_nnz, 1, MPIType<int64_t>(), MPI_SUM, CMG.layerWorld);
        if(CMG.RankInLayer == 0)
        {
            cout << "Global nonzeros in C is " << global_nnz << endl;
        }
    }
        
#endif
    delete mergedC;
    
    if(CMG.myrank == 0)
    {
        cout << "SUMMA Layer took " << time_mid-time_beg << " seconds" << endl;
        cout << "Reduce took " << time_end-time_mid << " seconds" << endl;
        //cout << "Total first run " << time_total << " seconds" << endl;
        //printf("comm_bcast = %f, comm_reduce = %f, comp_summa = %f, comp_reduce = %f\n", comm_bcast, comm_reduce, comp_summa, comp_reduce);
        double time_other = time_total - (comm_bcast + comm_reduce + comp_summa + comp_reduce + comp_reduce_layer + comp_result);
        printf("\n Processor Grid: %dx%dx%d \n", CMG.GridRows, CMG.GridCols, CMG.GridLayers);
        printf(" ----------------------------------------------------------------------------------------------\n");
        printf(" comm_bcast   comm_reduce comp_summa comp_merge  comp_reduce  comp_result     other      total\n");
        printf(" ----------------------------------------------------------------------------------------------\n");
        printf("%10lf %12lf %10lf %12lf %12lf %12lf %10lf %10lf\n\n", comm_bcast, comm_reduce, comp_summa, comp_reduce, comp_reduce_layer, comp_result, time_other, time_total);
    }
}
#endif
