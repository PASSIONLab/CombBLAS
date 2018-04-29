#ifndef _SUMMA_LAYER_H_
#define _SUMMA_LAYER_H_

#include <mpi.h>
#include <sys/time.h> 
#include <iostream>
#include <iomanip>
#include <functional>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>


#include "CombBLAS/CombBLAS.h"
#include "Glue.h"
#include "CombBLAS/mtSpGEMM.h"
#include "CCGrid.h"

namespace combblas {

// SplitB is already locally transposed
// Returns an array of unmerged lists in C
template <typename IT, typename NT>
void SUMMALayer (SpDCCols<IT,NT> & SplitA, SpDCCols<IT,NT> & SplitB, std::vector< SpTuples<IT,NT>* > & C, CCGrid & CMG, bool isBT, bool threaded)
{
	typedef PlusTimesSRing<NT,NT> PTDD;
	
	int stages = CMG.GridCols;	// total number of "essential" summa stages
	IT ** ARecvSizes = SpHelper::allocate2D<IT>(SpDCCols<IT,NT>::esscount, stages);
	IT ** BRecvSizes = SpHelper::allocate2D<IT>(SpDCCols<IT,NT>::esscount, stages);
	
	// Remotely fetched matrices are stored as pointers
	SpDCCols<IT,NT> * ARecv;
	SpDCCols<IT,NT> * BRecv;
	
	int Aself = CMG.RankInRow;
	int Bself = CMG.RankInCol;
		
    // Set the dimensions
    SpParHelper::GetSetSizes( SplitA, ARecvSizes, CMG.rowWorld);
    SpParHelper::GetSetSizes( SplitB, BRecvSizes, CMG.colWorld);
    
    for(int i = 0; i < stages; ++i)
    {
        double bcast_beg = MPI_Wtime();
        std::vector<IT> ess;
        
        if(i == Aself)  ARecv = &SplitA;	// shallow-copy
        else
        {
            ess.resize(SpDCCols<IT,NT>::esscount);
            for(int j=0; j< SpDCCols<IT,NT>::esscount; ++j)
                ess[j] = ARecvSizes[j][i];		// essentials of the ith matrix in this row
		
            ARecv = new SpDCCols<IT,NT>();				// first, create the object
        }
        SpParHelper::BCastMatrix(CMG.rowWorld, *ARecv, ess, i);	// then, receive its elements
        ess.clear();
        
        if(i == Bself)  BRecv = &SplitB;	// shallow-copy
        else
        {
            ess.resize(SpDCCols<IT,NT>::esscount);
            for(int j=0; j< SpDCCols<IT,NT>::esscount; ++j)
            {
                ess[j] = BRecvSizes[j][i];
            }
            BRecv = new SpDCCols<IT,NT>();
        }
        SpParHelper::BCastMatrix(CMG.colWorld, *BRecv, ess, i);	// then, receive its elements

        comm_bcast += (MPI_Wtime() - bcast_beg);
        double summa_beg = MPI_Wtime();
        SpTuples<IT,NT> * C_cont;
        if(threaded)
        {
                C_cont = LocalSpGEMM<PTDD, NT>
                (*ARecv, *BRecv, // parameters themselves
                 i != Aself, 	// 'delete A' condition
                 i != Bself);	// 'delete B' condition
        }
        else
        {
            C_cont = MultiplyReturnTuples<PTDD, NT>
                (*ARecv, *BRecv, // parameters themselves
                 false, isBT,	// transpose information (B is transposed)
                 i != Aself, 	// 'delete A' condition
                 i != Bself);	// 'delete B' condition
        }
        comp_summa += (MPI_Wtime() - summa_beg);
        C.push_back(C_cont);
	}
	
	SpHelper::deallocate2D(ARecvSizes, SpDCCols<IT,NT>::esscount);
	SpHelper::deallocate2D(BRecvSizes, SpDCCols<IT,NT>::esscount);
}

}

#endif
