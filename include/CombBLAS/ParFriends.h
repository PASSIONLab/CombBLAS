/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.6 -------------------------------------------------*/
/* date: 6/15/2017 ---------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc  --------------------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2017, The Regents of the University of California
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */


#ifndef _PAR_FRIENDS_H_
#define _PAR_FRIENDS_H_

#include "mpi.h"
#include <iostream>
#include <cstdarg>
#include "SpParMat.h"	
#include "SpParMat3D.h"	
#include "SpParHelper.h"
#include "MPIType.h"
#include "Friends.h"
#include "OptBuf.h"
#include "mtSpGEMM.h"
#include "MultiwayMerge.h"
#include <unistd.h>
#include <type_traits>

namespace combblas {

template <class IT, class NT, class DER>
class SpParMat;

/*************************************************************************************************/
/**************************** FRIEND FUNCTIONS FOR PARALLEL CLASSES ******************************/
/*************************************************************************************************/


/**
 ** Concatenate all the FullyDistVec<IT,NT> objects into a single one
 **/
template <typename IT, typename NT>
FullyDistVec<IT,NT> Concatenate ( std::vector< FullyDistVec<IT,NT> > & vecs)
{
	if(vecs.size() < 1)
	{
		SpParHelper::Print("Warning: Nothing to concatenate, returning empty ");
		return FullyDistVec<IT,NT>();
	}
	else if (vecs.size() < 2)
	{
		return vecs[1];
	
	}
	else 
	{
		typename std::vector< FullyDistVec<IT,NT> >::iterator it = vecs.begin();
		std::shared_ptr<CommGrid> commGridPtr = it->getcommgrid();
		MPI_Comm World = commGridPtr->GetWorld();
		
		IT nglen = it->TotalLength();	// new global length
		IT cumloclen = it->MyLocLength();	// existing cumulative local lengths 
		++it;
		for(; it != vecs.end(); ++it)
		{
			if(*(commGridPtr) != *(it->getcommgrid()))
			{
				SpParHelper::Print("Grids are not comparable for FullyDistVec<IT,NT>::EWiseApply\n");
				MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
			}
			nglen += it->TotalLength();
			cumloclen += it->MyLocLength();
		}
		FullyDistVec<IT,NT> ConCat (commGridPtr, nglen, NT());	
		int nprocs = commGridPtr->GetSize();
		
		std::vector< std::vector< NT > > data(nprocs);
		std::vector< std::vector< IT > > inds(nprocs);
		IT gloffset = 0;
		for(it = vecs.begin(); it != vecs.end(); ++it)
		{
			IT loclen = it->LocArrSize();
			for(IT i=0; i < loclen; ++i)
			{
				IT locind;
				IT loffset = it->LengthUntil();
				int owner = ConCat.Owner(gloffset+loffset+i, locind);	
				data[owner].push_back(it->arr[i]);
				inds[owner].push_back(locind);
			}
			gloffset += it->TotalLength();
		}
		
		int * sendcnt = new int[nprocs];
		int * sdispls = new int[nprocs];
		for(int i=0; i<nprocs; ++i)
			sendcnt[i] = (int) data[i].size();
		
		int * rdispls = new int[nprocs];
		int * recvcnt = new int[nprocs];
		MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, World);  // share the request counts
		sdispls[0] = 0;
		rdispls[0] = 0;
		for(int i=0; i<nprocs-1; ++i)
		{
			sdispls[i+1] = sdispls[i] + sendcnt[i];
			rdispls[i+1] = rdispls[i] + recvcnt[i];
		}
		IT totrecv = std::accumulate(recvcnt,recvcnt+nprocs,static_cast<IT>(0));
		NT * senddatabuf = new NT[cumloclen];
		for(int i=0; i<nprocs; ++i)
		{
      std::copy(data[i].begin(), data[i].end(), senddatabuf+sdispls[i]);
			std::vector<NT>().swap(data[i]);	// delete data vectors
		}
		NT * recvdatabuf = new NT[totrecv];
		MPI_Alltoallv(senddatabuf, sendcnt, sdispls, MPIType<NT>(), recvdatabuf, recvcnt, rdispls, MPIType<NT>(), World);  // send data
		delete [] senddatabuf;
		
		IT * sendindsbuf = new IT[cumloclen];
		for(int i=0; i<nprocs; ++i)
		{
      std::copy(inds[i].begin(), inds[i].end(), sendindsbuf+sdispls[i]);
			std::vector<IT>().swap(inds[i]);	// delete inds vectors
		}
		IT * recvindsbuf = new IT[totrecv];
		MPI_Alltoallv(sendindsbuf, sendcnt, sdispls, MPIType<IT>(), recvindsbuf, recvcnt, rdispls, MPIType<IT>(), World);  // send new inds
		DeleteAll(sendindsbuf, sendcnt, sdispls);

		for(int i=0; i<nprocs; ++i)
		{
			for(int j = rdispls[i]; j < rdispls[i] + recvcnt[i]; ++j)			
			{
				ConCat.arr[recvindsbuf[j]] = recvdatabuf[j];
			}
		}
		DeleteAll(recvindsbuf, recvcnt, rdispls);
		return ConCat;
	}
}

template <typename MATRIXA, typename MATRIXB>
bool CheckSpGEMMCompliance(const MATRIXA & A, const MATRIXB & B)
{
	if(A.getncol() != B.getnrow())
	{
		std::ostringstream outs;
		outs << "Can not multiply, dimensions does not match"<< std::endl;
		outs << A.getncol() << " != " << B.getnrow() << std::endl;
		SpParHelper::Print(outs.str());
		MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
		return false;
	}	
	if((void*) &A == (void*) &B)
	{
		std::ostringstream outs;
		outs << "Can not multiply, inputs alias (make a temporary copy of one of them first)"<< std::endl;
		SpParHelper::Print(outs.str());
		MPI_Abort(MPI_COMM_WORLD, MATRIXALIAS);
		return false;
	}	
	return true;
}	


// Combined logic for prune, recovery, and select
template <typename IT, typename NT, typename DER>
void MCLPruneRecoverySelect(SpParMat<IT,NT,DER> & A, NT hardThreshold, IT selectNum, IT recoverNum, NT recoverPct, int kselectVersion)
{
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    
#ifdef TIMING
    double t0, t1;
#endif
    
    // Prune and create a new pruned matrix
    SpParMat<IT,NT,DER> PrunedA = A.Prune([hardThreshold](NT val){ return val <= hardThreshold; }, false);

    // column-wise statistics of the pruned matrix
    FullyDistVec<IT,NT> colSums = PrunedA.Reduce(Column, std::plus<NT>(), 0.0);
    FullyDistVec<IT,NT> nnzPerColumnUnpruned = A.Reduce(Column, std::plus<NT>(), 0.0, [](NT val){return 1.0;});
    FullyDistVec<IT,NT> nnzPerColumn = PrunedA.Reduce(Column, std::plus<NT>(), 0.0, [](NT val){return 1.0;});
    //FullyDistVec<IT,NT> pruneCols(A.getcommgrid(), A.getncol(), hardThreshold);
    FullyDistVec<IT,NT> pruneCols(nnzPerColumn);
    pruneCols = hardThreshold;

    PrunedA.FreeMemory();

	FullyDistSpVec<IT,NT> recoverCols(nnzPerColumn, [recoverNum](const NT& val) { return val < recoverNum; });
    
    // recover only when nnzs in unprunned columns are greater than nnzs in pruned column
    recoverCols = EWiseApply<NT>(recoverCols, nnzPerColumnUnpruned,
                                 [](NT spval, NT dval){return spval;},
                                 [](NT spval, NT dval){return dval > spval;},
                                 false, NT());


    recoverCols = recoverPct;
    // columns with nnz < r AND sum < recoverPct (pct)
    recoverCols = EWiseApply<NT>(recoverCols, colSums,
                                 [](NT spval, NT dval){return spval;},
                                 [](NT spval, NT dval){return dval < spval;},
                                 false, NT());

    IT nrecover = recoverCols.getnnz();

    if(nrecover > 0)
    {
#ifdef TIMING
        t0=MPI_Wtime();
#endif
        A.Kselect(recoverCols, recoverNum, kselectVersion);

#ifdef TIMING
        t1=MPI_Wtime();
        mcl_kselecttime += (t1-t0);
#endif

        pruneCols.Set(recoverCols);

#ifdef COMBBLAS_DEBUG
        std::ostringstream outs;
        outs << "Number of columns needing recovery: " << nrecover << std::endl;
        SpParHelper::Print(outs.str());
#endif
        
    }

    if(selectNum>0)
    {
        // remaining columns will be up for selection
        FullyDistSpVec<IT,NT> selectCols = EWiseApply<NT>(recoverCols, colSums,
                                                          [](NT spval, NT dval){return spval;},
                                                          [](NT spval, NT dval){return spval==-1;},
                                                          true, static_cast<NT>(-1));
        
        selectCols = selectNum;
        selectCols = EWiseApply<NT>(selectCols, nnzPerColumn,
                                    [](NT spval, NT dval){return spval;},
                                    [](NT spval, NT dval){return dval > spval;},
                                    false, NT());
        IT nselect = selectCols.getnnz();
        
        if(nselect > 0 )
        {
#ifdef TIMING
            t0=MPI_Wtime();
#endif
            A.Kselect(selectCols, selectNum, kselectVersion); // PrunedA would also work
#ifdef TIMING
            t1=MPI_Wtime();
            mcl_kselecttime += (t1-t0);
#endif
        
            pruneCols.Set(selectCols);
#ifdef COMBBLAS_DEBUG
            std::ostringstream outs;
            outs << "Number of columns needing selection: " << nselect << std::endl;
            SpParHelper::Print(outs.str());
#endif
#ifdef TIMING
            t0=MPI_Wtime();
#endif
            SpParMat<IT,NT,DER> selectedA = A.PruneColumn(pruneCols, std::less<NT>(), false);
#ifdef TIMING
            t1=MPI_Wtime();
            mcl_prunecolumntime += (t1-t0);
#endif
            if(recoverNum>0 ) // recovery can be attempted after selection
            {

                FullyDistVec<IT,NT> nnzPerColumn1 = selectedA.Reduce(Column, std::plus<NT>(), 0.0, [](NT val){return 1.0;});
                FullyDistVec<IT,NT> colSums1 = selectedA.Reduce(Column, std::plus<NT>(), 0.0);
                selectedA.FreeMemory();
  
                // slected columns with nnz < recoverNum (r)
                selectCols = recoverNum;
                selectCols = EWiseApply<NT>(selectCols, nnzPerColumn1,
                                            [](NT spval, NT dval){return spval;},
                                            [](NT spval, NT dval){return dval < spval;},
                                            false, NT());
                
                // selected columns with sum < recoverPct (pct)
                selectCols = recoverPct;
                selectCols = EWiseApply<NT>(selectCols, colSums1,
                                            [](NT spval, NT dval){return spval;},
                                            [](NT spval, NT dval){return dval < spval;},
                                            false, NT());
                
                IT n_recovery_after_select = selectCols.getnnz();
                if(n_recovery_after_select>0)
                {
                    // mclExpandVector2 does it on the original vector
                    // mclExpandVector1 does it one pruned vector
#ifdef TIMING
                    t0=MPI_Wtime();
#endif
                    A.Kselect(selectCols, recoverNum, kselectVersion); // Kselect on PrunedA might give different result
#ifdef TIMING
                    t1=MPI_Wtime();
                    mcl_kselecttime += (t1-t0);
#endif
                    pruneCols.Set(selectCols);
                    
#ifdef COMBBLAS_DEBUG
                    std::ostringstream outs1;
                    outs1 << "Number of columns needing recovery after selection: " << nselect << std::endl;
                    SpParHelper::Print(outs1.str());
#endif
                }
                
            }
        }
    }

    // final prune
#ifdef TIMING
    t0=MPI_Wtime();
#endif
    A.PruneColumn(pruneCols, std::less<NT>(), true);
#ifdef TIMING
    t1=MPI_Wtime();
    mcl_prunecolumntime += (t1-t0);
#endif
    // Add loops for empty columns
    if(recoverNum<=0 ) // if recoverNum>0, recovery would have added nonzeros in empty columns
    {
        FullyDistVec<IT,NT> nnzPerColumnA = A.Reduce(Column, std::plus<NT>(), 0.0, [](NT val){return 1.0;});
        FullyDistSpVec<IT,NT> emptyColumns(nnzPerColumnA, [](NT val){return val == 0.0;});
        emptyColumns = 1.00;
        //Ariful: We need a selective AddLoops function with a sparse vector
        //A.AddLoops(emptyColumns);
    }

}

template <typename SR, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB> 
IU EstimateFLOP 
		(SpParMat<IU,NU1,UDERA> & A, SpParMat<IU,NU2,UDERB> & B, bool clearA = false, bool clearB = false)

{
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    typedef typename UDERA::LocalIT LIA;
    typedef typename UDERB::LocalIT LIB;
	int stages, dummy; 	// last two parameters of ProductGrid are ignored for Synch multiplication
	std::shared_ptr<CommGrid> GridC = ProductGrid((A.commGrid).get(), (B.commGrid).get(), stages, dummy, dummy);		
	IU C_m = A.spSeq->getnrow();
	IU C_n = B.spSeq->getncol();
	
	//const_cast< UDERB* >(B.spSeq)->Transpose(); // do not transpose for colum-by-column multiplication

    LIA ** ARecvSizes = SpHelper::allocate2D<LIA>(UDERA::esscount, stages);
    LIB ** BRecvSizes = SpHelper::allocate2D<LIB>(UDERB::esscount, stages);
	
	SpParHelper::GetSetSizes( *(A.spSeq), ARecvSizes, (A.commGrid)->GetRowWorld());
	SpParHelper::GetSetSizes( *(B.spSeq), BRecvSizes, (B.commGrid)->GetColWorld());

	// Remotely fetched matrices are stored as pointers
	UDERA * ARecv; 
	UDERB * BRecv;
    IU local_flops = 0;

	int Aself = (A.commGrid)->GetRankInProcRow();
	int Bself = (B.commGrid)->GetRankInProcCol();	
	
	for(int i = 0; i < stages; ++i) 
	{
		std::vector<IU> ess;	
		if(i == Aself)
		{	
			ARecv = A.spSeq;	// shallow-copy 
		}
		else
		{
			ess.resize(UDERA::esscount);
			for(int j=0; j< UDERA::esscount; ++j)	
			{
				ess[j] = ARecvSizes[j][i];		// essentials of the ith matrix in this row	
			}
			ARecv = new UDERA();				// first, create the object
		}
		SpParHelper::BCastMatrix(GridC->GetRowWorld(), *ARecv, ess, i);	// then, receive its elements
		ess.clear();
		
		if(i == Bself)
		{
			BRecv = B.spSeq;	// shallow-copy
		}
		else
		{
			ess.resize(UDERB::esscount);		
			for(int j=0; j< UDERB::esscount; ++j)	
			{
				ess[j] = BRecvSizes[j][i];	
			}	
			BRecv = new UDERB();
		}
		SpParHelper::BCastMatrix(GridC->GetColWorld(), *BRecv, ess, i);	// then, receive its elements

		local_flops += EstimateLocalFLOP<SR>
						(*ARecv, *BRecv, // parameters themselves
						i != Aself, 	// 'delete A' condition
						i != Bself);	// 'delete B' condition
	}

	if(clearA && A.spSeq != NULL) {	
		delete A.spSeq;
		A.spSeq = NULL;
	}	
	if(clearB && B.spSeq != NULL) {
		delete B.spSeq;
		B.spSeq = NULL;
	}

	SpHelper::deallocate2D(ARecvSizes, UDERA::esscount);
	SpHelper::deallocate2D(BRecvSizes, UDERB::esscount);

	//if(!clearB)
	//	const_cast< UDERB* >(B.spSeq)->Transpose();	// transpose back to original

    IU global_flops = 0;
    MPI_Allreduce(&local_flops, &global_flops, 1, MPI_LONG_LONG_INT, MPI_SUM, A.getcommgrid()->GetWorld());
    return global_flops;
}

/**
 * Broadcasts A multiple times (#phases) in order to save storage in the output
 * Only uses 1/phases of C memory if the threshold/max limits are proper
 * Parameters:
 *  - computationKernel: 1 means hash-based, 2 means heap-based
 */
template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB>
SpParMat<IU,NUO,UDERO> MemEfficientSpGEMM (SpParMat<IU,NU1,UDERA> & A, SpParMat<IU,NU2,UDERB> & B,
                                           int phases, NUO hardThreshold, IU selectNum, IU recoverNum, NUO recoverPct, int kselectVersion, int computationKernel, int64_t perProcessMemory)
{
    typedef typename UDERA::LocalIT LIA;
    typedef typename UDERB::LocalIT LIB;
    typedef typename UDERO::LocalIT LIC;
    
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    if(A.getncol() != B.getnrow())
    {
        std::ostringstream outs;
        outs << "Can not multiply, dimensions does not match"<< std::endl;
        outs << A.getncol() << " != " << B.getnrow() << std::endl;
        SpParHelper::Print(outs.str());
        MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
        return SpParMat< IU,NUO,UDERO >();
    }
    if(phases <1 || phases >= A.getncol())
    {
        SpParHelper::Print("MemEfficientSpGEMM: The value of phases is too small or large. Resetting to 1.\n");
        phases = 1;
    }
    
    int stages, dummy; 	// last two parameters of ProductGrid are ignored for Synch multiplication
    std::shared_ptr<CommGrid> GridC = ProductGrid((A.commGrid).get(), (B.commGrid).get(), stages, dummy, dummy);
    
    double t0, t1, t2, t3, t4, t5;
#ifdef TIMING
    MPI_Barrier(A.getcommgrid()->GetWorld());
    t0 = MPI_Wtime();
#endif
    if(perProcessMemory>0) // estimate the number of phases permitted by memory
    {
        int p;
        MPI_Comm World = GridC->GetWorld();
        MPI_Comm_size(World,&p);
        
        int64_t perNNZMem_in = sizeof(IU)*2 + sizeof(NU1);
        int64_t perNNZMem_out = sizeof(IU)*2 + sizeof(NUO);
        
        // max nnz(A) in a porcess
        int64_t lannz = A.getlocalnnz();
        int64_t gannz;
        MPI_Allreduce(&lannz, &gannz, 1, MPIType<int64_t>(), MPI_MAX, World);
        int64_t inputMem = gannz * perNNZMem_in * 4; // for four copies (two for SUMMA)
        
        // max nnz(A^2) stored by SUMMA in a porcess
        int64_t asquareNNZ = EstPerProcessNnzSUMMA(A,B, false);
		int64_t asquareMem = asquareNNZ * perNNZMem_out * 2; // an extra copy in multiway merge and in selection/recovery step
        
        
        // estimate kselect memory
        int64_t d = ceil( (asquareNNZ * sqrt(p))/ B.getlocalcols() ); // average nnz per column in A^2 (it is an overestimate because asquareNNZ is estimated based on unmerged matrices)
        // this is equivalent to (asquareNNZ * p) / B.getcol()
        int64_t k = std::min(int64_t(std::max(selectNum, recoverNum)), d );
        int64_t kselectmem = B.getlocalcols() * k * 8 * 3;
        
        // estimate output memory
        int64_t outputNNZ = (B.getlocalcols() * k)/sqrt(p);
        int64_t outputMem = outputNNZ * perNNZMem_in * 2;
        
        //inputMem + outputMem + asquareMem/phases + kselectmem/phases < memory
        int64_t remainingMem = perProcessMemory*1000000000 - inputMem - outputMem;
        if(remainingMem > 0)
        {
            phases = 1 + (asquareMem+kselectmem) / remainingMem;
        }
        
        
        if(myrank==0)
        {
            if(remainingMem < 0)
            {
                std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n Warning: input and output memory requirement is greater than per-process avaiable memory. Keeping phase to the value supplied at the command line. The program may go out of memory and crash! \n !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
            }
#ifdef SHOW_MEMORY_USAGE
            int64_t maxMemory = kselectmem/phases + inputMem + outputMem + asquareMem / phases;
            if(maxMemory>1000000000)
            std::cout << "phases: " << phases << ": per process memory: " << perProcessMemory << " GB asquareMem: " << asquareMem/1000000000.00 << " GB" << " inputMem: " << inputMem/1000000000.00 << " GB" << " outputMem: " << outputMem/1000000000.00 << " GB" << " kselectmem: " << kselectmem/1000000000.00 << " GB" << std::endl;
            else
            std::cout << "phases: " << phases << ": per process memory: " << perProcessMemory << " GB asquareMem: " << asquareMem/1000000.00 << " MB" << " inputMem: " << inputMem/1000000.00 << " MB" << " outputMem: " << outputMem/1000000.00 << " MB" << " kselectmem: " << kselectmem/1000000.00 << " MB" << std::endl;
#endif
            
        }
    }

    //if(myrank == 0){
        //fprintf(stderr, "[MemEfficientSpGEMM] Running with phase: %d\n", phases);
    //}

#ifdef TIMING
    MPI_Barrier(A.getcommgrid()->GetWorld());
    t1 = MPI_Wtime();
    mcl_symbolictime += (t1-t0);
#endif
    
    LIA C_m = A.spSeq->getnrow();
    LIB C_n = B.spSeq->getncol();
    
    std::vector< UDERB > PiecesOfB;
    UDERB CopyB = *(B.spSeq); // we allow alias matrices as input because of this local copy
    
    CopyB.ColSplit(phases, PiecesOfB); // CopyB's memory is destroyed at this point
    MPI_Barrier(GridC->GetWorld());
    
    LIA ** ARecvSizes = SpHelper::allocate2D<LIA>(UDERA::esscount, stages);
    LIB ** BRecvSizes = SpHelper::allocate2D<LIB>(UDERB::esscount, stages);
    
    static_assert(std::is_same<LIA, LIB>::value, "local index types for both input matrices should be the same");
    static_assert(std::is_same<LIA, LIC>::value, "local index types for input and output matrices should be the same");
    
    
    SpParHelper::GetSetSizes( *(A.spSeq), ARecvSizes, (A.commGrid)->GetRowWorld());
    
    // Remotely fetched matrices are stored as pointers
    UDERA * ARecv;
    UDERB * BRecv;
    
    std::vector< UDERO > toconcatenate;
    
    int Aself = (A.commGrid)->GetRankInProcRow();
    int Bself = (B.commGrid)->GetRankInProcCol();

    stringstream strn;

    for(int p = 0; p< phases; ++p)
    {
        SpParHelper::GetSetSizes( PiecesOfB[p], BRecvSizes, (B.commGrid)->GetColWorld());
        std::vector< SpTuples<LIC,NUO>  *> tomerge;
        for(int i = 0; i < stages; ++i)
        {
            std::vector<LIA> ess;
            if(i == Aself)  ARecv = A.spSeq;	// shallow-copy
            else
            {
                ess.resize(UDERA::esscount);
                for(int j=0; j< UDERA::esscount; ++j)
                    ess[j] = ARecvSizes[j][i];		// essentials of the ith matrix in this row
                ARecv = new UDERA();				// first, create the object
            }
            
#ifdef TIMING
            MPI_Barrier(A.getcommgrid()->GetWorld());
            t0 = MPI_Wtime();
#endif
            SpParHelper::BCastMatrix(GridC->GetRowWorld(), *ARecv, ess, i);	// then, receive its elements
#ifdef TIMING
            MPI_Barrier(A.getcommgrid()->GetWorld());
            t1 = MPI_Wtime();
            mcl_Abcasttime += (t1-t0);
            /*
            int64_t nnz_local = ARecv->getnnz();
            int64_t nnz_min;
            int64_t nnz_max;
            MPI_Allreduce(&nnz_local, &nnz_min, 1, MPI_LONG_LONG_INT, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&nnz_local, &nnz_max, 1, MPI_LONG_LONG_INT, MPI_MAX, MPI_COMM_WORLD);
            strn << "Phase: " << p << ", Stage: " << i << ", A_nnz_max: " << nnz_max << ", A_nnz_min: " << nnz_min << std::endl;;
            double time_local = t1-t0;
            double time_min;
            double time_max;
            MPI_Allreduce(&time_local, &time_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&time_local, &time_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            strn << "Phase: " << p << ", Stage: " << i << ", A_bcast_time_max: " << time_max << ", A_bcast_time_min: " << time_min << std::endl;;
            */

#endif
            ess.clear();

            if(i == Bself)  BRecv = &(PiecesOfB[p]);	// shallow-copy
            else
            {
                ess.resize(UDERB::esscount);
                for(int j=0; j< UDERB::esscount; ++j)
                    ess[j] = BRecvSizes[j][i];
                BRecv = new UDERB();
            }
#ifdef TIMING
            MPI_Barrier(A.getcommgrid()->GetWorld());
            double t2=MPI_Wtime();
#endif
            SpParHelper::BCastMatrix(GridC->GetColWorld(), *BRecv, ess, i);	// then, receive its elements
#ifdef TIMING
            MPI_Barrier(A.getcommgrid()->GetWorld());
            double t3=MPI_Wtime();
            mcl_Bbcasttime += (t3-t2);
            /*
            nnz_local = BRecv->getnnz();
            MPI_Allreduce(&nnz_local, &nnz_min, 1, MPI_LONG_LONG_INT, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&nnz_local, &nnz_max, 1, MPI_LONG_LONG_INT, MPI_MAX, MPI_COMM_WORLD);
            strn << "Phase: " << p << ", Stage: " << i << ", B_nnz_max: " << nnz_max << ", B_nnz_min: " << nnz_min << std::endl;;
            time_local = t3-t2;
            MPI_Allreduce(&time_local, &time_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&time_local, &time_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            strn << "Phase: " << p << ", Stage: " << i << ", B_bcast_time_max: " << time_max << ", B_bcast_time_min: " << time_min << std::endl;;
            */
#endif
            
            
#ifdef TIMING
            MPI_Barrier(A.getcommgrid()->GetWorld());
            double t4=MPI_Wtime();
#endif
            SpTuples<LIC,NUO> * C_cont;
            //if(computationKernel == 1) C_cont = LocalSpGEMMHash<SR, NUO>(*ARecv, *BRecv,i != Aself, i != Bself, false); // Hash SpGEMM without per-column sorting
            //else if(computationKernel == 2) C_cont=LocalSpGEMM<SR, NUO>(*ARecv, *BRecv,i != Aself, i != Bself);
            if(computationKernel == 1) C_cont = LocalSpGEMMHash<SR, NUO>(*ARecv, *BRecv, false, false, false); // Hash SpGEMM without per-column sorting
            else if(computationKernel == 2) C_cont=LocalSpGEMM<SR, NUO>(*ARecv, *BRecv, false, false);
            
            // Explicitly delete ARecv and BRecv because it effectively does not get freed inside LocalSpGEMM function
            if(i != Bself && (!BRecv->isZero())) delete BRecv;
            if(i != Aself && (!ARecv->isZero())) delete ARecv;

#ifdef TIMING
            MPI_Barrier(A.getcommgrid()->GetWorld());
            double t5=MPI_Wtime();
            mcl_localspgemmtime += (t5-t4);
            /*
            nnz_local = C_cont->getnnz();
            MPI_Allreduce(&nnz_local, &nnz_min, 1, MPI_LONG_LONG_INT, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&nnz_local, &nnz_max, 1, MPI_LONG_LONG_INT, MPI_MAX, MPI_COMM_WORLD);
            strn << "Phase: " << p << ", Stage: " << i << ", C_nnz_max: " << nnz_max << ", C_nnz_min: " << nnz_min << std::endl;;
            time_local = t5-t4;
            MPI_Allreduce(&time_local, &time_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&time_local, &time_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            strn << "Phase: " << p << ", Stage: " << i << ", spgemm_time_max: " << time_max << ", spgemm_time_min: " << time_min << std::endl;;
            */
#endif

            if(!C_cont->isZero())
                tomerge.push_back(C_cont);
            else
                delete C_cont;
            
        }   // all stages executed
        
#ifdef SHOW_MEMORY_USAGE
        int64_t gcnnz_unmerged, lcnnz_unmerged = 0;
         for(size_t i = 0; i < tomerge.size(); ++i)
         {
              lcnnz_unmerged += tomerge[i]->getnnz();
         }
        MPI_Allreduce(&lcnnz_unmerged, &gcnnz_unmerged, 1, MPIType<int64_t>(), MPI_MAX, MPI_COMM_WORLD);
        int64_t summa_memory = gcnnz_unmerged*20;//(gannz*2 + phase_nnz + gcnnz_unmerged + gannz + gannz/phases) * 20; // last two for broadcasts
        
        if(myrank==0)
        {
            if(summa_memory>1000000000)
                std::cout << p+1 << ". unmerged: " << summa_memory/1000000000.00 << "GB " ;
            else
                std::cout << p+1 << ". unmerged: " << summa_memory/1000000.00 << " MB " ;
            
        }
#endif

#ifdef TIMING
        MPI_Barrier(A.getcommgrid()->GetWorld());
        double t6=MPI_Wtime();
#endif
        // TODO: MultiwayMerge can directly return UDERO inorder to avoid the extra copy
        SpTuples<LIC,NUO> * OnePieceOfC_tuples;
        if(computationKernel == 1) OnePieceOfC_tuples = MultiwayMergeHash<SR>(tomerge, C_m, PiecesOfB[p].getncol(), true, false);
        else if(computationKernel == 2) OnePieceOfC_tuples = MultiwayMerge<SR>(tomerge, C_m, PiecesOfB[p].getncol(), true);
        
#ifdef SHOW_MEMORY_USAGE
        int64_t gcnnz_merged, lcnnz_merged ;
        lcnnz_merged = OnePieceOfC_tuples->getnnz();
        MPI_Allreduce(&lcnnz_merged, &gcnnz_merged, 1, MPIType<int64_t>(), MPI_MAX, MPI_COMM_WORLD);
       
        // TODO: we can remove gcnnz_merged memory here because we don't need to concatenate anymore
        int64_t merge_memory = gcnnz_merged*2*20;//(gannz*2 + phase_nnz + gcnnz_unmerged + gcnnz_merged*2) * 20;
        
        if(myrank==0)
        {
            if(merge_memory>1000000000)
                std::cout << " merged: " << merge_memory/1000000000.00 << "GB " ;
            else
                std::cout << " merged: " << merge_memory/1000000.00 << " MB " ;
        }
#endif
        
        
#ifdef TIMING
        MPI_Barrier(A.getcommgrid()->GetWorld());
        double t7=MPI_Wtime();
        mcl_multiwaymergetime += (t7-t6);
#endif
        UDERO * OnePieceOfC = new UDERO(* OnePieceOfC_tuples, false);
        delete OnePieceOfC_tuples;
        
        SpParMat<IU,NUO,UDERO> OnePieceOfC_mat(OnePieceOfC, GridC);
        MCLPruneRecoverySelect(OnePieceOfC_mat, hardThreshold, selectNum, recoverNum, recoverPct, kselectVersion);

#ifdef SHOW_MEMORY_USAGE
        int64_t gcnnz_pruned, lcnnz_pruned ;
        lcnnz_pruned = OnePieceOfC_mat.getlocalnnz();
        MPI_Allreduce(&lcnnz_pruned, &gcnnz_pruned, 1, MPIType<int64_t>(), MPI_MAX, MPI_COMM_WORLD);
        
        
        // TODO: we can remove gcnnz_merged memory here because we don't need to concatenate anymore
        int64_t prune_memory = gcnnz_pruned*2*20;//(gannz*2 + phase_nnz + gcnnz_pruned*2) * 20 + kselectmem; // 3 extra copies of OnePieceOfC_mat, we can make it one extra copy!
        //phase_nnz += gcnnz_pruned;
        
        if(myrank==0)
        {
            if(prune_memory>1000000000)
                std::cout << "Prune: " << prune_memory/1000000000.00 << "GB " << std::endl ;
            else
                std::cout << "Prune: " << prune_memory/1000000.00 << " MB " << std::endl ;
            
        }
#endif
        
        // ABAB: Change this to accept pointers to objects
        toconcatenate.push_back(OnePieceOfC_mat.seq());
    }
    SpParHelper::Print(strn.str());
    
    UDERO * C = new UDERO(0,C_m, C_n,0);
    C->ColConcatenate(toconcatenate); // ABAB: Change this to accept a vector of pointers to pointers to DER objects

    SpHelper::deallocate2D(ARecvSizes, UDERA::esscount);
    SpHelper::deallocate2D(BRecvSizes, UDERA::esscount);
    return SpParMat<IU,NUO,UDERO> (C, GridC);
}

template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB>
int CalculateNumberOfPhases (SpParMat<IU,NU1,UDERA> & A, SpParMat<IU,NU2,UDERB> & B,
        NUO hardThreshold, IU selectNum, IU recoverNum, NUO recoverPct, int kselectVersion, int64_t perProcessMemory){
    
    int phases;

    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    
    int stages, dummy; 	// last two parameters of ProductGrid are ignored for Synch multiplication
    std::shared_ptr<CommGrid> GridC = ProductGrid((A.commGrid).get(), (B.commGrid).get(), stages, dummy, dummy);
    
    double t0, t1, t2, t3, t4, t5;
    int p;
    MPI_Comm World = GridC->GetWorld();
    MPI_Comm_size(World,&p);
    
    int64_t perNNZMem_in = sizeof(IU)*2 + sizeof(NU1);
    int64_t perNNZMem_out = sizeof(IU)*2 + sizeof(NUO);
    
    // max nnz(A) in a porcess
    int64_t lannz = A.getlocalnnz();
    int64_t gannz;
    MPI_Allreduce(&lannz, &gannz, 1, MPIType<int64_t>(), MPI_MAX, World);
    int64_t inputMem = gannz * perNNZMem_in * 4; // for four copies (two for SUMMA)
    
    // max nnz(A^2) stored by SUMMA in a porcess
    int64_t asquareNNZ = EstPerProcessNnzSUMMA(A,B, false);
    int64_t asquareMem = asquareNNZ * perNNZMem_out * 2; // an extra copy in multiway merge and in selection/recovery step
    
    
    // estimate kselect memory
    int64_t d = ceil( (asquareNNZ * sqrt(p))/ B.getlocalcols() ); // average nnz per column in A^2 (it is an overestimate because asquareNNZ is estimated based on unmerged matrices)
    // this is equivalent to (asquareNNZ * p) / B.getcol()
    int64_t k = std::min(int64_t(std::max(selectNum, recoverNum)), d );
    int64_t kselectmem = B.getlocalcols() * k * 8 * 3;
    
    // estimate output memory
    int64_t outputNNZ = (B.getlocalcols() * d)/sqrt(p);
    //int64_t outputNNZ = (B.getlocalcols() * k)/sqrt(p); // if kselect is used
    int64_t outputMem = outputNNZ * perNNZMem_in * 2;
    
    //inputMem + outputMem + asquareMem/phases + kselectmem/phases < memory
    //int64_t remainingMem = perProcessMemory*1000000000 - inputMem - outputMem;
    int64_t remainingMem = perProcessMemory*1000000000 - inputMem; // if each phase result is discarded
    //if(remainingMem > 0)
    //{
        //phases = 1 + (asquareMem+kselectmem) / remainingMem;
    //}
    phases = 1 + asquareMem / remainingMem;
    return phases;
}

/*
 * A^2 with incremental MCL matrix
 * Non-zeroes are heavily skewed on the diagonals, hence SUMMA is suboptimal
 * We seprate diagonal elements from offdiagonals, M = D + A; D is diagonal and A is off-diagonal matrix;
 * D can be thought of sparse vector, but we use dense vector here to avoid technical difficult;
 * M^2 = (D+A)^2 = D^2 + A^2 + DxA + AxD
 * A^2: SUMMA (Verify whether SUMMA or 1D multiplication would be optimal?)
 * D^2: Elementwise squaring of vector
 * DxA: Vector dimapply along row of A
 * AxD: Vector dimapply along column of A
 * */
template <typename SR, typename ITA, typename NTA, typename DERA>
SpParMat<ITA, NTA, DERA> IncrementalMCLSquare(SpParMat<ITA, NTA, DERA> & A,
                                           int phases, NTA hardThreshold, ITA selectNum, ITA recoverNum, NTA recoverPct, int kselectVersion, int computationKernel, int64_t perProcessMemory)
{
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    double t0, t1, t2, t3, t4, t5;

    // Because we are squaring A, it is safe to assume that same commgrid as A would be used for all distributed matrices
    std::shared_ptr<CommGrid> commGrid = A.getcommgrid();
    MPI_Comm rowWorld = commGrid->GetRowWorld();
    MPI_Comm colWorld = commGrid->GetColWorld();

#ifdef TIMING
    MPI_Barrier(commGrid->GetWorld());
    t0 = MPI_Wtime();
#endif
    
    SpParMat<ITA, NTA, DERA> X(commGrid);
    {
        // Doing this inside here to force destruction of temporary objects once X is computed
        
        // MTH: What are mechanisms exist in CombBLAS to separate the diagonal elements?
        SpParMat<ITA, NTA, DERA> D(A);
        A.RemoveLoops(); // Remove diagonals, makes A as off-diagonal matrix
        D.SetDifference(A); // Remove offdiagonals

        FullyDistVec<ITA, NTA> diag = D.Reduce(Column, plus<NTA>(), 0.0); // diag: Vector with diagonal entries of D

        SpParMat<ITA, NTA, DERA> AD(A);
        AD.DimApply(Column, diag, [](NTA mv, NTA vv){return mv * vv;});
    	AD.Prune([](NTA val) { return val <= 1e-8; }, true);

        SpParMat<ITA, NTA, DERA> DA(A);
        DA.DimApply(Row, diag, [](NTA mv, NTA vv){return mv * vv;});

    	DA.Prune([](NTA val) { return val <= 1e-8; }, true);

        X = D;
    	X.Apply([](auto val) { return std::pow(val, 2); });

        X += DA;
        X += AD;
    }
#ifdef TIMING
    MPI_Barrier(commGrid->GetWorld());
    t1 = MPI_Wtime();
    if(myrank == 0){
        fprintf(stderr, "[IncrementalMCLSquare]:\tTime to calculate AD+DA+D^2: %lf\n", t1-t0);
    }
#endif

    if(phases < 1 || phases >= A.getncol())
    {
        SpParHelper::Print("[IncrementalMCLSquare]:\tThe value of phases is too small or large. Resetting to 1.\n");
        phases = 1;
    }
    
    int stages = commGrid->GetGridRows(); // As we use square grid number of rows would also mean number of columns in the grid
    float lb = A.LoadImbalance();
    //if(myrank == 0) fprintf(stderr, "[IncrementalMClSquare]:\tLoad imbalance of the matrix involved in SUMMA: %f\n", lb);

#ifdef TIMING
    MPI_Barrier(commGrid->GetWorld());
    t0 = MPI_Wtime();
#endif
    if(perProcessMemory>0) // estimate the number of phases permitted by memory
    {
        //int p;
        //MPI_Comm World = commGrid->GetWorld();
        //MPI_Comm_size(World,&p);
        
        //MPI_Barrier(commGrid->GetWorld());
        //if(myrank == 0) fprintf(stderr, "[IncrementalMCLSquare]:\tCheckpoint 1.1\n");

        //int64_t perNNZMem_in = sizeof(ITA)*2 + sizeof(NTA);
        //int64_t perNNZMem_out = sizeof(ITA)*2 + sizeof(NTA);

        //MPI_Barrier(commGrid->GetWorld());
        //if(myrank == 0) fprintf(stderr, "[IncrementalMCLSquare]:\tCheckpoint 1.2\n");
        
        //// max nnz(A) in a process
        //int64_t lannz = A.getlocalnnz();
        //int64_t gannz;
        //MPI_Allreduce(&lannz, &gannz, 1, MPIType<int64_t>(), MPI_MAX, World);
        //int64_t inputMem = gannz * perNNZMem_in * 5; // for five copies (two for SUMMA, one for X)
                                                    
        //MPI_Barrier(commGrid->GetWorld());
        //if(myrank == 0) fprintf(stderr, "[IncrementalMCLSquare]:\tCheckpoint 1.3\n");
        
        //// max nnz(A^2) stored by SUMMA in a process
        //SpParMat<ITA, NTA, DERA> B(A);
        //int64_t asquareNNZ = EstPerProcessNnzSUMMA(A,B, false);
        //int64_t asquareMem = asquareNNZ * perNNZMem_out * 2; // an extra copy in multiway merge and in selection/recovery step
                                                            
        //MPI_Barrier(commGrid->GetWorld());
        //if(myrank == 0) fprintf(stderr, "[IncrementalMCLSquare]:\tCheckpoint 1.4\n");
        
        //// estimate kselect memory
        //int64_t d = ceil( (asquareNNZ * sqrt(p))/ A.getlocalcols() ); // average nnz per column in A^2 (it is an overestimate because asquareNNZ is estimated based on unmerged matrices)
        //// this is equivalent to (asquareNNZ * p) / A.getcol()
        //int64_t k = std::min(int64_t(std::max(selectNum, recoverNum)), d );
        //int64_t kselectmem = A.getlocalcols() * k * 8 * 3;

        //MPI_Barrier(commGrid->GetWorld());
        //if(myrank == 0) fprintf(stderr, "[IncrementalMCLSquare]:\tCheckpoint 1.5\n");
        
        //// estimate output memory
        //int64_t outputNNZ = (A.getlocalcols() * k)/sqrt(p);
        //int64_t outputMem = outputNNZ * perNNZMem_in * 2;

        //MPI_Barrier(commGrid->GetWorld());
        //if(myrank == 0) fprintf(stderr, "[IncrementalMCLSquare]:\tCheckpoint 1.6\n");
        
        ////inputMem + outputMem + asquareMem/phases + kselectmem/phases < memory
        //int64_t remainingMem = perProcessMemory*1000000000 - inputMem - outputMem;
        //if(remainingMem > 0)
        //{
            //phases = 1 + (asquareMem+kselectmem) / remainingMem;
        //}
        //MPI_Barrier(commGrid->GetWorld());
        //if(myrank == 0) fprintf(stderr, "[IncrementalMCLSquare]:\tCheckpoint 1.7\n");
        
        
        //if(myrank==0)
        //{
            //if(remainingMem < 0)
            //{
                //std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n Warning: input and output memory requirement is greater than per-process avaiable memory. Keeping phase to the value supplied at the command line. The program may go out of memory and crash! \n !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
            //}
//#ifdef SHOW_MEMORY_USAGE
            //int64_t maxMemory = kselectmem/phases + inputMem + outputMem + asquareMem / phases;
            //if(maxMemory>1000000000)
            //std::cout << "phases: " << phases << ": per process memory: " << perProcessMemory << " GB asquareMem: " << asquareMem/1000000000.00 << " GB" << " inputMem: " << inputMem/1000000000.00 << " GB" << " outputMem: " << outputMem/1000000000.00 << " GB" << " kselectmem: " << kselectmem/1000000000.00 << " GB" << std::endl;
            //else
            //std::cout << "phases: " << phases << ": per process memory: " << perProcessMemory << " GB asquareMem: " << asquareMem/1000000.00 << " MB" << " inputMem: " << inputMem/1000000.00 << " MB" << " outputMem: " << outputMem/1000000.00 << " MB" << " kselectmem: " << kselectmem/1000000.00 << " MB" << std::endl;
//#endif
            
        //}
    }

    if(myrank == 0){
        fprintf(stderr, "[IncrementalMCLSquare]:\tRunning with phase: %d\n", phases);
    }

#ifdef TIMING
    MPI_Barrier(commGrid->GetWorld());
    t1 = MPI_Wtime();
    mcl_symbolictime += (t1-t0);
#endif
    
    ITA C_m = A.seqptr()->getnrow();
    ITA C_n = A.seqptr()->getncol();

    std::vector<DERA> PiecesOfB;
    DERA CopyA = *(A.seqptr()); // CopyA is effectively B because of A^2 computation
    CopyA.ColSplit(phases, PiecesOfB); // CopyA's memory is destroyed at this point
    
    std::vector<DERA> PiecesOfX;
    DERA CopyX = *(X.seqptr()); // Make a copy in order to use the ColSplit function
    CopyX.ColSplit(phases, PiecesOfX); // CopyX's memory is destroyed at this point

    X.FreeMemory(); // X is not needed anymore after splitting into `phases` pieces
    MPI_Barrier(commGrid->GetWorld());
    
    ITA ** ARecvSizes = SpHelper::allocate2D<ITA>(DERA::esscount, stages);
    ITA ** BRecvSizes = SpHelper::allocate2D<ITA>(DERA::esscount, stages);
    
    SpParHelper::GetSetSizes( *(A.seqptr()), ARecvSizes, commGrid->GetRowWorld());
    
    // Remotely fetched matrices are stored as pointers
    DERA * ARecv;
    DERA * BRecv;
    
    std::vector< DERA > toconcatenate;
    
    int Aself = commGrid->GetRankInProcRow();
    int Bself = commGrid->GetRankInProcCol();

    stringstream strn;

    for(int p = 0; p< phases; ++p)
    {
        SpParHelper::GetSetSizes( PiecesOfB[p], BRecvSizes, colWorld);
        std::vector< SpTuples<ITA,NTA>  *> tomerge;

        SpTuples<ITA,NTA> * PieceOfX = new SpTuples<ITA,NTA>(PiecesOfX[p]); // Convert target piece of X to SpTuples
        tomerge.push_back(PieceOfX); // Will be merged together with the result of A^2 with non-diagonal entries

        for(int i = 0; i < stages; ++i)
        {
            std::vector<ITA> ess;
            if(i == Aself)  ARecv = A.seqptr();	// shallow-copy
            else
            {
                ess.resize(DERA::esscount);
                for(int j=0; j< DERA::esscount; ++j)
                    ess[j] = ARecvSizes[j][i];		// essentials of the ith matrix in this row
                ARecv = new DERA();				// first, create the object
            }
            
#ifdef TIMING
            MPI_Barrier(commGrid->GetWorld());
            t0 = MPI_Wtime();
#endif
            SpParHelper::BCastMatrix(commGrid->GetRowWorld(), *ARecv, ess, i);	// then, receive its elements
#ifdef TIMING
            MPI_Barrier(commGrid->GetWorld());
            t1 = MPI_Wtime();
            mcl_Abcasttime += (t1-t0);
            /*
            int64_t nnz_local = ARecv->getnnz();
            int64_t nnz_min;
            int64_t nnz_max;
            MPI_Allreduce(&nnz_local, &nnz_min, 1, MPI_LONG_LONG_INT, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&nnz_local, &nnz_max, 1, MPI_LONG_LONG_INT, MPI_MAX, MPI_COMM_WORLD);
            strn << "Phase: " << p << ", Stage: " << i << ", A_nnz_max: " << nnz_max << ", A_nnz_min: " << nnz_min << std::endl;;
            double time_local = t1-t0;
            double time_min;
            double time_max;
            MPI_Allreduce(&time_local, &time_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&time_local, &time_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            strn << "Phase: " << p << ", Stage: " << i << ", A_bcast_time_max: " << time_max << ", A_bcast_time_min: " << time_min << std::endl;;
            */

#endif
            ess.clear();

            if(i == Bself)  BRecv = &(PiecesOfB[p]);	// shallow-copy
            else
            {
                ess.resize(DERA::esscount);
                for(int j=0; j< DERA::esscount; ++j)
                    ess[j] = BRecvSizes[j][i];
                BRecv = new DERA();
            }
#ifdef TIMING
            MPI_Barrier(commGrid->GetWorld());
            double t2=MPI_Wtime();
#endif
            SpParHelper::BCastMatrix(commGrid->GetColWorld(), *BRecv, ess, i);	// then, receive its elements
#ifdef TIMING
            MPI_Barrier(commGrid->GetWorld());
            double t3=MPI_Wtime();
            mcl_Bbcasttime += (t3-t2);
            /*
            nnz_local = BRecv->getnnz();
            MPI_Allreduce(&nnz_local, &nnz_min, 1, MPI_LONG_LONG_INT, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&nnz_local, &nnz_max, 1, MPI_LONG_LONG_INT, MPI_MAX, MPI_COMM_WORLD);
            strn << "Phase: " << p << ", Stage: " << i << ", B_nnz_max: " << nnz_max << ", B_nnz_min: " << nnz_min << std::endl;;
            time_local = t3-t2;
            MPI_Allreduce(&time_local, &time_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&time_local, &time_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            strn << "Phase: " << p << ", Stage: " << i << ", B_bcast_time_max: " << time_max << ", B_bcast_time_min: " << time_min << std::endl;;
            */
#endif
            
            
#ifdef TIMING
            MPI_Barrier(commGrid->GetWorld());
            double t4=MPI_Wtime();
#endif
            SpTuples<ITA,NTA> * C_cont;
            //if(computationKernel == 1) C_cont = LocalSpGEMMHash<SR, NUO>(*ARecv, *BRecv,i != Aself, i != Bself, false); // Hash SpGEMM without per-column sorting
            //else if(computationKernel == 2) C_cont=LocalSpGEMM<SR, NUO>(*ARecv, *BRecv,i != Aself, i != Bself);
            if(computationKernel == 1) C_cont = LocalSpGEMMHash<SR, NTA>(*ARecv, *BRecv, false, false, false); // Hash SpGEMM without per-column sorting
            else if(computationKernel == 2) C_cont=LocalSpGEMM<SR, NTA>(*ARecv, *BRecv, false, false);
            
            // Explicitly delete ARecv and BRecv because it effectively does not get freed inside LocalSpGEMM function
            if(i != Bself && (!BRecv->isZero())) delete BRecv;
            if(i != Aself && (!ARecv->isZero())) delete ARecv;

#ifdef TIMING
            MPI_Barrier(commGrid->GetWorld());
            double t5=MPI_Wtime();
            mcl_localspgemmtime += (t5-t4);
            /*
            nnz_local = C_cont->getnnz();
            MPI_Allreduce(&nnz_local, &nnz_min, 1, MPI_LONG_LONG_INT, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&nnz_local, &nnz_max, 1, MPI_LONG_LONG_INT, MPI_MAX, MPI_COMM_WORLD);
            strn << "Phase: " << p << ", Stage: " << i << ", C_nnz_max: " << nnz_max << ", C_nnz_min: " << nnz_min << std::endl;;
            time_local = t5-t4;
            MPI_Allreduce(&time_local, &time_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&time_local, &time_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            strn << "Phase: " << p << ", Stage: " << i << ", spgemm_time_max: " << time_max << ", spgemm_time_min: " << time_min << std::endl;;
            */
#endif

            if(!C_cont->isZero())
                tomerge.push_back(C_cont);
            else
                delete C_cont;
            
        }   // all stages executed
        
#ifdef SHOW_MEMORY_USAGE
        int64_t gcnnz_unmerged, lcnnz_unmerged = 0;
        for(size_t i = 0; i < tomerge.size(); ++i)
        {
             lcnnz_unmerged += tomerge[i]->getnnz();
        }
        MPI_Allreduce(&lcnnz_unmerged, &gcnnz_unmerged, 1, MPIType<int64_t>(), MPI_MAX, MPI_COMM_WORLD);
        int64_t summa_memory = gcnnz_unmerged*20;//(gannz*2 + phase_nnz + gcnnz_unmerged + gannz + gannz/phases) * 20; // last two for broadcasts
        
        if(myrank==0)
        {
            if(summa_memory>1000000000)
                std::cout << p+1 << ". unmerged: " << summa_memory/1000000000.00 << "GB " ;
            else
                std::cout << p+1 << ". unmerged: " << summa_memory/1000000.00 << " MB " ;
            
        }
#endif

#ifdef TIMING
        MPI_Barrier(commGrid->GetWorld());
        double t6=MPI_Wtime();
#endif
        // TODO: MultiwayMerge can directly return UDERO inorder to avoid the extra copy
        SpTuples<ITA,NTA> * OnePieceOfC_tuples;
        if(computationKernel == 1) OnePieceOfC_tuples = MultiwayMergeHash<SR>(tomerge, C_m, PiecesOfB[p].getncol(), true, false);
        else if(computationKernel == 2) OnePieceOfC_tuples = MultiwayMerge<SR>(tomerge, C_m, PiecesOfB[p].getncol(), true);
        
#ifdef SHOW_MEMORY_USAGE
        int64_t gcnnz_merged, lcnnz_merged ;
        lcnnz_merged = OnePieceOfC_tuples->getnnz();
        MPI_Allreduce(&lcnnz_merged, &gcnnz_merged, 1, MPIType<int64_t>(), MPI_MAX, MPI_COMM_WORLD);
       
        // TODO: we can remove gcnnz_merged memory here because we don't need to concatenate anymore
        int64_t merge_memory = gcnnz_merged*2*20;//(gannz*2 + phase_nnz + gcnnz_unmerged + gcnnz_merged*2) * 20;
        
        if(myrank==0)
        {
            if(merge_memory>1000000000)
                std::cout << " merged: " << merge_memory/1000000000.00 << "GB " ;
            else
                std::cout << " merged: " << merge_memory/1000000.00 << " MB " ;
        }
#endif
        
        
#ifdef TIMING
        MPI_Barrier(commGrid->GetWorld());
        double t7=MPI_Wtime();
        mcl_multiwaymergetime += (t7-t6);
#endif
        DERA * OnePieceOfC = new DERA(* OnePieceOfC_tuples, false);
        delete OnePieceOfC_tuples;
        
        SpParMat<ITA,NTA,DERA> OnePieceOfC_mat(OnePieceOfC, commGrid);
        MCLPruneRecoverySelect(OnePieceOfC_mat, hardThreshold, selectNum, recoverNum, recoverPct, kselectVersion);

#ifdef SHOW_MEMORY_USAGE
        int64_t gcnnz_pruned, lcnnz_pruned ;
        lcnnz_pruned = OnePieceOfC_mat.getlocalnnz();
        MPI_Allreduce(&lcnnz_pruned, &gcnnz_pruned, 1, MPIType<int64_t>(), MPI_MAX, MPI_COMM_WORLD);
        
        
        // TODO: we can remove gcnnz_merged memory here because we don't need to concatenate anymore
        int64_t prune_memory = gcnnz_pruned*2*20;//(gannz*2 + phase_nnz + gcnnz_pruned*2) * 20 + kselectmem; // 3 extra copies of OnePieceOfC_mat, we can make it one extra copy!
        //phase_nnz += gcnnz_pruned;
        
        if(myrank==0)
        {
            if(prune_memory>1000000000)
                std::cout << "Prune: " << prune_memory/1000000000.00 << "GB " << std::endl ;
            else
                std::cout << "Prune: " << prune_memory/1000000.00 << " MB " << std::endl ;
            
        }
#endif
        
        // ABAB: Change this to accept pointers to objects
        toconcatenate.push_back(OnePieceOfC_mat.seq());
    }
    SpParHelper::Print(strn.str());
    
    DERA * C = new DERA(0,C_m, C_n,0);
    C->ColConcatenate(toconcatenate); // ABAB: Change this to accept a vector of pointers to pointers to DERA objects

    SpHelper::deallocate2D(ARecvSizes, DERA::esscount);
    SpHelper::deallocate2D(BRecvSizes, DERA::esscount);
    return SpParMat<ITA,NTA,DERA> (C, commGrid);
}


/**
 * Parallel C = A*B routine that uses a double buffered broadcasting scheme 
 * @pre { Input matrices, A and B, should not alias }
 * Most memory efficient version available. Total stages: 2*sqrt(p)
 * Memory requirement during first sqrt(p) stages: <= (3/2)*(nnz(A)+nnz(B))+(1/2)*nnz(C)
 * Memory requirement during second sqrt(p) stages: <= nnz(A)+nnz(B)+nnz(C)
 * Final memory requirement: nnz(C) if clearA and clearB are true 
 **/  
template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB> 
SpParMat<IU,NUO,UDERO> Mult_AnXBn_DoubleBuff
		(SpParMat<IU,NU1,UDERA> & A, SpParMat<IU,NU2,UDERB> & B, bool clearA = false, bool clearB = false )

{
	if(!CheckSpGEMMCompliance(A,B) )
	{
		return SpParMat< IU,NUO,UDERO >();
	}
	typedef typename UDERA::LocalIT LIA;
	typedef typename UDERB::LocalIT LIB;
	typedef typename UDERO::LocalIT LIC;

	static_assert(std::is_same<LIA, LIB>::value, "local index types for both input matrices should be the same");
	static_assert(std::is_same<LIA, LIC>::value, "local index types for input and output matrices should be the same");

	int stages, dummy; 	// last two parameters of ProductGrid are ignored for Synch multiplication
	std::shared_ptr<CommGrid> GridC = ProductGrid((A.commGrid).get(), (B.commGrid).get(), stages, dummy, dummy);
	LIA C_m = A.spSeq->getnrow();
	LIB C_n = B.spSeq->getncol();
    
	UDERA * A1seq = new UDERA();
	UDERA * A2seq = new UDERA(); 
	UDERB * B1seq = new UDERB();
	UDERB * B2seq = new UDERB();
	(A.spSeq)->Split( *A1seq, *A2seq); 
	const_cast< UDERB* >(B.spSeq)->Transpose();
	(B.spSeq)->Split( *B1seq, *B2seq);
    
    // Transpose back for the column-by-column algorithm
    const_cast< UDERB* >(B1seq)->Transpose();
    const_cast< UDERB* >(B2seq)->Transpose();

	LIA ** ARecvSizes = SpHelper::allocate2D<LIA>(UDERA::esscount, stages);
	LIB ** BRecvSizes = SpHelper::allocate2D<LIB>(UDERB::esscount, stages);

	SpParHelper::GetSetSizes( *A1seq, ARecvSizes, (A.commGrid)->GetRowWorld());
	SpParHelper::GetSetSizes( *B1seq, BRecvSizes, (B.commGrid)->GetColWorld());

	// Remotely fetched matrices are stored as pointers
	UDERA * ARecv; 
	UDERB * BRecv;
	std::vector< SpTuples<LIC,NUO>  *> tomerge;

	int Aself = (A.commGrid)->GetRankInProcRow();
	int Bself = (B.commGrid)->GetRankInProcCol();	

	for(int i = 0; i < stages; ++i) 
	{
		std::vector<LIA> ess;	
		if(i == Aself)
		{	
			ARecv = A1seq;	// shallow-copy 
		}
		else
		{
			ess.resize(UDERA::esscount);
			for(int j=0; j< UDERA::esscount; ++j)	
			{
				ess[j] = ARecvSizes[j][i];		// essentials of the ith matrix in this row	
			}
			ARecv = new UDERA();				// first, create the object
		}
		SpParHelper::BCastMatrix(GridC->GetRowWorld(), *ARecv, ess, i);	// then, receive its elements	
		ess.clear();	
		if(i == Bself)
		{
			BRecv = B1seq;	// shallow-copy
		}
		else
		{
			ess.resize(UDERB::esscount);		
			for(int j=0; j< UDERB::esscount; ++j)	
			{
				ess[j] = BRecvSizes[j][i];	
			}	
			BRecv = new UDERB();
		}
		SpParHelper::BCastMatrix(GridC->GetColWorld(), *BRecv, ess, i);	// then, receive its elements
		
		// before activating this remove transposing B1seq
            
        //SpTuples<LIC,NUO> * C_cont = MultiplyReturnTuples<SR, NUO>
                        //(*ARecv, *BRecv,  //parameters themselves
                        //false, true,	 //transpose information (B is transposed)
                        //false, 	 //'delete A' condition
                        //false);	 // 'delete B' condition
        
		SpTuples<LIC,NUO> * C_cont = LocalHybridSpGEMM<SR, NUO>
			(*ARecv, *BRecv, // parameters themselves
			false,    // 'delete A' condition
			false);   // 'delete B' condition
        
        if(i != Bself && (!BRecv->isZero())) delete BRecv;
        if(i != Aself && (!ARecv->isZero())) delete ARecv;
        
        
		
		if(!C_cont->isZero())
			tomerge.push_back(C_cont);
		else
			delete C_cont;
	}
	if(clearA) delete A1seq;
	if(clearB) delete B1seq;
	
	// Set the new dimensions
	SpParHelper::GetSetSizes( *A2seq, ARecvSizes, (A.commGrid)->GetRowWorld());
	SpParHelper::GetSetSizes( *B2seq, BRecvSizes, (B.commGrid)->GetColWorld());

	// Start the second round
	for(int i = 0; i < stages; ++i) 
	{
		std::vector<LIA> ess;	
		if(i == Aself)
		{	
			ARecv = A2seq;	// shallow-copy 
		}
		else
		{
			ess.resize(UDERA::esscount);
			for(int j=0; j< UDERA::esscount; ++j)	
			{
				ess[j] = ARecvSizes[j][i];		// essentials of the ith matrix in this row	
			}
			ARecv = new UDERA();				// first, create the object
		}

		SpParHelper::BCastMatrix(GridC->GetRowWorld(), *ARecv, ess, i);	// then, receive its elements	
		ess.clear();	
		
		if(i == Bself)
		{
			BRecv = B2seq;	// shallow-copy
		}
		else
		{
			ess.resize(UDERB::esscount);		
			for(int j=0; j< UDERB::esscount; ++j)	
			{
				ess[j] = BRecvSizes[j][i];	
			}	
			BRecv = new UDERB();
		}
		SpParHelper::BCastMatrix(GridC->GetColWorld(), *BRecv, ess, i);	// then, receive its elements

        	// before activating this remove transposing B2seq
            
        //SpTuples<LIC,NUO> * C_cont = MultiplyReturnTuples<SR, NUO>
                        //(*ARecv, *BRecv,  //parameters themselves
                        //false, true,	 //transpose information (B is transposed)
                        //false, 	 //'delete A' condition
                        //false);	 //'delete B' condition
        
		SpTuples<LIC,NUO> * C_cont = LocalHybridSpGEMM<SR, NUO>
			(*ARecv, *BRecv, // parameters themselves
			false,    // 'delete A' condition
			false);   // 'delete B' condition


        if(i != Bself && (!BRecv->isZero())) delete BRecv;
        if(i != Aself && (!ARecv->isZero())) delete ARecv;
        
		if(!C_cont->isZero())
			tomerge.push_back(C_cont);
		else
			delete C_cont;
	}
	SpHelper::deallocate2D(ARecvSizes, UDERA::esscount);
	SpHelper::deallocate2D(BRecvSizes, UDERB::esscount);
	if(clearA) 
	{
		delete A2seq;
		delete A.spSeq;
		A.spSeq = NULL;
	}
	else
	{
		(A.spSeq)->Merge(*A1seq, *A2seq);
		delete A1seq;
		delete A2seq;
	}
	if(clearB) 
	{
		delete B2seq;
		delete B.spSeq;
		B.spSeq = NULL;	
	}
	else
	{
		B1seq->Transpose();
		B2seq->Transpose();
		(B.spSeq)->Merge(*B1seq, *B2seq);	
		delete B1seq;
		delete B2seq;
		const_cast< UDERB* >(B.spSeq)->Transpose();	// transpose back to original
	}

    SpTuples<LIC,NUO> * C_tuples = MultiwayMerge<SR>(tomerge, C_m, C_n,true); // Last parameter to delete input tuples
    UDERO * C = new UDERO(*C_tuples, false); // Last parameter to prevent transpose
    delete C_tuples;
	return SpParMat<IU,NUO,UDERO> (C, GridC);		// return the result object
}

/**
 * Parallel A = B*C routine that uses only MPI-1 features
 * Relies on simple blocking broadcast
 * @pre { Input matrices, A and B, should not alias }
 **/  
template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB> 
SpParMat<IU, NUO, UDERO> Mult_AnXBn_Synch 
		(SpParMat<IU,NU1,UDERA> & A, SpParMat<IU,NU2,UDERB> & B, bool clearA = false, bool clearB = false )

{
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    typedef typename UDERA::LocalIT LIA;
    typedef typename UDERB::LocalIT LIB;
    typedef typename UDERO::LocalIT LIC;
	if(!CheckSpGEMMCompliance(A,B) )
	{
		return SpParMat< IU,NUO,UDERO >();
	}
	int stages, dummy; 	// last two parameters of ProductGrid are ignored for Synch multiplication
	std::shared_ptr<CommGrid> GridC = ProductGrid((A.commGrid).get(), (B.commGrid).get(), stages, dummy, dummy);		
	LIA C_m = A.spSeq->getnrow();
	LIB C_n = B.spSeq->getncol();
	
	//const_cast< UDERB* >(B.spSeq)->Transpose(); // do not transpose for colum-by-column multiplication

    LIA ** ARecvSizes = SpHelper::allocate2D<LIA>(UDERA::esscount, stages);
    LIB ** BRecvSizes = SpHelper::allocate2D<LIB>(UDERB::esscount, stages);
	
	SpParHelper::GetSetSizes( *(A.spSeq), ARecvSizes, (A.commGrid)->GetRowWorld());
	SpParHelper::GetSetSizes( *(B.spSeq), BRecvSizes, (B.commGrid)->GetColWorld());

	// Remotely fetched matrices are stored as pointers
	UDERA * ARecv; 
	UDERB * BRecv;
	std::vector< SpTuples<LIC,NUO>  *> tomerge;

	int Aself = (A.commGrid)->GetRankInProcRow();
	int Bself = (B.commGrid)->GetRankInProcCol();	
	
	for(int i = 0; i < stages; ++i) 
	{
		std::vector<LIA> ess;	
		if(i == Aself)
		{	
			ARecv = A.spSeq;	// shallow-copy 
		}
		else
		{
			ess.resize(UDERA::esscount);
			for(int j=0; j< UDERA::esscount; ++j)	
			{
				ess[j] = ARecvSizes[j][i];		// essentials of the ith matrix in this row	
			}
			ARecv = new UDERA();				// first, create the object
		}
		SpParHelper::BCastMatrix(GridC->GetRowWorld(), *ARecv, ess, i);	// then, receive its elements	
		ess.clear();	
		
		if(i == Bself)
		{
			BRecv = B.spSeq;	// shallow-copy
		}
		else
		{
			ess.resize(UDERB::esscount);		
			for(int j=0; j< UDERB::esscount; ++j)	
			{
				ess[j] = BRecvSizes[j][i];	
			}	
			BRecv = new UDERB();
		}
		SpParHelper::BCastMatrix(GridC->GetColWorld(), *BRecv, ess, i);	// then, receive its elements
        
		SpTuples<LIC,NUO> * C_cont = LocalHybridSpGEMM<SR, NUO>
						(*ARecv, *BRecv, // parameters themselves
						false, 	// 'delete A' condition
						false);	// 'delete B' condition

        if(i != Bself && (!BRecv->isZero())) delete BRecv;
        if(i != Aself && (!ARecv->isZero())) delete ARecv;
		
		if(!C_cont->isZero()) 
			tomerge.push_back(C_cont);

#ifdef COMBBLAS_DEBUG
   		std::ostringstream outs;
		outs << i << "th SUMMA iteration"<< std::endl;
		SpParHelper::Print(outs.str());
#endif
	}

	if(clearA && A.spSeq != NULL) 
	{	
		delete A.spSeq;
		A.spSeq = NULL;
	}	
	if(clearB && B.spSeq != NULL) 
	{
		delete B.spSeq;
		B.spSeq = NULL;
	}

	SpHelper::deallocate2D(ARecvSizes, UDERA::esscount);
	SpHelper::deallocate2D(BRecvSizes, UDERB::esscount);

    SpTuples<LIC,NUO> * C_tuples = MultiwayMerge<SR>(tomerge, C_m, C_n,true); // Last parameter to delete input tuples
    UDERO * C = new UDERO(*C_tuples, false); // Last parameter to prevent transpose
    delete C_tuples;

	//if(!clearB)
	//	const_cast< UDERB* >(B.spSeq)->Transpose();	// transpose back to original

	return SpParMat<IU,NUO,UDERO> (C, GridC);		// return the result object
}

/*
 * Experimental SUMMA implementation with communication and computation overlap.
 * Not stable.
 * */
template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB>
SpParMat<IU, NUO, UDERO> Mult_AnXBn_Overlap
		(SpParMat<IU,NU1,UDERA> & A, SpParMat<IU,NU2,UDERB> & B, bool clearA = false, bool clearB = false )
{
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    typedef typename UDERA::LocalIT LIA;
    typedef typename UDERB::LocalIT LIB;
    typedef typename UDERO::LocalIT LIC;
	if(!CheckSpGEMMCompliance(A,B) )
	{
		return SpParMat< IU,NUO,UDERO >();
	}
	int stages, dummy; 	// last two parameters of ProductGrid are ignored for Synch multiplication
	std::shared_ptr<CommGrid> GridC = ProductGrid((A.commGrid).get(), (B.commGrid).get(), stages, dummy, dummy);
	IU C_m = A.spSeq->getnrow();
	IU C_n = B.spSeq->getncol();

	//const_cast< UDERB* >(B.spSeq)->Transpose(); // do not transpose for colum-by-column multiplication

    LIA ** ARecvSizes = SpHelper::allocate2D<LIA>(UDERA::esscount, stages);
    LIB ** BRecvSizes = SpHelper::allocate2D<LIB>(UDERB::esscount, stages);

	SpParHelper::GetSetSizes( *(A.spSeq), ARecvSizes, (A.commGrid)->GetRowWorld());
	SpParHelper::GetSetSizes( *(B.spSeq), BRecvSizes, (B.commGrid)->GetColWorld());

	// Remotely fetched matrices are stored as pointers
	UDERA ** ARecv = new UDERA* [stages]; 
	UDERB ** BRecv = new UDERB* [stages];

	Arr<IU,NU1> Aarrinfo = A.seqptr()->GetArrays();
	Arr<IU,NU2> Barrinfo = B.seqptr()->GetArrays();
    std::vector< std::vector<MPI_Request> > ABCastIndarrayReq;
    std::vector< std::vector<MPI_Request> > ABCastNumarrayReq;
    std::vector< std::vector<MPI_Request> > BBCastIndarrayReq;
    std::vector< std::vector<MPI_Request> > BBCastNumarrayReq;
    for(int i = 0; i < stages; i++){
        ABCastIndarrayReq.push_back( std::vector<MPI_Request>(Aarrinfo.indarrs.size(), MPI_REQUEST_NULL) );
        ABCastNumarrayReq.push_back( std::vector<MPI_Request>(Aarrinfo.numarrs.size(), MPI_REQUEST_NULL) );
        BBCastIndarrayReq.push_back( std::vector<MPI_Request>(Barrinfo.indarrs.size(), MPI_REQUEST_NULL) );
        BBCastNumarrayReq.push_back( std::vector<MPI_Request>(Barrinfo.numarrs.size(), MPI_REQUEST_NULL) );
    }

	int Aself = (A.commGrid)->GetRankInProcRow();
	int Bself = (B.commGrid)->GetRankInProcCol();

	std::vector< SpTuples<IU,NUO> *> tomerge;

	for(int i = 0; i < stages; ++i){
		std::vector<IU> ess;
		if(i == Aself) ARecv[i] = A.spSeq;	// shallow-copy 
		else{
			ess.resize(UDERA::esscount);
			for(int j=0; j< UDERA::esscount; ++j) ess[j] = ARecvSizes[j][i];		// essentials of the ith matrix in this row	
			ARecv[i] = new UDERA();				// first, create the object
	    }	
		SpParHelper::IBCastMatrix(GridC->GetRowWorld(), *(ARecv[i]), ess, i, ABCastIndarrayReq[i], ABCastNumarrayReq[i]);	// then, receive its elements	

		ess.clear();	
		
		if(i == Bself) BRecv[i] = B.spSeq;	// shallow-copy
		else{
			ess.resize(UDERB::esscount);		
			for(int j=0; j< UDERB::esscount; ++j) ess[j] = BRecvSizes[j][i];
			BRecv[i] = new UDERB();
		}
		SpParHelper::IBCastMatrix(GridC->GetColWorld(), *(BRecv[i]), ess, i, BBCastIndarrayReq[i], BBCastNumarrayReq[i]);	// then, receive its elements

		if(i > 0){
            MPI_Waitall(ABCastIndarrayReq[i-1].size(), ABCastIndarrayReq[i-1].data(), MPI_STATUSES_IGNORE);
            MPI_Waitall(ABCastNumarrayReq[i-1].size(), ABCastNumarrayReq[i-1].data(), MPI_STATUSES_IGNORE);
            MPI_Waitall(BBCastIndarrayReq[i-1].size(), BBCastIndarrayReq[i-1].data(), MPI_STATUSES_IGNORE);
            MPI_Waitall(BBCastNumarrayReq[i-1].size(), BBCastNumarrayReq[i-1].data(), MPI_STATUSES_IGNORE);

            SpTuples<IU,NUO> * C_cont = LocalHybridSpGEMM<SR, NUO>
                            (*(ARecv[i-1]), *(BRecv[i-1]), // parameters themselves
                            i-1 != Aself, 	// 'delete A' condition
                            i-1 != Bself);	// 'delete B' condition
            if(!C_cont->isZero()) tomerge.push_back(C_cont);

            SpTuples<IU,NUO> * C_tuples = MultiwayMerge<SR>(tomerge, C_m, C_n,true);
            std::vector< SpTuples<IU,NUO> *>().swap(tomerge);
            tomerge.push_back(C_tuples);
        }
        #ifdef COMBBLAS_DEBUG
        std::ostringstream outs;
        outs << i << "th SUMMA iteration"<< std::endl;
        SpParHelper::Print(outs.str());
        #endif
	}

    MPI_Waitall(ABCastIndarrayReq[stages-1].size(), ABCastIndarrayReq[stages-1].data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(ABCastNumarrayReq[stages-1].size(), ABCastNumarrayReq[stages-1].data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(BBCastIndarrayReq[stages-1].size(), BBCastIndarrayReq[stages-1].data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(BBCastNumarrayReq[stages-1].size(), BBCastNumarrayReq[stages-1].data(), MPI_STATUSES_IGNORE);

    SpTuples<IU,NUO> * C_cont = LocalHybridSpGEMM<SR, NUO>
                    (*(ARecv[stages-1]), *(BRecv[stages-1]), // parameters themselves
                    stages-1 != Aself, 	// 'delete A' condition
                    stages-1 != Bself);	// 'delete B' condition
    if(!C_cont->isZero()) tomerge.push_back(C_cont);

	if(clearA && A.spSeq != NULL) {	
		delete A.spSeq;
		A.spSeq = NULL;
	}	
	if(clearB && B.spSeq != NULL) {
		delete B.spSeq;
		B.spSeq = NULL;
	}

    delete[] ARecv;
    delete[] BRecv;

	SpHelper::deallocate2D(ARecvSizes, UDERA::esscount);
	SpHelper::deallocate2D(BRecvSizes, UDERB::esscount);

	// the last parameter to MergeAll deletes tomerge arrays
	SpTuples<IU,NUO> * C_tuples = MultiwayMerge<SR>(tomerge, C_m, C_n,true);
    std::vector< SpTuples<IU,NUO> *>().swap(tomerge);
	
    UDERO * C = new UDERO(*C_tuples, false);
    delete C_tuples;

	//if(!clearB)
	//	const_cast< UDERB* >(B.spSeq)->Transpose();	// transpose back to original

	return SpParMat<IU,NUO,UDERO> (C, GridC);		// return the result object
}

    
/**
  * Estimate the maximum nnz needed to store in a process from all stages of SUMMA before reduction
  * @pre { Input matrices, A and B, should not alias }
  **/
template <typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB>
int64_t EstPerProcessNnzSUMMA(SpParMat<IU,NU1,UDERA> & A, SpParMat<IU,NU2,UDERB> & B, bool hashEstimate)  
{
    	typedef typename UDERA::LocalIT LIA;
    	typedef typename UDERB::LocalIT LIB;
        static_assert(std::is_same<LIA, LIB>::value, "local index types for both input matrices should be the same");

        double t0, t1;

        int64_t nnzC_SUMMA = 0;
        
        if(A.getncol() != B.getnrow())
        {
            std::ostringstream outs;
            outs << "Can not multiply, dimensions does not match"<< std::endl;
            outs << A.getncol() << " != " << B.getnrow() << std::endl;
            SpParHelper::Print(outs.str());
            MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
            return nnzC_SUMMA;
        }
       
        int stages, dummy;     // last two parameters of ProductGrid are ignored for Synch multiplication
        std::shared_ptr<CommGrid> GridC = ProductGrid((A.commGrid).get(), (B.commGrid).get(), stages, dummy, dummy);
  
        MPI_Barrier(GridC->GetWorld());
        
        LIA ** ARecvSizes = SpHelper::allocate2D<LIA>(UDERA::esscount, stages);
        LIB ** BRecvSizes = SpHelper::allocate2D<LIB>(UDERB::esscount, stages);
        SpParHelper::GetSetSizes( *(A.spSeq), ARecvSizes, (A.commGrid)->GetRowWorld());
        SpParHelper::GetSetSizes( *(B.spSeq), BRecvSizes, (B.commGrid)->GetColWorld());
        
        // Remotely fetched matrices are stored as pointers
        UDERA * ARecv;
        UDERB * BRecv;

        int Aself = (A.commGrid)->GetRankInProcRow();
        int Bself = (B.commGrid)->GetRankInProcCol();
        
        
        for(int i = 0; i < stages; ++i)
        {
            std::vector<LIA> ess;
            if(i == Aself)
            {
                ARecv = A.spSeq;    // shallow-copy
            }
            else
            {
                ess.resize(UDERA::esscount);
                for(int j=0; j< UDERA::esscount; ++j)
                {
                    ess[j] = ARecvSizes[j][i];        // essentials of the ith matrix in this row
                }
                ARecv = new UDERA();                // first, create the object
            }

            SpParHelper::BCastMatrix(GridC->GetRowWorld(), *ARecv, ess, i);    // then, receive its elements
            ess.clear();
            
            if(i == Bself)
            {
                BRecv = B.spSeq;    // shallow-copy
            }
            else
            {
                ess.resize(UDERB::esscount);
                for(int j=0; j< UDERB::esscount; ++j)
                {
                    ess[j] = BRecvSizes[j][i];
                }
                BRecv = new UDERB();
            }

            SpParHelper::BCastMatrix(GridC->GetColWorld(), *BRecv, ess, i);    // then, receive its elements
            
    	    // no need to keep entries of colnnzC in larger precision 
	        // because colnnzC is of length nzc and estimates nnzs per column
			// @OGUZ-EDIT Using hash spgemm for estimation
            //LIB * colnnzC = estimateNNZ(*ARecv, *BRecv);
			LIB* flopC = estimateFLOP(*ARecv, *BRecv);
			LIB* colnnzC = estimateNNZ_Hash(*ARecv, *BRecv, flopC);
            LIB nzc = BRecv->GetDCSC()->nzc;

            if (flopC) delete [] flopC;
            if(colnnzC) delete [] colnnzC;

			// sampling-based estimation (comment the estimation above, and
			// comment out below to use)			
			// int64_t nnzC_stage = estimateNNZ_sampling(*ARecv, *BRecv);
			// nnzC_SUMMA += nnzC_stage;
            
            // delete received data
            if(i != Aself)
                delete ARecv;
            if(i != Bself)
                delete BRecv;
        }
        
        SpHelper::deallocate2D(ARecvSizes, UDERA::esscount);
        SpHelper::deallocate2D(BRecvSizes, UDERB::esscount);
        
        int64_t nnzC_SUMMA_max = 0;
        MPI_Allreduce(&nnzC_SUMMA, &nnzC_SUMMA_max, 1, MPIType<int64_t>(), MPI_MAX, GridC->GetWorld());
        
        return nnzC_SUMMA_max;
}
    
    
template <typename MATRIX, typename VECTOR>
void CheckSpMVCompliance(const MATRIX & A, const VECTOR & x)
{
	if(A.getncol() != x.TotalLength())
	{
		std::ostringstream outs;
		outs << "Can not multiply, dimensions does not match"<< std::endl;
		outs << A.getncol() << " != " << x.TotalLength() << std::endl;
		SpParHelper::Print(outs.str());
		MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
	}
	if(! ( *(A.getcommgrid()) == *(x.getcommgrid())) ) 		
	{
		std::cout << "Grids are not comparable for SpMV" << std::endl; 
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
	}
}			


template <typename SR, typename IU, typename NUM, typename UDER> 
FullyDistSpVec<IU,typename promote_trait<NUM,IU>::T_promote>  SpMV 
	(const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,IU> & x, bool indexisvalue, OptBuf<int32_t, typename promote_trait<NUM,IU>::T_promote > & optbuf);

template <typename SR, typename IU, typename NUM, typename UDER> 
FullyDistSpVec<IU,typename promote_trait<NUM,IU>::T_promote>  SpMV 
	(const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,IU> & x, bool indexisvalue)
{
	typedef typename promote_trait<NUM,IU>::T_promote T_promote;
	OptBuf<int32_t, T_promote > optbuf = OptBuf<int32_t, T_promote >();
	return SpMV<SR>(A, x, indexisvalue, optbuf);
}

/**
 * Step 1 of the sparse SpMV algorithm 
 * @param[in,out]   trxlocnz, lenuntil,trxinds,trxnums  { set or allocated }
 * @param[in] 	indexisvalue	
 **/
template<typename IU, typename NV>
void TransposeVector(MPI_Comm & World, const FullyDistSpVec<IU,NV> & x, int32_t & trxlocnz, IU & lenuntil, int32_t * & trxinds, NV * & trxnums, bool indexisvalue)
{
	int32_t xlocnz = (int32_t) x.getlocnnz();	
	int32_t roffst = (int32_t) x.RowLenUntil();	// since trxinds is int32_t
	int32_t roffset;
	IU luntil = x.LengthUntil();
	int diagneigh = x.commGrid->GetComplementRank();

	MPI_Status status;
	MPI_Sendrecv(&roffst, 1, MPIType<int32_t>(), diagneigh, TROST, &roffset, 1, MPIType<int32_t>(), diagneigh, TROST, World, &status);
	MPI_Sendrecv(&xlocnz, 1, MPIType<int32_t>(), diagneigh, TRNNZ, &trxlocnz, 1, MPIType<int32_t>(), diagneigh, TRNNZ, World, &status);
	MPI_Sendrecv(&luntil, 1, MPIType<IU>(), diagneigh, TRLUT, &lenuntil, 1, MPIType<IU>(), diagneigh, TRLUT, World, &status);
	
	// ABAB: Important observation is that local indices (given by x.ind) is 32-bit addressible
	// Copy them to 32 bit integers and transfer that to save 50% of off-node bandwidth
	trxinds = new int32_t[trxlocnz];
	int32_t * temp_xind = new int32_t[xlocnz];
#ifdef THREADED
#pragma omp parallel for
#endif
	for(int i=0; i< xlocnz; ++i)
        temp_xind[i] = (int32_t) x.ind[i];
	MPI_Sendrecv(temp_xind, xlocnz, MPIType<int32_t>(), diagneigh, TRI, trxinds, trxlocnz, MPIType<int32_t>(), diagneigh, TRI, World, &status);
	delete [] temp_xind;
	if(!indexisvalue)
	{
		trxnums = new NV[trxlocnz];
		MPI_Sendrecv(const_cast<NV*>(SpHelper::p2a(x.num)), xlocnz, MPIType<NV>(), diagneigh, TRX, trxnums, trxlocnz, MPIType<NV>(), diagneigh, TRX, World, &status);
	}
	// fullydist indexing (p pieces) -> matrix indexing (sqrt(p) pieces)
	std::transform(trxinds, trxinds+trxlocnz, trxinds, [roffset](int32_t  val){return val + roffset;});
}


/**
 * Step 2 of the sparse SpMV algorithm 
 * @param[in,out]   trxinds, trxnums { deallocated }
 * @param[in,out]   indacc, numacc { allocated }
 * @param[in,out]	accnz { set }
 * @param[in] 		trxlocnz, lenuntil, indexisvalue
 **/
template<typename IU, typename NV>
void AllGatherVector(MPI_Comm & ColWorld, int trxlocnz, IU lenuntil, int32_t * & trxinds, NV * & trxnums, 
					 int32_t * & indacc, NV * & numacc, int & accnz, bool indexisvalue)
{
    int colneighs, colrank;
	MPI_Comm_size(ColWorld, &colneighs);
	MPI_Comm_rank(ColWorld, &colrank);
	int * colnz = new int[colneighs];
	colnz[colrank] = trxlocnz;
	MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, colnz, 1, MPI_INT, ColWorld);
	int * dpls = new int[colneighs]();	// displacements (zero initialized pid) 
	std::partial_sum(colnz, colnz+colneighs-1, dpls+1);
	accnz = std::accumulate(colnz, colnz+colneighs, 0);
	indacc = new int32_t[accnz];
	numacc = new NV[accnz];
	
	// ABAB: Future issues here, colnz is of type int (MPI limitation)
	// What if the aggregate vector size along the processor row/column is not 32-bit addressible?
	// This will happen when n/sqrt(p) > 2^31
	// Currently we can solve a small problem (scale 32) with 4096 processor
	// For a medium problem (scale 35), we'll need 32K processors which gives sqrt(p) ~ 180
	// 2^35 / 180 ~ 2^29 / 3 which is not an issue !
	
#ifdef TIMING
	double t0=MPI_Wtime();
#endif
	MPI_Allgatherv(trxinds, trxlocnz, MPIType<int32_t>(), indacc, colnz, dpls, MPIType<int32_t>(), ColWorld);
	
	delete [] trxinds;
	if(indexisvalue)
	{
		IU lenuntilcol;
		if(colrank == 0)  lenuntilcol = lenuntil;
		MPI_Bcast(&lenuntilcol, 1, MPIType<IU>(), 0, ColWorld);
		for(int i=0; i< accnz; ++i)	// fill numerical values from indices
		{
			numacc[i] = indacc[i] + lenuntilcol;
		}
	}
	else
	{
		MPI_Allgatherv(trxnums, trxlocnz, MPIType<NV>(), numacc, colnz, dpls, MPIType<NV>(), ColWorld);
		delete [] trxnums;
	}	
#ifdef TIMING
	double t1=MPI_Wtime();
	cblas_allgathertime += (t1-t0);
#endif
	DeleteAll(colnz,dpls);
}	



/**
  * Step 3 of the sparse SpMV algorithm, with the semiring 
  * @param[in,out] optbuf {scratch space for all-to-all (fold) communication}
  * @param[in,out] indacc, numacc {index and values of the input vector, deleted upon exit}
  * @param[in,out] sendindbuf, sendnumbuf {index and values of the output vector, created}
 **/
template<typename SR, typename IVT, typename OVT, typename IU, typename NUM, typename UDER>
void LocalSpMV(const SpParMat<IU,NUM,UDER> & A, int rowneighs, OptBuf<int32_t, OVT > & optbuf, int32_t * & indacc, IVT * & numacc, 
			   int32_t * & sendindbuf, OVT * & sendnumbuf, int * & sdispls, int * sendcnt, int accnz, bool indexisvalue, PreAllocatedSPA<OVT> & SPA)
{
    if(optbuf.totmax > 0)	// graph500 optimization enabled
	{ 
		if(A.spSeq->getnsplit() > 0)
		{
			// optbuf.{inds/nums/dspls} and sendcnt are all pre-allocated and only filled by dcsc_gespmv_threaded
			generic_gespmv_threaded_setbuffers<SR> (*(A.spSeq), indacc, numacc, accnz, optbuf.inds, optbuf.nums, sendcnt, optbuf.dspls, rowneighs);
		}
		else
		{
			generic_gespmv<SR> (*(A.spSeq), indacc, numacc, accnz, optbuf.inds, optbuf.nums, sendcnt, optbuf.dspls, rowneighs, indexisvalue);
		}
		DeleteAll(indacc,numacc);
	}
	else
	{
		if(A.spSeq->getnsplit() > 0)
		{
			// sendindbuf/sendnumbuf/sdispls are all allocated and filled by dcsc_gespmv_threaded
			int totalsent = generic_gespmv_threaded<SR> (*(A.spSeq), indacc, numacc, accnz, sendindbuf, sendnumbuf, sdispls, rowneighs, SPA);
			
			DeleteAll(indacc, numacc);
			for(int i=0; i<rowneighs-1; ++i)
				sendcnt[i] = sdispls[i+1] - sdispls[i];
			sendcnt[rowneighs-1] = totalsent - sdispls[rowneighs-1];
		}
		else
		{
            // default SpMSpV
            std::vector< int32_t > indy;
            std::vector< OVT >  numy;
            generic_gespmv<SR>(*(A.spSeq), indacc, numacc, accnz, indy, numy, SPA);	
            
            DeleteAll(indacc, numacc);
            
            int32_t bufsize = indy.size();	// as compact as possible
            sendindbuf = new int32_t[bufsize];
            sendnumbuf = new OVT[bufsize];
            int32_t perproc = A.getlocalrows() / rowneighs;
            
            int k = 0;	// index to buffer
            for(int i=0; i<rowneighs; ++i)
            {
                int32_t end_this = (i==rowneighs-1) ? A.getlocalrows(): (i+1)*perproc;
                while(k < bufsize && indy[k] < end_this)
                {
                    sendindbuf[k] = indy[k] - i*perproc;
                    sendnumbuf[k] = numy[k];
                    ++sendcnt[i];
                    ++k;
                }
            }
            sdispls = new int[rowneighs]();
            std::partial_sum(sendcnt, sendcnt+rowneighs-1, sdispls+1);
            
//#endif

		}
	}

}



// non threaded
template <typename SR, typename IU, typename OVT>
void MergeContributions(int*  listSizes, std::vector<int32_t *> & indsvec, std::vector<OVT *> & numsvec, std::vector<IU>& mergedind, std::vector<OVT>& mergednum)
{
    
    int nlists = indsvec.size();
    // this condition is checked in the caller SpMV function.
    // I am still putting it here for completeness
    if(nlists == 1)
    {
        // simply copy data
        int veclen = listSizes[0];
        mergedind.resize(veclen);
        mergednum.resize(veclen);
        for(int i=0; i<veclen; i++)
        {
            mergedind[i] = indsvec[0][i];
            mergednum[i] = numsvec[0][i];
        }
        return;
    }

    int32_t hsize = 0;
    int32_t inf = std::numeric_limits<int32_t>::min();
    int32_t sup = std::numeric_limits<int32_t>::max();
    KNHeap< int32_t, int32_t > sHeap(sup, inf);
    int * processed = new int[nlists]();
    for(int i=0; i<nlists; ++i)
    {
        if(listSizes[i] > 0)
        {
            // key, list_id
            sHeap.insert(indsvec[i][0], i);
            ++hsize;
        }
    }
    int32_t key, locv;
    if(hsize > 0)
    {
        sHeap.deleteMin(&key, &locv);
        mergedind.push_back( static_cast<IU>(key));
        mergednum.push_back(numsvec[locv][0]);	// nothing is processed yet
        
        if( (++(processed[locv])) < listSizes[locv] )
            sHeap.insert(indsvec[locv][processed[locv]], locv);
        else
            --hsize;
    }
    while(hsize > 0)
    {
        sHeap.deleteMin(&key, &locv);
        if(mergedind.back() == static_cast<IU>(key))
        {
            mergednum.back() = SR::add(mergednum.back(), numsvec[locv][processed[locv]]);
            // ABAB: Benchmark actually allows us to be non-deterministic in terms of parent selection
            // We can just skip this addition operator (if it's a max/min select)
        }
        else
        {
            mergedind.push_back(static_cast<IU>(key));
            mergednum.push_back(numsvec[locv][processed[locv]]);
        }
        
        if( (++(processed[locv])) < listSizes[locv] )
            sHeap.insert(indsvec[locv][processed[locv]], locv);
        else
            --hsize;
    }
    DeleteAll(processed);
}



template <typename SR, typename IU, typename OVT>
void MergeContributions_threaded(int * & listSizes, std::vector<int32_t *> & indsvec, std::vector<OVT *> & numsvec, std::vector<IU> & mergedind, std::vector<OVT> & mergednum, IU maxindex)
{
    
    int nlists = indsvec.size();
    // this condition is checked in the caller SpMV function.
    // I am still putting it here for completeness
    if(nlists == 1)
    {
        // simply copy data
        int veclen = listSizes[0];
        mergedind.resize(veclen);
        mergednum.resize(veclen);
        
#ifdef THREADED
#pragma omp parallel for
#endif
        for(int i=0; i<veclen; i++)
        {
            mergedind[i] = indsvec[0][i];
            mergednum[i] = numsvec[0][i];
        }
        return;
    }
    
    int nthreads=1;
#ifdef THREADED
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
#endif
    int nsplits = 4*nthreads; // oversplit for load balance
    nsplits = std::min(nsplits, (int)maxindex);
    std::vector< std::vector<int32_t> > splitters(nlists);
    for(int k=0; k< nlists; k++)
    {
        splitters[k].resize(nsplits+1);
        splitters[k][0] = static_cast<int32_t>(0);
#pragma omp parallel for
        for(int i=1; i< nsplits; i++)
        {
            IU cur_idx = i * (maxindex/nsplits);
            auto it = std::lower_bound (indsvec[k], indsvec[k] + listSizes[k], cur_idx);
            splitters[k][i] = (int32_t) (it - indsvec[k]);
        }
        splitters[k][nsplits] = listSizes[k];
    }
    
    // ------ perform merge in parallel ------
    std::vector<std::vector<IU>> indsBuf(nsplits);
    std::vector<std::vector<OVT>> numsBuf(nsplits);
    //TODO: allocate these vectors here before calling MergeContributions
#pragma omp parallel for schedule(dynamic)
    for(int i=0; i< nsplits; i++)
    {
        std::vector<int32_t *> tIndsVec(nlists);
        std::vector<OVT *> tNumsVec(nlists);
        std::vector<int> tLengths(nlists);
        for(int j=0; j< nlists; ++j)
        {
            tIndsVec[j] = indsvec[j] + splitters[j][i];
            tNumsVec[j] = numsvec[j] + splitters[j][i];
            tLengths[j]= splitters[j][i+1] - splitters[j][i];
            
        }
        MergeContributions<SR>(tLengths.data(), tIndsVec, tNumsVec, indsBuf[i], numsBuf[i]);
    }

    // ------ concatenate merged tuples processed by threads ------
    std::vector<IU> tdisp(nsplits+1);
    tdisp[0] = 0;
    for(int i=0; i<nsplits; ++i)
    {
        tdisp[i+1] = tdisp[i] + indsBuf[i].size();
    }
    
    mergedind.resize(tdisp[nsplits]);
    mergednum.resize(tdisp[nsplits]);
    
    
#pragma omp parallel for schedule(dynamic)
    for(int i=0; i< nsplits; i++)
    {
        std::copy(indsBuf[i].data() , indsBuf[i].data() + indsBuf[i].size(), mergedind.data() + tdisp[i]);
        std::copy(numsBuf[i].data() , numsBuf[i].data() + numsBuf[i].size(), mergednum.data() + tdisp[i]);
    }
}


/** 
  * This version is the most flexible sparse matrix X sparse vector [Used in KDT]
  * It accepts different types for the matrix (NUM), the input vector (IVT) and the output vector (OVT)
  * without relying on automatic type promotion
  * Input (x) and output (y) vectors can be ALIASED because y is not written until the algorithm is done with x.
  */
template <typename SR, typename IVT, typename OVT, typename IU, typename NUM, typename UDER>
void SpMV (const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,IVT> & x, FullyDistSpVec<IU,OVT> & y, 
			bool indexisvalue, OptBuf<int32_t, OVT > & optbuf, PreAllocatedSPA<OVT> & SPA)
{
	CheckSpMVCompliance(A,x);
	optbuf.MarkEmpty();
    y.glen = A.getnrow(); // in case it is not set already
	
	MPI_Comm World = x.commGrid->GetWorld();
	MPI_Comm ColWorld = x.commGrid->GetColWorld();
	MPI_Comm RowWorld = x.commGrid->GetRowWorld();
	
	int accnz;
	int32_t trxlocnz;
	IU lenuntil;
	int32_t *trxinds, *indacc;
	IVT *trxnums, *numacc;
	
#ifdef TIMING
    double t0=MPI_Wtime();
#endif
    
	TransposeVector(World, x, trxlocnz, lenuntil, trxinds, trxnums, indexisvalue);
    
#ifdef TIMING
    double t1=MPI_Wtime();
    cblas_transvectime += (t1-t0);
#endif
    
    if(x.commGrid->GetGridRows() > 1)
    {
        AllGatherVector(ColWorld, trxlocnz, lenuntil, trxinds, trxnums, indacc, numacc, accnz, indexisvalue);   // trxindS/trxnums deallocated, indacc/numacc allocated, accnz set
    }
    else
    {
        accnz = trxlocnz;
        indacc = trxinds;   // aliasing ptr
        numacc = trxnums;   // aliasing ptr
    }
	
	int rowneighs;
	MPI_Comm_size(RowWorld, &rowneighs);
	int * sendcnt = new int[rowneighs]();	
	int32_t * sendindbuf;	
	OVT * sendnumbuf;
	int * sdispls;
    
#ifdef TIMING
    double t2=MPI_Wtime();
#endif
    
	LocalSpMV<SR>(A, rowneighs, optbuf, indacc, numacc, sendindbuf, sendnumbuf, sdispls, sendcnt, accnz, indexisvalue, SPA);	// indacc/numacc deallocated, sendindbuf/sendnumbuf/sdispls allocated

#ifdef TIMING
    double t3=MPI_Wtime();
    cblas_localspmvtime += (t3-t2);
#endif
	

    if(x.commGrid->GetGridCols() == 1)
    {
        y.ind.resize(sendcnt[0]);
        y.num.resize(sendcnt[0]);


		if(optbuf.totmax > 0 )	// graph500 optimization enabled
		{
#ifdef THREADED
#pragma omp parallel for
#endif
			for(int i=0; i<sendcnt[0]; i++)
			{
				y.ind[i] = optbuf.inds[i];
				y.num[i] = optbuf.nums[i];
			}
		}
		else
		{
#ifdef THREADED
#pragma omp parallel for
#endif
			for(int i=0; i<sendcnt[0]; i++)
			{
				y.ind[i] = sendindbuf[i];
				y.num[i] = sendnumbuf[i];
			}
			DeleteAll(sendindbuf, sendnumbuf,sdispls);
		}
		delete [] sendcnt;
        return;
    }
	int * rdispls = new int[rowneighs];
	int * recvcnt = new int[rowneighs];
	MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, RowWorld);       // share the request counts
	
	// receive displacements are exact whereas send displacements have slack
	rdispls[0] = 0;
	for(int i=0; i<rowneighs-1; ++i)
	{
		rdispls[i+1] = rdispls[i] + recvcnt[i];
	}
	
	int totrecv = std::accumulate(recvcnt,recvcnt+rowneighs,0);	
	int32_t * recvindbuf = new int32_t[totrecv];
	OVT * recvnumbuf = new OVT[totrecv];
	
#ifdef TIMING
	double t4=MPI_Wtime();
#endif
	if(optbuf.totmax > 0 )	// graph500 optimization enabled
	{
		MPI_Alltoallv(optbuf.inds, sendcnt, optbuf.dspls, MPIType<int32_t>(), recvindbuf, recvcnt, rdispls, MPIType<int32_t>(), RowWorld);
		MPI_Alltoallv(optbuf.nums, sendcnt, optbuf.dspls, MPIType<OVT>(), recvnumbuf, recvcnt, rdispls, MPIType<OVT>(), RowWorld);
		delete [] sendcnt;
	}
	else
    {
		MPI_Alltoallv(sendindbuf, sendcnt, sdispls, MPIType<int32_t>(), recvindbuf, recvcnt, rdispls, MPIType<int32_t>(), RowWorld);
		MPI_Alltoallv(sendnumbuf, sendcnt, sdispls, MPIType<OVT>(), recvnumbuf, recvcnt, rdispls, MPIType<OVT>(), RowWorld);
		DeleteAll(sendindbuf, sendnumbuf, sendcnt, sdispls);
	}
#ifdef TIMING
	double t5=MPI_Wtime();
	cblas_alltoalltime += (t5-t4);
#endif
	
#ifdef TIMING
    double t6=MPI_Wtime();
#endif
    //MergeContributions<SR>(y,recvcnt, rdispls, recvindbuf, recvnumbuf, rowneighs);
    // free memory of y, in case it was aliased
    std::vector<IU>().swap(y.ind);
    std::vector<OVT>().swap(y.num);
    
    std::vector<int32_t *> indsvec(rowneighs);
    std::vector<OVT *> numsvec(rowneighs);
    
#ifdef THREADED
#pragma omp parallel for
#endif
    for(int i=0; i<rowneighs; i++)
    {
        indsvec[i] = recvindbuf+rdispls[i];
        numsvec[i] = recvnumbuf+rdispls[i];
    }
#ifdef THREADED
    MergeContributions_threaded<SR>(recvcnt, indsvec, numsvec, y.ind, y.num, y.MyLocLength());
#else
    MergeContributions<SR>(recvcnt, indsvec, numsvec, y.ind, y.num);
#endif
    
    DeleteAll(recvcnt, rdispls,recvindbuf, recvnumbuf);
#ifdef TIMING
    double t7=MPI_Wtime();
    cblas_mergeconttime += (t7-t6);
#endif
    
}


template <typename SR, typename IVT, typename OVT, typename IU, typename NUM, typename UDER>
void SpMV (const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,IVT> & x, FullyDistSpVec<IU,OVT> & y, bool indexisvalue, PreAllocatedSPA<OVT> & SPA)
{
	OptBuf< int32_t, OVT > optbuf = OptBuf< int32_t,OVT >(); 
	SpMV<SR>(A, x, y, indexisvalue, optbuf, SPA);
}

template <typename SR, typename IVT, typename OVT, typename IU, typename NUM, typename UDER>
void SpMV (const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,IVT> & x, FullyDistSpVec<IU,OVT> & y, bool indexisvalue)
{
    OptBuf< int32_t, OVT > optbuf = OptBuf< int32_t,OVT >();
    PreAllocatedSPA<OVT> SPA;
    SpMV<SR>(A, x, y, indexisvalue, optbuf, SPA);
}

template <typename SR, typename IVT, typename OVT, typename IU, typename NUM, typename UDER>
void SpMV (const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,IVT> & x, FullyDistSpVec<IU,OVT> & y, bool indexisvalue, OptBuf<int32_t, OVT > & optbuf)
{
	PreAllocatedSPA<OVT> SPA;
	SpMV<SR>(A, x, y, indexisvalue, optbuf, SPA);
}


/**
 * Automatic type promotion is ONLY done here, all the callee functions (in Friends.h and below) are initialized with the promoted type
 * If indexisvalues = true, then we do not need to transfer values for x (happens for BFS iterations with boolean matrices and integer rhs vectors)
 **/
template <typename SR, typename IU, typename NUM, typename UDER>
FullyDistSpVec<IU,typename promote_trait<NUM,IU>::T_promote>  SpMV 
(const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,IU> & x, bool indexisvalue, OptBuf<int32_t, typename promote_trait<NUM,IU>::T_promote > & optbuf)
{		
	typedef typename promote_trait<NUM,IU>::T_promote T_promote;
	FullyDistSpVec<IU, T_promote> y ( x.getcommgrid(), A.getnrow());	// identity doesn't matter for sparse vectors
	SpMV<SR>(A, x, y, indexisvalue, optbuf);
	return y;
}

/**
 * Parallel dense SpMV
 **/ 
template <typename SR, typename IU, typename NUM, typename NUV, typename UDER> 
FullyDistVec<IU,typename promote_trait<NUM,NUV>::T_promote>  SpMV 
	(const SpParMat<IU,NUM,UDER> & A, const FullyDistVec<IU,NUV> & x )
{
	typedef typename promote_trait<NUM,NUV>::T_promote T_promote;
	CheckSpMVCompliance(A, x);

	MPI_Comm World = x.commGrid->GetWorld();
	MPI_Comm ColWorld = x.commGrid->GetColWorld();
	MPI_Comm RowWorld = x.commGrid->GetRowWorld();

	int xsize = (int) x.LocArrSize();
	int trxsize = 0;

	int diagneigh = x.commGrid->GetComplementRank();
	MPI_Status status;
	MPI_Sendrecv(&xsize, 1, MPI_INT, diagneigh, TRX, &trxsize, 1, MPI_INT, diagneigh, TRX, World, &status);
	
	NUV * trxnums = new NUV[trxsize];
	MPI_Sendrecv(const_cast<NUV*>(SpHelper::p2a(x.arr)), xsize, MPIType<NUV>(), diagneigh, TRX, trxnums, trxsize, MPIType<NUV>(), diagneigh, TRX, World, &status);

        int colneighs, colrank;
	MPI_Comm_size(ColWorld, &colneighs);
	MPI_Comm_rank(ColWorld, &colrank);
	int * colsize = new int[colneighs];
	colsize[colrank] = trxsize;
	MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, colsize, 1, MPI_INT, ColWorld);
	int * dpls = new int[colneighs]();	// displacements (zero initialized pid) 
	std::partial_sum(colsize, colsize+colneighs-1, dpls+1);
	int accsize = std::accumulate(colsize, colsize+colneighs, 0);
	NUV * numacc = new NUV[accsize];

	MPI_Allgatherv(trxnums, trxsize, MPIType<NUV>(), numacc, colsize, dpls, MPIType<NUV>(), ColWorld);
	delete [] trxnums;

	// serial SpMV with dense vector
	T_promote id = SR::id();
	IU ysize = A.getlocalrows();
	T_promote * localy = new T_promote[ysize];
	std::fill_n(localy, ysize, id);		

#ifdef THREADED
	dcsc_gespmv_threaded<SR>(*(A.spSeq), numacc, localy);
#else
	dcsc_gespmv<SR>(*(A.spSeq), numacc, localy);	
#endif
	

	DeleteAll(numacc,colsize, dpls);

	// FullyDistVec<IT,NT>(shared_ptr<CommGrid> grid, IT globallen, NT initval, NT id)
	FullyDistVec<IU, T_promote> y ( x.commGrid, A.getnrow(), id);
	
	int rowneighs;
	MPI_Comm_size(RowWorld, &rowneighs);

	IU begptr, endptr;
	for(int i=0; i< rowneighs; ++i)
	{
		begptr = y.RowLenUntil(i);
		if(i == rowneighs-1)
		{
			endptr = ysize;
		}
		else
		{
			endptr = y.RowLenUntil(i+1);
		}
		MPI_Reduce(localy+begptr, SpHelper::p2a(y.arr), endptr-begptr, MPIType<T_promote>(), SR::mpi_op(), i, RowWorld);
	}
	delete [] localy;
	return y;
}

	
/**
 * \TODO: Old version that is no longer considered optimal
 * Kept for legacy purposes
 * To be removed when other functionals are fully tested.
 **/ 
template <typename SR, typename IU, typename NUM, typename NUV, typename UDER> 
FullyDistSpVec<IU,typename promote_trait<NUM,NUV>::T_promote>  SpMV 
	(const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,NUV> & x)
{
	typedef typename promote_trait<NUM,NUV>::T_promote T_promote;
	CheckSpMVCompliance(A, x);

	MPI_Comm World = x.commGrid->GetWorld();
	MPI_Comm ColWorld = x.commGrid->GetColWorld();
	MPI_Comm RowWorld = x.commGrid->GetRowWorld();

	int xlocnz = (int) x.getlocnnz();
	int trxlocnz = 0;
	int roffst = x.RowLenUntil();
	int offset;

	int diagneigh = x.commGrid->GetComplementRank();
	MPI_Status status;
	MPI_Sendrecv(&xlocnz, 1, MPI_INT, diagneigh, TRX, &trxlocnz, 1, MPI_INT, diagneigh, TRX, World, &status);
	MPI_Sendrecv(&roffst, 1, MPI_INT, diagneigh, TROST, &offset, 1, MPI_INT, diagneigh, TROST, World, &status);
	
	IU * trxinds = new IU[trxlocnz];
	NUV * trxnums = new NUV[trxlocnz];
	MPI_Sendrecv(const_cast<IU*>(SpHelper::p2a(x.ind)), xlocnz, MPIType<IU>(), diagneigh, TRX, trxinds, trxlocnz, MPIType<IU>(), diagneigh, TRX, World, &status);
	MPI_Sendrecv(const_cast<NUV*>(SpHelper::p2a(x.num)), xlocnz, MPIType<NUV>(), diagneigh, TRX, trxnums, trxlocnz, MPIType<NUV>(), diagneigh, TRX, World, &status);
	// fullydist indexing (n pieces) -> matrix indexing (sqrt(p) pieces)
	std::transform(trxinds, trxinds+trxlocnz, trxinds, [offset](IU val){return val + offset;});

	int colneighs, colrank;
	MPI_Comm_size(ColWorld, &colneighs);
	MPI_Comm_rank(ColWorld, &colrank);
	int * colnz = new int[colneighs];
	colnz[colrank] = trxlocnz;
	MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, colnz, 1, MPI_INT, ColWorld);
	int * dpls = new int[colneighs]();	// displacements (zero initialized pid) 
	std::partial_sum(colnz, colnz+colneighs-1, dpls+1);
	int accnz = std::accumulate(colnz, colnz+colneighs, 0);
	IU * indacc = new IU[accnz];
	NUV * numacc = new NUV[accnz];

	// ABAB: Future issues here, colnz is of type int (MPI limitation)
	// What if the aggregate vector size along the processor row/column is not 32-bit addressible?
	MPI_Allgatherv(trxinds, trxlocnz, MPIType<IU>(), indacc, colnz, dpls, MPIType<IU>(), ColWorld);
	MPI_Allgatherv(trxnums, trxlocnz, MPIType<NUV>(), numacc, colnz, dpls, MPIType<NUV>(), ColWorld);
	DeleteAll(trxinds, trxnums);

	// serial SpMV with sparse vector
	std::vector< int32_t > indy;
	std::vector< T_promote >  numy;
	
        int32_t * tmpindacc = new int32_t[accnz];
        for(int i=0; i< accnz; ++i) tmpindacc[i] = indacc[i];
	delete [] indacc;

	dcsc_gespmv<SR>(*(A.spSeq), tmpindacc, numacc, accnz, indy, numy);	// actual multiplication

	DeleteAll(tmpindacc, numacc);
	DeleteAll(colnz, dpls);

	FullyDistSpVec<IU, T_promote> y ( x.commGrid, A.getnrow());	// identity doesn't matter for sparse vectors
	IU yintlen = y.MyRowLength();

	int rowneighs;
	MPI_Comm_size(RowWorld,&rowneighs);
	std::vector< std::vector<IU> > sendind(rowneighs);
	std::vector< std::vector<T_promote> > sendnum(rowneighs);
	typename std::vector<int32_t>::size_type outnz = indy.size();
	for(typename std::vector<IU>::size_type i=0; i< outnz; ++i)
	{
		IU locind;
		int rown = y.OwnerWithinRow(yintlen, static_cast<IU>(indy[i]), locind);
		sendind[rown].push_back(locind);
		sendnum[rown].push_back(numy[i]);
	}

	IU * sendindbuf = new IU[outnz];
	T_promote * sendnumbuf = new T_promote[outnz];
	int * sendcnt = new int[rowneighs];
	int * sdispls = new int[rowneighs];
	for(int i=0; i<rowneighs; ++i)
		sendcnt[i] = sendind[i].size();

	int * rdispls = new int[rowneighs];
	int * recvcnt = new int[rowneighs];
	MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, RowWorld);       // share the request counts

	sdispls[0] = 0;
	rdispls[0] = 0;
	for(int i=0; i<rowneighs-1; ++i)
	{
		sdispls[i+1] = sdispls[i] + sendcnt[i];
		rdispls[i+1] = rdispls[i] + recvcnt[i];
	}
	int totrecv = std::accumulate(recvcnt,recvcnt+rowneighs,0);
	IU * recvindbuf = new IU[totrecv];
	T_promote * recvnumbuf = new T_promote[totrecv];

	for(int i=0; i<rowneighs; ++i)
	{
    std::copy(sendind[i].begin(), sendind[i].end(), sendindbuf+sdispls[i]);
		std::vector<IU>().swap(sendind[i]);
	}
	for(int i=0; i<rowneighs; ++i)
	{
    std::copy(sendnum[i].begin(), sendnum[i].end(), sendnumbuf+sdispls[i]);
		std::vector<T_promote>().swap(sendnum[i]);
	}
	MPI_Alltoallv(sendindbuf, sendcnt, sdispls, MPIType<IU>(), recvindbuf, recvcnt, rdispls, MPIType<IU>(), RowWorld);
	MPI_Alltoallv(sendnumbuf, sendcnt, sdispls, MPIType<T_promote>(), recvnumbuf, recvcnt, rdispls, MPIType<T_promote>(), RowWorld);
	
	DeleteAll(sendindbuf, sendnumbuf);
	DeleteAll(sendcnt, recvcnt, sdispls, rdispls);
		
	// define a SPA-like data structure
	IU ysize = y.MyLocLength();
	T_promote * localy = new T_promote[ysize];
	bool * isthere = new bool[ysize];
	std::vector<IU> nzinds;	// nonzero indices		
  std::fill_n(isthere, ysize, false);
	
	for(int i=0; i< totrecv; ++i)
	{
		if(!isthere[recvindbuf[i]])
		{
			localy[recvindbuf[i]] = recvnumbuf[i];	// initial assignment
			nzinds.push_back(recvindbuf[i]);
			isthere[recvindbuf[i]] = true;
		} 
		else
		{
			localy[recvindbuf[i]] = SR::add(localy[recvindbuf[i]], recvnumbuf[i]);	
		}
	}
	DeleteAll(isthere, recvindbuf, recvnumbuf);
	sort(nzinds.begin(), nzinds.end());
	int nnzy = nzinds.size();
	y.ind.resize(nnzy);
	y.num.resize(nnzy);
	for(int i=0; i< nnzy; ++i)
	{
		y.ind[i] = nzinds[i];
		y.num[i] = localy[nzinds[i]]; 	
	}
	delete [] localy;
	return y;
}

// Aydin (June 2021):
// This currently duplicates the work of EWiseMult with exclude = true
// However, this is the right way of implementing it because it allows set difference when 
// the types of two matrices do not have a valid multiplication operator defined
// set difference should not require such an operator so we will move all code 
// bases that use EWiseMult(..., exclude=true) to this one
template <typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB>
SpParMat<IU,NU1,UDERA> SetDifference(const SpParMat<IU,NU1,UDERA> & A, const SpParMat<IU,NU2,UDERB> & B)
{
	if(*(A.commGrid) == *(B.commGrid))
        {
                UDERA * result = new UDERA( SetDifference(*(A.spSeq),*(B.spSeq)));
                return SpParMat<IU, NU1, UDERA> (result, A.commGrid);
        }
        else
        {
                std::cout << "Grids are not comparable for set difference" << std::endl;
                MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
                return SpParMat< IU,NU1,UDERA >();
        }

}

template <typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB> 
SpParMat<IU,typename promote_trait<NU1,NU2>::T_promote,typename promote_trait<UDERA,UDERB>::T_promote> EWiseMult 
	(const SpParMat<IU,NU1,UDERA> & A, const SpParMat<IU,NU2,UDERB> & B , bool exclude)
{
	typedef typename promote_trait<NU1,NU2>::T_promote N_promote;
	typedef typename promote_trait<UDERA,UDERB>::T_promote DER_promote;

	if(*(A.commGrid) == *(B.commGrid))	
	{
		DER_promote * result = new DER_promote( EWiseMult(*(A.spSeq),*(B.spSeq),exclude) );
		return SpParMat<IU, N_promote, DER_promote> (result, A.commGrid);
	}
	else
	{
		std::cout << "Grids are not comparable elementwise multiplication" << std::endl; 
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
		return SpParMat< IU,N_promote,DER_promote >();
	}
}
	
template <typename RETT, typename RETDER, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB, typename _BinaryOperation> 
SpParMat<IU,RETT,RETDER> EWiseApply 
	(const SpParMat<IU,NU1,UDERA> & A, const SpParMat<IU,NU2,UDERB> & B, _BinaryOperation __binary_op, bool notB, const NU2& defaultBVal)
{
	if(*(A.commGrid) == *(B.commGrid))	
	{
		RETDER * result = new RETDER( EWiseApply<RETT>(*(A.spSeq),*(B.spSeq), __binary_op, notB, defaultBVal) );
		return SpParMat<IU, RETT, RETDER> (result, A.commGrid);
	}
	else
	{
		std::cout << "Grids are not comparable elementwise apply" << std::endl; 
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
		return SpParMat< IU,RETT,RETDER >();
	}
}

template <typename RETT, typename RETDER, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB, typename _BinaryOperation, typename _BinaryPredicate> 
SpParMat<IU,RETT,RETDER> EWiseApply
	(const SpParMat<IU,NU1,UDERA> & A, const SpParMat<IU,NU2,UDERB> & B, _BinaryOperation __binary_op, _BinaryPredicate do_op, bool allowANulls, bool allowBNulls, const NU1& ANullVal, const NU2& BNullVal, const bool allowIntersect, const bool useExtendedBinOp)
{
	if(*(A.commGrid) == *(B.commGrid))	
	{
		RETDER * result = new RETDER( EWiseApply<RETT>(*(A.spSeq),*(B.spSeq), __binary_op, do_op, allowANulls, allowBNulls, ANullVal, BNullVal, allowIntersect) );
		return SpParMat<IU, RETT, RETDER> (result, A.commGrid);
	}
	else
	{
		std::cout << "Grids are not comparable elementwise apply" << std::endl; 
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
		return SpParMat< IU,RETT,RETDER >();
	}
}

// plain adapter
template <typename RETT, typename RETDER, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB, typename _BinaryOperation, typename _BinaryPredicate> 
SpParMat<IU,RETT,RETDER>
EWiseApply (const SpParMat<IU,NU1,UDERA> & A, const SpParMat<IU,NU2,UDERB> & B, _BinaryOperation __binary_op, _BinaryPredicate do_op, bool allowANulls, bool allowBNulls, const NU1& ANullVal, const NU2& BNullVal, const bool allowIntersect = true)
{
	return EWiseApply<RETT, RETDER>(A, B,
				EWiseExtToPlainAdapter<RETT, NU1, NU2, _BinaryOperation>(__binary_op),
				EWiseExtToPlainAdapter<bool, NU1, NU2, _BinaryPredicate>(do_op),
				allowANulls, allowBNulls, ANullVal, BNullVal, allowIntersect, true);
}
// end adapter

/**
 * if exclude is true, then we prune all entries W[i] != zero from V 
 * if exclude is false, then we perform a proper elementwise multiplication
**/
template <typename IU, typename NU1, typename NU2>
FullyDistSpVec<IU,typename promote_trait<NU1,NU2>::T_promote> EWiseMult 
	(const FullyDistSpVec<IU,NU1> & V, const FullyDistVec<IU,NU2> & W , bool exclude, NU2 zero)
{
	typedef typename promote_trait<NU1,NU2>::T_promote T_promote;

	if(*(V.commGrid) == *(W.commGrid))	
	{
		FullyDistSpVec< IU, T_promote> Product(V.commGrid);
		if(V.glen != W.glen)
		{
			std::cerr << "Vector dimensions don't match for EWiseMult\n";
			MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
		}
		else
		{
			Product.glen = V.glen;
			IU size= V.getlocnnz();
			if(exclude)
			{
				#if defined(_OPENMP) && defined(CBLAS_EXPERIMENTAL)	// not faster than serial
				int actual_splits = cblas_splits * 1;	// 1 is the parallel slackness
        std::vector <IU> tlosizes (actual_splits, 0);
        std::vector < std::vector<IU> > tlinds(actual_splits);
        std::vector < std::vector<T_promote> > tlnums(actual_splits);
				IU tlsize = size / actual_splits;
				#pragma omp parallel for //schedule(dynamic, 1)
				for(IU t = 0; t < actual_splits; ++t)
				{
					IU tlbegin = t*tlsize;
					IU tlend = (t==actual_splits-1)? size : (t+1)*tlsize;
					for(IU i=tlbegin; i<tlend; ++i)
					{
						if(W.arr[V.ind[i]] == zero) 	// keep only those
						{
							tlinds[t].push_back(V.ind[i]);
							tlnums[t].push_back(V.num[i]);
							tlosizes[t]++;
						}
					}
				}
        std::vector<IU> prefix_sum(actual_splits+1,0);
        std::partial_sum(tlosizes.begin(), tlosizes.end(), prefix_sum.begin()+1); 
				Product.ind.resize(prefix_sum[actual_splits]);
				Product.num.resize(prefix_sum[actual_splits]);
			
				#pragma omp parallel for //schedule(dynamic, 1)
				for(IU t=0; t< actual_splits; ++t)
				{
          std::copy(tlinds[t].begin(), tlinds[t].end(), Product.ind.begin()+prefix_sum[t]);
          std::copy(tlnums[t].begin(), tlnums[t].end(), Product.num.begin()+prefix_sum[t]);
				}
				#else
				for(IU i=0; i<size; ++i)
				{
					if(W.arr[V.ind[i]] == zero)     // keep only those
					{
                        	       		Product.ind.push_back(V.ind[i]);
                                		Product.num.push_back(V.num[i]);
                                      	}	
				}
				#endif
			}
			else
			{
				for(IU i=0; i<size; ++i)
				{
					if(W.arr[V.ind[i]] != zero) 	// keep only those
					{
						Product.ind.push_back(V.ind[i]);
						Product.num.push_back(V.num[i] * W.arr[V.ind[i]]);
					}
				}
			}
		}
		return Product;
	}
	else
	{
		std::cout << "Grids are not comparable elementwise multiplication" << std::endl; 
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
		return FullyDistSpVec< IU,T_promote>();
	}
}


/**
 Threaded EWiseApply. Only called internally from EWiseApply.
**/
template <typename RET, typename IU, typename NU1, typename NU2, typename _BinaryOperation, typename _BinaryPredicate>
FullyDistSpVec<IU,RET> EWiseApply_threaded
	(const FullyDistSpVec<IU,NU1> & V, const FullyDistVec<IU,NU2> & W , _BinaryOperation _binary_op, _BinaryPredicate _doOp, bool allowVNulls, NU1 Vzero, const bool useExtendedBinOp)
{
	typedef RET T_promote; //typedef typename promote_trait<NU1,NU2>::T_promote T_promote;
	if(*(V.commGrid) == *(W.commGrid))	
	{
		FullyDistSpVec< IU, T_promote> Product(V.commGrid);
		if(V.TotalLength() != W.TotalLength())
		{
			std::ostringstream outs;
			outs << "Vector dimensions don't match (" << V.TotalLength() << " vs " << W.TotalLength() << ") for EWiseApply (short version)\n";
			SpParHelper::Print(outs.str());
			MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
		}
		else
		{
            int nthreads=1;
#ifdef _OPENMP
#pragma omp parallel
            {
                nthreads = omp_get_num_threads();
            }
#endif

			Product.glen = V.glen;
			IU size= W.LocArrSize();
			IU spsize = V.getlocnnz();
            
            // temporary result vectors per thread
            std::vector<std::vector<IU>> tProductInd(nthreads);
            std::vector<std::vector<T_promote>> tProductVal(nthreads);
            IU perthread; //chunk of tProductInd or tProductVal allocated to each thread
            if (allowVNulls)
                perthread = size/nthreads;
            else
                perthread = spsize/nthreads;
            
#ifdef _OPENMP
#pragma omp parallel
#endif
            {
                int curthread = 0;
#ifdef _OPENMP
                curthread = omp_get_thread_num();
#endif
                IU tStartIdx = perthread * curthread;
                IU tNextIdx = perthread * (curthread+1);
                
                if (allowVNulls)
                {
                    if(curthread == nthreads-1) tNextIdx = size;
                    
                    // get sparse part for the current thread
                    auto it = std::lower_bound (V.ind.begin(), V.ind.end(), tStartIdx);
                    IU tSpIdx = (IU) std::distance(V.ind.begin(), it);
                    
                    // iterate over the dense vector
                    for(IU tIdx=tStartIdx; tIdx < tNextIdx; ++tIdx)
                    {
                        if(tSpIdx < spsize && V.ind[tSpIdx] < tNextIdx && V.ind[tSpIdx] == tIdx)
                        {
                            if (_doOp(V.num[tSpIdx], W.arr[tIdx], false, false))
                            {
                                tProductInd[curthread].push_back(tIdx);
                                tProductVal[curthread].push_back (_binary_op(V.num[tSpIdx], W.arr[tIdx], false, false));
                            }
                            tSpIdx++;
                        }
                        else
                        {
                            if (_doOp(Vzero, W.arr[tIdx], true, false))
                            {
                                tProductInd[curthread].push_back(tIdx);
                                tProductVal[curthread].push_back (_binary_op(Vzero, W.arr[tIdx], true, false));
                            }
                        }
                    }
                }
                else // iterate over the sparse vector
                {
                    if(curthread == nthreads-1) tNextIdx = spsize;
                    for(IU tSpIdx=tStartIdx; tSpIdx < tNextIdx; ++tSpIdx)
                    {
                        if (_doOp(V.num[tSpIdx], W.arr[V.ind[tSpIdx]], false, false))
                        {
                            
                            tProductInd[curthread].push_back( V.ind[tSpIdx]);
                            tProductVal[curthread].push_back (_binary_op(V.num[tSpIdx], W.arr[V.ind[tSpIdx]], false, false));
                        }
                    }
                }
            }
            
            std::vector<IU> tdisp(nthreads+1);
            tdisp[0] = 0;
            for(int i=0; i<nthreads; ++i)
            {
                tdisp[i+1] = tdisp[i] + tProductInd[i].size();
            }
            
            // copy results from temporary vectors
            Product.ind.resize(tdisp[nthreads]);
            Product.num.resize(tdisp[nthreads]);
            
#ifdef _OPENMP
#pragma omp parallel
#endif
            {
                int curthread = 0;
#ifdef _OPENMP
                curthread = omp_get_thread_num();
#endif
                std::copy(tProductInd[curthread].begin(), tProductInd[curthread].end(), Product.ind.data() + tdisp[curthread]);
                std::copy(tProductVal[curthread].begin() , tProductVal[curthread].end(), Product.num.data() + tdisp[curthread]);
            }
		}
		return Product;
	}
	else
	{
		std::cout << "Grids are not comparable for EWiseApply" << std::endl;
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
		return FullyDistSpVec< IU,T_promote>();
	}
}



/**
 * Performs an arbitrary binary operation _binary_op on the corresponding elements of two vectors with the result stored in a return vector ret.
 * The binary operatiation is only performed if the binary predicate _doOp returns true for those elements. Otherwise the binary operation is not
 * performed and ret does not contain an element at that position.
 * More formally the operation is defined as:
 * if (_doOp(V[i], W[i]))
 *    ret[i] = _binary_op(V[i], W[i])
 * else
 *    // ret[i] is not set
 * Hence _doOp can be used to implement a filter on either of the vectors.
 *
 * The above is only defined if both V[i] and W[i] exist (i.e. an intersection). To allow a union operation (ex. when V[i] doesn't exist but W[i] does)
 * the allowVNulls flag is set to true and the Vzero argument is used as the missing V[i] value.
 *
 * The type of each element of ret must not necessarily be related to the types of V or W, so the return type must be explicitly specified as a template parameter:
 * FullyDistSpVec<int, double> r = EWiseApply<double>(V, W, plus, retTrue, false, 0)
 **/
template <typename RET, typename IU, typename NU1, typename NU2, typename _BinaryOperation, typename _BinaryPredicate>
FullyDistSpVec<IU,RET> EWiseApply
(const FullyDistSpVec<IU,NU1> & V, const FullyDistVec<IU,NU2> & W , _BinaryOperation _binary_op, _BinaryPredicate _doOp, bool allowVNulls, NU1 Vzero, const bool useExtendedBinOp)
{
    
#ifdef _OPENMP
    return EWiseApply_threaded<RET>(V, W, _binary_op, _doOp, allowVNulls, Vzero, useExtendedBinOp);
    
#else
    typedef RET T_promote; //typedef typename promote_trait<NU1,NU2>::T_promote T_promote;
    if(*(V.commGrid) == *(W.commGrid))
    {
        FullyDistSpVec< IU, T_promote> Product(V.commGrid);
        //FullyDistVec< IU, NU1> DV (V); // Ariful: I am not sure why it was there??
        if(V.TotalLength() != W.TotalLength())
        {
          std::ostringstream outs;
            outs << "Vector dimensions don't match (" << V.TotalLength() << " vs " << W.TotalLength() << ") for EWiseApply (short version)\n";
            SpParHelper::Print(outs.str());
            MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
        }
        else
        {
            Product.glen = V.glen;
            IU size= W.LocArrSize();
            IU spsize = V.getlocnnz();
            IU sp_iter = 0;
            if (allowVNulls)
            {
                // iterate over the dense vector
                for(IU i=0; i<size; ++i)
                {
                    if(sp_iter < spsize && V.ind[sp_iter] == i)
                    {
                        if (_doOp(V.num[sp_iter], W.arr[i], false, false))
                        {
                            Product.ind.push_back(i);
                            Product.num.push_back(_binary_op(V.num[sp_iter], W.arr[i], false, false));
                        }
                        sp_iter++;
                    }
                    else
                    {
                        if (_doOp(Vzero, W.arr[i], true, false))
                        {
                            Product.ind.push_back(i);
                            Product.num.push_back(_binary_op(Vzero, W.arr[i], true, false));
                        }
                    }
                }
            }
            else
            {
                // iterate over the sparse vector
                for(sp_iter = 0; sp_iter < spsize; ++sp_iter)
                {
                    if (_doOp(V.num[sp_iter], W.arr[V.ind[sp_iter]], false, false))
                    {
                        Product.ind.push_back(V.ind[sp_iter]);
                        Product.num.push_back(_binary_op(V.num[sp_iter], W.arr[V.ind[sp_iter]], false, false));
                    }
                }
                
            }
        }
        return Product;
    }
    else
    {
      std::cout << "Grids are not comparable for EWiseApply" << std::endl; 
        MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
        return FullyDistSpVec< IU,T_promote>();
    }
#endif
}



/**
 * Performs an arbitrary binary operation _binary_op on the corresponding elements of two vectors with the result stored in a return vector ret. 
 * The binary operatiation is only performed if the binary predicate _doOp returns true for those elements. Otherwise the binary operation is not 
 * performed and ret does not contain an element at that position.
 * More formally the operation is defined as:
 * if (_doOp(V[i], W[i]))
 *    ret[i] = _binary_op(V[i], W[i])
 * else
 *    // ret[i] is not set
 * Hence _doOp can be used to implement a filter on either of the vectors.
 *
 * The above is only defined if both V[i] and W[i] exist (i.e. an intersection). To allow a union operation (ex. when V[i] doesn't exist but W[i] does) 
 * the allowVNulls flag is set to true and the Vzero argument is used as the missing V[i] value.
 * !allowVNulls && !allowWNulls => intersection
 * !allowVNulls &&  allowWNulls => operate on all elements of V 
 *  allowVNulls && !allowWNulls => operate on all elements of W
 *  allowVNulls &&  allowWNulls => union
 *
 * The type of each element of ret must not necessarily be related to the types of V or W, so the return type must be explicitly specified as a template parameter:
 * FullyDistSpVec<int, double> r = EWiseApply<double>(V, W, plus, ...)
 * For intersection, Vzero and Wzero are irrelevant
 * ABAB: \todo: Should allowIntersect be "false" for all SetDifference uses?
**/
template <typename RET, typename IU, typename NU1, typename NU2, typename _BinaryOperation, typename _BinaryPredicate>
FullyDistSpVec<IU,RET> EWiseApply 
	(const FullyDistSpVec<IU,NU1> & V, const FullyDistSpVec<IU,NU2> & W , _BinaryOperation _binary_op, _BinaryPredicate _doOp, bool allowVNulls, bool allowWNulls, NU1 Vzero, NU2 Wzero, const bool allowIntersect, const bool useExtendedBinOp)
{

	typedef RET T_promote; // typename promote_trait<NU1,NU2>::T_promote T_promote;
	if(*(V.commGrid) == *(W.commGrid))	
	{
		FullyDistSpVec< IU, T_promote> Product(V.commGrid);
		if(V.glen != W.glen)
		{
			std::ostringstream outs;
			outs << "Vector dimensions don't match (" << V.glen << " vs " << W.glen << ") for EWiseApply (full version)\n";
			SpParHelper::Print(outs.str());
			MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
		}
		else
		{
			Product.glen = V.glen;
			typename std::vector< IU  >::const_iterator indV = V.ind.begin();
			typename std::vector< NU1 >::const_iterator numV = V.num.begin();
			typename std::vector< IU  >::const_iterator indW = W.ind.begin();
			typename std::vector< NU2 >::const_iterator numW = W.num.begin();
			
			while (indV < V.ind.end() && indW < W.ind.end())
			{
				if (*indV == *indW)
				{
					// overlap
					if (allowIntersect)
					{
						if (_doOp(*numV, *numW, false, false))
						{
							Product.ind.push_back(*indV);
							Product.num.push_back(_binary_op(*numV, *numW, false, false));
						}
					}
					indV++; numV++;
					indW++; numW++;
				}
				else if (*indV < *indW)
				{
					// V has value but W does not
					if (allowWNulls)
					{
						if (_doOp(*numV, Wzero, false, true))
						{
							Product.ind.push_back(*indV);
							Product.num.push_back(_binary_op(*numV, Wzero, false, true));
						}
					}
					indV++; numV++;
				}
				else //(*indV > *indW)
				{
					// W has value but V does not
					if (allowVNulls)
					{
						if (_doOp(Vzero, *numW, true, false))
						{
							Product.ind.push_back(*indW);
							Product.num.push_back(_binary_op(Vzero, *numW, true, false));
						}
					}
					indW++; numW++;
				}
			}
			// clean up
			while (allowWNulls && indV < V.ind.end())
			{
				if (_doOp(*numV, Wzero, false, true))
				{
					Product.ind.push_back(*indV);
					Product.num.push_back(_binary_op(*numV, Wzero, false, true));
				}
				indV++; numV++;
			}
			while (allowVNulls && indW < W.ind.end())
			{
				if (_doOp(Vzero, *numW, true, false))
				{
					Product.ind.push_back(*indW);
					Product.num.push_back(_binary_op(Vzero, *numW, true, false));
				}
				indW++; numW++;
			}
		}
		return Product;
	}
	else
	{
		std::cout << "Grids are not comparable for EWiseApply" << std::endl; 
		MPI_Abort(MPI_COMM_WORLD, GRIDMISMATCH);
		return FullyDistSpVec< IU,T_promote>();
	}
}

// plain callback versions
template <typename RET, typename IU, typename NU1, typename NU2, typename _BinaryOperation, typename _BinaryPredicate>
FullyDistSpVec<IU,RET> EWiseApply 
	(const FullyDistSpVec<IU,NU1> & V, const FullyDistVec<IU,NU2> & W , _BinaryOperation _binary_op, _BinaryPredicate _doOp, bool allowVNulls, NU1 Vzero)
{


	return EWiseApply<RET>(V, W,
					EWiseExtToPlainAdapter<RET, NU1, NU2, _BinaryOperation>(_binary_op),
					EWiseExtToPlainAdapter<bool, NU1, NU2, _BinaryPredicate>(_doOp),
					allowVNulls, Vzero, true);
}



template <typename RET, typename IU, typename NU1, typename NU2, typename _BinaryOperation, typename _BinaryPredicate>
FullyDistSpVec<IU,RET> EWiseApply 
	(const FullyDistSpVec<IU,NU1> & V, const FullyDistSpVec<IU,NU2> & W , _BinaryOperation _binary_op, _BinaryPredicate _doOp, bool allowVNulls, bool allowWNulls, NU1 Vzero, NU2 Wzero, const bool allowIntersect = true)
{
	return EWiseApply<RET>(V, W,
					EWiseExtToPlainAdapter<RET, NU1, NU2, _BinaryOperation>(_binary_op),
					EWiseExtToPlainAdapter<bool, NU1, NU2, _BinaryPredicate>(_doOp),
					allowVNulls, allowWNulls, Vzero, Wzero, allowIntersect, true);
}





////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// sampling-based nnz estimation via SpMV
// @OGUZ-NOTE This is not based on SUMMA, do not use. Estimates the number of
// nonzeros in the final output matrix.


#define NROUNDS 5
typedef std::array<float, NROUNDS> samparr_t;

template <typename NZT>
struct promote_trait<NZT, samparr_t>
{
	typedef samparr_t T_promote;
};



class SamplesSaveHandler
{
public:
	template<typename c, typename t, typename V>
	void save(std::basic_ostream<c, t> &os,
			  std::array<V, NROUNDS> &sample_vec,
			  int64_t index)
	{
		for (auto it = sample_vec.begin(); it != sample_vec.end(); ++it)
			os << *it << " ";
	}
};



template<typename NZT>
struct SelectMinxSR
{
	static samparr_t id()
	{
		samparr_t arr;
		for (auto it = arr.begin(); it != arr.end(); ++it)
			*it = std::numeric_limits<float>::max();
		return arr;
	}


	static bool returnedSAID()
	{
		return false;
	}


	static samparr_t
	add (const samparr_t &arg1, const samparr_t &arg2)
	{
		samparr_t out;
		for (int i = 0; i < NROUNDS; ++i)
			out[i] = std::min(arg1[i], arg2[i]);
		return out;
	}


	static samparr_t
	multiply (const NZT arg1, const samparr_t &arg2)
	{
		return arg2;
	}


	static void axpy (const NZT a, const samparr_t &x, samparr_t &y)
	{
		y = add(y, multiply(a, x));
	}


	static MPI_Op mpi_op()
	{
		static MPI_Op mpiop;
		static bool exists = false;
		if (exists)
			return mpiop;
		else
		{
			MPI_Op_create(MPI_func, true, &mpiop);
			exists = true;
			return mpiop;
		}
	}


	static void
	MPI_func(void *invec, void *inoutvec, int *len, MPI_Datatype *datatype)
	{
		samparr_t   *in    = static_cast<samparr_t *>(invec);
		samparr_t   *inout = static_cast<samparr_t *>(inoutvec);
		for (int i = 0; i < *len; ++i)
			inout[i] = add(inout[i], in[i]);
	}
};



template <typename IU, typename NU1, typename NU2,
		  typename UDERA, typename UDERB>
int64_t
EstPerProcessNnzSpMV(
    SpParMat<IU, NU1, UDERA> &A, SpParMat<IU, NU2, UDERB> &B
	)  
{
	int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	float lambda = 1.0f;

	int nthds = 1;
	#ifdef THREADED
	#pragma omp parallel
	#endif
	{
		nthds = omp_get_num_threads();
	}

	if (myrank == 0)
		std::cout << "taking transposes." << std::endl;
	
	A.Transpose();
	B.Transpose();

	if (myrank == 0)
		std::cout << "setting initial samples." << std::endl;
	
	samparr_t sa;
	FullyDistVec<IU, samparr_t> samples_init(A.getcommgrid(), A.getncol(), sa);

	#ifdef THREADED
	#pragma omp parallel
	#endif
	{
		std::default_random_engine gen;
		std::exponential_distribution<float> exp_dist(lambda);

		#ifdef THREADED
		#pragma omp parallel for
		#endif
		for (IU i = 0; i < samples_init.LocArrSize(); ++i)
		{
			samparr_t tmp;
			for (auto it = tmp.begin(); it != tmp.end(); ++it)
				*it = exp_dist(gen);
			samples_init.SetLocalElement(i, tmp);
		}
	}

	// std::string fname("samples_init");
	// samples_init.ParallelWrite(fname, 1, SamplesSaveHandler(), true);
	
	if (myrank == 0)
		std::cout << "computing mid samples." << std::endl;

	FullyDistVec<IU, samparr_t> samples_mid =
		SpMV<SelectMinxSR<NU1> > (A, samples_init);

	// fname = "samples_mid";
	// samples_mid.ParallelWrite(fname, 1, SamplesSaveHandler(), true);

	if (myrank == 0)
		std::cout << "computing final samples." << std::endl;

	FullyDistVec<IU, samparr_t> samples_final =
		SpMV<SelectMinxSR<NU2> > (B, samples_mid);

	// fname = "samples_final";
	// samples_final.ParallelWrite(fname, 1, SamplesSaveHandler(), true);
	
	if (myrank == 0)
		std::cout << "computing nnz estimation." << std::endl;
	
	float nnzest = 0.0f;

	std::cout << myrank << "samples_final loc size: "
			  << samples_final.LocArrSize() << std::endl;

	const samparr_t *lsamples = samples_final.GetLocArr();
	
	#ifdef THREADED
	#pragma omp parallel for reduction (+:nnzest)
	#endif
	for (IU i = 0; i < samples_final.LocArrSize(); ++i)
	{
		float tmp = 0.0f;
		for (auto it = lsamples[i].begin(); it != lsamples[i].end(); ++it)
			tmp += *it;
		nnzest += static_cast<float>(NROUNDS - 1) / tmp;
	}

	if (myrank == 0)
		std::cout << "taking transposes again." << std::endl;

	int64_t nnzC_est = nnzest;
	int64_t nnzC_tot = 0;
	MPI_Allreduce(&nnzC_est, &nnzC_tot, 1, MPIType<int64_t>(), MPI_SUM,
				  (B.commGrid)->GetWorld());
	
	if (myrank == 0)
		std::cout << "sampling-based spmv est tot: " << nnzC_tot << std::endl;

	// revert back
	A.Transpose();
	B.Transpose();

	return nnzC_tot;
	
}

template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDER1, typename UDER2>
SpParMat3D<IU,NUO,UDERO> Mult_AnXBn_SUMMA3D(SpParMat3D<IU,NU1,UDER1> & A, SpParMat3D<IU,NU2,UDER2> & B){
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    typedef typename UDERO::LocalIT LIC;
    typedef typename UDER1::LocalIT LIA;
    typedef typename UDER2::LocalIT LIB;

#ifdef TIMING
    double t0, t1, t2, t3;
#endif

    /* 
     * Check if A and B are multipliable 
     * */
    if(A.getncol() != B.getnrow()){
        std::ostringstream outs;
        outs << "Can not multiply, dimensions does not match"<< std::endl;
        outs << A.getncol() << " != " << B.getnrow() << std::endl;
        SpParHelper::Print(outs.str());
        MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
    }
        
    /*
     * Calculate, accross fibers, which process should get how many columns after redistribution
     * */
    vector<LIB> divisions3d;
    // Calcuclate split boundaries as if all contents of the layer is being re-distributed along fiber
    // These boundaries will be used later on
    B.CalculateColSplitDistributionOfLayer(divisions3d); 

#ifdef TIMING
    t0 = MPI_Wtime();
#endif
    /*
     *  SUMMA Starts
     * */

    int stages, dummy; 	// last two parameters of ProductGrid are ignored for this multiplication
    std::shared_ptr<CommGrid> GridC = ProductGrid((A.GetLayerMat()->getcommgrid()).get(), 
                                                  (B.GetLayerMat()->getcommgrid()).get(), 
                                                  stages, dummy, dummy);		
    LIA C_m = A.GetLayerMat()->seqptr()->getnrow();
    LIB C_n = B.GetLayerMat()->seqptr()->getncol();

    LIA ** ARecvSizes = SpHelper::allocate2D<LIA>(UDER1::esscount, stages);
    LIB ** BRecvSizes = SpHelper::allocate2D<LIB>(UDER2::esscount, stages);
    
    SpParHelper::GetSetSizes( *(A.GetLayerMat()->seqptr()), ARecvSizes, (A.GetLayerMat()->getcommgrid())->GetRowWorld() );
    SpParHelper::GetSetSizes( *(B.GetLayerMat()->seqptr()), BRecvSizes, (B.GetLayerMat()->getcommgrid())->GetColWorld() );

    // Remotely fetched matrices are stored as pointers
    UDERO * ARecv; 
    UDER2 * BRecv;
    std::vector< SpTuples<LIC,NUO>  *> tomerge;

    int Aself = (A.GetLayerMat()->getcommgrid())->GetRankInProcRow();
    int Bself = (B.GetLayerMat()->getcommgrid())->GetRankInProcCol();	

    double Abcast_time = 0;
    double Bbcast_time = 0;
    double Local_multiplication_time = 0;
    
    for(int i = 0; i < stages; ++i) {
        std::vector<LIA> ess;

        if(i == Aself){
            ARecv = A.GetLayerMat()->seqptr();	// shallow-copy 
        }
        else{
            ess.resize(UDER1::esscount);
            for(int j=0; j<UDER1::esscount; ++j) {
                ess[j] = ARecvSizes[j][i];		// essentials of the ith matrix in this row	
            }
            ARecv = new UDER1();				// first, create the object
        }
#ifdef TIMING
        t2 = MPI_Wtime();
#endif
        if (Aself != i) {
            ARecv->Create(ess);
        }

        Arr<LIA,NU1> Aarrinfo = ARecv->GetArrays();

        for(unsigned int idx = 0; idx < Aarrinfo.indarrs.size(); ++idx) {
            MPI_Bcast(Aarrinfo.indarrs[idx].addr, Aarrinfo.indarrs[idx].count, MPIType<LIA>(), i, GridC->GetRowWorld());
        }

        for(unsigned int idx = 0; idx < Aarrinfo.numarrs.size(); ++idx) {
            MPI_Bcast(Aarrinfo.numarrs[idx].addr, Aarrinfo.numarrs[idx].count, MPIType<NU1>(), i, GridC->GetRowWorld());
        }
#ifdef TIMING
        t3 = MPI_Wtime();
        Abcast_time += (t3-t2);
#endif
        ess.clear();	
        if(i == Bself){
            BRecv = B.GetLayerMat()->seqptr();	// shallow-copy
        }
        else{
            ess.resize(UDER2::esscount);		
            for(int j=0; j<UDER2::esscount; ++j)	{
                ess[j] = BRecvSizes[j][i];	
            }	
            BRecv = new UDER2();
        }

        MPI_Barrier(A.GetLayerMat()->getcommgrid()->GetWorld());
#ifdef TIMING
        t2 = MPI_Wtime();
#endif
        if (Bself != i) {
            BRecv->Create(ess);	
        }
        Arr<LIB,NU2> Barrinfo = BRecv->GetArrays();

        for(unsigned int idx = 0; idx < Barrinfo.indarrs.size(); ++idx) {
            MPI_Bcast(Barrinfo.indarrs[idx].addr, Barrinfo.indarrs[idx].count, MPIType<LIB>(), i, GridC->GetColWorld());
        }
        for(unsigned int idx = 0; idx < Barrinfo.numarrs.size(); ++idx) {
            MPI_Bcast(Barrinfo.numarrs[idx].addr, Barrinfo.numarrs[idx].count, MPIType<NU2>(), i, GridC->GetColWorld());
        }
#ifdef TIMING
        t3 = MPI_Wtime();
        Bbcast_time += (t3-t2);
#endif

#ifdef TIMING
        t2 = MPI_Wtime();
#endif
        SpTuples<LIC,NUO> * C_cont = LocalSpGEMMHash<SR, NUO>
                            (*ARecv, *BRecv,    // parameters themselves
                            false,         // 'delete A' condition
                            false,         // 'delete B' condition
                            false);             // not to sort each column

        if(i != Bself && (!BRecv->isZero())) delete BRecv;
        if(i != Aself && (!ARecv->isZero())) delete ARecv;
#ifdef TIMING
        t3 = MPI_Wtime();
        Local_multiplication_time += (t3-t2);
#endif
        
        if(!C_cont->isZero()) tomerge.push_back(C_cont);
        
    }

    SpHelper::deallocate2D(ARecvSizes, UDER1::esscount);
    SpHelper::deallocate2D(BRecvSizes, UDER2::esscount);

#ifdef TIMING
    t2 = MPI_Wtime();
#endif
    SpTuples<LIC,NUO> * C_tuples = MultiwayMergeHash<SR>(tomerge, C_m, C_n, true, false); // Delete input arrays and do not sort
    //SpTuples<LIC,NUO> * C_tuples = MultiwayMergeHashSliding<SR>(tomerge, C_m, C_n, true, false); // Delete input arrays and do not sort
#ifdef TIMING
    t3 = MPI_Wtime();
#endif

#ifdef TIMING 
    if(myrank == 0){
        fprintf(stderr, "[SUMMA3D]\tAbcast_time: %lf\n", Abcast_time);
        fprintf(stderr, "[SUMMA3D]\tBbcast_time: %lf\n", Bbcast_time);
        fprintf(stderr, "[SUMMA3D]\tLocal_multiplication_time: %lf\n", Local_multiplication_time);
        fprintf(stderr, "[SUMMA3D]\tMerge_layer_time: %lf\n", (t3-t2));
    }
#endif
    /*
     *  SUMMA Ends
     * */
#ifdef TIMING
    t1 = MPI_Wtime();
    if(myrank == 0) fprintf(stderr, "[SUMMA3D]\tSUMMA time: %lf\n", (t1-t0));
#endif
    /*
     * 3d-reduction starts
     * */
#ifdef TIMING
    //MPI_Barrier(getcommgrid3D()->GetWorld());
    t0 = MPI_Wtime();
#endif
    MPI_Datatype MPI_tuple;
    MPI_Type_contiguous(sizeof(std::tuple<LIC,LIC,NUO>), MPI_CHAR, &MPI_tuple);
    MPI_Type_commit(&MPI_tuple);
    
    /*
     *  Create a profile with information regarding data to be sent and received between layers 
     *  These memory allocation needs to be `int` specifically because some of these arrays would be used in communication
     *  This is requirement is for MPI as MPI_Alltoallv takes pointer to integer exclusively as count and displacement
     * */
    int * sendcnt    = new int[A.getcommgrid3D()->GetGridLayers()];
    int * sendprfl   = new int[A.getcommgrid3D()->GetGridLayers()*3];
    int * sdispls    = new int[A.getcommgrid3D()->GetGridLayers()]();
    int * recvcnt    = new int[A.getcommgrid3D()->GetGridLayers()];
    int * recvprfl   = new int[A.getcommgrid3D()->GetGridLayers()*3];
    int * rdispls    = new int[A.getcommgrid3D()->GetGridLayers()]();

    vector<LIB> divisions3dPrefixSum(divisions3d.size());
    divisions3dPrefixSum[0] = 0;
    std::partial_sum(divisions3d.begin(), divisions3d.end()-1, divisions3dPrefixSum.begin()+1);
    ColLexiCompare<LIC,NUO> comp;
    IU totsend = C_tuples->getnnz();
    
#pragma omp parallel for
    for(int i=0; i < A.getcommgrid3D()->GetGridLayers(); ++i){
        LIB start_col = divisions3dPrefixSum[i];
        LIB end_col = divisions3dPrefixSum[i] + divisions3d[i];
        std::tuple<LIC, LIC, NUO> search_tuple_start(0, start_col, NUO());
        std::tuple<LIC, LIC, NUO> search_tuple_end(0, end_col, NUO());
        std::tuple<LIC, LIC, NUO>* start_it = std::lower_bound(C_tuples->tuples, C_tuples->tuples + C_tuples->getnnz(), search_tuple_start, comp);
        std::tuple<LIC, LIC, NUO>* end_it = std::lower_bound(C_tuples->tuples, C_tuples->tuples + C_tuples->getnnz(), search_tuple_end, comp);
        // This type casting is important from semantic point of view
        sendcnt[i] = (int)(end_it - start_it);
        sendprfl[i*3+0] = (int)(sendcnt[i]); // Number of nonzeros in ith chunk
        sendprfl[i*3+1] = (int)(A.GetLayerMat()->seqptr()->getnrow()); // Number of rows in ith chunk
        sendprfl[i*3+2] = (int)(divisions3d[i]); // Number of columns in ith chunk
    }
    std::partial_sum(sendcnt, sendcnt+A.getcommgrid3D()->GetGridLayers()-1, sdispls+1);

    // Send profile ready. Now need to update the tuples to reflect correct column id after column split.
    for(int i=0; i < A.getcommgrid3D()->GetGridLayers(); ++i){
#pragma omp parallel for schedule(static)
        for(int j = 0; j < sendcnt[i]; j++){
            std::get<1>(C_tuples->tuples[sdispls[i]+j]) = std::get<1>(C_tuples->tuples[sdispls[i]+j]) - divisions3dPrefixSum[i];
        }
    }

    MPI_Alltoall(sendprfl, 3, MPI_INT, recvprfl, 3, MPI_INT, A.getcommgrid3D()->GetFiberWorld());

    for(int i = 0; i < A.getcommgrid3D()->GetGridLayers(); i++) recvcnt[i] = recvprfl[i*3];
    std::partial_sum(recvcnt, recvcnt+A.getcommgrid3D()->GetGridLayers()-1, rdispls+1);
    IU totrecv = std::accumulate(recvcnt,recvcnt+A.getcommgrid3D()->GetGridLayers(), static_cast<IU>(0));
    std::tuple<LIC,LIC,NUO>* recvTuples = static_cast<std::tuple<LIC,LIC,NUO>*> (::operator new (sizeof(std::tuple<LIC,LIC,NUO>[totrecv])));

#ifdef TIMING
    t2 = MPI_Wtime();
#endif
    MPI_Alltoallv(C_tuples->tuples, sendcnt, sdispls, MPI_tuple, recvTuples, recvcnt, rdispls, MPI_tuple, A.getcommgrid3D()->GetFiberWorld());
    delete C_tuples;
#ifdef TIMING
    t3 = MPI_Wtime();
    if(myrank == 0) fprintf(stderr, "[SUMMA3D]\tAlltoallv: %lf\n", (t3-t2));
#endif
    vector<SpTuples<LIC, NUO>*> recvChunks(A.getcommgrid3D()->GetGridLayers());
#pragma omp parallel for
    for (int i = 0; i < A.getcommgrid3D()->GetGridLayers(); i++){
        recvChunks[i] = new SpTuples<LIC, NUO>(recvcnt[i], recvprfl[i*3+1], recvprfl[i*3+2], recvTuples + rdispls[i], true, false);
    }

    // Free all memory except tempTuples; Because that memory is holding data of newly created local matrices after receiving.
    DeleteAll(sendcnt, sendprfl, sdispls);
    DeleteAll(recvcnt, recvprfl, rdispls); 
    MPI_Type_free(&MPI_tuple);
    /*
     * 3d-reduction ends 
     * */
    
#ifdef TIMING
    t1 = MPI_Wtime();
    if(myrank == 0) fprintf(stderr, "[SUMMA3D]\tReduction time: %lf\n", (t1-t0));
#endif
#ifdef TIMING
    t0 = MPI_Wtime();
#endif
    /*
     * 3d-merge starts 
     * */
    SpTuples<LIC, NUO> * merged_tuples = MultiwayMergeHash<SR, LIC, NUO>(recvChunks, recvChunks[0]->getnrow(), recvChunks[0]->getncol(), false, false); // Do not delete
#ifdef TIMING
    t1 = MPI_Wtime();
    if(myrank == 0) fprintf(stderr, "[SUMMA3D]\tMerge_fiber_time: %lf\n", (t1-t0));
#endif
    //Create SpDCCol and delete merged_tuples;
    UDERO * localResultant = new UDERO(*merged_tuples, false);
    delete merged_tuples;

    // Do not delete elements of recvChunks, because that would give segmentation fault due to double free
    //delete [] recvTuples;
    ::operator delete(recvTuples);
    for(int i = 0; i < recvChunks.size(); i++){
        recvChunks[i]->tuples_deleted = true; // Temporary patch to avoid memory leak and segfault
        delete recvChunks[i];
    }
    vector<SpTuples<LIC,NUO>*>().swap(recvChunks);
    /*
     * 3d-merge ends
     * */

    std::shared_ptr<CommGrid3D> grid3d;
    grid3d.reset(new CommGrid3D(A.getcommgrid3D()->GetWorld(), A.getcommgrid3D()->GetGridLayers(), A.getcommgrid3D()->GetGridRows(), A.getcommgrid3D()->GetGridCols(), A.isSpecial()));
    SpParMat3D<IU, NUO, UDERO> C(localResultant, grid3d, A.isColSplit(), A.isSpecial());
    return C;
}

/*
 * Parameters:
 *  - computationKernel: 1 for hash-based, 2 for heap-based
 * */
template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB>
SpParMat3D<IU, NUO, UDERO> MemEfficientSpGEMM3D(SpParMat3D<IU, NU1, UDERA> & A, SpParMat3D<IU, NU2, UDERB> & B,
           int phases, NUO hardThreshold, IU selectNum, IU recoverNum, NUO recoverPct, int kselectVersion, int computationKernel, int64_t perProcessMemory){
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    typedef typename UDERA::LocalIT LIA;
    typedef typename UDERB::LocalIT LIB;
    typedef typename UDERO::LocalIT LIC;

    /* 
     * Check if A and B are multipliable 
     * */
    if(A.getncol() != B.getnrow()){
        std::ostringstream outs;
        outs << "Can not multiply, dimensions does not match"<< std::endl;
        outs << A.getncol() << " != " << B.getnrow() << std::endl;
        SpParHelper::Print(outs.str());
        MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
    }

    /* 
     * If provided number of phase is too low or too high then reset value of phase as 1 
     * */
    if(phases < 1 || phases >= B.getncol()){
        SpParHelper::Print("[MemEfficientSpGEMM3D]\tThe value of phases is too small or large. Resetting to 1.\n");
        phases = 1;
    }
    double t0, t1, t2, t3, t4, t5, t6, t7, t8, t9; // To time different parts of the function
#ifdef TIMING
    MPI_Barrier(B.getcommgrid3D()->GetWorld());
    t0 = MPI_Wtime();
#endif
    /* 
     * If per process memory is provided then calculate number of phases 
     * Otherwise, proceed to multiplication.
     * */
    if(perProcessMemory > 0) {
        int p, calculatedPhases;
        MPI_Comm_size(A.getcommgrid3D()->GetLayerWorld(),&p);
        int64_t perNNZMem_in = sizeof(IU)*2 + sizeof(NU1);
        int64_t perNNZMem_out = sizeof(IU)*2 + sizeof(NUO);

        int64_t lannz = A.GetLayerMat()->getlocalnnz();
        int64_t gannz = 0;
        // Get maximum number of nnz owned by one process
        MPI_Allreduce(&lannz, &gannz, 1, MPIType<int64_t>(), MPI_MAX, A.getcommgrid3D()->GetWorld()); 
        //int64_t ginputMem = gannz * perNNZMem_in * 4; // Four pieces per process: one piece of own A and B, one piece of received A and B
        int64_t ginputMem = gannz * perNNZMem_in * 5; // One extra copy for safety
        
        // Estimate per layer nnz after multiplication. After this estimation each process would know an estimation of
        // how many nnz the corresponding layer will have after the layerwise operation.
        int64_t asquareNNZ = EstPerProcessNnzSUMMA(*(A.GetLayerMat()), *(B.GetLayerMat()), true);
        int64_t gasquareNNZ;
        MPI_Allreduce(&asquareNNZ, &gasquareNNZ, 1, MPIType<int64_t>(), MPI_MAX, A.getcommgrid3D()->GetFiberWorld());

        // Atmost two copies, one of a process's own, another received from fiber reduction
        int64_t gasquareMem = gasquareNNZ * perNNZMem_out * 2; 
        // Calculate estimated average degree after multiplication
        int64_t d = ceil( ( ( gasquareNNZ / B.getcommgrid3D()->GetGridLayers() ) * sqrt(p) ) / B.GetLayerMat()->getlocalcols() );
        // Calculate per column nnz how left after k-select. Minimum of average degree and k-select parameters.
        int64_t k = std::min(int64_t(std::max(selectNum, recoverNum)), d );

        //estimate output memory
        int64_t postKselectOutputNNZ = ceil(( (B.GetLayerMat()->getlocalcols() / B.getcommgrid3D()->GetGridLayers() ) * k)/sqrt(p)); // If kselect is run
        int64_t postKselectOutputMem = postKselectOutputNNZ * perNNZMem_out * 2;
        double remainingMem = perProcessMemory*1000000000 - ginputMem - postKselectOutputMem;
        int64_t kselectMem = B.GetLayerMat()->getlocalcols() * k * sizeof(NUO) * 3;

        //inputMem + outputMem + asquareMem/phases + kselectmem/phases < memory
        if(remainingMem > 0){
            calculatedPhases = ceil( (gasquareMem + kselectMem) / remainingMem ); // If kselect is run
        }
        else calculatedPhases = -1;

        int gCalculatedPhases;
        MPI_Allreduce(&calculatedPhases, &gCalculatedPhases, 1, MPI_INT, MPI_MAX, A.getcommgrid3D()->GetFiberWorld());
        if(gCalculatedPhases > phases) phases = gCalculatedPhases;
    }
    else{
        // Do nothing
    }
#ifdef TIMING
    MPI_Barrier(B.getcommgrid3D()->GetWorld());
    t1 = MPI_Wtime();
    mcl3d_symbolictime+=(t1-t0);
    //if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tSymbolic stage time: %lf\n", (t1-t0));
#endif
        
        
    /*
     * Calculate, accross fibers, which process should get how many columns after redistribution
     * */
    vector<LIB> divisions3d;
    // Calculate split boundaries as if all contents of the layer is being re-distributed along fiber
    // These boundaries will be used later on
    B.CalculateColSplitDistributionOfLayer(divisions3d); 

    /*
     * Split B according to calculated number of phases
     * For better load balancing split B into nlayers*phases chunks
     * */
    vector<UDERB*> PiecesOfB;
    vector<UDERB*> tempPiecesOfB;
    UDERB CopyB = *(B.GetLayerMat()->seqptr());
    CopyB.ColSplit(divisions3d, tempPiecesOfB); // Split B into `nlayers` chunks at first
    for(int i = 0; i < tempPiecesOfB.size(); i++){
        vector<UDERB*> temp;
        tempPiecesOfB[i]->ColSplit(phases, temp); // Split each chunk of B into `phases` chunks
        for(int j = 0; j < temp.size(); j++){
            PiecesOfB.push_back(temp[j]);
        }
    }

    vector<UDERO> toconcatenate;
    //if(myrank == 0){
        //fprintf(stderr, "[MemEfficientSpGEMM3D]\tRunning with phase: %d\n", phases);
    //}

    for(int p = 0; p < phases; p++){
        /*
         * At the start of each phase take appropriate pieces from previously created pieces of local B matrix
         * Appropriate means correct pieces so that 3D-merge can be properly load balanced.
         * */
        vector<LIB> lbDivisions3d; // load balance friendly division
        LIB totalLocalColumnInvolved = 0;
        vector<UDERB*> targetPiecesOfB; // Pieces of B involved in current phase
        for(int i = 0; i < PiecesOfB.size(); i++){
            if(i % phases == p){
                targetPiecesOfB.push_back(new UDERB(*(PiecesOfB[i])));
                lbDivisions3d.push_back(PiecesOfB[i]->getncol());
                totalLocalColumnInvolved += PiecesOfB[i]->getncol();
            }
        }

        /*
         * Create new local matrix by concatenating appropriately picked pieces
         * */
        UDERB * OnePieceOfB = new UDERB(0, (B.GetLayerMat())->seqptr()->getnrow(), totalLocalColumnInvolved, 0);
        OnePieceOfB->ColConcatenate(targetPiecesOfB);
        vector<UDERB*>().swap(targetPiecesOfB);

        /*
         * Create a new layer-wise distributed matrix with the newly created local matrix for this phase
         * This matrix is used in SUMMA multiplication of respective layer
         * */
        SpParMat<IU, NU2, UDERB> OnePieceOfBLayer(OnePieceOfB, A.getcommgrid3D()->GetLayerWorld());
#ifdef TIMING
        t0 = MPI_Wtime();
#endif
        /*
         *  SUMMA Starts
         * */

        int stages, dummy; 	// last two parameters of ProductGrid are ignored for this multiplication
        std::shared_ptr<CommGrid> GridC = ProductGrid((A.GetLayerMat()->getcommgrid()).get(), 
                                                      (OnePieceOfBLayer.getcommgrid()).get(), 
                                                      stages, dummy, dummy);		
        LIA C_m = A.GetLayerMat()->seqptr()->getnrow();
        LIB C_n = OnePieceOfBLayer.seqptr()->getncol();

        LIA ** ARecvSizes = SpHelper::allocate2D<LIA>(UDERA::esscount, stages);
        LIB ** BRecvSizes = SpHelper::allocate2D<LIB>(UDERB::esscount, stages);
        
        SpParHelper::GetSetSizes( *(A.GetLayerMat()->seqptr()), ARecvSizes, (A.GetLayerMat()->getcommgrid())->GetRowWorld() );
        SpParHelper::GetSetSizes( *(OnePieceOfBLayer.seqptr()), BRecvSizes, (OnePieceOfBLayer.getcommgrid())->GetColWorld() );

        // Remotely fetched matrices are stored as pointers
        UDERA * ARecv; 
        UDERB * BRecv;
        std::vector< SpTuples<LIC,NUO>  *> tomerge;

        int Aself = (A.GetLayerMat()->getcommgrid())->GetRankInProcRow();
        int Bself = (OnePieceOfBLayer.getcommgrid())->GetRankInProcCol();	

        double Abcast_time = 0;
        double Bbcast_time = 0;
        double Local_multiplication_time = 0;
        
        for(int i = 0; i < stages; ++i) {
            std::vector<LIA> ess;	

            if(i == Aself){
                ARecv = A.GetLayerMat()->seqptr();	// shallow-copy 
            }
            else{
                ess.resize(UDERA::esscount);
                for(int j=0; j<UDERA::esscount; ++j) {
                    ess[j] = ARecvSizes[j][i];		// essentials of the ith matrix in this row	
                }
                ARecv = new UDERA();				// first, create the object
            }
#ifdef TIMING
            t2 = MPI_Wtime();
#endif
            if (Aself != i) {
                ARecv->Create(ess);
            }

            Arr<LIA,NU1> Aarrinfo = ARecv->GetArrays();

            for(unsigned int idx = 0; idx < Aarrinfo.indarrs.size(); ++idx) {
                MPI_Bcast(Aarrinfo.indarrs[idx].addr, Aarrinfo.indarrs[idx].count, MPIType<IU>(), i, GridC->GetRowWorld());
            }

            for(unsigned int idx = 0; idx < Aarrinfo.numarrs.size(); ++idx) {
                MPI_Bcast(Aarrinfo.numarrs[idx].addr, Aarrinfo.numarrs[idx].count, MPIType<NU1>(), i, GridC->GetRowWorld());
            }
#ifdef TIMING
            t3 = MPI_Wtime();
            mcl3d_Abcasttime += (t3-t2);
            Abcast_time += (t3-t2);
#endif
            ess.clear();	
            if(i == Bself){
                BRecv = OnePieceOfBLayer.seqptr();	// shallow-copy
            }
            else{
                ess.resize(UDERB::esscount);		
                for(int j=0; j<UDERB::esscount; ++j)	{
                    ess[j] = BRecvSizes[j][i];	
                }	
                BRecv = new UDERB();
            }

            MPI_Barrier(A.GetLayerMat()->getcommgrid()->GetWorld());
#ifdef TIMING
            t2 = MPI_Wtime();
#endif
            if (Bself != i) {
                BRecv->Create(ess);	
            }
            Arr<LIB,NU2> Barrinfo = BRecv->GetArrays();

            for(unsigned int idx = 0; idx < Barrinfo.indarrs.size(); ++idx) {
                MPI_Bcast(Barrinfo.indarrs[idx].addr, Barrinfo.indarrs[idx].count, MPIType<IU>(), i, GridC->GetColWorld());
            }
            for(unsigned int idx = 0; idx < Barrinfo.numarrs.size(); ++idx) {
                MPI_Bcast(Barrinfo.numarrs[idx].addr, Barrinfo.numarrs[idx].count, MPIType<NU2>(), i, GridC->GetColWorld());
            }
#ifdef TIMING
            t3 = MPI_Wtime();
            mcl3d_Bbcasttime += (t3-t2);
            Bbcast_time += (t3-t2);
#endif

#ifdef TIMING
            t2 = MPI_Wtime();
#endif
            SpTuples<LIC,NUO> * C_cont;
            
            if(computationKernel == 1){
                C_cont = LocalSpGEMMHash<SR, NUO>
                                    (*ARecv, *BRecv,    // parameters themselves
                                    false,         // 'delete A' condition
                                    false,         // 'delete B' condition
                                    false);             // not to sort each column
            }
            else if(computationKernel == 2){
                C_cont = LocalSpGEMM<SR, NUO>
                                    (*ARecv, *BRecv,    // parameters themselves
                                    false,         // 'delete A' condition
                                    false);        // 'delete B' condition
            
            }
            if(i != Bself && (!BRecv->isZero())) delete BRecv;
            if(i != Aself && (!ARecv->isZero())) delete ARecv;
            
#ifdef TIMING
            t3 = MPI_Wtime();
            mcl3d_localspgemmtime += (t3-t2);
            Local_multiplication_time += (t3-t2);
#endif
            
            if(!C_cont->isZero()) tomerge.push_back(C_cont);
        }

        SpHelper::deallocate2D(ARecvSizes, UDERA::esscount);
        SpHelper::deallocate2D(BRecvSizes, UDERB::esscount);

#ifdef TIMING
        t2 = MPI_Wtime();
#endif
        SpTuples<LIC,NUO> * C_tuples;
        if(computationKernel == 1) C_tuples = MultiwayMergeHash<SR>(tomerge, C_m, C_n, true, true); // Delete input arrays and sort
        else if(computationKernel == 2) C_tuples = MultiwayMerge<SR>(tomerge, C_m, C_n, true); // Delete input arrays and sort
        
#ifdef TIMING
        t3 = MPI_Wtime();
        mcl3d_SUMMAmergetime += (t3-t2);
#endif

#ifdef TIMING 
        if(myrank == 0){
            fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tAbcast_time: %lf\n", p, Abcast_time);
            fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tBbcast_time: %lf\n", p, Bbcast_time);
            fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tLocal_multiplication_time: %lf\n", p, Local_multiplication_time);
            fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tSUMMA Merge time: %lf\n", p, (t3-t2));
        }
#endif
        /*
         *  SUMMA Ends
         * */
#ifdef TIMING
        t1 = MPI_Wtime();
        mcl3d_SUMMAtime += (t1-t0);
        if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tSUMMA time: %lf\n", p, (t1-t0));
#endif

        /*
         * 3d-reduction starts
         * */
#ifdef TIMING
        t0 = MPI_Wtime();
        t2 = MPI_Wtime();
#endif
        MPI_Datatype MPI_tuple;
        MPI_Type_contiguous(sizeof(std::tuple<LIC,LIC,NUO>), MPI_CHAR, &MPI_tuple);
        MPI_Type_commit(&MPI_tuple);
        
        /*
         *  Create a profile with information regarding data to be sent and received between layers 
         *  These memory allocation needs to be `int` specifically because some of these arrays would be used in communication
         *  This is requirement is for MPI as MPI_Alltoallv takes pointer to integer exclusively as count and displacement
         * */
        int * sendcnt    = new int[A.getcommgrid3D()->GetGridLayers()];
        int * sendprfl   = new int[A.getcommgrid3D()->GetGridLayers()*3];
        int * sdispls    = new int[A.getcommgrid3D()->GetGridLayers()]();
        int * recvcnt    = new int[A.getcommgrid3D()->GetGridLayers()];
        int * recvprfl   = new int[A.getcommgrid3D()->GetGridLayers()*3];
        int * rdispls    = new int[A.getcommgrid3D()->GetGridLayers()]();

        vector<LIC> lbDivisions3dPrefixSum(lbDivisions3d.size());
        lbDivisions3dPrefixSum[0] = 0;
        std::partial_sum(lbDivisions3d.begin(), lbDivisions3d.end()-1, lbDivisions3dPrefixSum.begin()+1);
        ColLexiCompare<LIC,NUO> comp;
        LIC totsend = C_tuples->getnnz();
#ifdef TIMING
        t3 = MPI_Wtime();
        if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tAllocation of alltoall information: %lf\n", p, (t3-t2));
#endif
        
#ifdef TIMING
        t2 = MPI_Wtime();
#endif
#pragma omp parallel for
        for(int i=0; i < A.getcommgrid3D()->GetGridLayers(); ++i){
            LIC start_col = lbDivisions3dPrefixSum[i];
            LIC end_col = lbDivisions3dPrefixSum[i] + lbDivisions3d[i];
            std::tuple<LIC, LIC, NUO> search_tuple_start(0, start_col, NUO());
            std::tuple<LIC, LIC, NUO> search_tuple_end(0, end_col, NUO());
            std::tuple<LIC, LIC, NUO>* start_it = std::lower_bound(C_tuples->tuples, C_tuples->tuples + C_tuples->getnnz(), search_tuple_start, comp);
            std::tuple<LIC, LIC, NUO>* end_it = std::lower_bound(C_tuples->tuples, C_tuples->tuples + C_tuples->getnnz(), search_tuple_end, comp);
            // This type casting is important from semantic point of view
            sendcnt[i] = (int)(end_it - start_it);
            sendprfl[i*3+0] = (int)(sendcnt[i]); // Number of nonzeros in ith chunk
            sendprfl[i*3+1] = (int)(A.GetLayerMat()->seqptr()->getnrow()); // Number of rows in ith chunk
            sendprfl[i*3+2] = (int)(lbDivisions3d[i]); // Number of columns in ith chunk
        }
        std::partial_sum(sendcnt, sendcnt+A.getcommgrid3D()->GetGridLayers()-1, sdispls+1);
#ifdef TIMING
        t3 = MPI_Wtime();
        if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tGetting Alltoall data ready: %lf\n", p, (t3-t2));
#endif

        // Send profile ready. Now need to update the tuples to reflect correct column id after column split.
#ifdef TIMING
        t2 = MPI_Wtime();
#endif
        for(int i=0; i < A.getcommgrid3D()->GetGridLayers(); ++i){
#pragma omp parallel for schedule(static)
            for(int j = 0; j < sendcnt[i]; j++){
                std::get<1>(C_tuples->tuples[sdispls[i]+j]) = std::get<1>(C_tuples->tuples[sdispls[i]+j]) - lbDivisions3dPrefixSum[i];
            }
        }
#ifdef TIMING
        t3 = MPI_Wtime();
        if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tGetting Alltoallv data ready: %lf\n", p, (t3-t2));
#endif

#ifdef TIMING
        t2 = MPI_Wtime();
#endif
        MPI_Alltoall(sendprfl, 3, MPI_INT, recvprfl, 3, MPI_INT, A.getcommgrid3D()->GetFiberWorld());
#ifdef TIMING
        t3 = MPI_Wtime();
        if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tAlltoall: %lf\n", p, (t3-t2));
#endif
#ifdef TIMING
        t2 = MPI_Wtime();
#endif
        for(int i = 0; i < A.getcommgrid3D()->GetGridLayers(); i++) recvcnt[i] = recvprfl[i*3];
        std::partial_sum(recvcnt, recvcnt+A.getcommgrid3D()->GetGridLayers()-1, rdispls+1);
        LIC totrecv = std::accumulate(recvcnt,recvcnt+A.getcommgrid3D()->GetGridLayers(), static_cast<IU>(0));
        std::tuple<LIC,LIC,NUO>* recvTuples = static_cast<std::tuple<LIC,LIC,NUO>*> (::operator new (sizeof(std::tuple<LIC,LIC,NUO>[totrecv])));
#ifdef TIMING
        t3 = MPI_Wtime();
        if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tAllocation of receive data: %lf\n", p, (t3-t2));
#endif

#ifdef TIMING
        t2 = MPI_Wtime();
#endif
        MPI_Alltoallv(C_tuples->tuples, sendcnt, sdispls, MPI_tuple, recvTuples, recvcnt, rdispls, MPI_tuple, A.getcommgrid3D()->GetFiberWorld());
        delete C_tuples;
#ifdef TIMING
        t3 = MPI_Wtime();
        if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tAlltoallv: %lf\n", p, (t3-t2));
#endif
#ifdef TIMING
        t2 = MPI_Wtime();
#endif
        vector<SpTuples<LIC, NUO>*> recvChunks(A.getcommgrid3D()->GetGridLayers());
#pragma omp parallel for
        for (int i = 0; i < A.getcommgrid3D()->GetGridLayers(); i++){
            recvChunks[i] = new SpTuples<LIC, NUO>(recvcnt[i], recvprfl[i*3+1], recvprfl[i*3+2], recvTuples + rdispls[i], true, false);
        }
#ifdef TIMING
        t3 = MPI_Wtime();
        if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\trecvChunks creation: %lf\n", p, (t3-t2));
#endif

#ifdef TIMING
        t2 = MPI_Wtime();
#endif
        // Free all memory except tempTuples; Because that is holding data of newly created local matrices after receiving.
        DeleteAll(sendcnt, sendprfl, sdispls);
        DeleteAll(recvcnt, recvprfl, rdispls); 
        MPI_Type_free(&MPI_tuple);
#ifdef TIMING
        t3 = MPI_Wtime();
        if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tMemory freeing: %lf\n", p, (t3-t2));
#endif
        /*
         * 3d-reduction ends 
         * */
        
#ifdef TIMING
        t1 = MPI_Wtime();
        mcl3d_reductiontime += (t1-t0);
        if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tReduction time: %lf\n", p, (t1-t0));
#endif
#ifdef TIMING
        t0 = MPI_Wtime();
#endif
        /*
         * 3d-merge starts 
         * */
        SpTuples<LIC, NUO> * merged_tuples;

        if(computationKernel == 1) merged_tuples = MultiwayMergeHash<SR, LIC, NUO>(recvChunks, recvChunks[0]->getnrow(), recvChunks[0]->getncol(), false, false); // Do not delete
        else if(computationKernel == 2) merged_tuples = MultiwayMerge<SR, LIC, NUO>(recvChunks, recvChunks[0]->getnrow(), recvChunks[0]->getncol(), false); // Do not delete
#ifdef TIMING
        t1 = MPI_Wtime();
        mcl3d_3dmergetime += (t1-t0);
        if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\t3D Merge time: %lf\n", p, (t1-t0));
#endif
        /*
         * 3d-merge ends
         * */
#ifdef TIMING
        t0 = MPI_Wtime();
#endif
        // Do not delete elements of recvChunks, because that would give segmentation fault due to double free
        ::operator delete(recvTuples);
        for(int i = 0; i < recvChunks.size(); i++){
            recvChunks[i]->tuples_deleted = true; // Temporary patch to avoid memory leak and segfault
            delete recvChunks[i]; // As the patch is used, now delete each element of recvChunks
        }
        vector<SpTuples<LIC,NUO>*>().swap(recvChunks); // As the patch is used, now delete recvChunks

        // This operation is not needed if result can be used and discareded right away
        // This operation is being done because it is needed by MCLPruneRecoverySelect
        UDERO * phaseResultant = new UDERO(*merged_tuples, false);
        delete merged_tuples;
        SpParMat<IU, NUO, UDERO> phaseResultantLayer(phaseResultant, A.getcommgrid3D()->GetLayerWorld());
        MCLPruneRecoverySelect(phaseResultantLayer, hardThreshold, selectNum, recoverNum, recoverPct, kselectVersion);
#ifdef TIMING
        t1 = MPI_Wtime();
        mcl3d_kselecttime += (t1-t0);
        if(myrank == 0) fprintf(stderr, "[MemEfficientSpGEMM3D]\tPhase: %d\tMCLPruneRecoverySelect time: %lf\n",p, (t1-t0));
#endif
        toconcatenate.push_back(phaseResultantLayer.seq());
#ifdef TIMING
        if(myrank == 0) fprintf(stderr, "***\n");
#endif
    }
    for(int i = 0; i < PiecesOfB.size(); i++) delete PiecesOfB[i];

    std::shared_ptr<CommGrid3D> grid3d;
    grid3d.reset(new CommGrid3D(A.getcommgrid3D()->GetWorld(), A.getcommgrid3D()->GetGridLayers(), A.getcommgrid3D()->GetGridRows(), A.getcommgrid3D()->GetGridCols(), A.isSpecial()));
    UDERO * localResultant = new UDERO(0, A.GetLayerMat()->seqptr()->getnrow(), divisions3d[A.getcommgrid3D()->GetRankInFiber()], 0);
    localResultant->ColConcatenate(toconcatenate);
    SpParMat3D<IU, NUO, UDERO> C3D(localResultant, grid3d, A.isColSplit(), A.isSpecial());
    return C3D;
}

}


#endif

