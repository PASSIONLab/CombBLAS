#ifndef _GEN_RMAT_DIST_H_
#define _GEN_RMAT_DIST_H_

#include <mpi.h>
#include <sys/time.h> 
#include <iostream>
#include <iomanip>
#include <functional>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>

// These macros should be defined before stdint.h is included
#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif
#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
#endif
#include <stdint.h>

#include "../CombBLAS.h"
#include "Glue.h"   


template <typename IT, typename NT>
SpDCCols<IT,NT> * ReadMat(string filename, CCGrid & CMG, bool trans, bool permute, FullyDistVec<IT, IT>& p)
{
    double t01 = MPI_Wtime();
    double t02;
    if(CMG.layer_grid == 0)
    {
        shared_ptr<CommGrid> layerGrid;
        layerGrid.reset( new CommGrid(CMG.layerWorld, 0, 0) );
        SpParMat < IT, NT, SpDCCols<IT,NT> > *A = new SpParMat < IT, NT, SpDCCols<IT,NT> >(layerGrid);
        A->ReadDistribute(filename, 0, false);
        
        SpParHelper::Print("Read input file\n");
        ostringstream tinfo;
        t02 = MPI_Wtime();
        tinfo << "Reader took " << t02-t01 << " seconds" << endl;
        SpParHelper::Print(tinfo.str());
        
        // random permutations for load balance
        if(permute)
        {
            if(A->getnrow() == A->getncol())
            {
                if(p.TotalLength()!=A->getnrow())
                {
                    SpParHelper::Print("Generating random permutation vector.\n");
                    p.iota(A->getnrow(), 0);
                    p.RandPerm();
                }
                (*A)(p,p,true);// in-place permute to save memory
            }
            else
            {
                SpParHelper::Print("nrow != ncol. Can not apply symmetric permutation.\n");
            }
        }
        
        

        double trans_beg = MPI_Wtime();
        if(trans) A->Transpose(); // locally transpose
        comp_trans += (MPI_Wtime() - trans_beg);
        
        float balance = A->LoadImbalance();
        ostringstream outs;
        outs << "Load balance: " << balance << endl;
        SpParHelper::Print(outs.str());
        
        
        return A->seqptr();

    }
    else
        return new SpDCCols<IT,NT>();
}

template<typename IT, typename NT>
SpDCCols<IT,NT> * GenMat(CCGrid & CMG, unsigned scale, unsigned EDGEFACTOR, double initiator[4], bool trans, bool scramble)
{
    double t01 = MPI_Wtime();
    double t02;

    if(CMG.layer_grid == 0)
    {
         shared_ptr<CommGrid> layerGrid;
        layerGrid.reset( new CommGrid(CMG.layerWorld, 0, 0) );
        
        DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>(layerworld);
        
        ostringstream minfo;
        int nprocs = DEL->commGrid->GetSize();
        minfo << "Started Generation of scale "<< scale << endl;
        minfo << "Using " << nprocs << " MPI processes" << endl;
        SpParHelper::Print(minfo.str());
        
        DEL->GenGraph500Data(initiator, scale, EDGEFACTOR, scramble, false );
        // don't generate packed edges, that function uses MPI_COMM_WORLD which can not be used in a single layer!
        
        SpParHelper::Print("Generated renamed edge lists\n");
        ostringstream tinfo;
        t02 = MPI_Wtime();
        tinfo << "Generation took " << t02-t01 << " seconds" << endl;
        SpParHelper::Print(tinfo.str());
        
        SpParMat < IT, NT, SpDCCols<IT,NT> > *A = new SpParMat < IT, NT, SpDCCols<IT,NT> >(*DEL, false);
        
        delete DEL;
        SpParHelper::Print("Created Sparse Matrix\n");
        
        double trans_beg = MPI_Wtime();
        if(trans) A->Transpose(); // locally transpose
        comp_trans += (MPI_Wtime() - trans_beg);
        
        float balance = A->LoadImbalance();
        ostringstream outs;
        outs << "Load balance: " << balance << endl;
        SpParHelper::Print(outs.str());
        
        return A->seqptr();
    }
    else
        return new SpDCCols<IT,NT>();
}

/**
 ** \param[out] splitmat {split a matrix from layer 0 into CMG.GridLayers pieces}
 **/
template <typename IT, typename NT>
void SplitMat(CCGrid & CMG, SpDCCols<IT, NT> * localmat, SpDCCols<IT,NT> & splitmat)
{
    vector<IT> vecEss; // at layer_grid=0, this will have [CMG.GridLayers * SpDCCols<IT,NT>::esscount] entries
    vector< SpDCCols<IT, NT> > partsmat;    // only valid at layer_grid=0
    int nparts = CMG.GridLayers;
	if(CMG.layer_grid == 0)
	{
        double split_beg = MPI_Wtime();
        localmat->ColSplit(nparts, partsmat);     // split matrices are emplaced-back into partsmat vector, localmat destroyed

        for(int i=0; i< nparts; ++i)
        {
            vector<IT> ess = partsmat[i].GetEssentials();
            for(auto itr = ess.begin(); itr != ess.end(); ++itr)
            {
                vecEss.push_back(*itr);
            }
        }
        comp_split += (MPI_Wtime() - split_beg);
	}
    
    double scatter_beg = MPI_Wtime();   // timer on
    int esscnt = SpDCCols<IT,NT>::esscount; // necessary cast for MPI

    vector<IT> myess(esscnt);
    MPI_Scatter(vecEss.data(), esscnt, MPIType<IT>(), myess.data(), esscnt, MPIType<IT>(), 0, CMG.fiberWorld);
    
    if(CMG.layer_grid == 0) // senders
    {
        splitmat = partsmat[0]; // just copy the local split
        for(int recipient=1; recipient< nparts; ++recipient)    // scatter the others
        {
            int tag = 0;
            Arr<IT,NT> arrinfo = partsmat[recipient].GetArrays();
            for(unsigned int i=0; i< arrinfo.indarrs.size(); ++i)	// get index arrays
            {
                // MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
                MPI_Send(arrinfo.indarrs[i].addr, arrinfo.indarrs[i].count, MPIType<IT>(), recipient, tag++, CMG.fiberWorld);
            }
            for(unsigned int i=0; i< arrinfo.numarrs.size(); ++i)	// get numerical arrays
            {
                MPI_Send(arrinfo.numarrs[i].addr, arrinfo.numarrs[i].count, MPIType<NT>(), recipient, tag++, CMG.fiberWorld);
            }
        }
    }
    else // receivers
    {
        splitmat.Create(myess);		// allocate memory for arrays
        Arr<IT,NT> arrinfo = splitmat.GetArrays();

        int tag = 0;
        for(unsigned int i=0; i< arrinfo.indarrs.size(); ++i)	// get index arrays
        {
            MPI_Recv(arrinfo.indarrs[i].addr, arrinfo.indarrs[i].count, MPIType<IT>(), 0, tag++, CMG.fiberWorld, MPI_STATUS_IGNORE);
        }
        for(unsigned int i=0; i< arrinfo.numarrs.size(); ++i)	// get numerical arrays
        {
            MPI_Recv(arrinfo.numarrs[i].addr, arrinfo.numarrs[i].count, MPIType<NT>(), 0, tag++, CMG.fiberWorld, MPI_STATUS_IGNORE);
        }
    }
    comm_split += (MPI_Wtime() - scatter_beg);
}

#endif
