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

#include "CombBLAS/CombBLAS.h"
#include "Glue.h"   

namespace combblas {

template<typename IT, typename NT>
SpDCCols<IT,NT> * GenRMat(unsigned scale, unsigned EDGEFACTOR, double initiator[4], MPI_Comm & layerworld, bool scramble)
{
	double t01 = MPI_Wtime();
	double t02;
	
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

    float balance = A->LoadImbalance();
    ostringstream outs;
    outs << "Load balance: " << balance << endl;
    SpParHelper::Print(outs.str());
    
	return A->seqptr();
}

/**
 ** \param[out] splitmat {generated RMAT matrix, split into CMG.GridLayers pieces}
 **/
template <typename IT, typename NT>
void Generator(unsigned scale, unsigned EDGEFACTOR, double initiator[4], CCGrid & CMG, SpDCCols<IT,NT> & splitmat, bool trans, bool scramble)
{
    std::vector<IT> vecEss; // at layer_grid=0, this will have [CMG.GridLayers * SpDCCols<IT,NT>::esscount] entries
    std::vector< SpDCCols<IT, NT> > partsmat;    // only valid at layer_grid=0
    int nparts = CMG.GridLayers;
	if(CMG.layer_grid == 0)
	{
		SpDCCols<IT, NT> * localmat = GenRMat<IT,NT>(scale, EDGEFACTOR, initiator, CMG.layerWorld, scramble);
			
        double trans_beg = MPI_Wtime();
        if(trans) localmat->Transpose(); // locally transpose
        comp_trans += (MPI_Wtime() - trans_beg);

        double split_beg = MPI_Wtime();
        localmat->ColSplit(nparts, partsmat);     // split matrices are emplaced-back into partsmat vector, localmat destroyed

        for(int i=0; i< nparts; ++i)
        {
            std::vector<IT> ess = partsmat[i].GetEssentials();
            for(auto itr = ess.begin(); itr != ess.end(); ++itr)
            {
                vecEss.push_back(*itr);
            }
        }
        comp_split += (MPI_Wtime() - split_beg);
	}
    
    double scatter_beg = MPI_Wtime();   // timer on
    int esscnt = SpDCCols<IT,NT>::esscount; // necessary cast for MPI

    std::vector<IT> myess(esscnt);
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

}

#endif
