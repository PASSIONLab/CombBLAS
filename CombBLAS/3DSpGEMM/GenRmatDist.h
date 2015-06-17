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


template <class DER, class DELIT>
void MakeDCSC (const DistEdgeList<DELIT> & DEL, bool removeloops, DER ** spSeq)
{
	shared_ptr<CommGrid> commGrid = DEL.commGrid;		
	typedef typename DER::LocalIT LIT;
	typedef typename DER::LocalNT NT;

	int nprocs = commGrid->GetSize();
	int r = commGrid->GetGridRows();
	int s = commGrid->GetGridCols();

	ostringstream outs;
	outs << "MakeDCSC on an " << r << "-by-" << s << " grid" << endl;
	SpParHelper::Print(outs.str());
	
	vector< vector<LIT> > data(nprocs);	// enties are pre-converted to local indices before getting pushed into "data"

	LIT m_perproc = DEL.getGlobalV() / r;
	LIT n_perproc = DEL.getGlobalV() / s;

	if(sizeof(LIT) < sizeof(DELIT))
	{
		ostringstream outs;
		outs << "Warning: Using smaller indices for the matrix than DistEdgeList\n";
		outs << "Local matrices are " << m_perproc << "-by-" << n_perproc << endl;
		SpParHelper::Print(outs.str());
	}	
	
	// to lower memory consumption, form sparse matrix in stages
	LIT stages = MEM_EFFICIENT_STAGES;	
	
	// even if local indices (LIT) are 32-bits, we should work with 64-bits for global info
	int64_t perstage = DEL.getNumLocalEdges() / stages;
	LIT totrecv = 0;
	vector<LIT> alledges;

	int maxr = r-1;
	int maxs = s-1;	
	for(LIT s=0; s< stages; ++s)
	{
		int64_t n_befor = s*perstage;
		int64_t n_after= ((s==(stages-1))? DEL.getNumLocalEdges() : ((s+1)*perstage));

		// clear the source vertex by setting it to -1
		int realedges = 0;	// these are "local" realedges

		if(DEL.getPackedEdges())
		{
			for (int64_t i = n_befor; i < n_after; i++)
			{
				int64_t fr = get_v0_from_edge(&(DEL.getPackedEdges()[i]));
				int64_t to = get_v1_from_edge(&(DEL.getPackedEdges()[i]));

				if(fr >= 0 && to >= 0)	// otherwise skip
				{
					int rowowner = min(static_cast<int>(fr / m_perproc), maxr);
					int colowner = min(static_cast<int>(to / n_perproc), maxs); 
					int owner = commGrid->GetRank(rowowner, colowner);
					LIT rowid = fr - (rowowner * m_perproc);	
					LIT colid = to - (colowner * n_perproc);
					data[owner].push_back(rowid);	// row_id
					data[owner].push_back(colid);	// col_id
					++realedges;
				}
			}
		}
		else
		{
			for (int64_t i = n_befor; i < n_after; i++)
			{
				if(DEL.getEdges()[2*i+0] >= 0 && DEL.getEdges()[2*i+1] >= 0)	// otherwise skip
				{
					int rowowner = min(static_cast<int>(DEL.getEdges()[2*i+0] / m_perproc), maxr);
					int colowner = min(static_cast<int>(DEL.getEdges()[2*i+1] / n_perproc), maxs);
					int owner = commGrid->GetRank(rowowner, colowner);
					LIT rowid = DEL.getEdges()[2*i+0]- (rowowner * m_perproc);
					LIT colid = DEL.getEdges()[2*i+1]- (colowner * n_perproc);
					data[owner].push_back(rowid);	
					data[owner].push_back(colid);
					++realedges;
				}
			}
		}

  		LIT * sendbuf = new LIT[2*realedges];
		int * sendcnt = new int[nprocs];
		int * sdispls = new int[nprocs];
		for(int i=0; i<nprocs; ++i)
			sendcnt[i] = data[i].size();

		int * rdispls = new int[nprocs];
		int * recvcnt = new int[nprocs];

		MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, commGrid->GetWorld()); // share the counts

		sdispls[0] = 0;
		rdispls[0] = 0;
		for(int i=0; i<nprocs-1; ++i)
		{
			sdispls[i+1] = sdispls[i] + sendcnt[i];
			rdispls[i+1] = rdispls[i] + recvcnt[i];
		}
		for(int i=0; i<nprocs; ++i)
			copy(data[i].begin(), data[i].end(), sendbuf+sdispls[i]);

		// clear memory
		for(int i=0; i<nprocs; ++i)
			vector<LIT>().swap(data[i]);

		// ABAB: Total number of edges received might not be LIT-addressible
		// However, each edge_id is LIT-addressible
		int64_t thisrecv = accumulate(recvcnt,recvcnt+nprocs, static_cast<int64_t>(0));	// thisrecv = 2*locedges
		LIT * recvbuf = new LIT[thisrecv];
		totrecv += thisrecv;
			
		MPI_Alltoallv(sendbuf, sendcnt, sdispls, MPIType<LIT>(), recvbuf, recvcnt, rdispls, MPIType<LIT>(), commGrid->GetWorld());
		DeleteAll(sendcnt, recvcnt, sdispls, rdispls,sendbuf);
		copy (recvbuf,recvbuf+thisrecv,back_inserter(alledges));	// copy to all edges
		delete [] recvbuf;
	}

	int myprocrow = commGrid->GetRankInProcCol();
	int myproccol = commGrid->GetRankInProcRow();
	LIT locrows, loccols; 
	if(myprocrow != r-1)	locrows = m_perproc;
	else 	locrows = DEL.getGlobalV() - myprocrow * m_perproc;
	if(myproccol != s-1)	loccols = n_perproc;
	else	loccols = DEL.getGlobalV() - myproccol * n_perproc;

  	SpTuples<LIT,NT> A(totrecv/2, locrows, loccols, alledges, removeloops);  	// alledges is empty upon return
    
  	*spSeq = new DER(A,false);        // Convert SpTuples to DER

}

template<typename IT, typename NT>
SpDCCols<IT,NT> * GenRMat(unsigned scale, unsigned EDGEFACTOR, double initiator[4], MPI_Comm & layerworld)
{
	double t01 = MPI_Wtime();
	double t02;
	
	DistEdgeList<IT> * DEL = new DistEdgeList<IT>(layerworld);

	ostringstream minfo;
    int nprocs;
    MPI_Comm_size(DEL->commGrid->GetWorld(),&nprocs);
	minfo << "Started Generation of scale "<< scale << endl;
	minfo << "Using " << nprocs << " MPI processes" << endl;
	SpParHelper::Print(minfo.str());

	DEL->GenGraph500Data(initiator, scale, EDGEFACTOR, true, false );	// don't generate packed edges, that function uses MPI_COMM_WORLD which can not be used in a single layer!
	SpParHelper::Print("Generated renamed edge lists\n");
	ostringstream tinfo;
    t02 = MPI_Wtime();
	tinfo << "Generation took " << t02-t01 << " seconds" << endl;
	SpParHelper::Print(tinfo.str());

	SpDCCols<IT,NT> * LocalSpMat;
	MakeDCSC< SpDCCols<IT,NT> > (*DEL, false, &LocalSpMat);
	delete DEL;     // free memory before symmetricizing
	SpParHelper::Print("Created Sparse Matrix (with int32 local indices and values)\n");
	return LocalSpMat;
}

/**
 ** \param[out] splitmat {generated RMAT matrix, split into CMG.GridLayers pieces}
 **/
template <typename IT, typename NT>
void Generator(unsigned scale, unsigned EDGEFACTOR, double initiator[4], CCGrid & CMG, SpDCCols<IT,NT> & splitmat, bool trans)
{
    vector<IT> vecEss; // at layer_grid=0, this will have [CMG.GridLayers * SpDCCols<IT,NT>::esscount] entries
    vector< SpDCCols<IT, NT> > partsmat;    // only valid at layer_grid=0
    int nparts = CMG.GridLayers;
	if(CMG.layer_grid == 0)
	{
		SpDCCols<IT, NT> * localmat = GenRMat<IT,NT>(scale, EDGEFACTOR, initiator, CMG.layerWorld);
			
        double trans_beg = MPI_Wtime();
        if(trans) localmat->Transpose(); // locally transpose
        comp_trans += (MPI_Wtime() - trans_beg);

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
    vector<IT> myess(SpDCCols<IT,NT>::esscount);
	MPI_Scatter(vecEss.data(), esscnt, MPIType<IT>(), myess.data(), esscnt, MPIType<IT>(), 0, CMG.fiberWorld);
    
    if(CMG.layer_grid == 0) // senders
    {
        splitmat = partsmat[0]; // just copy the local split
        for(int i=1; i< nparts; ++i)    // scatter the others
        {
            int tag = 0;
            Arr<IT,NT> arrinfo = partsmat[i].GetArrays();
            for(unsigned int i=0; i< arrinfo.indarrs.size(); ++i)	// get index arrays
            {
                // MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
                MPI_Send(arrinfo.indarrs[i].addr, arrinfo.indarrs[i].count, MPIType<IT>(), i, tag++, CMG.fiberWorld);
            }
            for(unsigned int i=0; i< arrinfo.numarrs.size(); ++i)	// get numerical arrays
            {
                MPI_Send(arrinfo.numarrs[i].addr, arrinfo.numarrs[i].count, MPIType<NT>(), i, tag++, CMG.fiberWorld);
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
