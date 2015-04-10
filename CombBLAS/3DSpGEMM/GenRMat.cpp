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
#include "../SpParHelper.h"
#include "../DistEdgeList.h"
#include "Glue.h"

#ifdef BUPC
extern "C" void * VoidRMat(unsigned scale, unsigned EDGEFACTOR, double initiator[4], int layer_grid, int rankinlayer, void ** part1, void ** part2, bool trans);
#endif

extern void Split_GetEssensials(void * full, void ** part1, void ** part2, SpDCCol_Essentials * sess1, SpDCCol_Essentials * sess2);


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

SpDCCols<int32_t, double> * GenRMat(unsigned scale, unsigned EDGEFACTOR, double initiator[4], MPI_Comm & layerworld)
{
	double t01 = MPI_Wtime();
	double t02;
	
	DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>(layerworld);

	ostringstream minfo;
    int nprocs;
    MPI_Comm_size(DEL->commGrid->GetWorld(),&nprocs);
	minfo << "Started Generation of scale "<< scale << endl;
	minfo << "Using " << nprocs << " MPI processes" << endl;
	SpParHelper::Print(minfo.str());

	DEL->GenGraph500Data(initiator, scale, EDGEFACTOR, true, true );	// generate packed edges
	SpParHelper::Print("Generated renamed edge lists\n");
	t02 = MPI_Wtime();
	ostringstream tinfo;
	tinfo << "Generation took " << t02-t01 << " seconds" << endl;
	SpParHelper::Print(tinfo.str());

	SpDCCols<int32_t,double> * LocalSpMat;
	MakeDCSC< SpDCCols<int32_t,double> > (*DEL, false, &LocalSpMat);
	delete DEL;     // free memory before symmetricizing
	SpParHelper::Print("Created Sparse Matrix (with int32 local indices and values)\n");
	return LocalSpMat;
}

void LocalTranpose(void * matrix)
{
	static_cast< SpDCCols<int32_t, double> *>(matrix)->Transpose();
}

void * VoidRMat(unsigned scale, unsigned EDGEFACTOR, double initiator[4], int layer_grid, int rankinlayer, void ** part1, void ** part2, bool trans)
{
    
    // MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm)
    MPI_Comm layerWorld;
    MPI_Comm fiberWorld;
    MPI_Comm_split(MPI_COMM_WORLD, layer_grid, rankinlayer,&layerWorld);
    MPI_Comm_split(MPI_COMM_WORLD, rankinlayer, layer_grid,&fiberWorld);

	
	typedef SpDCCols<int32_t, double> LOC_SPMAT;

	LOC_SPMAT * localmat;
    
    MPI_Datatype esstype;
    MPI_Type_contiguous(sizeof(SpDCCol_Essentials), MPI_CHAR, &esstype );
    MPI_Type_commit(&esstype);

	SpDCCol_Essentials * sess1 = malloc(sizeof(SpDCCol_Essentials));	
	SpDCCol_Essentials * sess2 = malloc(sizeof(SpDCCol_Essentials));	

	LOC_SPMAT *A1, *A2;
	if(layer_grid == 0)
	{	
		#ifdef DEBUG	
		cout << MPI::COMM_WORLD.Get_rank() << " maps to " << layerWorld.Get_rank() << endl;
		#endif

		localmat = GenRMat(scale, EDGEFACTOR, initiator, layerWorld);
		
		#ifdef DEBUG	
		if(layerWorld.Get_rank() == 0)
		{
			cout << "Before transpose\n";
			localmat->PrintInfo();
			ofstream before("pre_trans.txt");
			localmat->put(before);
			before.close();
		}
		#endif
			
		// Timer start here
		if(trans)
			localmat->Transpose(); // locally transpose
		
		#ifdef DEBUG	
		if(layerWorld.Get_rank() == 0 && trans)
		{
			cout << "After transpose\n";
			localmat->PrintInfo();
			ofstream after("post_trans.txt");
			localmat->put(after);
			after.close();
		}
		#endif

			
		Split_GetEssensials(localmat, &A1, &A2, sess1, sess2);
		// Timer end here
		// Reduce timer on layerWorld and report as "local splitting and transposition time"
	}
	MPI_Bcast(sess1, 1, esstype, 0, fiberWorld);
	MPI_Bcast(sess2, 1, esstype, 0, fiberWorld);
	
	
	vector<int32_t> ess1(LOC_SPMAT::esscount);
	vector<int32_t> ess2(LOC_SPMAT::esscount);
	
	if (layer_grid != 0) 
	{
		A1 = new LOC_SPMAT();
		A2 = new LOC_SPMAT();
		
		ess1[0] = sess1->nnz;
		ess1[1] = sess1->m;
		ess1[2] = sess1->n;
		ess1[3] = sess1->nzc;

		ess2[0] = sess2->nnz;
		ess2[1] = sess2->m;
		ess2[2] = sess2->n;
		ess2[3] = sess2->nzc;
	}
    int fprocs, myrank;
    MPI_Comm_size(fiberWorld,&fprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

	if(fprocs > 1 && myrank == 1)
	{
		cout << "Dude #1 "<< endl;
		copy(ess1.begin(), ess1.end(), ostream_iterator<int32_t>(cout, " ")); cout << endl;
		copy(ess2.begin(), ess2.end(), ostream_iterator<int32_t>(cout, " ")); cout << endl;

	}
	// Start timer here
	SpParHelper::BCastMatrix(fiberWorld, *A1, ess1, 0);		// ess is not used at root
	SpParHelper::BCastMatrix(fiberWorld, *A2, ess2, 0);		// ess is not used at root
	*part1 = (void*) A1;
	*part2 = (void*) A2;
	// Timer end here
	// Reduce timer on everyone and report as "replication time"
	
	if(layer_grid == 0)
	{
		int64_t local_A1_nnz = A1->getnnz();
		int64_t local_A2_nnz = A2->getnnz();

		int64_t global_A1_nnz = 0, global_A2_nnz = 0;
        MPI_Reduce(&local_A1_nnz, &global_A1_nnz, 1, MPIType<int64_t>(), MPI_SUM, 0, layerWorld);
		MPI_Reduce(&local_A2_nnz, &global_A2_nnz, 1, MPIType<int64_t>(), MPI_SUM, 0, layerWorld);
	
        int layerrank;
        MPI_Comm_rank(layerWorld,&layerrank);
        
		if(layerrank == 0)
		{
			cout << "Global nonzeros in A1 is " << global_A1_nnz << endl;
			cout << "Global nonzeros in A2 is " << global_A2_nnz << endl;
		}
	}
	
	MPI_Comm_free(&layerWorld);
	MPI_Comm_free(&fiberWorld);
}
