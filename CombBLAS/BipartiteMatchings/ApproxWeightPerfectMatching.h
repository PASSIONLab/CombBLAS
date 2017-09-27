//
//  ApproxWeightPerfectMatching.h
//  
//
//  Created by Ariful Azad on 8/22/17.
//
//

#ifndef ApproxWeightPerfectMatching_h
#define ApproxWeightPerfectMatching_h



template <class IT, class NT>
vector<tuple<IT,IT,NT>> ExchangeData(vector<vector<tuple<IT,IT,NT>>> & tempTuples, MPI_Comm World)
{
	
	/* Create/allocate variables for vector assignment */
	MPI_Datatype MPI_tuple;
	MPI_Type_contiguous(sizeof(tuple<IT,IT,NT>), MPI_CHAR, &MPI_tuple);
	MPI_Type_commit(&MPI_tuple);
	
	int nprocs;
	MPI_Comm_size(World, &nprocs);
	
	int * sendcnt = new int[nprocs];
	int * recvcnt = new int[nprocs];
	int * sdispls = new int[nprocs]();
	int * rdispls = new int[nprocs]();
	
	// Set the newly found vector entries
	IT totsend = 0;
	for(IT i=0; i<nprocs; ++i)
	{
		sendcnt[i] = tempTuples[i].size();
		totsend += tempTuples[i].size();
	}
	
	MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, World);
	
	partial_sum(sendcnt, sendcnt+nprocs-1, sdispls+1);
	partial_sum(recvcnt, recvcnt+nprocs-1, rdispls+1);
	IT totrecv = accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));
	
	
	vector< tuple<IT,IT,NT> > sendTuples(totsend);
	for(int i=0; i<nprocs; ++i)
	{
		copy(tempTuples[i].begin(), tempTuples[i].end(), sendTuples.data()+sdispls[i]);
		vector< tuple<IT,IT,NT> >().swap(tempTuples[i]);	// clear memory
	}
	vector< tuple<IT,IT,NT> > recvTuples(totrecv);
	MPI_Alltoallv(sendTuples.data(), sendcnt, sdispls, MPI_tuple, recvTuples.data(), recvcnt, rdispls, MPI_tuple, World);
	DeleteAll(sendcnt, recvcnt, sdispls, rdispls); // free all memory
	MPI_Type_free(&MPI_tuple);
	return recvTuples;
	
}



template <class IT, class NT>
vector<tuple<IT,IT,IT,NT>> ExchangeData1(vector<vector<tuple<IT,IT,IT,NT>>> & tempTuples, MPI_Comm World)
{
	
	/* Create/allocate variables for vector assignment */
	MPI_Datatype MPI_tuple;
	MPI_Type_contiguous(sizeof(tuple<IT,IT,IT,NT>), MPI_CHAR, &MPI_tuple);
	MPI_Type_commit(&MPI_tuple);
	
	int nprocs;
	MPI_Comm_size(World, &nprocs);
	
	int * sendcnt = new int[nprocs];
	int * recvcnt = new int[nprocs];
	int * sdispls = new int[nprocs]();
	int * rdispls = new int[nprocs]();
	
	// Set the newly found vector entries
	IT totsend = 0;
	for(IT i=0; i<nprocs; ++i)
	{
		sendcnt[i] = tempTuples[i].size();
		totsend += tempTuples[i].size();
	}
	
	MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, World);
	
	partial_sum(sendcnt, sendcnt+nprocs-1, sdispls+1);
	partial_sum(recvcnt, recvcnt+nprocs-1, rdispls+1);
	IT totrecv = accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));
	
	vector< tuple<IT,IT,IT,NT> > sendTuples(totsend);
	for(int i=0; i<nprocs; ++i)
	{
		copy(tempTuples[i].begin(), tempTuples[i].end(), sendTuples.data()+sdispls[i]);
		vector< tuple<IT,IT,IT,NT> >().swap(tempTuples[i]);	// clear memory
	}
	vector< tuple<IT,IT,IT,NT> > recvTuples(totrecv);
	MPI_Alltoallv(sendTuples.data(), sendcnt, sdispls, MPI_tuple, recvTuples.data(), recvcnt, rdispls, MPI_tuple, World);
	DeleteAll(sendcnt, recvcnt, sdispls, rdispls); // free all memory
	MPI_Type_free(&MPI_tuple);
	return recvTuples;
}





template <class IT, class NT,class DER>
int OwnerProcs(SpParMat < IT, NT, DER > & A, IT grow, IT gcol, IT nrows, IT ncols)
{
	
	auto commGrid = A.getcommgrid();
	int procrows = commGrid->GetGridRows();
	int proccols = commGrid->GetGridCols();
	// remember that getnrow() and getncol() require collectives
	// Hence, we save them once and pass them to this function
	IT m_perproc = nrows / procrows;
	IT n_perproc = ncols / proccols;
	int pr, pc;
	if(m_perproc != 0)
		pr = std::min(static_cast<int>(grow / m_perproc), procrows-1);
	else	// all owned by the last processor row
		pr = procrows -1;
	if(n_perproc != 0)
		pc = std::min(static_cast<int>(gcol / n_perproc), proccols-1);
	else
		pc = proccols-1;
	if(grow > nrows)
	{
		cout << "grow > nrow: " << grow << " "<< nrows << endl;
		exit(1);
	}
	return commGrid->GetRank(pr, pc);
}



template <class IT>
vector<tuple<IT,IT>> MateBcast(vector<tuple<IT,IT>> sendTuples, MPI_Comm World)
{
	
	/* Create/allocate variables for vector assignment */
	MPI_Datatype MPI_tuple;
	MPI_Type_contiguous(sizeof(tuple<IT,IT>) , MPI_CHAR, &MPI_tuple);
	MPI_Type_commit(&MPI_tuple);
	
	
	int nprocs;
	MPI_Comm_size(World, &nprocs);
	
	int * recvcnt = new int[nprocs];
	int * rdispls = new int[nprocs]();
	int sendcnt  = sendTuples.size();
	
	
	MPI_Allgather(&sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, World);
	
	partial_sum(recvcnt, recvcnt+nprocs-1, rdispls+1);
	IT totrecv = accumulate(recvcnt,recvcnt+nprocs, static_cast<IT>(0));
	
	vector< tuple<IT,IT> > recvTuples(totrecv);
	
	
	MPI_Allgatherv(sendTuples.data(), sendcnt, MPI_tuple,
				   recvTuples.data(), recvcnt, rdispls,MPI_tuple,World );
	
	DeleteAll(recvcnt, rdispls); // free all memory
	MPI_Type_free(&MPI_tuple);
	return recvTuples;
	
}


// -----------------------------------------------------------
// replicate weights of mates
// Can be improved by removing AllReduce by All2All
// -----------------------------------------------------------

template <class IT, class NT,class DER>
void ReplicateMateWeights( SpParMat < IT, NT, DER > & A, vector<IT>& RepMateC2R, vector<NT>& RepMateWR2C, vector<NT>& RepMateWC2R, IT nrows, IT ncols)
{
	
	
	fill(RepMateWC2R.begin(), RepMateWC2R.end(), static_cast<NT>(0));
	fill(RepMateWR2C.begin(), RepMateWR2C.end(), static_cast<NT>(0));
	
	
	auto commGrid = A.getcommgrid();
	MPI_Comm World = commGrid->GetWorld();
	MPI_Comm ColWorld = commGrid->GetColWorld();
	MPI_Comm RowWorld = commGrid->GetRowWorld();
	int nprocs = commGrid->GetSize();
	int pr = commGrid->GetGridRows();
	int pc = commGrid->GetGridCols();
	int rowrank = commGrid->GetRankInProcRow();
	int colrank = commGrid->GetRankInProcCol();
	int diagneigh = commGrid->GetComplementRank();
	
	//Information about the matrix distribution
	//Assume that A is an nrow x ncol matrix
	//The local submatrix is an lnrow x lncol matrix
	IT m_perproc = nrows / pr;
	IT n_perproc = ncols / pc;
	DER* spSeq = A.seqptr(); // local submatrix
	IT lnrow = spSeq->getnrow();
	IT lncol = spSeq->getncol();
	IT localRowStart = colrank * m_perproc; // first row in this process
	IT localColStart = rowrank * n_perproc; // first col in this process
	
	
	for(auto colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit) // iterate over columns
	{
		IT lj = colit.colid(); // local numbering
		IT mj = RepMateC2R[lj]; // mate of j
		if(mj >= localRowStart && mj < (localRowStart+lnrow) )
		{
			
			for(auto nzit = spSeq->begnz(colit); nzit < spSeq->endnz(colit); ++nzit)
			{
				
				IT li = nzit.rowid();
				IT i = li + localRowStart;
				// TODO: use binary search to directly go to mj-th entry if more than 32 nonzero in this column
				if( i == mj)
				{
					RepMateWC2R[lj] = nzit.value();
					RepMateWR2C[mj-localRowStart] = nzit.value();
				}
			}
			/*
			 if(RepMateWC2R[lj]==0)
			 {
			 cout << " !!!!Error: " << lj << " " << mj << endl;
			 for(auto nzit = spSeq->begnz(colit); nzit < spSeq->endnz(colit); ++nzit)
			 {
			 
			 IT li = nzit.rowid();
			 IT i = li + localRowStart;
			 cout << i << " ";
			 // TODO: use binary search to directly go to mj-th entry if more than 32 nonzero in this column
			 
			 }
			 exit(1);
			 }
			 */
		}
		
	}
	MPI_Allreduce(MPI_IN_PLACE, RepMateWC2R.data(), RepMateWC2R.size(), MPIType<NT>(), MPI_SUM, ColWorld);
	MPI_Allreduce(MPI_IN_PLACE, RepMateWR2C.data(), RepMateWR2C.size(), MPIType<NT>(), MPI_SUM, RowWorld);
}




template <class IT, class NT,class DER>
void Trace( SpParMat < IT, NT, DER > & A)
{
	
	IT nrows = A.getnrow();
	IT ncols = A.getncol();
	auto commGrid = A.getcommgrid();
	MPI_Comm World = commGrid->GetWorld();
	int myrank=commGrid->GetRank();
	int pr = commGrid->GetGridRows();
	int pc = commGrid->GetGridCols();
	
	
	//Information about the matrix distribution
	//Assume that A is an nrow x ncol matrix
	//The local submatrix is an lnrow x lncol matrix
	int rowrank = commGrid->GetRankInProcRow();
	int colrank = commGrid->GetRankInProcCol();
	IT m_perproc = nrows / pr;
	IT n_perproc = ncols / pc;
	DER* spSeq = A.seqptr(); // local submatrix
	IT localRowStart = colrank * m_perproc; // first row in this process
	IT localColStart = rowrank * n_perproc; // first col in this process
	
	
	IT trnnz = 0;
	NT trace = 0.0;
	for(auto colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit) // iterate over columns
	{
		IT lj = colit.colid(); // local numbering
		IT j = lj + localColStart;
		
		for(auto nzit = spSeq->begnz(colit); nzit < spSeq->endnz(colit); ++nzit)
		{
			
			IT li = nzit.rowid();
			IT i = li + localRowStart;
			if( i == j)
			{
				trnnz ++;
				trace += nzit.value();
				
			}
		}
		
	}
	MPI_Allreduce(MPI_IN_PLACE, &trnnz, 1, MPIType<IT>(), MPI_SUM, World);
	MPI_Allreduce(MPI_IN_PLACE, &trace, 1, MPIType<NT>(), MPI_SUM, World);
	
	if(myrank==0)
		cout <<"nrows: " << nrows << " Nnz in the diag: " << trnnz << " sum of diag: " << trace << endl;
	
}


template <class NT>
NT MatchingWeight( vector<NT>& RepMateWC2R, MPI_Comm RowWorld, NT& minw)
{
	NT w = 0;
	minw = 99999999999999.0;
	for(int i=0; i<RepMateWC2R.size(); i++)
	{
		//w += fabs(RepMateWC2R[i]);
		//w += exp(RepMateWC2R[i]);
		//minw = min(minw, exp(RepMateWC2R[i]));
		
		w += RepMateWC2R[i];
		minw = min(minw, RepMateWC2R[i]);
	}
	
	MPI_Allreduce(MPI_IN_PLACE, &w, 1, MPIType<NT>(), MPI_SUM, RowWorld);
	MPI_Allreduce(MPI_IN_PLACE, &minw, 1, MPIType<NT>(), MPI_MIN, RowWorld);
	return w;
}





// update the distributed mate vectors from replicated mate vectors
template <class IT>
void UpdateMatching(FullyDistVec<IT, IT>& mateRow2Col, FullyDistVec<IT, IT>& mateCol2Row, vector<IT>& RepMateR2C, vector<IT>& RepMateC2R)
{
	
	auto commGrid = mateRow2Col.getcommgrid();
	MPI_Comm RowWorld = commGrid->GetRowWorld();
	int rowroot = commGrid->GetDiagOfProcRow();
	int pc = commGrid->GetGridCols();
	
	// mateRow2Col is easy
	IT localLenR2C = mateRow2Col.LocArrSize();
	//IT* localR2C = mateRow2Col.GetLocArr();
	for(IT i=0, j = mateRow2Col.RowLenUntil(); i<localLenR2C; i++, j++)
	{
		mateRow2Col.SetLocalElement(i, RepMateR2C[j]);
		//localR2C[i] = RepMateR2C[j];
	}
	
	
	// mateCol2Row requires communication
	vector <int> sendcnts(pc);
	vector <int> dpls(pc);
	dpls[0] = 0;
	for(int i=1; i<pc; i++)
	{
		dpls[i] = mateCol2Row.RowLenUntil(i);
		sendcnts[i-1] = dpls[i] - dpls[i-1];
	}
	sendcnts[pc-1] = RepMateC2R.size() - dpls[pc-1];
	
	IT localLenC2R = mateCol2Row.LocArrSize();
	IT* localC2R = mateCol2Row.GetLocArr();
	MPI_Scatterv(RepMateC2R.data(),sendcnts.data(), dpls.data(), MPIType<IT>(), localC2R, localLenC2R, MPIType<IT>(),rowroot, RowWorld);
}




template <class IT, class NT, class DER>
void TwoThirdApprox(SpParMat < IT, NT, DER > & A, FullyDistVec<IT, IT>& mateRow2Col, FullyDistVec<IT, IT>& mateCol2Row)
{
	
	// Information about CommGrid and matrix layout
	// Assume that processes are laid in (pr x pc) process grid
	auto commGrid = A.getcommgrid();
	int myrank=commGrid->GetRank();
	MPI_Comm World = commGrid->GetWorld();
	MPI_Comm ColWorld = commGrid->GetColWorld();
	MPI_Comm RowWorld = commGrid->GetRowWorld();
	int nprocs = commGrid->GetSize();
	int pr = commGrid->GetGridRows();
	int pc = commGrid->GetGridCols();
	int rowrank = commGrid->GetRankInProcRow();
	int colrank = commGrid->GetRankInProcCol();
	int diagneigh = commGrid->GetComplementRank();
	
	//Information about the matrix distribution
	//Assume that A is an nrow x ncol matrix
	//The local submatrix is an lnrow x lncol matrix
	IT nrows = A.getnrow();
	IT ncols = A.getncol();
	IT m_perproc = nrows / pr;
	IT n_perproc = ncols / pc;
	DER* spSeq = A.seqptr(); // local submatrix
	Dcsc<IT, NT>* dcsc = spSeq->GetDCSC();
	IT lnrow = spSeq->getnrow();
	IT lncol = spSeq->getncol();
	IT localRowStart = colrank * m_perproc; // first row in this process
	IT localColStart = rowrank * n_perproc; // first col in this process
	
	
	
	//mateRow2Col.DebugPrint();
	//mateCol2Row.DebugPrint();
	
	// -----------------------------------------------------------
	// replicate mate vectors for mateCol2Row
	// Communication cost: same as the first communication of SpMV
	// -----------------------------------------------------------
	int xsize = (int)  mateCol2Row.LocArrSize();
	int trxsize = 0;
	MPI_Status status;
	MPI_Sendrecv(&xsize, 1, MPI_INT, diagneigh, TRX, &trxsize, 1, MPI_INT, diagneigh, TRX, World, &status);
	vector<IT> trxnums(trxsize);
	MPI_Sendrecv(mateCol2Row.GetLocArr(), xsize, MPIType<IT>(), diagneigh, TRX, trxnums.data(), trxsize, MPIType<IT>(), diagneigh, TRX, World, &status);
	
	
	vector<int> colsize(pc);
	colsize[colrank] = trxsize;
	MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, colsize.data(), 1, MPI_INT, ColWorld);
	vector<int> dpls(pc,0);	// displacements (zero initialized pid)
	std::partial_sum(colsize.data(), colsize.data()+pc-1, dpls.data()+1);
	int accsize = std::accumulate(colsize.data(), colsize.data()+pc, 0);
	vector<IT> RepMateC2R(accsize);
	MPI_Allgatherv(trxnums.data(), trxsize, MPIType<IT>(), RepMateC2R.data(), colsize.data(), dpls.data(), MPIType<IT>(), ColWorld);
	// -----------------------------------------------------------
	
	
	//cout << endl;
	//for(int i=0; i<RepMateC2R.size(); i++ )
	//  cout << RepMateC2R[i] << " ";
	//cout << endl;
	
	// -----------------------------------------------------------
	// replicate mate vectors for mateRow2Col
	// Communication cost: same as the first communication of SpMV
	//                      (minus the cost of tranposing vector)
	// -----------------------------------------------------------
	
	
	xsize = (int)  mateRow2Col.LocArrSize();
	
	vector<int> rowsize(pr);
	rowsize[rowrank] = xsize;
	MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, rowsize.data(), 1, MPI_INT, RowWorld);
	vector<int> rdpls(pr,0);	// displacements (zero initialized pid)
	std::partial_sum(rowsize.data(), rowsize.data()+pr-1, rdpls.data()+1);
	accsize = std::accumulate(rowsize.data(), rowsize.data()+pr, 0);
	vector<IT> RepMateR2C(accsize);
	MPI_Allgatherv(mateRow2Col.GetLocArr(), xsize, MPIType<IT>(), RepMateR2C.data(), rowsize.data(), rdpls.data(), MPIType<IT>(), RowWorld);
	// -----------------------------------------------------------
	
	
	// -----------------------------------------------------------
	// replicate weights of mates
	// -----------------------------------------------------------
	vector<NT> RepMateWR2C(lnrow);
	vector<NT> RepMateWC2R(lncol);
	ReplicateMateWeights(A, RepMateC2R, RepMateWR2C, RepMateWC2R, nrows, ncols);
	
	//cout << endl;
	//for(int i=0; i<RepMateR2C.size(); i++ )
	//cout << RepMateR2C[i] << " ";
	//cout << endl;
	
	
	// Getting column pointers for all columns (for CSC-style access)
	vector<IT> colptr (lncol+1,-1);
	for(auto colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit) // iterate over all columns
	{
		IT lj = colit.colid(); // local numbering
		colptr[lj] = colit.colptr();
	}
	colptr[lncol] = spSeq->getnnz();
	if(colptr[0] == -1) colptr[0] = 0;
	for(IT k=1; k<=lncol; k++)
	{
		if(colptr[k] == -1) colptr[k] = colptr[k-1];
	}
	
	
	
	//Trace (A);
	
	int iterations = 0;
	NT minw;
	NT weightCur = MatchingWeight(RepMateWC2R, RowWorld, minw);
	NT weightPrev = weightCur - 999999999999;
	while(weightCur > weightPrev && iterations++ < 10)
	{
		
		
		if(myrank==0) cout << "Iteration " << iterations << ". matching weight: sum = "<< weightCur << " min = " << minw << endl;
		// C requests
		// each row is for a processor where C requests will be sent to
		double tstart = MPI_Wtime();
		vector<vector<tuple<IT,IT,NT>>> tempTuples (nprocs);
#ifdef THREADED
//#pragma omp parallel for
#endif
		for(auto colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit) // iterate over columns
		{
			
			IT lj = colit.colid(); // local numbering
			//if(myrank==0) cout << myrank << ") col: " << lj << " ********* "<< endl;
			IT j = lj + localColStart;
			IT mj = RepMateC2R[lj]; // mate of j
			//start nzit from mate colid;
			for(auto nzit = spSeq->begnz(colit); nzit < spSeq->endnz(colit); ++nzit)
			{
				IT li = nzit.rowid();
				IT i = li + localRowStart;
				IT mi = RepMateR2C[li];
				//if(myrank==0) cout << myrank << ") " << i << " " << mi << " "<< j << " " << mj << endl;
				// TODO: use binary search to directly start from RepMateC2R[colid]
				if( i > mj)
				{
					double w = nzit.value()- RepMateWR2C[li] - RepMateWC2R[lj];
					int owner = OwnerProcs(A, mj, mi, nrows, ncols); // think about the symmetry??
					tempTuples[owner].push_back(make_tuple(mj, mi, w));
					
				}
			}
		}
		
		//cout <<  myrank <<") Done Step1......: " << endl;
		//exchange C-request via All2All
		// there might be some empty mesages in all2all
		double t1Comp = MPI_Wtime() - tstart;
		tstart = MPI_Wtime();
		vector<tuple<IT,IT,NT>> recvTuples = ExchangeData(tempTuples, World);
		double t1Comm = MPI_Wtime() - tstart;
		tstart = MPI_Wtime();
		//tempTuples are cleared in ExchangeData function
		
		vector<vector<tuple<IT,IT, IT, NT>>> tempTuples1 (nprocs);
		for(int k=0; k<recvTuples.size(); ++k)
		{
			
			IT mj = get<0>(recvTuples[k]) ;
			IT mi = get<1>(recvTuples[k]) ;
			IT i = RepMateC2R[mi - localColStart];
			NT weight = get<2>(recvTuples[k]);
			
			//DER temp = (*spSeq)(mj - localRowStart, mi - localColStart);
			
			IT * ele = find(dcsc->ir+colptr[mi - localColStart], dcsc->ir+colptr[mi - localColStart+1], mj - localRowStart);
			
			// TODO: Add a function that returns the edge weight directly
			
			//if(!temp.isZero()) // this entry exists
			if (ele != dcsc->ir+colptr[mi - localColStart+1])
			{
				NT cw = weight + RepMateWR2C[mj - localRowStart]; //w+W[M'[j],M[i]];
				if (cw > 0)
				{
					IT j = RepMateR2C[mj - localRowStart];
					//if(myrank==0)
					//cout << k << " mj=" << mj << " mi="<< mi << " i=" << i<< " j="<< j << endl;
					//cout << i << " " << mi << " "<< j << " " << mj << endl;
					int owner = OwnerProcs(A,  mj, j, nrows, ncols); // (mj,j)
					if(owner > nprocs-1) cout << "error !!!\n";
					tempTuples1[owner].push_back(make_tuple(mj, mi, i, cw)); // @@@@@ send i as well
					//tempTuples[owner].push_back(make_tuple(mj, j, cw));
					
				}
			}
		}
		//vector< tuple<IT,IT,NT> >().swap(recvTuples);
		double t2Comp = MPI_Wtime() - tstart;
		tstart = MPI_Wtime();
		//exchange RC-requests via AllToAllv
		vector<tuple<IT,IT,IT,NT>> recvTuples1 = ExchangeData1(tempTuples1, World);
		double t2Comm = MPI_Wtime() - tstart;
		tstart = MPI_Wtime();
		
		vector<tuple<IT,IT,IT,NT>> bestTuplesPhase3 (lncol);
		for(int k=0; k<lncol; ++k)
		{
			bestTuplesPhase3[k] = make_tuple(-1,-1,-1,0); // fix this
		}
		
		for(int k=0; k<recvTuples1.size(); ++k)
		{
			IT mj = get<0>(recvTuples1[k]) ;
			IT mi = get<1>(recvTuples1[k]) ;
			IT i = get<2>(recvTuples1[k]) ;
			NT weight = get<3>(recvTuples1[k]);
			IT j = RepMateR2C[mj - localRowStart];
			IT lj = j - localColStart;
			
			// how can I get i from here ?? ***** // receive i as well
			
			// we can get rid of the first check if edge weights are non negative
			if( (get<0>(bestTuplesPhase3[lj]) == -1)  || (weight > get<3>(bestTuplesPhase3[lj])) )
			{
				bestTuplesPhase3[lj] = make_tuple(i,mi,mj,weight);
			}
		}
		
		
		for(int k=0; k<lncol; ++k)
		{
			if( get<0>(bestTuplesPhase3[k]) != -1)
			{
				//IT j = RepMateR2C[mj - localRowStart]; /// fix me
				
				IT i = get<0>(bestTuplesPhase3[k]) ;
				IT mi = get<1>(bestTuplesPhase3[k]) ;
				IT mj = get<2>(bestTuplesPhase3[k]) ;
				IT j = RepMateR2C[mj - localRowStart];
				NT weight = get<3>(bestTuplesPhase3[k]);
				int owner = OwnerProcs(A,  i, mi, nrows, ncols);
				tempTuples1[owner].push_back(make_tuple(i, j, mj, weight));
			}
		}
		
		//vector< tuple<IT,IT,IT, NT> >().swap(recvTuples1);
		double t3Comp = MPI_Wtime() - tstart;
		tstart = MPI_Wtime();
		recvTuples1 = ExchangeData1(tempTuples1, World);
		double t3Comm = MPI_Wtime() - tstart;
		tstart = MPI_Wtime();
		
		vector<tuple<IT,IT,IT,IT, NT>> bestTuplesPhase4 (lncol);
		// we could have used lnrow in both bestTuplesPhase3 and bestTuplesPhase4
		
		// Phase 4
		// at the owner of (i,mi)
		for(int k=0; k<lncol; ++k)
		{
			bestTuplesPhase4[k] = make_tuple(-1,-1,-1,-1,0);
		}
		
		for(int k=0; k<recvTuples1.size(); ++k)
		{
			IT i = get<0>(recvTuples1[k]) ;
			IT j = get<1>(recvTuples1[k]) ;
			IT mj = get<2>(recvTuples1[k]) ;
			IT mi = RepMateR2C[i-localRowStart];
			NT weight = get<3>(recvTuples1[k]);
			IT lmi = mi - localColStart;
			//IT lj = j - localColStart;
			
			// cout <<"****" << i << " " << mi << " "<< j << " " << mj << " " << get<0>(bestTuplesPhase4[lj]) << endl;
			// we can get rid of the first check if edge weights are non negative
			if( ((get<0>(bestTuplesPhase4[lmi]) == -1)  || (weight > get<4>(bestTuplesPhase4[lmi]))) && get<0>(bestTuplesPhase3[lmi])==-1 )
			{
				bestTuplesPhase4[lmi] = make_tuple(i,j,mi,mj,weight);
				//cout << "(("<< i << " " << mi << " "<< j << " " << mj << "))"<< endl;
			}
		}
		
		
		vector<vector<tuple<IT,IT,IT, IT>>> winnerTuples (nprocs);
		
		
		for(int k=0; k<lncol; ++k)
		{
			if( get<0>(bestTuplesPhase4[k]) != -1)
			{
				//int owner = OwnerProcs(A,  get<0>(bestTuples[k]), get<1>(bestTuples[k]), nrows, ncols); // (i,mi)
				//tempTuples[owner].push_back(bestTuples[k]);
				IT i = get<0>(bestTuplesPhase4[k]) ;
				IT j = get<1>(bestTuplesPhase4[k]) ;
				IT mi = get<2>(bestTuplesPhase4[k]) ;
				IT mj = get<3>(bestTuplesPhase4[k]) ;
				
				
				int owner = OwnerProcs(A,  mj, j, nrows, ncols);
				winnerTuples[owner].push_back(make_tuple(i, j, mi, mj));
				
				/// be very careful here
				// passing the opposite of the matching to the owner of (i,mi)
				owner = OwnerProcs(A,  i, mi, nrows, ncols);
				winnerTuples[owner].push_back(make_tuple(mj, mi, j, i));
			}
		}
		
		
		//vector< tuple<IT,IT,IT, NT> >().swap(recvTuples1);
		double t4Comp = MPI_Wtime() - tstart;
		tstart = MPI_Wtime();
		
		vector<tuple<IT,IT,IT,IT>> recvWinnerTuples = ExchangeData1(winnerTuples, World);
		
		double t4Comm = MPI_Wtime() - tstart;
		tstart = MPI_Wtime();
		
		// at the owner of (mj,j)
		vector<tuple<IT,IT>> rowBcastTuples(recvWinnerTuples.size()); //(mi,mj)
		vector<tuple<IT,IT>> colBcastTuples(recvWinnerTuples.size()); //(j,i)
		
		for(int k=0; k<recvWinnerTuples.size(); ++k)
		{
			IT i = get<0>(recvWinnerTuples[k]) ;
			IT j = get<1>(recvWinnerTuples[k]) ;
			IT mi = get<2>(recvWinnerTuples[k]) ;
			IT mj = get<3>(recvWinnerTuples[k]);
			
			
			
			
			colBcastTuples[k] = make_tuple(j,i);
			//rowBcastTuples.push_back(make_tuple(i,j));
			rowBcastTuples[k] = make_tuple(mj,mi);
			//colBcastTuples.push_back(make_tuple(mi,mj));
		}
		double t5Comp = MPI_Wtime() - tstart;
		tstart = MPI_Wtime();
		
		vector<tuple<IT,IT>> updatedR2C = MateBcast(rowBcastTuples, RowWorld);
		vector<tuple<IT,IT>> updatedC2R = MateBcast(colBcastTuples, ColWorld);
		
		double t5Comm = MPI_Wtime() - tstart;
		tstart = MPI_Wtime();
		
#ifdef THREADED
#pragma omp parallel for
#endif
		for(int k=0; k<updatedR2C.size(); k++)
		{
			IT row = get<0>(updatedR2C[k]);
			IT mate = get<1>(updatedR2C[k]);
			if( (row < localRowStart) || (row >= (localRowStart+lnrow)))
			{
				cout << "myrank: " << myrank << "row: " << row << "localRowStart: " << localRowStart << endl;
				exit(1);
			}
			RepMateR2C[row-localRowStart] = mate;
		}
		
#ifdef THREADED
#pragma omp parallel for
#endif
		for(int k=0; k<updatedC2R.size(); k++)
		{
			IT col = get<0>(updatedC2R[k]);
			IT mate = get<1>(updatedC2R[k]);
			RepMateC2R[col-localColStart] = mate;
		}
		
		
		double tUpdateMateComp = MPI_Wtime() - tstart;
		tstart = MPI_Wtime();
		// update weights of matched edges
		// we can do better than this since we are doing sparse updates
		ReplicateMateWeights(A, RepMateC2R, RepMateWR2C, RepMateWC2R, nrows, ncols);
		double tUpdateWeight = MPI_Wtime() - tstart;
		
		
		weightPrev = weightCur;
		weightCur = MatchingWeight(RepMateWC2R, RowWorld, minw);
		
		
		//UpdateMatching(mateRow2Col, mateCol2Row, RepMateR2C, RepMateC2R);
		//CheckMatching(mateRow2Col,mateCol2Row);
		
		if(myrank==0)
		{
			cout << " " << t1Comp << " " << t1Comm << " " << t2Comp << " " << t2Comm << " " << t3Comp << " " << t3Comm << " " << t4Comp << " " << t4Comm << " " << t5Comp << " " << t5Comm << " " << tUpdateMateComp << " " << tUpdateWeight << endl;
		}
	}
	
	
	
	// update the distributed mate vectors from replicated mate vectors
	UpdateMatching(mateRow2Col, mateCol2Row, RepMateR2C, RepMateC2R);
	//weightCur = MatchingWeight(RepMateWC2R, RowWorld);
	
	
	
}


#endif /* ApproxWeightPerfectMatching_h */
