/****************************************************************/
/* Sequential and Parallel Sparse Matrix Multiplication Library */
/* version 2.3 --------------------------------------------------/
/* date: 01/18/2009 ---------------------------------------------/
/* author: Aydin Buluc (aydin@cs.ucsb.edu) ----------------------/
/****************************************************************/

#include "SpParMPI2.h"


// If every processor has a distinct triples file such as {A_0, A_1, A_2,... A_p} for p processors
template <class IT, class NT, class DER>
SpParMPI2< IT,NT,DER >::SpParMPI2 (ifstream & input, MPI::Intracomm & world)
{
	if(!input.is_open())
	{
		perror("Input file doesn't exist\n");
		exit(-1);
	}
	commGrid = new CommGrid(world, 0, 0);
	input >> (*spSeq);
}

template <class IT, class NT, class DER>
SpParMPI2< IT,NT,DER >::SpParMPI2 (SpMat<IT,NT,DER> * myseq, MPI::Intracomm & world): spSeq(myseq)
{
	commGrid = new CommGrid(world, 0, 0);
}

// If there is a single file read by the master process only, use this and then call ReadDistribute()
template <class IT, class NT, class DER>
SpParMPI2< IT,NT,DER >::SpParMPI2 ()
{
	spSeq = new DER();
	commGrid = new CommGrid(MPI::COMM_WORLD, 0, 0);
}

template <class IT, class NT, class DER>
SpParMPI2< IT,NT,DER >::~SpParMPI2 ()
{
	if(spSeq != NULL) delete spSeq;
	if(commGrid != NULL) delete commGrid;
}


template <class IT, class NT, class DER>
SpParMPI2< IT,NT,DER >::SpParMPI2 (const SpParMPI2< IT,NT,DER > & rhs)
{
	if(rhs.spSeq != NULL)	
		spSeq = new SpMat< IT,NT,DER >(*(rhs.spSeq));  	// Deep copy of local block

	if(rhs.commGrid != NULL)	
		commGrid = new CommGrid(*(rhs.commGrid));  	// Deep copy of communication grid
}

template <class IT, class NT, class DER>
SpParMPI2< IT,NT,DER > & SpParMPI2< IT,NT,DER >::operator=(const SpParMPI2< IT,NT,DER > & rhs)
{
	if(this != &rhs)		
	{
		//! Check agains NULL is probably unneccessary, delete won't fail on NULL
		//! But useful in the presence of a user defined "operator delete" which fails to check NULL
		if(spSeq != NULL) delete spSeq;
		if(rhs.spSeq != NULL)	
			spSeq = new SpMat< IT,NT,DER >(*(rhs.spSeq));  // Deep copy of local block
		
		if(commGrid != NULL) delete commGrid;
		if(rhs.commGrid != NULL)	
			commGrid = new CommGrid(*(rhs.commGrid));  // Deep copy of communication grid
	}
	return *this;
}

template <class IT, class NT, class DER>
SpParMPI2< IT,NT,DER > & SpParMPI2< IT,NT,DER >::operator+=(const SpParMPI2< IT,NT,DER > & rhs)
{
	if(this != &rhs)		
	{
		if(commGrid == rhs.commGrid)	
		{
			(*spSeq) += (*(rhs.spSeq));
		}
		else
		{
			cout << "Grids are not comparable for parallel addition (A+B)" << endl; 
		}
	}
	else
	{
		cout<< "Missing feature (A+A): Use multiply with 2 instead !"<<endl;	
	}
	return *this;	
}

template <class IT, class NT, class DER>
IT SpParMPI2< IT,NT,DER >::getnnz() const
{
	IT totalnnz = 0;    
	IT localnnz = spSeq->getnnz();
	(commGrid->GetWorld()).Allreduce( &localnnz, &totalnnz, 1, MPIType<IT>(), MPI::SUM);
 	return totalnnz;  
}

template <class IT, class NT, class DER>
IT SpParMPI2< IT,NT,DER >::getnrow() const
{
	IT totalrows = 0;
	IT localrows = spSeq->getnrow();    
	(commGrid->GetColWorld()).Allreduce( &localrows, &totalrows, 1, MPIType<IT>(), MPI::SUM);
 	return totalrows;  
}

template <class IT, class NT, class DER>
IT SpParMPI2< IT,NT,DER >::getncol() const
{
	IT totalcols = 0;
	IT localcols = spSeq->getncol();    
	(commGrid->GetRowWorld()).Allreduce( &localcols, &totalcols, 1, MPIType<IT>(), MPI::SUM);
 	return totalcols;  
}


/** 
 * Create a submatrix of size m x (size(ncols) * s) on a r x s processor grid
 * Essentially fetches the columns ci[0], ci[1],... ci[size(ci)] from every submatrix
 *
template <class T>
SpParMatrix<T> * SpParMPI2<T>::SubsRefCol (const vector<ITYPE> & ci) const
{
	vector<ITYPE> ri;
	
 	shared_ptr< SpDCCols<T> > ARef (new SpDCCols<T> (spSeq->SubsRefCol(ci)));	

	return new SpParMPI2<T> (ARef, commGrid->commWorld);
} */

template <typename SR, typename IU, typename NU, typename UDER> 
SpParMPI2<IU,NU,UDER> Mult_AnXBn (const SpParMPI2<IU,NU,UDER> & A, const SpParMPI2<IU,NU,UDER> & B )
{
	double t1 = MPI::Wtime();

	if(A.getncol() != B.getnrow())
	{
		cout<<"Can not multiply, dimensions does not match"<<endl;
		MPI::COMM_WORLD.Abort(DIMMISMATCH);
		return SpParMPI2< IU,NU,UDER >();
	}

	int stages, Aoffset, Boffset; 	// stages = inner dimension of matrix blocks
	CommGrid * GridC = ProductGrid(A.commGrid, B.commGrid, stages, Aoffset, Boffset);		
		
	const_cast< SpMat<IU,NU,UDER>* >(B.spSeq)->Transpose();
	
	// set row & col window handles
	vector<MPI::Win> rowwindows, colwindows;
	SpParHelper::SetWindows((A.commGrid)->GetRowWorld(), *(A.spSeq), rowwindows);
	SpParHelper::SetWindows((B.commGrid)->GetColWorld(), *(B.spSeq), colwindows);

	IU ** ARecvSizes = SpHelper::allocate2D<IU>(UDER::esscount, stages);
	IU ** BRecvSizes = SpHelper::allocate2D<IU>(UDER::esscount, stages);
 
	SpParHelper::GetSetSizes( *(A.spSeq), ARecvSizes, (A.commGrid)->GetRowWorld(), (A.commGrid)->GetRankInProcRow());
	SpParHelper::GetSetSizes( *(B.spSeq), BRecvSizes, (B.commGrid)->GetColWorld(), (B.commGrid)->GetRankInProcCol());
	
	double t2 = MPI::Wtime();
	if(GridC->GetRank() == 0)
		fprintf(stdout, "setup (matrix transposition and memory registration) took %.6lf seconds\n", t2-t1);
	
	SpMat<IU,NU,UDER> * ARecv;
	SpMat<IU,NU,UDER> * BRecv; 

	SpMat<IU,NU,UDER> * C = new SpMat<IU,NU,UDER>();   // Create an empty object for the product	

	for(int i = 0; i < stages; ++i) 	// Robust generalization to non-square grids will require block-cyclic distibution	
	{
		int Aownind = (i+Aoffset) % stages;		
		int Bownind = (i+Boffset) % stages;

		if(Aownind == (A.commGrid)->GetRankInProcRow())
		{
			ARecv = A.spSeq;	// shallow-copy
		}
		else
		{
			// pack essentials to a vector
			vector<IU> ess(UDER::esscount);
			for(int j=0; j< UDER::esscount; ++j)	
			{
				ess[j] = ARecvSizes[j][Aownind];	
			}		
			SpParHelper::FetchMatrix(*ARecv, ess, rowwindows, Aownind);
		}
		if(Bownind == (B.commGrid)->GetRankInProcCol())
		{
			BRecv = B.spSeq;	// shallow-copy
		}
		else
		{
			// pack essentials to a vector
			vector<IU> ess(UDER::esscount);
			for(int j=0; j< UDER::esscount; ++j)	
			{
				ess[j] = BRecvSizes[j][Bownind];	
			}	
			SpParHelper::FetchMatrix(*BRecv, ess, colwindows, Bownind);	
		}
	
		SpParHelper::UnlockWindows(Aownind, rowwindows);	// unlock windows for A
		SpParHelper::UnlockWindows(Bownind, colwindows);	// unlock windows for B	

		C->template SpGEMM < SR > ( *ARecv, *BRecv, false, true);
		
		if(Aownind != (A.commGrid)->GetRankInProcRow()) delete ARecv;
		if(Bownind != (B.commGrid)->GetRankInProcCol()) delete BRecv; 
	} 

	(GridC->GetWorld()).Barrier();

	for(int i=0; i< rowwindows.size(); ++i)
	{
		rowwindows[i].Free();
	}
	for(int i=0; i< colwindows.size(); ++i)
	{
		colwindows[i].Free();
	}

	const_cast< SpMat<IU,NU,UDER>* >(B.spSeq)->Transpose();	// transpose back to original
	
	return SpParMPI2<IU,NU,UDER> (C, GridC);			// return the result object
}


//! Handles all sorts of orderings as long as there are no duplicates
//! May perform better when the data is already reverse column-sorted (i.e. in decreasing order)
template <class IT, class NT, class DER>
ifstream& SpParMPI2< IT,NT,DER >::ReadDistribute (ifstream& infile, int master)
{
	IT total_m, total_n, total_nnz;
	IT m_perproc, n_perproc;

	int colneighs = commGrid->GetGridRows();	// number of neighbors along this processor column (including oneself)
	int rowneighs = commGrid->GetGridCols();	// number of neighbors along this processor row (including oneself)

	IT buffpercolneigh = MEMORYINBYTES / (colneighs * (2 * sizeof(IT) + sizeof(NT)));
	IT buffperrowneigh = MEMORYINBYTES / (rowneighs * (2 * sizeof(IT) + sizeof(NT)));

	// make sure that buffperrowneigh >= buffpercolneigh to cover for this patological case:
	//   	-- all data received by a given column head (by vertical communication) are headed to a single processor along the row
	//   	-- then making sure buffperrowneigh >= buffpercolneigh guarantees that the horizontal buffer will never overflow
	buffperrowneigh = std::max(buffperrowneigh, buffpercolneigh);

	IT * cdispls = new IT[colneighs];
	for (int i=0; i<colneighs; ++i)
		cdispls[i] = i*buffpercolneigh;

	IT * rdispls = new IT[colneighs];
	for (int i=0; i<rowneighs; ++i)
		rdispls[i] = i*buffperrowneigh;		

	IT *ccurptrs, *rcurptrs;
	IT recvcount;

	IT * rows; 
	IT * cols;
	NT * vals;

	// Note: all other column heads that initiate the horizontal communication has the same "rankinrow" with the master
	int rankincol = commGrid->GetRankInProcCol(master);	// get master's rank in its processor column
	int rankinrow = commGrid->GetRankInProcRow(master);	
  	
	vector< tuple<IT, IT, NT> > localtuples;

	if(commGrid->GetRank() == master)	// 1 processor
	{
		
		cout << "masters rank is "<< rankincol << " x " << rankinrow << endl;
		cout << "buffers for neighbors are " << buffperrowneigh << " x " << buffpercolneigh << endl;
		
		// allocate buffers on the heap as stack space is usually limited
		rows = new IT [ buffpercolneigh * colneighs ];
		cols = new IT [ buffpercolneigh * colneighs ];
		vals = new NT [ buffpercolneigh * colneighs ];

		ccurptrs = new IT[colneighs];
		rcurptrs = new IT[rowneighs];
		fill_n(ccurptrs, colneighs, (IT) zero);	// fill with zero
		fill_n(rcurptrs, rowneighs, (IT) zero);	
		
		if (infile.is_open())
		{
			infile >> total_m >> total_n >> total_nnz;
			m_perproc = total_m / colneighs;
			n_perproc = total_n / rowneighs;
			cout << m_perproc << " " << n_perproc << endl;		
	
			(commGrid->commWorld).Bcast(&total_m, 1, MPIType<IT>(), master);
			(commGrid->commWorld).Bcast(&total_n, 1, MPIType<IT>(), master);
			
			IT temprow, tempcol;
			NT tempval;
			IT cnz = 0;
			while ( (!infile.eof()) && cnz < total_nnz)
			{
				infile >> temprow >> tempcol >> tempval;
				--temprow;	// file is 1-based where C-arrays are 0-based
				--tempcol;

				int colrec = temprow / m_perproc;	// precipient processor along the column
				rows[ colrec * buffpercolneigh + ccurptrs[colrec] ] = temprow;
				cols[ colrec * buffpercolneigh + ccurptrs[colrec] ] = tempcol;
				vals[ colrec * buffpercolneigh + ccurptrs[colrec] ] = tempval;
				++ (ccurptrs[colrec]);				

				if(ccurptrs[colrec] == buffpercolneigh || (cnz == (total_nnz-1)) )		// one buffer is full, or file is done !
				{
					// first, send the receive counts ...
					(commGrid->colWorld).Scatter(ccurptrs, 1, MPIType<IT>(), &recvcount, 1, MPIType<IT>(), rankincol);

					// generate space for own recv data ...
					vector<IT> temprows(recvcount);
					vector<IT> tempcols(recvcount);
					vector<NT> tempvals(recvcount);
					
					// then, send all buffers that to their recipients ...
					(commGrid->colWorld).Scatterv(rows, ccurptrs, cdispls, MPIType<IT>(), &temprows[0], recvcount,  MPIType<IT>(), rankincol); 
					(commGrid->colWorld).Scatterv(cols, ccurptrs, cdispls, MPIType<IT>(), &tempcols[0], recvcount,  MPIType<IT>(), rankincol); 
					(commGrid->colWorld).Scatterv(vals, ccurptrs, cdispls, MPIType<NT>(), &tempvals[0], recvcount,  MPIType<NT>(), rankincol); 

					// finally, reset current pointers !
					fill_n(ccurptrs, colneighs, (IT) zero);
					DeleteAll(rows, cols, vals);
			
					/* Begin horizontal distribution */

					rows = new IT [ buffperrowneigh * rowneighs ];
					cols = new IT [ buffperrowneigh * rowneighs ];
					vals = new NT [ buffperrowneigh * rowneighs ];
			
					// prepare to send the data along the horizontal
					for(IT i=zero; i< recvcount; ++i)
					{
						IT rowrec = tempcols[i] / n_perproc;
						rows[ rowrec * buffperrowneigh + rcurptrs[rowrec] ] = temprows[i];
						cols[ rowrec * buffperrowneigh + rcurptrs[rowrec] ] = tempcols[i];
						vals[ rowrec * buffperrowneigh + rcurptrs[rowrec] ] = tempvals[i];
						++ (rcurptrs[rowrec]);	
					}
				
					// Send the receive counts for horizontal communication ...
					(commGrid->rowWorld).Scatter(rcurptrs, 1, MPIType<IT>(), &recvcount, 1, MPIType<IT>(), rankinrow);
					vector<IT>(recvcount).swap(temprows);	// the data is now stored in rows/cols/vals, can reset temporaries
					vector<IT>(recvcount).swap(tempcols);	// sets size and capacity to recvcount
					vector<NT>(recvcount).swap(tempvals);

					// then, send all buffers that to their recipients ...
					(commGrid->rowWorld).Scatterv(rows, rcurptrs, rdispls, MPIType<IT>(), &temprows[0], recvcount,  MPIType<IT>(), rankinrow); 
					(commGrid->rowWorld).Scatterv(cols, rcurptrs, rdispls, MPIType<IT>(), &tempcols[0], recvcount,  MPIType<IT>(), rankinrow); 
					(commGrid->rowWorld).Scatterv(vals, rcurptrs, rdispls, MPIType<NT>(), &tempvals[0], recvcount,  MPIType<NT>(), rankinrow); 

					// now push what is ours to tuples
					IT moffset = commGrid->myprocrow * m_perproc; 
					IT noffset = commGrid->myproccol * n_perproc; 
					for(IT i=zero; i< recvcount; ++i)
					{					
						localtuples.push_back( 	make_tuple(temprows[i]-moffset, tempcols[i]-noffset, tempvals[i]) );
					}
					
					fill_n(rcurptrs, rowneighs, (IT) zero);
					DeleteAll(rows, cols, vals);		
					
					// reuse these buffers for the next vertical communication								
					rows = new IT [ buffpercolneigh * colneighs ];
					cols = new IT [ buffpercolneigh * colneighs ];
					vals = new NT [ buffpercolneigh * colneighs ];
				}
				++ cnz;
			}
			assert (cnz == total_nnz);
			
			// Signal the end of file to other processors along the column
			fill_n(ccurptrs, colneighs, numeric_limits<IT>::max());	
			(commGrid->colWorld).Scatter(ccurptrs, 1, MPIType<IT>(), &recvcount, 1, MPIType<IT>(), rankincol);

			// And along the row ...
			fill_n(rcurptrs, rowneighs, numeric_limits<IT>::max());				
			(commGrid->rowWorld).Scatter(rcurptrs, 1, MPIType<IT>(), &recvcount, 1, MPIType<IT>(), rankinrow);
			
		}
		DeleteAll(rows,cols,vals, ccurptrs, rcurptrs);

	}
	else if( commGrid->OnSameProcCol(master) ) 	// (r-1) processors
	{
		(commGrid->commWorld).Bcast(&total_m, 1, MPIType<IT>(), master);
		(commGrid->commWorld).Bcast(&total_n, 1, MPIType<IT>(), master);
		m_perproc = total_m / colneighs;
		n_perproc = total_n / rowneighs;

		// void MPI::Comm::Scatterv(const void* sendbuf, const int sendcounts[], const int displs[], const MPI::Datatype& sendtype,
		//				void* recvbuf, int recvcount, const MPI::Datatype & recvtype, int root) const
		// The outcome is as if the root executed n send operations,
    		//	MPI_Send(sendbuf + displs[i] * extent(sendtype), sendcounts[i], sendtype, i, ...)
		// and each process executed a receive,
   		// 	MPI_Recv(recvbuf, recvcount, recvtype, root, ...)
		// The send buffer is ignored for all nonroot processes.

		while(true)
		{
			// first receive the receive counts ...
			(commGrid->colWorld).Scatter(ccurptrs, 1, MPIType<IT>(), &recvcount, 1, MPIType<IT>(), rankincol);

			cerr << commGrid->myrank << " with recvcount " << recvcount << endl;
			if( recvcount == numeric_limits<IT>::max())
				break;
	
			// create space for incoming data ... 
			vector<IT> temprows(recvcount);
			vector<IT> tempcols(recvcount);
			vector<NT> tempvals(recvcount);

			// receive actual data ... (first 4 arguments are ignored in the receiver side)
			(commGrid->colWorld).Scatterv(rows, ccurptrs, cdispls, MPIType<IT>(), &temprows[0], recvcount,  MPIType<IT>(), rankincol); 
			(commGrid->colWorld).Scatterv(cols, ccurptrs, cdispls, MPIType<IT>(), &tempcols[0], recvcount,  MPIType<IT>(), rankincol); 
			(commGrid->colWorld).Scatterv(vals, ccurptrs, cdispls, MPIType<NT>(), &tempvals[0], recvcount,  MPIType<NT>(), rankincol); 

			// now, send the data along the horizontal
			rcurptrs = new IT[rowneighs];
			fill_n(rcurptrs, rowneighs, (IT) zero);	
		
			rows = new IT [ buffperrowneigh * rowneighs ];
			cols = new IT [ buffperrowneigh * rowneighs ];
			vals = new NT [ buffperrowneigh * rowneighs ];

			// prepare to send the data along the horizontal
			for(IT i=zero; i< recvcount; ++i)
			{
				IT rowrec = tempcols[i] / n_perproc;
				rows[ rowrec * buffperrowneigh + rcurptrs[rowrec] ] = temprows[i];
				cols[ rowrec * buffperrowneigh + rcurptrs[rowrec] ] = tempcols[i];
				vals[ rowrec * buffperrowneigh + rcurptrs[rowrec] ] = tempvals[i];
				++ (rcurptrs[rowrec]);	
			}
				
			// Send the receive counts for horizontal communication ...
			(commGrid->rowWorld).Scatter(rcurptrs, 1, MPIType<IT>(), &recvcount, 1, MPIType<IT>(), rankinrow);
			vector<IT>(recvcount).swap(temprows);	// the data is now stored in rows/cols/vals, can reset temporaries
			vector<IT>(recvcount).swap(tempcols);	// sets size and capacity to recvcount
			vector<NT>(recvcount).swap(tempvals);

			// then, send all buffers that to their recipients ...
			(commGrid->rowWorld).Scatterv(rows, rcurptrs, rdispls, MPIType<IT>(), &temprows[0], recvcount,  MPIType<IT>(), rankinrow); 
			(commGrid->rowWorld).Scatterv(cols, rcurptrs, rdispls, MPIType<IT>(), &tempcols[0], recvcount,  MPIType<IT>(), rankinrow); 
			(commGrid->rowWorld).Scatterv(vals, rcurptrs, rdispls, MPIType<NT>(), &tempvals[0], recvcount,  MPIType<NT>(), rankinrow); 

			// now push what is ours to tuples
			IT moffset = commGrid->myprocrow * m_perproc; 
			IT noffset = commGrid->myproccol * n_perproc;
			for(IT i=zero; i< recvcount; ++i)
			{					
				localtuples.push_back( 	make_tuple(temprows[i]-moffset, tempcols[i]-noffset, tempvals[i]) );
			}
					
			fill_n(rcurptrs, rowneighs, (IT) zero);
			DeleteAll(rows, cols, vals);	
		}
		// Signal the end of file to other processors along the row
		fill_n(rcurptrs, rowneighs, numeric_limits<IT>::max());				
		(commGrid->rowWorld).Scatter(rcurptrs, 1, MPIType<IT>(), &recvcount, 1, MPIType<IT>(), rankinrow);
		delete [] rcurptrs;	
	}
	else		// r * (s-1) processors that only participate in the horizontal communication step
	{
		(commGrid->commWorld).Bcast(&total_m, 1, MPIType<IT>(), master);
		(commGrid->commWorld).Bcast(&total_n, 1, MPIType<IT>(), master);
		m_perproc = total_m / colneighs;
		n_perproc = total_n / rowneighs;
		
		while (true)
		{
			// receive the receive count
			(commGrid->rowWorld).Scatter(rcurptrs, 1, MPIType<IT>(), &recvcount, 1, MPIType<IT>(), rankinrow);
			if( recvcount == numeric_limits<IT>::max())
				break;
		
			// create space for incoming data ... 
			vector<IT> temprows(recvcount);
			vector<IT> tempcols(recvcount);
			vector<NT> tempvals(recvcount);

			(commGrid->rowWorld).Scatterv(rows, rcurptrs, rdispls, MPIType<IT>(), &temprows[0], recvcount,  MPIType<IT>(), rankinrow); 
			(commGrid->rowWorld).Scatterv(cols, rcurptrs, rdispls, MPIType<IT>(), &tempcols[0], recvcount,  MPIType<IT>(), rankinrow); 
			(commGrid->rowWorld).Scatterv(vals, rcurptrs, rdispls, MPIType<NT>(), &tempvals[0], recvcount,  MPIType<NT>(), rankinrow);

			// now push what is ours to tuples
			IT moffset = commGrid->myprocrow * m_perproc; 
			IT noffset = commGrid->myproccol * n_perproc;
			for(IT i=zero; i< recvcount; ++i)
			{					
				localtuples.push_back( 	make_tuple(temprows[i]-moffset, tempcols[i]-noffset, tempvals[i]) );
			}
		}
	
	}
	
	DeleteAll(cdispls, rdispls);
	tuple<IT,IT,NT> * arrtuples = new tuple<IT,IT,NT>[localtuples.size()];	// the vector will go out of scope, make it stick !
	copy(localtuples.begin(), localtuples.end(), arrtuples);

 	IT localm = (commGrid->myprocrow != (commGrid->grrows-1))? m_perproc: (total_m - (m_perproc * (commGrid->grrows-1)));
 	IT localn = (commGrid->myproccol != (commGrid->grcols-1))? n_perproc: (total_n - (n_perproc * (commGrid->grcols-1)));
	
	spSeq->Create( localtuples.size(), localm, localn, arrtuples);		// the deletion of arrtuples[] is handled by SpMat::Create

	return infile;
}



template <class IT, class NT, class DER>
ofstream& SpParMPI2<IT,NT,DER>::put(ofstream& outfile) const
{
	outfile << (*spSeq) << endl;
}

template <class IU, class NU, class UDER>
ofstream& operator<<(ofstream& outfile, const SpParMPI2<IU, NU, UDER> & s)
{
	return s.put(outfile) ;	// use the right put() function

}
