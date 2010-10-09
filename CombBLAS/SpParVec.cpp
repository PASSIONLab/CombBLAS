#include <limits>
#include "SpParVec.h"
#include "SpDefs.h"
using namespace std;

template <class IT, class NT>
SpParVec<IT, NT>::SpParVec ( shared_ptr<CommGrid> grid): commGrid(grid), length(zero)
{
	if(commGrid->GetRankInProcRow() == commGrid->GetRankInProcCol())
		diagonal = true;
	else
		diagonal = false;	
};
	

template <class IT, class NT>
SpParVec<IT,NT> & SpParVec<IT, NT>::operator+=(const SpParVec<IT,NT> & rhs)
{
	if(this != &rhs)		
	{	
		if(diagonal)	// Only the diagonal processors hold values
		{
			vector< pair<IT, NT> > narr;
			IT lsize = arr.size();
			IT rsize = rhs.arr.size();
			narr.reserve(lsize+rsize);

			IT i=0, j=0;
			while(i < lsize && j < rsize)
			{
				// assignment won't change the size of vector, push_back is necessary
				if(arr[i].first > rhs.arr[j].first)
				{	
					narr.push_back( rhs.arr[j++] );
				}
				else if(arr[i].first < rhs.arr[j].first)
				{
					narr.push_back( arr[i++] );
				}
				else
				{
					narr.push_back( make_pair(arr[i].first, arr[i++].second + rhs.arr[j++].second) );
				}
			}
			arr.swap(narr);		// arr will contain the elements of narr with capacity shrunk-to-fit size
		} 	
	}	
	return *this;
};	

//! Called on an existing object
//! ABAB: This needs a rewrite to respect the semantics of SpParVec<> class
template <class IT, class NT>
ifstream& SpParVec<IT,NT>::ReadDistribute (ifstream& infile, int master)
{
	IT total_n, total_nnz, n_perproc;
	MPI::Intracomm diagprocs = commGrid->GetDiagWorld();
Get_rank()

	int neighs = diagprocs->Get_size();	// number of neighbors along diagonal (including oneself)
	IT buffperneigh = MEMORYINBYTES / (neighs * (sizeof(IT) + sizeof(NT)));

	IT * cdispls = new IT[colneighs];
	for (int i=0; i<colneighs; ++i)
		cdispls[i] = i*buffpercolneigh;

	IT *ccurptrs;
	IT recvcount;
	IT * inds; 
	NT * vals;

	// Note: all other column heads that initiate the horizontal communication has the same "rankinrow" with the master
	int rankincol = commGrid->GetRankInProcCol(master);	// get master's rank in its processor column
	int rankinrow = commGrid->GetRankInProcRow(master);	
  	
	vector< pair<IT,NT> > localvec;
	if(commGrid->GetRank() == master)	// 1 processor
	{		
		int diag;

		// allocate buffers on the heap as stack space is usually limited
		inds = new IT [ buffpercolneigh * colneighs ];
		vals = new NT [ buffpercolneigh * colneighs ];

		ccurptrs = new IT[colneighs];
		fill_n(ccurptrs, colneighs, (IT) zero);	// fill with zero
		
		if (infile.is_open())
		{
			infile >> total_n >> total_nnz;
			n_perproc = total_n / colneighs;
			commGrid->GetWorld().Bcast(&total_n, 1, MPIType<IT>(), master);			
	
			IT tempind;
			NT tempval;
			IT cnz = 0;
			while ( (!infile.eof()) && cnz < total_nnz)
			{
				infile >> tempind;
				infile >> tempval;
				tempind--;

				int colrec = std::min(tempind / n_perproc, colneighs-1);	// precipient processor along the column
				inds[ colrec * buffpercolneigh + ccurptrs[colrec] ] = tempind;
				vals[ colrec * buffpercolneigh + ccurptrs[colrec] ] = tempval;
				++ (ccurptrs[colrec]);				

				if(ccurptrs[colrec] == buffpercolneigh || (cnz == (total_nnz-1)) )		// one buffer is full, or file is done !
				{
					// first, send the receive counts ...
					commGrid->GetColWorld().Scatter(ccurptrs, 1, MPIType<IT>(), &recvcount, 1, MPIType<IT>(), rankincol);

					// generate space for own recv data ... (use arrays because vector<bool> is cripled, if NT=bool)
					IT * tempinds = new IT[recvcount];
					NT * tempvals = new NT[recvcount];
					
					// then, send all buffers that to their recipients ...
					commGrid->GetColWorld().Scatterv(inds, ccurptrs, cdispls, MPIType<IT>(), tempinds, recvcount,  MPIType<IT>(), rankincol); 
					commGrid->GetColWorld().Scatterv(vals, ccurptrs, cdispls, MPIType<NT>(), tempvals, recvcount,  MPIType<NT>(), rankincol); 

					// finally, reset current pointers !
					fill_n(ccurptrs, colneighs, (IT) zero);
					DeleteAll(inds, vals);
			
					/* Begin horizontal distribution, send data to the diagonal processor on this row */
					if(diagonal)
					{
						IT noffset = commGrid->GetRankInProcRow() * n_perproc; 
						for(IT i=zero; i< recvcount; ++i)
						{	
							localvec.push_back( make_pair(tempinds[i]-noffset, tempvals[i]) );
						}
					}
					else
					{
						diag = commGrid->GetDiagOfProcRow();
						commGrid->GetRowWorld().Send(&recvcount, 1, MPIType<IT>(), diag, RDTAGNNZ);	// send the size first						
						commGrid->GetRowWorld().Send(tempinds, recvcount, MPIType<IT>(), diag, RDTAGINDS);	// then the data
						commGrid->GetRowWorld().Send(tempvals, recvcount, MPIType<NT>(), diag, RDTAGVALS);
					}
					DeleteAll(tempinds, tempvals);		
					
					// reuse these buffers for the next vertical communication								
					inds = new IT [ buffpercolneigh * colneighs ];
					vals = new NT [ buffpercolneigh * colneighs ];
				}
				++ cnz;
			}
			assert (cnz == total_nnz);
			
			// Signal the end of file to other processors along the column
			fill_n(ccurptrs, colneighs, numeric_limits<IT>::max());	
			commGrid->GetColWorld().Scatter(ccurptrs, 1, MPIType<IT>(), &recvcount, 1, MPIType<IT>(), rankincol);

			if(!diagonal)
			{
				// And along the diagonal on this row ...
				recvcount = numeric_limits<IT>::max();
				commGrid->GetRowWorld().Send(&recvcount, 1, MPIType<IT>(), diag, RDTAGNNZ);	
			}
		}
		else	// input file does not exist !
		{
			total_n = 0;	
			commGrid->GetWorld().Bcast(&total_n, 1, MPIType<IT>(), master);						
		}
		DeleteAll(inds,vals, ccurptrs);
	}
	else if( commGrid->OnSameProcCol(master) ) 	// (r-1) processors
	{
		commGrid->GetWorld().Bcast(&total_n, 1, MPIType<IT>(), master);
		n_perproc = total_n / colneighs;
		int diag;

		while(total_n > 0)
		{
			// first receive the receive counts ...
			commGrid->GetColWorld().Scatter(ccurptrs, 1, MPIType<IT>(), &recvcount, 1, MPIType<IT>(), rankincol);

			if( recvcount == numeric_limits<IT>::max())
				break;
	
			// create space for incoming data ... 
			IT * tempinds = new IT[recvcount];
			NT * tempvals = new NT[recvcount];

			// receive actual data ... (first 4 arguments are ignored in the receiver side)
			commGrid->GetColWorld().Scatterv(inds, ccurptrs, cdispls, MPIType<IT>(), tempinds, recvcount,  MPIType<IT>(), rankincol); 
			commGrid->GetColWorld().Scatterv(vals, ccurptrs, cdispls, MPIType<NT>(), tempvals, recvcount,  MPIType<NT>(), rankincol); 

			/* Begin horizontal distribution, send data to the diagonal processor on this row */
			if(diagonal)
			{
				IT noffset = commGrid->GetRankInProcRow() * n_perproc; 
				for(IT i=zero; i< recvcount; ++i)
				{	
					localvec.push_back( make_pair(tempinds[i]-noffset, tempvals[i]) );
				}
			}
			else
			{
				diag = commGrid->GetDiagOfProcRow();
				commGrid->GetRowWorld().Send(&recvcount, 1, MPIType<IT>(), diag, RDTAGNNZ);	// send the size first						
				commGrid->GetRowWorld().Send(tempinds, recvcount, MPIType<IT>(), diag, RDTAGINDS);	// then the data
				commGrid->GetRowWorld().Send(tempvals, recvcount, MPIType<NT>(), diag, RDTAGVALS);
			}
			DeleteAll(tempinds, tempvals);	
		}
		if(!diagonal)
		{ 
			// Signal the end of data to the diagonal on this row	
			recvcount = numeric_limits<IT>::max();
			commGrid->GetRowWorld().Send(&recvcount, 1, MPIType<IT>(), diag, RDTAGNNZ);
		}
	}
	else		// remaining r * (s-1) processors 
	{		
		commGrid->GetWorld().Bcast(&total_n, 1, MPIType<IT>(), master);
		n_perproc = total_n / colneighs;

		if(diagonal)	// only the diagonals will receive the data
		{	
			while(total_n > 0)
			{
				commGrid->GetRowWorld().Recv(&recvcount, 1, MPIType<IT>(), rankinrow, RDTAGNNZ);	// receive the size first	
				if( recvcount == numeric_limits<IT>::max())
					break;
	
				// create space for incoming data ... 
				IT * tempinds = new IT[recvcount];
				NT * tempvals = new NT[recvcount];
		
				commGrid->GetRowWorld().Recv(tempinds, recvcount, MPIType<IT>(), rankinrow, RDTAGINDS);	// then receive the data
				commGrid->GetRowWorld().Recv(tempvals, recvcount, MPIType<NT>(), rankinrow, RDTAGVALS);

				IT noffset = commGrid->GetRankInProcRow() * n_perproc; 
				for(IT i=zero; i< recvcount; ++i)
				{	
					localvec.push_back( make_pair(tempinds[i]-noffset, tempvals[i]) );
				}
				DeleteAll(tempinds, tempvals);
			}
		}
	}	
	delete [] cdispls;
	arr = localvec;
 	length = (commGrid->GetRankInProcRow() != (commGrid->GetGridCols()-1))? n_perproc: (total_n - (n_perproc * (commGrid->GetGridCols()-1)));

	return infile;
}



