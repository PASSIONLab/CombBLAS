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
SpParVec<IT, NT>::SpParVec (): length(zero)
{
	commGrid.reset(new CommGrid(MPI::COMM_WORLD, 0, 0));
	
	if(commGrid->GetRankInProcRow() == commGrid->GetRankInProcCol())
		diagonal = true;
	else
		diagonal = false;	
};

template <class IT, class NT>
SpParVec<IT, IT> SpParVec<IT, NT>::sort()
{
	MPI::Intracomm DiagWorld = commGrid->GetDiagWorld();
	SpParVec<IT,IT> temp(commGrid);
	if(DiagWorld != MPI::COMM_NULL) // Diagonal processors only
	{
		pair<IT,IT> * vecpair = new pair<IT,IT>[length];

		int nproc = DiagWorld.Get_size();
		int diagrank = DiagWorld.Get_rank();

		long * dist = new long[nproc];
		dist[diagrank] = length;
		DiagWorld.Allgather(MPI::IN_PLACE, 0, MPI::DATATYPE_NULL, dist, 1, MPIType<long>());
		IT lengthuntil = accumulate(dist, dist+diagrank, 0);

		for(int i=0; i<length; ++i)
		{
			vecpair[i].first = num[i];	// we'll sort wrt numerical values
			vecpair[i].second = ind[i] + lengthuntil + 1;	
		}

		// less< pair<T1,T2> > works correctly (sorts wrt first elements)	
    		psort::parallel_sort (vecpair, vecpair + length,  dist, DiagWorld);

		vector< IT > nind(length);
		vector< IT > nnum(length);
		for(int i=0; i<length; ++i)
		{
			num[i] = vecpair[i].first;	// sorted range
			nind[i] = ind[i];		// make sure the sparsity distribution is the same
			nnum[i] = vecpair[i].second;	// inverse permutation stored as numerical values
		}
		delete [] vecpair;
		delete [] dist;

		temp.length = length;
		temp.ind = nind;
		temp.num = nnum;
	}
	return temp;
}
		
template <class IT, class NT>
SpParVec<IT,NT> & SpParVec<IT, NT>::operator+=(const SpParVec<IT,NT> & rhs)
{
	if(this != &rhs)		
	{	
		if(diagonal)	// Only the diagonal processors hold values
		{
			IT lsize = ind.size();
			IT rsize = rhs.ind.size();

			vector< IT > nind(lsize+rsize);
			vector< NT > nnum(lsize+rsize);

			IT i=0, j=0;
			while(i < lsize && j < rsize)
			{
				// assignment won't change the size of vector, push_back is necessary
				if(ind[i] > rhs.ind[j])
				{	
					nind.push_back( rhs.ind[j] );
					nnum.push_back( rhs.num[j++] );
				}
				else if(ind[i] < rhs.ind[j])
				{
					nind.push_back( ind[i] );
					nnum.push_back( num[i++] );
				}
				else
				{
					nind.push_back( ind[i].first );
					nnum.push_back( num[i++] + rhs.num[j++] );
				}
			}
			ind.swap(nind);		// ind will contain the elements of nind with capacity shrunk-to-fit size
			num.swap(nnum);
			length = ind.size();
		} 	
	}	
	return *this;
};	

//! Called on an existing object
template <class IT, class NT>
ifstream& SpParVec<IT,NT>::ReadDistribute (ifstream& infile, int master)
{
	length = zero;	// will be updated for diagonal processors
	IT total_n, total_nnz, n_perproc;
	MPI::Intracomm DiagWorld = commGrid->GetDiagWorld();
	if(DiagWorld != MPI::COMM_NULL)	// Diagonal processors only
	{
		int neighs = DiagWorld.Get_size();	// number of neighbors along diagonal (including oneself)
		IT buffperneigh = MEMORYINBYTES / (neighs * (sizeof(IT) + sizeof(NT)));

		IT * displs = new IT[neighs];
		for (int i=0; i<neighs; ++i)
			displs[i] = i*buffperneigh;

		IT * curptrs;
		IT recvcount;
		IT * inds; 
		NT * vals;

		int diagrank = DiagWorld.Get_rank();	
		if(diagrank == master)	// 1 processor only
		{		
			inds = new IT [ buffperneigh * neighs ];
			vals = new NT [ buffperneigh * neighs ];

			curptrs = new IT[neighs];
			fill_n(curptrs, neighs, (IT) zero);	// fill with zero
		
			if (infile.is_open())
			{
				infile.clear();
				infile.seekg(0);
				infile >> total_n >> total_nnz;
				n_perproc = total_n / neighs;	// the last proc gets the extras
				DiagWorld.Bcast(&total_n, 1, MPIType<IT>(), master);			
	
				IT tempind;
				NT tempval;
				IT cnz = 0;
				while ( (!infile.eof()) && cnz < total_nnz)
				{
					infile >> tempind;
					infile >> tempval;
					tempind--;

					int rec = std::min(tempind / n_perproc, neighs-1);	// precipient processor along the diagonal
					inds[ rec * buffperneigh + curptrs[rec] ] = tempind;
					vals[ rec * buffperneigh + curptrs[rec] ] = tempval;
					++ (curptrs[rec]);				

					if(curptrs[rec] == buffperneigh || (cnz == (total_nnz-1)) )		// one buffer is full, or file is done !
					{
						// first, send the receive counts ...
						DiagWorld.Scatter(curptrs, 1, MPIType<IT>(), &recvcount, 1, MPIType<IT>(), master);

						// generate space for own recv data ... (use arrays because vector<bool> is cripled, if NT=bool)
						IT * tempinds = new IT[recvcount];
						NT * tempvals = new NT[recvcount];
					
						// then, send all buffers that to their recipients ...
						DiagWorld.Scatterv(inds, curptrs, displs, MPIType<IT>(), tempinds, recvcount,  MPIType<IT>(), master); 
						DiagWorld.Scatterv(vals, curptrs, displs, MPIType<NT>(), tempvals, recvcount,  MPIType<NT>(), master); 
	
						// now push what is ours to tuples
						IT offset = master * n_perproc;
						for(IT i=zero; i< recvcount; ++i)
						{					
							ind.push_back( tempinds[i]-offset );
							num.push_back( tempvals[i] );
						}

						// reset current pointers so that we can reuse {inds,vals} buffers
						fill_n(curptrs, neighs, (IT) zero);
						DeleteAll(tempinds, tempvals);
					}
					++ cnz;
				}
				assert (cnz == total_nnz);
			
				// Signal the end of file to other processors along the diagonal
				fill_n(curptrs, neighs, numeric_limits<IT>::max());	
				DiagWorld.Scatter(curptrs, 1, MPIType<IT>(), &recvcount, 1, MPIType<IT>(), master);
			}
			else	// input file does not exist !
			{
				total_n = 0;	
				DiagWorld.Bcast(&total_n, 1, MPIType<IT>(), master);						
			}
			DeleteAll(inds,vals, curptrs);
		}
		else 	 	// (r-1) processors on the diagonal
		{
			DiagWorld.Bcast(&total_n, 1, MPIType<IT>(), master);
			n_perproc = total_n / neighs;

			while(total_n > 0)	// otherwise, input file do not exist
			{
				// first receive the receive counts ...
				DiagWorld.Scatter(curptrs, 1, MPIType<IT>(), &recvcount, 1, MPIType<IT>(), master);

				if( recvcount == numeric_limits<IT>::max())
					break;
	
				// create space for incoming data ... 
				IT * tempinds = new IT[recvcount];
				NT * tempvals = new NT[recvcount];

				// receive actual data ... (first 4 arguments are ignored in the receiver side)
				DiagWorld.Scatterv(inds, curptrs, displs, MPIType<IT>(), tempinds, recvcount,  MPIType<IT>(), master); 
				DiagWorld.Scatterv(vals, curptrs, displs, MPIType<NT>(), tempvals, recvcount,  MPIType<NT>(), master); 

				// now push what is ours to tuples
				IT offset = diagrank * n_perproc;
				for(IT i=zero; i< recvcount; ++i)
				{					
					ind.push_back( tempinds[i]-offset );
					num.push_back( tempvals[i] );
				}

				DeleteAll(tempinds, tempvals);
			}
		}
		delete [] displs;
 		length = (diagrank != neighs-1) ? n_perproc: (total_n - (n_perproc * (neighs-1)));
	}	
	commGrid->GetWorld().Barrier();
	return infile;
}




