/****************************************************************/
/* Sequential and Parallel Sparse Matrix Multiplication Library */
/* version 2.3 --------------------------------------------------/
/* date: 01/18/2009 ---------------------------------------------/
/* author: Aydin Buluc (aydin@cs.ucsb.edu) ----------------------/
/****************************************************************/

/**
 * Functions that are used by multiple parallel matrix classes, but don't need the "this" pointer
 **/

#ifndef _SP_PAR_HELPER_H_
#define _SP_PAR_HELPER_H_

#include <vector>
#include <mpi.h>
#include "LocArr.h"
#include "CommGrid.h"

using namespace std;

class SpParHelper
{
public:
	// input is already sorted
	template<typename _ForwardIter, typename IT>
	static void MEPSortInternal(_ForwardIter __first, _ForwardIter __last, IT * dist, MPI::Intracomm & comm)
	{
		
	}

	// Necessary because psort creates three 2D vectors of size p-by-p
	// One of those vector with 8 byte data uses 8*(4096)^2 = 128 MB space 
	// Per processor extra storage becomes:
	//	24 MB with 1K processors
	//	96 MB with 2K processors
	//	384 MB with 4K processors
	// 	1.5 GB with 8K processors
	template<typename KEY, typename VAL, typename IT>
	static void MemoryEfficientPSort(pair<KEY,VAL> * array, IT length, IT * dist, MPI::Intracomm & comm)
	{	
		int nprocs = comm.Get_size();
		if(nprocs < 2000)
		{
			psort::parallel_sort (array, array+length,  dist, comm);
		}
		else
		{
			IT gl_elements = accumulate(dist, dist+nprocs, 0);
			IT medianrank = gl_elements /2;	// global rank of the median element
			sort(array, array+length);	// re-sort because we might have swapped data in previous iterations

			int myrank = comm.Get_rank();
			int nsize = nprocs / 2;	// new size
			int color = myrank / nsize;
			halfcomm = comm.Split(color, myrank);	// split into two communicators
			
			IT begin = 0;
			IT end = length;	// initially everyone is active			
			KEY * medians = new KEY[nprocs];
			KEY * actives = new KEY[nprocs];	// KEY is float or double
			while(true)
			{
				KEY median = array[(begin + end)/2].first; 	// median of the active range
				IT active = end-begin;				// size of the active range

                		medians[myrank] = median;
				actives[myrank] = static_cast<KEY>(active);
	                	comm.Allgather(MPI::IN_PLACE, 0, MPIType<KEY>(), medians, 1, MPIType<KEY>());
	                	comm.Allgather(MPI::IN_PLACE, 0, MPIType<KEY>(), actives, 1, MPIType<KEY>());
				KEY totact = accumulate(actives, actives+nprocs, 0.0);
				transform(actives, actives+nprocs, actives, bind2nd(divides<KEY>(), totact));	// normalize
				transform(medians, medians+nprocs, actives, medians, multiplies<KEY>());	// weight medians
				nth_element( medians, medians+nprocs/2, medians+nprocs );
        	        	KEY wmm = medians[nprocs/2];	// weighted median of medians
				cout << "Weighted median of medians:" << wmm << endl;

				pair<KEY,VAL> wmmpair = make_pair(wmm, VAL());
				pair<KEY,VAL> * low =lower_bound (array+begin, array+end, wmmpair); 
				pair<KEY,VAL> * upp =lower_bound (array+begin, array+end, wmmpair); 
				KEY gl_low, gl_upp;
				comm.Allreduce( &low.first, &gl_low, 1, MPIType<KEY>(), MPI::SUM);
				comm.Allreduce( &upp.first, &gl_upp, 1, MPIType<KEY>(), MPI::SUM);
				cout << "GL_LOW: " << gl_low << ", GL_UPP: " << gl_upp << endl;

				if(gl_upp < gl_median)
				{
					begin = low - array;
				}
				else if(gl_median < gl_low)
				{
					
				}
				else
				{
					break;
				}
			}
			
	
			// __first and __last are local pointers
			if(color == 0)
				psort::parallel_sort (__first, __last,  dist, halfcomm);
			else
				psort::parallel_sort (__first, __last,  dist+nsize, halfcomm); 
		}
	}
	
	template<typename IT, typename NT, typename DER>
	static void FetchMatrix(SpMat<IT,NT,DER> & MRecv, const vector<IT> & essentials, vector<MPI::Win> & arrwin, int ownind);

	template<typename IT, typename NT, typename DER>	
	static void BCastMatrix(MPI::Intracomm & comm1d, SpMat<IT,NT,DER> & Matrix, const vector<IT> & essentials, int root);

	template<typename IT, typename NT, typename DER>
	static void SetWindows(MPI::Intracomm & comm1d, const SpMat< IT,NT,DER > & Matrix, vector<MPI::Win> & arrwin);

	template <typename IT, typename NT, typename DER>
	static void GetSetSizes(const SpMat<IT,NT,DER> & Matrix, IT ** & sizes, MPI::Intracomm & comm1d);

	template <typename IT, typename DER>
	static void AccessNFetch(DER * & Matrix, int owner, vector<MPI::Win> & arrwin, MPI::Group & group, IT ** sizes);

	template <typename IT, typename DER>
	static void LockNFetch(DER * & Matrix, int owner, vector<MPI::Win> & arrwin, MPI::Group & group, IT ** sizes);

	static void StartAccessEpoch(int owner, vector<MPI::Win> & arrwin, MPI::Group & group);
	static void PostExposureEpoch(int self, vector<MPI::Win> & arrwin, MPI::Group & group);
	static void LockWindows(int ownind, vector<MPI::Win> & arrwin);
	static void UnlockWindows(int ownind, vector<MPI::Win> & arrwin);
	static void SetWinErrHandler(vector<MPI::Win> & arrwin);	// set the error handler to THROW_EXCEPTIONS

	static void Print(const string & s)
	{
		int myrank = MPI::COMM_WORLD.Get_rank();
		if(myrank == 0)
		{
			cout << s;
		}
	}

	static void WaitNFree(vector<MPI::Win> & arrwin)
	{
		// End the exposure epochs for the arrays of the local matrices A and B
		// The Wait() call matches calls to Complete() issued by ** EACH OF THE ORIGIN PROCESSES ** 
		// that were granted access to the window during this epoch.

		for(int i=0; i< arrwin.size(); ++i)
		{
			arrwin[i].Wait();
		}
		FreeWindows(arrwin);
	}		
	
	static void FreeWindows(vector<MPI::Win> & arrwin)
	{
		for(int i=0; i< arrwin.size(); ++i)
		{
			arrwin[i].Free();
		}
	}
};

/**
  * @param[in,out] MRecv {an already existing, but empty SpMat<...> object}
  * @param[in] essentials {carries essential information (i.e. required array sizes) about ARecv}
  * @param[in] arrwin {windows array of size equal to the number of built-in arrays in the SpMat data structure}
  * @param[in] ownind {processor index (within this processor row/column) of the owner of the matrix to be received}
  * @remark {The communicator information is implicitly contained in the MPI::Win objects}
 **/
template <class IT, class NT, class DER>
void SpParHelper::FetchMatrix(SpMat<IT,NT,DER> & MRecv, const vector<IT> & essentials, vector<MPI::Win> & arrwin, int ownind)
{
	MRecv.Create(essentials);		// allocate memory for arrays
 
	Arr<IT,NT> arrinfo = MRecv.GetArrays();
	assert( (arrwin.size() == arrinfo.totalsize()));

	// C-binding for MPI::Get
	//	int MPI_Get(void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp,
        //    		int target_count, MPI_Datatype target_datatype, MPI_Win win)

	IT essk = 0;
	for(int i=0; i< arrinfo.indarrs.size(); ++i)	// get index arrays
	{
		//arrwin[essk].Lock(MPI::LOCK_SHARED, ownind, 0);
		arrwin[essk++].Get(arrinfo.indarrs[i].addr, arrinfo.indarrs[i].count, MPIType<IT>(), ownind, 0, arrinfo.indarrs[i].count, MPIType<IT>());
	}
	for(int i=0; i< arrinfo.numarrs.size(); ++i)	// get numerical arrays
	{
		//arrwin[essk].Lock(MPI::LOCK_SHARED, ownind, 0);
		arrwin[essk++].Get(arrinfo.numarrs[i].addr, arrinfo.numarrs[i].count, MPIType<NT>(), ownind, 0, arrinfo.numarrs[i].count, MPIType<NT>());
	}
}


/**
  * @param[in] Matrix {For the root processor, the local object to be sent to all others.
  * 		For all others, it is a (yet) empty object to be filled by the received data}
  * @param[in] essentials {irrelevant for the root}
 **/
template<typename IT, typename NT, typename DER>	
void SpParHelper::BCastMatrix(MPI::Intracomm & comm1d, SpMat<IT,NT,DER> & Matrix, const vector<IT> & essentials, int root)
{
	if(comm1d.Get_rank() != root)
	{
		Matrix.Create(essentials);		// allocate memory for arrays		
	}

	Arr<IT,NT> arrinfo = Matrix.GetArrays();
	for(int i=0; i< arrinfo.indarrs.size(); ++i)	// get index arrays
	{
		comm1d.Bcast(arrinfo.indarrs[i].addr, arrinfo.indarrs[i].count, MPIType<IT>(), root);					
	}
	for(int i=0; i< arrinfo.numarrs.size(); ++i)	// get numerical arrays
	{
		comm1d.Bcast(arrinfo.numarrs[i].addr, arrinfo.numarrs[i].count, MPIType<NT>(), root);					
	}			
}

template <class IT, class NT, class DER>
void SpParHelper::SetWindows(MPI::Intracomm & comm1d, const SpMat< IT,NT,DER > & Matrix, vector<MPI::Win> & arrwin) 
{	
	Arr<IT,NT> arrs = Matrix.GetArrays(); 
	 
	// static MPI::Win MPI::Win::create(const void *base, MPI::Aint size, int disp_unit, MPI::Info info, const MPI::Intracomm & comm);
	// The displacement unit argument is provided to facilitate address arithmetic in RMA operations
	// *** COLLECTIVE OPERATION ***, everybody exposes its own array to everyone else in the communicator
		
	for(int i=0; i< arrs.indarrs.size(); ++i)
	{
		arrwin.push_back(MPI::Win::Create(arrs.indarrs[i].addr, 
			arrs.indarrs[i].count * sizeof(IT), sizeof(IT), MPI::INFO_NULL, comm1d));
	}
	for(int i=0; i< arrs.numarrs.size(); ++i)
	{
		arrwin.push_back(MPI::Win::Create(arrs.numarrs[i].addr, 
			arrs.numarrs[i].count * sizeof(NT), sizeof(NT), MPI::INFO_NULL, comm1d));
	}	
}

inline void SpParHelper::LockWindows(int ownind, vector<MPI::Win> & arrwin)
{
	for(int i=0; i< arrwin.size(); ++i)
	{
		arrwin[i].Lock(MPI::LOCK_SHARED, ownind, 0);
	}
}

inline void SpParHelper::UnlockWindows(int ownind, vector<MPI::Win> & arrwin) 
{
	for(int i=0; i< arrwin.size(); ++i)
	{
		arrwin[i].Unlock(ownind);
	}
}

inline void SpParHelper::SetWinErrHandler(vector<MPI::Win> & arrwin)
{
	for(int i=0; i< arrwin.size(); ++i)
	{
		arrwin[i].Set_errhandler(MPI::ERRORS_THROW_EXCEPTIONS);
	}
}

/**
 * @param[in] owner {target processor rank within the processor group} 
 * @param[in] arrwin {start access epoch only to owner's arrwin (-windows) }
 */
inline void SpParHelper::StartAccessEpoch(int owner, vector<MPI::Win> & arrwin, MPI::Group & group)
{
	/* Now start using the whole comm as a group */
	int acc_ranks[1]; 
	acc_ranks[0] = owner;
	MPI::Group access = group.Incl(1, acc_ranks);	// take only the owner

	// begin the ACCESS epochs for the arrays of the remote matrices A and B
	// Start() *may* block until all processes in the target group have entered their exposure epoch
	for(int i=0; i< arrwin.size(); ++i)
		arrwin[i].Start(access, 0); 
}

/**
 * @param[in] self {rank of "this" processor to be excluded when starting the exposure epoch} 
 */
inline void SpParHelper::PostExposureEpoch(int self, vector<MPI::Win> & arrwin, MPI::Group & group)
{
	MPI::Group exposure = group;
	
	// begin the EXPOSURE epochs for the arrays of the local matrices A and B
	for(int i=0; i< arrwin.size(); ++i)
		arrwin[i].Post(exposure, MPI_MODE_NOPUT);
}

template <class IT, class DER>
void SpParHelper::AccessNFetch(DER * & Matrix, int owner, vector<MPI::Win> & arrwin, MPI::Group & group, IT ** sizes)
{
	StartAccessEpoch(owner, arrwin, group);	// start the access epoch to arrwin of owner

	vector<IT> ess(DER::esscount);			// pack essentials to a vector
	for(int j=0; j< DER::esscount; ++j)	
		ess[j] = sizes[j][owner];	

	Matrix = new DER();	// create the object first	
	FetchMatrix(*Matrix, ess, arrwin, owner);	// then start fetching its elements
}

template <class IT, class DER>
void SpParHelper::LockNFetch(DER * & Matrix, int owner, vector<MPI::Win> & arrwin, MPI::Group & group, IT ** sizes)
{
	LockWindows(owner, arrwin);

	vector<IT> ess(DER::esscount);			// pack essentials to a vector
	for(int j=0; j< DER::esscount; ++j)	
		ess[j] = sizes[j][owner];	

	Matrix = new DER();	// create the object first	
	FetchMatrix(*Matrix, ess, arrwin, owner);	// then start fetching its elements
}


/**
 * @param[in] sizes 2D array where 
 *  	sizes[i] is an array of size r/s representing the ith essential component of all local blocks within that row/col
 *	sizes[i][j] is the size of the ith essential component of the jth local block within this row/col
 */
template <class IT, class NT, class DER>
void SpParHelper::GetSetSizes(const SpMat<IT,NT,DER> & Matrix, IT ** & sizes, MPI::Intracomm & comm1d)
{
	vector<IT> essentials = Matrix.GetEssentials();
	int index = comm1d.Get_rank();
	for(IT i=0; i< essentials.size(); ++i)
	{
		sizes[i][index] = essentials[i]; 
		comm1d.Allgather(MPI::IN_PLACE, 1, MPIType<IT>(), sizes[i], 1, MPIType<IT>());
	}	
}

#endif
