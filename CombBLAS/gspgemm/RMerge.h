/**
 * @file
 *	RMerge.h
 *
 * @author
 *	Oguz Selvitopi
 *
 * @date
 *	2018.10.XX
 *
 * @brief
 *	Adapter for rmerge2
 *
 * @todo
 *
 * @note
 *
 */

#ifndef __RMERGE_H__
#define __RMERGE_H__

#define IGNORE_MKL

#include "tdef.h"

#include "mpi.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cusparse_v2.h"

#include "HostMatrix/SparseHostMatrixCSR.h"
#include "DeviceMatrix/SparseDeviceMatrixCSR.h"
#include "HostMatrix/SparseHostMatrixCSROperations.h"
#include "DeviceMatrix/SpmmWarp.h"
#include "DeviceMatrix/SpmmWarpN.h"
#include "DeviceMatrix/SpmmWarpNSorted.h"
#include "General/WallTime.h"
#include "SpmmFunctors.h"


template<typename NT>
class RMerge
{
private:
	SparseDeviceMatrixCSR<NT> d_A_;
	SparseDeviceMatrixCSR<NT> d_B_;
	
public:

	void
	copy(int A_nrow, int A_ncol, int A_nnz,
		 unsigned int *A_cp, unsigned int *A_ind, NT *A_val,
		 int B_nrow, int B_ncol, int B_nnz,
		 unsigned int *B_cp, unsigned int *B_ind, NT *B_val,
		 gstats *gst)
	{
		HostVector<unsigned int> A_cp_hv(A_cp, A_ncol+1);
		HostVector<unsigned int> A_ind_hv(A_ind, A_nnz);
		HostVector<NT> A_val_hv(A_val, A_nnz);
		SparseHostMatrixCSR<NT> h_A(A_nrow, A_ncol,
									A_val_hv, A_ind_hv, A_cp_hv);

		HostVector<unsigned int> B_cp_hv(B_cp, B_ncol+1);
		HostVector<unsigned int> B_ind_hv(B_ind, B_nnz);
		HostVector<NT> B_val_hv(B_val, B_nnz);
		SparseHostMatrixCSR<NT> h_B(B_nrow, B_ncol,
									B_val_hv, B_ind_hv, B_cp_hv);

		#ifdef DBG_SPEC_GPU_SPGEMM_TSTATS
		double t_tmp = MPI_Wtime();
		#endif

	    d_A_ = ToDevice(h_A);	// copies data to GPU
		d_B_ = ToDevice(h_B);	// copies data to GPU
		
		#ifdef DBG_SPEC_GPU_SPGEMM_TSTATS
		gst->t_pre_copy += MPI_Wtime() - t_tmp;
		#endif
	}

	
	template<typename IT>
	combblas::SpTuples<IT, NT> *
	mult(gstats *gst)
	{
		#ifdef DBG_SPEC_GPU_SPGEMM_TSTATS
		double t_tmp = MPI_Wtime();
		#endif
		
		RMerge2Functor mf = RMerge2Functor();
		SparseDeviceMatrixCSR<NT> d_C = mf(d_B_, d_A_);

		#ifdef DBG_SPEC_GPU_SPGEMM_TSTATS
		double t_spgemm = MPI_Wtime() - t_tmp;	
		gst->t_spgemm += t_spgemm;
		gst->compr_ratio +=
			static_cast<double>(gst->flops_current)/d_C.NonZeroCount()/2;
		gst->nnzs_C += d_C.NonZeroCount();	
		#endif

		#ifdef DBG_SPEC_GPU_SPGEMM_MSTATS
		size_t free_mem, total_mem;
		cudaMemGetInfo(&free_mem, &total_mem);
		gst->m_d_after_spgemm = total_mem - free_mem;
		#endif

		#ifdef DBG_SPEC_GPU_SPGEMM_TSTATS
	 	t_tmp = MPI_Wtime();
		#endif

		SparseHostMatrixCSR<NT> h_C = ToHost(d_C);

		#ifdef DBG_SPEC_GPU_SPGEMM_TSTATS
		gst->t_spgemm_copy += MPI_Wtime() - t_tmp;
		t_tmp = MPI_Wtime();
		#endif

		std::tuple<IT,IT,NT> * tuplesC =
			static_cast<std::tuple<IT,IT,NT> *>
			(::operator new (sizeof(std::tuple<IT,IT,NT>[h_C.NonZeroCount()])));

		NT				*values		= h_C.Values().Data();
		unsigned int	*colIndices = h_C.ColIndices().Data();
		unsigned int	*rowStarts	= h_C.RowStarts().Data();
		int				 height		= h_C.Height();
		for (IT r = 0; r < height; ++r)
			for (unsigned int c = rowStarts[r]; c < rowStarts[r+1]; ++c)
				tuplesC[c] =
					std::make_tuple(static_cast<IT>(colIndices[c]),
									r, values[c]);

		#ifdef DBG_SPEC_GPU_SPGEMM_TSTATS
		gst->t_post_tuple += MPI_Wtime() - t_tmp;
		#endif

		#ifdef DBG_SPEC_GPU_SPGEMM_MSTATS
		gst->m_h_output = ((h_C.Height()+1) * sizeof(unsigned int)) +
			(h_C.NonZeroCount() * sizeof(unsigned int)) +
			(h_C.NonZeroCount() * sizeof(NT));
		gst->m_h_ctuples = h_C.NonZeroCount() *
			(sizeof(IT) + sizeof(IT) + sizeof(NT));
		cudaMemGetInfo(&free_mem, &total_mem);
		gst->m_d_exit = total_mem - free_mem;
		#endif

		return new combblas::SpTuples<IT, NT>
			(h_C.NonZeroCount(), h_C.Width(),
			 h_C.Height(), tuplesC, true, true);
	}
};

#endif
