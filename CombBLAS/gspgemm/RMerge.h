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

	    d_A_ = ToDevice(h_A);
		d_B_ = ToDevice(h_B);
	}

	

	void
	mult(gstats *gst, mult_res<NT> *res, double *t_spgemm, double *t_spgemm_copy)
	{
		#ifdef DBG_SPEC_GPU_SPGEMM_TSTATS
		double t_tmp = MPI_Wtime();
		#endif
		
		RMerge2Functor mf = RMerge2Functor();
		SparseDeviceMatrixCSR<NT> d_C = mf(d_B_, d_A_);

		#ifdef DBG_SPEC_GPU_SPGEMM_TSTATS
		*t_spgemm = MPI_Wtime() - t_tmp; 
	 	t_tmp = MPI_Wtime();
		#endif

		res->nnzs	  = d_C.NonZeroCount();
		res->pidx_len = d_C.Height();
		res->pidx	  = new unsigned int[res->pidx_len+1];
		res->idx	  = new unsigned int[res->nnzs];
		res->vals	  = new NT[res->nnzs];
		res->ncols    = d_C.Width();
		cudaMemcpy(res->pidx, d_C.RowStarts().Data(),
				   sizeof(*res->pidx) * (res->pidx_len + 1),
				   cudaMemcpyDeviceToHost);
		cudaMemcpy(res->idx, d_C.ColIndices().Data(),
				   sizeof(*res->idx) * res->nnzs, cudaMemcpyDeviceToHost);
		cudaMemcpy(res->vals, d_C.Values().Data(),
				   sizeof(*res->vals) * res->nnzs, cudaMemcpyDeviceToHost);

		#ifdef DBG_SPEC_GPU_SPGEMM_TSTATS
		*t_spgemm_copy = MPI_Wtime() - t_tmp;
		#endif		

		return;
	}
};

#endif
