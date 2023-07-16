/**
 * @file
 *	NSparse.h
 *
 * @author
 *	Oguz Selvitopi
 *
 * @date
 *	2018.10.XX
 *
 * @brief
 *	Adapter for nsparse
 *
 * @todo
 *
 * @note
 *
 */

#ifndef __NSPARSE_H__
#define __NSPARSE_H__

#include "nsparse.hpp"
#include "CSR.hpp"
#include "SpGEMM.hpp"
#include "HashSpGEMM.hpp"


template<typename NT>
class NSparse
{
private:
	CSR<int, NT> A_;
	CSR<int, NT> B_;
	
public:
	void
	copy(unsigned int A_nrow, unsigned int A_ncol, unsigned int A_nnz,
		 unsigned int *A_cp, unsigned int *A_ind, NT *A_val,
		 unsigned int B_nrow, unsigned int B_ncol, unsigned int B_nnz,
		 unsigned int *B_cp, unsigned int *B_ind, NT *B_val,
		 gstats *gst)
	{	
		A_.rpt = (int *)A_cp; A_.colids = (int *)A_ind; A_.values = A_val;
		A_.nrow = A_ncol; A_.ncolumn = A_nrow; A_.nnz = A_nnz;
		A_.device_malloc = false;
		B_.rpt = (int *)B_cp; B_.colids = (int *)B_ind; B_.values = B_val;
		B_.nrow = B_ncol; B_.ncolumn = B_nrow; B_.nnz = B_nnz;
		B_.device_malloc = false;

		A_.memcpyHtD();
		B_.memcpyHtD();
	}



	void
	mult(gstats *gst, mult_res<NT> *res, double *t_spgemm, double *t_spgemm_copy)
	
	{	
		CSR<int, NT> C;

		#ifdef DBG_SPEC_GPU_SPGEMM_TSTATS
		double t_tmp = MPI_Wtime();
		#endif
		
		SpGEMM_Hash(B_, A_, C);

		#ifdef DBG_SPEC_GPU_SPGEMM_TSTATS
		*t_spgemm = MPI_Wtime() - t_tmp;
		t_tmp = MPI_Wtime();
		#endif

		C.memcpyDtH();

		res->nnzs	  = C.nnz;
		res->pidx_len = C.nrow;
		res->pidx	  = (unsigned int *) C.rpt;
		res->idx	  = (unsigned int *) C.colids;
		res->vals	  = C.values;
		res->ncols    = C.ncolumn;

		#ifdef DBG_SPEC_GPU_SPGEMM_TSTATS
		*t_spgemm_copy = MPI_Wtime() - t_tmp;
		#endif

		A_.release_csr();
		B_.release_csr();
		C.release_csr();

		return;
	}
};


#endif
