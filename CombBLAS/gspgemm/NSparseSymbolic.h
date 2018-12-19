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

#ifndef __NSPARSESYMBOLIC_H__
#define __NSPARSESYMBOLIC_H__

#include <cuda.h>
#include <helper_cuda.h>

#include "nsparse.hpp"
#include "CSR.hpp"
#include "SpGEMM.hpp"
#include "HashSpGEMM.hpp"


class NSparseSymbolic
{
public:
	template<typename IT, typename NT>
	IT
	mult(unsigned int A_nrow, unsigned int A_ncol, unsigned int A_nnz,
		 unsigned int *A_cp, unsigned int *A_ind, NT *A_val,
		 unsigned int B_nrow, unsigned int B_ncol, unsigned int B_nnz,
		 unsigned int *B_cp, unsigned int *B_ind, NT *B_val,
		 gstats *gst)
	{
		CSR<int, NT> A;
		CSR<int, NT> B;
		CSR<int, NT> C;

		A.rpt = (int *)A_cp; A.colids = (int *)A_ind; A.values = A_val;
		A.nrow = A_ncol; A.ncolumn = A_nrow; A.nnz = A_nnz;
		A.devise_malloc = false;
		B.rpt = (int *)B_cp; B.colids = (int *)B_ind; B.values = B_val;
		B.nrow = B_ncol; B.ncolumn = B_nrow; B.nnz = B_nnz;
		B.devise_malloc = false;

		#ifdef DBG_SPEC_GPU_SPGEMM_TSTATS
		double t_tmp = MPI_Wtime();
		#endif

		A.memcpyHtD();
		B.memcpyHtD();

		IT c_nnz = SpGEMM_Hash_Symbolic(B, A, C);

		#ifdef DBG_SPEC_GPU_SPGEMM_TSTATS
		t_tmp = MPI_Wtime() - t_tmp;
		gst->t_spgemm_symbolic += t_tmp;
		#endif

		A.release_csr();
		B.release_csr();
		// C is released in called function

		return c_nnz;
	}
};


#endif
