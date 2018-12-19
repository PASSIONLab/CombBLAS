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

#include <cuda.h>
#include <helper_cuda.h>

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
		A_.devise_malloc = false;
		B_.rpt = (int *)B_cp; B_.colids = (int *)B_ind; B_.values = B_val;
		B_.nrow = B_ncol; B_.ncolumn = B_nrow; B_.nnz = B_nnz;
		B_.devise_malloc = false;

		#ifdef DBG_SPEC_GPU_SPGEMM_TSTATS
		double t_tmp = MPI_Wtime();
		#endif

		A_.memcpyHtD();
		B_.memcpyHtD();

		#ifdef DBG_SPEC_GPU_SPGEMM_TSTATS
		gst->t_pre_copy += MPI_Wtime() - t_tmp;
		#endif
	}

	
	template<typename IT>
	combblas::SpTuples<IT, NT> *
	mult(gstats *gst)
	
	{	
		CSR<int, NT> C;

		#ifdef DBG_SPEC_GPU_SPGEMM_TSTATS
		double t_tmp = MPI_Wtime();
		#endif
		
		SpGEMM_Hash(B_, A_, C);

		#ifdef DBG_SPEC_GPU_SPGEMM_TSTATS
		double t_spgemm = MPI_Wtime() - t_tmp;
		gst->t_spgemm += t_spgemm;
		gst->compr_ratio +=
			static_cast<double>(gst->flops_current)/C.nnz/2;
		gst->nnzs_C += C.nnz;
		t_tmp = MPI_Wtime();
		#endif

		C.memcpyDtH();

		#ifdef DBG_SPEC_GPU_SPGEMM_TSTATS
		gst->t_spgemm_copy += MPI_Wtime() - t_tmp;
		#endif

		A_.release_csr();
		B_.release_csr();
		C.release_csr();

		#ifdef DBG_SPEC_GPU_SPGEMM_MSTATS
		size_t free_mem, total_mem;
		cudaMemGetInfo(&free_mem, &total_mem);
		gst->m_d_after_spgemm = total_mem - free_mem;
		#endif

		#ifdef DBG_SPEC_GPU_SPGEMM_TSTATS
		t_tmp = MPI_Wtime();
		#endif

		// std::cout << "C nrow: " << C.nrow
		// 		  << ", C ncol: " << C.ncolumn
		// 		  << ", C nnz: " << C.nnz << std::endl;

		std::tuple<IT,IT,NT> * tuplesC =
			static_cast<std::tuple<IT,IT,NT> *>
			(::operator new (sizeof(std::tuple<IT,IT,NT>[C.nnz])));
		for (int i = 0; i < C.nrow; ++i)
		{
			for (int c = C.rpt[i]; c < C.rpt[i+1]; ++c)
			{
				tuplesC[c] =
					std::make_tuple(static_cast<IT>(C.colids[c]), i,
									C.values[c]);
				// std::cout << "(" << C.colids[c] << ", " << i
				// 		  << ", " << C.values[c] << ") " << std::endl;
			}
		}

		#ifdef DBG_SPEC_GPU_SPGEMM_TSTATS
		gst->t_post_tuple += MPI_Wtime() - t_tmp;
		#endif

		#ifdef DBG_SPEC_GPU_SPGEMM_MSTATS
		gst->m_h_output = ((C.nrow+1) * sizeof(int)) +
			(C.nnz * sizeof(int)) + (C.nnz * sizeof(NT));
		gst->m_h_ctuples = C.nnz *
			(sizeof(IT) + sizeof(IT) + sizeof(NT));
		cudaMemGetInfo(&free_mem, &total_mem);
		gst->m_d_exit = total_mem - free_mem;
		#endif

		C.release_cpu_csr();

		return new combblas::SpTuples<IT, NT>
			(C.nnz, C.ncolumn, C.nrow, tuplesC, true, true);
	}
};


#endif
