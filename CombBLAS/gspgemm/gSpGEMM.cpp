/**
 * @file
 *	gSpGEMM.cpp
 *
 * @author
 *	Oguz Selvitopi
 *
 * @date
 *	2018.10.XX
 *
 * @brief
 *	Generic routines to perform SpGEMM on GPUs (implementation)
 *
 * @todo
 *
 * @note
 *
 */

#include <unistd.h>
#include <algorithm>
#include <cassert>
#include <string>
#include "mpi.h"

#include "gSpGEMM.h"
#include "RMerge.h"
#include "Bhsparse.h"
#include "NSparse.h"
#include "NSparseSymbolic.h"

#define CP  0
#define JC  1
#define IR	2
#define NUM 0


////////////////////////////////////////////////////////////////////////////////
// 							Constructors/Destructors                          //
////////////////////////////////////////////////////////////////////////////////
template<class NT>
GSpGEMM<NT>::GSpGEMM (
	int		rank,
	int		deviceId,
	bool	compute_flops
					  ):
	rank_(rank),
	deviceId_(deviceId),
	compute_flops_(compute_flops)
{
	gst_ = {};
}



template<class NT>
GSpGEMM<NT>::~GSpGEMM ()
{
}



////////////////////////////////////////////////////////////////////////////////
// 								Private members                               //
////////////////////////////////////////////////////////////////////////////////
/*
  Updates:
    type conversion assumes IT is 32-bit integer
 */
template<class NT>
template<typename IT>
void
GSpGEMM<NT>::prep_in (
    const combblas::SpDCCols<IT, NT> &A,
	const combblas::SpDCCols<IT, NT> &B
					  )
{
	#ifdef DBG_SPEC_GPU_SPGEMM_MSTATS
	size_t free_mem, total_mem;
	cudaMemGetInfo(&free_mem, &total_mem);
	gst_.m_d_before_pre = total_mem-free_mem;
	#endif

	combblas::Arr<IT, NT> Alocarrs = A.GetArrays();
	combblas::Arr<IT, NT> Blocarrs = B.GetArrays();

	#ifdef DBG_SPEC_GPU_SPGEMM_TSTATS
	double t_tmp = MPI_Wtime();
	#endif

	// prepare A
	A_cp_ = new unsigned int[A.getncol()+1];
	assert (A_cp_ != NULL && "malloc failed for A_cp_\n");
	A_cp_[0] = Alocarrs.indarrs[CP].addr[0];
	unsigned int curcol = 0;
	unsigned int curval = 0;
	for (unsigned int i = 0; i < A.getnzc(); ++i)
	{
		while (curcol != Alocarrs.indarrs[JC].addr[i])
			A_cp_[++curcol] = curval;
		curval = Alocarrs.indarrs[CP].addr[i+1];
	}
	curcol = Alocarrs.indarrs[JC].addr[A.getnzc()-1];
	curval = Alocarrs.indarrs[CP].addr[A.getnzc()];
	while (curcol < A.getncol()) // leftover
		A_cp_[++curcol] = curval;

	// prepare B
	B_cp_ = new unsigned int[B.getncol()+1];
	assert (B_cp_ != NULL && "malloc failed for B_cp_\n");
	B_cp_[0] = Blocarrs.indarrs[CP].addr[0];
	curcol = 0;
	curval = 0;
	for (unsigned int i = 0; i < B.getnzc(); ++i)
	{
		while (curcol != Blocarrs.indarrs[JC].addr[i])
			B_cp_[++curcol] = curval;
		curval = Blocarrs.indarrs[CP].addr[i+1];
	}
	curcol = Blocarrs.indarrs[JC].addr[B.getnzc()-1];
	curval = Blocarrs.indarrs[CP].addr[B.getnzc()];
	while (curcol < B.getncol()) // leftover
		B_cp_[++curcol] = curval;

	#ifdef DBG_SPEC_GPU_SPGEMM_TSTATS
	gst_.t_pre_decomp += MPI_Wtime() - t_tmp;
	gst_.nnzs_A += A.getnzc();
	gst_.nnzs_B += B.getnzc();
	t_tmp = MPI_Wtime();
	#endif

	// convert IR to unsigned int
	A_ind_ = (unsigned int *)Alocarrs.indarrs[IR].addr;
	// A_ind_ = new unsigned int[A.getnnz()];
	// assert (A_ind_ != NULL && "malloc failed for A_ind_\n");
	// for (unsigned int i = 0; i < A.getnnz(); ++i)
	//	A_ind_[i] = static_cast<unsigned int>(Alocarrs.indarrs[IR].addr[i]);

	B_ind_ = (unsigned int *)Blocarrs.indarrs[IR].addr;
	// B_ind_ = new unsigned int[B.getnnz()];
	// assert (B_ind_ != NULL && "malloc failed for B_ind_\n");
	// for (unsigned int i = 0; i < B.getnnz(); ++i)
	// 	B_ind_[i] = static_cast<unsigned int>(Blocarrs.indarrs[IR].addr[i]);

	#ifdef DBG_SPEC_GPU_SPGEMM_TSTATS
	gst_.t_pre_conv += MPI_Wtime() - t_tmp;
	t_tmp = MPI_Wtime();
	#endif

	A_val_ = Alocarrs.numarrs[NUM].addr;
	B_val_ = Blocarrs.numarrs[NUM].addr;

	#ifdef DBG_SPEC_GPU_SPGEMM_MSTATS
	// A_cp_ and B_cp_
	gst_.m_h_dcsc = (A.getncol()+1) * sizeof(unsigned int) +
		(B.getncol()+1) * sizeof(unsigned int);
	// A_ind_ and B_ind_
	gst_.m_h_conv = (A.getnnz()) * sizeof(unsigned int) +
		(B.getnnz()) * sizeof(unsigned int);
	// input mats
	gst_.m_d_input = ((A.getncol()+1) * sizeof(unsigned int)) +
		(A.getnnz() * sizeof(unsigned int)) + (A.getnnz() * sizeof(NT)) +
		((B.getncol()+1) * sizeof(unsigned int)) +
		(B.getnnz() * sizeof(unsigned int)) + (B.getnnz() * sizeof(NT));
	free_mem, total_mem;
	cudaMemGetInfo(&free_mem, &total_mem);
	gst_.m_d_after_pre = total_mem-free_mem;
	#endif

	return;
}



template<class NT>
void
GSpGEMM<NT>::report_memusage (
    const std::string &unitstr
							  )
{
	int			prec  = 0;
	int			width = 4;
	double		div	  = 1<<20;
	std::string unit  = " MB";
	if (!unitstr.compare(std::string("GB")))
	{
		prec = 2;
		div	 = 1<<30;
		unit = " GB";
	}

	// lfile_ << "mem stats, unit" << unit << std::endl;
	// lfile_ << std::fixed << std::setprecision(prec) << std::setw(width);
	// lfile_ << "[gpu] mem usage before pre "
	// 	   << (gst_.m_d_before_pre/div) << unit << std::endl;
	// lfile_ << "[cpu] extra mem for " << std::endl;
	// lfile_ << "  dcsc " << (gst_.m_h_dcsc/div) << unit << std::endl;
	// lfile_ << "  conv " << (gst_.m_h_conv/div) << unit << std::endl;
	// lfile_ << "[gpu] device input mats "
	// 	   << (gst_.m_d_input/div) << unit << std::endl;
	// lfile_ << "[gpu] mem usage after pre "
	// 	   << (gst_.m_d_after_pre/div) << unit << std::endl;
	// lfile_ << "[cpu] output mat size copied to host "
	// 	   << (gst_.m_h_output/div) << unit << std::endl;
	// lfile_ << "[gpu] mem usage after spgemm "
	// 	   << (gst_.m_d_after_spgemm/div) << unit << std::endl;
	// lfile_ << "[cpu] output tuple size "
	// 	   << (gst_.m_h_ctuples/div) << unit << std::endl;
	// double m_cpu_total = gst_.m_h_dcsc + gst_.m_h_conv + gst_.m_h_output +
	// 	gst_.m_h_ctuples;
	// lfile_ << "[cpu] extra mem used in cpu for pre "
	// 	   << ((gst_.m_h_dcsc + gst_.m_h_conv)/div) << unit << std::endl;
	// lfile_ << "[cpu] extra mem used in cpu for post "
	// 	   << ((gst_.m_h_output + gst_.m_h_ctuples)/div) << unit << std::endl;
	// lfile_ << "[gpu] mem usage upon exit "
	// 	   << (gst_.m_d_exit/div) << unit << std::endl;

	return;
}



// make sure prep_in called before calling this
template<class NT>
int64_t
GSpGEMM<NT>::get_flops (void)
{
	// note we compute B^T A^T
	HostVector<int64_t> tmp(B_ncols_);

	#ifdef THREADED
	#pragma omp parallel for
	#endif
	for (unsigned int r = 0; r < B_ncols_; ++r)
	{
		int64_t nops = 0;
		for (unsigned int c = B_cp_[r]; c < B_cp_[r+1]; ++c)
		{
			unsigned int cidx = B_ind_[c];
			nops += A_cp_[cidx + 1] - A_cp_[cidx];
		}
		tmp[r] = nops;
	}

	return 2 * Sum(tmp);
}



////////////////////////////////////////////////////////////////////////////////
// 								 Public members                               //
////////////////////////////////////////////////////////////////////////////////
template<class NT>
void
GSpGEMM<NT>::report_time (
    std::ofstream	&lfile,
	int				 ncalls
						  )
{
	double	t_pre_total	   = gst_.t_pre_decomp + gst_.t_pre_conv + gst_.t_pre_copy +
		gst_.t_pre_flops;
	double	t_spgemm_total = gst_.t_spgemm + gst_.t_spgemm_copy;
	double	t_total		   = t_pre_total + t_spgemm_total + gst_.t_post_tuple;
	lfile << "time in seconds" << std::endl;
	lfile << std::fixed;
	lfile << "pre: "
		  << std::setprecision(4)
		  << t_pre_total << " "
		  << std::setw(2) << std::setprecision(0)
		  << (t_pre_total/t_total*100) << "%"
		  << std::endl;
	lfile << "  decomp: "
		  << std::setprecision(4)
		  << gst_.t_pre_decomp << " "
		  << std::setw(2) << std::setprecision(0)
		  << (gst_.t_pre_decomp/t_pre_total*100) << "%"
		  << std::endl;
	lfile << "  conv  : "
		  << std::setprecision(4)
		  << gst_.t_pre_conv << " "
		  << std::setw(2) << std::setprecision(0)
		  << (gst_.t_pre_conv/t_pre_total*100) << "%"
		  << std::endl;
	lfile << "  copy  : "
		  << std::setprecision(4)
		  << gst_.t_pre_copy << " "
		  << std::setw(2) << std::setprecision(0)
		  << (gst_.t_pre_copy/t_pre_total*100) << "%"
		  << std::endl;
	lfile << "  flops : "
		  << std::setprecision(4)
		  << gst_.t_pre_flops << " "
		  << std::setw(2) << std::setprecision(0)
		  << (gst_.t_pre_flops/t_pre_total*100) << "%"
		  << std::endl;

	lfile << "spgemm: "
		  << std::setprecision(4)
		  << t_spgemm_total << " "
		  << std::setw(2) << std::setprecision(0)
		  << (t_spgemm_total/t_total*100) << "%"
		  << std::endl;
	lfile << "  mult: "
		  << std::setprecision(4)
		  << gst_.t_spgemm << " "
		  << std::setw(2) << std::setprecision(0)
		  << (gst_.t_spgemm/t_spgemm_total*100) << "%"
		  << std::endl;
	lfile << "  copy: "
		  << std::setprecision(4)
		  << gst_.t_spgemm_copy << " "
		  << std::setw(2) << std::setprecision(0)
		  << (gst_.t_spgemm_copy/t_spgemm_total*100) << "%"
		  << std::endl;

	lfile << "post: "
		  << std::setprecision(4)
		  << gst_.t_post_tuple << " "
		  << std::setw(2) << std::setprecision(0)
		  << (gst_.t_post_tuple/t_total*100) << "%"
		  << std::endl;

	lfile << "total: "
		  << std::setprecision(4)
		  << t_total << std::endl;

	lfile << "total (measured): "
		  << std::setprecision(4)
		  << gst_.t_total << std::endl;

	lfile << "cumulative gflops: "
		  << std::setprecision(2)
		  << (static_cast<double>(gst_.flops_total)/
			  (1e9*gst_.t_spgemm)) << std::endl;

	lfile << "compression ratio: "
		  << std::setprecision(2)
		  << (gst_.compr_ratio/ncalls) << std::endl;

	lfile << "nnzs A: " << (gst_.nnzs_A/ncalls) << std::endl;
	lfile << "nnzs B: " << (gst_.nnzs_B/ncalls) << std::endl;
	lfile << "nnzs C: " << (gst_.nnzs_C/ncalls) << std::endl;
}



template<class NT>
template<typename IT, typename MulFunctor>
combblas::SpTuples<IT, NT> *
GSpGEMM<NT>::mult (
	const combblas::SpDCCols<IT, NT>	&A,
	const combblas::SpDCCols<IT, NT>	&B,
	bool								 clearA,
	bool								 clearB,
	MulFunctor							 mf,
	int									 iter,
	int									 phase,
	int									 stage,
	pthread_cond_t						*input_freed,
	pthread_mutex_t						*mutex
				   )
{
	// cudaSetDevice(deviceId_);
	// #ifdef LOG_GNRL_GPU_SPGEMM
	// lfile_ << "rank " << rank_ << " device " << deviceId_ << std::endl;
	// lfile_ << "iter " << iter << " phase "  << phase
	// 	   << " stage " << stage << std::endl;
	// lfile_ << "A nrows " << A.getnrow() << " ncols " << A.getncol()
	// 	   << " nnzs " << A.getnnz()
	// 	   <<  " B nrows " << B.getnrow() << " ncols " << B.getncol()
	// 	   << " nnzs " << B.getnnz() << std::endl;
	// #endif

	// report_memusage("GB");
	A_nrows_ = A.getnrow(); A_ncols_ = A.getncol(); A_nnzs_ = A.getnnz();
	B_nrows_ = B.getnrow(); B_ncols_ = B.getncol(); B_nnzs_ = B.getnnz();

	#ifdef DBG_SPEC_GPU_SPGEMM_TSTATS
	double t_tmp_all = MPI_Wtime();
	#endif

	prep_in(A, B);

	if (compute_flops_)
	{
		#ifdef DBG_SPEC_GPU_SPGEMM_TSTATS
		double t_tmp = MPI_Wtime();
		#endif

		gst_.flops_current	= get_flops();
		gst_.flops_total   += gst_.flops_current;

		#ifdef DBG_SPEC_GPU_SPGEMM_TSTATS
		gst_.t_pre_flops += MPI_Wtime() - t_tmp;
		#endif
	}

	mf.copy(A_nrows_, A_ncols_, A_nnzs_, A_cp_, A_ind_, A_val_,
			B_nrows_, B_ncols_, B_nnzs_, B_cp_, B_ind_, B_val_, &gst_);

	if(clearA)
        delete const_cast<combblas::SpDCCols<IT, NT> *>(&A);
    if(clearB)
        delete const_cast<combblas::SpDCCols<IT, NT> *>(&B);

	// main thread is assured to be waiting for this condition
	pthread_mutex_lock(mutex);
	pthread_cond_signal(input_freed);
	pthread_mutex_unlock(mutex);

	combblas::SpTuples<IT, NT> *ctuples = mf.mult<IT>(&gst_);

	#ifdef DBG_SPEC_GPU_SPGEMM_MSTATS
	report_memusage("GB");
	#endif

	
	#ifdef DBG_SPEC_GPU_SPGEMM_TSTATS
	gst_.t_total += MPI_Wtime() - t_tmp_all;
	// report_time(lfile_);
	#endif

	delete[] A_cp_;
	// delete[] A_ind_;
	delete[] B_cp_;
	// delete[] B_ind_;

	return ctuples;
}




template<class NT>
template<typename IT, typename MulFunctor>
IT
GSpGEMM<NT>::mult_symb (
    const combblas::SpDCCols<IT, NT>	&A,
	const combblas::SpDCCols<IT, NT>	&B,
	MulFunctor							 mf
						)
{
	// report_memusage("GB");
	A_nrows_ = A.getnrow(); A_ncols_ = A.getncol(); A_nnzs_ = A.getnnz();
	B_nrows_ = B.getnrow(); B_ncols_ = B.getncol(); B_nnzs_ = B.getnnz();

	prep_in(A, B);

	IT c_nnz = mf.mult<IT, NT>(A_nrows_, A_ncols_, A_nnzs_,
							   A_cp_, A_ind_, A_val_,
							   B_nrows_, B_ncols_, B_nnzs_,
							   B_cp_, B_ind_, B_val_,
							   &gst_);

	delete[] A_cp_;
	// delete[] A_ind_;
	delete[] B_cp_;
	// delete[] B_ind_;

	return c_nnz;
}
