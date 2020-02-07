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
#include <thread>
#include <tuple>
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
	bool	compute_flops
					  ):
	rank_(rank),
	compute_flops_(compute_flops)
	
{
	gst_ = {};
	cudaGetDeviceCount(&ndevices_);
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
template<typename IT, template<typename T> class Mul>
combblas::SpTuples<IT, NT> *
GSpGEMM<NT>::mult (
	const combblas::SpDCCols<IT, NT>	&A,
	const combblas::SpDCCols<IT, NT>	&B,
	bool								 clearA,
	bool								 clearB,
	int									 iter,
	int									 phase,
	int									 stage,
	pthread_cond_t						*input_freed,
	pthread_mutex_t						*mutex
				   )
{
	// report_memusage("GB");
	A_nrows_ = A.getnrow(); A_ncols_ = A.getncol(); A_nnzs_ = A.getnnz();
	B_nrows_ = B.getnrow(); B_ncols_ = B.getncol(); B_nnzs_ = B.getnnz();

	#ifdef LOG_GNRL_GPU_SPGEMM
	int64_t mem_A_cp = (A_ncols_ + 1) * sizeof(unsigned int);
	int64_t mem_A_ind = A_nnzs_ * sizeof(unsigned int);
	int64_t mem_A_val = A_nnzs_ * sizeof(NT);
	int64_t mem_B_cp = (B_ncols_ + 1) * sizeof(unsigned int);
	int64_t mem_B_ind = B_nnzs_ * sizeof(unsigned int);
	int64_t mem_B_val = B_nnzs_ * sizeof(NT);
	size_t free_mem, total_mem;
	cudaMemGetInfo(&free_mem, &total_mem);
	(*lfile_) << std::fixed << std::setprecision(2)
			  << "mem info: Acp " << static_cast<double>(mem_A_cp) / (1<<30)
			  << " Aind " << static_cast<double>(mem_A_ind) / (1<<30)
			  << " Aval " << static_cast<double>(mem_A_val) / (1<<30)
			  << " Bcp " << static_cast<double>(mem_B_cp) / (1<<30)
			  << " Bind " << static_cast<double>(mem_B_ind) / (1<<30)
			  << " Bval " << static_cast<double>(mem_B_val) / (1<<30)
			  << " --- gpu total " << static_cast<double>(total_mem) / (1<<30)
			  << " free " <<  static_cast<double>(free_mem) / (1<<30)
			  << std::endl;
	#endif
	
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

	
	////////////////////////////////////////////////////////////////////////////
	// multi-gpu spgemm
	#ifdef LOG_GNRL_GPU_SPGEMM
	(*lfile_) << std::fixed;
	(*lfile_) << std::setprecision(5);
	(*lfile_) << "rank " << rank_ << " #devices " << ndevices_ << std::endl;
	#endif
	
	unsigned int	nelems	= B_ncols_ / ndevices_;
	unsigned int	nnz_tot = 0;
	vector<mult_res<NT> > 	mres(ndevices_);
	vector<unsigned int>	nnz_cnts(ndevices_);
	double t_spgemm, t_spgemm_copy, t_post_tuple;
	std::tuple<IT,IT,NT>   *tuplesC;
	omp_set_num_threads(ndevices_);
	#pragma omp parallel reduction(max: t_spgemm, t_spgemm_copy, t_post_tuple)
	{
		int tid = omp_get_thread_num();
		cudaSetDevice(tid);

		int gpu_id = -1;
		int num_cpu_threads = omp_get_num_threads();
		cudaGetDevice(&gpu_id);
		
		unsigned int beg = tid * nelems;
		unsigned int end = -1;
		if (tid == ndevices_ - 1)
			end = B_ncols_;
		else
			end = (tid + 1) * nelems;
		unsigned int ncols = end - beg;

		unsigned int *B_cp_tmp = (unsigned int *)
			malloc(sizeof(*B_cp_tmp) * (ncols + 1));

		for (int i = 0; i <= ncols; ++i)
			B_cp_tmp[i] = B_cp_[beg + i] - B_cp_[beg];

		unsigned int nnzs = B_cp_[end] - B_cp_[beg];

		#ifdef DBG_SPEC_GPU_SPGEMM_TSTATS
		double t_tmp = MPI_Wtime();
		#endif

		Mul<NT> mf;
		mf.copy(A_nrows_, A_ncols_, A_nnzs_, A_cp_, A_ind_, A_val_,
				B_nrows_, end - beg, B_cp_[end] - B_cp_[beg],
				B_cp_tmp, &(B_ind_[B_cp_[beg]]),
				&(B_val_[B_cp_[beg]]), &gst_);
		
		free(B_cp_tmp);

		#pragma omp barrier

		#ifdef DBG_SPEC_GPU_SPGEMM_TSTATS
		gst_.t_pre_copy += MPI_Wtime() - t_tmp;
		#endif
		
		if (tid == 0)
		{
			if(clearA)
				delete const_cast<combblas::SpDCCols<IT, NT> *>(&A);
			if(clearB)
				delete const_cast<combblas::SpDCCols<IT, NT> *>(&B);

			// main thread is assured to be waiting for this condition
			pthread_mutex_lock(mutex);
			pthread_cond_signal(input_freed);
			pthread_mutex_unlock(mutex);
		}
		
		mf.mult(&gst_, &mres[tid], &t_spgemm, &t_spgemm_copy);
				
		#pragma omp barrier

		// concatenate results
		if (tid == 0)
		{
			nnz_cnts[0]	= 0;
			for (int i = 0; i < ndevices_; ++i)
			{
				nnz_tot += mres[i].nnzs;
				if (i < ndevices_ - 1)
					nnz_cnts[i + 1] = nnz_cnts[i] + mres[i].nnzs;
			}
			tuplesC = static_cast<std::tuple<IT,IT,NT> *>
				(::operator new (sizeof(std::tuple<IT, IT, NT>[nnz_tot])));
		}

		#pragma omp barrier

		#ifdef DBG_SPEC_GPU_SPGEMM_TSTATS
		t_post_tuple = MPI_Wtime();
		#endif
		
		unsigned int	 i	 = nnz_cnts[tid];
		mult_res<NT>	*res = &mres[tid];
		for (unsigned int j = 0; j < res->pidx_len; ++j)
		{
			for (unsigned int k = res->pidx[j]; k < res->pidx[j+1]; ++k)
			{
				tuplesC[i++] = std::make_tuple(static_cast<IT>(res->idx[k]),
											   j + (tid * nelems),
											   res->vals[k]);
			}
		}
		
		#ifdef DBG_SPEC_GPU_SPGEMM_TSTATS
		t_post_tuple = MPI_Wtime() - t_post_tuple;
		#endif

		delete [] mres[tid].pidx;
		delete [] mres[tid].idx;
		delete [] mres[tid].vals;
	}
	
	#ifdef DBG_SPEC_GPU_SPGEMM_TSTATS
	gst_.t_spgemm	   += t_spgemm;
	gst_.t_spgemm_copy += t_spgemm_copy;
	gst_.t_post_tuple  += t_post_tuple;
	(*lfile_) << "mult: " << t_spgemm
			  << " copy: " << t_spgemm_copy
			  << " post-tuple: " << t_post_tuple << std::endl;
	gst_.t_total += MPI_Wtime() - t_tmp_all;
	// report_time(lfile_);
	#endif
	////////////////////////////////////////////////////////////////////////////
	

	#ifdef DBG_SPEC_GPU_SPGEMM_MSTATS
	report_memusage("GB");
	#endif

	delete[] A_cp_;
	// delete[] A_ind_;
	delete[] B_cp_;
	// delete[] B_ind_;

	return new combblas::SpTuples<IT, NT>
		(nnz_tot, A_nrows_, B_ncols_, tuplesC, true, true);
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

	IT c_nnz = mf.template mult<IT, NT>(A_nrows_, A_ncols_, A_nnzs_,
		A_cp_, A_ind_, A_val_, B_nrows_, B_ncols_, B_nnzs_,
		B_cp_, B_ind_, B_val_, &gst_);

	delete[] A_cp_;
	// delete[] A_ind_;
	delete[] B_cp_;
	// delete[] B_ind_;

	return c_nnz;
}



template<class NT>
void
GSpGEMM<NT>::print_res (mult_res<NT> *res, std::ofstream &out)
{
	out << "pidx_len " << res->pidx_len
		<< " nnzs " << res->nnzs << std::endl;
	for (unsigned int i = 0; i < res->pidx_len; ++i)
	{
		out << "col/row " << i << " beg " << res->pidx[i]
			<< " end " << res->pidx[i+1] << " -> ";
		for (unsigned int j = res->pidx[i]; j < res->pidx[i+1]; ++j)
			out << res->idx[j] << " " << res->vals[j] << " ### ";
		out << std::endl;
	}	
}
 


template<class NT>
void
GSpGEMM<NT>::print_mat (void)
{
	(*lfile_) << "A: \n"
			  << "  nrows " << A_nrows_ << " ncols " << A_ncols_
			  << " nnz " << A_nnzs_ << "\n";
	for (unsigned int i = 0; i < A_ncols_; ++i)
	{
		(*lfile_) << "  col " << i << ": [" << A_cp_[i] << ", "
				  << A_cp_[i+1] << ") -> ";
		for (unsigned int j = A_cp_[i]; j < A_cp_[i+1]; ++j)
		{
			(*lfile_) << "(" << A_ind_[j] << ", " << A_val_[j] << ") ";
		}
		(*lfile_) << "\n";
	}

	(*lfile_) << "\n";

	(*lfile_) << "B: \n"
			  << "  nrows " << B_nrows_ << " ncols " << B_ncols_
			  << " nnz " << B_nnzs_ << "\n";
	for (unsigned int i = 0; i < B_ncols_; ++i)
	{
		(*lfile_) << "  col " << i << ": [" << B_cp_[i] << ", "
				  << B_cp_[i+1] << ") -> ";
		for (unsigned int j = B_cp_[i]; j < B_cp_[i+1]; ++j)
		{
			(*lfile_) << "(" << B_ind_[j] << ", " << B_val_[j] << ") ";
		}
		(*lfile_) << "\n";
	}
}



template<class NT>
void
GSpGEMM<NT>::print_mat_v2 (
    unsigned int nrows, unsigned int ncols, unsigned int nnzs,
	unsigned int *pidx, unsigned int *idx, NT *val)
{
	(*lfile_) << "matrix: \n"
			  << "  nrows " << nrows << " ncols " << ncols
			  << " nnz " << nnzs << "\n";
	for (unsigned int i = 0; i < ncols; ++i)
	{
		(*lfile_) << "  col " << i << ": [" << pidx[i] << ", "
				  << pidx[i+1] << ") -> ";
		for (unsigned int j = pidx[i]; j < pidx[i+1]; ++j)
		{
			(*lfile_) << "(" << idx[j] << ", " << val[j] << ") ";
		}
		(*lfile_) << "\n";
	}

}
