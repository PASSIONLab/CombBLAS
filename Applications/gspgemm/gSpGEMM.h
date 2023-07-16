/**
 * @file
 *	gSpGEMM.h
 *
 * @author
 *	Oguz Selvitopi
 *
 * @date
 *	2018.10.XX
 *
 * @brief
 *	Generic routines to perform SpGEMM on GPUs
 *
 * @todo
 *
 * @note
 *
 */

#ifndef __GSPGEMM_H__
#define __GSPGEMM_H__

#include <pthread.h>

#include "tdef.h"
#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/SpParMat.h"


// IT belongs to SpDCCols
template<class NT>
class GSpGEMM
{
	// mpi rank bound to the gpu instance
	int rank_;

	// sparse matrix stuff
	unsigned int     A_nrows_, A_ncols_, A_nnzs_;
	unsigned int	*A_cp_, *A_ind_;
	NT				*A_val_;
	unsigned int     B_nrows_, B_ncols_, B_nnzs_;
	unsigned int	*B_cp_, *B_ind_;
	NT				*B_val_;

	// multi-gpu
	int ndevices_;

	// stats & debugging
	bool compute_flops_;

	// prepare arrays
	template<typename IT>
	void prep_in(const combblas::SpDCCols<IT, NT> &A,
				 const combblas::SpDCCols<IT, NT> &B);

	void report_memusage(const std::string &unitstr);
	int64_t get_flops(void);
	void print_mat(void);
	void print_mat_v2(unsigned int nrows, unsigned int ncols, unsigned int nnzs,
					  unsigned int *pidx, unsigned int *idx, NT *val);
	void print_res(mult_res<NT> *res, std::ofstream &out);


public:
	gstats			 gst_;
	std::ofstream	*lfile_;
	
	GSpGEMM (int rank, bool computeFlops = true);
	~GSpGEMM ();

	template<typename IT, template<typename T> class Mul>
	combblas::SpTuples<IT, NT> *
	mult (const combblas::SpDCCols<IT, NT> &A,
		  const combblas::SpDCCols<IT, NT> &B,
		  bool clearA, bool clearB,
		  int iter, int phase, int stage,
		  pthread_cond_t *input_freed, pthread_mutex_t *mutex);
	
	template<typename IT, typename MulFunctor>
	IT
	mult_symb (const combblas::SpDCCols<IT, NT> &A,
			   const combblas::SpDCCols<IT, NT> &B,
			   MulFunctor mf);

	void report_time(std::ofstream &lfile, int ncalls);	
};


#include "gSpGEMM.cpp"

#endif
