/**
 * @file
 *	tdef.h
 *
 * @author
 *	Oguz Selvitopi
 *
 * @date
 *	XXXX.XX.XX
 *
 * @brief
 *	typedefs and global variables
 *
 * @todo
 *
 * @note
 *
 */

#ifndef __TDEF_H__
#define __TDEF_H__

#include <inttypes.h>

// #define DBG_SPEC_GPU_SPGEMM_TSTATS
// #define DBG_SPEC_GPU_SPGEMM_MSTATS
// #define LOG_GNRL_GPU_SPGEMM


typedef struct _gstats
{
	// time (cumulative)
	double t_pre_decomp;
	double t_pre_conv;
	double t_pre_copy;
	double t_pre_flops;			// auxiliary, turn off in actual runs
	double t_spgemm;
	double t_spgemm_copy;
	double t_post_tuple;
	double t_total;				// measured, not result of additions
	double t_spgemm_symbolic;	// includes copy

	// memory (overwritten)
	double m_d_before_pre;
	double m_h_dcsc;
	double m_h_conv;
	double m_d_input;
	double m_d_after_pre;
	double m_d_after_spgemm;
	double m_h_output;
	double m_h_ctuples;
	double m_d_exit;

	// flops (cumulative)
	int64_t flops_total;
	int64_t flops_current;		// used to compute compression ratio

	// compression ratio (cumulative)
	double compr_ratio;

	// nnzs (cumulative)
	int64_t nnzs_A;
	int64_t nnzs_B;
	int64_t nnzs_C;
} gstats;



// structure for thread return data
template<typename NT>
struct mult_res
{
	unsigned int	*pidx;
	unsigned int	*idx;
	NT				*vals;
	unsigned int	 pidx_len;
	int64_t			 nnzs;
	int64_t			 ncols;
};

#endif
