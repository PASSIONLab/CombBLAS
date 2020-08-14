#include "sort_profiler.h"

#ifdef HAVE_PAPI
#include <papi.h>
#endif

#include <omp.h>

sort_profiler_t 				total_sort;

sort_profiler_t 				sample_get_splitters;
sort_profiler_t 				sample_sort_splitters;
sort_profiler_t 				sample_prepare_scatter;
sort_profiler_t 				sample_do_all2all;

sort_profiler_t					hyper_compute_splitters;
sort_profiler_t					hyper_communicate;
sort_profiler_t					hyper_merge;
sort_profiler_t 				hyper_comm_split;

sort_profiler_t 				seq_sort;
sort_profiler_t					sort_partitionw;

long                    total_bytes;

sort_profiler_t::sort_profiler_t () {
	seconds  = 0.0;   // openmp wall time
	p_flpops =   0;   // papi floating point operations
	
	_pri_seconds  = 0.0;
	_pri_p_flpops =   0;
}

sort_profiler_t::~sort_profiler_t () {
	
}

void
sort_profiler_t::start() {
	_pri_seconds = omp_get_wtime();
	flops_papi();
}

void
sort_profiler_t::stop() {
	seconds -= _pri_seconds;
	p_flpops -= _pri_p_flpops;
	
	_pri_seconds = omp_get_wtime();
	flops_papi(); 
	
	seconds  += _pri_seconds;
	p_flpops += _pri_p_flpops;
}

void
sort_profiler_t::clear() {
	seconds  = 0.0;   
	p_flpops =   0;   
	
	_pri_seconds  = 0.0;
	_pri_p_flpops =   0;
}

void 
sort_profiler_t::flops_papi() {
#ifdef HAVE_PAPI
	int 		retval;
	float rtime, ptime, mflops;
	retval  = PAPI_flops(&rtime, &ptime, &_pri_p_flpops, &mflops);
	// assert (retval == PAPI_OK);
#else
	_pri_p_flpops =   0;
#endif	
}
