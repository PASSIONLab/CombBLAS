#ifndef SORT_PROFILER_H_1CK8Y26
#define SORT_PROFILER_H_1CK8Y26

class sort_profiler_t
{
public:
	sort_profiler_t ();
	virtual ~sort_profiler_t ();

	void									start();
	void									stop();
	void									clear();
			
public:           			
	double								seconds;  // openmp wall time
	long long							p_flpops; // papi floating point operations
	
private:
	void 									flops_papi();
				
protected:        			
	double								_pri_seconds;  // openmp wall time
	long long							_pri_p_flpops; // papi floating point operations
};                			

extern sort_profiler_t 				total_sort;
			
extern sort_profiler_t 				sample_get_splitters;
extern sort_profiler_t 				sample_sort_splitters;
extern sort_profiler_t 				sample_prepare_scatter;
extern sort_profiler_t 				sample_do_all2all;

extern sort_profiler_t				hyper_compute_splitters;
extern sort_profiler_t				hyper_communicate;
extern sort_profiler_t				hyper_merge;
extern sort_profiler_t 				hyper_comm_split;
	
extern sort_profiler_t 				seq_sort;
extern sort_profiler_t				sort_partitionw;

extern long                   total_bytes;

#endif /* end of include guard: SORT_PROFILER_H_1CK8Y26 */
