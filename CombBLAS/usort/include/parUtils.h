
/**
  @file parUtils.h
  @brief A set of parallel utilities.	
  @author Hari Sundar, hsundar@gmail.com
  */ 

#ifndef __PAR_UTILS_H_
#define __PAR_UTILS_H_

#define KEEP_HIGH 100
#define KEEP_LOW  101

#ifdef __DEBUG__
#ifndef __DEBUG_PAR__
#define __DEBUG_PAR__
#endif
#endif

#include "mpi.h"
#include <vector>

#ifdef __USE_64_BIT_INT__
#define DendroIntL long long
#define DendroIntLSpecifier %lld
#define DendroUIntLSpecifier %llu
#else
#define DendroIntL int
#define DendroIntLSpecifier %d
#define DendroUIntLSpecifier %u
#endif

/**
  @namespace par
  @author Hari Sundar   hsundar@gmail.com
  @brief Collection of Generic Parallel Functions: Sorting, Partitioning, Searching,...
  */
namespace par {

  template <typename T>
    int Mpi_Isend(T* buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request* request);

  template <typename T>
    int Mpi_Issend(T* buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request* request);

  template <typename T>
    int Mpi_Recv(T* buf, int count, int source, int tag, MPI_Comm comm, MPI_Status* status);

  template <typename T>
    int Mpi_Irecv(T* buf, int count, int source, int tag, MPI_Comm comm, MPI_Request* request);

  template <typename T>
    int Mpi_Gather( T* sendBuffer, T* recvBuffer, int count, int root, MPI_Comm comm);

  template <typename T, typename S>
    int Mpi_Sendrecv( T* sendBuf, int sendCount, int dest, int sendTag,
        S* recvBuf, int recvCount, int source, int recvTag,
        MPI_Comm comm, MPI_Status* status);

  template <typename T>
    int Mpi_Bcast( T* buffer, int count, int root, MPI_Comm comm);

  template <typename T>
    int Mpi_Scan( T* sendbuf, T* recvbuf, int count, MPI_Op op, MPI_Comm comm);

  template <typename T>
    int Mpi_Reduce( T* sendbuf, T* recvbuf, int count, MPI_Op op, int root, MPI_Comm comm);

  template <typename T> 
    int Mpi_Allreduce( T* sendbuf, T* recvbuf, int count, MPI_Op op, MPI_Comm comm);

  template <typename T>
    int Mpi_Alltoall(T* sendbuf, T* recvbuf, int count, MPI_Comm comm); 

  template <typename T>
    int Mpi_Allgatherv(T* sendbuf, int sendcount, T* recvbuf,
        int* recvcounts, int* displs, MPI_Comm comm);

  template <typename T>
    int Mpi_Allgather(T* sendbuf, T* recvbuf, int count, MPI_Comm comm);

  template <typename T>
    int Mpi_Alltoallv_sparse(T* sendbuf, int* sendcnts, int* sdispls, 
        T* recvbuf, int* recvcnts, int* rdispls, MPI_Comm comm);

  template <typename T>
    int Mpi_Alltoallv_dense(T* sendbuf, int* sendcnts, int* sdispls, 
        T* recvbuf, int* recvcnts, int* rdispls, MPI_Comm comm);
  
	
	template<typename T>
    unsigned int defaultWeight(const T *a);
	
  /**
    @brief A parallel weighted partitioning function. In our implementation, we do not pose any 
    restriction on the input or the number of processors. This function can be used with an odd number of processors as well.
    Some processors can pass an empty vector as input. The relative ordering of the elements is preserved.
    @author Hari Sundar
    @author Rahul Sampath
    @param vec the input vector
    @param getWeight function pointer to compute the weight of each element. If you pass NULL, 
    then every element will get a weight equal to 1.    
    @param comm the communicator
    */
  template<typename T>
    int partitionW(std::vector<T>& vec,
        unsigned int (*getWeight)(const T *), MPI_Comm comm);


		template<typename T>
		void rankSamples(std::vector<T>& arr, std::vector<T> samples, MPI_Comm comm);

	template<typename T>
		std::vector<T> Sorted_Sample_Select(std::vector<T>& arr, unsigned int kway, std::vector<unsigned int>& min_idx, std::vector<unsigned int>& max_idx, std::vector<DendroIntL>& splitter_ranks, MPI_Comm comm);
	template<typename T>
		std::vector<T> Sorted_approx_Select(std::vector<T>& arr, unsigned int k, MPI_Comm comm);
	//! new one to handle skewed distributions ...
  template<typename T>
		std::vector<std::pair<T, DendroIntL> > Sorted_approx_Select_skewed(std::vector<T>& arr, unsigned int k, MPI_Comm comm);
	
  template<typename T>
		std::vector<T> GetRangeMean(std::vector<T>& arr, std::vector<unsigned int> range_min, std::vector<unsigned int> range_max, MPI_Comm comm);
	template<typename T>
		std::vector<T> GuessRangeMedian(std::vector<T>& arr, std::vector<unsigned int> range_min, std::vector<unsigned int> range_max, MPI_Comm comm);

		/** @brief A parallel k-selection algorithms
			
			@author Hari Sundar 
			@date 2013-01-10
			@param  arr				arr from which samples are to be selected
			@param  k					number of samples
			@param  isSorted	if false, arr is locally sorted first.
			@return list of k keys. 
		**/
		template<typename T>
				std::vector<T> Sorted_k_Select(std::vector<T>& arr, std::vector<unsigned int> min_idx, std::vector<unsigned int> max_idx, std::vector<DendroIntL> K, std::vector<T> guess, MPI_Comm comm);
	
  /**
    @brief A parallel hyper quick sort implementation.
    @author Dhairya Malhotra
    @param in the input vector
    @param out the output vector
    @param comm the communicator
    */
  template<typename T>
    int HyperQuickSort(std::vector<T>& in, std::vector<T> & out, MPI_Comm comm); 

  template<typename T>
    int HyperQuickSort_kway(std::vector<T>& in, std::vector<T> & out, MPI_Comm comm); 
  
  template<typename T>
    int HyperQuickSort_cav(std::vector<T>& in, std::vector<T> & out, MPI_Comm comm); 
		
  /* mem-efficient version */
  template<typename T>
    int HyperQuickSort(std::vector<T>& arr, MPI_Comm comm);
		
  template<typename T>
    int HyperQuickSort_kway(std::vector<T>& in, MPI_Comm comm); 
  
  template<typename T>
    int HyperQuickSort_cav(std::vector<T>& in, MPI_Comm comm); 
  
  /**
    @brief A parallel sample sort implementation. In our implementation, we do not pose any 
    restriction on the input or the number of processors. This function can be used with an odd number of processors as well.
    Some processors can pass an empty vector as input. If the total number of elements in the vector (globally) is fewer 
    than 10*p^2, where p is the number of processors, then we will use bitonic sort instead of sample sort to sort the vector.
    We use a paralle bitonic sort to sort the samples in the sample sort algorithm. Hence, the complexity of the algorithm
    is O(n/p log n/p) + O(p log p). Here, n is the global length of the vector and p is the number of processors.
    @author Hari Sundar
    @param in the input vector
    @param out the output vector
    @param comm the communicator
    */
  template<typename T>
    int sampleSort(std::vector<T>& in, std::vector<T> & out, MPI_Comm comm); 

  template<typename T>
    int sampleSort(std::vector<T>& in, MPI_Comm comm); 

  /**
    @brief Splits a communication group into two, one containing processors that passed a value of 'false' 
    for the parameter 'iAmEmpty' and the another containing processors that passed a value of 'true' 
    for the parameter. Both the groups are sorted in the ascending order of their ranks in the old comm.
    @author Rahul Sampath
    @param iAmEmpty     Some flag to determine which group the calling processor will be combined into. 	
    @param orig_comm    The comm group that needs to be split.
    @param new_comm     The new comm group.
    */
  int splitComm2way(bool iAmEmpty, MPI_Comm* new_comm, MPI_Comm orig_comm);

  /**
    @brief Splits a communication group into two depending on the values in isEmptyList.
    Both the groups are sorted in the ascending order of their ranks in the old comm.
    All processors must call this function with the same 'isEmptyList' array.
    @author Rahul Sampath
    @param isEmptyList  flags (of length equal to the number of processors) to determine whether each processor is active or not. 	
    @param orig_comm    The comm group that needs to be split.
    @param new_comm     The new comm group.
    */
  int splitComm2way(const bool* isEmptyList, MPI_Comm* new_comm, MPI_Comm orig_comm);

  /*
     @author Rahul Sampath
     @brief Splits a communication group into two, processors with 
     ranks less than splittingRank form one group and the other 
     processors form the second group. Both the groups are sorted in 
     the ascending order of their ranks in the old comm.
     @param splittingRank The rank used for splitting the communicator
     @param orig_comm    The comm group that needs to be split.
     @param new_comm     The new comm group.
     */
  int splitCommUsingSplittingRank(int splittingRank, MPI_Comm* new_comm, MPI_Comm orig_comm);

  /** 
   * @brief Splits a communication group into two, the first having a power of 2
   * number of processors and the other having the remainder. The first group
   * is sorted in the ascending order of their ranks in the old comm and the second group
   * is sorted in the descending order of their ranks in the old comm
   * @author Hari Sundar
   * @param orig_comm    The comm group that needs to be split.
   * @param new_comm     The new comm group.
   */
  unsigned int splitCommBinary( MPI_Comm orig_comm, MPI_Comm* new_comm);


  /** 
   * @brief Splits a communication group into two, the first having a power of 2
   * number of processors and the other having the remainder. Both the groups
   * are sorted in the ascending order of their ranks in the old comm.
   * @author Hari Sundar
   * @param orig_comm    The comm group that needs to be split.
   * @param new_comm     The new comm group.
   */
  unsigned int splitCommBinaryNoFlip( MPI_Comm orig_comm, MPI_Comm* new_comm);

  /** 
    @author Hari Sundar
   * @brief Merges lists A, and B, retaining either the low or the High in list A.      
   * 
   * @param listA   	Input list, and where the output is stored.
   * @param listB   	Second input list.
   * @param KEEP_WHAT 	determines whether to retain the High or the low values
   * 			from A and B. One of KEEP_HIGH or KEEP_LOW.
   *
   * Merging the two lists when their sizes are not the same is a bit involved.
   * The major condition that needs to be used is that all elements that are less
   * than max(min(A), min(B)) are retained by the KEEP_LOW processor, and
   * similarly all elements that are larger larger than min(max(A), max(B)) are
   * retained by the KEEP_HIGH processor.
   *
   * The reason for this is that, on the Keep_Low side,
   *
   *   max(min(A), min(B)) > min(A) > max(A-)
   *
   * and similarly on the Keep_high side,
   *  
   *   min(max(A), max(B)) < max(A) < min(A+)
   *
   * which guarantees that the merged lists remain bitonic.   
   */
  template <typename T>
    void MergeLists( std::vector<T> &listA, std::vector<T> &listB, int KEEP_WHAT) ;

  /**
    @author Hari Sundar
    @brief The main operation in the parallel bitonic sort algorithm. This implements the compare-split operation.
   * @param which_keys is one of KEEP_HIGH or KEEP_LOW
   * @param partner    is the processor with which to Merge and Split.
   @param local_list the input vector
   @param comm the communicator
   */
  template <typename T>
    void MergeSplit( std::vector<T> &local_list, int which_keys, int partner, MPI_Comm  comm);

  /**
    @author Hari Sundar
    */
  template <typename T>
    void Par_bitonic_sort_incr( std::vector<T> &local_list, int proc_set_size, MPI_Comm  comm );

  /**
    @author Hari Sundar
    */
  template <typename T>
    void Par_bitonic_sort_decr( std::vector<T> &local_list, int proc_set_size, MPI_Comm  comm);

  /**
    @author Hari Sundar
    */
  template <typename T>
    void Par_bitonic_merge_incr( std::vector<T> &local_list, int proc_set_size, MPI_Comm  comm );

  /**
    @brief An implementation of parallel bitonic sort that expects the number of processors to be a power of 2.
    However, unlike most implementations, we do not expect the length of the vector 
    (neither locally nor globally) to be a power of 2 or even. Moreover, each processor can call 
    this with a different number of elements. However, we do expect that 'in' atleast has 1
    element on each processor.
    @param in the vector to be sorted		
    @author Hari Sundar  	
    */
  template <typename T>
    void bitonicSort_binary(std::vector<T> & in, MPI_Comm comm) ;

  /**
    @brief An implementation of parallel bitonic sort that does not expect the number of processors to
    be a power of 2. In fact, the number of processors can even be odd.
    Moreover, we do not even expect the length of the vector 
    (neither locally nor globally) to be a power of 2 or even. Moreover, each processor can call 
    this with a different number of elements. However, we do expect that 'in' atleast has 1
    element on each processor. This recursively calls the function bitonicSort_binary, followed by a
    special parallel merge.
    @param in  the vector to be sorted
    @author Hari Sundar
    @see bitonicSort_binary
    */
  template <typename T>
    void bitonicSort(std::vector<T> & in, MPI_Comm comm) ;

}//end namespace

#include "parUtils.tcc"

#endif

