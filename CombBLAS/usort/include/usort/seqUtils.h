
/**
  @file seqUtils.h
  @brief A set of sequential utilities.	
  @author Rahul S. Sampath, rahul.sampath@gmail.com
  */ 

#ifndef __SEQ_UTILS_H_
#define __SEQ_UTILS_H_

#include <vector>

/**
  @namespace seq
  @author Rahul Sampath
  @brief Collection of Generic Sequential Functions.
  */
namespace seq {

  /** 
   * @brief Flash sort algo to sort an array in O(n).
   * 
   * @param a    The array to be sorted
   * @param n    The number of elements in a
   * @param m    Size of index vector. typically m = 0.1*n
   * @param ctr  The number of times flashsort was called.
   * 
   * Sorts array a with n elements by use of the index vector l of
   * dimension m (with m about 0.1 n). 
   * The routine runs fastest with a uniform distribution of elements.
   * The vector l is declare dynamically using the calloc function.
   * The variable ctr counts the number of times that flashsort is called.
   * THRESHOLD is a very important constant.  It is the minimum number of 
   * elements required in a subclass before recursion is used. 
   *
   * Templated version of flashsort based on original C code by 
   * Karl-Dietrich Neubert.
   */
  template <typename T> void flashsort(T* a, int n, int m, int *ctr);

  /**
    @brief Removes duplicates from the vector.  
    @author		Rahul Sampath
    @param vec    The vector to be made free of duplicates.
    @param isSorted Pass 'true' if the input is sorted.    

    If the vector is not sorted,
    it is first sorted before removing duplicates.
    */
  template <typename T> 
    void makeVectorUnique(std::vector<T>& vec, bool isSorted) ;

  /**
    @brief A binary search implementation.	
    @author Rahul Sampath
    @param arr A sorted array
    @param nelem The length of the array
    @param key The search key
    @param idx 0-based index of the position of the key in the array
    @return 'true' if the key exists in the array and 'false' otherwise	
    */
  template <typename T>
    bool BinarySearch(const T* arr, unsigned int nelem, const T & key, unsigned int *idx) ;

  /**
    @brief Finds the index of the smallest upper bound of the search key in the array.
    @author Santi Swaroop Adavani
    @param arr A sorted array
    @param nelem The length of the array
    @param startIdx The starting location to search from
    @param key The search key
    @return the index of the first element in the array >= key	
    */
  template <typename T>
    int UpperBound (unsigned int nelem,const  T * arr,unsigned int startIdx, const T & key);

  /**
    @brief Finds the index of the greatest lower bound of the search key in the array. 
    The implementation uses a simple modification of the binary search algorithm.
    @author Rahul Sampath
    @param arr A sorted array
    @param key The search key
    @param retIdx The index of the position of the last element in the array <= key
    @param leftIdx If this is not NULL, then the search will be limited to elements at positions >= *leftIdx
    @param rightIdx if this is not NULL, then the search will be limited to elements at positions <= *rightIdx
    @return 'true' if the search was successful
    */
  template <typename T>
    bool maxLowerBound(const std::vector<T> & arr,const T & key, unsigned int & retIdx,
        unsigned int* leftIdx, unsigned int* rightIdx);

/*
  template <class T,class StrictWeakOrdering>
    typename std::iterator_traits<T>::difference_type BinSearch(T A,T A_last,
	const typename std::iterator_traits<T>::value_type& val,StrictWeakOrdering comp);

  template <class T,class StrictWeakOrdering>
    void Merge(T A_,T A_last,T B_,T B_last,T C_,StrictWeakOrdering comp);
*/

}//end namespace

#include "seqUtils.tcc"

#endif

