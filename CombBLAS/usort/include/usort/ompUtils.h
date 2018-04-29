#ifndef _OMP_UTILS_H_
#define _OMP_UTILS_H_

namespace omp_par {

  template <class T,class StrictWeakOrdering>
    void merge(T A_,T A_last,T B_,T B_last,T C_,int p,StrictWeakOrdering comp);

  template <class T,class StrictWeakOrdering>
    void merge_sort(T A,T A_last,StrictWeakOrdering comp);

  template <class T>
    void merge_sort(T A,T A_last);

  template <class T>
    void merge_sort_ptrs(T A,T A_last);

  template <class T, class I>
    T reduce(T* A, I cnt);

  template <class T, class I>
    void scan(T* A, T* B,I cnt);

}

#include "ompUtils.tcc"

#endif

