#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if USE_MPI
#include <mpi.h>
#endif
#include "graph.h"

#define PQSORT_SWAP(a, b, size)			      \
  do									      \
    {									      \
      register size_t __size = (size);	      \
      register char *__a = (a), *__b = (b);   \
      do								      \
	{								      \
	  char __tmp = *__a;			      \
	  *__a++ = *__b;				      \
	  *__b++ = __tmp;				      \
	} while (--__size > 0);			      \
    } while (0)

static int partition (int64_t p, int64_t r, void *data, 
        int (*cmpfn)(const void *x, const void *y),
        size_t elem_size) {
    
    int64_t k = p;
    int64_t l = r+1;
    int cmp_result;

    char *datac = (char *) data;
    
    while (1) {
        k++;
        while (k < r) {
            cmp_result = (*cmpfn)(datac+k*elem_size, datac+p*elem_size);
            if (cmp_result <= 0) {
                k++;
            } else {
                break;
            }
        }

        l--;
        while (1) {
            cmp_result = (*cmpfn)(datac+l*elem_size, datac+p*elem_size);
            if (cmp_result <= 0) {
                break;
            } else {
                l--;
            }
        }

        while (k < l) {

            /* swap k and l */
            PQSORT_SWAP(datac+k*elem_size, datac+l*elem_size, elem_size);
        
            k++;
            while (1) {
                cmp_result = (*cmpfn)(datac+k*elem_size, datac+p*elem_size);
                if (cmp_result <= 0) {
                    k++;
                } else {
                    break;
                }
            }

            l--;
            while (1) {
                cmp_result = (*cmpfn)(datac+l*elem_size, datac+p*elem_size);
                if (cmp_result <= 0) {
                    break;
                } else {
                    l--;
                }
            }

        }

        /* swap p and l */
        PQSORT_SWAP(datac+p*elem_size, datac+l*elem_size, elem_size);
        return l;
    }
}

static void parallel_quicksort_rec (int64_t p, int64_t r, 
        void *data, int64_t low_limit,
        int (*cmpfn)(const void *x, const void *y),
        size_t elem_size) {
    
    if (p < r) {
        char *datac = (char *) data;
        if ((r-p) < low_limit) 
            qsort(datac + p*elem_size, r-p+1, elem_size, cmpfn);
        else {
            int q = partition (p, r, data, cmpfn, elem_size);
#pragma omp task
            parallel_quicksort_rec (p, q-1, data, low_limit, cmpfn, elem_size);
#pragma omp task
            parallel_quicksort_rec (q+1, r, data, low_limit, cmpfn, elem_size);
        }
    }
}

int local_parallel_quicksort (void *data, int64_t n, size_t elem_size, 
        int (*cmpfn)(const void *x, const void *y)) {

#ifdef _OPENMP
    int omp_dyn_val = omp_get_dynamic(); 
    int omp_nested_val = omp_get_nested();
    omp_set_dynamic (0);
    omp_set_nested (1);
#endif

    int64_t low_limit = (n-1)/64 + 1;
#pragma omp parallel
{
#pragma omp single nowait
    parallel_quicksort_rec (0, n-1, data, low_limit, cmpfn, elem_size);
}

#ifdef _OPENMP
    omp_set_dynamic (omp_dyn_val);
    omp_set_nested (omp_nested_val);
#endif
    return 0;
}

double get_seconds() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double) (tp.tv_sec + ((1e-6)*tp.tv_usec));
}

