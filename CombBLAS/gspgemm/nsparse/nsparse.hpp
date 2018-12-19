#include <typeinfo>

#ifndef NSPARSE_H
#define NSPARSE_H

#define div_round_up(a, b) ((a % b == 0)? a / b : a / b + 1)

/* Hardware Specific Parameters */
#define warp_BIT 5
#define warp 32
#define MAX_LOCAL_THREAD_NUM 1024
#define MAX_THREAD_BLOCK (MAX_LOCAL_THREAD_NUM / warp)

/* Number of SpMV Execution for Evaluation or Test */
#define TRI_NUM 101
#define TEST_NUM 2
#define SpGEMM_TRI_NUM 11

/* Define 2 related */
#define sfFLT_MAX 1000000000
#define SHORT_MAX 32768
#define SHORT_MAX_BIT 15
#define USHORT_MAX 65536
#define USHORT_MAX_BIT 16

#define SCL_BORDER 16
#define SCL_BIT ((1 << SCL_BORDER) - 1)

#define MAX_BLOCK_SIZE 20

/* Check the answer */
#define sfDEBUG

/* Structure of Formats*/
/* Initializing vector */
template <class idType, class valType>
void init_vector(valType *x, int row)
{
    int i;

    srand48((unsigned)time(NULL));

    for (i = 0; i < row; i++) {
        x[i] = drand48();
    }
}

/* Compare the vectors */
template <class idType, class valType>
void check_answer(valType *csr_ans, valType *ans_vec, idType nrow)
{
    idType i;
    int total_fail = 10;
    valType delta, base;
    valType scale;
    if (typeid(valType) == typeid(float)) {
        scale = 1000;
    }
    else {
        scale = 1000 * 1000;
    }
  
    for (i = 0; i < nrow; i++) {
        delta = ans_vec[i] - csr_ans[i];
        base = ans_vec[i];

        if (delta < 0) {
            delta *= -1;
        }
        if (base < 0) {
            base *= -1;
        }
        if (delta * 100 * scale > base) {
            printf("i=%d, ans=%e, csr=%e, delta=%e\n", i, ans_vec[i], csr_ans[i], delta);
            total_fail--;
            if(total_fail == 0)
                break;
        }
    }
    if (total_fail != 10){
        printf("Calculation Result is Incorrect\n");
    }
    else {
        printf("Calculation Result is Correct\n");
    }
}

#endif

/*
 * Release MemObjects of Each Format structure
 */
/* void release_cpu_amb(sfAMB mat); */
/* void release_amb(sfAMB mat); */

/*
 * Converting matrix to AMB format
 */
/* void init_plan(sfPlan *plan); */
/* void set_plan(sfPlan *plan, size_t seg_size, int block_size); */
/* void sf_csr2amb(sfAMB *mat, sfCSR *csr_mat, real *d_x, sfPlan *plan); */

/*
 * SpMV Kernel
 */
/* void csr_ans_check(real *val, int *col, int *rpt, real *rhs_vec, real *csr_ans, int N); */
/* void sf_spmv_amb(real *d_y, sfAMB *mat, real *d_x, sfPlan *plan); */

