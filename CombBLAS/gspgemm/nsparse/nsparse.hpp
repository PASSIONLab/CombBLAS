#include <typeinfo>

#ifndef NSPARSE_H
#define NSPARSE_H

#define div_round_up(a, b) ((a % b == 0)? a / b : a / b + 1)

/* Hardware Specific Parameters */
#define nsp_warp_BIT 5
#define nsp_warp_sz 32
#define MAX_LOCAL_THREAD_NUM 1024
#define MAX_THREAD_BLOCK (MAX_LOCAL_THREAD_NUM / nsp_warp_sz)

/* Number of SpMV Execution for Evaluation or Test */
#define TRI_NUM 101
#define TEST_NUM 2
#define SpGEMM_TRI_NUM 11

#endif

