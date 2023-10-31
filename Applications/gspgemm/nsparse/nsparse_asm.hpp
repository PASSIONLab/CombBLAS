#ifndef NSPARSE_ASM_H
#define NSPARSE_ASM_H



// @OGUZ-TODO
// this is probably not the right instruction to use here
__device__ __inline__
int ld_gbl_col(const int *col)
{
    int return_value;
    // asm("ld.global.cv.s32 %0, [%1];" : "=r"(return_value) : "l"(col));
	asm("flat_load_dword %0, %1": "=v"(return_value) : "v"(col));
    return return_value;
}



// @OGUZ-DONT-USE
__device__ __inline__
short ld_gbl_col(const short *col)
{
    // short return_value;
    // asm("ld.global.cv.u16 %0, [%1];" : "=h"(return_value) : "l"(col));
	// return return_value;
	return 0;
}



// @OGUZ-DONT-USE
__device__ __inline__
unsigned short ld_gbl_col(const unsigned short *col)
{
    // unsigned short return_value;
    // asm("ld.global.cv.u16 %0, [%1];" : "=h"(return_value) : "l"(col));	
    // return return_value;
	return 0;
}


#endif
