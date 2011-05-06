/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.1 -------------------------------------------------*/
/* date: 12/25/2010 --------------------------------------------*/
/* authors: Aydin Buluc (abuluc@lbl.gov), Adam Lugowski --------*/
/****************************************************************/


/**
 * Deterministic vertex scrambling functions from V2.1 of the reference implementation
 **/

#ifndef _REF_GEN_2_1_H_
#define _REF_GEN_2_1_H_

#include <vector>
#include <limits>
#include "SpDefs.h"
#include "StackEntry.h"
#include "promote.h"
#include "Isect.h"
#include "HeapEntry.h"
#include "SpImpl.h"
#include "graph500-1.2/generator/graph_generator.h"
#include "graph500-1.2/generator/utils.h"


#ifdef __cplusplus
	#ifndef __STDC_CONSTANT_MACROS
 	#define __STDC_CONSTANT_MACROS
	#endif
	#ifndef __STDC_LIMIT_MACROS
	#define __STDC_LIMIT_MACROS
	#endif
 	#ifdef _STDINT_H
  		#undef _STDINT_H
 	#endif
 	#include <stdint.h>
#endif


using namespace std;

class RefGen21
{
public:
	/* Reverse bits in a number; this should be optimized for performance
 	* (including using bit- or byte-reverse intrinsics if your platform has them).
 	* */
	static inline uint64_t bitreverse(uint64_t x) 
	{
		#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3)
		#define USE_GCC_BYTESWAP /* __builtin_bswap* are in 4.3 but not 4.2 */
		#endif

		#ifdef FAST_64BIT_ARITHMETIC

 		 /* 64-bit code */
		#ifdef USE_GCC_BYTESWAP
  		 x = __builtin_bswap64(x);
		#else
  		 x = (x >> 32) | (x << 32);
  		 x = ((x >> 16) & UINT64_C(0x0000FFFF0000FFFF)) | ((x & UINT64_C(0x0000FFFF0000FFFF)) << 16);
  		 x = ((x >>  8) & UINT64_C(0x00FF00FF00FF00FF)) | ((x & UINT64_C(0x00FF00FF00FF00FF)) <<  8);
		#endif
  		x = ((x >>  4) & UINT64_C(0x0F0F0F0F0F0F0F0F)) | ((x & UINT64_C(0x0F0F0F0F0F0F0F0F)) <<  4);
  		x = ((x >>  2) & UINT64_C(0x3333333333333333)) | ((x & UINT64_C(0x3333333333333333)) <<  2);
 		x = ((x >>  1) & UINT64_C(0x5555555555555555)) | ((x & UINT64_C(0x5555555555555555)) <<  1);
  		return x;

		#else

  		/* 32-bit code */
 		uint32_t h = (uint32_t)(x >> 32);
  		uint32_t l = (uint32_t)(x & UINT32_MAX);
		#ifdef USE_GCC_BYTESWAP
  		 h = __builtin_bswap32(h);
  		 l = __builtin_bswap32(l);
		#else
 		 h = (h >> 16) | (h << 16);
 		 l = (l >> 16) | (l << 16);
		 h = ((h >> 8) & UINT32_C(0x00FF00FF)) | ((h & UINT32_C(0x00FF00FF)) << 8);
 		 l = ((l >> 8) & UINT32_C(0x00FF00FF)) | ((l & UINT32_C(0x00FF00FF)) << 8);
		#endif
  		h = ((h >> 4) & UINT32_C(0x0F0F0F0F)) | ((h & UINT32_C(0x0F0F0F0F)) << 4);
  		l = ((l >> 4) & UINT32_C(0x0F0F0F0F)) | ((l & UINT32_C(0x0F0F0F0F)) << 4);
  		h = ((h >> 2) & UINT32_C(0x33333333)) | ((h & UINT32_C(0x33333333)) << 2);
  		l = ((l >> 2) & UINT32_C(0x33333333)) | ((l & UINT32_C(0x33333333)) << 2);
  		h = ((h >> 1) & UINT32_C(0x55555555)) | ((h & UINT32_C(0x55555555)) << 1);
  		l = ((l >> 1) & UINT32_C(0x55555555)) | ((l & UINT32_C(0x55555555)) << 1);
  		return ((uint64_t)l << 32) | h; /* Swap halves */

		#endif
	}


	/* Apply a permutation to scramble vertex numbers; a randomly generated
 	* permutation is not used because applying it at scale is too expensive. */
	static inline int64_t scramble(int64_t v0, int lgN, uint64_t val0, uint64_t val1) 
	{
  		uint64_t v = (uint64_t)v0;
  		v += val0 + val1;
  		v *= (val0 | UINT64_C(0x4519840211493211));
 	 	v = (RefGen21::bitreverse(v) >> (64 - lgN));
  		assert ((v >> lgN) == 0);
  		v *= (val1 | UINT64_C(0x3050852102C843A5));
  		v = (RefGen21::bitreverse(v) >> (64 - lgN));
  		assert ((v >> lgN) == 0);
  		return (int64_t)v;
	}
};

#endif
