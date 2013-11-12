/* Copyright (C) 2010-2013 The Trustees of Indiana University.             */
/*                                                                         */
/* Use, modification and distribution is subject to the Boost Software     */
/* License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at */
/* http://www.boost.org/LICENSE_1_0.txt)                                   */
/*                                                                         */
/*  Authors: Jeremiah Willcock                                             */
/*           Andrew Lumsdaine                                              */

#ifndef BITMAP_HPP
#define BITMAP_HPP

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif

#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <limits.h>
#include <mpi.h>
#include <assert.h>
#include <stdio.h>
#ifdef __IBMCPP__
#include <builtins.h>
#endif

// On BG/P:
/* Bitmaps are represented as arrays of unsigned longs; each unsigned long
 * represents sizeof(unsigned long) * CHAR_BIT elements of the bitmap, with
 * bits within that numbered from 0 at the MSB (that encoding is used to allow
 * (x << b) < 0 to test a bit). */

// On x86: Bitmaps represent element i as bit (i % BITMAP_ULONG_BITS) from the
// LSB of word (i / BITMAP_ULONG_BITS).

// #define BITMAP_ULONG_BITS (sizeof(unsigned long) * CHAR_BIT)
static const unsigned int BITMAP_ULONG_BITS = (sizeof(unsigned long) * CHAR_BIT);
// #define BITMAP_ULONG_HIGHBIT (1UL << (BITMAP_ULONG_BITS - 1))

class bitmap {
  unsigned long* data;
  uint64_t length;
  size_t nwords;

  public:
  explicit bitmap(uint64_t len = 0);
  void ensure_size(uint64_t newlen);
  void clear();
  ~bitmap();
  void reset_all();
  const unsigned long* get_data(size_t start, size_t len) const {(void)len; assert (start <= nwords); assert (start + len <= nwords); return data + start;}
  unsigned long* get_data(size_t start, size_t len) {(void)len; assert (start <= nwords); assert (start + len <= nwords); return data + start;}
  size_t get_nwords() const {return nwords;}
  uint64_t get_length() const {return length;}

  void union_all(const bitmap& b_in, MPI_Comm comm);
  void union_all_inplace(MPI_Comm comm);

  // a = a &~ b; returns true if any bits set in result
  bool and_not(const bitmap&);
  // a = b &~ a
  void and_not_rev(const bitmap&);
  bitmap& operator|=(const bitmap&);
  bitmap& operator=(const bitmap&);
  void swap(bitmap&);

  bool get(uint64_t idx) const;
  void set(uint64_t idx);
  void set_non_atomic(uint64_t idx);
  void reset(uint64_t idx);
  bool test_and_set(uint64_t idx);
  bool test_and_set_non_atomic(uint64_t idx);

  bool are_all_reset() const;
};

#if 0
inline unsigned long rotate_left(unsigned long x, unsigned int count) {
#if defined(__powerpc__)
  unsigned long y;
#if defined(__powerpc64__)
  if (__builtin_constant_p(count)) {
    y = (x << (count & (BITMAP_ULONG_BITS - 1))) | (x >> ((-count) & (BITMAP_ULONG_BITS - 1)));
  } else {
#ifdef __IBMCPP__
    y = __rotatel8(x, count);
#else
    asm ("rotld %0, %1, %2" : "=r" (y) : "r" (x), "r" (count));
#endif
  }
#else
  if (__builtin_constant_p(count)) {
    y = (x << (count & (BITMAP_ULONG_BITS - 1))) | (x >> ((-count) & (BITMAP_ULONG_BITS - 1)));
  } else {
#ifdef __IBMCPP__
    y = __rotatel4(x, count);
#else
    asm ("rotlw %0, %1, %2" : "=r" (y) : "r" (x), "r" (count));
#endif
  }
#endif
  return y;
#else
  return (x << (count % BITMAP_ULONG_BITS)) | (x >> (-count % BITMAP_ULONG_BITS));
#endif
}

inline unsigned long rotate_right(unsigned long x, unsigned int count) {
#if defined(__powerpc__)
  return rotate_left(x, -count);
#else
  return (x >> (count % BITMAP_ULONG_BITS)) | (x << (-count % BITMAP_ULONG_BITS));
#endif
}

inline unsigned int high_bit_to_mask_u(unsigned int x) {
  return (unsigned int)((signed int)x >> ((sizeof(int) * CHAR_BIT) - 1));
}

inline unsigned long high_bit_to_mask_ul(unsigned long x) {
  return (unsigned long)((signed long)x >> (BITMAP_ULONG_BITS - 1));
}

inline uint32_t high_bit_to_mask_u32(uint32_t x) {
  return (uint32_t)((int32_t)x >> 31);
}

inline bool high_bit_set_u(unsigned int x) {return (signed int)x < 0;}
inline bool high_bit_set_ul(unsigned long x) {return (signed long)x < 0;}
inline bool high_bit_set_u32(uint32_t x) {return (int32_t)x < 0;}
#endif

inline bool bitmap::get(uint64_t idx) const {
  assert (idx < this->length);
  bool result = (this->data[idx / BITMAP_ULONG_BITS] & (1UL << (idx % BITMAP_ULONG_BITS))) != 0;
  // bool result = high_bit_set_ul(rotate_left(this->data[idx / BITMAP_ULONG_BITS], (unsigned int)idx));
  // assert (result == high_bit_set_ul(this->data[idx / BITMAP_ULONG_BITS] << (idx % BITMAP_ULONG_BITS)));
  return result;
}

static inline bool bitmap_fast_get(uint64_t idx, const unsigned long* bm_ptr, const uint64_t bm_len) {
  (void)bm_len;
  assert (idx < bm_len);
  bool result = (bm_ptr[idx / BITMAP_ULONG_BITS] & (1UL << (idx % BITMAP_ULONG_BITS))) != 0;
  // bool result = high_bit_set_ul(rotate_left(bm_ptr[idx / BITMAP_ULONG_BITS], (unsigned int)idx));
  // assert (result == high_bit_set_ul(bm_ptr[idx / BITMAP_ULONG_BITS] << (idx % BITMAP_ULONG_BITS)));
  return result;
}

inline void bitmap::set(uint64_t idx) {
  assert (idx < this->length);
  unsigned long mask = 1UL << (idx % BITMAP_ULONG_BITS);
#pragma omp atomic
  this->data[idx / BITMAP_ULONG_BITS] |= mask;
}

inline void bitmap::set_non_atomic(uint64_t idx) {
  assert (idx < this->length);
  // unsigned long mask = rotate_right(BITMAP_ULONG_HIGHBIT, (unsigned int)(idx));
  unsigned long mask = 1UL << (idx % BITMAP_ULONG_BITS);
  this->data[idx / BITMAP_ULONG_BITS] |= mask;
}

inline bool bitmap::test_and_set(uint64_t idx) {
  assert (idx < this->length);
  unsigned long mask = 1UL << (idx % BITMAP_ULONG_BITS);
  // unsigned long mask = rotate_right(BITMAP_ULONG_HIGHBIT, (unsigned int)(idx));
  // assert (mask == BITMAP_ULONG_HIGHBIT >> (idx % BITMAP_ULONG_BITS));
  unsigned long old_val = __sync_fetch_and_or((volatile unsigned long*)&this->data[idx / BITMAP_ULONG_BITS], mask);
  return (old_val & mask) != 0;
}

#if 0
static inline unsigned long bitmap_fast_tas_na(uint64_t idx, unsigned long* bm_ptr, const uint64_t bm_len) { // Returns answer in high bit of result
  (void)bm_len;
  assert (idx < bm_len);
#if 0 // __x86_64__
  char result;
  asm ("bts %[idx], %[base_ptr]; setc %[result]" : [base_ptr] "=m" (*bm_ptr), [result] "=r" (result) : [idx] "r" (idx ^ (BITMAP_ULONG_BITS - 1)) : "cc", "memory");
  return (unsigned long)(result) << 63;
#elif defined(__powerpc__)
  unsigned long data = bm_ptr[idx / BITMAP_ULONG_BITS];
  unsigned long mask = rotate_left(1, ~(unsigned int)idx);
  // unsigned long mask = BITMAP_ULONG_HIGHBIT >> (idx % BITMAP_ULONG_BITS);
  unsigned long temp2 = data & mask;
  bm_ptr[idx / BITMAP_ULONG_BITS] = data | mask;
  unsigned long result = rotate_left(temp2, (unsigned int)idx);
  // unsigned long result = temp2 ? BITMAP_ULONG_HIGHBIT : 0;
  /* if (result != 0 && result != BITMAP_ULONG_HIGHBIT) {fprintf(stderr, "Invalid value %08lx for bitmap_fast_tas_na result\n", result); abort();} */
  return result;
#else
  unsigned long data = bm_ptr[idx / BITMAP_ULONG_BITS];
  unsigned long result = rotate_left(data, (unsigned int)(idx));
  data = rotate_right(1UL | rotate_left(result, 1), (unsigned int)(idx + 1));
  bm_ptr[idx / BITMAP_ULONG_BITS] = data;
  return result & BITMAP_ULONG_HIGHBIT;
#endif
}

static inline unsigned long bitmap_fast_tas_na_inv(uint64_t idx, unsigned long* bm_ptr, const uint64_t bm_len) { // Returns inverted answer in high bit of result
  (void)bm_len;
  assert (idx < bm_len);
#if 0 // __x86_64__
  char result;
  asm ("bts %[idx], %[base_ptr]; setnc %[result]" : [base_ptr] "=m" (*bm_ptr), [result] "=r" (result) : [idx] "r" (idx ^ (BITMAP_ULONG_BITS - 1)) : "cc", "memory");
  return (unsigned long)(result) << 63;
#elif defined(__powerpc__)
  unsigned long data = bm_ptr[idx / BITMAP_ULONG_BITS];
  unsigned long mask = rotate_left(1UL, (unsigned int)(~idx));
  // unsigned long mask = BITMAP_ULONG_HIGHBIT >> (idx % BITMAP_ULONG_BITS);
  unsigned long temp2 = ~data & mask;
  bm_ptr[idx / BITMAP_ULONG_BITS] = data | mask;
  unsigned long result = rotate_left(temp2, (unsigned int)(idx));
  // unsigned long result = temp2 ? BITMAP_ULONG_HIGHBIT : 0;
  /* if (result != 0 && result != BITMAP_ULONG_HIGHBIT) {fprintf(stderr, "Invalid value %08lx for bitmap_fast_tas_na_inv result\n", result); abort();} */
  return result;
#else
  unsigned long data = bm_ptr[idx / BITMAP_ULONG_BITS];
  unsigned long result = rotate_left(~data, (unsigned int)(idx));
  data = rotate_right(1UL | rotate_left(result, 1), (unsigned int)(idx + 1));
  bm_ptr[idx / BITMAP_ULONG_BITS] = data;
  return result & BITMAP_ULONG_HIGHBIT;
#endif
}
#endif

inline bool bitmap::test_and_set_non_atomic(uint64_t idx) {
  assert (idx < this->length);
  unsigned long mask = 1UL << (idx % BITMAP_ULONG_BITS);
  // unsigned long mask = rotate_right(1UL, (unsigned int)(idx + 1));
  // assert (mask == BITMAP_ULONG_HIGHBIT >> (idx % BITMAP_ULONG_BITS));
  unsigned long* ptr = &this->data[idx / BITMAP_ULONG_BITS];
  unsigned long old_val = *ptr;
  if (old_val & mask) {
    return true;
  } else {
    *ptr = old_val | mask;
    return false;
  }
}

inline void bitmap::reset(uint64_t idx) {
  assert (idx < this->length);
  unsigned long mask = 1UL << (idx % BITMAP_ULONG_BITS);
  // unsigned long mask = rotate_right(1UL, (unsigned int)(idx + 1));
  // assert (mask == BITMAP_ULONG_HIGHBIT >> (idx % BITMAP_ULONG_BITS));
#ifdef __IBMCPP__
#ifdef __powerpc64__
  __fetch_and_andlp((volatile unsigned long*)&this->data[idx / BITMAP_ULONG_BITS], ~mask);
#else
  __fetch_and_and((volatile unsigned int*)&this->data[idx / BITMAP_ULONG_BITS], ~mask);
#endif
#else
  __sync_fetch_and_and((volatile unsigned long*)&this->data[idx / BITMAP_ULONG_BITS], ~mask);
#endif
}

#define BITMAP_FOR_EACH_WORD_BEGIN(bitmap_expr, bitidxvar, wordvar, par) \
  do { \
    const bitmap& bitmap = (bitmap_expr); \
    const ptrdiff_t nwords = ptrdiff_t(bitmap.get_nwords()); \
    const unsigned long* const restrict data = bitmap.get_data(0, nwords); \
    par \
    for (ptrdiff_t wordidx = 0; wordidx < nwords; ++wordidx) { \
      const uint64_t bitidxvar = uint64_t(wordidx) * BITMAP_ULONG_BITS; \
      const unsigned long wordvar = data[wordidx];
#define BITMAP_FOR_EACH_WORD_END \
    } \
  } while (0);

#define PP_CAT2(a, b) a ## b
#define PP_CAT(a, b) PP_CAT2(a, b)

// This code assumes that the bitmap is all reset before starting
#define BITMAP_DECLARE_SET_OPERATION(bitmap_expr, var) \
  size_t const restrict PP_CAT(var, _bmdatalen) = (bitmap_expr).get_nwords(); \
  unsigned long* restrict PP_CAT(var, _bmdata) = (bitmap_expr).get_data(0, PP_CAT(var, _bmdatalen)); \
  unsigned long* restrict PP_CAT(var, _bmdataend) = PP_CAT(var, _bmdata) + PP_CAT(var, _bmdatalen); \
  (void)PP_CAT(var, _bmdataend); \
  unsigned long PP_CAT(var, _data) = 0; \
  unsigned int PP_CAT(var, _counter) = 0;
#define BITMAP_STEP_SET_OPERATION(var) \
  do { \
    ++PP_CAT(var, _counter); \
    PP_CAT(var, _counter) &= (BITMAP_ULONG_BITS - 1); \
    if (PP_CAT(var, _counter) == 0) { /* Check for 1 */ \
      *PP_CAT(var, _bmdata) = PP_CAT(var, _data); \
      ++PP_CAT(var, _bmdata); \
      PP_CAT(var, _data) = 0; \
    } \
  } while (0)
#define BITMAP_WRITE_SET_OPERATION(var) \
  do { \
    assert (PP_CAT(var, _bmdata) < PP_CAT(var, _bmdataend)); \
    const unsigned long PP_CAT(var, _mask) = 1UL << PP_CAT(var, _counter); \
    PP_CAT(var, _data) |= PP_CAT(var, _mask); \
  } while (0)
#if 0
#define BITMAP_WRITE_SET_OPERATION_COND(var, cond) /* Cond tells whether to set bit; must be BITMAP_ULONG_HIGHBIT or 0 */ \
  do { \
    assert (PP_CAT(var, _bmdata) < PP_CAT(var, _bmdataend)); \
    const unsigned long PP_CAT(var, _mask) = (cond) >> PP_CAT(var, _counter); \
    PP_CAT(var, _data) |= PP_CAT(var, _mask); \
  } while (0)
#endif
#if 0
#define BITMAP_WRITE_SET_OPERATION_COND(var, cond) /* High bit of cond tells whether to set bit; must be BITMAP_ULONG_HIGHBIT or 0 */ \
  do { \
    unsigned long condx = (cond); \
    assert (PP_CAT(var, _bmdata) < PP_CAT(var, _bmdataend)); \
    /* if (condx != 0 && condx != BITMAP_ULONG_HIGHBIT) {fprintf(stderr, "Invalid value %08lx for BITMAP_WRITE_SET_OPERATION_COND condition\n", condx); abort();} */ \
    const unsigned long PP_CAT(var, _mask) = condx << PP_CAT(var, _counter); \
    PP_CAT(var, _data) |= PP_CAT(var, _mask); \
  } while (0)
#endif
#if 0
#define BITMAP_WRITE_SET_OPERATION_COND_NOT(var, cond) /* Complemented high bit of cond tells whether to set bit; must be BITMAP_ULONG_HIGHBIT or 0 */ \
  do { \
    unsigned long condx = (cond); \
    assert (PP_CAT(var, _bmdata) < PP_CAT(var, _bmdataend)); \
    /* if (condx != 0 && condx != BITMAP_ULONG_HIGHBIT) {fprintf(stderr, "Invalid value %08lx for BITMAP_WRITE_SET_OPERATION_COND_NOT condition\n", condx); abort();} */ \
    const unsigned long PP_CAT(var, _mask) = (condx ^ BITMAP_ULONG_HIGHBIT) >> PP_CAT(var, _counter); \
    PP_CAT(var, _data) |= PP_CAT(var, _mask); \
  } while (0)
#endif
#if 0
#define BITMAP_WRITE_SET_OPERATION_COND1(var, cond) /* Cond must be either 0 or 1 */ \
  do { \
    assert (PP_CAT(var, _bmdata) < PP_CAT(var, _bmdataend)); \
    assert (cond == 0 || cond == 1); \
    const unsigned long PP_CAT(var, _mask) = 1UL << PP_CAT(var, _counter); \
    PP_CAT(var, _data) |= PP_CAT(var, _mask) & -(unsigned long)(cond); \
  } while (0)
#define BITMAP_WRITE_SET_OPERATION_COND1_NOT(var, cond) /* Cond must be either 0 or 1 */ \
  do { \
    assert (PP_CAT(var, _bmdata) < PP_CAT(var, _bmdataend)); \
    assert (cond == 0 || cond == 1); \
    const unsigned long PP_CAT(var, _mask) = 1UL << PP_CAT(var, _counter); \
    PP_CAT(var, _data) |= PP_CAT(var, _mask) & ((unsigned long)(cond) - 1); \
  } while (0)
#endif
#define BITMAP_FLUSH_SET_OPERATION(var) \
  do { \
    if (PP_CAT(var, _bmdata) < PP_CAT(var, _bmdataend)) { \
      *PP_CAT(var, _bmdata) = PP_CAT(var, _data); \
    } \
  } while (0)

#define BITMAP_DECLARE_READ_OPERATION(bitmap_expr, var) \
  size_t const restrict PP_CAT(var, _bmdatalen) = (bitmap_expr).get_nwords(); \
  const unsigned long* restrict PP_CAT(var, _bmdata) = (bitmap_expr).get_data(0, PP_CAT(var, _bmdatalen)); \
  const unsigned long* restrict PP_CAT(var, _bmdataend) = PP_CAT(var, _bmdata) + PP_CAT(var, _bmdatalen); \
  (void)PP_CAT(var, _bmdataend); \
  unsigned long PP_CAT(var, _mask) = 1UL; \
  unsigned long PP_CAT(var, _data) = (PP_CAT(var, _bmdata) == PP_CAT(var, _bmdataend) ? 0 : *PP_CAT(var, _bmdata));
#define BITMAP_STEP_READ_OPERATION(var) \
  do { \
    PP_CAT(var, _mask) <<= 1; \
    if (PP_CAT(var, _mask) == 0) { \
      ++PP_CAT(var, _bmdata); \
      PP_CAT(var, _mask) = 1UL; \
      PP_CAT(var, _data) = (PP_CAT(var, _bmdata) == PP_CAT(var, _bmdataend) ? 0 : *PP_CAT(var, _bmdata)); \
    } \
  } while (0)
#define BITMAP_GET_FROM_READ_OPERATION(var) ((PP_CAT(var, _mask) & PP_CAT(var, _data)) != 0)

// In both of these, offset_expr must be a multiple of BITMAP_ULONG_BITS
#define BITMAP_SAVE_PTRS_INTO_LOCAL_VARS(bitmap_expr, offset_expr, var) \
  bitmap& PP_CAT(var, _saveref) = (bitmap_expr); \
  uint64_t const PP_CAT(var, _bmdatalen) = PP_CAT(var, _saveref).get_length(); \
  uint64_t const PP_CAT(var, _bmoffset) = (offset_expr); \
  assert (offset_expr % BITMAP_ULONG_BITS == 0); \
  unsigned long* const restrict PP_CAT(var, _bmdata) = PP_CAT(var, _saveref).get_data(size_t(PP_CAT(var, _bmoffset) / BITMAP_ULONG_BITS), size_t(PP_CAT(var, _bmdatalen) / BITMAP_ULONG_BITS - PP_CAT(var, _bmoffset) / BITMAP_ULONG_BITS));
#define BITMAP_SAVE_PTRS_INTO_LOCAL_VARS_CONST(bitmap_expr, offset_expr, var) \
  const bitmap& PP_CAT(var, _saveref) = (bitmap_expr); \
  uint64_t const PP_CAT(var, _bmdatalen) = PP_CAT(var, _saveref).get_length(); \
  uint64_t const PP_CAT(var, _bmoffset) = (offset_expr); \
  const unsigned long* const restrict PP_CAT(var, _bmdata) = PP_CAT(var, _saveref).get_data(size_t(PP_CAT(var, _bmoffset) / BITMAP_ULONG_BITS), size_t(PP_CAT(var, _bmdatalen) / BITMAP_ULONG_BITS - PP_CAT(var, _bmoffset) / BITMAP_ULONG_BITS));
#define BITMAP_GET_LOCAL(var, idx) (bitmap_fast_get((idx), PP_CAT(var, _bmdata), PP_CAT(var, _bmdatalen)))
#define BITMAP_TEST_AND_SET_NON_ATOMIC_LOCAL(var, idx) (bitmap_fast_tas_na((idx), PP_CAT(var, _bmdata), PP_CAT(var, _bmdatalen)))
#define BITMAP_TEST_AND_SET_INV_NON_ATOMIC_LOCAL(var, idx) (bitmap_fast_tas_na_inv((idx), PP_CAT(var, _bmdata), PP_CAT(var, _bmdatalen)))

#if 0
#define BITMAP_ITER_BEGIN_COMMON(b, idxvar, maskvar, in_parallel) do { \
  size_t bitmap_word_idx, bitmap_word_idx_end = (b).nwords; \
  unsigned long* restrict b_data_copy = (b).data; \
  in_parallel for (bitmap_word_idx = 0; bitmap_word_idx < bitmap_word_idx_end; ++bitmap_word_idx) { \
    unsigned long bitmap_word_data = b_data_copy[bitmap_word_idx]; \
    /* Iterate through set bits in word; increment here clears the most
     * significant set bit. */ \
    for (; bitmap_word_data != 0; bitmap_word_data &= ~(BITMAP_ULONG_HIGHBIT >> __builtin_clzl(bitmap_word_data))) { \
      int bitmap_bit_idx = __builtin_clzl(bitmap_word_data); \
      uint64_t idxvar = (uint64_t)bitmap_word_idx * BITMAP_ULONG_BITS + bitmap_bit_idx; \
      unsigned long maskvar = bitmap_word_data & (BITMAP_ULONG_HIGHBIT >> bitmap_bit_idx);

#define BITMAP_ITER_BEGIN(b, idxvar, maskvar) BITMAP_ITER_BEGIN_COMMON(b, idxvar, maskvar, /**/)
#define BITMAP_ITER_BEGIN_PAR(b, idxvar, maskvar) BITMAP_ITER_BEGIN_COMMON(b, idxvar, maskvar, _Pragma("omp parallel for"))

#define BITMAP_ITER_END }}} while (0)
#endif

/* Compute b = A * x, where A is an unsymmetric matrix in coordinate form (high
 * 32-bit word is row, low word is column); mark the element in A that triggers
 * the first setting of an element in b in marks.  The b and marks bitmaps will
 * not be updated by any other threads, and can be assumed to be cleared and
 * correctly sized. */
void bitmap_matvec_and_mark(const uint64_t* const A, size_t Alen, const bitmap& x, uint64_t x_offset, uint64_t xlen, bitmap& b, uint64_t b_offset, uint64_t blen, bitmap& marks);

/* Compute b = A^T * x, where A is an unsymmetric matrix in coordinate form
 * (high 32-bit word is row, low word is column); mark the element in A that
 * triggers the first setting of an element in b in marks.  The b and marks
 * bitmaps will not be updated by any other threads, and can be assumed to be
 * cleared and correctly sized. */
void bitmap_matvec_trans_and_mark(const uint64_t* const A, size_t Alen, const bitmap& x, uint64_t x_offset, uint64_t xlen, bitmap& b, uint64_t b_offset, uint64_t blen, bitmap& marks);

/* Compute b = (A | A^T) * x, where A is an unsymmetric matrix in coordinate
 * form (high 32-bit word is row, low word is column); mark the element in A
 * that triggers the first setting of an element in b in marks, and the element
 * in A^T ... in marks_t.  The b, marks, and marks_t bitmaps will not be
 * updated by any other threads, and can be assumed to be cleared and correctly
 * sized. */
void bitmap_matvec_symm_and_mark(const uint64_t* const A, size_t Alen, const bitmap& x, uint64_t x_offset, uint64_t xlen, bitmap& b, uint64_t b_offset, uint64_t blen, bitmap& marks, bitmap& marks_t);

#endif /* BITMAP_HPP */
