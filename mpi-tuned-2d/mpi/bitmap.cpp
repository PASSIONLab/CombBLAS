/* Copyright (C) 2010-2013 The Trustees of Indiana University.             */
/*                                                                         */
/* Use, modification and distribution is subject to the Boost Software     */
/* License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at */
/* http://www.boost.org/LICENSE_1_0.txt)                                   */
/*                                                                         */
/*  Authors: Jeremiah Willcock                                             */
/*           Andrew Lumsdaine                                              */

#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
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
#include <utility>
#include "bitmap.hpp"
#include "common.hpp"
#include "kernels.h"

bitmap::bitmap(uint64_t len) {
  assert ((len + CHAR_BIT - 1) / CHAR_BIT < PTRDIFF_MAX);
  this->length = len;
  this->nwords = (size_t)((len + BITMAP_ULONG_BITS - 1) / BITMAP_ULONG_BITS);
  memalloc("bitmap constructor", this->nwords * sizeof(unsigned long));
  this->data = (unsigned long*)xmalloc(this->nwords * sizeof(unsigned long));
  this->reset_all();
}

void bitmap::ensure_size(uint64_t newlen) {
  assert ((newlen + CHAR_BIT - 1) / CHAR_BIT < PTRDIFF_MAX);
  if (newlen <= this->length) return;
  if (newlen <= (uint64_t)(this->nwords * BITMAP_ULONG_BITS)) {this->length = newlen; return;}
  size_t old_nwords = this->nwords;
  memfree("bitmap ensure_size", old_nwords * sizeof(unsigned long));
  this->nwords = (size_t)((newlen + BITMAP_ULONG_BITS - 1) / BITMAP_ULONG_BITS);
  memalloc("bitmap ensure_size", this->nwords * sizeof(unsigned long));
  this->data = (unsigned long*)xrealloc(this->data, this->nwords * sizeof(unsigned long));
  memset(this->data + old_nwords, 0, (this->nwords - old_nwords) * sizeof(unsigned long));
  this->length = newlen;
}

void bitmap::clear() {
  memfree("bitmap clear", this->nwords * sizeof(unsigned long));
  xfree(this->data); this->data = NULL;
  this->length = 0;
  this->nwords = 0;
}

bitmap::~bitmap() {
  this->clear();
}

void bitmap::reset_all() {
  memset(this->data, 0, this->nwords * sizeof(unsigned long));
}

void bitmap::union_all(const bitmap& restrict b_in, MPI_Comm comm) {
  assert (b_in.length == this->length);
  MPI_Allreduce(b_in.data, this->data, (int)b_in.nwords, MPI_UNSIGNED_LONG, MPI_BOR, comm);
}

void bitmap::union_all_inplace(MPI_Comm comm) {
  MPI_Allreduce(MPI_IN_PLACE, this->data, (int)this->nwords, MPI_UNSIGNED_LONG, MPI_BOR, comm);
}

/* a = a &~ b */
bool bitmap::and_not(const bitmap& restrict b) {
  assert (this->length == b.length);
  unsigned long* restrict this_data = this->data;
  const unsigned long* restrict b_data = b.data;
  ptrdiff_t nwords = (ptrdiff_t)this->nwords;
  ptrdiff_t i;
  unsigned long any_set = 0;
#pragma omp parallel for reduction(|:any_set)
  for (i = 0; i < nwords; ++i) {
    unsigned long temp;
    temp = b_data[i];
    temp = ~temp & this_data[i];
    this_data[i] = temp;
    // if (temp) any_set |= 1;
    any_set |= temp;
  }
  return any_set != 0;
}

/* a = b &~ a */
void bitmap::and_not_rev(const bitmap& restrict b) {
  assert (this->length == b.length);
  unsigned long* restrict this_data = this->data;
  const unsigned long* restrict b_data = b.data;
  ptrdiff_t nwords = (ptrdiff_t)this->nwords;
  ptrdiff_t i;
#pragma omp parallel for
  for (i = 0; i < nwords; ++i) {
    this_data[i] = (b_data[i] & ~this_data[i]);
  }
}

/* a = a | b */
bitmap& bitmap::operator|=(const bitmap& restrict b) {
  assert (this->length == b.length);
  unsigned long* restrict this_data = this->data;
  const unsigned long* restrict b_data = b.data;
  bitmap_or(this_data, this->length, b_data);
  return *this;
}

/* a = b */
bitmap& bitmap::operator=(const bitmap& restrict b) {
  assert (this->length == b.length);
  unsigned long* restrict this_data = this->data;
  const unsigned long* restrict b_data = b.data;
  size_t nwords = this->nwords;
  memcpy(this_data, b_data, nwords * sizeof(unsigned long));
  return *this;
}

void bitmap::swap(bitmap& b) {
  if (&b == this) return;
  std::swap(this->data, b.data);
  std::swap(this->length, b.length);
  std::swap(this->nwords, b.nwords);
}

bool bitmap::are_all_reset() const {
  bool acc = false;
  for (size_t i = 0; i < this->nwords; ++i) if (this->data[i]) {acc = true; break;}
  return !acc;
}

/* In all of these, note that indexes into the bitmaps are relative to x_offset
 * and b_offset. */

#if 0
/* Compute b = A * x, where A is an unsymmetric matrix in coordinate form (high
 * 32-bit word is row, low word is column); mark the element in A that triggers
 * the first setting of an element in b in marks.  The b and marks bitmaps will
 * not be updated by any other threads, and can be assumed to be cleared and
 * correctly sized. */
void bitmap_matvec_and_mark(const uint64_t* const A, size_t Alen, const bitmap& x, uint64_t x_offset, uint64_t xlen, bitmap& b, uint64_t b_offset, uint64_t blen, bitmap& marks) {
  size_t const marks_bmdatalen = marks.get_nwords();
  unsigned long* restrict marks_bmdata = marks.get_data(0, marks_bmdatalen);
  BITMAP_SAVE_PTRS_INTO_LOCAL_VARS_CONST(x, x_offset, x_data)
  BITMAP_SAVE_PTRS_INTO_LOCAL_VARS(b, b_offset, b_data)
  bitmap_matvec_and_mark_impl(A, Alen, x_data_bmdata, xlen, b_data_bmdata, blen, marks_bmdata);
}
#endif

#if 0
/* Compute b = A^T * x, where A is an unsymmetric matrix in coordinate form
 * (high 32-bit word is row, low word is column); mark the element in A that
 * triggers the first setting of an element in b in marks.  The b and marks
 * bitmaps will not be updated by any other threads, and can be assumed to be
 * cleared and correctly sized. */
void bitmap_matvec_trans_and_mark(const uint64_t* const A, size_t Alen, const bitmap& x, uint64_t x_offset, uint64_t xlen, bitmap& b, uint64_t b_offset, uint64_t blen, bitmap& marks) {
  size_t const marks_bmdatalen = marks.get_nwords();
  unsigned long* restrict marks_bmdata = marks.get_data(0, marks_bmdatalen);
  BITMAP_SAVE_PTRS_INTO_LOCAL_VARS_CONST(x, x_offset, x_data)
  BITMAP_SAVE_PTRS_INTO_LOCAL_VARS(b, b_offset, b_data)
  bitmap_matvec_trans_and_mark_impl(A, Alen, x_data_bmdata, xlen, b_data_bmdata, blen, marks_bmdata);
}
#endif

#if 0
/* Compute b = (A | A^T) * x, where A is an unsymmetric matrix in coordinate
 * form (high 32-bit word is row, low word is column); mark the element in A
 * that triggers the first setting of an element in b in marks, and the element
 * in A^T ... in marks_t.  The b, marks, and marks_t bitmaps will not be
 * updated by any other threads, and can be assumed to be cleared and correctly
 * sized. */
void bitmap_matvec_symm_and_mark(const uint64_t* const A, size_t Alen, const bitmap& x, uint64_t x_offset, uint64_t xlen, bitmap& b, uint64_t b_offset, uint64_t blen, bitmap& marks, bitmap& marks_t) {
  size_t const marks_bmdatalen = marks.get_nwords();
  unsigned long* restrict marks_bmdata = marks.get_data(0, marks_bmdatalen);
  size_t const marks_t_bmdatalen = marks_t.get_nwords();
  unsigned long* restrict marks_t_bmdata = marks_t.get_data(0, marks_t_bmdatalen);
  BITMAP_SAVE_PTRS_INTO_LOCAL_VARS_CONST(x, x_offset, x_data)
  BITMAP_SAVE_PTRS_INTO_LOCAL_VARS(b, b_offset, b_data)
  bitmap_matvec_symm_and_mark_impl(A, Alen, x_data_bmdata, xlen, b_data_bmdata, blen, marks_bmdata, marks_t_bmdata);
}
#endif
