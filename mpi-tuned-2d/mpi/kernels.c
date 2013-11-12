/* Copyright (C) 2010-2013 The Trustees of Indiana University.             */
/*                                                                         */
/* Use, modification and distribution is subject to the Boost Software     */
/* License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at */
/* http://www.boost.org/LICENSE_1_0.txt)                                   */
/*                                                                         */
/*  Authors: Jeremiah Willcock                                             */
/*           Andrew Lumsdaine                                              */

#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <limits.h>
#include <assert.h>
#include <string.h>
#include "kernels.h"

#ifdef USE_VALGRIND
#include <valgrind/callgrind.h>
void callgrind_start(void) {CALLGRIND_START_INSTRUMENTATION;}
void callgrind_stop(void) {CALLGRIND_STOP_INSTRUMENTATION;}
#else
void callgrind_start(void) {}
void callgrind_stop(void) {}
#endif

// static inline double get_time_base(void) {return MPI_Wtime();}
static inline double get_time_base(void) {struct timeval tv; gettimeofday(&tv, NULL); return tv.tv_sec + tv.tv_usec * 1.e-6;}

#define BITMAP_ULONG_BITS (sizeof(unsigned long) * CHAR_BIT)
#define BITMAP_ULONG_HIGHBIT (1UL << (BITMAP_ULONG_BITS - 1))

#ifdef __GNUC__
#define EXPECT(a, b) __builtin_expect((a), (b))
#else
#define EXPECT(a, b) (a)
#endif

#ifdef __IBMC__
#if defined(__powerpc64__)
#define rotatel_ul __rotatel8
#define popcount_ul __popcnt8
#define cntlz_ul __cntlz8
#define cnttz_ul __cnttz8
#else
#define rotatel_ul __rotatel4
// #define popcount_ul __popcnt4
#define popcount_ul popcount_word
#define cntlz_ul __cntlz4
#define cnttz_ul __cnttz4
#endif
#elif defined(_CRAYC_) && defined(__x86_64__)
#define popcount_ul _popcnt64
#define cntlz_ul _leadz64
#else
#define popcount_ul __builtin_popcountl
#define cntlz_ul __builtin_clzl
#define cnttz_ul __builtin_ctzl
#endif

#if defined(__powerpc__) && !defined(__powerpc64__)
static inline int popcount_word(uint32_t x) {
  x -= (x >> 1) & UINT32_C(0x55555555);
  x = (x & UINT32_C(0x33333333)) + ((x >> 2) & UINT32_C(0x33333333));
  x = (x + (x >> 4)) & UINT32_C(0x0F0F0F0F);
  return ((x * UINT32_C(0x01010101)) >> 24) & 0xFF;
}
#endif

/* Even though runs at 128 nodes show that the count == UINT8_MAX condition is
 * often true, it is basically never true at or above 2048 ranks because of the
 * larger, sparser graphs used at those sizes.  Thus, it is marked as
 * infrequent. */

#ifndef USE_ASM_VERSIONS_FOR_MATVEC

/* Compute b = A * x, where A is an unsymmetric matrix in coordinate form (high
 * 32-bit word is row, low word is column); mark the element in A that triggers
 * the first setting of an element in b in marks.  The b and marks bitmaps will
 * not be updated by any other threads; both must be accumulated into. */
void bitmap_matvec_and_mark_impl(const uint16_t* restrict const A_row_low_halfwords, const uint16_t* restrict const A_col_low_halfwords, size_t Alen, const uint8_t* restrict const edge_block_counts_by_block, const unsigned long* restrict x, uint64_t xlen, const uint8_t* restrict const xempty, unsigned long* restrict b, uint64_t blen, unsigned long* restrict marks) {
  double start_time[2], end_time[2];
  (void)Alen;
  size_t i = 0, byte_idx = 0;
  for (size_t row_blk = 0; row_blk < (xlen >> 16); ++row_blk) {
    if (!xempty[row_blk]) {
      const unsigned long* restrict this_x_base = x + row_blk * ((1 << 16) / BITMAP_ULONG_BITS);
      for (size_t col_blk = 0; col_blk < (blen >> 16); ++col_blk) {
        size_t count = edge_block_counts_by_block[byte_idx++];
        if (EXPECT(count == UINT8_MAX, 0)) {
#ifdef __IBMC__
#pragma execution_frequency (very_low)
#endif
          while (1) {
            uint8_t code = edge_block_counts_by_block[byte_idx++];
            count += code;
            if (code != UINT8_MAX) break;
          }
        }
        unsigned long* restrict this_b_base = b + col_blk * ((1 << 16) / BITMAP_ULONG_BITS);
        const uint16_t* restrict this_A_row_low_halfwords_base = A_row_low_halfwords + i;
        const uint16_t* restrict this_A_col_low_halfwords_base = A_col_low_halfwords + i;
        for (size_t j = 0; j < count; ++j) {
          unsigned int row_low = this_A_row_low_halfwords_base[j];
#if 0
          if ((row >> 16) != row_blk) {
            fprintf(stderr, "Edge location mismatch: on block %u, found edge (%" PRIu32 ", %" PRIu32 ")\n", row_blk, row, col);
            abort();
          }
#endif
          unsigned long temp = this_x_base[row_low / BITMAP_ULONG_BITS];
          if (temp & (1UL << (row_low % BITMAP_ULONG_BITS))) {
            unsigned int col_low = this_A_col_low_halfwords_base[j];
            unsigned long temp2 = this_b_base[col_low / BITMAP_ULONG_BITS];
            if (!(temp2 & (1UL << (col_low % BITMAP_ULONG_BITS)))) {
              this_b_base[col_low / BITMAP_ULONG_BITS] = temp2 | (1UL << (col_low % BITMAP_ULONG_BITS));
              marks[(i + j) / BITMAP_ULONG_BITS] |= 1UL << ((i + j) % BITMAP_ULONG_BITS); // rotatel_ul(mask & ~temp2, col_low - (i + j));
            }
          }
        }
        i += count;
      }
    } else {
#if 0
      for (size_t j = 0; j < count; ++j) {
        uint64_t elt = A[i + j];
        uint32_t row = (uint32_t)(elt >> 32);
        uint32_t col = (uint32_t)(elt);
        if ((row >> 16) != row_blk) {
          fprintf(stderr, "Edge location mismatch: on block %u, found edge (%" PRIu32 ", %" PRIu32 ")\n", row_blk, row, col);
          abort();
        }
      }
#endif
      for (size_t col_blk = 0; col_blk < (xlen >> 16); ++col_blk) {
        size_t count = edge_block_counts_by_block[byte_idx++];
        if (EXPECT(count == UINT8_MAX, 0)) {
#ifdef __IBMC__
#pragma execution_frequency (very_low)
#endif
          while (1) {
            uint8_t code = edge_block_counts_by_block[byte_idx++];
            count += code;
            if (code != UINT8_MAX) break;
          }
        }
        i += count;
      }
    }
  }
}

/* Compute b = A^T * x, where A is an unsymmetric matrix in coordinate form
 * (high 32-bit word is row, low word is column); mark the element in A that
 * triggers the first setting of an element in b in marks.  The b and marks
 * bitmaps will not be updated by any other threads; both must be accumulated
 * into. */
void bitmap_matvec_trans_and_mark_impl(const uint16_t* restrict const A_row_low_halfwords, const uint16_t* restrict const A_col_low_halfwords, size_t Alen, const uint8_t* restrict const edge_block_counts_by_block, const unsigned long* restrict x, uint64_t xlen, const uint8_t* restrict const xempty, unsigned long* restrict b, uint64_t blen, unsigned long* restrict marks) {
  double start_time[2], end_time[2];
  (void)Alen;
  size_t i = 0, byte_idx = 0;
  for (size_t row_blk = 0; row_blk < (blen >> 16); ++row_blk) {
    unsigned long* restrict this_b_base = b + row_blk * ((1 << 16) / BITMAP_ULONG_BITS);
    for (size_t col_blk = 0; col_blk < (xlen >> 16); ++col_blk) {
      size_t count = edge_block_counts_by_block[byte_idx++];
      if (EXPECT(count == UINT8_MAX, 0)) {
#ifdef __IBMC__
#pragma execution_frequency (very_low)
#endif
        while (1) {
          uint8_t code = edge_block_counts_by_block[byte_idx++];
          count += code;
          if (code != UINT8_MAX) break;
        }
      }
      if (!xempty[col_blk]) {
        const unsigned long* restrict this_x_base = x + col_blk * ((1 << 16) / BITMAP_ULONG_BITS);
        const uint16_t* restrict this_A_row_low_halfwords_base = A_row_low_halfwords + i;
        const uint16_t* restrict this_A_col_low_halfwords_base = A_col_low_halfwords + i;
        for (size_t j = 0; j < count; ++j) {
          // uint64_t elt = A[i + j];
          unsigned int col_low = this_A_col_low_halfwords_base[j];
#if 0
          if ((row >> 16) != row_blk || (col >> 16) != col_blk) {
            fprintf(stderr, "Edge location mismatch: on block (%u, %u), found edge (%" PRIu32 ", %" PRIu32 ")\n", row_blk, col_blk, row, col);
            abort();
          }
#endif
          unsigned long temp = this_x_base[col_low / BITMAP_ULONG_BITS];
          if (temp & (1UL << (col_low % BITMAP_ULONG_BITS))) { // rotatel_ul(temp, col_low) >= BITMAP_ULONG_HIGHBIT) {
            unsigned int row_low = this_A_row_low_halfwords_base[j];
            unsigned long temp2 = this_b_base[row_low / BITMAP_ULONG_BITS];
            if (!(temp2 & (1UL << (row_low % BITMAP_ULONG_BITS)))) {
              this_b_base[row_low / BITMAP_ULONG_BITS] = temp2 | (1UL << (row_low % BITMAP_ULONG_BITS));
              marks[(i + j) / BITMAP_ULONG_BITS] |= 1UL << ((i + j) % BITMAP_ULONG_BITS); // rotatel_ul(mask & ~temp2, col_low - (i + j));
            }
          }
        }
      } else {
#if 0
        for (unsigned int j = 0; j < count; ++j) {
          uint64_t elt = A[i + j];
          uint32_t row = (uint32_t)(elt >> 32);
          uint32_t col = (uint32_t)(elt);
          if ((row >> 16) != row_blk || (col >> 16) != col_blk) {
            fprintf(stderr, "Edge location mismatch: on block (%u, %u), found edge (%" PRIu32 ", %" PRIu32 ")\n", row_blk, col_blk, row, col);
            abort();
          }
        }
#endif
      }
      i += count;
    }
  }
}

#endif // End of section replaced by assembly code

/* Copy data into all elements of an array where bit is set in bitmap. */
void write_levels_for_in_queue(int64_t* restrict const out, const int64_t data, const unsigned long* restrict const bm, const uint64_t bmlen) {
  const size_t bmlen_in_words = (size_t)((bmlen + BITMAP_ULONG_BITS - 1) / BITMAP_ULONG_BITS);
  for (size_t i = 0; i < bmlen_in_words; ++i) {
#if 1
    for (unsigned long w = bm[i]; w != 0; w &= w - 1) { // Iterate set bits from LSB to MSB
      int bit_idx = cnttz_ul(w); // Get position of current LSB
      assert (out[i * BITMAP_ULONG_BITS + bit_idx] == UINT64_MAX);
      out[i * BITMAP_ULONG_BITS + bit_idx] = data;
    }
#else
    for (int bit_idx = 0; bit_idx < BITMAP_ULONG_BITS; ++bit_idx) {
      if (bm[i] & (1UL << bit_idx)) {
        out[i * BITMAP_ULONG_BITS + bit_idx] = data;
      }
    }
#endif
  }
}

/* Clear marks for edges whose targets are not set in oq */
void clear_marks_by_target(const uint64_t* restrict const edges, const size_t nedges, const unsigned long* restrict const oq, const uint64_t oqlen, unsigned long* restrict const marks) {
  const size_t marks_len_in_words = (size_t)((nedges + BITMAP_ULONG_BITS - 1) / BITMAP_ULONG_BITS);
  for (size_t i = 0; i < marks_len_in_words; ++i) {
    unsigned long mark_data = marks[i];
    if (mark_data == 0) continue;
    for (int bit = 0; bit < BITMAP_ULONG_BITS; ++bit) {
      if (mark_data & (1UL << bit)) {
        const uint32_t tgt = (uint32_t)(edges[i * BITMAP_ULONG_BITS + bit]);
        assert (tgt < oqlen); (void)oqlen;
        const unsigned long tgt_data = oq[tgt / BITMAP_ULONG_BITS];
        if (!(tgt_data & (1UL << (tgt % BITMAP_ULONG_BITS)))) { // Not set
          mark_data &= ~(1UL << bit); // Clear bit in mark array
        }
      }
    }
    marks[i] = mark_data;
  }
}

/* Clear marks for edges whose targets are not set in oq (transposed) */
void clear_marks_by_target_trans(const uint64_t* restrict const edges, const size_t nedges, const unsigned long* restrict const oq, const uint64_t oqlen, unsigned long* restrict const marks) {
  const size_t marks_len_in_words = (size_t)((nedges + BITMAP_ULONG_BITS - 1) / BITMAP_ULONG_BITS);
  for (size_t i = 0; i < marks_len_in_words; ++i) {
    unsigned long mark_data = marks[i];
    if (mark_data == 0) continue;
    for (int bit = 0; bit < BITMAP_ULONG_BITS; ++bit) {
      if (mark_data & (1UL << bit)) {
        const uint32_t tgt = (uint32_t)(edges[i * BITMAP_ULONG_BITS + bit] >> 32);
        assert (tgt < oqlen); (void)oqlen;
        const unsigned long tgt_data = oq[tgt / BITMAP_ULONG_BITS];
        if (!(tgt_data & (1UL << (tgt % BITMAP_ULONG_BITS)))) { // Not set
          mark_data &= ~(1UL << bit); // Clear bit in mark array
        }
      }
    }
    marks[i] = mark_data;
  }
}

void bitmap_or(unsigned long* restrict b, uint64_t blen, const unsigned long* restrict a) {
  for (size_t i = 0; i < (blen + BITMAP_ULONG_BITS - 1) / BITMAP_ULONG_BITS; ++i) {
    b[i] |= a[i];
  }
}

void bitmap_andc_outofplace(unsigned long* restrict a, uint64_t alen, const unsigned long* restrict b, const unsigned long* restrict c) {
  for (size_t i = 0; i < (alen + BITMAP_ULONG_BITS - 1) / BITMAP_ULONG_BITS; ++i) {
    a[i] = b[i] & ~c[i];
  }
}

bool bitmap_andc(unsigned long* restrict a, uint64_t alen, const unsigned long* restrict b) {
  unsigned long combined = 0;
  for (size_t i = 0; i < (alen + BITMAP_ULONG_BITS - 1) / BITMAP_ULONG_BITS; ++i) {
    unsigned long temp = a[i] & ~b[i];
    a[i] = temp;
    combined |= temp;
  }
  return (combined != 0);
}

void bitmap_andc_then_or(unsigned long* restrict c, uint64_t alen, const unsigned long* restrict a, const unsigned long* restrict b) {
  for (size_t i = 0; i < (alen + BITMAP_ULONG_BITS - 1) / BITMAP_ULONG_BITS; ++i) {
    unsigned long temp = a[i] & ~b[i];
    c[i] |= temp;
  }
}

void bitmap_or_andc(unsigned long* restrict a, uint64_t alen, unsigned long* restrict b) {
  for (size_t i = 0; i < (alen + BITMAP_ULONG_BITS - 1) / BITMAP_ULONG_BITS; ++i) {
    unsigned long av = a[i];
    unsigned long bv = b[i];
    a[i] = av | bv;
    b[i] = bv & ~av;
  }
}

void bitmap_andc_rev(unsigned long* restrict b, uint64_t blen, const unsigned long* restrict a) {
  for (size_t i = 0; i < (blen + BITMAP_ULONG_BITS - 1) / BITMAP_ULONG_BITS; ++i) {
    b[i] = a[i] & ~b[i];
  }
}

bool bitmap_difference_and_or(unsigned long* restrict a, uint64_t alen, unsigned long* restrict b, unsigned long* restrict c) {
  unsigned long combined = 0;
  for (size_t i = 0; i < (alen + BITMAP_ULONG_BITS - 1) / BITMAP_ULONG_BITS; ++i) {
    unsigned long adata = a[i];
    unsigned long bdata = b[i];
    unsigned long cdata = c[i];
    a[i] = adata & ~cdata;
    b[i] = bdata & ~cdata;
    c[i] = adata | bdata | cdata;
    combined |= adata & ~cdata;
  }
  return (combined != 0);
}

size_t popcount_range(const unsigned long* restrict a, uint64_t alen) {
  size_t c = 0;
  size_t wordcount = (size_t)((alen + BITMAP_ULONG_BITS - 1) / BITMAP_ULONG_BITS);
  for (size_t i = 0; i < wordcount; ++i) c += (size_t)popcount_ul(a[i]);
  return c;
}

#if 1
void write_pred_scatter_data(const uint64_t* restrict const A, size_t Alen, const unsigned long* restrict marks, const unsigned long* restrict out_queue, const int* restrict owners, int* restrict insert_positions, addr_data_pair* restrict send_data, int owner_idx_shift, const uint32_t* restrict local_biases, const uint32_t data_bias) {
  for (size_t edge_idx_outer = 0; edge_idx_outer < Alen; edge_idx_outer += BITMAP_ULONG_BITS) {
    unsigned long mark_data = marks[edge_idx_outer / BITMAP_ULONG_BITS];
    for (; mark_data != 0; mark_data &= mark_data - 1) { // Iterate through set bits from LSB to MSB
      size_t edge_idx_inner = cnttz_ul(mark_data);
      size_t edge_idx = edge_idx_outer + edge_idx_inner;
      uint64_t e = A[edge_idx];
      uint32_t v0_local = (uint32_t)(e >> 32), v1_local = (uint32_t)(e);
      unsigned int idx = v1_local >> owner_idx_shift;
      if (out_queue[v1_local / BITMAP_ULONG_BITS] & (1UL << (v1_local % BITMAP_ULONG_BITS))) {
        int pos = insert_positions[owners[idx]]++;
        send_data[pos].addr = local_biases[idx] + v1_local;
        send_data[pos].data = data_bias + v0_local;
      }
    }
  }
}

void write_pred_scatter_data_trans(const uint64_t* restrict const A, size_t Alen, const unsigned long* restrict marks_t, const unsigned long* restrict out_queue, const int* restrict owners, int* restrict insert_positions, addr_data_pair* restrict send_data, int owner_idx_shift, const uint32_t* restrict local_biases, const uint32_t data_bias) {
  for (size_t edge_idx_outer = 0; edge_idx_outer < Alen; edge_idx_outer += BITMAP_ULONG_BITS) {
    unsigned long mark_t_data = marks_t[edge_idx_outer / BITMAP_ULONG_BITS];
    for (; mark_t_data != 0; mark_t_data &= mark_t_data - 1) { // Iterate through set bits from LSB to MSB
      size_t edge_idx_inner = cnttz_ul(mark_t_data);
      size_t edge_idx = edge_idx_outer + edge_idx_inner;
      uint64_t e = A[edge_idx];
      uint32_t v0_local = (uint32_t)(e >> 32), v1_local = (uint32_t)(e);
      unsigned int idx = v0_local >> owner_idx_shift;
      if (out_queue[v0_local / BITMAP_ULONG_BITS] & (1UL << (v0_local % BITMAP_ULONG_BITS))) {
        int pos = insert_positions[owners[idx]]++;
        send_data[pos].addr = local_biases[idx] + v0_local;
        send_data[pos].data = data_bias + v1_local;
      }
    }
  }
}

#else
void write_pred_scatter_data(const uint64_t* restrict const A, size_t Alen, const unsigned long* restrict marks, const unsigned long* restrict out_queue, const int* restrict owners, int* restrict insert_positions, uint32_t* restrict send_addr, uint64_t* restrict send_data, int owner_idx_shift, const uint32_t* restrict local_biases, const uint64_t data_bias) {
  for (size_t edge_idx_outer = 0; edge_idx_outer + BITMAP_ULONG_BITS - 1 < Alen; edge_idx_outer += BITMAP_ULONG_BITS) {
    unsigned long mark_data = marks[edge_idx_outer / BITMAP_ULONG_BITS];
    if (mark_data == 0) continue;
    for (size_t edge_idx_inner = 0; edge_idx_inner < BITMAP_ULONG_BITS; ++edge_idx_inner) {
      size_t edge_idx = edge_idx_outer + edge_idx_inner;
      uint64_t e = A[edge_idx];
      uint32_t v0_local = (uint32_t)(e >> 32), v1_local = (uint32_t)(e);
      unsigned int idx = v1_local >> owner_idx_shift;
      if (mark_data & (1UL << edge_idx_inner)) {
        if (out_queue[v1_local / BITMAP_ULONG_BITS] & (1UL << (v1_local % BITMAP_ULONG_BITS))) { // (signed long)rotatel_ul(out_queue[v1_local / BITMAP_ULONG_BITS], v1_local) < 0) {
          int pos = insert_positions[owners[idx]]++;
          send_addr[pos] = local_biases[idx] + v1_local;
          send_data[pos] = data_bias | v0_local;
          assert ((data_bias | v0_local) == (data_bias + v0_local));
        }
      }
    }
  }
  if (Alen % BITMAP_ULONG_BITS != 0) {
    unsigned long mark_data = marks[Alen / BITMAP_ULONG_BITS];
    for (size_t edge_idx_inner = 0; edge_idx_inner < Alen % BITMAP_ULONG_BITS; ++edge_idx_inner) {
      size_t edge_idx = Alen - Alen % BITMAP_ULONG_BITS + edge_idx_inner;
      uint64_t e = A[edge_idx];
      uint32_t v0_local = (uint32_t)(e >> 32), v1_local = (uint32_t)(e);
      unsigned int idx = v1_local >> owner_idx_shift;
      if (mark_data & (1UL << edge_idx_inner)) {
        if (out_queue[v1_local / BITMAP_ULONG_BITS] & (1UL << (v1_local % BITMAP_ULONG_BITS))) { // (signed long)rotatel_ul(out_queue[v1_local / BITMAP_ULONG_BITS], v1_local) < 0) {
          int pos = insert_positions[owners[idx]]++;
          send_addr[pos] = local_biases[idx] + v1_local;
          send_data[pos] = data_bias | v0_local;
          assert ((data_bias | v0_local) == (data_bias + v0_local));
        }
      }
    }
  }
}

void write_pred_scatter_data_trans(const uint64_t* restrict const A, size_t Alen, const unsigned long* restrict marks_t, const unsigned long* restrict out_queue, const int* restrict owners, int* restrict insert_positions, uint32_t* restrict send_addr, uint64_t* restrict send_data, int owner_idx_shift, const uint32_t* restrict local_biases, const uint64_t data_bias) {
  for (size_t edge_idx_outer = 0; edge_idx_outer + BITMAP_ULONG_BITS - 1 < Alen; edge_idx_outer += BITMAP_ULONG_BITS) {
    unsigned long mark_t_data = marks_t[edge_idx_outer / BITMAP_ULONG_BITS];
    if (mark_t_data == 0) continue;
    for (size_t edge_idx_inner = 0; edge_idx_inner < BITMAP_ULONG_BITS; ++edge_idx_inner) {
      size_t edge_idx = edge_idx_outer + edge_idx_inner;
      uint64_t e = A[edge_idx];
      uint32_t v0_local = (uint32_t)(e >> 32), v1_local = (uint32_t)(e);
      unsigned int idx = v0_local >> owner_idx_shift;
      if (mark_t_data & (1UL << edge_idx_inner)) {
        if (out_queue[v0_local / BITMAP_ULONG_BITS] & (1UL << (v0_local % BITMAP_ULONG_BITS))) { // (signed long)rotatel_ul(out_queue[v0_local / BITMAP_ULONG_BITS], v0_local) < 0) {
          int pos = insert_positions[owners[idx]]++;
          send_addr[pos] = local_biases[idx] + v0_local;
          send_data[pos] = data_bias | v1_local;
          assert ((data_bias | v1_local) == (data_bias + v1_local));
        }
      }
    }
  }
  if (Alen % BITMAP_ULONG_BITS != 0) {
    unsigned long mark_t_data = marks_t[Alen / BITMAP_ULONG_BITS];
    for (size_t edge_idx_inner = 0; edge_idx_inner < Alen % BITMAP_ULONG_BITS; ++edge_idx_inner) {
      size_t edge_idx = Alen - Alen % BITMAP_ULONG_BITS + edge_idx_inner;
      uint64_t e = A[edge_idx];
      uint32_t v0_local = (uint32_t)(e >> 32), v1_local = (uint32_t)(e);
      unsigned int idx = v0_local >> owner_idx_shift;
      if (mark_t_data & (1UL << edge_idx_inner)) {
        if (out_queue[v0_local / BITMAP_ULONG_BITS] & (1UL << (v0_local % BITMAP_ULONG_BITS))) { // (signed long)rotatel_ul(out_queue[v0_local / BITMAP_ULONG_BITS], v0_local) < 0) {
          int pos = insert_positions[owners[idx]]++;
          send_addr[pos] = local_biases[idx] + v0_local;
          send_data[pos] = data_bias | v1_local;
          assert ((data_bias | v1_local) == (data_bias + v1_local));
        }
      }
    }
  }
}
#endif

// #define DISABLE_COMPRESSION

static void assert_compressed_bitmap_correct(const unsigned long* restrict const bm_compressed, size_t bm_compressed_size, uint64_t bmlen) {
#ifndef NDEBUG
  assert (bmlen % BITMAP_ULONG_BITS == 0);
  const size_t bmlen_in_words = (size_t)((bmlen + BITMAP_ULONG_BITS - 1) / BITMAP_ULONG_BITS);
  assert (bmlen_in_words % BITMAP_ULONG_BITS == 0);
  const size_t summary_len = bmlen_in_words / BITMAP_ULONG_BITS;
  if (bm_compressed_size == bmlen_in_words) {
    // Not compressed; no further checks possible
  } else if (bm_compressed_size == 0) { // Empty
    // Empty
  } else if (bm_compressed_size < summary_len) {
    // List of elements
    for (size_t i = 0; i < bm_compressed_size; ++i) {
      assert (bm_compressed[i] < bmlen);
    }
  } else {
    assert (bm_compressed_size >= summary_len);
    assert (bm_compressed_size < bmlen_in_words);
    assert (bm_compressed_size - summary_len == popcount_range(bm_compressed, summary_len * BITMAP_ULONG_BITS));
  }
#else
  // Suppress warnings
  (void)bm_compressed;
  (void)bm_compressed_size;
  (void)bmlen;
#endif
}

size_t compress_bitmap(const unsigned long* restrict const bm, uint64_t bmlen, unsigned long* restrict const compressed, int compression_level) {
  assert (bmlen % BITMAP_ULONG_BITS == 0);
  const size_t bmlen_in_words = (size_t)((bmlen + BITMAP_ULONG_BITS - 1) / BITMAP_ULONG_BITS);
  const size_t summary_len = (bmlen_in_words + BITMAP_ULONG_BITS - 1) / BITMAP_ULONG_BITS;
  assert (bmlen_in_words % BITMAP_ULONG_BITS == 0);
#ifdef DISABLE_COMPRESSION
  compression_level = 0;
#endif
  compression_loop:
  if (compression_level == 0) {
    memcpy(compressed, bm, bmlen_in_words * sizeof(unsigned long));
    return bmlen_in_words;
  } else {
    size_t nzwords = 0, popcount = 0;
    for (size_t i = 0; i < bmlen_in_words; ++i) {
      const unsigned long data = bm[i];
      if (data != 0) {
        ++nzwords;
        popcount += popcount_ul(data);
      }
    }
    if (compression_level >= 2 && nzwords == 0) { // Allow special casing of empty bitmaps
      return 0;
    }
    if (nzwords + summary_len >= bmlen_in_words) { // Not profitable to compress
      compression_level = 0;
      goto compression_loop;
    }
    if (compression_level >= 3 && popcount < summary_len) {
      // Copy elements directly
      size_t out_pos = 0;
      for (size_t i = 0; i < bmlen_in_words; ++i) {
        unsigned long data = bm[i];
        if (data == 0) continue;
        for (; data != 0; data &= data - 1) {
          compressed[out_pos++] = i * BITMAP_ULONG_BITS + cnttz_ul(data);
        }
      }
      assert (out_pos == popcount);
      return popcount;
    }
    // Otherwise, do compression level 1
    const unsigned long* bm_ptr = bm;
    unsigned long* restrict write_ptr = compressed + summary_len;
    for (size_t i = 0; i < bmlen_in_words / BITMAP_ULONG_BITS /* Rounding down */; ++i) {
      size_t offset = i * BITMAP_ULONG_BITS;
      unsigned long summary_data = 0UL;
      // Compiler is not smart enough to fully constant propagate this if I just use #pragma unroll
#ifndef __IBMC__
      for (int j = 0; j < BITMAP_ULONG_BITS; ++j) {
        unsigned long data = bm[offset + (j)];
        if (data != 0) {
          summary_data |= 1UL << (j);
          *write_ptr++ = data;
        }
      }
#else
#define JLOOP_ITER(j) \
      do { \
        unsigned long data = bm[offset + (j)]; \
        if (data != 0) { \
          summary_data |= 1UL << (j); \
          *write_ptr++ = data; \
        } \
      } while (0)
      JLOOP_ITER(0);
      JLOOP_ITER(1);
      JLOOP_ITER(2);
      JLOOP_ITER(3);
      JLOOP_ITER(4);
      JLOOP_ITER(5);
      JLOOP_ITER(6);
      JLOOP_ITER(7);
      JLOOP_ITER(8);
      JLOOP_ITER(9);
      JLOOP_ITER(10);
      JLOOP_ITER(11);
      JLOOP_ITER(12);
      JLOOP_ITER(13);
      JLOOP_ITER(14);
      JLOOP_ITER(15);
      JLOOP_ITER(16);
      JLOOP_ITER(17);
      JLOOP_ITER(18);
      JLOOP_ITER(19);
      JLOOP_ITER(20);
      JLOOP_ITER(21);
      JLOOP_ITER(22);
      JLOOP_ITER(23);
      JLOOP_ITER(24);
      JLOOP_ITER(25);
      JLOOP_ITER(26);
      JLOOP_ITER(27);
      JLOOP_ITER(28);
      JLOOP_ITER(29);
      JLOOP_ITER(30);
      JLOOP_ITER(31);
#ifndef __bgp__
      if (BITMAP_ULONG_BITS == 64) {
        JLOOP_ITER(32);
        JLOOP_ITER(33);
        JLOOP_ITER(34);
        JLOOP_ITER(35);
        JLOOP_ITER(36);
        JLOOP_ITER(37);
        JLOOP_ITER(38);
        JLOOP_ITER(39);
        JLOOP_ITER(40);
        JLOOP_ITER(41);
        JLOOP_ITER(42);
        JLOOP_ITER(43);
        JLOOP_ITER(44);
        JLOOP_ITER(45);
        JLOOP_ITER(46);
        JLOOP_ITER(47);
        JLOOP_ITER(48);
        JLOOP_ITER(49);
        JLOOP_ITER(50);
        JLOOP_ITER(51);
        JLOOP_ITER(52);
        JLOOP_ITER(53);
        JLOOP_ITER(54);
        JLOOP_ITER(55);
        JLOOP_ITER(56);
        JLOOP_ITER(57);
        JLOOP_ITER(58);
        JLOOP_ITER(59);
        JLOOP_ITER(60);
        JLOOP_ITER(61);
        JLOOP_ITER(62);
        JLOOP_ITER(63);
      }
#endif
#undef JLOOP_ITER
#endif
      compressed[i] = summary_data;
    }
    if (bmlen_in_words % BITMAP_ULONG_BITS != 0) {
      size_t i = (bmlen_in_words - bmlen_in_words % BITMAP_ULONG_BITS) / BITMAP_ULONG_BITS;
      size_t offset = i * BITMAP_ULONG_BITS;
      unsigned long summary_data = 0UL;
      for (unsigned int j = 0; j < bmlen_in_words % BITMAP_ULONG_BITS; ++j) {
        unsigned long data = bm[offset + j];
        if (data != 0) {
          summary_data |= 1UL << (j % BITMAP_ULONG_BITS); // rotatel_ul(1UL, ~j);
          assert (write_ptr <= compressed + bmlen_in_words);
          *write_ptr++ = data;
        }
      }
      compressed[i] = summary_data;
    }
    assert_compressed_bitmap_correct(compressed, (size_t)(write_ptr - compressed), bmlen);
    return (size_t)(write_ptr - compressed);
  }
}

static size_t compress_bitmap_adding_elements(const unsigned long* restrict const bm, uint64_t bmlen, unsigned long* restrict const compressed, int compression_level, const unsigned long* new_elements, size_t num_new_elements) {
  assert (bmlen % BITMAP_ULONG_BITS == 0);
  const size_t bmlen_in_words = (size_t)((bmlen + BITMAP_ULONG_BITS - 1) / BITMAP_ULONG_BITS);
  const size_t summary_len = (bmlen_in_words + BITMAP_ULONG_BITS - 1) / BITMAP_ULONG_BITS;
  assert (bmlen_in_words % BITMAP_ULONG_BITS == 0);
#ifdef DISABLE_COMPRESSION
  compression_level = 0;
#endif
  compression_loop:
  if (compression_level == 0) {
    memcpy(compressed, bm, bmlen_in_words * sizeof(unsigned long));
    for (size_t i = 0; i < num_new_elements; ++i) {
      unsigned long data = new_elements[i];
      compressed[data / BITMAP_ULONG_BITS] |= 1UL << (data % BITMAP_ULONG_BITS);
    }
    return bmlen_in_words;
  } else {
    size_t nzwords = 0, popcount = 0;
    size_t new_idx = 0;
    for (size_t i = 0; i < bmlen_in_words; ++i) {
      unsigned long data = bm[i];
      while (new_idx < num_new_elements && new_elements[new_idx] / BITMAP_ULONG_BITS < i) {++new_idx;}
      for (; new_idx < num_new_elements && new_elements[new_idx] / BITMAP_ULONG_BITS == i; ++new_idx) {
        data |= 1UL << (new_elements[new_idx] % BITMAP_ULONG_BITS);
      }
      if (data != 0) {
        ++nzwords;
        popcount += popcount_ul(data);
      }
    }
    if (compression_level >= 2 && nzwords == 0) { // Allow special casing of empty bitmaps
      memcpy(compressed, new_elements, num_new_elements * sizeof(unsigned long));
      return num_new_elements;
    }
    if (nzwords + summary_len >= bmlen_in_words) { // Not profitable to compress
      compression_level = 0;
      goto compression_loop;
    }
    if (compression_level >= 3 && popcount < summary_len) {
      // Copy elements directly
      size_t out_pos = 0;
      size_t new_idx = 0;
      for (size_t i = 0; i < bmlen_in_words; ++i) {
        unsigned long data = bm[i];
        while (new_idx < num_new_elements && new_elements[new_idx] / BITMAP_ULONG_BITS < i) {++new_idx;}
        for (; new_idx < num_new_elements && new_elements[new_idx] / BITMAP_ULONG_BITS == i; ++new_idx) {
          data |= 1UL << (new_elements[new_idx] % BITMAP_ULONG_BITS);
        }
        if (data == 0) continue;
        for (; data != 0; data &= data - 1) {
          compressed[out_pos++] = i * BITMAP_ULONG_BITS + cnttz_ul(data);
        }
      }
      assert (out_pos == popcount);
      return popcount;
    }
    // Otherwise, do compression level 1
    {
      const unsigned long* bm_ptr = bm;
      unsigned long* restrict write_ptr = compressed + summary_len;
      size_t new_idx = 0;
      for (size_t i = 0; i < bmlen_in_words / BITMAP_ULONG_BITS /* Rounding down */; ++i) {
        size_t offset = i * BITMAP_ULONG_BITS;
        unsigned long summary_data = 0UL;
        for (int j = 0; j < BITMAP_ULONG_BITS; ++j) {
          size_t full_offset = offset + j;
          unsigned long data = bm[full_offset];
          while (new_idx < num_new_elements && new_elements[new_idx] / BITMAP_ULONG_BITS < full_offset) {++new_idx;}
          for (; new_idx < num_new_elements && new_elements[new_idx] / BITMAP_ULONG_BITS == full_offset; ++new_idx) {
            data |= 1UL << (new_elements[new_idx] % BITMAP_ULONG_BITS);
          }
          if (data != 0) {
            summary_data |= 1UL << (j);
            *write_ptr++ = data;
          }
        }
        compressed[i] = summary_data;
      }
      if (bmlen_in_words % BITMAP_ULONG_BITS != 0) {
        size_t i = (bmlen_in_words - bmlen_in_words % BITMAP_ULONG_BITS) / BITMAP_ULONG_BITS;
        size_t offset = i * BITMAP_ULONG_BITS;
        unsigned long summary_data = 0UL;
        for (unsigned int j = 0; j < bmlen_in_words % BITMAP_ULONG_BITS; ++j) {
          size_t full_offset = offset + j;
          unsigned long data = bm[full_offset];
          while (new_idx < num_new_elements && new_elements[new_idx] / BITMAP_ULONG_BITS < full_offset) {++new_idx;}
          for (; new_idx < num_new_elements && new_elements[new_idx] / BITMAP_ULONG_BITS == full_offset; ++new_idx) {
            data |= 1UL << (new_elements[new_idx] % BITMAP_ULONG_BITS);
          }
          if (data != 0) {
            summary_data |= 1UL << (j % BITMAP_ULONG_BITS); // rotatel_ul(1UL, ~j);
            assert (write_ptr <= compressed + bmlen_in_words);
            *write_ptr++ = data;
          }
        }
        compressed[i] = summary_data;
      }
      assert_compressed_bitmap_correct(compressed, (size_t)(write_ptr - compressed), bmlen);
      return (size_t)(write_ptr - compressed);
    }
  }
}

void decompress_bitmap(const unsigned long* restrict const compressed, size_t compressed_size, unsigned long* restrict const bm, uint64_t bmlen) {
  assert (compressed != bm);
  assert_compressed_bitmap_correct(compressed, compressed_size, bmlen);
  assert (bmlen % BITMAP_ULONG_BITS == 0);
  size_t bmlen_in_words = (size_t)((bmlen + BITMAP_ULONG_BITS - 1) / BITMAP_ULONG_BITS);
  assert (bmlen_in_words % BITMAP_ULONG_BITS == 0);
  size_t summary_len = (bmlen_in_words + BITMAP_ULONG_BITS - 1) / BITMAP_ULONG_BITS;
  if (compressed_size == bmlen_in_words) {
    // Not compressed
    memcpy(bm, compressed, compressed_size * sizeof(unsigned long));
    return;
  } else if (compressed_size == 0) { // Empty
    memset(bm, 0, bmlen_in_words * sizeof(unsigned long));
    return;
  } else if (compressed_size < summary_len) { // List of distinct elements
    memset(bm, 0, bmlen_in_words * sizeof(unsigned long));
    for (size_t i = 0; i < compressed_size; ++i) {
      unsigned long data = compressed[i];
      bm[data / BITMAP_ULONG_BITS] |= 1UL << (data % BITMAP_ULONG_BITS);
    }
    return;
  }
  {
    const unsigned long* restrict read_ptr = compressed + summary_len;
    unsigned long* restrict write_ptr = bm;
    for (size_t i = 0; i < summary_len; ++i) {
      unsigned long summary_data = compressed[i];
      unsigned long read_data = (summary_data == 0 ? 0UL : *read_ptr);
      unsigned int jmax = BITMAP_ULONG_BITS;
      if ((i + 1) * BITMAP_ULONG_BITS > bmlen_in_words) jmax = bmlen_in_words - i * BITMAP_ULONG_BITS;
      if (jmax == BITMAP_ULONG_BITS) {
        for (unsigned int j = 0; j < BITMAP_ULONG_BITS; ++j) {
          // write_ptr[j] = ((signed long)rotatel_ul(summary_data, j) < 0) ? read_data : 0UL;
          write_ptr[j] = (summary_data & (1UL << (j % BITMAP_ULONG_BITS))) ? read_data : 0UL;
          if (summary_data & (1UL << (j % BITMAP_ULONG_BITS))) { // (signed long)rotatel_ul(summary_data, j) < 0) {
            read_data = *++read_ptr;
          }
        }
        write_ptr += BITMAP_ULONG_BITS;
        } else {
        for (unsigned int j = 0; j < jmax; ++j, ++write_ptr) {
          *write_ptr = (summary_data & (1UL << (j % BITMAP_ULONG_BITS))) ? read_data : 0UL;
          // *write_ptr = ((signed long)rotatel_ul(summary_data, j) < 0) ? read_data : 0UL;
          if (summary_data & (1UL << (j % BITMAP_ULONG_BITS))) { // (signed long)rotatel_ul(summary_data, j) < 0) {
            read_data = *++read_ptr;
          }
        }
      }
    }
    assert (read_ptr - compressed == compressed_size);
  }
}

size_t bitmap_or_compressed(unsigned long* restrict c_compressed,
                            const unsigned long* restrict a_compressed, size_t a_compressed_size, uint64_t alen,
                            const unsigned long* restrict b, uint64_t blen,
                            int compression_level) {
  assert (c_compressed != a_compressed);
  assert (c_compressed != b);
  assert_compressed_bitmap_correct(a_compressed, a_compressed_size, alen);
  assert (alen == blen); (void)blen;
  assert (alen % BITMAP_ULONG_BITS == 0);
  const size_t alen_in_words = (size_t)((alen + BITMAP_ULONG_BITS - 1) / BITMAP_ULONG_BITS);
  assert (alen_in_words % BITMAP_ULONG_BITS == 0);
  const size_t summary_len = (alen_in_words + BITMAP_ULONG_BITS - 1) / BITMAP_ULONG_BITS;
  assert (a_compressed_size == 0 || a_compressed_size >= summary_len);
  if (a_compressed_size == alen_in_words) {
    /* a is not actually compressed, so c shouldn't be either. */
    memcpy(c_compressed, a_compressed, alen_in_words * sizeof(unsigned long));
    bitmap_or(c_compressed, alen, b);
    return alen_in_words;
  } else if (a_compressed_size == 0) { // a is empty
    return compress_bitmap(b, blen, c_compressed, compression_level);
  } else if (a_compressed_size < summary_len) { // List of elements
    return compress_bitmap_adding_elements(b, blen, c_compressed, compression_level, a_compressed, a_compressed_size);
  }
  { // Extra scope to make xlc restrict rules happy
    unsigned long* restrict c_write_ptr = c_compressed + summary_len;
    const unsigned long* restrict a_read_ptr = a_compressed + summary_len;
    const unsigned long* restrict b_ptr = b;
    for (size_t i = 0; i < summary_len; ++i) {
      unsigned int jmax = BITMAP_ULONG_BITS;
      if ((i + 1) * BITMAP_ULONG_BITS > alen) jmax = (unsigned int)(alen - i * BITMAP_ULONG_BITS);
      const unsigned long a_summary_data = a_compressed[i];
      unsigned long c_summary_data = a_summary_data;
      if (EXPECT(c_write_ptr + jmax >= c_compressed + alen_in_words, 0)) {
#ifdef __IBMC__
#pragma execution_frequency (very_low)
#endif
        // Not profitable to compress result
        decompress_bitmap(a_compressed, a_compressed_size, c_compressed, alen);
        bitmap_or(c_compressed, alen, b);
        return alen_in_words;
      }
      for (unsigned int j = 0; j < jmax; ++j, ++b_ptr) {
        assert (b_ptr - b < alen_in_words);
        const unsigned long b_data = *b_ptr;
        if (a_summary_data & (1UL << (j % BITMAP_ULONG_BITS))) { // (signed long)rotatel_ul(a_summary_data, j) < 0) {
          const unsigned long a_data = *a_read_ptr++;
          *c_write_ptr++ = a_data | b_data;
        } else if (b_data != 0) {
          c_summary_data |= 1UL << (j % BITMAP_ULONG_BITS); // rotatel_ul(1UL, ~j);
          *c_write_ptr++ = b_data;
        } else {
          /* Nothing */
        }
        assert (c_write_ptr - c_compressed <= alen_in_words);
        assert (a_read_ptr - a_compressed <= a_compressed_size);
      }
      c_compressed[i] = c_summary_data;
    }
    assert (a_read_ptr - a_compressed == a_compressed_size);
    assert_compressed_bitmap_correct(c_compressed, (size_t)(c_write_ptr - c_compressed), alen);
    return (size_t)(c_write_ptr - c_compressed);
  }
}

void bitmap_andc_compressed(unsigned long* restrict b, uint64_t blen,
                            const unsigned long* restrict a_compressed, size_t a_compressed_size, uint64_t alen) {
  assert (b != a_compressed);
  assert_compressed_bitmap_correct(a_compressed, a_compressed_size, alen);
  assert (alen == blen); (void)blen;
  assert (alen % BITMAP_ULONG_BITS == 0);
  const size_t alen_in_words = (size_t)((alen + BITMAP_ULONG_BITS - 1) / BITMAP_ULONG_BITS);
  assert (alen_in_words % BITMAP_ULONG_BITS == 0);
  const size_t summary_len = (alen_in_words + BITMAP_ULONG_BITS - 1) / BITMAP_ULONG_BITS;
  if (a_compressed_size == alen_in_words) {
    /* a is not actually compressed. */
    bitmap_andc(b, blen, a_compressed);
    return;
  } else if (a_compressed_size == 0) { // a empty
    return;
  } else if (a_compressed_size < summary_len) { // list of elements
    for (size_t i = 0; i < a_compressed_size; ++i) {
      unsigned long data = a_compressed[i];
      b[data / BITMAP_ULONG_BITS] &= ~(1UL << (data % BITMAP_ULONG_BITS));
    }
    return;
  }
  { // Extra scope to make xlc restrict rules happy
    const unsigned long* restrict a_read_ptr = a_compressed + summary_len;
    unsigned long* restrict b_ptr = b;
    for (size_t i = 0; i < summary_len; ++i) {
      unsigned int jmax = BITMAP_ULONG_BITS;
      if ((i + 1) * BITMAP_ULONG_BITS > alen) jmax = (unsigned int)(alen - i * BITMAP_ULONG_BITS);
      const unsigned long a_summary_data = a_compressed[i];
      for (unsigned int j = 0; j < jmax; ++j, ++b_ptr) {
        assert (b_ptr - b < alen_in_words);
        const unsigned long b_data = *b_ptr;
        if (a_summary_data & (1UL << (j % BITMAP_ULONG_BITS))) { // (signed long)rotatel_ul(a_summary_data, j) < 0) {
          const unsigned long a_data = *a_read_ptr++;
          const unsigned long new_b_data = b_data & ~a_data;
          *b_ptr = new_b_data;
        }
        assert (a_read_ptr - a_compressed <= a_compressed_size);
      }
    }
    assert (a_read_ptr - a_compressed == a_compressed_size);
    return;
  }
}

void bitmap_andc_compressed_then_or(unsigned long* restrict c, uint64_t clen, const unsigned long* restrict b, uint64_t blen,
                                    const unsigned long* restrict a_compressed, size_t a_compressed_size, uint64_t alen) {
  assert (c != a_compressed);
  assert (c != b);
  assert_compressed_bitmap_correct(a_compressed, a_compressed_size, alen);
  assert (alen == blen); (void)blen;
  assert (alen == clen); (void)clen;
  assert (alen % BITMAP_ULONG_BITS == 0);
  const size_t alen_in_words = (size_t)((alen + BITMAP_ULONG_BITS - 1) / BITMAP_ULONG_BITS);
  assert (alen_in_words % BITMAP_ULONG_BITS == 0);
  const size_t summary_len = (alen_in_words + BITMAP_ULONG_BITS - 1) / BITMAP_ULONG_BITS;
  assert (a_compressed_size == 0 || a_compressed_size >= summary_len);
  if (a_compressed_size == alen_in_words) {
    /* a is not actually compressed. */
    bitmap_andc_then_or(c, clen, b, a_compressed);
    return;
  } else if (a_compressed_size == 0) { // a empty
    bitmap_or(c, clen, b);
    return;
  } else if (a_compressed_size < summary_len) { // list of elements
    size_t a_idx = 0;
    for (size_t i = 0; i < alen_in_words; ++i) {
      unsigned long temp = b[i];
      while (a_idx < a_compressed_size && a_compressed[a_idx] / BITMAP_ULONG_BITS < i) {++a_idx;}
      for (; a_idx < a_compressed_size && a_compressed[a_idx] / BITMAP_ULONG_BITS == i; ++a_idx) {
        temp &= ~(1UL << (a_compressed[a_idx] % BITMAP_ULONG_BITS));
      }
      c[i] |= b[i] & temp;
    }
    return;
  }
  { // Extra scope to make xlc restrict rules happy
    const unsigned long* restrict a_read_ptr = a_compressed + summary_len;
    const unsigned long* restrict b_ptr = b;
    unsigned long* restrict c_ptr = c;
    for (size_t i = 0; i < summary_len; ++i) {
      unsigned int jmax = BITMAP_ULONG_BITS;
      if ((i + 1) * BITMAP_ULONG_BITS > alen) jmax = (unsigned int)(alen - i * BITMAP_ULONG_BITS);
      const unsigned long a_summary_data = a_compressed[i];
      for (unsigned int j = 0; j < jmax; ++j, ++b_ptr, ++c_ptr) {
        assert (b_ptr - b < alen_in_words);
        assert (c_ptr - c < alen_in_words);
        const unsigned long b_data = *b_ptr;
        if (a_summary_data & (1UL << (j % BITMAP_ULONG_BITS))) {
          const unsigned long a_data = *a_read_ptr++;
          *c_ptr |= b_data & ~a_data;
        } else {
          *c_ptr |= b_data;
        }
        assert (a_read_ptr - a_compressed <= a_compressed_size);
      }
    }
    assert (a_read_ptr - a_compressed == a_compressed_size);
    return;
  }
}
