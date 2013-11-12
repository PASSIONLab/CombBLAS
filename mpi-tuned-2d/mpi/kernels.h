/* Copyright (C) 2010-2013 The Trustees of Indiana University.             */
/*                                                                         */
/* Use, modification and distribution is subject to the Boost Software     */
/* License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at */
/* http://www.boost.org/LICENSE_1_0.txt)                                   */
/*                                                                         */
/*  Authors: Jeremiah Willcock                                             */
/*           Andrew Lumsdaine                                              */

#ifndef KERNELS_H
#define KERNELS_H

/* These are particular kernels that are important to overall algorithm
 * performance, and thus are written in C to use the IBM compilers.  They can
 * also be switched to assembly later. */

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

void callgrind_start(void);
void callgrind_stop(void);

typedef struct addr_data_pair { // To use below and in bfs_custom.cpp
  uint32_t data;
  uint32_t addr;
} addr_data_pair;

/* All lengths in here are in bits. */

/* Compute b = A * x, where A is an unsymmetric matrix in coordinate form (high
 * 32-bit word is row, low word is column); mark the element in A that triggers
 * the first setting of an element in b in marks.  The b and marks bitmaps will
 * not be updated by any other threads; both must be accumulated into. */
void bitmap_matvec_and_mark_impl(const uint16_t* restrict const A_row_low_halfwords, const uint16_t* restrict const A_col_low_halfwords, size_t Alen, const uint8_t* restrict const edge_block_counts_by_block, const unsigned long* restrict x, uint64_t xlen, const uint8_t* restrict const xempty, unsigned long* restrict b, uint64_t blen, unsigned long* restrict marks);

/* Compute b = A^T * x, where A is an unsymmetric matrix in coordinate form
 * (high 32-bit word is row, low word is column); mark the element in A that
 * triggers the first setting of an element in b in marks.  The b and marks
 * bitmaps will not be updated by any other threads; both must be accumulated
 * into. */
void bitmap_matvec_trans_and_mark_impl(const uint16_t* restrict const A_row_low_halfwords, const uint16_t* restrict const A_col_low_halfwords, size_t Alen, const uint8_t* restrict const edge_block_counts_by_block, const unsigned long* restrict x, uint64_t xlen, const uint8_t* restrict const xempty, unsigned long* restrict b, uint64_t blen, unsigned long* restrict marks);

/* Copy data into all elements of an array where bit is set in bitmap. */
void write_levels_for_in_queue(int64_t* restrict const out, const int64_t data, const unsigned long* restrict const bm, const uint64_t bmlen);

/* Clear marks for edges whose targets are not set in oq */
void clear_marks_by_target(const uint64_t* restrict const edges, const size_t nedges, const unsigned long* restrict const oq, const uint64_t oqlen, unsigned long* restrict const marks);
void clear_marks_by_target_trans(const uint64_t* restrict const edges, const size_t nedges, const unsigned long* restrict const oq, const uint64_t oqlen, unsigned long* restrict const marks);

/* Compute b = b | a, where a and b are bitmaps. */
void bitmap_or(unsigned long* restrict b, uint64_t blen, const unsigned long* restrict a);

/* Compute a = b & ~c, where a, b, and c are bitmaps. */
void bitmap_andc_outofplace(unsigned long* restrict a, uint64_t alen, const unsigned long* restrict b, const unsigned long* restrict c);

/* Compute a = a & ~b, where a and b are bitmaps; return true iff a is nonzero
 * at the end. */
bool bitmap_andc(unsigned long* restrict a, uint64_t alen, const unsigned long* restrict b);

/* Compute c |= a & ~b, where a and b are bitmaps; return true iff a is nonzero
 * at the end. */
void bitmap_andc_then_or(unsigned long* restrict c, uint64_t clen, const unsigned long* restrict a, const unsigned long* restrict b);

/* Compute a = a | b, b = b & ~a (parallel assignment), where a and b are
 * bitmaps. */
void bitmap_or_andc(unsigned long* restrict a, uint64_t alen, unsigned long* restrict b);

/* Compute b = a & ~b, where a and b are bitmaps. */
void bitmap_andc_rev(unsigned long* restrict b, uint64_t blen, const unsigned long* restrict a);

/* Compute a = a & ~c, b = b & ~c, c = c | a | b on bitmaps; return true iff a
 * is nonzero at the end. */
bool bitmap_difference_and_or(unsigned long* restrict a, uint64_t alen, unsigned long* restrict b, unsigned long* restrict c);

/* Compute number of 1 bits in a range. */
size_t popcount_range(const unsigned long* restrict a, uint64_t alen);

/* Write data to scatter for the pred map.  Pseudocode is:
 * for i = 0 to Alen
 *   s = source(A[i]); t = target(A[i])
 *   if (marks[i] && out_queue[t])
 *     pos = insert_positions[owners[t >> owner_idx_shift]]++
 *     send_addr[pos] = local_biases[t >> owner_idx_shift] + t
 *     send_data[pos] = data_bias + s
 */
void write_pred_scatter_data(const uint64_t* restrict const A, size_t Alen, const unsigned long* restrict marks, const unsigned long* restrict out_queue, const int* restrict owners, int* restrict insert_positions, addr_data_pair* restrict send_data, int owner_idx_shift, const uint32_t* restrict local_biases, const uint32_t data_bias);

/* Write data to scatter for the pred map (transposed chunk).  Pseudocode is:
 * for i = 0 to Alen
 *   s = source(A[i]); t = target(A[i])
 *   if (marks_t[i] && out_queue[s])
 *     pos = insert_positions[owners[s >> owner_idx_shift]]++
 *     send_addr[pos] = local_biases[s >> owner_idx_shift] + s
 *     send_data[pos] = data_bias + t
 */
void write_pred_scatter_data_trans(const uint64_t* restrict const A, size_t Alen, const unsigned long* restrict marks_t, const unsigned long* restrict out_queue, const int* restrict owners, int* restrict insert_positions, addr_data_pair* restrict send_data, int owner_idx_shift, const uint32_t* restrict local_biases, const uint32_t data_bias);

/* Compress a bitmap, returning the final size.  The final size must be at most
 * the original size (the size itself can be used as metadata to tell whether
 * the buffer was actually compressed or just copied as is).  The input and
 * output buffers must be disjoint.
 */
size_t compress_bitmap(const unsigned long* restrict const bm, uint64_t bmlen, unsigned long* restrict const compressed, int compression_level);

/* Decompress a bitmap. */
void decompress_bitmap(const unsigned long* restrict const compressed, size_t compressed_size, unsigned long* restrict const bm, uint64_t bmlen);

/* Compute c = a | b, where a and b are bitmaps (a compressed, b not
 * compressed).  Return the compressed size of c.
 * */
size_t bitmap_or_compressed(unsigned long* restrict c_compressed,
                            const unsigned long* restrict a_compressed, size_t a_compressed_size, uint64_t alen,
                            const unsigned long* restrict b, uint64_t blen,
                            int compression_level);

/* Compute b = b & ~a, where a and b are bitmaps (a compressed, b not
 * compressed). */
void bitmap_andc_compressed(unsigned long* restrict b, uint64_t blen,
                            const unsigned long* restrict a_compressed, size_t a_compressed_size, uint64_t alen);

/* Compute c |= b & ~a, where a and b are bitmaps (a compressed, b and c not
 * compressed). */
void bitmap_andc_compressed_then_or(unsigned long* c, uint64_t clen, const unsigned long* restrict b, uint64_t blen,
                                    const unsigned long* restrict a_compressed, size_t a_compressed_size, uint64_t alen);

#ifdef __cplusplus
}
#endif

#endif /* KERNELS_H */
