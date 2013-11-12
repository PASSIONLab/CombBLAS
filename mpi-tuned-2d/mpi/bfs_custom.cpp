/* Copyright (C) 2010-2013 The Trustees of Indiana University.             */
/*                                                                         */
/* Use, modification and distribution is subject to the Boost Software     */
/* License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at */
/* http://www.boost.org/LICENSE_1_0.txt)                                   */
/*                                                                         */
/*  Authors: Jeremiah Willcock                                             */
/*           Andrew Lumsdaine                                              */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
#endif

#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif

// #define PROCESS_COUNT 512 -- in Makefile now

// #define KEEP_GRAPH_STATS

#include "common.hpp"
#include "mpi_workarounds.hpp"
#include "onesided.hpp"
#include "coroutine.hpp"
#include "kernels.h"
#include <mpi.h>
#ifdef __bgp__
#include <mpix.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#include <math.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <limits.h>
#include <cassert>
#include <algorithm>
#include <utility>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include "bitmap.hpp"
#if 0
extern "C" {
#include <mpt.h>
}
#endif
// #include "/u/jewillco/valgrind-Linux-install/include/valgrind/callgrind.h"

#if defined(CRAYPAT) && 0
#include <pat_api.h>
#endif

#ifdef _CRAYC
extern "C" {
#include <rca_lib.h>
}
#endif

#ifndef PROCESS_COUNT
#error "Define PROCESS_COUNT to your targeted number of processes"
#endif

// #define WITH_MPI_LOCK _Pragma("omp critical (mpi)")
#define WITH_MPI_LOCK /**/

static const unsigned int constant_size = PROCESS_COUNT;

template <unsigned int N> struct ct_log2_rounded_up {static const unsigned int value = ct_log2_rounded_up<(N + 1) / 2>::value + 1;};
template <> struct ct_log2_rounded_up<1> {static const unsigned int value = 0;};

template <unsigned int N> struct ct_log2_rounded_down {static const unsigned int value = ct_log2_rounded_down<N / 2>::value + 1;};
template <> struct ct_log2_rounded_down<1> {static const unsigned int value = 0;};

#ifdef __IBMCPP__
// Work around compiler bug by reducing error checking
template <unsigned int N> struct ct_log2: ct_log2_rounded_up<N> {};
#else
template <bool V> struct enable_if {};
template <> struct enable_if<true> {typedef void type;};
template <unsigned int N, typename Enable = void> struct ct_log2 {};
template <unsigned int N> struct ct_log2<N, typename enable_if<N != 0 && (N & (N - 1)) == 0>::type>: ct_log2_rounded_up<N> {};
#endif

template <unsigned int N, unsigned int LB = 1, unsigned int UB_minus_LB = (N + 1) / 2 - 1>
struct integer_sqrt_rounded_down {
  static const unsigned int middle = LB + UB_minus_LB / 2;
  static const int compare = (middle * middle < N) ? (((middle + 1) * (middle + 1) > N) ? 0 : 1) : -1;
  static const unsigned int new_lb = (compare <= 0) ? LB : middle;
  static const unsigned int new_ub = (compare >= 0) ? LB + UB_minus_LB : middle;
  static const unsigned int value =
    integer_sqrt_rounded_down<N, new_lb, new_ub - new_lb>::value;
  // static_assert (value * value <= N && (value + 1) * (value + 1) > N, "zzz");
};

template <unsigned int N, unsigned int B>
struct integer_sqrt_rounded_down<N, B, 0> {
  static const unsigned int value = B;
  // static_assert (value * value <= N && (value + 1) * (value + 1) > N, "zzz");
};

template <unsigned int N, unsigned int B>
struct integer_sqrt_rounded_down<N, B, 1> {
  static const unsigned int value = (B + 1) * (B + 1) >= N ? B + 1 : B;
  // static_assert (value * value <= N && (value + 1) * (value + 1) > N, "zzz");
};

template <unsigned int N>
struct integer_sqrt_rounded_up {
  static const unsigned int dn = integer_sqrt_rounded_down<N>::value;
  static const unsigned int value = (dn * dn == N) ? dn : dn + 1;
};

#ifdef SIZE_MUST_BE_A_POWER_OF_TWO
static const unsigned int const_lgsize = ct_log2<constant_size>::value;
static const unsigned int nprocbits = ((const_lgsize + 1) / 2);
static const unsigned int nprocbits_rounded_down = nprocbits;
static const unsigned int lg_triangle_side = nprocbits;
static const unsigned int triangle_side = (1 << nprocbits);
static const unsigned int half_triangle_side = (1 << (nprocbits - 1));
static const unsigned int lg_subsplit_side = (nprocbits >= 5 ? 5 : nprocbits); // BFS can only use nsubranks_1d * 2^lg_subsplit_side threads effectively; lg_subsplit_side must be <= nprocbits
static const unsigned int subsplit_side = (1 << lg_subsplit_side);

static const unsigned int nsubranks_1d = 2;
static const unsigned int nlocal_slots = 1 << (2 * lg_subsplit_side + 1);
#else
static const unsigned int triangle_side = integer_sqrt_rounded_down<2 * constant_size>::value;
// static const unsigned int lg_triangle_side_up = ct_log2_rounded_up<actual_triangle_side>::value;
// static const unsigned int rounded_triangle_side = 1U << lg_triangle_side_up;
// Make these non-static to avoid "unused variable" warnings on xlC
int check0[(triangle_side % 2 == 0) ? 1 : -1];
int check1[(triangle_side * triangle_side == 2 * constant_size) ? 1 : -1];
static const unsigned int half_triangle_side = triangle_side / 2;

static const unsigned int lg_triangle_side_down = ct_log2_rounded_down<triangle_side>::value;
static const unsigned int lg_triangle_side_up = ct_log2_rounded_up<triangle_side>::value;
static const unsigned int nprocbits_rounded_down = lg_triangle_side_down;
static const unsigned int triangle_side_down = 1u << lg_triangle_side_down;
static const unsigned int triangle_side_up = 1u << lg_triangle_side_up;

static const unsigned int lg_subsplit_side = (lg_triangle_side_down >= 5 ? 5 : lg_triangle_side_down); // BFS can only use nsubranks_1d * subsplit_side threads effectively; lg_subsplit_side must be <= nprocbits
static const unsigned int subsplit_side = 1u << lg_subsplit_side;

static const unsigned int nsubranks_1d = 2;
static const unsigned int nlocal_slots = 2 * subsplit_side * subsplit_side;
#endif

static inline int lg_rounded_up(uint64_t n) {
  int acc = 0;
  while (true) {
    if (n == 1) return acc;
    ++acc;
    n = (n + 1) / 2;
  }
  return -1; // Should not happen
}

struct vertex_bitmap {
  bitmap bm;
  uint64_t start_vertex;

  explicit vertex_bitmap(uint64_t start_vertex = 0): bm(0), start_vertex(start_vertex) {}
  void check(uint64_t expected_v) const {assert (start_vertex == expected_v); (void)expected_v;}
  void check(const vertex_bitmap& b) const {assert (start_vertex == b.start_vertex); (void)b;}
  void swap(vertex_bitmap& b) {
    if (&b == this) return;
    this->bm.swap(b.bm);
    std::swap(this->start_vertex, b.start_vertex);
  }

  private: vertex_bitmap(const vertex_bitmap&);
};

struct chunk_info {
  bool valid;
  size_t edge_slot;
  unsigned int in_bitmap_subrank, out_bitmap_subrank;
  uint32_t in_bitmap_offset, out_bitmap_offset;
  uint64_t base_src, base_tgt;
  enum {normal, transposed} edge_flip_status;
  bitmap total_edge_marks; // Accumulated over all levels
  bitmap total_edge_marks_transposed;

  chunk_info(): valid(false), total_edge_marks(0), total_edge_marks_transposed(0) {}
};

struct edge_compare {
  bool operator()(uint64_t e1, uint64_t e2) const {
    e1 = (e1 & UINT64_C(0xFFFF00000000FFFF)) | ((e1 >> 16) & UINT64_C(0x00000000FFFF0000)) | ((e1 & UINT64_C(0x00000000FFFF0000)) << 16);
    e2 = (e2 & UINT64_C(0xFFFF00000000FFFF)) | ((e2 >> 16) & UINT64_C(0x00000000FFFF0000)) | ((e2 & UINT64_C(0x00000000FFFF0000)) << 16);
    return e1 < e2;
  }
  typedef bool result_type;
  typedef uint64_t first_argument_type;
  typedef uint64_t second_argument_type;
};

template <typename RA, typename Cmp>
void my_sort(RA begin, RA end, const Cmp& cmp) {
  std::make_heap(begin, end, cmp);
  std::sort_heap(begin, end, cmp);
}

struct edge_range {
  uint64_t nrows, ncols;
  std::vector<uint64_t> data;
  /* Encoding for edge_block_counts_by_block:
   * There is a linear stream of bytes; 0-UINT8_MAX-1 represent themselves,
   * UINT8_MAX is an escape code that adds UINT8_MAX to this same decoding
   * starting at the next byte.  The size of the array may not be the same as
   * the number of elements because of this. */
  std::vector<uint8_t> edge_block_counts_by_block;
  std::vector<uint16_t> row_low_halfwords;
  std::vector<uint16_t> col_low_halfwords;

  edge_range(): nrows(0), ncols(0), data() {}
  void set_size(uint64_t nrows_, uint64_t ncols_) {nrows = nrows_; ncols = ncols_;}
  void clear() {data.clear();}
  void append(const uint64_t* range_begin, size_t len) {
    // fprintf(stderr, "%d: edge_range::append(%p, %zu) onto %zu old elements\n", rank, range_begin, len, data.size());
    if (len > 0) {
      data.insert(data.end(), range_begin, range_begin + len);
      // data.resize(data.size() + len);
      // fprintf(stderr, "%d: Copy from range %p len %zx * %zx to %p\n", rank, range_begin, len, sizeof(*range_begin), &*(data.end() - len));
      // std::copy(range_begin, range_begin + len, data.end() - len);
      my_sort(data.end() - len, data.end(), edge_compare());
      std::inplace_merge(data.begin(), data.end() - len, data.end(), edge_compare());
      // fprintf(stderr, "%d: Sort from range %p len %zx\n", rank, &*data.begin(), data.end() - data.begin());
      // std::sort(data.begin(), data.end(), edge_compare());
      // my_sort(data.begin(), data.end(), edge_compare());
    }
    // fprintf(stderr, "%d: edge_range::append(%p, %zu) after: %zu elements\n", rank, range_begin, len, data.size());
  }
  void finish() {
    data.erase(std::unique(data.begin(), data.end()), data.end());
#ifdef KEEP_GRAPH_STATS
    size_t block_distrib[1000];
    memset(block_distrib, 0, sizeof(block_distrib));
#endif
    size_t nrow_blocks = (size_t)((nrows + (1 << 16) - 1) >> 16); // This block size needs to match the one in edge_compare
    size_t ncol_blocks = (size_t)((ncols + (1 << 16) - 1) >> 16); // This block size needs to match the one in edge_compare
    // if (rank == 0) fprintf(stderr, "nrows %% 65536 = %u, ncols %% 65536 = %u\n", (unsigned int)(nrows % 65536), (unsigned int)(ncols % 65536));
    int avg_edges_per_block = (int)((data.size() + nrow_blocks * ncol_blocks - 1) / (nrow_blocks * ncol_blocks));
    size_t count_byte_idx = 0, count_byte_limit = nrow_blocks * ncol_blocks * (avg_edges_per_block >= 64 ? (avg_edges_per_block + 63) / 64 : 1) + nrow_blocks * ncol_blocks / 32 + 1024;
    edge_block_counts_by_block.clear();
    edge_block_counts_by_block.resize(count_byte_limit, 0);
    row_low_halfwords.clear();
    row_low_halfwords.resize(data.size());
    col_low_halfwords.clear();
    col_low_halfwords.resize(data.size());
    size_t last_blk_num = 0, last_blk_count = 0;
    for (size_t i = 0; i < data.size(); ++i) {
      uint64_t elt = data[i];
      uint32_t row = (uint32_t)(elt >> 32);
      uint32_t col = (uint32_t)elt;
      uint32_t row_blk = row >> 16;
      uint32_t col_blk = col >> 16; /* Shift done after cast to uint32_t */
      size_t blk_num = row_blk * ncol_blocks + col_blk;
      // fprintf(stderr, "%d: edge %zu: blk_num = %zu at index %zu with val %" PRIu8 "\n", rank, i, blk_num, count_byte_idx, edge_block_counts_by_block[count_byte_idx]);
      if (__builtin_expect(blk_num < last_blk_num, 0)) {
        fprintf(stderr, "%d: Blocks out of order: found %zu after %zu\n", rank, blk_num, last_blk_num);
        abort();
      }
      while (last_blk_num < blk_num) {
#ifdef KEEP_GRAPH_STATS
        ++block_distrib[(std::min)(last_blk_count, (size_t)255)];
        if (rank == 73 && last_blk_count >= 255) {fprintf(stderr, "Block %zu count: %zu\n", last_blk_num, last_blk_count);}
#endif
        ++last_blk_num;
        last_blk_count = 0;
        ++count_byte_idx;
        if (__builtin_expect(count_byte_idx >= count_byte_limit, 0)) goto count_array_full;
      }
      ++edge_block_counts_by_block[count_byte_idx];
      ++last_blk_count;
      if (edge_block_counts_by_block[count_byte_idx] == UINT8_MAX) {
        // if (rank == 73) fprintf(stderr, "%d: Block %zu over UINT8_MAX edges, avg_edges_per_block = %d\n", rank, blk_num, avg_edges_per_block);
        ++count_byte_idx;
        if (__builtin_expect(count_byte_idx >= count_byte_limit, 0)) goto count_array_full;
        edge_block_counts_by_block[count_byte_idx] = 0;
      }
      row_low_halfwords[i] = (uint16_t)row;
      col_low_halfwords[i] = (uint16_t)col;
      if (false) {
count_array_full:
        fprintf(stderr, "%d: Overflow of count array of size %zu at block %zu, starting with %zu edges in %zu blocks.\n", rank, count_byte_limit, blk_num, data.size(), nrow_blocks * ncol_blocks);
        abort();
      }
    }
#ifdef KEEP_GRAPH_STATS
    ++block_distrib[(std::min)(last_blk_count, (size_t)255)];
    if (rank == 73 && last_blk_count >= 255) {fprintf(stderr, "Block %zu count: %zu\n", last_blk_num, last_blk_count);}
    ++last_blk_num;
    while (last_blk_num < nrow_blocks * ncol_blocks) {++last_blk_num; ++block_distrib[0];}
    if (rank == 73) {
      ++count_byte_idx;
      fprintf(stderr, "%d: Finishing block loop with %zu of %zu byte(s) for %zu block(s)\n", rank, count_byte_idx, count_byte_limit, nrow_blocks * ncol_blocks);
      for (int i = 0; i < 256; ++i) fprintf(stderr, "Block size histogram: %d -> %zu\n", i, block_distrib[i]);
    }
#endif
  }
  bool empty() const {return data.empty();}
  size_t edge_count() const {return data.size();}
  size_t compressed_size() const {return 2 * this->edge_count();}
};

static MPI_Datatype addr_data_pair_datatype;

static struct graph_data {
  unsigned int lg_nvertices; /* Global count */
  // int64_t nedges; /* Global count */
  checked_array<nlocal_slots, edge_range> edge_ranges;
  /* These buffers are per element in nsubranks_1d */
  /* We only ever use two of these at a time, but which two changes over time */
  checked_array<nsubranks_1d, vertex_bitmap> in_queue;
  checked_array<nsubranks_1d, std::vector<uint8_t> > in_blocks_empty;
  checked_array<nsubranks_1d, vertex_bitmap> out_queue;
  checked_array<nsubranks_1d, vertex_bitmap> out_queue_last;
  checked_array<nsubranks_1d, vertex_bitmap> my_assigned_targets; // Vertices which I am responsible for sending pred writes for
  checked_array<nsubranks_1d, vertex_bitmap> visited;
  checked_array<nlocal_slots * 2, chunk_info> chunks_to_process;
  checked_array<nsubranks_1d, MPI_Comm> comm_per_row;
  checked_array<nsubranks_1d, unsigned int> rank_in_row;
} g;

static uint32_t owner_table[triangle_side][triangle_side]; // Subowner is x % nsubranks_1d, owner is x / nsubranks_1d
static uint32_t reverse_owner_table[constant_size][nsubranks_1d]; // High half is a, low half is b
static unsigned int my_coord_row_memberships[nsubranks_1d];
static MPI_Comm bfs_communicator;
static int bfs_rank;

static inline uint64_t nvertices(const unsigned int lg_nvertices) {
  return uint64_t(1) << lg_nvertices;
}

static inline uint64_t actual_nvertices_local_1d_per_subowner(const unsigned int lg_nvertices) {
  return (nvertices(lg_nvertices) + triangle_side * triangle_side - 1) / triangle_side / triangle_side;
}

static inline int compute_lg_nvertices_local_1d_per_subowner(const unsigned int lg_nvertices) {
  return lg_rounded_up(actual_nvertices_local_1d_per_subowner(lg_nvertices));
}

static inline size_t nvertices_local_1d_per_subowner(const unsigned int /* lg_nvertices */, const unsigned int lg_nvertices_local_1d_per_subowner) {
  return size_t(1) << lg_nvertices_local_1d_per_subowner;
}

static inline size_t actual_nvertices_local_1d_total(const unsigned int lg_nvertices, const unsigned int /*lg_nvertices_local_1d_per_subowner*/) {
  return size_t(nsubranks_1d) * size_t(actual_nvertices_local_1d_per_subowner(lg_nvertices));
}

static inline size_t nvertices_local_1d_total(const unsigned int lg_nvertices, const unsigned int lg_nvertices_local_1d_per_subowner) {
  return size_t(nsubranks_1d) * nvertices_local_1d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner);
}

static inline int lg_nvertices_local_2d_per_subowner(const unsigned int /*lg_nvertices*/, const unsigned int lg_nvertices_local_1d_per_subowner) {
  return lg_nvertices_local_1d_per_subowner + lg_triangle_side_up;
}

static inline uint32_t nvertices_local_2d_per_subowner(const unsigned int lg_nvertices, const unsigned int lg_nvertices_local_1d_per_subowner) {
  return uint32_t(1) << lg_nvertices_local_2d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner);
}

static inline uint64_t nvertices_local_2d_total(const unsigned int lg_nvertices, const unsigned int lg_nvertices_local_1d_per_subowner) {
  return uint64_t(nsubranks_1d) * nvertices_local_2d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner);
}

static inline uint32_t nvertices_per_subpart_2d(const unsigned int lg_nvertices, const unsigned int lg_nvertices_local_1d_per_subowner) {
  return uint32_t(1) << (lg_nvertices_local_2d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner) - lg_subsplit_side);
}

static inline int lg_nvertices_per_subpart_2d(const unsigned int lg_nvertices, const unsigned int lg_nvertices_local_1d_per_subowner) {
  return lg_nvertices_local_2d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner) - lg_subsplit_side;
}

static const int oned_parts_per_2d_subpart = triangle_side_up / subsplit_side;

uint32_t compute_coord_owner_info(const unsigned int a, const unsigned int b) {
  assert (a < triangle_side);
  assert (b < triangle_side);
  const unsigned int subowner = ((a > b) || (a == b && a >= triangle_side / 2));
  unsigned int aa = (std::min)(a, b);
  unsigned int bb = (std::max)(a, b);
  const unsigned int e = ((a >= triangle_side / 2) ^ (b >= triangle_side / 2));
  assert (e < 2);
  /* Note that these two conditions are opposites */
  if (aa >= triangle_side / 2) aa = triangle_side - 1 - aa;
  if (bb < triangle_side / 2) bb = triangle_side - 1 - bb;
  assert (aa < triangle_side / 2);
  assert (bb >= triangle_side / 2);
  bb -= triangle_side / 2;
  unsigned int owner;
  owner = bb * triangle_side + aa * 2 + e;
  assert (owner < constant_size);
  assert (subowner < nsubranks_1d);
  return owner * nsubranks_1d + subowner;
}

#ifdef __bgp__
uint32_t compute_coord_owner_info_bgp(const unsigned int a, const unsigned int b, int xlen, int ylen, int zlen, int tlen) {
  assert (a < triangle_side);
  assert (b < triangle_side);
  const unsigned int subowner = ((a > b) || (a == b && a >= triangle_side / 2));
  unsigned int aa = (std::min)(a, b);
  unsigned int bb = (std::max)(a, b);
  const unsigned int e = (a ^ b) / (triangle_side / 2);
  assert (e < 2);
  /* Note that these two conditions are opposites */
  if (aa >= triangle_side / 2) aa = triangle_side - 1 - aa;
  if (bb < triangle_side / 2) bb = triangle_side - 1 - bb;
  assert (aa < triangle_side / 2);
  assert (bb >= triangle_side / 2);
  bb -= triangle_side / 2;
  unsigned int owner;
  const unsigned int aa_upper = aa / ylen;
  const unsigned int bb_upper = bb / zlen;
#if 1
  const unsigned int aa_upper_encoded = e ? (triangle_side / 2 / ylen - 1 - aa_upper) : aa_upper;
  const unsigned int bb_upper_encoded = (aa_upper_encoded & 1) ? (triangle_side / 2 / zlen - 1 - bb_upper) : bb_upper;
  // assert (aa_upper < upper_field_limit);
  // assert (bb_upper < upper_field_limit);
  // Encode aa_upper and bb_upper in Gray code
  const unsigned int bg_x = e * ((triangle_side / 2 / ylen) * (triangle_side / 2 / zlen) / tlen)
                          + aa_upper_encoded * ((triangle_side / 2 / zlen) / tlen)
                          + bb_upper_encoded / tlen;
  const unsigned int bg_t = bb_upper_encoded % tlen;
  assert (bg_x < (unsigned int)xlen); (void)(xlen);
  assert (bg_t < (unsigned int)tlen);
  // Flip every other row in y and z to localize communication more
  const unsigned int bg_y = (aa_upper % 2) ? (ylen - 1 - aa % ylen) : (aa % ylen);
  assert (bg_y < (unsigned int)ylen);
  const unsigned int bg_z = (bb_upper % 2) ? (zlen - 1 - bb % zlen) : (bb % zlen);
  assert (bg_z < (unsigned int)zlen);
#else
  unsigned int bg_x = e * ((triangle_side / 2 / ylen) * (triangle_side / 2 / zlen) / tlen)
                    + aa_upper * ((triangle_side / 2 / zlen) / tlen)
                    + bb_upper / tlen;
  unsigned int bg_t = bb_upper % tlen;
  assert (bg_x < (unsigned int)xlen); (void)(xlen);
  assert (bg_t < (unsigned int)tlen);
  // Flip every other row in y and z to localize communication more
  unsigned int bg_y = aa % ylen;
  assert (bg_y < (unsigned int)ylen);
  unsigned int bg_z = bb % zlen;
  assert (bg_z < (unsigned int)zlen);
  // xor embedding from "Embedding hypercubes into rings and toruses" by
  // Gonzalez, Valero-Garcia, and Diaz de Cerio
  bg_x ^= ((bg_x & (xlen / 2)) >> 1);
  bg_y ^= ((bg_y & (ylen / 2)) >> 1);
  bg_z ^= ((bg_z & (zlen / 2)) >> 1);
  bg_t ^= ((bg_t & (tlen / 2)) >> 1);
#endif
  int coords[4] = {int(bg_t), int(bg_z), int(bg_y), int(bg_x)};
  MPI_Cart_rank(bfs_communicator, coords, (int*)&owner);
  assert (owner < constant_size);
  assert (subowner < nsubranks_1d);
  // if (rank == 0) fprintf(stderr, "(%u, %u) -> (x=%u, y=%u, z=%u, t=%u) -> %u sub %u\n", a, b, bg_x, bg_y, bg_z, bg_t, owner, subowner);
  return owner * nsubranks_1d + subowner;
}
#endif

void setup_owner_table() {
  for (unsigned int i = 0; i < constant_size; ++i) {
    for (unsigned int j = 0; j < nsubranks_1d; ++j) {
      reverse_owner_table[i][j] = (uint32_t)(-1);
    }
  }
  for (unsigned int i = 0; i < triangle_side; ++i) {
    for (unsigned int j = 0; j < triangle_side; ++j) {
      uint32_t x = compute_coord_owner_info(i, j);
      owner_table[i][j] = x;
      reverse_owner_table[x / nsubranks_1d][x % nsubranks_1d] = (i << 16) | j;
    }
  }
  my_coord_row_memberships[0] = reverse_owner_table[bfs_rank][0] >> 16;
  my_coord_row_memberships[1] = reverse_owner_table[bfs_rank][1] >> 16;
  assert (my_coord_row_memberships[0] < triangle_side);
  assert (my_coord_row_memberships[1] < triangle_side);
  if (my_coord_row_memberships[0] == my_coord_row_memberships[1]) {
    my_coord_row_memberships[1] = triangle_side - 1 - my_coord_row_memberships[0];
  }
}

#ifdef __bgp__
void setup_owner_table_bgp() {
  int ndims;
  MPI_Cartdim_get(bfs_communicator, &ndims);
  if (ndims != 4) {
    if (rank == 0) {fprintf(stderr, "BG/P must have 4 dimensions.\n");}
    abort();
  }
  int dims[4], periods[4], coords[4];
  MPI_Cart_get(bfs_communicator, 4, dims, periods, coords);
  unsigned int xlen = (unsigned int)dims[3];
  unsigned int ylen = (unsigned int)dims[2];
  unsigned int zlen = (unsigned int)dims[1];
  unsigned int tlen = (unsigned int)dims[0];
  // if (rank == 0) fprintf(stderr, "xlen=%u, ylen=%u, zlen=%u, tlen=%u\n", xlen, ylen, zlen, tlen);
  for (unsigned int i = 0; i < triangle_side; ++i) {
    for (unsigned int j = 0; j < triangle_side; ++j) {
      uint32_t x = compute_coord_owner_info_bgp(i, j, xlen, ylen, zlen, tlen);
      owner_table[i][j] = x;
      reverse_owner_table[x / nsubranks_1d][x % nsubranks_1d] = (i << 16) | j;
    }
  }
  my_coord_row_memberships[0] = reverse_owner_table[bfs_rank][0] >> 16;
  my_coord_row_memberships[1] = reverse_owner_table[bfs_rank][1] >> 16;
  // fprintf(stderr, "%d: my_coord_row_memberships = {%u, %u}\n", bfs_rank, my_coord_row_memberships[0], my_coord_row_memberships[1]);
}
#endif

inline void coord_owner_info(const unsigned int a, const unsigned int b, unsigned int& owner, unsigned int& subowner) {
  assert (a < triangle_side);
  assert (b < triangle_side);
  uint32_t val = owner_table[a][b];
  assert (val < nsubranks_1d * constant_size);
  owner = val / nsubranks_1d;
  subowner = val % nsubranks_1d;
}

inline void coord_to_global(const unsigned int o, const unsigned int so, unsigned int& a, unsigned int& b) {
  assert (o < constant_size);
  assert (so < nsubranks_1d);
  uint32_t x = reverse_owner_table[o][so];
  assert (x != (uint32_t)(-1));
  a = (x >> 16);
  b = (x & ((1 << 16) - 1));
  assert (a < triangle_side);
  assert (b < triangle_side);
}

inline unsigned int coord_row_memberships(const unsigned int index /* 0 ... nsubranks_1d - 1 */ ) {
  assert (index < nsubranks_1d);
  return my_coord_row_memberships[index];
}

#ifdef SIZE_MUST_BE_A_POWER_OF_TWO
static inline uint64_t encode_vertex(const uint64_t v, const unsigned int /*lg_nvertices*/) {return v;}
static inline uint64_t decode_vertex(const uint64_t v, const unsigned int /*lg_nvertices*/) {return v;}
#else
static inline uint64_t encode_vertex(const uint64_t v, const unsigned int lg_nvertices) {
  const uint64_t actual_verts_per_side = actual_nvertices_local_1d_per_subowner(lg_nvertices);
  const unsigned int lg_nvertices_local_1d_per_subowner = compute_lg_nvertices_local_1d_per_subowner(lg_nvertices);
  const uint64_t v_low = v % actual_verts_per_side;
  const uint64_t v_high = v / actual_verts_per_side;
  return (((v_high / triangle_side) << lg_triangle_side_up) + (v_high % triangle_side)) * nvertices_local_1d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner) + v_low;
}
static inline uint64_t decode_vertex(const uint64_t v, const unsigned int lg_nvertices) {
  const uint64_t actual_verts_per_side = actual_nvertices_local_1d_per_subowner(lg_nvertices);
  const unsigned int lg_nvertices_local_1d_per_subowner = compute_lg_nvertices_local_1d_per_subowner(lg_nvertices);
  const uint64_t v_low = v % nvertices_local_1d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner);
  const uint64_t v_high = v / nvertices_local_1d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner);
  return (((v_high >> lg_triangle_side_up) * triangle_side) + (v_high % (1 << lg_triangle_side_up))) * actual_verts_per_side + v_low;
}
#endif

inline void split_vertex_number_1d(const uint64_t v, unsigned int& owner_high, unsigned int& owner_low, size_t& local_raw, const unsigned int lg_nvertices, const unsigned int lg_nvertices_local_1d_per_subowner) {
  assert (decode_vertex(v, lg_nvertices) < nvertices(lg_nvertices));
  const unsigned int h = (unsigned int)(v >> lg_nvertices_local_1d_per_subowner);
  owner_high = h / triangle_side_up;
  owner_low = h % triangle_side_up;
  local_raw = (size_t(v) & (nvertices_local_1d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner) - 1));
  assert (local_raw == (size_t)(v & (nvertices_local_1d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner) - 1)));
  assert (local_raw < nvertices_local_1d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner));
}

static inline uint64_t build_vertex_number_1d(const unsigned int owner_high, const unsigned int owner_low, const size_t local_raw, const unsigned int lg_nvertices, const unsigned int lg_nvertices_local_1d_per_subowner) {
  (void)lg_nvertices;
  (void)lg_nvertices_local_1d_per_subowner;
  assert (owner_high < triangle_side);
  assert (owner_low < triangle_side);
  assert (local_raw < nvertices_local_1d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner));
  const uint64_t v = (uint64_t(owner_high * triangle_side_up + owner_low) << lg_nvertices_local_1d_per_subowner) + local_raw;
#ifndef NDEBUG
  unsigned int owner_high_check, owner_low_check;
  size_t local_raw_check;
  split_vertex_number_1d(v, owner_high_check, owner_low_check, local_raw_check, lg_nvertices, lg_nvertices_local_1d_per_subowner);
  assert (owner_high_check == owner_high);
  assert (owner_low_check == owner_low);
  assert (local_raw_check == local_raw);
  assert (decode_vertex(v, lg_nvertices) < nvertices(lg_nvertices));
#endif
  return v;
}

static inline void vertex_owner_and_local_1d(const uint64_t v, const unsigned int lg_nvertices, const unsigned int lg_nvertices_local_1d_per_subowner, unsigned int& owner, size_t& local) {
  unsigned int owner_high, owner_low, subowner;
  size_t local_raw;
  split_vertex_number_1d(v, owner_high, owner_low, local_raw, lg_nvertices, lg_nvertices_local_1d_per_subowner);
  coord_owner_info(owner_high, owner_low, owner, subowner);
  assert (owner < constant_size);
  local = local_raw + (size_t(subowner) << lg_nvertices_local_1d_per_subowner);
}

static inline uint64_t vertex_to_global_1d(const unsigned int owner, const size_t local, const unsigned int lg_nvertices, const unsigned int lg_nvertices_local_1d_per_subowner) {
  assert (owner < constant_size);
  assert (local < (nsubranks_1d << lg_nvertices_local_1d_per_subowner));
  const unsigned int so = (unsigned int)(local >> lg_nvertices_local_1d_per_subowner);
  unsigned int a, b;
  coord_to_global(owner, so, a, b);
  const uint64_t v = build_vertex_number_1d(a, b, (size_t)(local & (nvertices_local_1d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner) - 1)), lg_nvertices, lg_nvertices_local_1d_per_subowner);
#ifndef NDEBUG
  unsigned int owner_check;
  size_t local_check;
  vertex_owner_and_local_1d(v, lg_nvertices, lg_nvertices_local_1d_per_subowner, owner_check, local_check);
  assert (owner_check == owner);
  assert (local_check == local);
#endif
  return v;
}

static inline void split_global_vertex_number_2d(const uint64_t v, unsigned int& owner_part, uint32_t& local_part, const unsigned int lg_nvertices, const unsigned int lg_nvertices_local_1d_per_subowner) {
  assert (decode_vertex(v, lg_nvertices) < nvertices(lg_nvertices));
  owner_part = (unsigned int)(v >> lg_nvertices_local_2d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner));
  assert (owner_part < triangle_side);
  local_part = uint32_t(v & (nvertices_local_2d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner) - 1));
}

static inline uint64_t build_global_vertex_number_2d(const unsigned int owner_part, const uint32_t local_part, const unsigned int lg_nvertices, const unsigned int lg_nvertices_local_1d_per_subowner) {
  assert (owner_part < triangle_side);
  assert (local_part < nvertices_local_2d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner));
  uint64_t v = (uint64_t(owner_part) << lg_nvertices_local_2d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner)) + local_part;
  assert (decode_vertex(v, lg_nvertices) < nvertices(lg_nvertices));
  return v;
}

static inline void edge_owner_and_slot(const uint64_t a, const uint64_t b, const unsigned int lg_nvertices, const unsigned int lg_nvertices_local_1d_per_subowner, unsigned int& owner, unsigned int& slot) {
#ifndef NDEBUG
  // fprintf(stderr, "a=%" PRIx64 " decode(a)=%" PRIx64 "\n", a, decode_vertex(a, lg_nvertices));
  // fprintf(stderr, "b=%" PRIx64 " decode(b)=%" PRIx64 "\n", b, decode_vertex(b, lg_nvertices));
  assert ((decode_vertex(a, lg_nvertices) >> lg_nvertices) == 0);
  assert ((decode_vertex(b, lg_nvertices) >> lg_nvertices) == 0);
#endif
  // Order of endpoints doesn't matter here
  unsigned int owner_part_a = (unsigned int)(a >> lg_nvertices_local_2d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner));
  unsigned int owner_part_b = (unsigned int)(b >> lg_nvertices_local_2d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner));
  uint64_t local_part_a = (a & (nvertices_local_2d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner) - 1));
  uint64_t local_part_b = (b & (nvertices_local_2d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner) - 1));
  unsigned int split_num_a = (unsigned int)(local_part_a >> lg_nvertices_per_subpart_2d(lg_nvertices, lg_nvertices_local_1d_per_subowner));
  unsigned int split_num_b = (unsigned int)(local_part_b >> lg_nvertices_per_subpart_2d(lg_nvertices, lg_nvertices_local_1d_per_subowner));
  unsigned int subowner;
  coord_owner_info(owner_part_a, owner_part_b, owner, subowner);
  assert (owner < constant_size);
  slot =
    (subowner << (2 * lg_subsplit_side)) |
    (split_num_a << lg_subsplit_side) |
    split_num_b;
  assert (slot < nlocal_slots);
}

static inline uint64_t make_edge_local(const uint32_t a_local, const uint32_t b_local) {
  return (uint64_t(a_local) << 32) | b_local;
}

static inline uint64_t get_pred(uint64_t l_and_p) {
  const uint64_t mask = UINT64_C(0x0000ffffffffffff);
  if ((~l_and_p) == 0) {
    return UINT64_MAX;
  } else {
    uint64_t pr = (l_and_p & mask);
    assert (decode_vertex(pr, g.lg_nvertices) < nvertices(g.lg_nvertices));
    return pr;
  }
}

static inline uint16_t level(uint64_t l_and_p) {
  return (uint16_t)(l_and_p >> 48);
}

static inline uint64_t make_lp(uint16_t lev, uint64_t pre) {
  // Requires un-encoded vertex in pre
  assert ((lev == UINT16_MAX && pre == UINT64_MAX) ||
          (lev != UINT16_MAX && pre < nvertices(g.lg_nvertices)));
  static const uint64_t mask = UINT64_C(0x0000ffffffffffff);
  uint64_t lp = (uint64_t)(((uint64_t)lev << 48) | (pre & mask));
  if (lev == UINT16_MAX && pre == UINT64_MAX) {assert (lp == UINT64_MAX);}
  assert (get_pred(lp) == pre);
  assert (level(lp) == lev);
  return lp;
}

extern "C" int PMI_Cart_comm_create(MPI_Comm*);

// static void print_node_topology();

void make_graph_data_structure(const tuple_graph* const tg) {
  if ((unsigned int)size != constant_size) {
    if (bfs_rank == 0) fprintf(stderr, "Number of ranks %d does not match compile-time-set value of %u.\n", size, constant_size);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

#ifdef __bgp__
  {
    int err = PMI_Cart_comm_create(&bfs_communicator);
    if (err != MPI_SUCCESS) {
      if (bfs_rank == 0) fprintf(stderr, "Topology mapping failed\n");
      MPI_Comm_dup(MPI_COMM_WORLD, &bfs_communicator);
      MPI_Comm_rank(bfs_communicator, &bfs_rank);
      setup_owner_table();
    } else {
      MPI_Comm_rank(bfs_communicator, &bfs_rank);
      setup_owner_table_bgp();
    }
  }
#else
  MPI_Comm_dup(MPI_COMM_WORLD, &bfs_communicator);
  MPI_Comm_rank(bfs_communicator, &bfs_rank);
  setup_owner_table();
#endif

  {
    int blocklengths[3] = {1, 1, 1};
    MPI_Aint displs[3] = {offsetof(addr_data_pair, data), offsetof(addr_data_pair, addr), sizeof(addr_data_pair)};
    MPI_Datatype types[3] = {MPI_UINT32_T, MPI_UINT32_T, MPI_UB};
    MPI_Type_create_struct(3, blocklengths, displs, types, &addr_data_pair_datatype);
    MPI_Type_commit(&addr_data_pair_datatype);
  }

  // print_node_topology();

  scoped_array<int> counts;
  counts.reset_to_new(constant_size);

  scoped_array<int> count_totals, incoming_counts, incoming_displs;
  count_totals.reset_to_new(constant_size);
  incoming_counts.reset_to_new(constant_size);
  incoming_displs.reset_to_new(constant_size + 1);
  scoped_array<uint64_t> write_buffer;
  scoped_array<size_t> write_offsets;
  write_offsets.reset_to_new(constant_size);
  scoped_array<int> write_displs;
  write_displs.reset_to_new(constant_size + 1);

  {
    uint64_t vertex_mask = 0;
    ITERATE_TUPLE_GRAPH_BEGIN(tg, buf, bufsize) {
      if (bfs_rank == 0) fprintf(stderr, "Getting vertex count block %zu of %zu\n", size_t(ITERATE_TUPLE_GRAPH_BLOCK_NUMBER), size_t(ITERATE_TUPLE_GRAPH_BLOCK_COUNT(tg)));
#pragma omp parallel for reduction(|:vertex_mask)
      for (int i = 0; i < bufsize; ++i) {
        uint64_t v0 = (uint64_t)get_v0_from_edge(&buf[i]);
        uint64_t v1 = (uint64_t)get_v1_from_edge(&buf[i]);
        if (v0 == v1) continue;
        vertex_mask |= (v0 | v1);
      }
    } ITERATE_TUPLE_GRAPH_END;
    MPI_Allreduce(MPI_IN_PLACE, &vertex_mask, 1, MPI_UINT64_T, MPI_BOR, bfs_communicator);
#ifdef __IBMCPP__
    const int mask_leading_zeros = __cntlz8(vertex_mask);
#else
    const int mask_leading_zeros = __builtin_clzll(vertex_mask);
#endif
    unsigned int lg_nvertices = (vertex_mask == 0) ? 0 : (unsigned int)(sizeof(unsigned long long) * CHAR_BIT - mask_leading_zeros); /* Rounds up */
    if (lg_nvertices > 32 + nprocbits_rounded_down) {
      if (bfs_rank == 0) {
        fprintf(stderr, "Vertex count is too large for this processor count: found vertex count 2**%u, limit is 2**%u\n",
                lg_nvertices, 32 + nprocbits_rounded_down);
      }
      MPI_Abort(bfs_communicator, 2);
    }
    g.lg_nvertices = lg_nvertices; // Used in edge_owner
  }

  const unsigned int lg_nvertices = g.lg_nvertices;
  const unsigned int lg_nvertices_local_1d_per_subowner = compute_lg_nvertices_local_1d_per_subowner(lg_nvertices);
  uint64_t total_local_edges = 0;
  const uint32_t subpart_2d_mask = nvertices_per_subpart_2d(lg_nvertices, lg_nvertices_local_1d_per_subowner) - 1;

  for (int i = 0; i < int(nlocal_slots); ++i) {
    edge_range& this_edge_range = g.edge_ranges[i];
    this_edge_range.set_size(nvertices_per_subpart_2d(lg_nvertices, lg_nvertices_local_1d_per_subowner), nvertices_per_subpart_2d(lg_nvertices, lg_nvertices_local_1d_per_subowner));
  }
  ITERATE_DESTRUCTIVE_TUPLE_GRAPH_BEGIN(tg, buf, bufsize) {
    for (unsigned int i = 0; i < constant_size; ++i) counts[i] = 0;
#pragma omp parallel for
    for (int i = 0; i < bufsize; ++i) {
      uint64_t v0_raw = (uint64_t)get_v0_from_edge(&buf[i]);
      uint64_t v1_raw = (uint64_t)get_v1_from_edge(&buf[i]);
      if (v0_raw == v1_raw) continue;
      assert (v0_raw < nvertices(lg_nvertices));
      assert (v1_raw < nvertices(lg_nvertices));
      unsigned int owner, slot;
      edge_owner_and_slot(encode_vertex(v0_raw, lg_nvertices), encode_vertex(v1_raw, lg_nvertices), lg_nvertices, lg_nvertices_local_1d_per_subowner, owner, slot);
#pragma omp atomic
      ++counts[owner];
    }
    /* Put this everywhere to avoid needing to do a scan and various other
     * collectives. */
    MPI_Allreduce(counts.get(0, constant_size), count_totals.get(0, constant_size), constant_size, MPI_INT, MPI_SUM, bfs_communicator);
    MPI_Alltoall(counts.get(0, constant_size), 1, MPI_INT, incoming_counts.get(0, constant_size), 1, MPI_INT, bfs_communicator);
    incoming_displs[0] = 0;
    for (unsigned int i = 0; i < constant_size; ++i) {
      incoming_displs[i + 1] = incoming_displs[i] + incoming_counts[i];
    }
    write_displs[0] = 0;
    for (unsigned int i = 0; i < constant_size; ++i) {
      write_displs[i + 1] = write_displs[i] + counts[i];
      assert (write_displs[i + 1] >= write_displs[i]);
    }
    write_buffer.reset_to_new(write_displs[constant_size]);
    for (unsigned int i = 0; i < constant_size; ++i) {
      write_offsets[i] = write_displs[i];
    }
    if (bfs_rank == 0) fprintf(stderr, "Distributing graph block %zu of %zu\n", size_t(ITERATE_DESTRUCTIVE_TUPLE_GRAPH_BLOCK_NUMBER), size_t(ITERATE_DESTRUCTIVE_TUPLE_GRAPH_BLOCK_COUNT(tg)));
    // if (bfs_rank == 0) fprintf(stderr, "%d before writing to send buffer of size %zu\n", bfs_rank, (size_t)write_displs[constant_size]);
#pragma omp parallel for
    for (int i = 0; i < bufsize; ++i) {
      uint64_t v0_raw = (uint64_t)get_v0_from_edge(&buf[i]);
      uint64_t v1_raw = (uint64_t)get_v1_from_edge(&buf[i]);
      if (v0_raw == v1_raw) continue;
      if (v0_raw > v1_raw) std::swap(v0_raw, v1_raw);
      uint64_t v0 = encode_vertex(v0_raw, lg_nvertices);
      uint64_t v1 = encode_vertex(v1_raw, lg_nvertices);
      unsigned int owner, slot;
      edge_owner_and_slot(v0, v1, lg_nvertices, lg_nvertices_local_1d_per_subowner, owner, slot);
      uint64_t local = uint64_t(v0 & subpart_2d_mask) * nvertices_per_subpart_2d(lg_nvertices, lg_nvertices_local_1d_per_subowner) + uint32_t(v1 & subpart_2d_mask);
      assert (owner < constant_size);
#ifdef __IBMCPP__
#ifdef __powerpc64__
      size_t offs = __fetch_and_addlp((volatile long long*)&write_offsets[owner], 1);
#else
      size_t offs = __fetch_and_add((volatile int*)&write_offsets[owner], 1);
#endif
#else
      size_t offs = __sync_fetch_and_add(&write_offsets[owner], 1);
#endif
      assert (offs < (size_t)write_displs[owner + 1]);
      write_buffer[offs] = local + uint64_t(slot) * nvertices_per_subpart_2d(lg_nvertices, lg_nvertices_local_1d_per_subowner) * nvertices_per_subpart_2d(lg_nvertices, lg_nvertices_local_1d_per_subowner);
    }
    ITERATE_DESTRUCTIVE_TUPLE_GRAPH_RELEASE_BUFFER;
    // if (bfs_rank == 0) fprintf(stderr, "%d after writing to send buffer\n", bfs_rank);
    assert (incoming_displs[constant_size] == count_totals[bfs_rank]);
    scoped_array<uint64_t> incoming_data;
    incoming_data.reset_to_new(incoming_displs[constant_size]);
    // if (bfs_rank == 0) fprintf(stderr, "%d before data transfer of %d out, %d in\n", bfs_rank, write_displs[constant_size], incoming_displs[constant_size]);
    MPI_Alltoallv(write_buffer.get(0, write_displs[constant_size]), counts.get(0, constant_size), write_displs.get(0, constant_size), MPI_UINT64_T,
                  incoming_data.get(0, incoming_displs[constant_size]), incoming_counts.get(0, constant_size), incoming_displs.get(0, constant_size), MPI_UINT64_T,
                  bfs_communicator);
    // if (bfs_rank == 0) fprintf(stderr, "%d after data transfer\n", bfs_rank);
    write_buffer.reset();
    double sortstart = MPI_Wtime(); (void)sortstart;
    // This sort must be in numerical order so that lower_bound() below will
    // work; sorting for cache locality is done in edge_range::append. */
    std::sort(incoming_data.get(0, incoming_displs[constant_size]), incoming_data.get(incoming_displs[constant_size], 0));
    double sortend = MPI_Wtime(); (void)sortend;
    // if (bfs_rank == 0) fprintf(stderr, "%d: sort took %f\n", bfs_rank, sortend - sortstart);
    checked_array<nlocal_slots + 1, size_t> slot_start;
    int recv_total = incoming_displs[constant_size];
    slot_start[nlocal_slots] = recv_total;
    for (unsigned int i = 0; i < nlocal_slots; ++i) {
      slot_start[i] = std::lower_bound(incoming_data.get(0, incoming_displs[constant_size]), incoming_data.get(incoming_displs[constant_size], 0), (uint64_t(i) << (2 * lg_nvertices_per_subpart_2d(lg_nvertices, lg_nvertices_local_1d_per_subowner)))) - incoming_data.get(0, 0);
      assert (slot_start[i] <= (size_t)recv_total);
      if (i > 0) assert (slot_start[i] >= slot_start[i - 1]);
    }
    assert (slot_start[nlocal_slots] >= slot_start[nlocal_slots - 1]);
#ifndef NDEBUG
    for (unsigned int i = 0; i < nlocal_slots; ++i) {
      for (size_t j = slot_start[i]; j < slot_start[i + 1]; ++j) {
        assert ((incoming_data[j] >> (2 * lg_nvertices_per_subpart_2d(lg_nvertices, lg_nvertices_local_1d_per_subowner))) == i);
      }
    }
#endif
#pragma omp parallel for
    for (int i = 0; i < recv_total; ++i) {
      uint64_t d = incoming_data[i];
      uint32_t v0 = uint32_t(d >> lg_nvertices_per_subpart_2d(lg_nvertices, lg_nvertices_local_1d_per_subowner)) & subpart_2d_mask;
      uint32_t v1 = uint32_t(d) & subpart_2d_mask;
      incoming_data[i] = make_edge_local(v0, v1);
    }
#pragma omp parallel for reduction(+:total_local_edges)
    for (int i = 0; i < int(nlocal_slots); ++i) {
      size_t nedges_here = slot_start[i + 1] - slot_start[i];
      edge_range& this_edge_range = g.edge_ranges[i];
      this_edge_range.append(incoming_data.get(slot_start[i], nedges_here), nedges_here);
      total_local_edges += nedges_here;
    }
    incoming_data.reset();
  } ITERATE_DESTRUCTIVE_TUPLE_GRAPH_END;
#pragma omp parallel for
  for (int i = 0; i < int(nlocal_slots); ++i) {
    edge_range& this_edge_range = g.edge_ranges[i];
    this_edge_range.finish();
  }

  uint64_t total_compressed_size = 0;
  for (int i = 0; i < int(nlocal_slots); ++i) {
    total_compressed_size += g.edge_ranges[i].compressed_size();
  }
  MPI_Allreduce(MPI_IN_PLACE, &total_compressed_size, 1, MPI_UINT64_T, MPI_MAX, bfs_communicator);

  if (bfs_rank == 0) {
    fprintf(stderr, "Done building graph data structure; running with vertex count 2**%d (from a limit of 2**%d)\n", g.lg_nvertices, (int)(32 + nprocbits_rounded_down));
    fprintf(stderr, "Maximum compressed data size is %" PRIu64 " words = %" PRIu64 " bytes = 0x%" PRIx64 " bytes\n", total_compressed_size, total_compressed_size * sizeof(uint32_t), total_compressed_size * sizeof(uint32_t));
  }
#if 0
  // Just looking at graph statistics
  if (bfs_rank == 0) fprintf(stderr, "Exiting\n");
  MPI_Finalize();
  exit(0);
#endif
  // fprintf(stderr, "Rank %d has %" PRIu64 " edge(s)\n", bfs_rank, total_local_edges);

#if 0
  int64_t nedges_local = 0;
  char* msg;
  asprintf(&msg, "%d has %d range(s): ", bfs_rank, (int)g.n_edge_ranges);
  for (size_t i = 0; i < g.n_edge_ranges; ++i) {
    char* newmsg;
    {asprintf(&newmsg, "%s%s[len=%zu]", msg, (i == 0 ? "" : ", "), g.edge_ranges[i].compressed_size); xfree(msg); msg = newmsg;}
    nedges_local += g.edge_ranges[i].compressed_size;
  }
  fprintf(stderr, "%s; %" PRId64 " compressed word(s) of edges total\n", msg, nedges_local);
  xfree(msg);
#endif

  // Initialization of queues moved to top of BFS
  for (unsigned int i = 0; i < nsubranks_1d; ++i) {
    uint64_t start = build_global_vertex_number_2d(coord_row_memberships(i), 0, lg_nvertices, lg_nvertices_local_1d_per_subowner);
    g.in_queue[i].start_vertex = start;
    g.out_queue[i].start_vertex = start;
    g.out_queue_last[i].start_vertex = start;
    g.my_assigned_targets[i].start_vertex = start;
    g.visited[i].start_vertex = start;
  }
  for (unsigned int row = 0; row < triangle_side; ++row) {
    unsigned int subrank = UINT_MAX;
    unsigned int rank_in_row = (unsigned int)MPI_UNDEFINED;
    for (unsigned int i = 0; i < nsubranks_1d; ++i) {
      unsigned int a, b;
      coord_to_global((unsigned int)bfs_rank, i, a, b);
      if (a == row) {
        subrank = i;
        rank_in_row = b;
        break;
      }
    }
    MPI_Comm dummy;
    MPI_Comm_split(bfs_communicator, (subrank == UINT_MAX ? MPI_UNDEFINED : 0), rank_in_row, (subrank == UINT_MAX ? &dummy : &g.comm_per_row[subrank]));
    if (subrank != UINT_MAX) {
      g.rank_in_row[subrank] = rank_in_row;
    }
  }
  for (unsigned int i = 0; i < nsubranks_1d; ++i) assert (g.comm_per_row[i] != MPI_COMM_NULL);

  for (unsigned int i = 0; i < nlocal_slots; ++i) {
    chunk_info& ch = g.chunks_to_process[2 * i + 0];
    chunk_info& ch2 = g.chunks_to_process[2 * i + 1];
    if (g.edge_ranges[i].empty()) {
      ch.valid = false;
      ch2.valid = false;
      continue;
    }
    ch.valid = true;
    ch.edge_slot = i;
    unsigned int a, b;
    coord_to_global(bfs_rank, (i >> (2 * lg_subsplit_side)), a, b);
    ch.in_bitmap_subrank = UINT_MAX;
    for (unsigned int j = 0; j < nsubranks_1d; ++j) {
      if (coord_row_memberships(j) == a) ch.in_bitmap_subrank = j;
    }
#ifndef NDEBUG
    if (ch.in_bitmap_subrank == UINT_MAX) {
      fprintf(stderr, "%d -> %d: Could not find in subrank for slot %u (shifted = %u), coords are %u %u\n", rank, bfs_rank, i, (i >> (2 * lg_subsplit_side)), a, b);
      abort();
    }
#endif
    ch.out_bitmap_subrank = UINT_MAX;
    for (unsigned int j = 0; j < nsubranks_1d; ++j) {
      if (coord_row_memberships(j) == b) ch.out_bitmap_subrank = j;
    }
#ifndef NDEBUG
    if (ch.out_bitmap_subrank == UINT_MAX) {
      fprintf(stderr, "%d -> %d: Could not find out subrank for slot %u (shifted = %u), coords are %u %u\n", rank, bfs_rank, i, (i >> (2 * lg_subsplit_side)), a, b);
      abort();
    }
#endif
    ch.in_bitmap_offset = uint32_t(((i >> lg_subsplit_side) % subsplit_side) * uint64_t(nvertices_per_subpart_2d(lg_nvertices, lg_nvertices_local_1d_per_subowner)));
    ch.out_bitmap_offset = uint32_t((i % subsplit_side) * uint64_t(nvertices_per_subpart_2d(lg_nvertices, lg_nvertices_local_1d_per_subowner)));
    ch.base_src = build_global_vertex_number_2d(a, ch.in_bitmap_offset, lg_nvertices, lg_nvertices_local_1d_per_subowner);
    ch.base_tgt = build_global_vertex_number_2d(b, ch.out_bitmap_offset, lg_nvertices, lg_nvertices_local_1d_per_subowner);
    g.in_queue[ch.in_bitmap_subrank].check(ch.base_src - ch.in_bitmap_offset);
    g.out_queue[ch.out_bitmap_subrank].check(ch.base_tgt - ch.out_bitmap_offset);
    g.my_assigned_targets[ch.out_bitmap_subrank].check(ch.base_tgt - ch.out_bitmap_offset);
    ch.edge_flip_status = chunk_info::normal;
    ch.total_edge_marks.ensure_size(g.edge_ranges[i].edge_count() + 128);
    ch2.valid = true;
    ch2.edge_slot = i;
    ch2.in_bitmap_subrank = ch.out_bitmap_subrank;
    ch2.out_bitmap_subrank = ch.in_bitmap_subrank;
    ch2.in_bitmap_offset = ch.out_bitmap_offset;
    ch2.out_bitmap_offset = ch.in_bitmap_offset;
    ch2.base_src = ch.base_tgt;
    ch2.base_tgt = ch.base_src;
    ch2.edge_flip_status = chunk_info::transposed;
    ch2.total_edge_marks_transposed.ensure_size(g.edge_ranges[ch.edge_slot].edge_count() + 128);
  }
}

void free_graph_data_structure(void) {
  MPI_Type_free(&addr_data_pair_datatype);
  for (unsigned int i = 0; i < nlocal_slots; ++i) g.edge_ranges[i].clear();

  for (unsigned int i = 0; i < nsubranks_1d; ++i) {
    MPI_Comm_free(&g.comm_per_row[i]);
  }
  MPI_Comm_free(&bfs_communicator);
}

int bfs_writes_depth_map() {
  return 1;
}

MPI_Comm get_communicator_for_validation() {
  return bfs_communicator;
}

void get_vertex_distribution_for_pred(size_t count, const int64_t* vertex_p, int* owner_p, size_t* local_p) {
  const int64_t* restrict vertex = vertex_p;
  int* restrict owner = owner_p;
  size_t* restrict local = local_p;
  const unsigned int lg_nvertices = g.lg_nvertices;
  const unsigned int lg_nvertices_local_1d_per_subowner = compute_lg_nvertices_local_1d_per_subowner(g.lg_nvertices);
#pragma omp parallel for
  for (ptrdiff_t i = 0; i < (ptrdiff_t)count; ++i) {
    if (vertex[i] == -1) continue;
    vertex_owner_and_local_1d(encode_vertex(vertex[i], lg_nvertices), lg_nvertices, lg_nvertices_local_1d_per_subowner, (unsigned int&)owner[i], local[i]);
  }
}

int64_t vertex_to_global_for_pred(int v_rank, size_t v_local) {
  return (int64_t)decode_vertex(vertex_to_global_1d((unsigned int)v_rank, v_local, g.lg_nvertices, compute_lg_nvertices_local_1d_per_subowner(g.lg_nvertices)), g.lg_nvertices);
}

size_t get_nlocalverts_for_pred(void) {
  return nvertices_local_1d_total(g.lg_nvertices, compute_lg_nvertices_local_1d_per_subowner(g.lg_nvertices));
}

static unsigned long long n_compressed_bits, n_uncompressed_bits;

static void reset_compression_ratio() {n_compressed_bits = n_uncompressed_bits = 0;}
static inline void add_compression_ratio(unsigned long n_compressed, unsigned long n_uncompressed) {
#pragma omp critical (compression_ratio_update)
  {
    n_compressed_bits += n_compressed;
    n_uncompressed_bits += n_uncompressed;
  }
}
static void print_compression_ratio(int compression_level) {
  unsigned long long total_compressed_bits, total_uncompressed_bits;
  MPI_Reduce(&n_compressed_bits, &total_compressed_bits, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, bfs_communicator);
  MPI_Reduce(&n_uncompressed_bits, &total_uncompressed_bits, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, bfs_communicator);
  if (bfs_rank == 0) {
    fprintf(stderr, "Compression ratio for compression level %d is %.2lf%%\n", compression_level, 100. * (1. - (double)total_compressed_bits/(double)total_uncompressed_bits));
  }
}

enum event_type {e_scan_waiting_for_chunk, e_scan_received_chunk, e_scan_ready_to_send_chunk, e_scan_waiting_for_send, e_scan_send_completed, e_scan_accumulating_assigned_bm, e_scan_done_accumulating_assigned_bm, e_scan_accumulating_assigned_bm_trivial, e_scan_done_accumulating_assigned_bm_trivial};
const char* event_type_names[] = {"scan_waiting_for_chunk", "scan_received_chunk", "scan_ready_to_send_chunk", "scan_waiting_for_send", "scan_send_completed", "scan_accumulating_assigned_bm", "scan_done_accumulating_assigned_bm", "scan_accumulating_assigned_bm_trivial", "scan_done_accumulating_assigned_bm_trivial"};
struct event {
  event_type et;
  size_t chunk_idx;
  unsigned int rank, row, rank_in_row;
  int thr;
  double time;
  event(event_type et, size_t chunk_idx, unsigned int rank, unsigned int row, unsigned int rank_in_row, int thr, double time)
    : et(et), chunk_idx(chunk_idx), rank(rank), row(row), rank_in_row(rank_in_row), thr(thr), time(time) {}
  friend std::ostream& operator<<(std::ostream& os, const event& e) {
    return os << std::setprecision(8) << e.time << ": " << e.rank << "=(" << e.row << ", " << e.rank_in_row << "), thr " << e.thr << ": " << event_type_names[e.et] << " chunk " << e.chunk_idx;
  }
};
static checked_array<4, std::vector<event> > events;

static inline void add_event(event_type et, size_t chunk_idx, unsigned int row, unsigned int rank_in_row) {
  (void)et;
  (void)chunk_idx;
  (void)row;
  (void)rank_in_row;
#if 0
#ifdef _OPENMP
  int thr = omp_get_thread_num();
#else
  int thr = 0;
#endif
  double time = MPI_Wtime();
  events[thr].push_back(event(et, chunk_idx, bfs_rank, row, rank_in_row, thr, time));
#endif
}

static void clear_events() {
  for (int thr = 0; thr < 4; ++thr) {
    events[thr].clear();
  }
}

static void dump_events() {
#if 0
  for (int thr = 0; thr < 4; ++thr) {
    for (std::vector<event>::const_iterator i = events[thr].begin(); i != events[thr].end(); ++i) {
      std::ostringstream os;
      os << *i;
      fprintf(stderr, "event: %s\n", os.str().c_str());
    }
  }
#endif
}

/* pred is distributed by edge target */
void run_bfs(int64_t root_raw, int64_t* pred, const bfs_settings& settings) {

  clear_events();
  if (bfs_rank == 0) fprintf(stderr, "Starting BFS with compression = %d\n", (int)settings.compression_level);
  reset_compression_ratio();
  // CALLGRIND_START_INSTRUMENTATION;
  callgrind_start();
  // MT_trace_start();
  assert (pred);
  const unsigned int bfs_rank = (unsigned int)(::bfs_rank);
  const unsigned int lg_nvertices = g.lg_nvertices;
  const unsigned int lg_nvertices_local_1d_per_subowner = compute_lg_nvertices_local_1d_per_subowner(lg_nvertices);
  uint64_t root = encode_vertex(root_raw, lg_nvertices);

#if 0
  if (bfs_rank == 0) {
    std::ostringstream msg;
    msg << "Queue pairs: ";
    for (unsigned int i = 0; i < g.nchunks_to_process; ++i) {
      if (i > 0) msg << ", ";
      msg << "<" << g.chunks_to_process[i].in_bitmap_index << " -> " << g.chunks_to_process[i].out_bitmap_index << ">";
    }
    msg << "\n";
    std::string msg_str = msg.str();
    fprintf(stderr, "%s", msg_str.c_str());
  }
#endif

#if 0
  fprintf(stderr, "nlocal* = (%zu sources, %zu targets), nglobalvertices = %" PRId64 "\n", nlocalsources, nlocaltargets, g.nglobalverts);
#endif

  if (nvertices_local_2d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner) % (1 << 16) != 0) {
    if (bfs_rank == 0) {
      fprintf(stderr, "Graph too small: nvertices_local_2d_per_subowner = %zu, must be a multiple of 1 << 16\n", (size_t)nvertices_local_2d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner));
    }
    abort();
  }

  for (unsigned int i = 0; i < nsubranks_1d; ++i) {
    g.in_queue[i].bm.ensure_size(nvertices_local_2d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner));
    g.in_blocks_empty[i].clear();
    g.in_blocks_empty[i].resize(((nvertices_local_2d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner) + (1 << 16) - 1) >> 16), 1);
    g.out_queue[i].bm.ensure_size(nvertices_local_2d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner));
    g.out_queue_last[i].bm.ensure_size(0);
    g.my_assigned_targets[i].bm.ensure_size(nvertices_local_2d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner));
    g.visited[i].bm.ensure_size(nvertices_local_2d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner));
  }
  scoped_array<int> elts_per_dest;
  scoped_array<int> displs_per_dest;
  elts_per_dest.reset_to_new(constant_size);
  displs_per_dest.reset_to_new(constant_size + 1);
  scoped_array<int> insert_positions;
  insert_positions.reset_to_new(constant_size);

  bool root_in_this_subrank[nsubranks_1d];

  for (unsigned int i = 0; i < nsubranks_1d; ++i) {
    // ensure_size clears data
    // g.in_queue[i].bm.reset_all();
    // g.out_queue[i].bm.reset_all();
    // g.visited[i].bm.reset_all();
    unsigned int root_owner_part;
    uint32_t root_local_part;
    split_global_vertex_number_2d(root, root_owner_part, root_local_part, lg_nvertices, lg_nvertices_local_1d_per_subowner);
    if (root_owner_part == coord_row_memberships(i)) {
      g.in_queue[i].check(root - root_local_part);
      g.out_queue[i].check(root - root_local_part);
      g.my_assigned_targets[i].check(root - root_local_part);
      g.in_queue[i].bm.set(root_local_part);
      g.in_blocks_empty[i][root_local_part >> 16] = 0;
      g.out_queue[i].bm.set(root_local_part);
      g.visited[i].bm.set(root_local_part);
    }
    root_in_this_subrank[i] = (root_owner_part == coord_row_memberships(i));
  }

  memset(pred, 0xFF, nvertices_local_1d_total(lg_nvertices, lg_nvertices_local_1d_per_subowner) * sizeof(int64_t));
  int root_is_mine_1d = 0;
  {
    unsigned int root_owner;
    size_t root_local;
    vertex_owner_and_local_1d(root, lg_nvertices, lg_nvertices_local_1d_per_subowner, root_owner, root_local);
    if (root_owner == bfs_rank) {
      pred[root_local] = make_lp(0, root_raw);
      root_is_mine_1d = 1;
    }
  }

  // Clear edge marks
  {
    size_t range_idx, range_idx_end = 2 * nlocal_slots;
    for (range_idx = 0; range_idx < range_idx_end; ++range_idx) {
      chunk_info& chunk = g.chunks_to_process[range_idx];
      if (!chunk.valid) continue;
      chunk.total_edge_marks.reset_all();
      chunk.total_edge_marks_transposed.reset_all();
    }
  }

  std::vector<double> marking_times, termination_check_times, scan_times, broadcast_times, pop_count_times, collecting_data_times, scatter_times;

  uint64_t nvisited = 0, total_out_queue_size = 0;
  uint16_t curlevel = 1;
  while (1) {
    if (curlevel >= UINT16_MAX) {
      if (bfs_rank == 0) fprintf(stderr, "BFS level number hit UINT16_MAX, but the level map uses uint16_t's.\n");
      MPI_Abort(bfs_communicator, 1);
    }
    if (bfs_rank == 0) fprintf(stderr, "Level %" PRIu16 " -- %f\n", curlevel, MPI_Wtime());

    // Mark in_queue vertices as having this level number
    if (curlevel != 1) { // Skip this for root
      for (unsigned int i = 0; i < nsubranks_1d; ++i) {
#if defined(CRAYPAT) && 0
        PAT_region_begin(1, "Adding level numbers to in_queue vertices");
#endif
        BITMAP_SAVE_PTRS_INTO_LOCAL_VARS_CONST(g.in_queue[i].bm, 0, inq_data)
        write_levels_for_in_queue(pred + (uint64_t(i) * nvertices_local_1d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner)),
                                  INT64_C(-0x10000) | (curlevel - 1),
                                  inq_data_bmdata + size_t((uint64_t(g.rank_in_row[i]) << lg_nvertices_local_1d_per_subowner) / BITMAP_ULONG_BITS),
                                  actual_nvertices_local_1d_per_subowner(lg_nvertices));
#if defined(CRAYPAT) && 0
        PAT_region_end(1);
#endif
      }
    }
    // fprintf(stderr, "%d: starting level %d\n", curlevel);
    // bitmap_clear_all(&g.out_queue[0]);
    // bitmap_clear_all(&g.out_queue[1]);
    // MPI_Barrier(bfs_communicator);

#ifndef NDEBUG
// #define EXHAUSTIVE_IN_QUEUE_LEVEL_CHECK
#endif

#ifdef EXHAUSTIVE_IN_QUEUE_LEVEL_CHECK
    {
      if (bfs_rank == 0) fprintf(stderr, "Exhaustive in queue level check\n");
      bitmap global_correct_level(nvertices_local_1d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner) * triangle_side_up * triangle_side_up);
      for (size_t i = 0; i < nvertices_local_1d_total(lg_nvertices); ++i) {
        if (i % nvertices_local_1d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner) >= actual_nvertices_local_1d_per_subowner(lg_nvertices)) {
          continue;
        }
        if ((curlevel == 1 && pred[i] == root) ||
            (curlevel != 1 && /* pred[i] != root && */ (pred[i] & UINT64_C(0xFFFF)) == uint64_t(curlevel) - 1)) {
          global_correct_level.set_non_atomic(vertex_to_global_1d(bfs_rank, (i / actual_nvertices_local_1d_per_subowner(lg_nvertices) * nvertices_local_1d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner)) + i % actual_nvertices_local_1d_per_subowner(lg_nvertices), lg_nvertices));
        }
      }
      global_correct_level.union_all_inplace(bfs_communicator);
      for (unsigned int i = 0; i < nsubranks_1d; ++i) {
        g.in_queue[i].check(build_global_vertex_number_2d(coord_row_memberships(i), 0, lg_nvertices));
        for (size_t j1 = 0; j1 < triangle_side; ++j1) {
          for (size_t j2 = 0; j2 < actual_nvertices_local_1d_per_subowner(lg_nvertices); ++j2) {
            size_t j = j1 * nvertices_local_1d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner) + j2;
            if (decode_vertex(j, lg_nvertices) >= nvertices(lg_nvertices)) break;
            bool bm_value = g.in_queue[i].bm.get(j);
            uint64_t vn_2d = build_global_vertex_number_2d(coord_row_memberships(i), uint32_t(j), lg_nvertices);
            bool correct = global_correct_level.get(vn_2d);
            assert (bm_value == correct);
          }
        }
      }
    }
#endif

    // MPI_Barrier(bfs_communicator);
    const double marking_start = MPI_Wtime();
#if defined(CRAYPAT) && 0
    PAT_region_begin(2, "Edge marking");
#endif

    for (unsigned int i = 0; i < nsubranks_1d; ++i) {
      BITMAP_SAVE_PTRS_INTO_LOCAL_VARS_CONST(g.in_queue[i].bm, 0, inq_data)
      const size_t block_size_in_words = (1 << 16) / BITMAP_ULONG_BITS;
      ptrdiff_t blk_end = ptrdiff_t((nvertices_local_2d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner) + (1 << 16) - 1) >> 16);
#pragma omp parallel for
      for (ptrdiff_t blk = 0; blk < blk_end; ++blk) {
        uint8_t empty = 1;
        size_t jend_bits = (1 << 16);
        if ((((uint64_t)blk + 1) << 16) > nvertices_local_2d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner)) {
          jend_bits = size_t(nvertices_local_2d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner) - ((uint64_t)blk << 16));
        }
        for (size_t j = 0; j < (jend_bits + BITMAP_ULONG_BITS - 1) / BITMAP_ULONG_BITS; ++j) {
          if (inq_data_bmdata[blk * block_size_in_words + j] != 0) {
            empty = 0;
            break;
          }
        }
        g.in_blocks_empty[i][blk] = empty;
      }
    }

    if (1) {
#pragma omp parallel // Every thread runs every index here
      {
        double marking_start_this_thread = MPI_Wtime(); (void)marking_start_this_thread;
        size_t range_idx, range_idx_end = 2 * nlocal_slots;
        for (range_idx = 0; range_idx < range_idx_end; ++range_idx) {
          chunk_info& chunk = g.chunks_to_process[range_idx];
          if (!chunk.valid) continue;
          if (curlevel == 1 && !root_in_this_subrank[chunk.in_bitmap_subrank]) continue;

          // Distribute work to threads by out_bitmap_index
#ifdef _OPENMP
          if ((int)(chunk.out_bitmap_offset >> lg_nvertices_per_subpart_2d(lg_nvertices, lg_nvertices_local_1d_per_subowner)) % omp_get_num_threads() != omp_get_thread_num()) continue;
#endif

          size_t edge_slot = chunk.edge_slot;
          unsigned int in_bitmap_subrank = chunk.in_bitmap_subrank;
          unsigned int out_bitmap_subrank = chunk.out_bitmap_subrank;
          uint64_t in_bitmap_offset = chunk.in_bitmap_offset;
          uint64_t out_bitmap_offset = chunk.out_bitmap_offset;
          uint64_t base_src = chunk.base_src;
          uint64_t base_tgt = chunk.base_tgt;
          (void)base_src;
          (void)base_tgt;
#ifndef NDEBUG
          assert (edge_slot < nlocal_slots);
          assert (in_bitmap_subrank < nsubranks_1d);
          assert (out_bitmap_subrank < nsubranks_1d);
          assert (in_bitmap_offset < nvertices_local_2d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner));
          assert (out_bitmap_offset < nvertices_local_2d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner));
          assert (decode_vertex(base_src, lg_nvertices) < nvertices(lg_nvertices));
          assert (decode_vertex(base_tgt, lg_nvertices) < nvertices(lg_nvertices));
#endif

          edge_range& cur_edge_range = g.edge_ranges[edge_slot];

          switch (chunk.edge_flip_status) {
            case chunk_info::normal: {
              size_t const marks_bmdatalen = chunk.total_edge_marks.get_nwords();
              unsigned long* restrict marks_bmdata = chunk.total_edge_marks.get_data(0, marks_bmdatalen + 1);
              BITMAP_SAVE_PTRS_INTO_LOCAL_VARS_CONST(g.in_queue[in_bitmap_subrank].bm, in_bitmap_offset, x_data)
              BITMAP_SAVE_PTRS_INTO_LOCAL_VARS(g.out_queue[out_bitmap_subrank].bm, out_bitmap_offset, b_data)
              bitmap_matvec_and_mark_impl(&cur_edge_range.row_low_halfwords[0], &cur_edge_range.col_low_halfwords[0], cur_edge_range.data.size(), &cur_edge_range.edge_block_counts_by_block[0], x_data_bmdata, nvertices_per_subpart_2d(lg_nvertices, lg_nvertices_local_1d_per_subowner), &g.in_blocks_empty[in_bitmap_subrank][size_t(in_bitmap_offset >> 16)], b_data_bmdata, nvertices_per_subpart_2d(lg_nvertices, lg_nvertices_local_1d_per_subowner), marks_bmdata);
              break;
            }
            case chunk_info::transposed: {
              size_t const marks_bmdatalen = chunk.total_edge_marks_transposed.get_nwords();
              unsigned long* restrict marks_bmdata = chunk.total_edge_marks_transposed.get_data(0, marks_bmdatalen + 1);
              BITMAP_SAVE_PTRS_INTO_LOCAL_VARS_CONST(g.in_queue[in_bitmap_subrank].bm, in_bitmap_offset, x_data)
              BITMAP_SAVE_PTRS_INTO_LOCAL_VARS(g.out_queue[out_bitmap_subrank].bm, out_bitmap_offset, b_data)
              bitmap_matvec_trans_and_mark_impl(&cur_edge_range.row_low_halfwords[0], &cur_edge_range.col_low_halfwords[0], cur_edge_range.data.size(), &cur_edge_range.edge_block_counts_by_block[0], x_data_bmdata, nvertices_per_subpart_2d(lg_nvertices, lg_nvertices_local_1d_per_subowner), &g.in_blocks_empty[in_bitmap_subrank][size_t(in_bitmap_offset >> 16)], b_data_bmdata, nvertices_per_subpart_2d(lg_nvertices, lg_nvertices_local_1d_per_subowner), marks_bmdata);
              break;
            }
            default: {fprintf(stderr, "%u: Invalid edge flip status %d\n", bfs_rank, (int)chunk.edge_flip_status); abort();}
          }
          // fprintf(stderr, "%u: ending range %d\n", bfs_rank, range_idx);
        }
        double marking_end_this_thread = MPI_Wtime(); (void)marking_end_this_thread;
        // if (bfs_rank == 0 || false) fprintf(stderr, "Finished marking pass (thread %d) for level %d in %lf\n", (int)omp_get_thread_num(), (int)curlevel, marking_end_this_thread - marking_start_this_thread);
      } // End of omp parallel
    }
    // MPI_Barrier(bfs_communicator);
    const double marking_end = MPI_Wtime();
    marking_times.push_back(marking_end - marking_start);
#if defined(CRAYPAT) && 0
    PAT_region_end(2);
#endif
    // if (bfs_rank == 0 || false) fprintf(stderr, "Finished marking pass for level %d in %lf\n", (int)curlevel, marking_end - marking_start);

#if defined(CRAYPAT) && 0
    PAT_region_begin(3, "Termination checking");
#endif
    const double termination_check_start = MPI_Wtime();
    int done = 1; // int for MPI

    // This loop is not parallel because bitmap::and_not is.
    for (unsigned int i = 0; i < nsubranks_1d; ++i) {
      g.out_queue[i].check(g.visited[i]);
      int local_done = !g.out_queue[i].bm.and_not(g.visited[i].bm);
      done &= local_done;
    }

    // MT_trace_stop();
    MPI_Allreduce(MPI_IN_PLACE, &done, 1, MPI_INT, MPI_LAND, bfs_communicator);
    // MT_trace_start();
    const double termination_check_end = MPI_Wtime();
    termination_check_times.push_back(termination_check_end - termination_check_start);
#if defined(CRAYPAT) && 0
    PAT_region_end(3);
#endif
    // if (bfs_rank == 0 || false) fprintf(stderr, "Finished termination check for level %d in %lf\n", (int)curlevel, termination_check_end - termination_check_start);
    if (done) {break;}

    uint64_t this_total_out_queue_size;
    if (settings.keep_queue_stats) {
      uint64_t out_queue_size = 0;
#pragma omp parallel for reduction(+:out_queue_size)
      for (int i = 0; i < (int)nsubranks_1d; ++i) {
        out_queue_size += popcount_range(g.out_queue[i].bm.get_data(0, g.out_queue[i].bm.get_nwords()), g.out_queue[i].bm.get_length());
      }
      MPI_Reduce(&out_queue_size, &this_total_out_queue_size, 1, MPI_UINT64_T, MPI_SUM, 0, bfs_communicator);
      total_out_queue_size += this_total_out_queue_size;
    }

    // At this point, we no longer need the old in_queue, so use that for out_queue_last
    g.in_queue.swap(g.out_queue_last);

#if defined(CRAYPAT) && 0
    PAT_region_begin(4, "Scans and broadcasts");
#endif
    // do_one_scan_and_diff does the or reduction into out_queue_last, and puts
    // only the new elements into out_queue.
    {
      struct do_one_scan_and_diff: coroutine {
        COROUTINE_STATE_DECL
        MPI_Request req;
        unsigned int i;
        unsigned int rank_in_row;
        unsigned int row;
        unsigned int bfs_rank;
        size_t chunk_offset, chunk_stride;
        const bfs_settings& settings;
        size_t word_offset_in_bitmap, chunk_len, word_len_in_bitmap, chunk_idx;
        size_t cur_offset, cur_size;
        int recvcount, sendcount;
        MPI_Status st;
        scoped_array<unsigned long> oql_incoming_compressed;
        scoped_array<size_t> incoming_compressed_counts;
        scoped_array<size_t>& compressed_counts;
        // scoped_array<double> times;

        do_one_scan_and_diff(unsigned int i, unsigned int bfs_rank, size_t chunk_offset, size_t chunk_stride, const bfs_settings& settings, size_t word_offset_in_bitmap, size_t chunk_len, size_t word_len_in_bitmap, scoped_array<size_t>& compressed_counts): COROUTINE_STATE_INIT, req(MPI_REQUEST_NULL), i(i), bfs_rank(bfs_rank), chunk_offset(chunk_offset), chunk_stride(chunk_stride), settings(settings), word_offset_in_bitmap(word_offset_in_bitmap), chunk_len(chunk_len), word_len_in_bitmap(word_len_in_bitmap), chunk_idx(0), compressed_counts(compressed_counts) {}

        COROUTINE_PROC_START {
          rank_in_row = g.rank_in_row[i];
          row = coord_row_memberships(i);
          oql_incoming_compressed.reset_to_new(chunk_len * ((word_len_in_bitmap + chunk_len * chunk_stride - 1) / (chunk_len * chunk_stride)));
          incoming_compressed_counts.reset_to_new((word_len_in_bitmap + chunk_len * chunk_stride - 1) / (chunk_len * chunk_stride));
          // times.reset_to_new((word_len_in_bitmap + chunk_len - 1) / chunk_len * 4);
          for (chunk_idx = chunk_offset; chunk_idx * chunk_len < word_len_in_bitmap; chunk_idx += chunk_stride) {
            // times[chunk_idx * 4 + 0] = MPI_Wtime();
            cur_offset = word_offset_in_bitmap + chunk_len * chunk_idx;
            cur_size = (std::min)(cur_offset + chunk_len, word_len_in_bitmap) - cur_offset;
            if (rank_in_row == 0) {
              unsigned long* restrict oql_bm = g.out_queue_last[i].bm.get_data(cur_offset, cur_size);
              // times[chunk_idx * 4 + 1] = MPI_Wtime();
              sendcount = (int)compress_bitmap(g.out_queue[i].bm.get_data(cur_offset, cur_size), cur_size * BITMAP_ULONG_BITS, oql_bm, settings.compression_level);
              if (settings.keep_queue_stats) add_compression_ratio((sendcount + 1) * BITMAP_ULONG_BITS, cur_size * BITMAP_ULONG_BITS);
              incoming_compressed_counts[chunk_idx / chunk_stride] = 0;
            } else {
              WITH_MPI_LOCK
                MPI_Irecv(oql_incoming_compressed.get(chunk_len * (chunk_idx / chunk_stride), cur_size),
                          (int)cur_size,
                          MPI_UNSIGNED_LONG,
                          rank_in_row - 1,
                          1000 + (int)chunk_offset,
                          g.comm_per_row[i],
                          &req);
              add_event(e_scan_waiting_for_chunk, chunk_idx, row, rank_in_row);
              COROUTINE_MPI_WAIT(1, &req, &st);
              add_event(e_scan_received_chunk, chunk_idx, row, rank_in_row);
              // times[chunk_idx * 4 + 1] = MPI_Wtime();
              WITH_MPI_LOCK MPI_Get_count(&st, MPI_UNSIGNED_LONG, &recvcount);
              assert (recvcount >= 0 && (size_t)recvcount <= cur_size);
              incoming_compressed_counts[chunk_idx / chunk_stride] = recvcount;
              const unsigned long* restrict oq_bm = g.out_queue[i].bm.get_data(cur_offset, cur_size);
              unsigned long* restrict oql_bm = g.out_queue_last[i].bm.get_data(cur_offset, cur_size);
              sendcount = (int)bitmap_or_compressed(oql_bm,
                                                    oql_incoming_compressed.get(chunk_len * (chunk_idx / chunk_stride), recvcount), recvcount, cur_size * BITMAP_ULONG_BITS,
                                                    oq_bm, cur_size * BITMAP_ULONG_BITS,
                                                    settings.compression_level);
              if (settings.keep_queue_stats) add_compression_ratio((sendcount + 1) * BITMAP_ULONG_BITS, cur_size * BITMAP_ULONG_BITS);
            }
            compressed_counts[chunk_idx] = sendcount;
            // times[chunk_idx * 4 + 2] = MPI_Wtime();
            add_event(e_scan_ready_to_send_chunk, chunk_idx, row, rank_in_row);
            if (rank_in_row == triangle_side - 1) {
              // Nothing
            } else {
              {
                unsigned long* restrict oql_bm = g.out_queue_last[i].bm.get_data(cur_offset, cur_size);
                WITH_MPI_LOCK
                  MPI_Isend(oql_bm,
                            sendcount,
                            MPI_UNSIGNED_LONG,
                            rank_in_row + 1,
                            1000 + (int)chunk_offset,
                            g.comm_per_row[i],
                            &req);
              }
              add_event(e_scan_waiting_for_send, chunk_idx, row, rank_in_row);
              COROUTINE_MPI_WAIT(2, &req, MPI_STATUS_IGNORE);
              add_event(e_scan_send_completed, chunk_idx, row, rank_in_row);
            }
            // times[chunk_idx * 4 + 3] = MPI_Wtime();
          }
          // double oq_update_start = MPI_Wtime();
          if (rank_in_row != 0) {
            for (size_t chunk_idx = chunk_offset; chunk_idx * chunk_len < word_len_in_bitmap; chunk_idx += chunk_stride) {
              add_event(e_scan_accumulating_assigned_bm, chunk_idx, row, rank_in_row);
              size_t cur_offset = word_offset_in_bitmap + chunk_len * chunk_idx;
              size_t cur_size = (std::min)(cur_offset + chunk_len, word_len_in_bitmap) - cur_offset;
              size_t cur_compressed_size = incoming_compressed_counts[chunk_idx / chunk_stride];
              unsigned long* restrict oq_bm = g.out_queue[i].bm.get_data(cur_offset, cur_size);
              unsigned long* restrict assigned_bm = g.my_assigned_targets[i].bm.get_data(cur_offset, cur_size);
              bitmap_andc_compressed_then_or(assigned_bm, cur_size * BITMAP_ULONG_BITS, oq_bm, cur_size * BITMAP_ULONG_BITS,
                                             oql_incoming_compressed.get(chunk_len * (chunk_idx / chunk_stride), cur_compressed_size), cur_compressed_size, cur_size * BITMAP_ULONG_BITS);
              add_event(e_scan_done_accumulating_assigned_bm, chunk_idx, row, rank_in_row);
            }
          } else {
            for (size_t chunk_idx = chunk_offset; chunk_idx * chunk_len < word_len_in_bitmap; chunk_idx += chunk_stride) {
              add_event(e_scan_accumulating_assigned_bm_trivial, chunk_idx, row, rank_in_row);
              size_t cur_offset = word_offset_in_bitmap + chunk_len * chunk_idx;
              size_t cur_size = (std::min)(cur_offset + chunk_len, word_len_in_bitmap) - cur_offset;
              const unsigned long* restrict oq_bm = g.out_queue[i].bm.get_data(cur_offset, cur_size);
              unsigned long* restrict assigned_bm = g.my_assigned_targets[i].bm.get_data(cur_offset, cur_size);
              bitmap_or(assigned_bm, cur_size * BITMAP_ULONG_BITS, oq_bm);
              add_event(e_scan_done_accumulating_assigned_bm_trivial, chunk_idx, row, rank_in_row);
            }
          }
          // double oq_update_end = MPI_Wtime();
#if 0
          for (chunk_idx = 0; chunk_idx < word_len_in_bitmap / chunk_len; ++chunk_idx) {
            fprintf(stderr, "<%u,%u> = %d: Started scan step for chunk %zu at %lf\n", row, rank_in_row, bfs_rank, chunk_idx, times[chunk_idx * 4 + 0]);
            fprintf(stderr, "<%u,%u> = %d: Scan recv done for chunk %zu at %lf\n", row, rank_in_row, bfs_rank, chunk_idx, times[chunk_idx * 4 + 1]);
            fprintf(stderr, "<%u,%u> = %d: Scan update done for chunk %zu at %lf\n", row, rank_in_row, bfs_rank, chunk_idx, times[chunk_idx * 4 + 2]);
            fprintf(stderr, "<%u,%u> = %d: Scan send done for chunk %zu at %lf\n", row, rank_in_row, bfs_rank, chunk_idx, times[chunk_idx * 4 + 3]);
          }
          fprintf(stderr, "<%u,%u> = %d: Started out_queue update at %lf\n", row, rank_in_row, bfs_rank, oq_update_start);
          fprintf(stderr, "<%u,%u> = %d: Update of out_queue done at %lf\n", row, rank_in_row, bfs_rank, oq_update_end);
#endif
          memfree("do_one_scan_and_diff", sizeof(do_one_scan_and_diff));
          delete this;
          COROUTINE_EXIT;
        } COROUTINE_PROC_END;
      };

      // static const unsigned int tree_step = (triangle_side <= 32 ? 4 : (1U << (lg_triangle_side / 2)));
      struct do_one_broadcast: coroutine {
        COROUTINE_STATE_DECL
        MPI_Request req, req2;
        unsigned int i;
        unsigned int rank_in_row;
        unsigned int row;
        unsigned int bfs_rank;
        size_t chunk_offset, chunk_stride;
        unsigned int delta, parent, child1, child2;
        size_t word_offset_in_bitmap, chunk_len, word_len_in_bitmap, chunk_idx;
        size_t cur_offset, cur_size;
        MPI_Status st;
        int recvcount;
        scoped_array<unsigned long> temp_buf;
        const bfs_settings& settings;
        const scoped_array<size_t>& compressed_counts;

        do_one_broadcast(unsigned int i, unsigned int bfs_rank, size_t chunk_offset, size_t chunk_stride, size_t word_offset_in_bitmap, size_t chunk_len, size_t word_len_in_bitmap, const bfs_settings& settings, const scoped_array<size_t>& compressed_counts): COROUTINE_STATE_INIT, req(MPI_REQUEST_NULL), i(i), bfs_rank(bfs_rank), chunk_offset(chunk_offset), chunk_stride(chunk_stride), word_offset_in_bitmap(word_offset_in_bitmap), chunk_len(chunk_len), word_len_in_bitmap(word_len_in_bitmap), chunk_idx(0), settings(settings), compressed_counts(compressed_counts) {}

        COROUTINE_PROC_START {
          rank_in_row = g.rank_in_row[i];
          row = coord_row_memberships(i);
          temp_buf.reset_to_new(chunk_len);
          for (chunk_idx = chunk_offset; chunk_idx * chunk_len < word_len_in_bitmap; chunk_idx += chunk_stride) {
            cur_offset = word_offset_in_bitmap + chunk_len * chunk_idx;
            cur_size = (std::min)(cur_offset + chunk_len, word_len_in_bitmap) - cur_offset;
            if (rank_in_row == triangle_side - 1) {
              // Start broadcast in reverse direction
              g.out_queue_last[i].check(g.visited[i]);
              g.out_queue_last[i].check(g.in_queue[i]);
              req = MPI_REQUEST_NULL;
              recvcount = (int)compressed_counts[chunk_idx];
              memcpy(temp_buf.get(0, recvcount), g.out_queue_last[i].bm.get_data(cur_offset, recvcount), recvcount * sizeof(unsigned long));
            } else {
              // Receive broadcast
              // unsigned int parent = triangle_side - 1 - ((triangle_side - 2 - rank_in_row) / 2);
              parent = (rank_in_row % settings.bcast_step_size == (triangle_side - 1) % settings.bcast_step_size) ? rank_in_row + settings.bcast_step_size : rank_in_row + 1;
              WITH_MPI_LOCK
                MPI_Irecv(temp_buf.get(0, cur_size),
                          (int)cur_size,
                          MPI_UNSIGNED_LONG,
                          parent,
                          2000 + (int)chunk_offset,
                          g.comm_per_row[i],
                          &req);
              COROUTINE_MPI_WAIT(3, &req, &st);
              WITH_MPI_LOCK MPI_Get_count(&st, MPI_UNSIGNED_LONG, &recvcount);
            }
            {
              // unsigned int child1 = triangle_side - 1 - (2 * (triangle_side - 1 - rank_in_row) + 1);
              // unsigned int child2 = triangle_side - 1 - (2 * (triangle_side - 1 - rank_in_row) + 2);
              child1 = ((rank_in_row % settings.bcast_step_size == (triangle_side - 1) % settings.bcast_step_size) &&
                        rank_in_row >= (unsigned int)settings.bcast_step_size)
                       ? rank_in_row - settings.bcast_step_size
                       : constant_size;
              child2 = (rank_in_row == 0 || (rank_in_row % settings.bcast_step_size == triangle_side % settings.bcast_step_size)) ? constant_size : rank_in_row - 1;
              // Multiple active sends from the same buffer are not allowed by
              // some versions of the MPI standard, but I believe 2.2 allows
              // them and they work in practice.
              if (child1 < constant_size) {
                WITH_MPI_LOCK
                  MPI_Isend(temp_buf.get(0, recvcount),
                            (int)recvcount,
                            MPI_UNSIGNED_LONG,
                            child1,
                            2000 + (int)chunk_offset,
                            g.comm_per_row[i],
                            &req);
              } else {
                req = MPI_REQUEST_NULL;
              }
              if (child2 < constant_size) {
                WITH_MPI_LOCK
                  MPI_Isend(temp_buf.get(0, recvcount),
                            (int)recvcount,
                            MPI_UNSIGNED_LONG,
                            child2,
                            2000 + (int)chunk_offset,
                            g.comm_per_row[i],
                            &req2);
              } else {
                req2 = MPI_REQUEST_NULL;
              }
            }
            COROUTINE_MPI_WAIT(4, &req, MPI_STATUS_IGNORE);
            COROUTINE_MPI_WAIT(5, &req2, MPI_STATUS_IGNORE);
            decompress_bitmap(temp_buf.get(0, recvcount), recvcount, g.out_queue_last[i].bm.get_data(cur_offset, cur_size), cur_size * BITMAP_ULONG_BITS);
          }
          memfree("do_one_broadcast", sizeof(do_one_broadcast));
          delete this;
          COROUTINE_EXIT;
        } COROUTINE_PROC_END;
      };

      {
        const double scan_start = MPI_Wtime();

        const size_t len = (1 << 16) / sizeof(unsigned long);
        scoped_array<size_t> compressed_counts[nsubranks_1d];
        for (int i = 0; i < (int)nsubranks_1d; ++i) {
          compressed_counts[i].reset_to_new((g.out_queue_last[i].bm.get_nwords() + len - 1) / len); // Different threads use disjoint elements in the vector
        }
#pragma omp parallel
        {
          coroutine* action = skip();
          for (int i = 0; i < (int)nsubranks_1d; ++i) {
            // const size_t len = (8192 + sizeof(unsigned long) - 1) / sizeof(unsigned long);
            memalloc("do_one_scan_and_diff", sizeof(do_one_scan_and_diff));
            action = par(action,
                         new do_one_scan_and_diff(i, bfs_rank, (size_t)omp_get_thread_num(), (size_t)omp_get_num_threads(), settings, 0, len, g.out_queue_last[i].bm.get_nwords(), compressed_counts[i]));
          }
          run(action);
        }
        const double scan_end = MPI_Wtime();
        scan_times.push_back(scan_end - scan_start);

        const double broadcast_start = MPI_Wtime();
        // if (bfs_rank == 0) fprintf(stderr, "Finished scans, starting broadcasts -- %f\n", MPI_Wtime());
#pragma omp parallel
        {
          coroutine* action = skip();
          for (int i = 0; i < (int)nsubranks_1d; ++i) {
            // const size_t len = (8192 + sizeof(unsigned long) - 1) / sizeof(unsigned long);
            memalloc("do_one_broadcast", sizeof(do_one_broadcast));
            action = par(action,
                         new do_one_broadcast(i, bfs_rank, (size_t)omp_get_thread_num(), (size_t)omp_get_num_threads(), 0, len, g.out_queue_last[i].bm.get_nwords(), settings, compressed_counts[i]));
          }
          run(action);
        }
        const double broadcast_end = MPI_Wtime();
        broadcast_times.push_back(broadcast_end - broadcast_start);
      }
    }
#if defined(CRAYPAT) && 0
    PAT_region_end(4);
#endif

    // if (bfs_rank == 0 || false) fprintf(stderr, "Finished scans and broadcasts for level %d in %lf\n", (int)curlevel, scan_and_broadcast_end - scan_and_broadcast_start);
    
#if 0 && !defined(NDEBUG)
    {
      uint64_t total_assigned_targets = 0, total_edge_marks = 0;
      size_t range_idx, range_idx_end = 2 * nlocal_slots;
      for (range_idx = 0; range_idx < range_idx_end; ++range_idx) {
        chunk_info& chunk = g.chunks_to_process[range_idx];
        if (!chunk.valid) continue;

        // Distribute work to threads by out_bitmap_index
#ifdef _OPENMP
        // if ((int)(chunk.out_bitmap_offset >> lg_nvertices_per_subpart_2d(lg_nvertices)) % omp_get_num_threads() != omp_get_thread_num()) continue;
#endif

        size_t edge_slot = chunk.edge_slot;
        unsigned int in_bitmap_subrank = chunk.in_bitmap_subrank;
        unsigned int out_bitmap_subrank = chunk.out_bitmap_subrank;
        uint64_t in_bitmap_offset = chunk.in_bitmap_offset;
        uint64_t out_bitmap_offset = chunk.out_bitmap_offset;
        uint64_t base_src = chunk.base_src;
        uint64_t base_tgt = chunk.base_tgt;
        (void)base_src;
        (void)base_tgt;
#ifndef NDEBUG
        assert (edge_slot < nlocal_slots);
        assert (in_bitmap_subrank < nsubranks_1d);
        assert (out_bitmap_subrank < nsubranks_1d);
        assert (in_bitmap_offset < nvertices_local_2d_per_subowner(lg_nvertices));
        assert (out_bitmap_offset < nvertices_local_2d_per_subowner(lg_nvertices));
        assert (decode_vertex(base_src, lg_nvertices) < nvertices(lg_nvertices));
        assert (decode_vertex(base_tgt, lg_nvertices) < nvertices(lg_nvertices));
#endif

        edge_range& cur_edge_range = g.edge_ranges[edge_slot];

        switch (chunk.edge_flip_status) {
          case chunk_info::normal: {
            size_t const marks_bmdatalen = chunk.total_edge_marks.get_nwords();
            unsigned long* restrict total_marks_bmdata = chunk.total_edge_marks.get_data(0, marks_bmdatalen);
            uint64_t count = popcount_range(total_marks_bmdata, marks_bmdatalen * BITMAP_ULONG_BITS);
            fprintf(stderr, "Range %zu has %" PRIu64 " marks\n", range_idx, count);
            total_edge_marks += count;
            break;
          }
          case chunk_info::transposed: {
            size_t const marks_bmdatalen = chunk.total_edge_marks_transposed.get_nwords();
            unsigned long* restrict total_marks_bmdata = chunk.total_edge_marks_transposed.get_data(0, marks_bmdatalen);
            uint64_t count = popcount_range(total_marks_bmdata, marks_bmdatalen * BITMAP_ULONG_BITS);
            total_edge_marks += count;
            fprintf(stderr, "Range %zu has %" PRIu64 " marks\n", range_idx, count);
            break;
          }
          default: {fprintf(stderr, "%u: Invalid edge flip status %d\n", bfs_rank, (int)chunk.edge_flip_status); abort();}
        }
        // fprintf(stderr, "%u: ending range %d\n", bfs_rank, range_idx);
      }
      for (int i = 0; i < int(nsubranks_1d); ++i) {
        uint64_t nbits = g.my_assigned_targets[i].bm.get_length();
        size_t nwords = g.my_assigned_targets[i].bm.get_nwords();
        uint64_t at = popcount_range(g.my_assigned_targets[i].bm.get_data(0, nwords), nbits);
        total_assigned_targets += at;
        fprintf(stderr, "Subrank %d has %" PRIu64 " assigned target(s)\n", i, at);
      }
      assert (total_assigned_targets == total_edge_marks);
    }
#endif

    // Now, out_queue_last contains the next level's in_queue, and out_queue
    // should become the next out_queue.
    g.in_queue.swap(g.out_queue_last);

    for (int i = 0; i < int(nsubranks_1d); ++i) {
      g.visited[i].bm |= g.in_queue[i].bm;
      g.out_queue[i].bm = g.visited[i].bm;
    }

    ++curlevel;
  }

  // Build predecessor map (other than root, which was already written)
  int total_elts_to_receive = 0; // Used to count visited vertices later
  {
#if defined(CRAYPAT) && 0
    PAT_region_begin(5, "Population counts");
#endif
    const double pop_count_start = MPI_Wtime();
    assert (nvertices_local_1d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner) % BITMAP_ULONG_BITS == 0);
#pragma omp parallel for reduction(+:total_elts_to_receive)
    for (int i = 0; i < (int)nsubranks_1d; ++i) {
      // Count visited vertices whose pred map entries are owned by me
      total_elts_to_receive +=
        (int)popcount_range(g.visited[i].bm.get_data(
                              g.rank_in_row[i] * nvertices_local_1d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner) / BITMAP_ULONG_BITS,
                              nvertices_local_1d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner) / BITMAP_ULONG_BITS),
                            nvertices_local_1d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner));
    }
    if (root_is_mine_1d) --total_elts_to_receive; // We will not be getting a pred entry for the root
    for (size_t i = 0; i < nsubranks_1d; ++i) memset(elts_per_dest.get(0, constant_size), 0, constant_size * sizeof(int));

#pragma omp parallel
    {
      for (int i = 0; i < (int)nsubranks_1d; ++i) {
        const uint64_t start_v = g.my_assigned_targets[i].start_vertex;
        const size_t start_v_in_subowners = size_t(start_v / nvertices_local_1d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner));
        BITMAP_SAVE_PTRS_INTO_LOCAL_VARS_CONST(g.my_assigned_targets[i].bm, 0, at);
        const size_t nwords = size_t(at_bmdatalen / BITMAP_ULONG_BITS);
        const ptrdiff_t delta = nvertices_local_1d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner) / BITMAP_ULONG_BITS;
#pragma omp for
        for (ptrdiff_t j_in_delta = 0; j_in_delta < (ptrdiff_t)nwords / delta; ++j_in_delta) {
          unsigned int full_j_in_subowners = (unsigned int)start_v_in_subowners + (unsigned int)(j_in_delta);
          if (full_j_in_subowners % triangle_side_up >= triangle_side ||
              full_j_in_subowners / triangle_side_up >= triangle_side) {
            continue;
          }
          unsigned int owner;
          size_t local;
          vertex_owner_and_local_1d(full_j_in_subowners * nvertices_local_1d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner), lg_nvertices, lg_nvertices_local_1d_per_subowner, owner, local);
          int count = (int)popcount_range(&at_bmdata[j_in_delta * delta], nvertices_local_1d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner));
#pragma omp atomic
          elts_per_dest[owner] += count;
        }
      }
    }
    const double pop_count_end = MPI_Wtime();
    pop_count_times.push_back(pop_count_end - pop_count_start);
#if defined(CRAYPAT) && 0
    PAT_region_end(5);
#endif
    if (bfs_rank == 0 || false) fprintf(stderr, "Finished population counts in %lf\n", pop_count_end - pop_count_start);
#if defined(CRAYPAT) && 0
    PAT_region_begin(6, "Collecting data for pred map");
#endif
    const double collecting_data_start = MPI_Wtime();
    displs_per_dest[0] = 0;
    for (unsigned int j = 0; j < constant_size; ++j) {
      displs_per_dest[j + 1] = displs_per_dest[j] + elts_per_dest[j];
    }
    const int send_total = displs_per_dest[constant_size];
    if (settings.keep_queue_stats) {
      uint64_t union_assigned_target_size = 0, send_total_u64 = (uint64_t)send_total;
      MPI_Reduce(&send_total_u64, &union_assigned_target_size, 1, MPI_UINT64_T, MPI_SUM, 0, bfs_communicator);
      if (bfs_rank == 0) {
        fprintf(stderr, "Found %" PRIu64 " elements in unioned out queues vs. %" PRIu64 " elements in sum; savings is %.2lf%%\n",
                union_assigned_target_size, total_out_queue_size,
                100. * (1. - (double)union_assigned_target_size / (double)total_out_queue_size));
      }
    }
    scoped_array<addr_data_pair> send_data;
    send_data.reset_to_new(send_total);
    memcpy(insert_positions.get(0, constant_size), displs_per_dest.get(0, constant_size), constant_size * sizeof(int));
    for (unsigned int i = 0; i < nsubranks_1d; ++i) {
      const bitmap& at = g.my_assigned_targets[i].bm;
#pragma omp parallel
      {
        int range_idx_end = 2 * nlocal_slots;
        for (int range_idx = 0; range_idx < range_idx_end; ++range_idx) {
          const chunk_info& chunk = g.chunks_to_process[range_idx];
          if (!chunk.valid) {continue;}
          if (chunk.out_bitmap_subrank != i) {continue;}
          assert (chunk.edge_slot < nlocal_slots);

          const size_t edge_slot = chunk.edge_slot;
          const uint64_t base_src = chunk.base_src;
          const uint64_t base_tgt = chunk.base_tgt;

#ifdef _OPENMP
          if (int64_t(chunk.out_bitmap_offset >> lg_nvertices_per_subpart_2d(lg_nvertices, lg_nvertices_per_subpart_2d(lg_nvertices, lg_nvertices_local_1d_per_subowner))) % omp_get_num_threads()
                != omp_get_thread_num()) {
            continue; // Subdivide work
          }
#endif

          BITMAP_SAVE_PTRS_INTO_LOCAL_VARS_CONST(at, chunk.out_bitmap_offset, at_data);

          int owners[oned_parts_per_2d_subpart];
          uint32_t local_biases[oned_parts_per_2d_subpart];
          for (int j = 0; j < oned_parts_per_2d_subpart; ++j) {
            size_t local;
            vertex_owner_and_local_1d(base_tgt + (uint64_t(j) << lg_nvertices_local_1d_per_subowner), lg_nvertices, lg_nvertices_local_1d_per_subowner, (unsigned int&)owners[j], local);
            local_biases[j] = (uint32_t)local - (uint32_t(j) << lg_nvertices_local_1d_per_subowner);
          }
          const uint32_t data_bias = uint32_t(base_src % nvertices_local_2d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner)) | (uint32_t(chunk.in_bitmap_subrank) << 31);

          switch (chunk.edge_flip_status) {
            case chunk_info::normal: {
              // assert (owner_part_a == row);
              // assert (owner_part_b == rank_in_row);
              if (chunk.total_edge_marks.are_all_reset()) break;
              BITMAP_SAVE_PTRS_INTO_LOCAL_VARS_CONST(chunk.total_edge_marks, 0, total_edge_marks);
              const edge_range& the_range = (g.edge_ranges[edge_slot]);
              size_t range_len = the_range.edge_count();
              const uint64_t* restrict range_data = &the_range.data[0];
              write_pred_scatter_data(range_data, range_len, total_edge_marks_bmdata, at_data_bmdata, owners, insert_positions.get(0, constant_size), send_data.get(0, send_total), lg_nvertices_local_1d_per_subowner, local_biases, data_bias);
              break;
            }
            case chunk_info::transposed: {
              // assert (owner_part_a == rank_in_row);
              // assert (owner_part_b == row);
              if (chunk.total_edge_marks_transposed.are_all_reset()) break;
              BITMAP_SAVE_PTRS_INTO_LOCAL_VARS_CONST(chunk.total_edge_marks_transposed, 0, total_edge_marks_transposed);
              const edge_range& the_range = (g.edge_ranges[edge_slot]);
              size_t range_len = the_range.edge_count();
              const uint64_t* restrict range_data = &the_range.data[0];
              write_pred_scatter_data_trans(range_data, range_len, total_edge_marks_transposed_bmdata, at_data_bmdata, owners, insert_positions.get(0, constant_size), send_data.get(0, send_total), lg_nvertices_local_1d_per_subowner, local_biases, data_bias);
              break;
            }
            default: {fprintf(stderr, "%u: Invalid edge flip status %d\n", bfs_rank, (int)chunk.edge_flip_status); abort();}
          }
        }
      }
    }
    const double collecting_data_end = MPI_Wtime();
    collecting_data_times.push_back(collecting_data_end - collecting_data_start);
#if defined(CRAYPAT) && 0
    PAT_region_end(6);
#endif
    if (bfs_rank == 0 || false) fprintf(stderr, "Finished collecting send data for pred map for level %d in %lf\n", (int)curlevel, collecting_data_end - collecting_data_start);
#if 0
#ifndef NDEBUG
    scoped_array<int> data_counts;
    data_counts.reset_to_new(constant_size);
    memset(data_counts.get(0, constant_size), 0, constant_size * sizeof(int));
    for (unsigned int i = 0; i < nsubranks_1d; ++i) {
      for (unsigned int j = 0; j < triangle_side; ++j) {
        unsigned int owner, subowner;
        coord_owner_info(coord_row_memberships(i), j, owner, subowner);
        data_counts[owner] += elts_per_dest[i][j];
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, data_counts.get(0, constant_size), constant_size, MPI_INT, MPI_SUM, bfs_communicator);
    int total_elts_to_receive_2 = data_counts[bfs_rank];
    if (total_elts_to_receive != total_elts_to_receive_2) {
      fprintf(stderr, "%d: Got incorrect number of elements: pop count got %d, allreduce got %d\n", bfs_rank, total_elts_to_receive, total_elts_to_receive_2);
      abort();
    }
#endif
#endif
    // if (bfs_rank == 0) fprintf(stderr, "Finished allreduce (before barrier) -- %lf\n", MPI_Wtime());
    // MPI_Barrier(bfs_communicator);
    // if (bfs_rank == 0) fprintf(stderr, "Finished allreduce -- %lf\n", MPI_Wtime());
#if defined(CRAYPAT) && 0
    PAT_region_begin(7, "Sending pred map data");
#endif
    const double scatter_start = MPI_Wtime();
    static const int chunk_size_for_pred_map = 16384;
    static const size_t recv_depth = 32;
    checked_array<constant_size, MPI_Request> send_reqs;
    for (size_t i = 0; i < constant_size; ++i) send_reqs[i] = MPI_REQUEST_NULL;
    int data_received = 0, data_sent = 0;
    checked_array<nsubranks_1d * recv_depth, scoped_array<addr_data_pair> > recv_data;
    checked_array<nsubranks_1d * recv_depth, MPI_Request> recv_reqs;
    for (int i = 0; i < (int)nsubranks_1d * (int)recv_depth; ++i) {
      recv_data[i].reset_to_new(chunk_size_for_pred_map);
      WITH_MPI_LOCK
        MPI_Irecv(recv_data[i].get(0, chunk_size_for_pred_map), chunk_size_for_pred_map, addr_data_pair_datatype, MPI_ANY_SOURCE, MPI_ANY_TAG, bfs_communicator, &recv_reqs[i]);
    }
    checked_array<constant_size, int> send_offset;
    memset(send_offset.get(0, constant_size), 0, constant_size * sizeof(int));
    for (int dest = 0; dest < (int)constant_size; ++dest) {
      send_offset[dest] = 0;
      int count = (std::min)(send_offset[dest] + chunk_size_for_pred_map, elts_per_dest[dest]) - send_offset[dest];
      if (count != 0) {
        WITH_MPI_LOCK
          MPI_Isend(send_data.get(displs_per_dest[dest] + send_offset[dest], count), 
                    count,
                    addr_data_pair_datatype,
                    dest, 4000, bfs_communicator, &send_reqs[dest]);
        // fprintf(stderr, "Row %d rank %d (subrank %d) sending %d from offset %d (of %d) to %d\n", coord_row_memberships(subrank), g.rank_in_row[subrank], (int)subrank, count, send_offset[subrank][dest], elts_per_dest[subrank][dest], dest);
        send_offset[dest] += count;
        data_sent += count;
        assert (send_offset[dest] <= elts_per_dest[dest]);
      }
    }
    while (data_received < total_elts_to_receive || data_sent < send_total) {
      int index, flag;
      MPI_Status st;
      WITH_MPI_LOCK MPI_Testany(nsubranks_1d * recv_depth, &recv_reqs[0], &index, &flag, &st);
      if (flag && index != MPI_UNDEFINED) {
        const scoped_array<addr_data_pair>& data = recv_data[index];
        int count;
        WITH_MPI_LOCK MPI_Get_count(&st, addr_data_pair_datatype, &count);
        assert (count > 0 && count <= chunk_size_for_pred_map);
        checked_array<nsubranks_1d, uint64_t> data_bias;
        for (unsigned int sr = 0; sr < nsubranks_1d; ++sr) {
          unsigned int a, b;
          coord_to_global((unsigned int)st.MPI_SOURCE, sr, a, b);
          data_bias[sr] = uint64_t(a) << lg_nvertices_local_2d_per_subowner(lg_nvertices, lg_nvertices_local_1d_per_subowner);
        }
        // fprintf(stderr, "Row %d rank %d (subrank %d) received %d from %d\n", coord_row_memberships(subrank), g.rank_in_row[subrank], (int)subrank, count, st.MPI_SOURCE);
        const addr_data_pair* data_ptr = data.get(0, count);
#pragma omp parallel for if (count >= 1024)
        for (int j = 0; j < count; ++j) {
          size_t idx = size_t(data_ptr[j].addr);
          assert (idx < nvertices_local_1d_total(lg_nvertices, lg_nvertices_local_1d_per_subowner));
          const unsigned int src_subrank = (unsigned int)(data_ptr[j].data >> 31);
          assert ((uint64_t(pred[idx]) | 0xFFFF) == UINT64_MAX);
          assert (pred[idx] != -1);
          pred[idx] =
            make_lp(uint16_t(pred[idx] & 0xFFFF),
                    decode_vertex((data_ptr[j].data & 0x7FFFFFFFU) + data_bias[src_subrank], lg_nvertices));
        }
        data_received += count;
        WITH_MPI_LOCK
          MPI_Irecv(recv_data[index].get(0, chunk_size_for_pred_map), chunk_size_for_pred_map, addr_data_pair_datatype,
                    MPI_ANY_SOURCE, 4000, bfs_communicator, &recv_reqs[index]);
      }
      WITH_MPI_LOCK MPI_Testany(constant_size, &send_reqs[0], &index, &flag, &st);
      if (flag && index != MPI_UNDEFINED) {
        int dest = index;
        if (send_offset[dest] < elts_per_dest[dest]) { // Send another message
          int sendcount = (std::min)(send_offset[dest] + chunk_size_for_pred_map, elts_per_dest[dest]) - send_offset[dest];
          WITH_MPI_LOCK
            MPI_Isend(send_data.get(displs_per_dest[dest] + send_offset[dest], sendcount), 
                      sendcount,
                      addr_data_pair_datatype,
                      dest, 4000, bfs_communicator, &send_reqs[index]);
          // fprintf(stderr, "Row %d rank %d sending %d from offset %d (of %d) to %d\n", coord_row_memberships(subrank), g.rank_in_row[subrank], sendcount, send_offset[subrank][dest], elts_per_dest[subrank][dest], dest);
          send_offset[dest] += sendcount;
          data_sent += sendcount;
          assert (send_offset[dest] <= elts_per_dest[dest]);
        }
      }
    }
    WITH_MPI_LOCK MPI_Waitall((int)constant_size, send_reqs.get(0, constant_size), MPI_STATUSES_IGNORE);
    for (int i = 0; i < (int)nsubranks_1d * (int)recv_depth; ++i) {
      WITH_MPI_LOCK MPI_Cancel(&recv_reqs[i]);
      MPI_Wait(&recv_reqs[i], MPI_STATUS_IGNORE);
      recv_data[i].reset();
    }
#ifndef NDEBUG
    for (unsigned int j = 0; j < constant_size; ++j) {
      assert (send_offset[j] == elts_per_dest[j]);
    }
#endif
    const double scatter_end = MPI_Wtime();
    scatter_times.push_back(scatter_end - scatter_start);
#if defined(CRAYPAT) && 0
    PAT_region_end(7);
#endif
  }

  // if (bfs_rank == 0) fprintf(stderr, "Done with BFS (before barrier) -- %f\n", MPI_Wtime()); 
  // MPI_Barrier(bfs_communicator);
  if (bfs_rank == 0) fprintf(stderr, "Done with BFS -- %f\n", MPI_Wtime()); 

  for (unsigned int i = 0; i < nsubranks_1d; ++i) {
    g.in_queue[i].bm.clear();
    g.out_queue[i].bm.clear();
    g.out_queue_last[i].bm.clear();
    g.my_assigned_targets[i].bm.clear();
    g.visited[i].bm.clear();
  }

  nvisited = 0;
  uint64_t nvisited_local = total_elts_to_receive;
#ifndef NDEBUG
  uint64_t nvisited_local_2 = 0;
  ptrdiff_t vi_end = (ptrdiff_t)nvertices_local_1d_total(lg_nvertices, lg_nvertices_local_1d_per_subowner);
#pragma omp parallel for reduction(+:nvisited_local_2)
  for (ptrdiff_t vi = 0; vi < vi_end; ++vi) {
    if (level(pred[vi]) != UINT16_MAX) {
      ++nvisited_local_2;
    }
  }
  assert (nvisited_local + (root_is_mine_1d ? 1 : 0) == nvisited_local_2);
#endif
  MPI_Allreduce(&nvisited_local, &nvisited, 1, MPI_INT64_T, MPI_SUM, bfs_communicator);

  if (bfs_rank == 0) fprintf(stderr, "Visited %" PRId64 " vertices\n", nvisited + 1 /* root */);
  // CALLGRIND_STOP_INSTRUMENTATION;
  callgrind_stop();
  // MT_trace_stop();
  if (settings.keep_queue_stats) print_compression_ratio(settings.compression_level);
  if (settings.keep_average_level_times) {
    std::vector<double>* ptrs[] = {&marking_times, &termination_check_times, &scan_times, &broadcast_times, &pop_count_times, &collecting_data_times, &scatter_times};
    const char* formats[] = {
      "Finished marking pass for level %d in %lf\n",
      "Finished termination check for level %d in %lf\n",
      "Finished scans for level %d in %lf\n",
      "Finished broadcasts for level %d in %lf\n",
      "Finished population counts for level %d in %lf\n",
      "Finished collecting send data for pred map for level %d in %lf\n",
      "Finished scatters for level %d in %lf\n"
    };
    for (size_t i = 0; i < sizeof(ptrs) / sizeof(*ptrs); ++i) {
      if (bfs_rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, &(*ptrs[i])[0], (int)ptrs[i]->size(), MPI_DOUBLE, MPI_SUM, 0, bfs_communicator);
        for (size_t level = 0; level < ptrs[i]->size(); ++level) {
          fprintf(stderr, formats[i], (int)(level + 1), (*ptrs[i])[level] / constant_size);
        }
        fprintf(stderr, "-----------------------------\n");
      } else {
        MPI_Reduce(&(*ptrs[i])[0], MPI_BOTTOM, (int)ptrs[i]->size(), MPI_DOUBLE, MPI_SUM, 0, bfs_communicator);
      }
    }
  }
  dump_events();
}

#if 0
static void print_node_topology() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<int> allnids(rank == 0 ? size : 0);
  {
    FILE* f = fopen("/proc/cray_xt/nid", "r");
    if (!f) {perror ("open nid"); abort();}
    int nid;
    int count = fscanf(f, "%d", &nid);
    if (count != 1) {perror("fscanf"); abort();}
    fclose(f);

    MPI_Gather(&nid, 1, MPI_INT, &allnids[0], 1, MPI_INT, 0, MPI_COMM_WORLD);
  }

  if (rank != 0) return;

  {
    int num_x = 0, num_y = 0, num_z = 0;
    for (int i = 0; i < size; ++i) {
      int this_nid = allnids[i];
      rca_mesh_coord_t coord;
      rca_get_meshcoord(uint32_t(this_nid), &coord);
      num_x = (std::max)(num_x, coord.mesh_x + 1);
      num_y = (std::max)(num_y, coord.mesh_y + 1);
      num_z = (std::max)(num_z, coord.mesh_z + 1);
    }
    FILE* f = fopen("/mnt/lustre/lus0/jewillco/points", "w");
    if (!f) {perror("open points"); abort();}
    fprintf(f, "%d\n", size);
    for (int i = 0; i < size; ++i) {
      int this_nid = allnids[i];
      rca_mesh_coord_t coord;
      rca_get_meshcoord(uint32_t(this_nid), &coord);
      fprintf(f, "%d %lf %lf %lf\n", i, allnids[i], (coord.mesh_x + .5) / num_x, (coord.mesh_y + .5) / num_y, (coord.mesh_z + .5) / num_z);
    }
    fclose(f);
  }

  FILE* f = fopen("/mnt/lustre/lus0/jewillco/grid_topology.dot", "w");
  if (!f) {perror("open file"); abort();}

  fprintf(f, "graph topology {\n");
  for (int i = 0; i < triangle_side; ++i) {
    for (int j = 0; j < triangle_side; ++j) {
      unsigned int this_owner, this_subowner;
      coord_owner_info(i, j, this_owner, this_subowner);
      if (i != triangle_side - 1) {
        unsigned int next_owner, next_subowner;
        coord_owner_info(i + 1, j, next_owner, next_subowner);
        fprintf(f, "%u -- %u\n", this_owner, next_owner);
      }
      if (j != triangle_side - 1) {
        unsigned int next_owner, next_subowner;
        coord_owner_info(i, j + 1, next_owner, next_subowner);
        fprintf(f, "%u -- %u\n", this_owner, next_owner);
      }
    }
  }
  fprintf(f, "}\n");

  fclose(f);
}
#endif
