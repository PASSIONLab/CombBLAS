/* Copyright (C) 2010-2013 The Trustees of Indiana University.             */
/*                                                                         */
/* Use, modification and distribution is subject to the Boost Software     */
/* License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at */
/* http://www.boost.org/LICENSE_1_0.txt)                                   */
/*                                                                         */
/*  Authors: Jeremiah Willcock                                             */
/*           Andrew Lumsdaine                                              */

#ifndef COMMON_HPP
#define COMMON_HPP

#include <stdint.h>
#include <mpi.h>
#include <stddef.h>
#include <stdio.h>
#include <assert.h>
#include <algorithm>
#include <utility>
#include "../generator/graph_generator.h"
#include "mpi_workarounds.hpp"

// #define SIZE_MUST_BE_A_POWER_OF_TWO

extern int rank, size;
#ifdef SIZE_MUST_BE_A_POWER_OF_TWO
extern int lgsize;
#endif
extern MPI_Datatype packed_edge_mpi_type; /* MPI datatype for packed_edge struct */

/* Distribute edges by their endpoints (make two directed copies of each input
 * undirected edge); distribution is 1-d and cyclic. */
#ifdef SIZE_MUST_BE_A_POWER_OF_TWO
#define MOD_SIZE(v) ((v) & ((1 << lgsize) - 1))
#define DIV_SIZE(v) ((v) >> lgsize)
#define MUL_SIZE(x) ((x) << lgsize)
#else
#define MOD_SIZE(v) ((v) % size)
#define DIV_SIZE(v) ((v) / size)
#define MUL_SIZE(x) ((x) * size)
#endif
#define VERTEX_OWNER(v) ((int)(MOD_SIZE(v)))
#define VERTEX_LOCAL(v) ((size_t)(DIV_SIZE(v)))
#define VERTEX_TO_GLOBAL(r, i) ((int64_t)(MUL_SIZE((uint64_t)i) + (int)(r)))

static void memalloc(const char* msg, size_t sz) {
  (void)msg;
  (void)sz;
  // if (rank == 5) fprintf(stderr, "%d: %s alloc: %zu\n", rank, msg, sz);
}

static void memfree(const char* msg, size_t sz) {
  (void)msg;
  (void)sz;
  // if (rank == 5) fprintf(stderr, "%d: %s free: %zu\n", rank, msg, sz);
}

template <typename T>
class scoped_array {
  T* restrict ptr;
  size_t len;

  public:
  scoped_array(): ptr(0), len(0) {}
#if 0
  scoped_array(T* ptr, size_t len): ptr(ptr), len(len) {
    // if (rank == 0 && ptr) fprintf(stderr, "MEMORY: alloc %zu %p sa_constructor\n", len * sizeof(T), ptr);
  }
#endif

  ~scoped_array() {
    reset();
  }

  void swap(scoped_array<T>& o) {
    std::swap(ptr, o.ptr);
    std::swap(len, o.len);
  }

  void unsafe_reset() {
    ptr = 0;
    len = 0;
  }
#if 0
  void unsafe_reset(T* new_ptr, size_t new_len) {
    // if (rank == 0) fprintf(stderr, "MEMORY: alloc %zu %p sa_reset_unsafe\n", new_len * sizeof(T), new_ptr);
    ptr = new_ptr;
    len = new_len;
  }
#endif
  void reset() {
    // if (rank == 0) fprintf(stderr, "MEMORY: free ? %p delete_array\n", ptr);
    if (ptr) {memfree("scoped_array reset", len * sizeof(T));}
    delete[] ptr;
    ptr = 0;
    len = 0;
  }
#if 0
  void reset(T* new_ptr, size_t new_len) {
    // if (rank == 0 && ptr) fprintf(stderr, "MEMORY: free ? %p delete_array\n", ptr);
    delete[] ptr;
    // if (rank == 0 && new_ptr) fprintf(stderr, "MEMORY: alloc %zu %p sa_reset\n", new_len * sizeof(T), new_ptr);
    ptr = new_ptr;
    len = new_len;
  }
#endif
  void reset_to_new(size_t new_len) { // Put in new array
    if (ptr) {memfree("scoped_array reset_to_new", len * sizeof(T));}
    delete[] ptr;
    memalloc("scoped_array reset_to_new", new_len * sizeof(T));
    ptr = new T[new_len];
    len = new_len;
  }
  T& operator[](size_t i) const {assert (i < len); return ptr[i];}
  T* get(size_t start, size_t rlen) const {assert (start <= len); assert (start + rlen <= len); (void)rlen; return ptr + start;}
  size_t size() const {return len;}

  private: scoped_array(const scoped_array&);
};

template <size_t Len, typename T>
struct checked_array {
  T data[Len];

  checked_array() {}

  T& operator[](size_t i) {assert (i < Len); return data[i];}
  const T& operator[](size_t i) const {assert (i < Len); return data[i];}

  T* get(size_t start, size_t len) {
    assert (start <= Len);
    assert (start + len <= Len);
    (void)len;
    return data + start;
  }

  void swap(checked_array& a) {
    if (&a == this) return;
    for (size_t i = 0; i < Len; ++i) this->data[i].swap(a.data[i]);
  }
};

typedef struct tuple_graph {
  scoped_array<scoped_array<packed_edge> > edgememory;
  uint64_t nglobaledges; /* Number of edges in graph, in both cases */
} tuple_graph;

#define FILE_CHUNKSIZE ((MPI_Offset)(1) << 23) /* Size of one file I/O block or memory block to be processed in one step, in edges */

static inline size_t block_count_for_tg(const tuple_graph* const tg) {
  return tg->edgememory.size();
}

/* Simple iteration of edge data or file; cannot be nested. */
#define ITERATE_TUPLE_GRAPH_BLOCK_COUNT(tg) block_count_for_tg((tg))
#define ITERATE_TUPLE_GRAPH_BEGIN(tg, user_buf, user_buf_count) \
  do { \
    const tuple_graph* const tg_save = (tg); \
    const size_t block_limit = ITERATE_TUPLE_GRAPH_BLOCK_COUNT(tg_save); \
    size_t block_idx; \
    bool break_from_block_loop = false; \
    for (block_idx = 0; block_idx < block_limit; ++block_idx) { \
      scoped_array<packed_edge>& actual_edge_buffer = tg_save->edgememory[block_idx]; \
      ptrdiff_t const user_buf_count = (ptrdiff_t)actual_edge_buffer.size(); \
      const packed_edge* restrict const user_buf = actual_edge_buffer.get(0, actual_edge_buffer.size()); \
      assert (user_buf != NULL); \
      assert (user_buf_count >= 0); \
      assert (tuple_graph_max_bufsize(tg_save) >= user_buf_count); \
      int iteration_count = 0; \
      while (1) { \
        /* Prevent continue */ assert (iteration_count == 0); \
        {
#define ITERATE_TUPLE_GRAPH_BLOCK_NUMBER (block_idx)
#define ITERATE_TUPLE_GRAPH_BREAK /* Must be done collectively and before ITERATE_TUPLE_GRAPH_RELEASE_BUFFER */ \
          break_from_block_loop = true; \
          break
#define ITERATE_TUPLE_GRAPH_RELEASE_BUFFER \
          do {} while (0)
#define ITERATE_TUPLE_GRAPH_END \
          ITERATE_TUPLE_GRAPH_RELEASE_BUFFER; \
        } \
        if (break_from_block_loop) ITERATE_TUPLE_GRAPH_RELEASE_BUFFER; \
        iteration_count = 1; \
        break; \
      } \
      /* Prevent user break */ assert (iteration_count == 1); \
      if (break_from_block_loop) break; \
    } \
  } while (0)

/* Simple iteration of edge data or file; cannot be nested. */
#define ITERATE_DESTRUCTIVE_TUPLE_GRAPH_BLOCK_COUNT(tg) block_count_for_tg((tg))
#define ITERATE_DESTRUCTIVE_TUPLE_GRAPH_BEGIN(idtg_tg, idtg_user_buf, idtg_user_buf_count) \
  do { \
    const tuple_graph* const idtg_tg_save = (idtg_tg); \
    const size_t idtg_block_limit = ITERATE_DESTRUCTIVE_TUPLE_GRAPH_BLOCK_COUNT(idtg_tg_save); \
    size_t idtg_block_idx; \
    bool idtg_break_from_block_loop = false; \
    for (idtg_block_idx = 0; idtg_block_idx < idtg_block_limit; ++idtg_block_idx) { \
      scoped_array<packed_edge>& idtg_actual_edge_buffer = idtg_tg_save->edgememory[idtg_block_idx]; \
      ptrdiff_t const idtg_user_buf_count = (ptrdiff_t)idtg_actual_edge_buffer.size(); \
      const packed_edge* restrict const idtg_user_buf = idtg_actual_edge_buffer.get(0, idtg_actual_edge_buffer.size()); \
      assert (idtg_user_buf != NULL); \
      assert (idtg_user_buf_count >= 0); \
      assert (tuple_graph_max_bufsize(idtg_tg_save) >= idtg_user_buf_count); \
      int idtg_iteration_count = 0; \
      while (1) { \
        /* Prevent continue */ assert (idtg_iteration_count == 0); \
        {
#define ITERATE_DESTRUCTIVE_TUPLE_GRAPH_BLOCK_NUMBER (idtg_block_idx)
#define ITERATE_DESTRUCTIVE_TUPLE_GRAPH_BREAK /* Must be done collectively and before ITERATE_DESTRUCTIVE_TUPLE_GRAPH_RELEASE_BUFFER */ \
          idtg_break_from_block_loop = true; \
          break
#define ITERATE_DESTRUCTIVE_TUPLE_GRAPH_RELEASE_BUFFER \
          do { \
            idtg_actual_edge_buffer.reset(); \
          } while (0)
#define ITERATE_DESTRUCTIVE_TUPLE_GRAPH_END \
          ITERATE_DESTRUCTIVE_TUPLE_GRAPH_RELEASE_BUFFER; \
        } \
        if (idtg_break_from_block_loop) ITERATE_DESTRUCTIVE_TUPLE_GRAPH_RELEASE_BUFFER; \
        idtg_iteration_count = 1; \
        break; \
      } \
      /* Prevent user break */ assert (idtg_iteration_count == 1); \
      if (idtg_break_from_block_loop) break; \
    } \
  } while (0)

extern uint_fast32_t seed[5];
extern int SCALE; // Users cannot use this -- it is only for infrastructure components

/* Simple iteration of edge data or file; cannot be nested.  Distribution used
 * here does not necessarily match what the generator produces. */
#define REGEN_TUPLE_GRAPH_BLOCK_COUNT(tg) (size_t(((tg)->nglobaledges + FILE_CHUNKSIZE * size - 1) / (FILE_CHUNKSIZE * size)))
#define REGEN_TUPLE_GRAPH_BEGIN(tg, user_buf, user_buf_count) \
  do { \
    const tuple_graph* const tg_save = (tg); \
    const size_t block_limit = REGEN_TUPLE_GRAPH_BLOCK_COUNT(tg_save); \
    size_t block_idx; \
    memalloc("regen_tuple_graph buf", FILE_CHUNKSIZE * sizeof(packed_edge)); \
    packed_edge* const edge_data = (packed_edge*)(xmalloc(FILE_CHUNKSIZE * sizeof(packed_edge))); \
    bool break_from_block_loop = false; \
    for (block_idx = 0; block_idx < block_limit; ++block_idx) { \
      uint64_t start_edge_index = (std::min)(uint64_t(FILE_CHUNKSIZE) * (MUL_SIZE(block_idx) + rank), tg_save->nglobaledges); \
      uint64_t end_edge_index = (std::min)(start_edge_index + uint64_t(FILE_CHUNKSIZE), tg_save->nglobaledges); \
      size_t edge_count = (size_t)(end_edge_index - start_edge_index); \
      generate_kronecker_range(seed, SCALE, start_edge_index, start_edge_index + edge_count, edge_data); \
      const packed_edge* restrict const user_buf = edge_data; \
      ptrdiff_t const user_buf_count = (ptrdiff_t)edge_count; \
      assert (user_buf != NULL); \
      assert (user_buf_count >= 0); \
      assert (tuple_graph_max_bufsize(tg_save) >= user_buf_count); \
      int iteration_count = 0; \
      while (1) { \
        /* Prevent continue */ assert (iteration_count == 0); \
        {
#define REGEN_TUPLE_GRAPH_BLOCK_NUMBER (block_idx)
#define REGEN_TUPLE_GRAPH_BREAK /* Must be done collectively and before REGEN_TUPLE_GRAPH_RELEASE_BUFFER */ \
          break_from_block_loop = true; \
          break
#define REGEN_TUPLE_GRAPH_RELEASE_BUFFER \
          do {} while (0)
#define REGEN_TUPLE_GRAPH_END \
        } \
        if (break_from_block_loop) REGEN_TUPLE_GRAPH_RELEASE_BUFFER; \
        iteration_count = 1; \
        break; \
      } \
      /* Prevent user break */ assert (iteration_count == 1); \
      if (break_from_block_loop) break; \
    } \
    {memfree("regen_tuple_graph buf", FILE_CHUNKSIZE * sizeof(packed_edge)); xfree(edge_data);} \
  } while (0)

static inline int64_t tuple_graph_max_bufsize(const tuple_graph* tg) {
  (void)tg;
  return FILE_CHUNKSIZE;
}

#ifdef __cplusplus
#define EXTERN_C extern "C"
#else
#define EXTERN_C /**/
#endif

void setup_globals(void); /* In utils.c */
void cleanup_globals(void); /* In utils.c */
int lg_int64_t(int64_t x); /* In utils.c */
void* xMPI_Alloc_mem(size_t nbytes); /* In utils.c */
EXTERN_C void* xmalloc(size_t nbytes); /* In utils.c */
void* xmalloc_aligned(size_t nbytes); /* In utils.c */
void* xcalloc(size_t n, size_t unit); /* In utils.c */
void* xrealloc(void* p, size_t nbytes); /* In utils.c */
EXTERN_C void xfree(void* p); /* In utils.c */
void xMPI_Free_mem(void* p); /* In utils.c */

int validate_bfs_result(const tuple_graph* const tg, const int64_t nglobalverts, const size_t nlocalverts, const int64_t root, int64_t* const pred, int64_t* const edge_visit_count_ptr); /* In validate.c */

struct bfs_settings {
  int compression_level;
  size_t nbfs;
  bool keep_queue_stats;
  bool keep_average_level_times;
  bool official_results;
  int bcast_step_size;
};

/* Definitions in each BFS file, using static global variables for internal
 * storage: */
void make_graph_data_structure(const tuple_graph* const tg);
void free_graph_data_structure(void);
int bfs_writes_depth_map(void); /* True if high 16 bits of pred entries are zero-based level numbers, or UINT16_MAX for unreachable. */
void run_bfs(int64_t root, int64_t* pred, const bfs_settings& settings);
void get_vertex_distribution_for_pred(size_t count, const int64_t* vertices, int* owners, size_t* locals);
int64_t vertex_to_global_for_pred(int v_rank, size_t v_local); /* Only used for error messages */
size_t get_nlocalverts_for_pred(void);
MPI_Comm get_communicator_for_validation(void);

/* Chunk size for blocks of one-sided operations; a fence is inserted after (at
 * most) each CHUNKSIZE one-sided operations. */
#define CHUNKSIZE (1 << 22)
#define HALF_CHUNKSIZE ((CHUNKSIZE) / 2)

#endif /* COMMON_HPP */
