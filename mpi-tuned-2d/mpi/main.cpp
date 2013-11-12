/* Copyright (C) 2010-2013 The Trustees of Indiana University.             */
/*                                                                         */
/* Use, modification and distribution is subject to the Boost Software     */
/* License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at */
/* http://www.boost.org/LICENSE_1_0.txt)                                   */
/*                                                                         */
/*  Authors: Jeremiah Willcock                                             */
/*           Andrew Lumsdaine                                              */

/* These need to be before any possible inclusions of stdint.h or inttypes.h.
 * */
#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
#endif
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif

#include "../generator/make_graph.h"
#include "../generator/utils.h"
// #include "/home/jewillco/mpe-install/include/mpe.h"
#include "common.hpp"
#include "bitmap.hpp"
#include "onesided.hpp"
#include <math.h>
#include <mpi.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <stdarg.h>
#include <limits.h>
#include <stdint.h>
#include <inttypes.h>
#include <algorithm>
#include <execinfo.h>
#include <unistd.h>
#include <iostream>
#include <iomanip>

#if defined(CRAYPAT) && 0
#include <pat_api.h>
#warning "CrayPAT on"
#endif

static int compare_doubles(const void* a, const void* b) {
  double aa = *(const double*)a;
  double bb = *(const double*)b;
  return (aa < bb) ? -1 : (aa == bb) ? 0 : 1;
}

enum {s_minimum, s_firstquartile, s_median, s_thirdquartile, s_maximum, s_mean, s_std, s_LAST};
static void get_statistics(const double x[], int n, double r[s_LAST]) {
  double temp;
  /* Compute mean. */
  temp = 0.;
  for (int i = 0; i < n; ++i) temp += x[i];
  temp /= n;
  r[s_mean] = temp;
  /* Compute std. dev. */
  temp = 0.;
  for (int i = 0; i < n; ++i) temp += (x[i] - r[s_mean]) * (x[i] - r[s_mean]);
  temp /= double(n - 1);
  r[s_std] = sqrt(temp);
  /* Sort x. */
  double* xx = (double*)xmalloc(n * sizeof(double));
  memcpy(xx, x, n * sizeof(double));
  qsort(xx, n, sizeof(double), compare_doubles);
  /* Get order statistics. */
  r[s_minimum] = xx[0];
  r[s_firstquartile] = (xx[(n - 1) / 4] + xx[n / 4]) * .5;
  r[s_median] = (xx[(n - 1) / 2] + xx[n / 2]) * .5;
  r[s_thirdquartile] = (xx[n - 1 - (n - 1) / 4] + xx[n - 1 - n / 4]) * .5;
  r[s_maximum] = xx[n - 1];
  /* Clean up. */
  xfree(xx);
}

void do_trace(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void do_trace(const char* fmt, ...) {
  (void)fmt;
#if 0
  if (rank != 0) return;
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fflush(stderr);
  void* trace[20];
  int len = backtrace(trace, 20), i;
  fprintf(stderr, "Trace length %d\n", len);
  for (i = 0; i < len; ++i) fprintf(stderr, "%d: %p\n", i, trace[i]);
  backtrace_symbols_fd(trace, len, 2);
  fsync(2);
#endif
}

std::ostream& operator<<(std::ostream& os, const bfs_settings& s) {
  os << "compression=" << s.compression_level << " nbfs=" << s.nbfs << " keep_queue_stats=" << std::boolalpha << s.keep_queue_stats << " keep_average_level_times=" << s.keep_average_level_times << " official_results=" << s.official_results << " bcast_step_size=" << s.bcast_step_size;
  return os;
}

uint_fast32_t seed[5]; // These two are used in common.hpp for graph regen
int SCALE;

int main(int argc, char** argv) {
#ifdef _OPENMP
  int thr_wanted;
#if 0
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &thr_wanted);
  if (!(thr_wanted >= MPI_THREAD_SERIALIZED)) {
    if (rank == 0) {
      fprintf(stderr, "When compiled with OpenMP, this code requires MPI_THREAD_SERIALIZED\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 50);
  }
#endif
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thr_wanted);
  if (!(thr_wanted >= MPI_THREAD_MULTIPLE)) {
    if (rank == 0) {
      fprintf(stderr, "When compiled with OpenMP, this code requires MPI_THREAD_MULTIPLE\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 50);
  }
#else
  MPI_Init(&argc, &argv);
#endif
#ifdef _OPENMP
#pragma omp parallel
  fprintf(stderr, "%s", ""); // Make sure threads run
#endif
  // MPE_Stop_log();

#if defined(CRAYPAT) && 0
  PAT_record(PAT_STATE_OFF);
#endif

  setup_globals();

  /* Parse arguments. */
  SCALE = 26;
  int edgefactor = 16; /* nedges / nvertices, i.e., 2*avg. degree */
  if (argc >= 2) SCALE = atoi(argv[1]);
  if (argc >= 3) edgefactor = atoi(argv[2]);
  if (argc <= 1 || argc >= 4 || SCALE == 0 || edgefactor == 0) {
    if (rank == 0) {
      fprintf(stderr, "Usage: %s SCALE edgefactor\n  SCALE = log_2(# vertices) [integer, required]\n  edgefactor = (# edges) / (# vertices) = .5 * (average vertex degree) [integer, defaults to 16]\n(Random number seed and Kronecker initiator are in main.c)\n", argv[0]);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  uint64_t seed1 = 2, seed2 = 3;

  tuple_graph tg;
  tg.nglobaledges = (uint64_t)(edgefactor) << SCALE;
  uint64_t nglobalverts = (uint64_t)(1) << SCALE;

  /* Make the raw graph edges. */
  /* Get roots for BFS runs, plus maximum vertex with non-zero degree (used by
   * validator). */
  unsigned int num_bfs_roots = 64;

  do_trace("Before graph_generation\n");

  double make_graph_start = MPI_Wtime();
  scoped_array<uint64_t> bfs_roots;
  uint64_t max_used_vertex = 0;
  {
    /* Spread the two 64-bit numbers into five nonzero values in the correct
     * range. */
    make_mrg_seed(seed1, seed2, seed);

    /* Generate the graph. */
    size_t block_limit = size_t(DIV_SIZE((tg.nglobaledges + size * FILE_CHUNKSIZE - 1) / FILE_CHUNKSIZE));
    tg.edgememory.reset_to_new(block_limit);
    for (size_t block_idx = 0; block_idx < block_limit; ++block_idx) {
      if (rank == 0) fprintf(stderr, "%d: Generating block %d of %d\n", rank, (int)block_idx, (int)block_limit);
      uint64_t start_edge_index = (std::min)(uint64_t(FILE_CHUNKSIZE) * (MUL_SIZE(block_idx) + rank), tg.nglobaledges);
      size_t edge_count = size_t((std::min)(tg.nglobaledges - start_edge_index, uint64_t(FILE_CHUNKSIZE)));
      assert (start_edge_index <= tg.nglobaledges);
      assert (edge_count <= size_t(FILE_CHUNKSIZE));
      scoped_array<packed_edge>& buf = tg.edgememory[block_idx];
      buf.reset_to_new(edge_count);
      generate_kronecker_range(seed, SCALE, start_edge_index, start_edge_index + edge_count, buf.get(0, edge_count));
    }

    /* Find root vertices and the largest vertex number with incident edges.
     * We first generate a larger number of candidate roots, then scan the
     * edges to find which of them have incident edges.  Hopefully enough do;
     * otherwise, we need to iterate until we find them.  */
    do_trace("Before finding roots\n");
    {
      const unsigned int candidates_to_scan_per_iter = 4 * num_bfs_roots;

      bfs_roots.reset_to_new(num_bfs_roots);
      unsigned int bfs_root_idx = 0;
      scoped_array<uint64_t> candidate_roots;
      candidate_roots.reset_to_new(candidates_to_scan_per_iter);
      scoped_array<int> candidate_root_marks;
      candidate_root_marks.reset_to_new(candidates_to_scan_per_iter);
      unsigned int pass_num = 0;
      while (bfs_root_idx < (unsigned int)(num_bfs_roots)) {
        ++pass_num;
        if (rank == 0) fprintf(stderr, "Finding roots, pass %u\n", pass_num);
        for (unsigned int i = 0; i < candidates_to_scan_per_iter; ++i) {
          double d[2];
          make_random_numbers(2, seed1, seed2, 2 * (pass_num * candidates_to_scan_per_iter + i), d);
          uint64_t root = uint64_t((d[0] + d[1]) * double(nglobalverts)) % nglobalverts;
          assert (root < nglobalverts);
          candidate_roots[i] = root;
          candidate_root_marks[i] = 0;
        }
        ITERATE_TUPLE_GRAPH_BEGIN(&tg, edgebuf, edgebuf_size) {
          if (rank == 0) fprintf(stderr, "Block %zu of %zu\n", size_t(ITERATE_TUPLE_GRAPH_BLOCK_NUMBER), size_t(ITERATE_TUPLE_GRAPH_BLOCK_COUNT(&tg)));
          // Global max code from http://www.openmp.org/pipermail/omp/2005/000257.html
#pragma omp parallel
          {
            uint64_t my_max_used_vertex = max_used_vertex;
#pragma omp for
            for (ptrdiff_t i = 0; i < edgebuf_size; ++i) {
              uint64_t src = uint64_t(get_v0_from_edge(&edgebuf[i]));
              uint64_t tgt = uint64_t(get_v1_from_edge(&edgebuf[i]));
              if (src == tgt) continue;
              for (unsigned int j = 0; j < candidates_to_scan_per_iter; ++j) {
                if (candidate_roots[j] == src || candidate_roots[j] == tgt) {
#pragma omp atomic
                  candidate_root_marks[j] |= true;
                }
                if (src > my_max_used_vertex) {
                  my_max_used_vertex = src;
                }
                if (tgt > my_max_used_vertex) {
                  my_max_used_vertex = tgt;
                }
              }
            }
#pragma omp critical
            {
              if (my_max_used_vertex > max_used_vertex) max_used_vertex = my_max_used_vertex;
            }
          }
        } ITERATE_TUPLE_GRAPH_END;
        MPI_Allreduce(MPI_IN_PLACE, candidate_root_marks.get(0, candidates_to_scan_per_iter), candidates_to_scan_per_iter, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &max_used_vertex, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
        /* This control flow is common to all nodes. */
        for (unsigned int i = 0; i < candidates_to_scan_per_iter; ++i) {
          if (candidate_root_marks[i] == 1) {
            bool is_duplicate = false;
            for (unsigned int j = 0; j < bfs_root_idx; ++j) {
              if (candidate_roots[i] == bfs_roots[j]) {
                is_duplicate = true;
                break;
              }
            }
            if (!is_duplicate) {
              bfs_roots[bfs_root_idx++] = candidate_roots[i];
              if (bfs_root_idx >= (unsigned int)(num_bfs_roots)) break;
            }
          }
        }
      }
    }
  }

  double make_graph_stop = MPI_Wtime();
  double make_graph_time = make_graph_stop - make_graph_start;
  if (rank == 0) { /* Not an official part of the results */
    fprintf(stderr, "graph_generation:               %f s\n", make_graph_time);
  }

  do_trace("After graph_generation\n");

  // int begin_bfs_event = MPE_Log_get_event_number();
  // int end_bfs_event = MPE_Log_get_event_number();
  // if (rank == 0) MPE_Describe_state(begin_bfs_event, end_bfs_event, "BFS", "red");

  /* Make user's graph data structure. */
  double data_struct_start = MPI_Wtime();
  make_graph_data_structure(&tg);
  double data_struct_stop = MPI_Wtime();
  double data_struct_time = data_struct_stop - data_struct_start;
  if (rank == 0) { /* Not an official part of the results */
    fprintf(stderr, "construction_time:              %f s\n", data_struct_time);
  }

  do_trace("After user graph data structure construction\n");

  tg.edgememory.reset();
  do_trace("After cleanup\n");

  /* Number of edges visited in each BFS; a double so get_statistics can be
   * used directly. */
  double* edge_counts = (double*)xmalloc(num_bfs_roots * sizeof(double));

  /* Run BFS. */
  int validation_passed = 1;
  double* bfs_times = (double*)xmalloc(num_bfs_roots * sizeof(double));
  double* validate_times = (double*)xmalloc(num_bfs_roots * sizeof(double));
  uint64_t nlocalverts = get_nlocalverts_for_pred();
  assert (nlocalverts * sizeof(int64_t) <= SIZE_MAX);
  memalloc("pred", size_t(nlocalverts) * sizeof(int64_t));
  int64_t* pred = (int64_t*)xMPI_Alloc_mem(size_t(nlocalverts) * sizeof(int64_t));

  // const int default_bcast_step_size = 8;

  const bfs_settings settings[] = {
    // {2, num_bfs_roots, false, false, true, default_bcast_step_size},
    {3, 1, false, true, false, 4},
    {3, 1, false, true, false, 6},
    {3, 1, false, true, false, 8},
    {3, 1, false, true, false, 12},
    {3, 1, false, true, false, 16},
    {3, 1, false, true, false, 24},
    {3, 1, false, true, false, 32},
    {3, 1, false, true, false, 48},
    {3, 1, false, true, false, 64},
    {3, 1, false, true, false, 96},
    {3, 1, false, true, false, 128},
    {3, 1, false, true, false, 192},
    {3, 1, false, true, false, 256},
    {3, num_bfs_roots, false, false, true, -1},
    {3, 1, true, true, false, -1},
    // {2, 1, false, false, false, -1},
    // {1, 1, false, true, false, -1},
    // {0, 1, false, true, false, -1},
  };
  bfs_settings best_settings = settings[0];
  double best_teps = 0.;

  for (size_t i = 0; i < sizeof(settings) / sizeof(*settings); ++i) {

    bfs_settings s = settings[i];
    if (s.nbfs > (size_t)num_bfs_roots) {
      if (rank == 0) fprintf(stderr, "Root count %zu in settings too large for %u generated roots\n", s.nbfs, num_bfs_roots);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (s.bcast_step_size == -1) s.bcast_step_size = best_settings.bcast_step_size;

    for (int bfs_root_idx = 0; bfs_root_idx < (int)s.nbfs; ++bfs_root_idx) {

      int64_t root = bfs_roots[bfs_root_idx];

      if (rank == 0) fprintf(stderr, "Running settings %d BFS %d\n", (int)i, bfs_root_idx);

      /* Clear the pred array. */
      memset(pred, 0, size_t(nlocalverts) * sizeof(int64_t));

      do_trace("Before BFS %d\n", bfs_root_idx);

      /* Do the actual BFS. */
      // MPE_Start_log();
      // MPE_Log_event(begin_bfs_event, 0, "Start of BFS");
      MPI_Barrier(MPI_COMM_WORLD);
#if defined(CRAYPAT) && 0
      if (s.official_results) PAT_record(PAT_STATE_ON);
#endif
      double bfs_start = MPI_Wtime();
      run_bfs(root, &pred[0], s);
      MPI_Barrier(MPI_COMM_WORLD);
      // MPE_Stop_log();
      // MPE_Log_event(end_bfs_event, 0, "End of BFS");
      double bfs_stop = MPI_Wtime();
#if defined(CRAYPAT) && 0
      if (s.official_results) PAT_record(PAT_STATE_OFF);
#endif
      bfs_times[bfs_root_idx] = bfs_stop - bfs_start;
      if (rank == 0) fprintf(stderr, "Time for BFS %d is %f\n", bfs_root_idx, bfs_times[bfs_root_idx]);

      do_trace("After BFS %d\n", bfs_root_idx);

#if 1
      /* Validate result. */
      if (rank == 0) fprintf(stderr, "Validating BFS %d\n", bfs_root_idx);

      double validate_start = MPI_Wtime();
      int64_t edge_visit_count;
      // fprintf(stderr, "%d: validate_bfs_result(nglobalverts = %" PRId64 ", nlocalverts = %" PRIu64 ", root = %" PRId64 ")\n", rank, max_used_vertex + 1, nlocalverts, root);
      int validation_passed_one = validate_bfs_result(&tg, max_used_vertex + 1, size_t(nlocalverts), root, pred, &edge_visit_count);
      double validate_stop = MPI_Wtime();

      do_trace("After validating BFS %d\n", bfs_root_idx);

      validate_times[bfs_root_idx] = validate_stop - validate_start;
      if (rank == 0) fprintf(stderr, "Validate time for BFS %d is %f\n", bfs_root_idx, validate_times[bfs_root_idx]);
      edge_counts[bfs_root_idx] = (double)edge_visit_count;
      if (rank == 0) {
        std::cerr << "Settings: " << s << std::endl;
        fprintf(stderr, "TEPS for BFS %d is %g\n", bfs_root_idx, double(edge_visit_count) / bfs_times[bfs_root_idx]);
      }

      if (!validation_passed_one) {
        validation_passed = 0;
        if (rank == 0) fprintf(stderr, "Validation failed for this BFS root; aborting.\n");
        abort();
      }
#endif

      if (double(edge_visit_count) / bfs_times[bfs_root_idx] > best_teps) {
        best_settings = s;
        best_teps = double(edge_visit_count) / bfs_times[bfs_root_idx];
      }
    }

    /* Print results. */
    if (rank == 0 && s.official_results) {
      if (!validation_passed) {
        fprintf(stdout, "No results printed for invalid run.\n");
      } else {
        fprintf(stdout, "compression_level:              %d\n", (int)s.compression_level);
        fprintf(stdout, "SCALE:                          %d\n", SCALE);
        fprintf(stdout, "edgefactor:                     %d\n", edgefactor);
        fprintf(stdout, "NBFS:                           %zu\n", s.nbfs);
        fprintf(stdout, "graph_generation:               %g\n", make_graph_time);
        fprintf(stdout, "num_mpi_processes:              %d\n", size);
        fprintf(stdout, "construction_time:              %g\n", data_struct_time);
        double stats[s_LAST];
        get_statistics(bfs_times, s.nbfs, stats);
        fprintf(stdout, "min_time:                       %g\n", stats[s_minimum]);
        fprintf(stdout, "firstquartile_time:             %g\n", stats[s_firstquartile]);
        fprintf(stdout, "median_time:                    %g\n", stats[s_median]);
        fprintf(stdout, "thirdquartile_time:             %g\n", stats[s_thirdquartile]);
        fprintf(stdout, "max_time:                       %g\n", stats[s_maximum]);
        fprintf(stdout, "mean_time:                      %g\n", stats[s_mean]);
        fprintf(stdout, "stddev_time:                    %g\n", stats[s_std]);
        get_statistics(edge_counts, s.nbfs, stats);
        fprintf(stdout, "min_nedge:                      %.11g\n", stats[s_minimum]);
        fprintf(stdout, "firstquartile_nedge:            %.11g\n", stats[s_firstquartile]);
        fprintf(stdout, "median_nedge:                   %.11g\n", stats[s_median]);
        fprintf(stdout, "thirdquartile_nedge:            %.11g\n", stats[s_thirdquartile]);
        fprintf(stdout, "max_nedge:                      %.11g\n", stats[s_maximum]);
        fprintf(stdout, "mean_nedge:                     %.11g\n", stats[s_mean]);
        fprintf(stdout, "stddev_nedge:                   %.11g\n", stats[s_std]);
        double* secs_per_edge = (double*)xmalloc(s.nbfs * sizeof(double));
        for (size_t i = 0; i < s.nbfs; ++i) secs_per_edge[i] = bfs_times[i] / edge_counts[i];
        get_statistics(secs_per_edge, s.nbfs, stats);
        fprintf(stdout, "min_TEPS:                       %g\n", 1. / stats[s_maximum]);
        fprintf(stdout, "firstquartile_TEPS:             %g\n", 1. / stats[s_thirdquartile]);
        fprintf(stdout, "median_TEPS:                    %g\n", 1. / stats[s_median]);
        fprintf(stdout, "thirdquartile_TEPS:             %g\n", 1. / stats[s_firstquartile]);
        fprintf(stdout, "max_TEPS:                       %g\n", 1. / stats[s_minimum]);
        fprintf(stdout, "harmonic_mean_TEPS:             %g\n", 1. / stats[s_mean]);
        /* Formula from:
         * Title: The Standard Errors of the Geometric and Harmonic Means and
         *        Their Application to Index Numbers
         * Author(s): Nilan Norris
         * Source: The Annals of Mathematical Statistics, Vol. 11, No. 4 (Dec., 1940), pp. 445-448
         * Publisher(s): Institute of Mathematical Statistics
         * Stable URL: http://www.jstor.org/stable/2235723
         * (same source as in specification). */
        fprintf(stdout, "harmonic_stddev_TEPS:           %g\n", stats[s_std] / (stats[s_mean] * stats[s_mean] * sqrt(s.nbfs - 1)));
        xfree(secs_per_edge); secs_per_edge = NULL;
        get_statistics(validate_times, s.nbfs, stats);
        fprintf(stdout, "min_validate:                   %g\n", stats[s_minimum]);
        fprintf(stdout, "firstquartile_validate:         %g\n", stats[s_firstquartile]);
        fprintf(stdout, "median_validate:                %g\n", stats[s_median]);
        fprintf(stdout, "thirdquartile_validate:         %g\n", stats[s_thirdquartile]);
        fprintf(stdout, "max_validate:                   %g\n", stats[s_maximum]);
        fprintf(stdout, "mean_validate:                  %g\n", stats[s_mean]);
        fprintf(stdout, "stddev_validate:                %g\n", stats[s_std]);
#if 0
        for (int i = 0; i < s.nbfs; ++i) {
          fprintf(stdout, "Run %3d:                        %g s, validation %g s\n", i + 1, bfs_times[i], validate_times[i]);
        }
#endif
        fprintf(stdout, "\n");
      }
    }
  }
  memfree("pred", size_t(nlocalverts) * sizeof(int64_t));
  xfree(edge_counts); edge_counts = NULL;
  xMPI_Free_mem(pred);
  bfs_roots.reset();
  free_graph_data_structure();

  xfree(bfs_times);
  xfree(validate_times);

  cleanup_globals();
  MPI_Finalize();
  return 0;
}
