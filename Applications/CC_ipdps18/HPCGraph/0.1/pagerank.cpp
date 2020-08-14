/*
//@HEADER
// *****************************************************************************
//
//  HPCGraph: Graph Computation on High Performance Computing Systems
//              Copyright (2016) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions?  Contact  George M. Slota   (gmslota@sandia.gov)
//                      Siva Rajamanickam (srajama@sandia.gov)
//                      Kamesh Madduri    (madduri@cse.psu.edu)
//
// *****************************************************************************
//@HEADER
*/


#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fstream>

#include "dist_graph.h"
#include "comms.h"
#include "util.h"
#include "pagerank.h"

#define DAMPING_FACTOR 0.85

extern int procid, nprocs;
extern bool verbose, debug, verify;


int run_pagerank(dist_graph_t* g, mpi_data_t* comm, 
                 double*& pageranks, uint32_t num_iter)
{ 
  if (debug) { printf("Task %d run_pagerank() start\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  double* pageranks_next = (double*)malloc(g->n_total*sizeof(double));
  double sum_noouts = 0.0;
  double sum_noouts_next = 0.0;

  for (int32_t i = 0; i < nprocs; ++i)
    comm->sendcounts_temp[i] = 0;

  comm->global_queue_size = 1;
#pragma omp parallel default(shared)
{
  thread_comm_t tc;
  init_thread_comm_flt(&tc);

#pragma omp for reduction(+:sum_noouts_next)
  for (uint64_t i = 0; i < g->n_local; ++i)
  {
    pageranks[i] = (1.0 / (double)g->n);
    uint64_t out_degree = out_degree(g, i);
    if (out_degree > 0)
      pageranks[i] /= (double)out_degree;
    else
    {
      pageranks[i] /= (double)g->n;
      sum_noouts_next += pageranks[i];
    }
  }
#pragma omp for
  for (uint64_t i = g->n_local; i < g->n_total; ++i)
    pageranks[i] = (1.0 / (double)g->n) / (double)g->n;
#pragma omp for
  for (uint64_t i = 0; i < g->n_total; ++i)    
    pageranks_next[i] = pageranks[i];

#pragma omp for schedule(guided) nowait
  for (uint64_t i = 0; i < g->n_local; ++i)
    update_sendcounts_thread_out(g, &tc, i);

  for (int32_t i = 0; i < nprocs; ++i)
  {
#pragma omp atomic
    comm->sendcounts_temp[i] += tc.sendcounts_thread[i];

    tc.sendcounts_thread[i] = 0;
  }
#pragma omp barrier

#pragma omp single
{
  init_sendbuf_vid_data_flt(comm);
  init_recvbuf_vid_data_flt(comm);
}

#pragma omp for schedule(guided) nowait
  for (uint64_t i = 0; i < g->n_local; ++i)
    update_vid_data_queues_out(g, &tc, comm, i, pageranks[i]);

  empty_vid_data_flt(&tc, comm);
#pragma omp barrier

#pragma omp single
{
  exchange_verts(comm);
  exchange_data_flt(comm);

  sum_noouts = 0.0;
  MPI_Allreduce(&sum_noouts_next, &sum_noouts, 1, 
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  sum_noouts_next = 0.0;
}

#pragma omp for
  for (uint64_t i = 0; i < comm->total_recv; ++i)
  {
    uint64_t index = get_value(&g->map, comm->recvbuf_vert[i]);
    pageranks[index] = comm->recvbuf_data_flt[i];
    comm->recvbuf_vert[i] = index;
  }

#pragma omp for
  for (uint64_t i = 0; i < comm->total_send; ++i)
  {
    uint64_t index = get_value(&g->map, comm->sendbuf_vert[i]);
    comm->sendbuf_vert[i] = index;
  } 

  for (uint32_t iter = 0; iter < num_iter; ++iter)
  {
    if (debug && tc.tid == 0) {
      printf("Task %d Iter %u run_pagerank() sink contribution sum %e\n", procid, iter, sum_noouts); 
    }

#pragma omp for schedule(guided) reduction(+:sum_noouts_next) nowait
    for (uint64_t i = 0; i < g->n_local; ++i)
    {
      uint64_t vert_index = i;
      double vert_pagerank = sum_noouts;

      uint64_t in_degree = in_degree(g, vert_index);
      uint64_t* ins = in_vertices(g, vert_index);
      for (uint64_t j = 0; j < in_degree; ++j)
        vert_pagerank += pageranks[ins[j]];

      vert_pagerank *= DAMPING_FACTOR;
      vert_pagerank += ((1.0 - DAMPING_FACTOR) / (double)g->n);

      uint64_t out_degree = out_degree(g, vert_index);
      if (out_degree > 0)
        vert_pagerank /= (double)out_degree;
      else
      {
        vert_pagerank /= (double)g->n;
        sum_noouts_next += vert_pagerank;
      }

      pageranks_next[vert_index] = vert_pagerank;
    }  

#pragma omp for
    for (uint64_t i = 0; i < comm->total_send; ++i)
      comm->sendbuf_data_flt[i] = pageranks_next[comm->sendbuf_vert[i]];

#pragma omp single
{
    exchange_data_flt(comm);
    sum_noouts = 0.0;
    MPI_Allreduce(&sum_noouts_next, &sum_noouts, 1, 
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    sum_noouts_next = 0.0;
}

#pragma omp for
    for (uint64_t i = 0; i < comm->total_recv; ++i)
      pageranks_next[comm->recvbuf_vert[i]] = comm->recvbuf_data_flt[i];

#pragma omp single
{
    double* temp = pageranks;
    pageranks = pageranks_next;
    pageranks_next = temp;
}
  } // end for loop

  clear_thread_comm(&tc);
} // end parallel

  clear_allbuf_vid_data(comm);
  free(pageranks_next);

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d, run_pagerank() time %9.6f (s)\n", procid, elt);
  }
  if (debug) { printf("Task %d run_pagerank() success\n", procid); }

  return 0;
}


int pagerank_output(dist_graph_t* g, double* pageranks, char* output_file)
{
  if (debug) printf("Task %d pageranks to %s\n", procid, output_file); 

  double* global_pageranks = (double*)malloc(g->n*sizeof(double));
  
#pragma omp parallel for
  for (uint64_t i = 0; i < g->n; ++i)
    global_pageranks[i] = -1.0;

#pragma omp parallel for
  for (uint64_t i = 0; i < g->n_local; ++i)
  {
    uint64_t out_degree = out_degree(g, i);
    assert(g->local_unmap[i] < g->n);
    if (out_degree > 0)
      global_pageranks[g->local_unmap[i]] = pageranks[i] * (double)out_degree;
    else
      global_pageranks[g->local_unmap[i]] = pageranks[i] * (double)g->n;
  }

  if (procid == 0)
    MPI_Reduce(MPI_IN_PLACE, global_pageranks, (int32_t)g->n,
      MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  else
    MPI_Reduce(global_pageranks, global_pageranks, (int32_t)g->n,
      MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (procid == 0)
  {
    if (debug)
      for (uint64_t i = 0; i < g->n; ++i)
        if (global_pageranks[i] == -1.0)
        {
          printf("Pagerank error: %lu not assigned\n", i);
          global_pageranks[i] = 0.0;
        }
        
    std::ofstream outfile;
    outfile.open(output_file);

    for (uint64_t i = 0; i < g->n; ++i)
      outfile << global_pageranks[i] << std::endl;

    outfile.close();
  }

  free(global_pageranks);

  if (debug) printf("Task %d done writing pageranks\n", procid); 

  return 0;
}


int pagerank_verify(dist_graph_t* g, double* pageranks)
{
  if (debug) { printf("Task %d pagerank_verify() start\n", procid); }

  double* global_pageranks = (double*)malloc(g->n*sizeof(double));
  
#pragma omp parallel for
  for (uint64_t i = 0; i < g->n; ++i)
    global_pageranks[i] = -1.0;
   
#pragma omp parallel for 
  for (uint64_t i = 0; i < g->n_local; ++i)
  {
    uint64_t out_degree = out_degree(g, i);
    if (out_degree > 0)
      global_pageranks[g->local_unmap[i]] = pageranks[i] * (double)out_degree;
    else
      global_pageranks[g->local_unmap[i]] = pageranks[i] * (double)g->n;
  }

  if (procid == 0)
    MPI_Reduce(MPI_IN_PLACE, global_pageranks, (int32_t)g->n,
      MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  else
    MPI_Reduce(global_pageranks, global_pageranks, (int32_t)g->n,
      MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (procid == 0)
  {
    double pr_sum = 0.0;
    for (uint64_t i = 0; i < g->n; ++i)
      pr_sum += global_pageranks[i];

    printf("PageRanks sum (should be 1.0): %9.6lf\n", pr_sum);
  }

  free(global_pageranks);

  if (debug) { printf("Task %d pagerank_verify() success\n", procid); }

  return 0;
}

int pagerank_dist(dist_graph_t *g, mpi_data_t* comm, 
                  uint32_t num_iter, char* output_file)
{  
  if (debug) { printf("Task %d pagerank_dist() start\n", procid); }

  MPI_Barrier(MPI_COMM_WORLD); 
  double elt = omp_get_wtime();

  double* pageranks = (double*)malloc(g->n_total*sizeof(double));
  run_pagerank(g, comm, pageranks, num_iter);

  MPI_Barrier(MPI_COMM_WORLD); 
  elt = omp_get_wtime() - elt;
  if (procid == 0) printf("PageRank time %9.6f (s)\n", elt);

  if (output) {
    pagerank_output(g, pageranks, output_file);
  }

  if (verify) { 
    pagerank_verify(g, pageranks);
  }

  free(pageranks);

  if (debug)  printf("Task %d pagerank_dist() success\n", procid); 
  return 0;
}

