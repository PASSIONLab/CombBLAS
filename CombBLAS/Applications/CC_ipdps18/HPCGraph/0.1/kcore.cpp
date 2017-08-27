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
#include "kcore.h"

extern int procid, nprocs;
extern bool verbose, debug, verify;

#define KCORE_NOT_ASSIGNED 18446744073709551615
#define MAX_ITER 10000

int run_kcore(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q,
              uint64_t* kcores, uint32_t num_iter, bool run_approx)
{  
  if (debug) { printf("Task %d run_kcore() start\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  uint64_t global_changes = g->n_local;
  uint32_t iter = 0;
  if (!run_approx) num_iter = MAX_ITER;

  for (int32_t i = 0; i < nprocs; ++i)
    comm->sendcounts_temp[i] = 0;

#pragma omp parallel default(shared)
{
  thread_comm_t tc;
  init_thread_comm(&tc);

#pragma omp for
  for (uint64_t i = 0; i < g->n_local; ++i)
    kcores[i] = out_degree(g, i) + in_degree(g, i);

#pragma omp for schedule(guided) nowait
  for (uint64_t i = 0; i < g->n_local; ++i)
    update_sendcounts_thread(g, &tc, i);

  for (int32_t i = 0; i < nprocs; ++i)
  {
#pragma omp atomic
    comm->sendcounts_temp[i] += tc.sendcounts_thread[i];

    tc.sendcounts_thread[i] = 0;
  }
#pragma omp barrier

#pragma omp single
{
  init_sendbuf_vid_data(comm);
  init_recvbuf_vid_data(comm);
}

#pragma omp for schedule(guided) nowait
  for (uint64_t i = 0; i < g->n_local; ++i)
    update_vid_data_queues(g, &tc, comm, i, kcores[i]);

  empty_vid_data(&tc, comm);
#pragma omp barrier

#pragma omp single
{
  exchange_verts(comm);
  exchange_data(comm);
}

#pragma omp for
  for (uint64_t i = 0; i < comm->total_recv; ++i)
  {
    uint64_t index = get_value(&g->map, comm->recvbuf_vert[i]);
    kcores[index] = comm->recvbuf_data[i];
    comm->recvbuf_vert[i] = index;
  }

#pragma omp for
  for (uint64_t i = 0; i < comm->total_send; ++i)
  {
    uint64_t index = get_value(&g->map, comm->sendbuf_vert[i]);
    comm->sendbuf_vert[i] = index;
  } 

  while (global_changes && iter < num_iter)
  {
    if (debug && tc.tid == 0) {
      printf("Task %d iter %lu changes %u run_kcore()\n", procid, iter, global_changes); 
    }

#pragma omp barrier
#pragma omp single
{
    global_changes = 0;
    ++iter;
}

  
#pragma omp for schedule(guided) reduction(+:global_changes) 
    for (uint64_t v = 0; v < g->n_local; ++v)
    {
      uint64_t vert_index = v;
      uint64_t vert_kcore = kcores[vert_index];
      uint64_t* counts = (uint64_t*)malloc((vert_kcore+1)*sizeof(uint64_t));
      for (uint64_t j = 0; j < (vert_kcore+1); ++j)
        counts[j] = 0;

      uint64_t out_degree = out_degree(g, vert_index);
      uint64_t* outs = out_vertices(g, vert_index);
      for (uint64_t j = 0; j < out_degree; ++j)
      {
        uint64_t out_index = outs[j];
        uint64_t kcore_out = kcores[out_index] < vert_kcore ? 
                             kcores[out_index] : vert_kcore;
        ++counts[kcore_out];
      }
      uint64_t in_degree = in_degree(g, vert_index);
      uint64_t* ins = in_vertices(g, vert_index);
      for (uint64_t j = 0; j < in_degree; ++j)
      {
        uint64_t in_index = ins[j];
        uint64_t kcore_in = kcores[in_index] < vert_kcore ? 
                            kcores[in_index] : vert_kcore;
        ++counts[kcore_in];
      }

      for (uint64_t j = vert_kcore; j > 0; --j)
        counts[j - 1] = counts[j - 1] + counts[j];

      uint64_t new_kcore = vert_kcore;
      while (new_kcore > 2 && counts[new_kcore] < new_kcore)
        --new_kcore;

      if (new_kcore != vert_kcore)
        ++global_changes;

      assert(new_kcore <= vert_kcore);

      kcores[vert_index] = new_kcore;
      free(counts);
    }

#pragma omp for
    for (uint64_t i = 0; i < comm->total_send; ++i)
      comm->sendbuf_data[i] = kcores[comm->sendbuf_vert[i]];

#pragma omp single
{
    exchange_data(comm);
    MPI_Allreduce(MPI_IN_PLACE, &global_changes, 1,
                  MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD); 
}

#pragma omp for
    for (uint64_t i = 0; i < comm->total_recv; ++i)
      kcores[comm->recvbuf_vert[i]] = comm->recvbuf_data[i];

  } // end for loop

  clear_thread_comm(&tc);
} // end parallel

  clear_allbuf_vid_data(comm);

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d, run_kcore() time %9.6f (s)\n", procid, elt);
  }
  if (debug) { printf("Task %d run_kcore() success\n", procid); }

  return 0;
}

int kcore_output(dist_graph_t* g, uint64_t* kcores, char* output_file)
{
  if (debug) printf("Task %d kcores to %s\n", procid, output_file); 

  uint64_t* global_kcores = (uint64_t*)malloc(g->n*sizeof(uint64_t));
  
#pragma omp parallel for
  for (uint64_t i = 0; i < g->n; ++i)
    global_kcores[i] = KCORE_NOT_ASSIGNED;

#pragma omp parallel for
  for (uint64_t i = 0; i < g->n_local; ++i)
    global_kcores[g->local_unmap[i]] = kcores[i];


  if (procid == 0)
    MPI_Reduce(MPI_IN_PLACE, global_kcores, (int32_t)g->n,
      MPI_UINT64_T, MPI_MIN, 0, MPI_COMM_WORLD);
  else
    MPI_Reduce(global_kcores, global_kcores, (int32_t)g->n,
      MPI_UINT64_T, MPI_MIN, 0, MPI_COMM_WORLD);

  if (procid == 0)
  {
    if (debug)
      for (uint64_t i = 0; i < g->n; ++i)
        if (global_kcores[i] == KCORE_NOT_ASSIGNED)
        {
          printf("Kcores error: %lu not assigned\n", i);
          global_kcores[i] = 0;
        }
        
    std::ofstream outfile;
    outfile.open(output_file);

    for (uint64_t i = 0; i < g->n; ++i)
      outfile << global_kcores[i] << std::endl;

    outfile.close();
  }

  free(global_kcores);

  if (debug) printf("Task %d done writing kcores\n", procid); 

  return 0;
}

int kcore_verify(dist_graph_t* g, uint64_t* kcores, uint64_t num_to_output)
{
  if (debug) { printf("Task %d kcore_verify() start\n", procid); }

  uint64_t* global_kcores = (uint64_t*)malloc(g->n*sizeof(uint64_t));
  uint64_t* kcores_counts = (uint64_t*)malloc(g->n*sizeof(uint64_t));
  
#pragma omp parallel for
  for (uint64_t i = 0; i < g->n; ++i)
    global_kcores[i] = KCORE_NOT_ASSIGNED;

#pragma omp parallel for 
  for (uint64_t i = 0; i < g->n_local; ++i)
    global_kcores[g->local_unmap[i]] = kcores[i];

  if (procid == 0)
    MPI_Reduce(MPI_IN_PLACE, global_kcores, (int32_t)g->n,
      MPI_UINT64_T, MPI_MIN, 0, MPI_COMM_WORLD);
  else
    MPI_Reduce(global_kcores, global_kcores, (int32_t)g->n,
      MPI_UINT64_T, MPI_MIN, 0, MPI_COMM_WORLD);

  if (procid == 0)
  {
#pragma omp parallel for
    for (uint64_t i = 0; i < g->n; ++i)
      kcores_counts[i] = 0;

    uint64_t max_k = 0;
    uint64_t max_v = 0;

    for (uint64_t i = 0; i < g->n; ++i)
    {
      ++kcores_counts[global_kcores[i]];
      if (global_kcores[i] > max_k)
      {
        max_k = global_kcores[i];
        max_v = i;
      }
    }

    printf("KC MAX K: %lu, vert: %lu\n", max_k, max_v);

    for (uint64_t i = 0; i < num_to_output; ++i)
      printf("KC VERIFY: coreness: %lu, number: %lu\n", i, kcores_counts[i]);
  }

  free(global_kcores);
  free(kcores_counts);

  if (debug) { printf("Task %d kcore_verify() success\n", procid); }

  return 0;
}

int kcore_dist(dist_graph_t *g, mpi_data_t* comm, queue_data_t* q, 
               uint32_t num_iter, char* output_file, bool run_approx)
{  
  if (debug) { printf("Task %d kcore_dist() start\n", procid); }

  MPI_Barrier(MPI_COMM_WORLD);
  double elt = omp_get_wtime();

  uint64_t* kcores = (uint64_t*)malloc(g->n_total*sizeof(uint64_t));
  run_kcore(g, comm, q, kcores, num_iter, run_approx);

  MPI_Barrier(MPI_COMM_WORLD);
  elt = omp_get_wtime() - elt;
  if (procid == 0) printf("Kcore time %9.6f (s)\n", elt);
  
  if (output) { 
    kcore_output(g, kcores, output_file);
  }

  if (verify) { 
    kcore_verify(g, kcores, 20);
  }

  free(kcores);

  if (debug)  printf("Task %d kcore_dist() success\n", procid); 
  return 0;
}

