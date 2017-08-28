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

#define WCC_NOT_VISITED 18446744073709551615
#define WCC_VISITED 18446744073709551614

extern int procid, nprocs;
extern bool verbose, debug, verify, output;


int wcc_bfs(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q,
            uint64_t* wcc, uint64_t root)
{  
  if (debug) { printf("procid %d wcc_bfs() start\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  q->queue_size = 0;
  q->next_size = 0;
  q->send_size = 0;

  uint64_t root_index = get_value(&g->map, root);
  if (root_index != NULL_KEY && root_index < g->n_local)    
  {
    q->queue[0] = root;
    q->queue_size = 1;
  }

  uint64_t iter = 0;
  comm->global_queue_size = 1;
#pragma omp parallel default(shared)
{
  thread_queue_t tq;
  init_thread_queue(&tq);  

#pragma omp for
  for (uint64_t i = 0; i < g->n_local; ++i)
    if (out_degree(g, i) == 0 && in_degree(g, i) == 0)
      wcc[i] = g->local_unmap[i];
    else
      wcc[i] = WCC_NOT_VISITED;
#pragma omp for
    for (uint64_t i = g->n_local; i < g->n_total; ++i)
      wcc[i] = WCC_NOT_VISITED;

  while (comm->global_queue_size)
  {
    if (debug && tq.tid == 0) { 
      printf("Task: %d wcc_bfs() GQ: %lu, TQ: %lu\n", 
        procid, comm->global_queue_size, q->queue_size); 
    }

#pragma omp for schedule(guided) nowait
    for (uint64_t i = 0; i < q->queue_size; ++i)
    {
      uint64_t vert = q->queue[i];
      uint64_t vert_index = get_value(&g->map, vert);
      if (wcc[vert_index] == root)
        continue;
      wcc[vert_index] = root;

      uint64_t out_degree = out_degree(g, vert_index);
      uint64_t* outs = out_vertices(g, vert_index);
      for (uint64_t j = 0; j < out_degree; ++j)
      {
        uint64_t out_index = outs[j];
        if (wcc[out_index] == WCC_NOT_VISITED)
        {
          wcc[out_index] = WCC_VISITED;

          if (out_index < g->n_local)
            add_vid_to_queue(&tq, q, g->local_unmap[out_index]);
          else
            add_vid_to_send(&tq, q, out_index);
        }
      }

      uint64_t in_degree = in_degree(g, vert_index);
      uint64_t* ins = in_vertices(g, vert_index);
      for (uint64_t j = 0; j < in_degree; ++j)
      { 
        uint64_t in_index = ins[j];
        if (wcc[in_index] == WCC_NOT_VISITED)
        {
          wcc[in_index] = WCC_VISITED;

          if (in_index < g->n_local)
            add_vid_to_queue(&tq, q, g->local_unmap[in_index]);
          else
            add_vid_to_send(&tq, q, in_index);
        }
      }
    }  

    empty_queue(&tq, q);
    empty_send(&tq, q);
#pragma omp barrier

#pragma omp single
    {
      exchange_verts(g, comm, q);
      ++iter;
    }
  } // end while

  clear_thread_queue(&tq);
} // end parallel

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d wcc_bfs() time %9.6f (s)\n", procid, elt);
  }
  if (debug) { printf("Task %d wcc_bfs() success\n", procid); }

  return 0;
}


int wcc_color(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q,
              uint64_t* wcc)
{ 
  if (debug) { printf("Task %d wcc_color() start\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  q->queue_size = 0;
  q->next_size = 0;
  q->send_size = 0;

  for (int32_t i = 0; i < nprocs; ++i)
    comm->sendcounts_temp[i] = 0;

  uint64_t iter = 0;
  comm->global_queue_size = 1;
#pragma omp parallel default(shared)
{
  thread_queue_t tq;
  thread_comm_t tc;
  init_thread_queue(&tq);
  init_thread_comm(&tc);

#pragma omp for
  for (uint64_t i = 0; i < g->n_local; ++i)
  {
    if (wcc[i] == WCC_NOT_VISITED)
    {
      wcc[i] = g->local_unmap[i];
      add_vid_to_queue(&tq, q, i);
    }
  }

#pragma omp for
  for (uint64_t i = g->n_local; i < g->n_total; ++i)
    if (wcc[i] == WCC_NOT_VISITED)
      wcc[i] = g->ghost_unmap[i - g->n_local];

  empty_queue(&tq, q);
#pragma omp barrier

#pragma omp single
{
  q->queue_size = q->next_size;
  q->next_size = 0;

  uint64_t* temp = q->queue;
  q->queue = q->queue_next;
  q->queue_next = temp;
}

  while (comm->global_queue_size)
  {
    if (debug && tq.tid == 0) { 
      printf("Task %d Iter %lu wcc_color() GQ: %lu, TQ: %lu\n", 
        procid, iter, comm->global_queue_size, q->queue_size); 
    }

#pragma omp for schedule(guided) nowait
    for (uint64_t i = 0; i < q->queue_size; ++i)
    {
      uint64_t vert_index = q->queue[i];
      bool send = false;

      uint64_t out_degree = out_degree(g, vert_index);
      uint64_t* outs = out_vertices(g, vert_index);
      for (uint64_t j = 0; j < out_degree; ++j)
      {
        uint64_t out_index = outs[j];
        if (wcc[out_index] > wcc[vert_index])
        {
          wcc[vert_index] = wcc[out_index];
          send = true;
        }
      }

      uint64_t in_degree = in_degree(g, vert_index);
      uint64_t* ins = in_vertices(g, vert_index);
      for (uint64_t j = 0; j < in_degree; ++j)
      { 
        uint64_t in_index = ins[j];
        if (wcc[in_index] > wcc[vert_index])
        {
          wcc[vert_index] = wcc[in_index];
          send = true;
        }
      }

      if (send)
        add_vid_to_send(&tq, q, vert_index);
    }  

    empty_send(&tq, q);
#pragma omp barrier

    for (int32_t i = 0; i < nprocs; ++i)
      tc.sendcounts_thread[i] = 0;

#pragma omp for schedule(guided) nowait
    for (uint64_t i = 0; i < q->send_size; ++i)
    {
      uint64_t vert_index = q->queue_send[i];
      update_sendcounts_thread(g, &tc, vert_index);
    }

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
}

#pragma omp for schedule(guided) nowait
    for (uint64_t i = 0; i < q->send_size; ++i)
    {
      uint64_t vert_index = q->queue_send[i];
      update_vid_data_queues(g, &tc, comm,
                             vert_index, wcc[vert_index]);
    }

    empty_vid_data(&tc, comm);
#pragma omp barrier

#pragma omp single
{
    exchange_vert_data(g, comm, q);
} // end single


#pragma omp for
    for (uint64_t i = 0; i < comm->total_recv; ++i)
    {
      uint64_t index = get_value(&g->map, comm->recvbuf_vert[i]);
      wcc[index] = comm->recvbuf_data[i];
    }

#pragma omp single
{
    clear_allbuf_vid_data(comm);
    ++iter;
}
  }// end while

  clear_thread_queue(&tq);
  clear_thread_comm(&tc);
} // end parallel

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d, wcc_color() time %9.6f (s)\n", procid, elt);
  }
  if (debug) { printf("Task %d wcc_color() success\n", procid); }

  return 0;
}

int wcc_verify(dist_graph_t* g, uint64_t* wcc)
{
  MPI_Barrier(MPI_COMM_WORLD);

  uint64_t* counts = (uint64_t*)malloc(g->n*sizeof(uint64_t));
  uint64_t unassigned = 0;

  for (uint64_t i = 0; i < g->n; ++i)
    counts[i] = 0;
  for (uint64_t i = 0; i < g->n_local; ++i)
    if (wcc[i] != WCC_NOT_VISITED)
      ++counts[wcc[i]];
    else
      ++unassigned;

  MPI_Allreduce(MPI_IN_PLACE, counts, (int32_t)g->n, 
                MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

  uint64_t num_wccs = 0;
  uint64_t max_wcc = 0;
  for (uint64_t i = 0; i < g->n; ++i)
    if (counts[i])
    {
      ++num_wccs;
      if (counts[i] > max_wcc)
        max_wcc = counts[i];
    }

  if (procid == 0)
    printf("Num CCs: %lu, Max CC: %lu, Unassigned %lu\n",
            num_wccs, max_wcc, unassigned); 

  free(counts);

  return 0;
}


int wcc_output(dist_graph_t* g, uint64_t* wcc, char* output_file)
{
  if (verbose) printf("Task %d wcc assignments to %s\n", procid, output_file); 

  uint64_t* global_wcc = (uint64_t*)malloc(g->n*sizeof(uint64_t));
  
#pragma omp parallel for
  for (uint64_t i = 0; i < g->n; ++i)
    global_wcc[i] = WCC_NOT_VISITED;

#pragma omp parallel for
  for (uint64_t i = 0; i < g->n_local; ++i)
    global_wcc[g->local_unmap[i]] = wcc[i];

  if (procid == 0)
    MPI_Reduce(MPI_IN_PLACE, global_wcc, (int32_t)g->n,
      MPI_UINT64_T, MPI_MIN, 0, MPI_COMM_WORLD);
  else
    MPI_Reduce(global_wcc, global_wcc, (int32_t)g->n,
      MPI_UINT64_T, MPI_MIN, 0, MPI_COMM_WORLD);

  if (procid == 0)
  {
    if (debug)
      for (uint64_t i = 0; i < g->n; ++i)
        if (global_wcc[i] == WCC_NOT_VISITED)
        {
          printf("WCC error: %lu not assigned\n", i);
          global_wcc[i] = 0;
        }
        
    std::ofstream outfile;
    outfile.open(output_file);

    for (uint64_t i = 0; i < g->n; ++i)
      outfile << global_wcc[i] << std::endl;

    outfile.close();
  }

  free(global_wcc);

  if (verbose) printf("Task %d done writing assignments\n", procid); 

  return 0;
}

int wcc_dist(dist_graph_t *g, mpi_data_t* comm, queue_data_t* q, 
             uint64_t root, char* output_file)
{  
  if (debug) { printf("Task %d wcc_dist() start\n", procid); }

  MPI_Barrier(MPI_COMM_WORLD);
  double elt = omp_get_wtime();

  uint64_t* wcc = (uint64_t*)malloc(g->n_total*sizeof(uint64_t));
  wcc_bfs(g, comm, q, wcc, root);
  wcc_color(g, comm, q, wcc);
  
  MPI_Barrier(MPI_COMM_WORLD);
  elt = omp_get_wtime() - elt;
  if (procid == 0) printf("WCC time %9.6f (s)\n", elt);

  if (output) {
    wcc_output(g, wcc, output_file);
  }

  if (verify) { 
    wcc_verify(g, wcc);
  }

  free(wcc);

  if (debug)  printf("Task %d wcc_dist() success\n", procid); 
  return 0;
}

