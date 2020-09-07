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

#define SCC_NOT_VISITED 18446744073709551615
#define SCC_VISITED_FW 18446744073709551614
#define SCC_EXPLORED_FW 18446744073709551613
#define SCC_VISITED_BW 18446744073709551612
#define SCC_EXPLORED_BW 18446744073709551611
#define SCC_MARKED 18446744073709551610

extern int procid, nprocs;
extern bool verbose, debug, verify, output;


int scc_bfs_fw(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q,
               uint64_t* scc, uint64_t root)
{  
  if (debug) { printf("procid %d scc_bfs_fw() start\n", procid); }
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
    if (out_degree(g, i) == 0 || in_degree(g, i) == 0)
      scc[i] = g->local_unmap[i];
    else
      scc[i] = SCC_NOT_VISITED;
#pragma omp for
    for (uint64_t i = g->n_local; i < g->n_total; ++i)
      scc[i] = SCC_NOT_VISITED;

  while (comm->global_queue_size)
  {
    if (debug && tq.tid == 0) { 
      printf("Task: %d scc_bfs_fw() GQ: %lu, TQ: %lu\n", 
        procid, comm->global_queue_size, q->queue_size); 
    }

#pragma omp for schedule(guided) nowait
    for (uint64_t i = 0; i < q->queue_size; ++i)
    {
      uint64_t vert = q->queue[i];
      uint64_t vert_index = get_value(&g->map, vert);
      if (scc[vert_index] != SCC_NOT_VISITED &&
          scc[vert_index] != SCC_VISITED_FW)
        continue;
      scc[vert_index] = SCC_EXPLORED_FW;

      uint64_t out_degree = out_degree(g, vert_index);
      uint64_t* outs = out_vertices(g, vert_index);
      for (uint64_t j = 0; j < out_degree; ++j)
      {
        uint64_t out_index = outs[j];
        if (scc[out_index] == SCC_NOT_VISITED)
        {
          scc[out_index] = SCC_VISITED_FW;

          if (out_index < g->n_local)
            add_vid_to_queue(&tq, q, g->local_unmap[out_index]);
          else
            add_vid_to_send(&tq, q, out_index);
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
    printf("Task %d scc_bfs_fw() time %9.6f (s)\n", procid, elt);
  }
  if (debug) { printf("Task %d scc_bfs_fw() success\n", procid); }

  return 0;
}

uint64_t scc_bfs_bw(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q,
               uint64_t* scc, uint64_t root)
{  
  if (debug) { printf("procid %d scc_bfs_bw() start\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  for (int32_t i = 0; i < nprocs; ++i)
    comm->sendcounts_temp[i] = 0;

  q->queue_size = 0;
  q->next_size = 0;
  q->send_size = 0;
  uint64_t num_unassigned = 0;

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
  thread_comm_t tc;
  init_thread_queue(&tq);
  init_thread_comm(&tc);

  while (comm->global_queue_size)
  {
    if (debug && tq.tid == 0) { 
      printf("Task: %d scc_bfs_bw() GQ: %lu, TQ: %lu\n", 
        procid, comm->global_queue_size, q->queue_size); 
    }

#pragma omp for schedule(guided) nowait
    for (uint64_t i = 0; i < q->queue_size; ++i)
    {
      uint64_t vert = q->queue[i];
      uint64_t vert_index = get_value(&g->map, vert);
      if (scc[vert_index] != SCC_VISITED_BW &&
          scc[vert_index] != SCC_EXPLORED_FW)
        continue;
      scc[vert_index] = SCC_EXPLORED_BW;

      uint64_t in_degree = in_degree(g, vert_index);
      uint64_t* ins = in_vertices(g, vert_index);
      for (uint64_t j = 0; j < in_degree; ++j)
      {
        uint64_t in_index = ins[j];
        if ((in_index < g->n_local && scc[in_index] == SCC_EXPLORED_FW) ||
            (in_index >= g->n_local && scc[in_index] != SCC_VISITED_BW))
        {
          scc[in_index] = SCC_VISITED_BW;

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

#pragma omp for schedule(guided) reduction(+:num_unassigned) nowait
  for (uint64_t i = 0; i < g->n_local; ++i)
    if (scc[i] == SCC_EXPLORED_BW)
      scc[i] = root;
    else if (scc[i] == SCC_EXPLORED_FW)
    {
      scc[i] = SCC_NOT_VISITED;
      ++num_unassigned;
    }
    else if (scc[i] == SCC_NOT_VISITED)
      ++num_unassigned;

  for (int32_t i = 0; i < nprocs; ++i)
    tc.sendcounts_thread[i] = 0;

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
    update_vid_data_queues(g, &tc, comm, i, scc[i]);

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
    uint64_t vert_index = get_value(&g->map, comm->recvbuf_vert[i]);
    scc[vert_index] = comm->recvbuf_data[i];
  }

#pragma omp single
{
  clear_allbuf_vid_data(comm);
}

  clear_thread_queue(&tq);
  clear_thread_comm(&tc);
} // end parallel

  MPI_Allreduce(MPI_IN_PLACE, &num_unassigned, 1, 
                MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d scc_bfs_bw() time %9.6f (s)\n", procid, elt);
  }
  if (debug) { printf("Task %d scc_bfs_bw() success\n", procid); }

  return num_unassigned;
}


int scc_color(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q,
              uint64_t* scc, uint64_t* colors)
{ 
  if (debug) { printf("Task %d scc_color() start\n", procid); }
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
    if (scc[i] == SCC_NOT_VISITED)
    {
      colors[i] = g->local_unmap[i];
      add_vid_to_queue(&tq, q, i);
    }
    else
      colors[i] = SCC_MARKED;
  }

#pragma omp for
  for (uint64_t i = g->n_local; i < g->n_total; ++i)
    if (scc[i] == SCC_NOT_VISITED)
      colors[i] = g->ghost_unmap[i - g->n_local];
    else
      colors[i] = SCC_MARKED;

#pragma omp for
  for (uint64_t i = 0; i < g->n_total; ++i)
    if (colors[i] == SCC_MARKED && scc[i] >= g->n)
      printf("SCC assignment is %lu, out of bounds for n=%lu\n", scc[i], g->n);

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
      printf("Task %d Iter %lu scc_color() GQ: %lu, TQ: %lu\n", 
        procid, iter, comm->global_queue_size, q->queue_size); 
    }

#pragma omp for schedule(guided) nowait
    for (uint64_t i = 0; i < q->queue_size; ++i)
    {
      uint64_t vert_index = q->queue[i];
      bool send = false;

      uint64_t in_degree = in_degree(g, vert_index);
      uint64_t* ins = in_vertices(g, vert_index);
      for (uint64_t j = 0; j < in_degree; ++j)
      {
        uint64_t in_index = ins[j];
        if (colors[in_index] != SCC_MARKED &&
            colors[in_index] > colors[vert_index])
        {
          colors[vert_index] = colors[in_index];
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
                             vert_index, colors[vert_index]);
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
      uint64_t vert_index = get_value(&g->map, comm->recvbuf_vert[i]);
      colors[vert_index] = comm->recvbuf_data[i];
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
    printf("Task %d, scc_color() time %9.6f (s)\n", procid, elt);
  }
  if (debug) { printf("Task %d scc_color() success\n", procid); }

  return 0;
}



uint64_t scc_find_sccs(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q,
                  uint64_t* scc, uint64_t* colors)
{  
  if (debug) { printf("procid %d scc_find_sccs() start\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  q->queue_size = 0;
  q->next_size = 0;
  q->send_size = 0;
  uint64_t num_unassigned = 0;

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
    if (g->local_unmap[i] == colors[i])
      add_vid_to_queue(&tq, q, g->local_unmap[i]);

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
      printf("Task: %d scc_find_sccs() GQ: %lu, TQ: %lu\n", 
        procid, comm->global_queue_size, q->queue_size); 
    }

#pragma omp for schedule(guided) nowait
    for (uint64_t i = 0; i < q->queue_size; ++i)
    {
      uint64_t vert = q->queue[i];
      uint64_t vert_index = get_value(&g->map, vert);
      if (scc[vert_index] != SCC_NOT_VISITED &&
          scc[vert_index] != SCC_VISITED_FW)
        continue;
      scc[vert_index] = colors[vert_index];

      uint64_t in_degree = in_degree(g, vert_index);
      uint64_t* ins = in_vertices(g, vert_index);
      for (uint64_t j = 0; j < in_degree; ++j)
      {
        uint64_t in_index = ins[j];
        if (scc[in_index] == SCC_NOT_VISITED &&
            colors[in_index] == colors[vert_index])
        {          
          scc[in_index] = SCC_VISITED_FW;
          
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

#pragma omp for schedule(guided) reduction(+:num_unassigned) nowait
  for (uint64_t i = 0; i < g->n_local; ++i)
    if (scc[i] == SCC_EXPLORED_FW)
    {
      scc[i] = SCC_NOT_VISITED;
      ++num_unassigned;
    }
    else if (scc[i] == SCC_NOT_VISITED)
      ++num_unassigned;

  for (int32_t i = 0; i < nprocs; ++i)
    tc.sendcounts_thread[i] = 0;

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
    update_vid_data_queues(g, &tc, comm, i, scc[i]);

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
    uint64_t vert_index = get_value(&g->map, comm->recvbuf_vert[i]);
    scc[vert_index] = comm->recvbuf_data[i];
  }

#pragma omp single
{
  clear_allbuf_vid_data(comm);
}

  clear_thread_queue(&tq);
  clear_thread_comm(&tc);
} // end parallel

  MPI_Allreduce(MPI_IN_PLACE, &num_unassigned, 1, 
                MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d scc_find_sccs() time %9.6f (s)\n", procid, elt);
  }
  if (debug) { printf("Task %d scc_find_sccs() success\n", procid); }

  return num_unassigned;
}


int scc_verify(dist_graph_t* g, uint64_t* scc)
{
  MPI_Barrier(MPI_COMM_WORLD);

  uint64_t* counts = (uint64_t*)malloc(g->n*sizeof(uint64_t));
  uint64_t unassigned = 0;

  for (uint64_t i = 0; i < g->n; ++i)
    counts[i] = 0;
  for (uint64_t i = 0; i < g->n_local; ++i)
    //if (scc[i] != SCC_NOT_VISITED)
    if (scc[i] < g->n)
      ++counts[scc[i]];
    else
      ++unassigned;

  MPI_Allreduce(MPI_IN_PLACE, counts, (int32_t)g->n, 
                MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &unassigned, 1, 
                MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

  uint64_t num_sccs = 0;
  uint64_t num_trivial = 0;
  uint64_t max_scc = 0;
  for (uint64_t i = 0; i < g->n; ++i)
    if (counts[i])
    {
      ++num_sccs;
      if (counts[i] > max_scc)
        max_scc = counts[i];
      if (counts[i] == 1)
        ++num_trivial;
    }

  if (procid == 0)
    printf("Num SCCs: %lu, Max SCC: %lu, Trivial: %lu, Unassigned %lu\n",
            num_sccs, max_scc, num_trivial, unassigned); 

  free(counts);

  return 0;
}


int scc_output(dist_graph_t* g, uint64_t* scc, char* output_file)
{
  if (verbose) printf("Task %d scc assignments to %s\n", procid, output_file); 

  uint64_t* global_scc = (uint64_t*)malloc(g->n*sizeof(uint64_t));
  
#pragma omp parallel for
  for (uint64_t i = 0; i < g->n; ++i)
    global_scc[i] = SCC_NOT_VISITED;

#pragma omp parallel for
  for (uint64_t i = 0; i < g->n_local; ++i)
    global_scc[g->local_unmap[i]] = scc[i];

  if (procid == 0)
    MPI_Reduce(MPI_IN_PLACE, global_scc, (int32_t)g->n,
      MPI_UINT64_T, MPI_MIN, 0, MPI_COMM_WORLD);
  else
    MPI_Reduce(global_scc, global_scc, (int32_t)g->n,
      MPI_UINT64_T, MPI_MIN, 0, MPI_COMM_WORLD);

  if (procid == 0)
  {
    if (debug)
      for (uint64_t i = 0; i < g->n; ++i)
        if (global_scc[i] == SCC_NOT_VISITED)
        {
          printf("SCC error: %lu not assigned\n", i);
          global_scc[i] = 0;
        }
        
    std::ofstream outfile;
    outfile.open(output_file);

    for (uint64_t i = 0; i < g->n; ++i)
      outfile << global_scc[i] << std::endl;

    outfile.close();
  }

  free(global_scc);

  if (verbose) printf("Task %d done writing assignments\n", procid); 

  return 0;
}

int scc_dist(dist_graph_t *g, mpi_data_t* comm, queue_data_t* q, 
             uint64_t root, char* output_file)
{  
  if (debug) { printf("Task %d scc_dist() start\n", procid); }

  MPI_Barrier(MPI_COMM_WORLD);
  double elt = omp_get_wtime();

  uint64_t* scc = (uint64_t*)malloc(g->n_total*sizeof(uint64_t));
  uint64_t* colors = (uint64_t*)malloc(g->n_total*sizeof(uint64_t));
  uint64_t num_unassigned = 0;
  scc_bfs_fw(g, comm, q, scc, root);
  num_unassigned = scc_bfs_bw(g, comm, q, scc, root);
  
  while (num_unassigned)
  {
    scc_color(g, comm, q, scc, colors);
    num_unassigned = scc_find_sccs(g, comm, q, scc, colors);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  elt = omp_get_wtime() - elt;
  if (procid == 0) printf("SCC time %9.6f (s)\n", elt);

  if (output) {
    scc_output(g, scc, output_file);
  }

  if (verify) { 
    scc_verify(g, scc);
  }

  free(scc);

  if (debug)  printf("Task %d scc_dist() success\n", procid); 
  return 0;
}

