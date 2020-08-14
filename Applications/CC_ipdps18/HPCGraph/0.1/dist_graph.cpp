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
#include <string.h>
#include <fstream>

#include "fast_map.h"
#include "dist_graph.h"
#include "util.h"
#include "comms.h"

extern int procid, nprocs;
extern bool verbose, debug, verify, output;

int create_graph(graph_gen_data_t *ggi, dist_graph_t *g)
{  
  if (debug) { printf("Task %d create_graph() start\n", procid); }

  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  g->n = ggi->n;
  g->n_local = ggi->n_local;
  g->n_offset = ggi->n_offset;
  g->m = ggi->m;
  g->m_local_out = ggi->m_local_out;
  g->m_local_in = ggi->m_local_in;
  init_map(&g->map);

  uint64_t* out_edges = (uint64_t*)malloc(g->m_local_out*sizeof(uint64_t));
  uint64_t* out_degree_list = (uint64_t*)malloc((g->n_local+1)*sizeof(uint64_t));
  uint64_t* temp_counts = (uint64_t*)malloc(g->n_local*sizeof(uint64_t));
  if (out_edges == NULL || out_degree_list == NULL || temp_counts == NULL)
    throw_err("create_graph(), unable to allocate out edge storage", procid);

#pragma omp parallel
{
#pragma omp for nowait
  for (uint64_t i = 0; i < g->n_local+1; ++i)
    out_degree_list[i] = 0;
#pragma omp for
  for (uint64_t i = 0; i < g->n_local; ++i)
    temp_counts[i] = 0;
}

  for (uint64_t i = 0; i < g->m_local_out*2; i+=2)
    ++temp_counts[ggi->gen_edges[i] - g->n_offset];
  for (uint64_t i = 0; i < g->n_local; ++i)
    out_degree_list[i+1] = out_degree_list[i] + temp_counts[i];
  memcpy(temp_counts, out_degree_list, g->n_local*sizeof(uint64_t));


  for (uint64_t i = 0; i < g->m_local_out*2; i+=2)
    out_edges[temp_counts[ggi->gen_edges[i] - g->n_offset]++] = ggi->gen_edges[i+1];
  
  free(ggi->gen_edges);
  g->out_edges = out_edges;
  g->out_degree_list = out_degree_list;

  uint64_t* in_edges = (uint64_t*)malloc(g->m_local_in*sizeof(uint64_t));
  uint64_t* in_degree_list = (uint64_t*)malloc((g->n_local+1)*sizeof(uint64_t));
  if (in_edges == NULL || in_degree_list == NULL)
    throw_err("create_graph(), unable to allocate in edge storage\n", procid);

#pragma omp parallel
{
#pragma omp for nowait
  for (uint64_t i = 0; i < g->n_local+1; ++i)
    in_degree_list[i] = 0;
#pragma omp for nowait
  for (uint64_t i = 0; i < g->n_local; ++i)
    temp_counts[i] = 0;
}

  for (uint64_t i = 0; i < g->m_local_in*2; i+=2)
    ++temp_counts[ggi->gen_edges_rev[i+1] - g->n_offset];
  for (uint64_t i = 0; i < g->n_local; ++i)
    in_degree_list[i+1] = in_degree_list[i] + temp_counts[i];
  memcpy(temp_counts, in_degree_list, g->n_local*sizeof(uint64_t));
  for (uint64_t i = 0; i < g->m_local_in*2; i+=2)
    in_edges[temp_counts[ggi->gen_edges_rev[i+1] - g->n_offset]++] = ggi->gen_edges_rev[i];

  free(temp_counts);
  free(ggi->gen_edges_rev);
  g->in_edges = in_edges;
  g->in_degree_list = in_degree_list;

  g->local_unmap = (uint64_t*)malloc(g->n_local*sizeof(uint64_t));
  if (g->local_unmap == NULL)
    throw_err("create_graph(), unable to allocate unmap", procid);

#pragma omp parallel for
  for (uint64_t i = 0; i < g->n_local; ++i)
    g->local_unmap[i] = i + g->n_offset;

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d create_graph() %9.6f (s)\n", procid, elt);
  }

  if (debug) { printf("Task %d create_graph() success\n", procid); }
  return 0;
}

int create_graph_serial(graph_gen_data_t *ggi, dist_graph_t *g)
{
  if (debug) { printf("Task %d create_graph_serial() success\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  g->n = ggi->n;
  g->n_local = ggi->n_local;
  g->n_offset = ggi->n_offset;
  g->m = ggi->m;
  g->m_local_out = ggi->m;
  g->m_local_in = ggi->m;
  g->n_ghost = 0;
  g->n_total = g->n_local;
  init_map(&g->map);

  uint64_t* out_edges = (uint64_t*)malloc(g->m_local_out*sizeof(uint64_t));
  uint64_t* out_degree_list = (uint64_t*)malloc((g->n_local+1)*sizeof(uint64_t));
  uint64_t* temp_counts = (uint64_t*)malloc(g->n_local*sizeof(uint64_t));
  if (out_edges == NULL || out_degree_list == NULL || temp_counts == NULL)
  throw_err("create_graph_serial(), unable to allocate out edge storage\n", procid);

#pragma omp parallel
{
#pragma omp for nowait
  for (uint64_t i = 0; i < g->n_local+1; ++i)
    out_degree_list[i] = 0;
#pragma omp for nowait
  for (uint64_t i = 0; i < g->n_local; ++i)
    temp_counts[i] = 0;
}

  for (uint64_t i = 0; i < g->m_local_out*2; i+=2)
    ++temp_counts[ggi->gen_edges[i] - g->n_offset];
  for (uint64_t i = 0; i < g->n_local; ++i)
    out_degree_list[i+1] = out_degree_list[i] + temp_counts[i];
  memcpy(temp_counts, out_degree_list, g->n_local*sizeof(uint64_t));
  for (uint64_t i = 0; i < g->m_local_out*2; i+=2)
    out_edges[temp_counts[ggi->gen_edges[i] - (uint64_t)g->n_offset]++] = ggi->gen_edges[i+1];
  
  g->out_edges = out_edges;
  g->out_degree_list = out_degree_list;

  uint64_t* in_edges = (uint64_t*)malloc(g->m_local_in*sizeof(uint64_t));
  uint64_t* in_degree_list = (uint64_t*)malloc((g->n_local+1)*sizeof(uint64_t));
  if (in_edges == NULL || in_degree_list == NULL)
    throw_err("create_graph_serial(), unable to allocate in edge storage\n", procid);

#pragma omp parallel
{
#pragma omp for nowait
  for (uint64_t i = 0; i < g->n_local+1; ++i)
    in_degree_list[i] = 0;
#pragma omp for nowait
  for (uint64_t i = 0; i < g->n_local; ++i)
    temp_counts[i] = 0;
}

  for (uint64_t i = 0; i < g->m_local_in*2; i+=2)
    ++temp_counts[ggi->gen_edges[i+1] - g->n_offset];
  for (uint64_t i = 0; i < g->n_local; ++i)
    in_degree_list[i+1] = in_degree_list[i] + temp_counts[i];
  memcpy(temp_counts, in_degree_list, g->n_local*sizeof(uint64_t));
  for (uint64_t i = 0; i < g->m_local_in*2; i+=2)
    in_edges[temp_counts[ggi->gen_edges[i+1] - (uint64_t)g->n_offset]++] = ggi->gen_edges[i];

  free(temp_counts);
  free(ggi->gen_edges);
  g->in_edges = in_edges;
  g->in_degree_list = in_degree_list;

  g->local_unmap = (uint64_t*)malloc(g->n_local*sizeof(uint64_t));  
  if (g->local_unmap == NULL)
    throw_err("create_graph(), unable to allocate unmap\n", procid);

  for (uint64_t i = 0; i < g->n_local; ++i)
    g->local_unmap[i] = i + g->n_offset;

  //int64_t total_edges = g->m_local_in + g->m_local_out;
  init_map_nohash(&g->map, g->n);
  for (uint64_t i = 0; i < g->n_local; ++i)
    set_value_uq(&g->map, i, i);


  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d create_graph_serial() %9.6f (s)\n", procid, elt);
  }
  if (debug) { printf("Task %d create_graph_serial() success\n", procid); }
  return 0;
}


int clear_graph(dist_graph_t *g)
{
  if (debug) { printf("Task %d clear_graph() start\n", procid); }

  free(g->out_edges);
  free(g->in_edges);
  free(g->out_degree_list);
  free(g->in_degree_list);
  free(g->local_unmap);
  clear_map(&g->map);
  if (nprocs > 1) {
    free(g->ghost_unmap);
    free(g->ghost_tasks);
  }

  if (debug) { printf("Task %d clear_graph() success\n", procid); }
  return 0;
} 


int relabel_edges(dist_graph_t *g)
{
  if (debug) { printf("Task %d relabel_edges() start\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  uint64_t cur_label = g->n_local;
  uint64_t total_edges = g->m_local_in + g->m_local_out;

  if (is_init(&g->map))
    clear_map(&g->map);
  init_map(&g->map, total_edges*2);

  for (uint64_t i = 0; i < g->n_local; ++i)
  {
    uint64_t vert = g->local_unmap[i];
    set_value(&g->map, vert, i);
  }

  for (uint64_t i = 0; i < g->m_local_out; ++i)
  {
    uint64_t out = g->out_edges[i];
    uint64_t val = get_value(&g->map, out);
    if (val == NULL_KEY)
    {
      set_value_uq(&g->map, out, cur_label);
      g->out_edges[i] = cur_label++;
    }
    else        
      g->out_edges[i] = val;
  }
  for (uint64_t i = 0; i < g->m_local_in; ++i)
  {
    uint64_t in = g->in_edges[i];
    uint64_t val = get_value(&g->map, in);
    if (val == NULL_KEY)
    {
      set_value_uq(&g->map, in, cur_label);
      g->in_edges[i] = cur_label++;
    }
    else
      g->in_edges[i] = val;
  }

  g->n_ghost = (uint64_t)g->map.num_unique;
  g->n_total = g->n_ghost + g->n_local;

  if (debug)
    printf("Task %d, n_ghost %lu\n", procid, g->n_ghost);

  g->ghost_unmap = (uint64_t*)malloc(g->n_ghost*sizeof(uint64_t));
  g->ghost_tasks = (uint64_t*)malloc(g->n_ghost*sizeof(uint64_t));
  if (g->ghost_unmap == NULL || g->ghost_tasks == NULL)
    throw_err("relabel_edges(), unable to allocate ghost unmaps", procid);

  uint64_t n_per_rank = g->n / (uint64_t)nprocs + 1;  

#pragma omp parallel for
  for (uint64_t i = 0; i < g->n_ghost; ++i)
  {
    uint64_t cur_index = get_value(&g->map, g->map.unique_keys[i]);

    cur_index -= g->n_local;
    g->ghost_unmap[cur_index] = g->map.unique_keys[i];
    g->ghost_tasks[cur_index] = g->map.unique_keys[i] / n_per_rank;
  }

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf(" Task %d relabel_edges() %9.6f (s)\n", procid, elt); 
  }

  if (debug) { printf("Task %d relabel_edges() success\n", procid); }
  return 0;
}


int relabel_edges(dist_graph_t *g, int32_t* global_parts)
{
  if (debug) { printf("Task %d relabel_edges() start\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  uint64_t cur_label = g->n_local;
  uint64_t total_edges = g->m_local_in + g->m_local_out;

  clear_map(&g->map);
  init_map(&g->map, total_edges*2);

  for (uint64_t i = 0; i < g->n_local; ++i)
  {
    uint64_t vert = g->local_unmap[i];
    set_value(&g->map, vert, i);
  }

  for (uint64_t i = 0; i < g->m_local_out; ++i)
  {
    uint64_t out = g->out_edges[i];
    uint64_t val = get_value(&g->map, out);
    if (val == NULL_KEY)
    {
      set_value_uq(&g->map, out, cur_label);
      g->out_edges[i] = cur_label++;
    }
    else        
      g->out_edges[i] = val;
  }
  for (uint64_t i = 0; i < g->m_local_in; ++i)
  {
    uint64_t in = g->in_edges[i];
    uint64_t val = get_value(&g->map, in);
    if (val == NULL_KEY)
    {
      set_value_uq(&g->map, in, cur_label);
      g->in_edges[i] = cur_label++;
    }
    else
      g->in_edges[i] = val;
  }

  g->n_ghost = (uint64_t)g->map.num_unique;
  g->n_total = g->n_ghost + g->n_local;

  if (debug)
    printf("Task %d, n_ghost %lu\n", procid, g->n_ghost);

  free(g->ghost_unmap);
  free(g->ghost_tasks);
  g->ghost_unmap = (uint64_t*)malloc(g->n_ghost*sizeof(uint64_t));
  g->ghost_tasks = (uint64_t*)malloc(g->n_ghost*sizeof(uint64_t));
  if (g->ghost_unmap == NULL || g->ghost_tasks == NULL)
    throw_err("relabel_edges(), unable to allocate ghost unmaps", procid);

  
#pragma omp parallel for
  for (uint64_t i = 0; i < g->n_ghost; ++i)
  {
    uint64_t cur_index = get_value(&g->map, g->map.unique_keys[i]);

    cur_index -= g->n_local;
    g->ghost_unmap[cur_index] = g->map.unique_keys[i];
    g->ghost_tasks[cur_index] = global_parts[g->map.unique_keys[i]];
  }

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf(" Task %d relabel_edges() %9.6f (s)\n", procid, elt); 
  }

  if (debug) { printf("Task %d relabel_edges() success\n", procid); }
  return 0;
}

int repart_graph(dist_graph_t *g, mpi_data_t* comm, char* part_file)
{
  if (debug) { printf("Task %d repart_graph() start\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  int32_t* global_parts = (int32_t*)malloc(g->n*sizeof(int32_t));
  int32_t* local_parts = (int32_t*)malloc(g->n_local*sizeof(int32_t));

#pragma omp parallel for
  for (uint64_t i = 0; i < g->n; ++i)
    global_parts[i] = -1;

  if (procid == 0)
  {
    std::ifstream outfile;
    outfile.open(part_file);

    for (uint64_t i = 0; i < g->n; ++i)
      outfile >> global_parts[i];

    outfile.close();

    if (debug)
      for (uint64_t i = 0; i < g->n; ++i)
        if (global_parts[i] == -1)
        {
          printf("Part Error: %lu not assigned\n", i);
          global_parts[i] = 0;
        }
  }

  MPI_Bcast(global_parts, (int32_t)g->n, MPI_INT32_T, 0, MPI_COMM_WORLD);
  
#pragma omp parallel for
    for (uint64_t i = 0; i < g->n_local; ++i)
      local_parts[i] = global_parts[g->local_unmap[i]];
  
  repart_graph(g, comm, local_parts);
  relabel_edges(g, global_parts);
  if (debug) {
    for (uint64_t i = 0; i < g->n_local; ++i)
      if (global_parts[g->local_unmap[i]] != procid)
        printf("Part Error: task %d received %lu which was assigned to %d\n", procid, g->local_unmap[i], global_parts[g->local_unmap[i]]);
    
  }
  free(local_parts);
  free(global_parts);

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d repart_graph() %9.6f (s)\n", procid, elt); 
  }

  if (debug) { printf("Task %d repart_graph() success\n", procid); }
  return 0;
}


int repart_graph(dist_graph_t *g, mpi_data_t* comm, int32_t* local_parts)
{
   for (int i = 0; i < nprocs; ++i)
  {
    comm->sendcounts_temp[i] = 0;
    comm->recvcounts_temp[i] = 0;
  }

  for (uint64_t i = 0; i < g->n_local; ++i)
  {
    int32_t rank = local_parts[i];
    ++comm->sendcounts_temp[rank];
  }

  MPI_Alltoall(comm->sendcounts_temp, 1, MPI_UINT64_T, 
               comm->recvcounts_temp, 1, MPI_UINT64_T, MPI_COMM_WORLD);
  
  uint64_t total_recv = 0;
  uint64_t total_send = 0;
  for (int32_t i = 0; i < nprocs; ++i)
  {
    total_recv += comm->recvcounts_temp[i];
    total_send += comm->sendcounts_temp[i];
  }

  uint64_t* recvbuf_vids = (uint64_t*)malloc(total_recv*sizeof(uint64_t));
  uint64_t* recvbuf_deg_out = (uint64_t*)malloc(total_recv*sizeof(uint64_t));
  uint64_t* recvbuf_deg_in = (uint64_t*)malloc(total_recv*sizeof(uint64_t));
  if (recvbuf_vids == NULL ||
      recvbuf_deg_out == NULL || recvbuf_deg_in == NULL)
    throw_err("repart_graph(), unable to allocate buffers\n", procid);

  uint64_t max_transfer = total_send > total_recv ? total_send : total_recv;
  uint64_t num_comms = max_transfer / (uint64_t)(MAX_SEND_SIZE/(g->m/g->n))+ 1;
  MPI_Allreduce(MPI_IN_PLACE, &num_comms, 1, 
                MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);

  if (debug) 
    printf("Task %d repart_graph() num_comms %lu total_send %lu total_recv %lu\n", procid, num_comms, total_send, total_recv);

  uint64_t sum_recv_deg = 0;
  for (uint64_t c = 0; c < num_comms; ++c)
  {
    uint64_t send_begin = (g->n_local * c) / num_comms;
    uint64_t send_end = (g->n_local * (c + 1)) / num_comms;
    if (c == (num_comms-1))
      send_end = g->n_local;

    for (int32_t i = 0; i < nprocs; ++i)
    {
      comm->sendcounts[i] = 0;
      comm->recvcounts[i] = 0;
    }

    if (debug)
      printf("Task %d send_begin %lu send_end %lu\n", procid, send_begin, send_end);
    for (uint64_t i = send_begin; i < send_end; ++i)
    {
      int32_t rank = local_parts[i];
      ++comm->sendcounts[rank];
    }

    MPI_Alltoall(comm->sendcounts, 1, MPI_INT32_T, 
                 comm->recvcounts, 1, MPI_INT32_T, MPI_COMM_WORLD);

    comm->sdispls[0] = 0;
    comm->sdispls_cpy[0] = 0;
    comm->rdispls[0] = 0;
    for (int32_t i = 1; i < nprocs; ++i)
    {
      comm->sdispls[i] = comm->sdispls[i-1] + comm->sendcounts[i-1];
      comm->rdispls[i] = comm->rdispls[i-1] + comm->recvcounts[i-1];
      comm->sdispls_cpy[i] = comm->sdispls[i];
    }

    int32_t cur_send = comm->sdispls[nprocs-1] + comm->sendcounts[nprocs-1];
    int32_t cur_recv = comm->rdispls[nprocs-1] + comm->recvcounts[nprocs-1];
    uint64_t* sendbuf_vids = (uint64_t*)malloc((uint64_t)cur_send*sizeof(uint64_t));
    uint64_t* sendbuf_deg_out = (uint64_t*)malloc((uint64_t)cur_send*sizeof(uint64_t));
    uint64_t* sendbuf_deg_in = (uint64_t*)malloc((uint64_t)cur_send*sizeof(uint64_t));
    if (sendbuf_vids == NULL || 
        sendbuf_deg_out == NULL || sendbuf_deg_in == NULL)
      throw_err("repart_graph(), unable to allocate buffers\n", procid);

    for (uint64_t i = send_begin; i < send_end; ++i)
    {
      int32_t rank = local_parts[i];
      int32_t snd_index = comm->sdispls_cpy[rank]++;
      sendbuf_vids[snd_index] = g->local_unmap[i];
      sendbuf_deg_out[snd_index] = (uint64_t)out_degree(g, i);
      sendbuf_deg_in[snd_index] = (uint64_t)in_degree(g, i);
    }

    MPI_Alltoallv(
      sendbuf_vids, comm->sendcounts, comm->sdispls, MPI_UINT64_T, 
      recvbuf_vids+sum_recv_deg, comm->recvcounts, comm->rdispls,
      MPI_UINT64_T, MPI_COMM_WORLD);    
    MPI_Alltoallv(
      sendbuf_deg_out, comm->sendcounts, comm->sdispls, MPI_UINT64_T, 
      recvbuf_deg_out+sum_recv_deg, comm->recvcounts, comm->rdispls,
      MPI_UINT64_T, MPI_COMM_WORLD);
    MPI_Alltoallv(
      sendbuf_deg_in, comm->sendcounts, comm->sdispls, MPI_UINT64_T, 
      recvbuf_deg_in+sum_recv_deg, comm->recvcounts, comm->rdispls,
      MPI_UINT64_T, MPI_COMM_WORLD);
    sum_recv_deg += (uint64_t)cur_recv;
    free(sendbuf_vids);
    free(sendbuf_deg_out);
    free(sendbuf_deg_in);
  }






  for (int i = 0; i < nprocs; ++i)
  {
    comm->sendcounts_temp[i] = 0;
    comm->recvcounts_temp[i] = 0;
  }

  for (uint64_t i = 0; i < g->n_local; ++i)
  {
    int32_t rank = local_parts[i];
    comm->sendcounts_temp[rank] += (uint64_t)out_degree(g, i);
  }

  MPI_Alltoall(comm->sendcounts_temp, 1, MPI_UINT64_T, 
               comm->recvcounts_temp, 1, MPI_UINT64_T, MPI_COMM_WORLD);
  
  total_recv = 0;
  total_send = 0;
  for (int32_t i = 0; i < nprocs; ++i)
  {
    total_recv += comm->recvcounts_temp[i];
    total_send += comm->sendcounts_temp[i];
  }

  uint64_t* recvbuf_e_out = (uint64_t*)malloc(total_recv*sizeof(uint64_t));
  if (recvbuf_e_out == NULL)
    throw_err("repart_graph(), unable to allocate buffer\n", procid);

  // max_transfer = total_send > total_recv ? total_send : total_recv;
  // num_comms = max_transfer / (uint64_t)MAX_SEND_SIZE + 1;
  // MPI_Allreduce(MPI_IN_PLACE, &num_comms, 1, 
  //               MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);

  if (debug) 
    printf("Task %d repart_graph() num_comms %lu total_send %lu total_recv %lu\n", procid, num_comms, total_send, total_recv);
  
  uint64_t sum_recv_e_out = 0;
  for (uint64_t c = 0; c < num_comms; ++c)
  {
    uint64_t send_begin = (g->n_local * c) / num_comms;
    uint64_t send_end = (g->n_local * (c + 1)) / num_comms;
    if (c == (num_comms-1))
      send_end = g->n_local;

    for (int32_t i = 0; i < nprocs; ++i)
    {
      comm->sendcounts[i] = 0;
      comm->recvcounts[i] = 0;
    }

    for (uint64_t i = send_begin; i < send_end; ++i)
    {
      uint32_t rank = local_parts[i];
      comm->sendcounts[rank] += (int32_t)out_degree(g, i);
    }

    MPI_Alltoall(comm->sendcounts, 1, MPI_INT32_T, 
                 comm->recvcounts, 1, MPI_INT32_T, MPI_COMM_WORLD);

    comm->sdispls[0] = 0;
    comm->sdispls_cpy[0] = 0;
    comm->rdispls[0] = 0;
    for (int32_t i = 1; i < nprocs; ++i)
    {
      comm->sdispls[i] = comm->sdispls[i-1] + comm->sendcounts[i-1];
      comm->rdispls[i] = comm->rdispls[i-1] + comm->recvcounts[i-1];
      comm->sdispls_cpy[i] = comm->sdispls[i];
    }

    int32_t cur_send = comm->sdispls[nprocs-1] + comm->sendcounts[nprocs-1];
    int32_t cur_recv = comm->rdispls[nprocs-1] + comm->recvcounts[nprocs-1];
    uint64_t* sendbuf_e_out = (uint64_t*)malloc((uint64_t)cur_send*sizeof(uint64_t));
    if (sendbuf_e_out == NULL)
      throw_err("repart_graph(), unable to allocate buffer\n", procid);

    for (uint64_t i = send_begin; i < send_end; ++i)
    {
      uint64_t out_degree = out_degree(g, i);
      uint64_t* outs = out_vertices(g, i);
      int32_t rank = local_parts[i];
      int32_t snd_index = comm->sdispls_cpy[rank];
      comm->sdispls_cpy[rank] += out_degree;
      for (uint64_t j = 0; j < out_degree; ++j)
      {
        uint64_t out;
        if (outs[j] < g->n_local)
          out = g->local_unmap[outs[j]];
        else
          out = g->ghost_unmap[outs[j]-g->n_local];
        sendbuf_e_out[snd_index++] = out;
      }
    }

    MPI_Alltoallv(sendbuf_e_out, comm->sendcounts, comm->sdispls, MPI_UINT64_T, 
                  recvbuf_e_out+sum_recv_e_out, comm->recvcounts, comm->rdispls,
                  MPI_UINT64_T, MPI_COMM_WORLD);
    sum_recv_e_out += (uint64_t)cur_recv;
    free(sendbuf_e_out);
  }

  free(g->out_edges);
  free(g->out_degree_list);
  g->out_edges = recvbuf_e_out;
  g->m_local_out = (uint64_t)sum_recv_e_out;
  g->out_degree_list = (uint64_t*)malloc((sum_recv_deg+1)*sizeof(uint64_t));
  g->out_degree_list[0] = 0;
  for (uint64_t i = 0; i < sum_recv_deg; ++i)
    g->out_degree_list[i+1] = g->out_degree_list[i] + recvbuf_deg_out[i];
  assert(g->out_degree_list[sum_recv_deg] == g->m_local_out);
  free(recvbuf_deg_out);







  for (int i = 0; i < nprocs; ++i)
  {
    comm->sendcounts_temp[i] = 0;
    comm->recvcounts_temp[i] = 0;
  }

  for (uint64_t i = 0; i < g->n_local; ++i)
  {
    int32_t rank = local_parts[i];
    comm->sendcounts_temp[rank] += (uint64_t)in_degree(g, i);
  }

  MPI_Alltoall(comm->sendcounts_temp, 1, MPI_UINT64_T, 
               comm->recvcounts_temp, 1, MPI_UINT64_T, MPI_COMM_WORLD);
  
  total_recv = 0;
  total_send = 0;
  for (int32_t i = 0; i < nprocs; ++i)
  {
    total_recv += comm->recvcounts_temp[i];
    total_send += comm->sendcounts_temp[i];
  }

  uint64_t* recvbuf_e_in = (uint64_t*)malloc(total_recv*sizeof(uint64_t));
  if (recvbuf_e_in == NULL)
    throw_err("repart_graph(), unable to allocate buffer\n", procid);

  // max_transfer = total_send > total_recv ? total_send : total_recv;
  // num_comms = max_transfer / (uint64_t)MAX_SEND_SIZE + 1;
  // MPI_Allreduce(MPI_IN_PLACE, &num_comms, 1, 
  //               MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);

  if (debug) 
    printf("Task %d repart_graph() num_comms %lu total_send %lu total_recv %lu\n", procid, num_comms, total_send, total_recv);
  
  uint64_t sum_recv_e_in = 0;
  for (uint64_t c = 0; c < num_comms; ++c)
  {
    uint64_t send_begin = (g->n_local * c) / num_comms;
    uint64_t send_end = (g->n_local * (c + 1)) / num_comms;
    if (c == (num_comms-1))
      send_end = g->n_local;

    for (int32_t i = 0; i < nprocs; ++i)
    {
      comm->sendcounts[i] = 0;
      comm->recvcounts[i] = 0;
    }

    for (uint64_t i = send_begin; i < send_end; ++i)
    {
      int32_t rank = local_parts[i];
      comm->sendcounts[rank] += (int32_t)in_degree(g, i);
    }

    MPI_Alltoall(comm->sendcounts, 1, MPI_INT32_T, 
                 comm->recvcounts, 1, MPI_INT32_T, MPI_COMM_WORLD);

    comm->sdispls[0] = 0;
    comm->sdispls_cpy[0] = 0;
    comm->rdispls[0] = 0;
    for (int32_t i = 1; i < nprocs; ++i)
    {
      comm->sdispls[i] = comm->sdispls[i-1] + comm->sendcounts[i-1];
      comm->rdispls[i] = comm->rdispls[i-1] + comm->recvcounts[i-1];
      comm->sdispls_cpy[i] = comm->sdispls[i];
    }

    int32_t cur_send = comm->sdispls[nprocs-1] + comm->sendcounts[nprocs-1];
    int32_t cur_recv = comm->rdispls[nprocs-1] + comm->recvcounts[nprocs-1];
    uint64_t* sendbuf_e_in = (uint64_t*)malloc((uint64_t)cur_send*sizeof(uint64_t));
    if (sendbuf_e_in == NULL)
      throw_err("repart_graph(), unable to allocate buffer\n", procid);

    for (uint64_t i = send_begin; i < send_end; ++i)
    {
      uint64_t in_degree = in_degree(g, i);
      uint64_t* ins = in_vertices(g, i);
      int32_t rank = local_parts[i];
      int32_t snd_index = comm->sdispls_cpy[rank];
      comm->sdispls_cpy[rank] += in_degree;
      for (uint32_t j = 0; j < in_degree; ++j)
      {
        uint64_t in;
        if (ins[j] < g->n_local)
          in = g->local_unmap[ins[j]];
        else
          in = g->ghost_unmap[ins[j]-g->n_local];
        sendbuf_e_in[snd_index++] = in;
      }
    }

    MPI_Alltoallv(sendbuf_e_in, comm->sendcounts, comm->sdispls, MPI_UINT64_T, 
                  recvbuf_e_in+sum_recv_e_in, comm->recvcounts, comm->rdispls,
                  MPI_UINT64_T, MPI_COMM_WORLD);
    sum_recv_e_in += (uint64_t)cur_recv;
    free(sendbuf_e_in);
  }

  free(g->in_edges);
  free(g->in_degree_list);
  g->in_edges = recvbuf_e_in;
  g->m_local_in = (uint64_t)sum_recv_e_in;
  g->in_degree_list = (uint64_t*)malloc((sum_recv_deg+1)*sizeof(uint64_t));
  g->in_degree_list[0] = 0;
  for (uint64_t i = 0; i < sum_recv_deg; ++i)
    g->in_degree_list[i+1] = g->in_degree_list[i] + recvbuf_deg_in[i];
  assert(g->in_degree_list[sum_recv_deg] == g->m_local_in);
  free(recvbuf_deg_in);



  free(g->local_unmap);
  g->local_unmap = (uint64_t*)malloc(sum_recv_deg*sizeof(uint64_t));
  for (uint64_t i = 0; i < sum_recv_deg; ++i)
    g->local_unmap[i] = recvbuf_vids[i];
  free(recvbuf_vids);

  g->n_local = sum_recv_deg;
}


int get_max_degree_vert(dist_graph_t *g)
{ 
  if (debug) { printf("Task %d get_max_degree_vert() start\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  uint64_t my_max_degree = 0;
  uint64_t my_max_out_degree = 0;
  uint64_t my_max_in_degree = 0;
  uint64_t my_max_vert = -1;
  for (uint64_t i = 0; i < g->n_local; ++i)
  {
    uint64_t out_degree = out_degree(g, i);
    uint64_t in_degree = in_degree(g, i);
    uint64_t this_degree = (uint64_t)in_degree;
    if (this_degree > my_max_degree)
    {
      my_max_degree = this_degree;
      my_max_vert = g->local_unmap[i];
    }
    if (out_degree > my_max_out_degree)
      my_max_out_degree = out_degree;
    if (in_degree > my_max_in_degree)
      my_max_in_degree = in_degree;
  }

  uint64_t max_in_degree;
  uint64_t max_out_degree;
  uint64_t max_degree;
  uint64_t max_vert;

  MPI_Allreduce(&my_max_out_degree, &max_out_degree, 1, MPI_UINT64_T, 
                MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&my_max_in_degree, &max_in_degree, 1, MPI_UINT64_T, 
                MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&my_max_degree, &max_degree, 1, MPI_UINT64_T,
                MPI_MAX, MPI_COMM_WORLD);
  if (my_max_degree == max_degree)
    max_vert = my_max_vert;
  else
    max_vert = NULL_KEY;
  MPI_Allreduce(MPI_IN_PLACE, &max_vert, 1, MPI_UINT64_T,
                MPI_MIN, MPI_COMM_WORLD);

  g->max_degree_vert = max_vert;
  g->max_out_degree = max_out_degree;
  g->max_in_degree = max_in_degree;

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d, max_degree %lu, max_vert %lu, max_in_degree %lu, max_out_degree %lu, %f (s)\n", 
           procid, max_degree, max_vert, max_in_degree, max_out_degree, elt);
  }

  if (debug) { printf("Task %d get_max_degree_vert() success\n", procid); }
  return 0;
}
