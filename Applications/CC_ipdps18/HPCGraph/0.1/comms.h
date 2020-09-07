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

#ifndef _COMMS_H_
#define _COMMS_H_

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

#include "dist_graph.h"
#include "util.h"

extern int procid, nprocs;
extern bool verbose, debug, verify, output;

#define MAX_SEND_SIZE 2147483648
#define THREAD_QUEUE_SIZE 1024

struct mpi_data_t {
  int32_t* sendcounts;
  int32_t* recvcounts;
  int32_t* sdispls;
  int32_t* rdispls;
  int32_t* sdispls_cpy;

  uint64_t* recvcounts_temp;
  uint64_t* sendcounts_temp;
  uint64_t* sdispls_temp;
  uint64_t* rdispls_temp;
  uint64_t* sdispls_cpy_temp;

  uint64_t* sendbuf_vert;
  uint64_t* sendbuf_data;
  double* sendbuf_data_flt;
  uint64_t* recvbuf_vert;
  uint64_t* recvbuf_data;
  double* recvbuf_data_flt;

  uint64_t total_recv;
  uint64_t total_send;
  uint64_t global_queue_size;
} ;

struct queue_data_t {
  uint64_t* queue;
  uint64_t* queue_next;
  uint64_t* queue_send;

  uint64_t queue_size;
  uint64_t next_size;
  uint64_t send_size;
} ;

struct thread_queue_t {
  int32_t tid;
  uint64_t* thread_queue;
  uint64_t* thread_send;
  uint64_t thread_queue_size;
  uint64_t thread_send_size;
} ;

struct thread_comm_t {
  int32_t tid;
  bool* v_to_rank;
  uint64_t* sendcounts_thread;
  uint64_t* sendbuf_vert_thread;
  uint64_t* sendbuf_data_thread;
  double* sendbuf_data_thread_flt;
  int32_t* sendbuf_rank_thread;
  uint64_t* thread_starts;
  uint64_t thread_queue_size;
} ;

void init_queue_data(dist_graph_t* g, queue_data_t* q);
void clear_queue_data(queue_data_t* q);
void init_comm_data(mpi_data_t* comm);
void clear_comm_data(mpi_data_t* comm);

void init_thread_queue(thread_queue_t* tq);
void clear_thread_queue(thread_queue_t* tq);
void init_thread_comm(thread_comm_t* tc);
void clear_thread_comm(thread_comm_t* tc);
void init_thread_comm_flt(thread_comm_t* tc);
void clear_thread_commflt(thread_comm_t* tc);

void init_sendbuf_vid_data(mpi_data_t* comm);
void init_recvbuf_vid_data(mpi_data_t* comm);
void init_sendbuf_vid_data_flt(mpi_data_t* comm);
void init_recvbuf_vid_data_flt(mpi_data_t* comm);
void clear_recvbuf_vid_data(mpi_data_t* comm);
void clear_allbuf_vid_data(mpi_data_t* comm);

inline void exchange_verts(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q);
inline void exchange_verts(mpi_data_t* comm);
inline void exchange_data(mpi_data_t* comm);
inline void exchange_data_flt(mpi_data_t* comm);
inline void exchange_vert_data(dist_graph_t* g, mpi_data_t* comm, 
                               queue_data_t* q);
inline void exchange_vert_data(dist_graph_t* g, mpi_data_t* comm);


inline void update_sendcounts_thread(dist_graph_t* g, 
                              thread_comm_t* tc, uint64_t vert_index);
inline void update_sendcounts_thread_out(dist_graph_t* g, 
                              thread_comm_t* tc, uint64_t vert_index);

inline void update_vid_data_queues(dist_graph_t* g, 
                            thread_comm_t* tc, mpi_data_t* comm,
                            uint64_t vert_index, uint64_t data);
inline void update_vid_data_queues_out(dist_graph_t* g, 
                            thread_comm_t* tc, mpi_data_t* comm,
                            uint64_t vert_index, uint64_t data);
inline void update_vid_data_queues_out(dist_graph_t* g, 
                            thread_comm_t* tc, mpi_data_t* comm,
                            uint64_t vert_index, double data);


inline void add_vid_to_queue(thread_queue_t* tq, queue_data_t* q, 
                            uint64_t vertex_id);
inline void empty_queue(thread_queue_t* tq, queue_data_t* q);


inline void add_vid_to_send(thread_queue_t* tq, queue_data_t* q, 
                            uint64_t vertex_id);
inline void empty_send(thread_queue_t* tq, queue_data_t* q);


inline void add_vid_data_to_send(thread_comm_t* tc, mpi_data_t* comm,
  uint64_t vertex_id, uint64_t data_val, int32_t send_rank);
inline void add_vid_data_to_send_flt(thread_comm_t* tc, mpi_data_t* comm,
  uint64_t vertex_id, double data_val, int32_t send_rank);

inline void empty_vid_data(thread_comm_t* tc, mpi_data_t* comm);
inline void empty_vid_data_flt(thread_comm_t* tc, mpi_data_t* comm);



inline void exchange_verts(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q)
{
  comm->global_queue_size = 0;
  uint64_t task_queue_size = q->next_size + q->send_size;
  MPI_Allreduce(&task_queue_size, &comm->global_queue_size, 1, 
                MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);      
  
  uint64_t num_comms = comm->global_queue_size / (uint64_t)MAX_SEND_SIZE + 1;
  uint64_t sum_recv = 0;
  for (uint64_t c = 0; c < num_comms; ++c)
  {
    uint64_t send_begin = (q->send_size * c) / num_comms;
    uint64_t send_end = (q->send_size * (c + 1)) / num_comms;
    if (c == (num_comms-1))
      send_end = q->send_size;

    for (int32_t i = 0; i < nprocs; ++i)
    {
      comm->sendcounts[i] = 0;
      comm->recvcounts[i] = 0;
    }
    for (uint64_t i = send_begin; i < send_end; ++i)
    {
      uint64_t ghost_index = q->queue_send[i] - g->n_local;
      uint64_t ghost_task = g->ghost_tasks[ghost_index];
      ++comm->sendcounts[ghost_task];
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
    comm->sendbuf_vert = (uint64_t*)malloc((uint64_t)(cur_send+1)*sizeof(uint64_t));
    if (comm->sendbuf_vert == NULL)
      throw_err("exchange_verts(), unable to allocate comm buffers", procid);

    for (uint64_t i = send_begin; i < send_end; ++i)
    {
      uint64_t ghost_index = q->queue_send[i] - g->n_local;
      uint64_t ghost_task = g->ghost_tasks[ghost_index];
      uint64_t vert = g->ghost_unmap[ghost_index];
      comm->sendbuf_vert[comm->sdispls_cpy[ghost_task]++] = vert; 
    }

    MPI_Alltoallv(comm->sendbuf_vert, 
                  comm->sendcounts, comm->sdispls, MPI_UINT64_T, 
                  q->queue_next+q->next_size+sum_recv, 
                  comm->recvcounts, comm->rdispls, MPI_UINT64_T, 
                  MPI_COMM_WORLD);
    free(comm->sendbuf_vert);
    sum_recv += cur_recv;
  }

  q->queue_size = q->next_size + sum_recv;
  q->next_size = 0;
  q->send_size = 0;
  uint64_t* temp = q->queue;
  q->queue = q->queue_next;
  q->queue_next = temp;
}



inline void exchange_vert_data(dist_graph_t* g, mpi_data_t* comm, 
                               queue_data_t* q)
{
  for (int32_t i = 0; i < nprocs; ++i)
    comm->recvcounts_temp[i] = 0;

  MPI_Alltoall(comm->sendcounts_temp, 1, MPI_UINT64_T, 
               comm->recvcounts_temp, 1, MPI_UINT64_T, MPI_COMM_WORLD);

  comm->total_recv = 0;
  for (int i = 0; i < nprocs; ++i)
    comm->total_recv += comm->recvcounts_temp[i];

  comm->recvbuf_vert = (uint64_t*)malloc(comm->total_recv*sizeof(uint64_t));
  comm->recvbuf_data = (uint64_t*)malloc(comm->total_recv*sizeof(uint64_t));
  comm->recvbuf_data_flt = NULL;
  if (comm->recvbuf_vert == NULL || comm->recvbuf_data == NULL)
    throw_err("exchange_vert_data() unable to allocate comm buffers", procid);


  comm->global_queue_size = 0;
  uint64_t task_queue_size = comm->total_send;
  MPI_Allreduce(&task_queue_size, &comm->global_queue_size, 1, 
                MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
  
  uint64_t num_comms = comm->global_queue_size / (uint64_t)MAX_SEND_SIZE + 1;
  uint64_t sum_recv = 0;
  uint64_t sum_send = 0;
  for (uint64_t c = 0; c < num_comms; ++c)
  {
    for (int32_t i = 0; i < nprocs; ++i)
    {
      uint64_t send_begin = (comm->sendcounts_temp[i] * c) / num_comms;
      uint64_t send_end = (comm->sendcounts_temp[i] * (c + 1)) / num_comms;
      if (c == (num_comms-1))
        send_end = comm->sendcounts_temp[i];
      comm->sendcounts[i] = (int32_t)(send_end - send_begin);
      assert(comm->sendcounts[i] >= 0);
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
    uint64_t* buf_v = (uint64_t*)malloc((uint64_t)(cur_send)*sizeof(uint64_t));
    uint64_t* buf_d = (uint64_t*)malloc((uint64_t)(cur_send)*sizeof(uint64_t));
    if (buf_v == NULL || buf_d == NULL)
      throw_err("exchange_verts(), unable to allocate comm buffers", procid);

    for (int32_t i = 0; i < nprocs; ++i)
    {
      uint64_t send_begin = (comm->sendcounts_temp[i] * c) / num_comms;
      uint64_t send_end = (comm->sendcounts_temp[i] * (c + 1)) / num_comms;
      if (c == (num_comms-1))
        send_end = comm->sendcounts_temp[i];

      for (uint64_t j = send_begin; j < send_end; ++j)
      {
        uint64_t vert = comm->sendbuf_vert[comm->sdispls_temp[i]+j];
        uint64_t data = comm->sendbuf_data[comm->sdispls_temp[i]+j];
        buf_v[comm->sdispls_cpy[i]] = vert;
        buf_d[comm->sdispls_cpy[i]++] = data;
      }
    }

    MPI_Alltoallv(buf_v, comm->sendcounts, 
                  comm->sdispls, MPI_UINT64_T, 
                  comm->recvbuf_vert+sum_recv, comm->recvcounts, 
                  comm->rdispls, MPI_UINT64_T, MPI_COMM_WORLD);
    MPI_Alltoallv(buf_d, comm->sendcounts, 
                  comm->sdispls, MPI_UINT64_T, 
                  comm->recvbuf_data+sum_recv, comm->recvcounts, 
                  comm->rdispls, MPI_UINT64_T, MPI_COMM_WORLD);
    free(buf_v);
    free(buf_d);
    sum_recv += cur_recv;
    sum_send += cur_send;
  }

  assert(sum_recv == comm->total_recv);
  assert(sum_send == comm->total_send);

  comm->global_queue_size = 0;
  task_queue_size = comm->total_recv + q->next_size;
  MPI_Allreduce(&task_queue_size, &comm->global_queue_size, 1, 
                MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);  

  q->send_size = 0;
}


inline void exchange_verts(mpi_data_t* comm)
{
  if (debug) { printf("Task %d exchange_verts() start\n", procid); }

  uint64_t num_comms = comm->global_queue_size / (uint64_t)MAX_SEND_SIZE + 1;
  uint64_t sum_recv = 0;
  uint64_t sum_send = 0;
  for (uint64_t c = 0; c < num_comms; ++c)
  {
    for (int32_t i = 0; i < nprocs; ++i)
    {
      uint64_t send_begin = (comm->sendcounts_temp[i] * c) / num_comms;
      uint64_t send_end = (comm->sendcounts_temp[i] * (c + 1)) / num_comms;
      if (c == (num_comms-1))
        send_end = comm->sendcounts_temp[i];
      comm->sendcounts[i] = (int32_t)(send_end - send_begin);
      assert(comm->sendcounts[i] >= 0);
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
    uint64_t* buf_v = (uint64_t*)malloc((uint64_t)(cur_send)*sizeof(uint64_t));
    if (buf_v == NULL)
      throw_err("exchange_verts(), unable to allocate comm buffers", procid);

    for (int32_t i = 0; i < nprocs; ++i)
    {
      uint64_t send_begin = (comm->sendcounts_temp[i] * c) / num_comms;
      uint64_t send_end = (comm->sendcounts_temp[i] * (c + 1)) / num_comms;
      if (c == (num_comms-1))
        send_end = comm->sendcounts_temp[i];

      for (uint64_t j = send_begin; j < send_end; ++j)
      {
        uint64_t vert = comm->sendbuf_vert[comm->sdispls_temp[i]+j];
        buf_v[comm->sdispls_cpy[i]++] = vert;
      }
    }

    MPI_Alltoallv(buf_v, comm->sendcounts, 
                  comm->sdispls, MPI_UINT64_T, 
                  comm->recvbuf_vert+sum_recv, comm->recvcounts, 
                  comm->rdispls, MPI_UINT64_T, MPI_COMM_WORLD);
    free(buf_v);
    sum_recv += cur_recv;
    sum_send += cur_send;
  }

  assert(sum_recv == comm->total_recv);
  assert(sum_send == comm->total_send);

  if (debug) { printf("Task %d exchange_verts() success\n", procid); }

}

inline void exchange_data(mpi_data_t* comm)
{
  if (debug) { printf("Task %d exchange_data() start\n", procid); }

  uint64_t num_comms = comm->global_queue_size / (uint64_t)MAX_SEND_SIZE + 1;
  uint64_t sum_recv = 0;
  uint64_t sum_send = 0;
  for (uint64_t c = 0; c < num_comms; ++c)
  {
    for (int32_t i = 0; i < nprocs; ++i)
    {
      uint64_t send_begin = (comm->sendcounts_temp[i] * c) / num_comms;
      uint64_t send_end = (comm->sendcounts_temp[i] * (c + 1)) / num_comms;
      if (c == (num_comms-1))
        send_end = comm->sendcounts_temp[i];
      comm->sendcounts[i] = (int32_t)(send_end - send_begin);
      assert(comm->sendcounts[i] >= 0);
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
    uint64_t* buf_d = (uint64_t*)malloc((uint64_t)(cur_send)*sizeof(uint64_t));
    if (buf_d == NULL)
      throw_err("exchange_data(), unable to allocate comm buffers", procid);

    for (int32_t i = 0; i < nprocs; ++i)
    {
      uint64_t send_begin = (comm->sendcounts_temp[i] * c) / num_comms;
      uint64_t send_end = (comm->sendcounts_temp[i] * (c + 1)) / num_comms;
      if (c == (num_comms-1))
        send_end = comm->sendcounts_temp[i];

      for (uint64_t j = send_begin; j < send_end; ++j)
      {
        uint64_t data = comm->sendbuf_data[comm->sdispls_temp[i]+j];
        buf_d[comm->sdispls_cpy[i]++] = data;
      }
    }

    MPI_Alltoallv(buf_d, comm->sendcounts, 
                  comm->sdispls, MPI_UINT64_T, 
                  comm->recvbuf_data+sum_recv, comm->recvcounts, 
                  comm->rdispls, MPI_UINT64_T, MPI_COMM_WORLD);
    free(buf_d);
    sum_recv += cur_recv;
    sum_send += cur_send;
  }

  assert(sum_recv == comm->total_recv);
  assert(sum_send == comm->total_send);

  if (debug) { printf("Task %d exchange_data() success\n", procid); }
}

inline void exchange_data_flt(mpi_data_t* comm)
{
  if (debug) { printf("Task %d exchange_data_flt() start\n", procid); }

  uint64_t num_comms = comm->global_queue_size / (uint64_t)MAX_SEND_SIZE + 1;
  uint64_t sum_recv = 0;
  uint64_t sum_send = 0;
  for (uint64_t c = 0; c < num_comms; ++c)
  {
    for (int32_t i = 0; i < nprocs; ++i)
    {
      uint64_t send_begin = (comm->sendcounts_temp[i] * c) / num_comms;
      uint64_t send_end = (comm->sendcounts_temp[i] * (c + 1)) / num_comms;
      if (c == (num_comms-1))
        send_end = comm->sendcounts_temp[i];
      comm->sendcounts[i] = (int32_t)(send_end - send_begin);
      assert(comm->sendcounts[i] >= 0);
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
    double* buf_d = (double*)malloc((double)(cur_send)*sizeof(double));
    if (buf_d == NULL)
      throw_err("exchange_data_flt(), unable to allocate comm buffers", procid);

    for (int32_t i = 0; i < nprocs; ++i)
    {
      uint64_t send_begin = (comm->sendcounts_temp[i] * c) / num_comms;
      uint64_t send_end = (comm->sendcounts_temp[i] * (c + 1)) / num_comms;
      if (c == (num_comms-1))
        send_end = comm->sendcounts_temp[i];

      for (uint64_t j = send_begin; j < send_end; ++j)
      {
        double data = comm->sendbuf_data_flt[comm->sdispls_temp[i]+j];
        buf_d[comm->sdispls_cpy[i]++] = data;
      }
    }

    MPI_Alltoallv(buf_d, comm->sendcounts, 
                  comm->sdispls, MPI_DOUBLE, 
                  comm->recvbuf_data_flt+sum_recv, comm->recvcounts, 
                  comm->rdispls, MPI_DOUBLE, MPI_COMM_WORLD);
    free(buf_d);
    sum_recv += cur_recv;
    sum_send += cur_send;
  }

  assert(sum_recv == comm->total_recv);
  assert(sum_send == comm->total_send);

  if (debug) { printf("Task %d exchange_data_flt() success\n", procid); }
}


inline void update_sendcounts_thread(dist_graph_t* g, 
                                     thread_comm_t* tc, 
                                     uint64_t vert_index)
{
  for (int32_t i = 0; i < nprocs; ++i)
    tc->v_to_rank[i] = false;

  uint64_t out_degree = out_degree(g, vert_index);
  uint64_t* outs = out_vertices(g, vert_index);
  for (uint64_t j = 0; j < out_degree; ++j)
  {
    uint64_t out_index = outs[j];
    if (out_index >= g->n_local)
    {
      int32_t out_rank = g->ghost_tasks[out_index-g->n_local];
      if (!tc->v_to_rank[out_rank])
      {
        tc->v_to_rank[out_rank] = true;
        ++tc->sendcounts_thread[out_rank];
      }
    }
  }
  uint64_t in_degree = in_degree(g, vert_index);
  uint64_t* ins = in_vertices(g, vert_index);
  for (uint64_t j = 0; j < in_degree; ++j)
  {
    uint64_t in_index = ins[j];
    if (in_index >= g->n_local)
    {
      int32_t in_rank = g->ghost_tasks[in_index-g->n_local];
      if (!tc->v_to_rank[in_rank])
      {
        tc->v_to_rank[in_rank] = true;
        ++tc->sendcounts_thread[in_rank];
      }
    }
  }
}

inline void update_sendcounts_thread_out(dist_graph_t* g, 
                                           thread_comm_t* tc, 
                                           uint64_t vert_index)
{
  for (int32_t i = 0; i < nprocs; ++i)
    tc->v_to_rank[i] = false;

  uint64_t out_degree = out_degree(g, vert_index);
  uint64_t* outs = out_vertices(g, vert_index);
  for (uint64_t j = 0; j < out_degree; ++j)
  {
    uint64_t out_index = outs[j];
    if (out_index >= g->n_local)
    {
      int32_t out_rank = g->ghost_tasks[out_index-g->n_local];
      if (!tc->v_to_rank[out_rank])
      {
        tc->v_to_rank[out_rank] = true;
        ++tc->sendcounts_thread[out_rank];
      }
    }
  }
}



inline void update_vid_data_queues(dist_graph_t* g, 
                                   thread_comm_t* tc, mpi_data_t* comm,
                                   uint64_t vert_index, uint64_t data)
{
  for (int32_t i = 0; i < nprocs; ++i)
    tc->v_to_rank[i] = false;

  uint64_t out_degree = out_degree(g, vert_index);
  uint64_t* outs = out_vertices(g, vert_index);
  for (uint64_t j = 0; j < out_degree; ++j)
  {
    uint64_t out_index = outs[j];
    if (out_index >= g->n_local)
    {
      int32_t out_rank = g->ghost_tasks[out_index - g->n_local];
      if (!tc->v_to_rank[out_rank])
      {
        tc->v_to_rank[out_rank] = true;
        add_vid_data_to_send(tc, comm,
          g->local_unmap[vert_index], data, out_rank);
      }
    }
  }

  uint64_t in_degree = in_degree(g, vert_index);
  uint64_t* ins = in_vertices(g, vert_index);
  for (uint64_t j = 0; j < in_degree; ++j)
  {
    uint64_t in_index = ins[j];
    if (in_index >= g->n_local)
    {
      int32_t in_rank = g->ghost_tasks[in_index - g->n_local];
      if (!tc->v_to_rank[in_rank])
      {
        tc->v_to_rank[in_rank] = true;
        add_vid_data_to_send(tc, comm,
          g->local_unmap[vert_index], data, in_rank);
      }
    }
  }
}

inline void update_vid_data_queues_out(dist_graph_t* g, 
                                       thread_comm_t* tc, mpi_data_t* comm,
                                       uint64_t vert_index, double data)
{
  for (int32_t i = 0; i < nprocs; ++i)
    tc->v_to_rank[i] = false;

  uint64_t out_degree = out_degree(g, vert_index);
  uint64_t* outs = out_vertices(g, vert_index);
  for (uint64_t j = 0; j < out_degree; ++j)
  {
    uint64_t out_index = outs[j];
    if (out_index >= g->n_local)
    {
      int32_t out_rank = g->ghost_tasks[out_index - g->n_local];
      if (!tc->v_to_rank[out_rank])
      {
        tc->v_to_rank[out_rank] = true;
        add_vid_data_to_send_flt(tc, comm,
          g->local_unmap[vert_index], data, out_rank);
      }
    }
  }
}

inline void update_vid_data_queues_out(dist_graph_t* g, 
                                       thread_comm_t* tc, mpi_data_t* comm,
                                       uint64_t vert_index, uint64_t data)
{
  for (int32_t i = 0; i < nprocs; ++i)
    tc->v_to_rank[i] = false;

  uint64_t out_degree = out_degree(g, vert_index);
  uint64_t* outs = out_vertices(g, vert_index);
  for (uint64_t j = 0; j < out_degree; ++j)
  {
    uint64_t out_index = outs[j];
    if (out_index >= g->n_local)
    {
      int32_t out_rank = g->ghost_tasks[out_index - g->n_local];
      if (!tc->v_to_rank[out_rank])
      {
        tc->v_to_rank[out_rank] = true;
        add_vid_data_to_send(tc, comm,
          g->local_unmap[vert_index], data, out_rank);
      }
    }
  }
}


inline void add_vid_to_queue(thread_queue_t* tq, queue_data_t* q, 
                            uint64_t vertex_id)
{
  tq->thread_queue[tq->thread_queue_size++] = vertex_id;

  if (tq->thread_queue_size == THREAD_QUEUE_SIZE)
    empty_queue(tq, q);
}

inline void empty_queue(thread_queue_t* tq, queue_data_t* q)
{
  uint64_t start_offset;

#pragma omp atomic capture
  start_offset = q->next_size += tq->thread_queue_size;

  start_offset -= tq->thread_queue_size;
  for (uint64_t i = 0; i < tq->thread_queue_size; ++i)
    q->queue_next[start_offset + i] = tq->thread_queue[i];
  tq->thread_queue_size = 0;
}

inline void add_vid_to_send(thread_queue_t* tq, queue_data_t* q, 
                            uint64_t vertex_id)
{
  tq->thread_send[tq->thread_send_size++] = vertex_id;

  if (tq->thread_send_size == THREAD_QUEUE_SIZE)
    empty_send(tq, q);
}

inline void empty_send(thread_queue_t* tq, queue_data_t* q)
{
  uint64_t start_offset;

#pragma omp atomic capture
  start_offset = q->send_size += tq->thread_send_size;
  
  start_offset -= tq->thread_send_size;
  for (uint64_t i = 0; i < tq->thread_send_size; ++i)
    q->queue_send[start_offset + i] = tq->thread_send[i];
  tq->thread_send_size = 0;
}


inline void add_vid_data_to_send(thread_comm_t* tc, mpi_data_t* comm,
  uint64_t vertex_id, uint64_t data_val, int32_t send_rank)
{
  tc->sendbuf_vert_thread[tc->thread_queue_size] = vertex_id;
  tc->sendbuf_data_thread[tc->thread_queue_size] = data_val;
  tc->sendbuf_rank_thread[tc->thread_queue_size] = send_rank;
  ++tc->thread_queue_size;
  ++tc->sendcounts_thread[send_rank];

  if (tc->thread_queue_size == THREAD_QUEUE_SIZE)
    empty_vid_data(tc, comm);
}

inline void add_vid_data_to_send_flt(thread_comm_t* tc, mpi_data_t* comm,
  uint64_t vertex_id, double data_val, int32_t send_rank)
{
  tc->sendbuf_vert_thread[tc->thread_queue_size] = vertex_id;
  tc->sendbuf_data_thread_flt[tc->thread_queue_size] = data_val;
  tc->sendbuf_rank_thread[tc->thread_queue_size] = send_rank;
  ++tc->thread_queue_size;
  ++tc->sendcounts_thread[send_rank];

  if (tc->thread_queue_size == THREAD_QUEUE_SIZE)
    empty_vid_data_flt(tc, comm);
}

inline void empty_vid_data(thread_comm_t* tc, mpi_data_t* comm)
{
  for (int32_t i = 0; i < nprocs; ++i)
  {
#pragma omp atomic capture
    tc->thread_starts[i] = comm->sdispls_cpy_temp[i] += tc->sendcounts_thread[i];

    tc->thread_starts[i] -= tc->sendcounts_thread[i];
  }

  for (uint64_t i = 0; i < tc->thread_queue_size; ++i)
  {
    int32_t cur_rank = tc->sendbuf_rank_thread[i];
    comm->sendbuf_vert[tc->thread_starts[cur_rank]] = 
      tc->sendbuf_vert_thread[i];
    comm->sendbuf_data[tc->thread_starts[cur_rank]] = 
      tc->sendbuf_data_thread[i];
    ++tc->thread_starts[cur_rank];
  }
  
  for (int32_t i = 0; i < nprocs; ++i)
  {
    tc->thread_starts[i] = 0;
    tc->sendcounts_thread[i] = 0;
  }
  tc->thread_queue_size = 0;
}

inline void empty_vid_data_flt(thread_comm_t* tc, mpi_data_t* comm)
{
  for (int32_t i = 0; i < nprocs; ++i)
  {
#pragma omp atomic capture
    tc->thread_starts[i] = comm->sdispls_cpy_temp[i] += tc->sendcounts_thread[i];

    tc->thread_starts[i] -= tc->sendcounts_thread[i];
  }

  for (uint64_t i = 0; i < tc->thread_queue_size; ++i)
  {
    int32_t cur_rank = tc->sendbuf_rank_thread[i];
    comm->sendbuf_vert[tc->thread_starts[cur_rank]] = 
      tc->sendbuf_vert_thread[i];
    comm->sendbuf_data_flt[tc->thread_starts[cur_rank]] = 
      tc->sendbuf_data_thread_flt[i];
    ++tc->thread_starts[cur_rank];
  }
  
  for (int32_t i = 0; i < nprocs; ++i)
  {
    tc->thread_starts[i] = 0;
    tc->sendcounts_thread[i] = 0;
  }
  tc->thread_queue_size = 0;
}



#endif
