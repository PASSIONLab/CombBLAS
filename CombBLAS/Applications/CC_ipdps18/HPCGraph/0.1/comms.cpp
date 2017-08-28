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

#include "comms.h"
#include "util.h"

extern int procid, nprocs;
extern bool verbose, debug, verify, output;

void init_queue_data(dist_graph_t* g, queue_data_t* q)
{  
  if (debug) { printf("Task %d init_queue_data() start\n", procid); }

  uint64_t queue_size = g->m_local_in + g->m_local_out;
  q->queue = (uint64_t*)malloc(queue_size*sizeof(uint64_t));
  q->queue_next = (uint64_t*)malloc(queue_size*sizeof(uint64_t));
  q->queue_send = (uint64_t*)malloc(queue_size*sizeof(uint64_t));
  if (q->queue == NULL || q->queue_next == NULL || q->queue_send == NULL)
    throw_err("init_queue_data(), unable to allocate resources\n", procid);

  q->queue_size = 0;
  q->next_size = 0;
  q->send_size = 0;  
  if (debug) { printf("Task %d init_queue_data() success\n", procid); }
}

void clear_queue_data(queue_data_t* q)
{
  if (debug) { printf("Task %d clear_queque_data() start\n", procid); }
  free(q->queue);
  free(q->queue_next);
  free(q->queue_send);
  if (debug) { printf("Task %d clear_queque_data() success\n", procid); }
}

void init_comm_data(mpi_data_t* comm)
{
  if (debug) { printf("Task %d init_comm_data() start\n", procid); }

  comm->sendcounts = (int32_t*)malloc(nprocs*sizeof(int32_t));
  comm->sendcounts_temp = (uint64_t*)malloc(nprocs*sizeof(uint64_t));
  comm->recvcounts = (int32_t*)malloc(nprocs*sizeof(int32_t));
  comm->recvcounts_temp = (uint64_t*)malloc(nprocs*sizeof(uint64_t));
  comm->sdispls = (int32_t*)malloc(nprocs*sizeof(int32_t));
  comm->rdispls = (int32_t*)malloc(nprocs*sizeof(int32_t));
  comm->rdispls_temp = (uint64_t*)malloc(nprocs*sizeof(uint64_t));
  comm->sdispls_cpy = (int32_t*)malloc(nprocs*sizeof(int32_t));
  comm->sdispls_temp = (uint64_t*)malloc(nprocs*sizeof(int64_t));
  comm->sdispls_cpy_temp = (uint64_t*)malloc(nprocs*sizeof(int64_t));

  if (comm->sendcounts == NULL || comm->sendcounts_temp == NULL ||
      comm->recvcounts == NULL || comm->sdispls == NULL || 
      comm->rdispls == NULL || comm->rdispls_temp == NULL ||
      comm->sdispls_cpy == NULL || comm->sdispls_cpy_temp == NULL)
    throw_err("init_comm_data(), unable to allocate resources\n", procid);

  comm->total_recv = 0;
  comm->total_send = 0;
  comm->global_queue_size = 0;
  if (debug) { printf("Task %d init_comm_data() success\n", procid); }
}

void clear_comm_data(mpi_data_t* comm)
{
  if (debug) { printf("Task %d clear_comm_data() start\n", procid); }
  free(comm->sendcounts);
  free(comm->sendcounts_temp);
  free(comm->recvcounts);
  free(comm->recvcounts_temp);
  free(comm->sdispls);
  free(comm->rdispls);
  free(comm->sdispls_cpy);
  free(comm->sdispls_temp);
  if (debug) { printf("Task %d clear_comm_data() success\n", procid); }
}

void clear_thread_queue_comm_data(mpi_data_t* comm)
{
  if (debug) { printf("Task %d clear_thread_queue_comm_data() start\n", procid); }
  free(comm->sendcounts);
  free(comm->recvcounts);
  free(comm->sdispls);
  free(comm->rdispls);
  free(comm->sdispls_cpy);
  if (debug) { printf("Task %d clear_thread_queue_comm_data() success\n", procid); }
}

void init_thread_queue(thread_queue_t* tq)
{
  tq->tid = omp_get_thread_num();
  if (debug) { 
    printf("Task %d Thread %d init_thread_queue() start\n", procid, tq->tid); 
  }

  tq->thread_queue = (uint64_t*)malloc(THREAD_QUEUE_SIZE*sizeof(uint64_t));
  tq->thread_send = (uint64_t*)malloc(THREAD_QUEUE_SIZE*sizeof(uint64_t));
  if (tq->thread_queue == NULL || tq->thread_send == NULL)
    throw_err("init_thread_queue(), unable to allocate resources\n", procid, tq->tid);

  tq->tid = omp_get_thread_num();
  tq->thread_queue_size = 0;
  tq->thread_send_size = 0;
  if (debug) { 
    printf("Task %d Thread %d init_thread_queue() success\n", procid, tq->tid); 
  }
  
}

void clear_thread_queue(thread_queue_t* tq)
{  
  if (debug) { 
    printf("Task %d Thread %d clear_thread_queue() start\n", procid, tq->tid); 
  }
  free(tq->thread_queue);
  free(tq->thread_send);
  if (debug) { 
    printf("Task %d Thread %d clear_thread_queue() success\n", procid, tq->tid); 
  }
}

void init_thread_comm(thread_comm_t* tc)
{
  tc->tid = omp_get_thread_num();
  if (debug) { 
    printf("Task %d Thread %d init_thread_comm() start\n", procid, tc->tid); 
  }

  tc->v_to_rank = (bool*)malloc(nprocs*sizeof(bool));
  tc->sendcounts_thread = (uint64_t*)malloc(nprocs*sizeof(uint64_t));
  tc->sendbuf_vert_thread = 
    (uint64_t*)malloc(THREAD_QUEUE_SIZE*sizeof(uint64_t));
  tc->sendbuf_data_thread = 
    (uint64_t*)malloc(THREAD_QUEUE_SIZE*sizeof(uint64_t));
  tc->sendbuf_rank_thread = (int32_t*)malloc(THREAD_QUEUE_SIZE*sizeof(int32_t));
  tc->thread_starts = (uint64_t*)malloc(nprocs*sizeof(uint64_t));
  if (tc->v_to_rank == NULL || tc->sendcounts_thread == NULL || 
      tc->sendbuf_vert_thread == NULL || tc->sendbuf_data_thread == NULL || 
      tc->sendbuf_rank_thread == NULL || tc->thread_starts == NULL)
    throw_err("init_thread_comm(), unable to allocate resources\n", procid, tc->tid);

  for (int32_t i = 0; i < nprocs; ++i)
    tc->sendcounts_thread[i] = 0;

  tc->thread_queue_size = 0;

  if (debug) { 
    printf("Task %d Thread %d init_thread_comm() success\n", procid, tc->tid); 
  }
}

void clear_thread_comm(thread_comm_t* tc)
{
  free(tc->v_to_rank);
  free(tc->sendcounts_thread);
  free(tc->sendbuf_vert_thread);
  free(tc->sendbuf_data_thread);
  free(tc->sendbuf_rank_thread);
  free(tc->thread_starts);
}

void init_thread_comm_flt(thread_comm_t* tc)
{
  tc->tid = omp_get_thread_num();
  if (debug) { 
    printf("Task %d Thread %d init_thread_comm_flt() start\n", procid, tc->tid); 
  }

  tc->v_to_rank = (bool*)malloc(nprocs*sizeof(bool));
  tc->sendcounts_thread = (uint64_t*)malloc(nprocs*sizeof(uint64_t));
  tc->sendbuf_vert_thread = 
    (uint64_t*)malloc(THREAD_QUEUE_SIZE*sizeof(uint64_t));
  tc->sendbuf_data_thread = NULL;
  tc->sendbuf_data_thread_flt = 
    (double*)malloc(THREAD_QUEUE_SIZE*sizeof(double));
  tc->sendbuf_rank_thread = (int32_t*)malloc(THREAD_QUEUE_SIZE*sizeof(int32_t));
  tc->thread_starts = (uint64_t*)malloc(nprocs*sizeof(uint64_t));
  if (tc->v_to_rank == NULL || tc->sendcounts_thread == NULL || 
      tc->sendbuf_vert_thread == NULL || tc->sendbuf_data_thread_flt == NULL || 
      tc->sendbuf_rank_thread == NULL || tc->thread_starts == NULL)
    throw_err("init_thread_comm(), unable to allocate resources\n", procid, tc->tid);

  tc->thread_queue_size = 0;

  for (int32_t i = 0; i < nprocs; ++i)
    tc->sendcounts_thread[i] = 0;

  if (debug) { 
    printf("Task %d Thread %d init_thread_comm_flt() success\n", procid, tc->tid); 
  }
}

void clear_thread_comm_flt(thread_comm_t* tc)
{
  free(tc->v_to_rank);
  free(tc->sendcounts_thread);
  free(tc->sendbuf_vert_thread);
  free(tc->sendbuf_data_thread);
  free(tc->sendbuf_rank_thread);
  free(tc->thread_starts);
}


void init_sendbuf_vid_data(mpi_data_t* comm)
{
  if (debug) { printf("Task %d init_sendbuf_vid_data() start\n", procid); }
  
  comm->sdispls_temp[0] = 0;
  comm->sdispls_cpy_temp[0] = 0;
  for (int32_t i = 1; i < nprocs; ++i)
  {
    comm->sdispls_temp[i] = comm->sdispls_temp[i-1] + comm->sendcounts_temp[i-1];
    comm->sdispls_cpy_temp[i] = comm->sdispls_temp[i];
  }

  comm->total_send = comm->sdispls_temp[nprocs-1] + comm->sendcounts_temp[nprocs-1];
  comm->sendbuf_vert = (uint64_t*)malloc(comm->total_send*sizeof(uint64_t));
  comm->sendbuf_data = (uint64_t*)malloc(comm->total_send*sizeof(uint64_t));
  comm->sendbuf_data_flt = NULL; 
  if (comm->sendbuf_vert == NULL || comm->sendbuf_data == NULL)
    throw_err("init_sendbuf_vid_data(), unable to allocate resources\n", procid);

  comm->global_queue_size = 0;
  uint64_t task_queue_size = comm->total_send;
  MPI_Allreduce(&task_queue_size, &comm->global_queue_size, 1, 
                MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

  if (debug) { printf("Task %d init_sendbuf_vid_data() success\n", procid); }
}

void init_recvbuf_vid_data(mpi_data_t* comm)
{
  if (debug) { printf("Task %d init_recvbuf_vid_data() start\n", procid); }

  for (int32_t i = 0; i < nprocs; ++i)
    comm->recvcounts_temp[i] = 0;

  MPI_Alltoall(comm->sendcounts_temp, 1, MPI_UINT64_T, 
               comm->recvcounts_temp, 1, MPI_UINT64_T, MPI_COMM_WORLD);

  comm->rdispls_temp[0] = 0;

  for (int i = 1; i < nprocs; ++i)
  {
    comm->rdispls_temp[i] = comm->rdispls_temp[i-1] + comm->recvcounts_temp[i-1];
  }

  comm->total_recv = comm->rdispls_temp[nprocs-1] + comm->recvcounts_temp[nprocs-1];
  comm->recvbuf_vert = (uint64_t*)malloc(comm->total_recv*sizeof(uint64_t));
  comm->recvbuf_data = (uint64_t*)malloc(comm->total_recv*sizeof(uint64_t));
  comm->recvbuf_data_flt = NULL;
  if (comm->recvbuf_vert == NULL || comm->recvbuf_data == NULL)
    throw_err("init_recvbuf_vid_data() unable to allocate comm buffers", procid);

  if (debug) { printf("Task %d init_recvbuf_vid_data() success\n", procid); }
}

void init_sendbuf_vid_data_flt(mpi_data_t* comm)
{
  if (debug) { printf("Task %d init_sendbuf_vid_data_flt() start\n", procid); }
  
  comm->sdispls_temp[0] = 0;
  comm->sdispls_cpy_temp[0] = 0;
  for (int32_t i = 1; i < nprocs; ++i)
  {
    comm->sdispls_temp[i] = comm->sdispls_temp[i-1] + comm->sendcounts_temp[i-1];
    comm->sdispls_cpy_temp[i] = comm->sdispls_temp[i];
  }

  comm->total_send = comm->sdispls_temp[nprocs-1] + comm->sendcounts_temp[nprocs-1];
  comm->sendbuf_vert = (uint64_t*)malloc(comm->total_send*sizeof(uint64_t));
  comm->sendbuf_data = NULL;
  comm->sendbuf_data_flt = (double*)malloc(comm->total_send*sizeof(double)); 
  if (comm->sendbuf_vert == NULL || comm->sendbuf_data_flt == NULL)
    throw_err("init_sendbuf_vid_data_flt(), unable to allocate resources\n", procid);

  comm->global_queue_size = 0;
  uint64_t task_queue_size = comm->total_send;
  MPI_Allreduce(&task_queue_size, &comm->global_queue_size, 1, 
                MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

  if (debug) { printf("Task %d init_sendbuf_vid_data_flt() success\n", procid); }
}

void init_recvbuf_vid_data_flt(mpi_data_t* comm)
{
  if (debug) { printf("Task %d init_recvbuf_vid_data_flt() start\n", procid); }

  for (int32_t i = 0; i < nprocs; ++i)
    comm->recvcounts_temp[i] = 0;

  MPI_Alltoall(comm->sendcounts_temp, 1, MPI_UINT64_T, 
               comm->recvcounts_temp, 1, MPI_UINT64_T, MPI_COMM_WORLD);

  comm->rdispls_temp[0] = 0;

  for (int i = 1; i < nprocs; ++i)
  {
    comm->rdispls_temp[i] = comm->rdispls_temp[i-1] + comm->recvcounts_temp[i-1];
  }

  comm->total_recv = comm->rdispls_temp[nprocs-1] + comm->recvcounts_temp[nprocs-1];
  comm->recvbuf_vert = (uint64_t*)malloc(comm->total_recv*sizeof(uint64_t));
  comm->recvbuf_data = NULL;
  comm->recvbuf_data_flt = (double*)malloc(comm->total_recv*sizeof(double));
  if (comm->recvbuf_vert == NULL || comm->recvbuf_data_flt == NULL)
    throw_err("init_recvbuf_vid_data_flt() unable to allocate comm buffers", procid);

  if (debug) { printf("Task %d init_recvbuf_vid_data_flt() success\n", procid); }
}

void clear_recvbuf_vid_data(mpi_data_t* comm)
{
  free(comm->recvbuf_vert);
  free(comm->recvbuf_data);

  for (int32_t i = 0; i < nprocs; ++i)
    comm->sendcounts[i] = 0;
  for (int32_t i = 0; i < nprocs; ++i)
    comm->sendcounts_temp[i] = 0;
}

void clear_allbuf_vid_data(mpi_data_t* comm)
{  
  free(comm->sendbuf_vert);
  free(comm->recvbuf_vert);

  if (comm->sendbuf_data != NULL)
    free(comm->sendbuf_data);
  if (comm->recvbuf_data != NULL)
    free(comm->recvbuf_data);
  if (comm->sendbuf_data_flt != NULL)
    free(comm->sendbuf_data_flt);
  if (comm->recvbuf_data_flt != NULL)
    free(comm->recvbuf_data_flt);

  for (int32_t i = 0; i < nprocs; ++i)
    comm->sendcounts[i] = 0;
  for (int32_t i = 0; i < nprocs; ++i)
    comm->sendcounts_temp[i] = 0;
}
