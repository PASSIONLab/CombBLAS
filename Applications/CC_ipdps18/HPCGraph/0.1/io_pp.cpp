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

#include "io_pp.h"
#include "comms.h"
#include "util.h"

extern int procid, nprocs;
extern bool verbose, debug, verify, output;


int load_graph_edges_32(char *input_filename, graph_gen_data_t *ggi) 
{  
  if (debug) { 
    printf("Task %d load_graph_edges() %s start\n", procid, input_filename); 
  }

  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  FILE *infp = fopen(input_filename, "rb");
  if(infp == NULL)
    throw_err("load_graph_edges() unable to open input file", procid);

  fseek(infp, 0L, SEEK_END);
  uint64_t file_size = ftell(infp);
  fseek(infp, 0L, SEEK_SET);

  uint64_t nedges_global = file_size/(2*sizeof(uint32_t));
  ggi->m = nedges_global;

  uint64_t read_offset_start = procid*2*sizeof(uint32_t)*(nedges_global/nprocs);
  uint64_t read_offset_end = (procid+1)*2*sizeof(uint32_t)*(nedges_global/nprocs);

  if (procid == nprocs - 1)
    read_offset_end = 2*sizeof(uint32_t)*nedges_global;

  uint64_t nedges = (read_offset_end - read_offset_start)/8;
  ggi->m_local_read = nedges;

  if (debug) {
    printf("Task %d, read_offset_start %ld, read_offset_end %ld, nedges_global %ld, nedges: %ld\n", procid, read_offset_start, read_offset_end, nedges_global, nedges);
  }

  uint32_t* gen_edges_read = (uint32_t*)malloc(2*nedges*sizeof(uint32_t));
  uint64_t* gen_edges = (uint64_t*)malloc(2*nedges*sizeof(uint64_t));
  if (gen_edges_read == NULL || gen_edges == NULL)
    throw_err("load_graph_edges(), unable to allocate buffer", procid);

  fseek(infp, read_offset_start, SEEK_SET);
  fread(gen_edges_read, nedges, 2*sizeof(uint32_t), infp);
  fclose(infp);

  for (uint64_t i = 0; i < nedges*2; ++i)
    gen_edges[i] = (uint64_t)gen_edges_read[i];

  free(gen_edges_read);
  ggi->gen_edges = gen_edges;

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d read %lu edges, %9.6f (s)\n", procid, nedges, elt);
  }
  
  uint64_t max_n = 0;
  for (uint64_t i = 0; i < ggi->m_local_read*2; ++i)
    if (gen_edges[i] > max_n)
      max_n = gen_edges[i];

  uint64_t n_global;
  MPI_Allreduce(&max_n, &n_global, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
  
  ggi->n = n_global+1;
  ggi->n_offset = procid*(ggi->n/nprocs + 1);
  ggi->n_local = ggi->n/nprocs + 1;
  if (procid == nprocs - 1)
    ggi->n_local = n_global - ggi->n_offset + 1; 

  if (verbose) {
    printf("Task %d, n %lu, n_offset %lu, n_local %lu\n", 
           procid, ggi->n, ggi->n_offset, ggi->n_local);
  }

  if (debug) { printf("Task %d load_graph_edges() success\n", procid); }
  return 0;
}


int load_graph_edges_64(char *input_filename, graph_gen_data_t *ggi) 
{  
  if (debug) { printf("Task %d load_graph_edges() start\n", procid); }

  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  FILE *infp = fopen(input_filename, "rb");
  if(infp == NULL)
    throw_err("load_graph_edges() unable to open input file", procid);

  fseek(infp, 0L, SEEK_END);
  uint64_t file_size = ftell(infp);
  fseek(infp, 0L, SEEK_SET);

  uint64_t nedges_global = file_size/(2*sizeof(uint64_t));
  ggi->m = nedges_global;

  uint64_t read_offset_start = procid*2*sizeof(uint64_t)*(nedges_global/nprocs);
  uint64_t read_offset_end = (procid+1)*2*sizeof(uint64_t)*(nedges_global/nprocs);

  if (procid == nprocs - 1)
    read_offset_end = 2*sizeof(uint64_t)*nedges_global;

  uint64_t nedges = (read_offset_end - read_offset_start)/8;
  ggi->m_local_read = nedges;

  if (debug) {
    printf("Task %d, read_offset_start %ld, read_offset_end %ld, nedges_global %ld, nedges: %ld\n", procid, read_offset_start, read_offset_end, nedges_global, nedges);
  }

  uint64_t* gen_edges = (uint64_t*)malloc(2*nedges*sizeof(uint64_t));
  if (gen_edges == NULL)
    throw_err("load_graph_edges(), unable to allocate buffer", procid);

  fseek(infp, read_offset_start, SEEK_SET);
  fread(gen_edges, nedges, 2*sizeof(uint64_t), infp);
  fclose(infp);

  ggi->gen_edges = gen_edges;

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d read %ld edges, %9.6f (s)\n", procid, nedges, elt);
  }
  
  uint64_t max_n = 0;
  for (uint64_t i = 0; i < ggi->m_local_read*2; ++i)
    if (gen_edges[i] > max_n)
      max_n = gen_edges[i];

  uint64_t n_global;
  MPI_Allreduce(&max_n, &n_global, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
  
  ggi->n = n_global+1;
  ggi->n_offset = procid*(ggi->n/nprocs + 1);
  ggi->n_local = ggi->n/nprocs + 1;
  if (procid == nprocs - 1)
    ggi->n_local = n_global - ggi->n_offset + 1; 

  if (verbose) {
    printf("Task %d, n %lu, n_offset %lu, n_local %lu\n", 
           procid, ggi->n, ggi->n_offset, ggi->n_local);
  }

  if (debug) { printf("Task %d load_graph_edges() success\n", procid); }
  return 0;
}

int exchange_out_edges(graph_gen_data_t *ggi, mpi_data_t* comm)
{
  if (debug) { printf("Task %d exchange_out_edges() start\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  uint64_t* temp_sendcounts = (uint64_t*)malloc(nprocs*sizeof(uint64_t));
  uint64_t* temp_recvcounts = (uint64_t*)malloc(nprocs*sizeof(uint64_t));
  for (int i = 0; i < nprocs; ++i)
  {
    temp_sendcounts[i] = 0;
    temp_recvcounts[i] = 0;
  }

  uint64_t n_per_rank = ggi->n / nprocs + 1;
  for (uint64_t i = 0; i < ggi->m_local_read*2; i+=2)
  {
    uint64_t vert = ggi->gen_edges[i];
    int32_t vert_task = (int32_t)(vert / n_per_rank);
    temp_sendcounts[vert_task] += 2;
  }

  MPI_Alltoall(temp_sendcounts, 1, MPI_UINT64_T, 
               temp_recvcounts, 1, MPI_UINT64_T, MPI_COMM_WORLD);
  
  uint64_t total_recv = 0;
  uint64_t total_send = 0;
  for (int32_t i = 0; i < nprocs; ++i)
  {
    total_recv += temp_recvcounts[i];
    total_send += temp_sendcounts[i];
  }
  free(temp_sendcounts);
  free(temp_recvcounts);

  uint64_t* recvbuf = (uint64_t*)malloc(total_recv*sizeof(uint64_t));
  if (recvbuf == NULL)
  { 
    fprintf(stderr, "Task %d Error: exchange_out_edges(), unable to allocate buffer\n", procid);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }  

  uint64_t max_transfer = total_send > total_recv ? total_send : total_recv;
  uint64_t num_comms = max_transfer / (uint64_t)MAX_SEND_SIZE + 1;
  MPI_Allreduce(MPI_IN_PLACE, &num_comms, 1, 
                MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);

  if (debug) 
    printf("Task %d exchange_out_edges() num_comms %lu total_send %lu total_recv %lu\n", procid, num_comms, total_send, total_recv);

  uint64_t sum_recv = 0;
  for (uint64_t c = 0; c < num_comms; ++c)
  {
    uint64_t send_begin = (ggi->m_local_read * c) / num_comms;
    uint64_t send_end = (ggi->m_local_read * (c + 1)) / num_comms;
    if (c == (num_comms-1))
      send_end = ggi->m_local_read;

    for (int32_t i = 0; i < nprocs; ++i)
    {
      comm->sendcounts[i] = 0;
      comm->recvcounts[i] = 0;
    }

    for (int64_t i = send_begin; i < send_end; ++i)
    {
      uint64_t vert = ggi->gen_edges[i*2];
      int32_t vert_task = (int32_t)(vert / n_per_rank);
      comm->sendcounts[vert_task] += 2;
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
    uint64_t* sendbuf = (uint64_t*) malloc((uint64_t)cur_send*sizeof(uint64_t));
    if (sendbuf == NULL)
    { 
      fprintf(stderr, "Task %d Error: exchange_out_edges(), unable to allocate comm buffers", procid);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (uint64_t i = send_begin; i < send_end; ++i)
    {
      uint64_t vert1 = ggi->gen_edges[2*i];
      uint64_t vert2 = ggi->gen_edges[2*i+1];
      int32_t vert_task = (int32_t)(vert1 / n_per_rank);

      sendbuf[comm->sdispls_cpy[vert_task]++] = vert1; 
      sendbuf[comm->sdispls_cpy[vert_task]++] = vert2;
    }

    MPI_Alltoallv(sendbuf, comm->sendcounts, comm->sdispls, MPI_UINT64_T, 
                  recvbuf+sum_recv, comm->recvcounts, comm->rdispls,
                  MPI_UINT64_T, MPI_COMM_WORLD);
    sum_recv += cur_recv;
    free(sendbuf);
  }

  free(ggi->gen_edges);
  ggi->gen_edges = recvbuf;
  ggi->m_local_out = total_recv / 2;

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d exchange_out_edges() sent %lu, recv %lu, m_local_out %lu, %9.6f (s)\n", procid, total_send, total_recv, ggi->m_local_out, elt);
  }

  if (debug) { printf("Task %d exchange_out_edges() success\n", procid); }
  return 0;
}

int exchange_in_edges(graph_gen_data_t *ggi, mpi_data_t* comm)
{
  if (debug) { printf("Task %d exchange_in_edges() start\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  uint64_t* temp_sendcounts = (uint64_t*)malloc(nprocs*sizeof(uint64_t));
  uint64_t* temp_recvcounts = (uint64_t*)malloc(nprocs*sizeof(uint64_t));
  for (int i = 0; i < nprocs; ++i)
  {
    temp_sendcounts[i] = 0;
    temp_recvcounts[i] = 0;
  }

  uint64_t n_per_rank = ggi->n / nprocs + 1;
  for (uint64_t i = 0; i < ggi->m_local_out; ++i)
  {
    uint64_t vert = ggi->gen_edges[2*i+1];
    int32_t vert_task = (int32_t)(vert / n_per_rank);
    temp_sendcounts[vert_task] += 2;
  }

  MPI_Alltoall(temp_sendcounts, 1, MPI_UINT64_T, 
               temp_recvcounts, 1, MPI_UINT64_T, MPI_COMM_WORLD);
  
  uint64_t total_recv = 0;
  uint64_t total_send = 0;
  for (int32_t i = 0; i < nprocs; ++i)
  {
    total_recv += temp_recvcounts[i];
    total_send += temp_sendcounts[i];
  }
  free(temp_sendcounts);
  free(temp_recvcounts);

  uint64_t* recvbuf = (uint64_t*) malloc(total_recv*sizeof(uint64_t));
  if (recvbuf == NULL)
  { 
    fprintf(stderr, "Task %d Error: exchange_in_edges(), unable to allocate buffer\n", procid);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }  

  uint64_t max_transfer = total_send > total_recv ? total_send : total_recv;
  uint64_t num_comms = max_transfer / (uint64_t)MAX_SEND_SIZE + 1;
  MPI_Allreduce(MPI_IN_PLACE, &num_comms, 1, 
                MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);

  if (debug) 
    printf("Task %d exchange_in_edges() num_comms %li total_send %li total_recv %li\n", procid, num_comms, total_send, total_recv);

  uint64_t sum_recv = 0;
  for (int64_t c = 0; c < num_comms; ++c)
  {
    uint64_t send_begin = (ggi->m_local_out * c) / num_comms;
    uint64_t send_end = (ggi->m_local_out * (c + 1)) / num_comms;
    if (c == (num_comms-1))
      send_end = ggi->m_local_out;

    for (int32_t i = 0; i < nprocs; ++i)
    {
      comm->sendcounts[i] = 0;
      comm->recvcounts[i] = 0;
    }

    for (uint64_t i = send_begin; i < send_end; ++i)
    {
      uint64_t vert = ggi->gen_edges[i*2+1];
      int32_t vert_task = (int32_t)(vert / n_per_rank);
      comm->sendcounts[vert_task] += 2;
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
    uint64_t* sendbuf = (uint64_t*) malloc((uint64_t)cur_send*sizeof(uint64_t));
    if (sendbuf == NULL)
    { 
      fprintf(stderr, "Task %d Error: exchange_in_edges(), unable to allocate comm buffers\n", procid);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (uint64_t i = send_begin; i < send_end; ++i)
    {
      uint64_t vert1 = ggi->gen_edges[2*i];
      uint64_t vert2 = ggi->gen_edges[2*i+1];
      int32_t vert_task = (int32_t)(vert2 / n_per_rank);

      sendbuf[comm->sdispls_cpy[vert_task]++] = vert1; 
      sendbuf[comm->sdispls_cpy[vert_task]++] = vert2;
    }

    MPI_Alltoallv(sendbuf, comm->sendcounts, comm->sdispls, MPI_UINT64_T, 
                  recvbuf+sum_recv, comm->recvcounts, comm->rdispls,
                  MPI_UINT64_T, MPI_COMM_WORLD);
    sum_recv += cur_recv;
    free(sendbuf);
  }

  ggi->gen_edges_rev = recvbuf;
  ggi->m_local_in = total_recv / 2;

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d exchange_in_edges() sent %ld, recv %ld, m_local_out %ld, %9.6f (s)\n", procid, total_send, total_recv, ggi->m_local_out, elt);
  }

  if (debug) { printf("Task %d exchange_in_edges() success\n", procid); }
  return 0;
}
