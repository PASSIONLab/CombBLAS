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
#include <unordered_map>
#include <vector>

#include "dist_graph.h"
#include "comms.h"
#include "util.h"
#include "labelprop.h"

#define LABEL_NOT_ASSIGNED 18446744073709551615

extern int procid, nprocs;
extern bool verbose, debug, verify;


int run_labelprop(dist_graph_t* g, mpi_data_t* comm, 
                 uint64_t*& labels, uint32_t num_iter)
{ 
  if (debug) { printf("Task %d run_labelprop() start\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  uint64_t* labels_next = (uint64_t*)malloc(g->n_total*sizeof(uint64_t));

  for (int32_t i = 0; i < nprocs; ++i)
    comm->sendcounts_temp[i] = 0;

#pragma omp parallel default(shared)
{
  thread_comm_t tc;
  init_thread_comm(&tc);
  //std::unordered_map<uint64_t, uint64_t> map;
  //std::vector<uint64_t> max_labels;
  fast_map map;
  init_map(&map, (g->max_out_degree + g->max_in_degree)*2);

#pragma omp for
  for (uint64_t i = 0; i < g->n_local; ++i)
    labels[i] = g->local_unmap[i];
#pragma omp for
  for (uint64_t i = 0; i < g->n_local; ++i)
    labels_next[i] = g->local_unmap[i];
#pragma omp for
  for (uint64_t i = 0; i < g->n_ghost; ++i)
    labels[i+g->n_local] = g->ghost_unmap[i];
#pragma omp for
  for (uint64_t i = 0; i < g->n_ghost; ++i)
    labels_next[i+g->n_local] = g->ghost_unmap[i];


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
    update_vid_data_queues(g, &tc, comm, i, labels[i]);

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
    labels[index] = comm->recvbuf_data[i];
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
      printf("Task %d Iter %u run_labelprop()\n", procid, iter); 
    }

#pragma omp for schedule(guided) nowait
    for (uint64_t i = 0; i < g->n_local; ++i)
    {
      uint64_t vert_index = i;
      empty_map(&map);
      //map.clear();

      // uint64_t max_label = labels[vert_index];
      // uint64_t max_label_count = 0;

      uint64_t out_degree = out_degree(g, vert_index);
      uint64_t* outs = out_vertices(g, vert_index);
      for (uint64_t j = 0; j < out_degree; ++j)
      {
        uint64_t out_index = outs[j];
        uint64_t label_out = labels[out_index];
        // if (map.count(label_out) == 0)
        //   map[label_out] = 1;
        // else
        //   map[label_out] = map[label_out] + 1;
        
        // if (map[label_out] > max_label_count)
        // {
        //   max_label = label_out;
        //   max_labels.clear();
        //   max_labels.push_back(label_out);
        // }
        // else if (map[label_out] == max_label_count)
        // {
        //   max_labels.push_back(label_out);
        //   max_label = max_labels[rand() % max_labels.size()];
        // }
        uint64_t val = get_value(&map, label_out);
        if (val == NULL_KEY)
          set_value_uq(&map, label_out, 1);
        else
          set_value(&map, label_out, val+1);

      }
      uint64_t in_degree = in_degree(g, vert_index);
      uint64_t* ins = in_vertices(g, vert_index);
      for (uint64_t j = 0; j < in_degree; ++j)
      {
        uint64_t in_index = ins[j];
        uint64_t label_in = labels[in_index];
        // if (map.count(label_in) == 0)
        //   map[label_in] = 1;
        // else
        //   map[label_in] = map[label_in] + 1;

        // if (map[label_in] > max_label_count)
        // {
        //   max_label = label_in;
        //   max_labels.clear();
        //   max_labels.push_back(label_in);
        // }
        // else if (map[label_in] == max_label_count)
        // {
        //   max_labels.push_back(label_in);
        //   max_label = max_labels[rand() % max_labels.size()];
        // }
        uint64_t val = get_value(&map, label_in);
        if (val == NULL_KEY)
          set_value_uq(&map, label_in, 1);
        else
          set_value(&map, label_in, val+1);
      }
      // labels_next[vert_index] = max_label;

      labels_next[vert_index] = get_max_key(&map);
      if (labels_next[vert_index] == NULL_KEY)
        labels_next[vert_index] = labels[vert_index];
    }  

#pragma omp for
    for (uint64_t i = 0; i < comm->total_send; ++i)
      comm->sendbuf_data[i] = labels_next[comm->sendbuf_vert[i]];

#pragma omp single
{
    exchange_data(comm);
}

#pragma omp for
    for (uint64_t i = 0; i < comm->total_recv; ++i)
      labels_next[comm->recvbuf_vert[i]] = comm->recvbuf_data[i];

#pragma omp single
{
    uint64_t* temp = labels;
    labels = labels_next;
    labels_next = temp;
}
  } // end for loop

  clear_thread_comm(&tc);
} // end parallel

  clear_allbuf_vid_data(comm);
  free(labels_next);

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d, run_labelprop() time %9.6f (s)\n", procid, elt);
  }
  if (debug) { printf("Task %d run_labelprop() success\n", procid); }

  return 0;
}



int labelprop_output(dist_graph_t* g, uint64_t* labels, char* output_file)
{
  if (debug) printf("Task %d labels to %s\n", procid, output_file); 

  uint64_t* global_labels = (uint64_t*)malloc(g->n*sizeof(uint64_t));
  
#pragma omp parallel for
  for (uint64_t i = 0; i < g->n; ++i)
    global_labels[i] = LABEL_NOT_ASSIGNED;

#pragma omp parallel for
  for (uint64_t i = 0; i < g->n_local; ++i)
    global_labels[g->local_unmap[i]] = labels[i];


  if (procid == 0)
    MPI_Reduce(MPI_IN_PLACE, global_labels, (int32_t)g->n,
      MPI_UINT64_T, MPI_MIN, 0, MPI_COMM_WORLD);
  else
    MPI_Reduce(global_labels, global_labels, (int32_t)g->n,
      MPI_UINT64_T, MPI_MIN, 0, MPI_COMM_WORLD);

  if (procid == 0)
  {
    if (debug)
      for (uint64_t i = 0; i < g->n; ++i)
        if (global_labels[i] == LABEL_NOT_ASSIGNED)
        {
          printf("Labels error: %lu not assigned\n", i);
          global_labels[i] = 0;
        }
        
    std::ofstream outfile;
    outfile.open(output_file);

    for (uint64_t i = 0; i < g->n; ++i)
      outfile << global_labels[i] << std::endl;

    outfile.close();
  }

  free(global_labels);

  if (debug) printf("Task %d done writing labels\n", procid); 

  return 0;
}

int labelprop_verify(dist_graph_t* g, uint64_t* labels, uint64_t num_to_output)
{
  if (debug) { printf("Task %d labelprop_verify() start\n", procid); }

  uint64_t* global_labels = (uint64_t*)malloc(g->n*sizeof(uint64_t));
  uint64_t* label_counts = (uint64_t*)malloc(g->n*sizeof(uint64_t));
  
#pragma omp parallel for
  for (uint64_t i = 0; i < g->n; ++i)
    global_labels[i] = LABEL_NOT_ASSIGNED;

#pragma omp parallel for 
  for (uint64_t i = 0; i < g->n_local; ++i)
    global_labels[g->local_unmap[i]] = labels[i];

  if (procid == 0)
    MPI_Reduce(MPI_IN_PLACE, global_labels, (int32_t)g->n,
      MPI_UINT64_T, MPI_MIN, 0, MPI_COMM_WORLD);
  else
    MPI_Reduce(global_labels, global_labels, (int32_t)g->n,
      MPI_UINT64_T, MPI_MIN, 0, MPI_COMM_WORLD);

  if (procid == 0)
  {
#pragma omp parallel for
    for (uint64_t i = 0; i < g->n; ++i)
      label_counts[i] = 0;

    for (uint64_t i = 0; i < g->n; ++i)
      ++label_counts[global_labels[i]];

    quicksort_dec(label_counts, global_labels, (int64_t)0, (int64_t)g->n-1);

    for (uint64_t i = 0; i < num_to_output; ++i)
      printf("LP VERIFY %lu: label: %lu, size: %lu\n", i, global_labels[i], label_counts[i]);
  }

  free(global_labels);
  free(label_counts);

  if (debug) { printf("Task %d labelprop_verify() success\n", procid); }

  return 0;
}

int labelprop_dist(dist_graph_t* g, mpi_data_t* comm, 
                   uint32_t num_iter, char* output_file)
{  
  if (debug) { printf("Task %d labelprop_dist() start\n", procid); }

  MPI_Barrier(MPI_COMM_WORLD);
  double elt = omp_get_wtime();

  uint64_t* labels = (uint64_t*)malloc(g->n_total*sizeof(uint64_t));
  run_labelprop(g, comm, labels, num_iter);

  MPI_Barrier(MPI_COMM_WORLD);
  elt = omp_get_wtime() - elt;
  if (procid == 0) printf("LabelProp time %9.6f (s)\n", elt);

  if (output) {
    labelprop_output(g, labels, output_file);
  }

  if (verify) { 
     labelprop_verify(g, labels, 100);
  }

  free(labels);

  if (debug)  printf("Task %d labelprop_dist() success\n", procid); 
  return 0;
}

