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
#include "harmonic.h"

#define HC_NOT_VISITED 18446744073709551615
#define HC_VISITED 18446744073709551614

extern int procid, nprocs;
extern bool verbose, debug, verify;


int run_harmonic(dist_graph_t* g, mpi_data_t* comm, queue_data_t* q,
            uint64_t* distances, uint64_t root)
{  
  if (debug) { printf("Task %d run_harmonic() start\n", procid); }
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

  uint64_t distance = 0;
  comm->global_queue_size = 1;
#pragma omp parallel default(shared)
{
  thread_queue_t tq;
  init_thread_queue(&tq);  

#pragma omp for
  for (uint64_t i = 0; i < g->n_total; ++i)
    distances[i] = HC_NOT_VISITED;

  while (comm->global_queue_size)
  {
    if (debug && tq.tid == 0) { 
      printf("Task %d Distance %u run_harmonic() GQ: %li, TQ: %u\n", 
        procid, distance, comm->global_queue_size, q->queue_size); 
    }

#pragma omp for schedule(guided) nowait
    for (int64_t i = 0; i < q->queue_size; ++i)
    {
      uint64_t vert = q->queue[i];
      uint64_t vert_index = get_value(&g->map, vert);
      if (distances[vert_index] != HC_NOT_VISITED && 
          distances[vert_index] != HC_VISITED)
        continue;
      distances[vert_index] = distance;

      uint64_t in_degree = in_degree(g, vert_index);
      uint64_t* ins = in_vertices(g, vert_index);
      for (uint64_t j = 0; j < in_degree; ++j)
      {
        uint64_t in_index = ins[j];
        if (distances[in_index] == HC_NOT_VISITED)
        {
          distances[in_index] = HC_VISITED;

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
      ++distance;
    }
  } // end while

  clear_thread_queue(&tq);
} // end parallel

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d run_harmonic() time %9.6f (s)\n", procid, elt);
  }
  if (debug) { printf("Task %d run_harmonic() success\n", procid); }

  return 0;
}

int harmonic_output(dist_graph_t* g, uint64_t num_to_output,
               uint64_t* input_list, double* centralities,
               char* output_file)
{
  if (verbose) printf("Task %d centralities to %s\n", procid, output_file); 

  if (procid == 0)
  {     
    std::ofstream outfile;
    outfile.open(output_file);

    outfile << "Vertex, Centrality" << std::endl;

    for (uint64_t i = 0; i < num_to_output; ++i)
      outfile << input_list[i] << "," << centralities[i] << std::endl;

    outfile.close();
  }

  if (verbose) printf("Task %d done writing centralities\n", procid); 

  return 0;
}

double harmonic_calc(dist_graph_t* g, uint64_t* distances, uint64_t root)
{
  double my_hc = 0.0;
  uint64_t count = 0;
  for (uint64_t i = 0; i < g->n_local; ++i)
    if (distances[i] > 0 && 
        distances[i] != HC_VISITED && distances[i] != HC_NOT_VISITED)
    {
      my_hc += (1.0 / (double)distances[i]);
      ++count;
    }

  double global_hc = 0.0;
  MPI_Allreduce(&my_hc, &global_hc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  if (procid == 0)
    printf("Vertex %lu - Harmonic Centrality %lf\n", root, global_hc);

  return global_hc;
}

int harmonic_dist(dist_graph_t *g, mpi_data_t* comm, queue_data_t* q, 
                  char* output_file, 
                  uint64_t num_to_output, uint64_t* input_list)
{  
  if (debug) { printf("Task %d harmonic_dist() start\n", procid); }

  MPI_Barrier(MPI_COMM_WORLD);
  double elt = omp_get_wtime();

  uint64_t* distances = (uint64_t*)malloc(g->n_total*sizeof(uint64_t));  
  double* centralities = (double*)malloc(num_to_output*sizeof(double));  

  for (uint64_t i = 0; i < num_to_output; ++i)
  {
    uint64_t root = input_list[i];
    run_harmonic(g, comm, q, distances, root); 
    centralities[i] = harmonic_calc(g, distances, root);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  elt = omp_get_wtime() - elt;
  if (procid == 0) printf("Harmonic time %9.6f (s)\n", elt);

  if (output) {
    harmonic_output(g, num_to_output, input_list, centralities, output_file);
  }

  free(distances);

  if (debug)  printf("Task %d harmonic_dist() success\n", procid); 
  return 0;
}

