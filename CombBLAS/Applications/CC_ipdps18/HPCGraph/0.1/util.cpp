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
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

#include "util.h"

extern int procid, nprocs;
extern bool verbose, debug, verify;

void throw_err(char const* err_message)
{
  fprintf(stderr, "Error: %s\n", err_message);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

void throw_err(char const* err_message, int32_t task)
{
  fprintf(stderr, "Task %d Error: %s\n", task, err_message);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

void throw_err(char const* err_message, int32_t task, int32_t thread)
{
  fprintf(stderr, "Task %d Thread %d Error: %s\n", task, thread, err_message);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

void quicksort_dec(uint64_t* arr1, uint64_t* arr2, int64_t left, int64_t right) 
{
  int64_t i = left;
  int64_t j = right;
  uint64_t temp; uint64_t temp2;
  uint64_t pivot = arr1[(left + right) / 2];

  while (i <= j) 
  {
    while (arr1[i] > pivot) {i++;}
    while (arr1[j] < pivot) {j--;}
  
    if (i <= j) 
    {
      temp = arr1[i];
      arr1[i] = arr1[j];
      arr1[j] = temp;
      temp2 = arr2[i];
      arr2[i] = arr2[j];
      arr2[j] = temp2;
      ++i;
      --j;
    }
  }

  if (j > left)
    quicksort_dec(arr1, arr2, left, j);
  if (i < right)
    quicksort_dec(arr1, arr2, i, right);
}


uint64_t* str_to_array(char *input_list_str, uint64_t* num)
{
  char *cp = strtok(input_list_str, ",");
  if (cp == NULL) {
    return (uint64_t*)malloc((*num)*sizeof(uint64_t));
  }

  int64_t my_index = -1;
  uint64_t n;
  if (sscanf(cp, "%lu", &n) == 1) {
      my_index = (int64_t)*num;
      *num += 1;
  } else {
      printf("Invalid integer token '%s'\n", cp);
  }
  uint64_t *array = str_to_array(NULL, num);
  if (my_index >= 0) {
      array[my_index] = n;
  }
  return array;
}
