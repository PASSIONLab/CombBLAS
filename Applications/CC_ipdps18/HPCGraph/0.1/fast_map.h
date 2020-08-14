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

#ifndef _FAST_MAP_H_
#define _FAST_MAP_H_

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>

extern int procid, nprocs;
extern bool verbose, debug, verify, output;

#define NULL_KEY 18446744073709551615

struct fast_map {
  uint64_t* arr;
  uint64_t* unique_keys;
  uint64_t* unique_indexes;
  uint64_t capacity;
  uint64_t num_unique;
  bool hashing;
} ;

bool is_init(fast_map* map);
void init_map(fast_map* map);
void init_map(fast_map* map, uint64_t init_size);
void init_map_nohash(fast_map* map, uint64_t init_size);
void clear_map(fast_map* map);
void empty_map(fast_map* map);

inline uint64_t mult_hash(fast_map* map, uint64_t key);
inline void set_value(fast_map* map, uint64_t key, uint64_t value);
inline void set_value_uq(fast_map* map, uint64_t key, uint64_t value);
inline uint64_t get_value(fast_map* map, uint64_t key);
inline uint64_t get_max_key(fast_map* map);

inline uint64_t mult_hash(fast_map* map, uint64_t key)
{
  if (map->hashing)
    return (key*2654435761 % map->capacity);
  else
    return key;
}

inline void set_value(fast_map* map, uint64_t key, uint64_t value)
{
  uint64_t cur_index = mult_hash(map, key)*2;
  uint64_t count = 0;
  while (map->arr[cur_index] != key && map->arr[cur_index] != NULL_KEY)
  {
    cur_index = (cur_index + 2) % (map->capacity*2);
    ++count;
    if (debug && count % 100 == 0)
      fprintf(stderr, "Warning: fast_map set_value(): Big Count %d -- %lu - %lu, %lu, %lu\n", 
              procid, count, cur_index, key, value);
  }
  if (map->arr[cur_index] == NULL_KEY)
  {  
    map->arr[cur_index] = key;
  }
  map->arr[cur_index+1] = value;
}

inline void set_value_uq(fast_map* map, uint64_t key, uint64_t value)
{
  uint64_t cur_index = mult_hash(map, key)*2;
  uint64_t count = 0;
  while (map->arr[cur_index] != key && map->arr[cur_index] != NULL_KEY)
  {
    cur_index = (cur_index + 2) % (map->capacity*2);
    ++count;
    if (debug && count % 100 == 0)
      fprintf(stderr, "Warning: fast_map set_value_uq(): Big Count %d -- %lu - %lu, %lu, %lu\n", 
              procid, count, cur_index, key, value);
  }
  if (map->arr[cur_index] == NULL_KEY)
  {  
    map->arr[cur_index] = key;
    map->unique_keys[map->num_unique] = key;
    map->unique_indexes[map->num_unique] = cur_index;
    ++map->num_unique;
  }
  map->arr[cur_index+1] = value;
}


inline uint64_t get_value(fast_map* map, uint64_t key)
{
  uint64_t cur_index = mult_hash(map, key)*2;
  while (map->arr[cur_index] != key && map->arr[cur_index] != NULL_KEY)
    cur_index = (cur_index + 2) % (map->capacity*2);
  if (map->arr[cur_index] == NULL_KEY)
    return NULL_KEY;
  else
    return map->arr[cur_index+1];
}

inline uint64_t get_max_key(fast_map* map)
{
  uint64_t max_val = 0;
  uint64_t max_key = NULL_KEY;
  std::vector<uint64_t> vec;
  for (uint64_t i = 0; i < map->num_unique; ++i)
    if (map->arr[map->unique_indexes[i]+1] > max_val)
    {
      max_val = map->arr[map->unique_indexes[i]+1];
      vec.clear();
      vec.push_back(map->arr[map->unique_indexes[i]]);
    }
    else if (map->arr[map->unique_indexes[i]+1] == max_val)
    {
      vec.push_back(map->arr[map->unique_indexes[i]]);
    }

  if (vec.size() > 0)
    max_key = vec[(int)rand() % vec.size()];

  return max_key;
}


#endif
