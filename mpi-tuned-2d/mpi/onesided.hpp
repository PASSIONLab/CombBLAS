/* Copyright (C) 2011-2013 The Trustees of Indiana University.             */
/*                                                                         */
/* Use, modification and distribution is subject to the Boost Software     */
/* License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at */
/* http://www.boost.org/LICENSE_1_0.txt)                                   */
/*                                                                         */
/*  Authors: Jeremiah Willcock                                             */
/*           Andrew Lumsdaine                                              */

#include "common.hpp"
#include "bitmap.hpp"
#include <mpi.h>
// #include <nbc.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#ifndef ONESIDED_HPP
#define ONESIDED_HPP

/* Adapted from histogram_sort_inplace in
 * boost/graph/detail/histogram_sort.hpp; now done out-of-place for
 * performance. */
template <typename V1>
void histogram_sort_inplace1
       (int* restrict keys,
        const int* restrict rowstart,
        int numkeys,
        V1* restrict values1) {
  scoped_array<int> insert_positions;
  insert_positions.reset_to_new(numkeys);
  memcpy(insert_positions.get(0, numkeys), rowstart, numkeys * sizeof(int));
  assert (rowstart[numkeys] >= 0 && rowstart[numkeys] <= PTRDIFF_MAX);
  size_t element_count = size_t(rowstart[numkeys]);
  scoped_array<V1> values1_copy;
  values1_copy.reset_to_new(element_count);
  memcpy(values1_copy.get(0, element_count), values1, element_count * sizeof(V1));
// #pragma omp parallel for if(numkeys >= 16)
  for (ptrdiff_t i = 0; i < ptrdiff_t(element_count); ++i) {
    int key = keys[i];
    assert (key >= 0 && key < numkeys);
    // int target_pos = __sync_fetch_and_add(&insert_positions[key], 1);
    int target_pos = insert_positions[key]++;
    assert (target_pos < rowstart[key + 1]);
    values1[target_pos] = values1_copy[i];
  }
}

template <typename V1, typename V2>
void histogram_sort_inplace2
       (int* restrict keys,
        const int* restrict rowstart,
        int numkeys,
        V1* restrict values1,
        V2* restrict values2) {
  scoped_array<int> insert_positions;
  insert_positions.reset_to_new(numkeys);
  memcpy(insert_positions.get(0, numkeys), rowstart, numkeys * sizeof(int));
  assert (rowstart[numkeys] >= 0 && rowstart[numkeys] <= PTRDIFF_MAX);
  size_t element_count = size_t(rowstart[numkeys]);
  scoped_array<V1> values1_copy;
  values1_copy.reset_to_new(element_count);
  memcpy(values1_copy.get(0, element_count), values1, element_count * sizeof(V1));
  scoped_array<V2> values2_copy;
  values2_copy.reset_to_new(element_count);
  memcpy(values2_copy.get(0, element_count), values2, element_count * sizeof(V2));
// #pragma omp parallel for if(numkeys >= 16)
  for (ptrdiff_t i = 0; i < ptrdiff_t(element_count); ++i) {
    int key = keys[i];
    assert (key >= 0 && key < numkeys);
    // int target_pos = __sync_fetch_and_add(&insert_positions[key], 1);
    int target_pos = insert_positions[key]++;
    assert (target_pos < rowstart[key + 1]);
    values1[target_pos] = values1_copy[i];
    values2[target_pos] = values2_copy[i];
  }
}

/* Gather from one array into another. */
template <typename InElt, typename OutElt, typename F>
class gather {
  const InElt* input;
  size_t input_count;
  OutElt* output;
  size_t output_count;
  size_t nrequests_max;
  MPI_Datatype out_datatype;
  bool epoch_active;
  MPI_Comm comm;
  scoped_array<size_t> local_indices;
  scoped_array<int> remote_ranks;
  scoped_array<MPI_Aint> remote_indices;
  int comm_size;
  scoped_array<int> send_counts;
  scoped_array<int> send_offsets;
  scoped_array<int> recv_counts;
  scoped_array<int> recv_offsets;
  F f;

  public:
  gather(const InElt* input, size_t input_count, OutElt* output, size_t output_count, size_t nrequests_max, MPI_Datatype out_dt, const F& f, MPI_Comm comm)
    : input(input), input_count(input_count), output(output), output_count(output_count),
      nrequests_max(nrequests_max), out_datatype(out_dt), epoch_active(false), f(f)
  {
    MPI_Comm_dup(comm, &this->comm);
    this->local_indices.reset_to_new(nrequests_max);
    this->remote_ranks.reset_to_new(nrequests_max);
    this->remote_indices.reset_to_new(nrequests_max);
    MPI_Comm_size(this->comm, &this->comm_size);
    int size = this->comm_size;
    this->send_counts.reset_to_new(size + 1);
    this->send_offsets.reset_to_new(size + 2);
    this->recv_counts.reset_to_new(size);
    this->recv_offsets.reset_to_new(size + 1);
#ifndef NDEBUG
    int datasize;
    MPI_Type_size(out_dt, &datasize);
    assert (datasize == (int)sizeof(OutElt));
#endif
  }

  ~gather() {
    assert (!this->epoch_active);
    this->local_indices.reset();
    this->remote_ranks.reset();
    this->remote_indices.reset();
    MPI_Comm_free(&this->comm);
    this->send_counts.reset();
    this->send_offsets.reset();
    this->recv_counts.reset();
    this->recv_offsets.reset();
    this->input = NULL;
    this->output = NULL;
    this->input_count = 0;
    this->output_count = 0;
  }

  void begin_epoch() {
    assert (!this->epoch_active);
    ptrdiff_t nr = ptrdiff_t(this->nrequests_max);
    int size = this->comm_size;
    int* restrict remote_ranks = this->remote_ranks.get(0, nr);
// #pragma omp parallel for
    for (ptrdiff_t i = 0; i < nr; ++i) remote_ranks[i] = size;
    this->epoch_active = true;
  }

  void add_request(size_t local_idx, int remote_rank, size_t remote_idx, size_t req_id) {
    assert (this->epoch_active);
    assert (remote_rank >= 0 && remote_rank < this->comm_size);
    assert (req_id < this->nrequests_max);
    assert (local_idx < this->output_count);
    this->local_indices[req_id] = local_idx;
    assert (this->remote_ranks[req_id] == this->comm_size);
    this->remote_ranks[req_id] = remote_rank;
    this->remote_indices[req_id] = (MPI_Aint)remote_idx;
  }

  void end_epoch() {
    assert (this->epoch_active);
    int size = this->comm_size;
    int* restrict send_counts = this->send_counts.get(0, size + 1);
    int* restrict send_offsets = this->send_offsets.get(0, size + 2);
    int* restrict recv_counts = this->recv_counts.get(0, size);
    int* restrict recv_offsets = this->recv_offsets.get(0, size + 1);
    size_t* restrict local_indices = this->local_indices.get(0, this->nrequests_max);
    int* restrict remote_ranks = this->remote_ranks.get(0, this->nrequests_max);
    MPI_Aint* restrict remote_indices = this->remote_indices.get(0, this->nrequests_max);
    const InElt* restrict input = this->input;
    OutElt* restrict output = this->output;
    MPI_Comm comm = this->comm;
    MPI_Datatype out_datatype = this->out_datatype;
    assert (nrequests_max <= PTRDIFF_MAX);
    ptrdiff_t nrequests_max = ptrdiff_t(this->nrequests_max);
#ifndef NDEBUG
    size_t input_count = this->input_count;
#endif
    memset(send_counts, 0, (size + 1) * sizeof(int));
// #pragma omp parallel for
    for (ptrdiff_t i = 0; i < nrequests_max; ++i) {
      assert (remote_ranks[i] >= 0 && remote_ranks[i] < size + 1);
// #pragma omp atomic
      ++send_counts[remote_ranks[i]];
    }
    send_offsets[0] = 0;
    for (int i = 0; i < size + 1; ++i) {
      assert (send_counts[i] >= 0);
      send_offsets[i + 1] = send_offsets[i] + send_counts[i];
    }
    assert (send_offsets[size + 1] == (int)nrequests_max);
    histogram_sort_inplace2<size_t, MPI_Aint>(remote_ranks, send_offsets, size + 1, local_indices, remote_indices);

    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, comm);

    int max_to_one_dest = 0;
    for (int i = 0; i < size; ++i) max_to_one_dest = (std::max)(max_to_one_dest, send_counts[i]);
    MPI_Allreduce(MPI_IN_PLACE, &max_to_one_dest, 1, MPI_INT, MPI_MAX, comm);

    size_t group_limit = (std::min)(size_t(max_to_one_dest), 2 * size_t((CHUNKSIZE + size - 1) / size));
    int group_count = (group_limit == 0 ? 0 : int((max_to_one_dest + group_limit - 1) / group_limit));

    scoped_array<OutElt> recv_reply_data;
    recv_reply_data.reset_to_new(send_offsets[size]);
    scoped_array<int> send_counts_for_mpi;
    send_counts_for_mpi.reset_to_new(size);
    scoped_array<int> recv_counts_for_mpi;
    recv_counts_for_mpi.reset_to_new(size);
    scoped_array<int> send_offsets_for_mpi;
    send_offsets_for_mpi.reset_to_new(size);
    scoped_array<MPI_Aint> recv_data;
    recv_data.reset_to_new(group_limit * size);
    scoped_array<OutElt> reply_data;
    reply_data.reset_to_new(group_limit * size);

    for (int group = 0; group < group_count; ++group) {
      recv_offsets[0] = 0;
      for (int i = 0; i < size; ++i) {
        send_counts_for_mpi[i] = int((std::min)(group_limit, (std::max)(group_limit * group, size_t(send_counts[i])) - group_limit * group));
        recv_counts_for_mpi[i] = int((std::min)(group_limit, (std::max)(group_limit * group, size_t(recv_counts[i])) - group_limit * group));
        assert (recv_counts[i] >= 0);
        send_offsets_for_mpi[i] = send_offsets[i] + int((std::min)(group_limit * group, size_t(send_counts[i])));
        recv_offsets[i + 1] = recv_offsets[i] + recv_counts_for_mpi[i];
      }
      int recvtotal = recv_offsets[size];
      assert (recvtotal >= 0 && recvtotal <= int(group_limit * size));
      MPI_Alltoallv(remote_indices, send_counts_for_mpi.get(0, size), send_offsets_for_mpi.get(0, size), MPI_AINT, recv_data.get(0, recvtotal), recv_counts_for_mpi.get(0, size), recv_offsets, MPI_AINT, comm);
#pragma omp parallel for
      for (int i = 0; i < recvtotal; ++i) {
        assert (recv_data[i] >= 0 && recv_data[i] < MPI_Aint(input_count));
        reply_data[i] = this->f(input[recv_data[i]]);
      }
      MPI_Alltoallv(reply_data.get(0, recvtotal), recv_counts_for_mpi.get(0, size), recv_offsets, out_datatype, recv_reply_data.get(0, send_offsets[size]), send_counts_for_mpi.get(0, size), send_offsets_for_mpi.get(0, size), out_datatype, comm);
#pragma omp parallel for
      for (int i = 0; i < send_offsets[size]; ++i) {
        output[local_indices[i]] = recv_reply_data[i];
      }
    }
    this->epoch_active = false;
  }
};

struct scatter_bitmap_set {
  bitmap& array;
  size_t nrequests_max;
  int epoch_active;
  MPI_Comm comm;
  scoped_array<int> remote_ranks;
  scoped_array<MPI_Aint> remote_indices;
  int comm_size;
  scoped_array<int> send_counts;
  scoped_array<int> send_offsets;
  scoped_array<int> recv_counts;
  scoped_array<int> recv_offsets;

  scatter_bitmap_set(bitmap& array, size_t nrequests_max, MPI_Comm comm) 
    : array(array), nrequests_max(nrequests_max), epoch_active(false)
  {
    MPI_Comm_dup(comm, &this->comm);
    this->remote_ranks.reset_to_new(nrequests_max);
    this->remote_indices.reset_to_new(nrequests_max);
    MPI_Comm_size(this->comm, &this->comm_size);
    int size = this->comm_size;
    this->send_counts.reset_to_new(size + 1);
    this->send_offsets.reset_to_new(size + 2);
    this->recv_counts.reset_to_new(size);
    this->recv_offsets.reset_to_new(size + 1);
  }

  ~scatter_bitmap_set() {
    assert (!this->epoch_active);
    MPI_Comm_free(&this->comm);
  }

  void begin_epoch() {
    assert (!this->epoch_active);
    ptrdiff_t nr = ptrdiff_t(this->nrequests_max);
    int size = this->comm_size;
    int* restrict remote_ranks = this->remote_ranks.get(0, nr);
// #pragma omp parallel for
    for (ptrdiff_t i = 0; i < nr; ++i) remote_ranks[i] = size;
    this->epoch_active = true;
  }

  void add_request(int remote_rank, size_t remote_idx, size_t req_id) {
    assert (this->epoch_active);
    assert (remote_rank >= 0 && remote_rank < this->comm_size);
    assert (req_id < this->nrequests_max);
    assert (this->remote_ranks[req_id] == this->comm_size);
    this->remote_ranks[req_id] = remote_rank;
    this->remote_indices[req_id] = (MPI_Aint)remote_idx;
  }

  void end_epoch() {
    assert (this->epoch_active);
    int size = this->comm_size;
    int* restrict send_counts = this->send_counts.get(0, size + 1);
    int* restrict send_offsets = this->send_offsets.get(0, size + 2);
    int* restrict recv_counts = this->recv_counts.get(0, size);
    int* restrict recv_offsets = this->recv_offsets.get(0, size + 1);
    int* restrict remote_ranks = this->remote_ranks.get(0, this->nrequests_max);
    MPI_Aint* restrict remote_indices = this->remote_indices.get(0, this->nrequests_max);
    bitmap& array = this->array;
    MPI_Comm comm = this->comm;
    ptrdiff_t nrequests_max = this->nrequests_max;
    memset(send_counts, 0, (size + 1) * sizeof(int));
// #pragma omp parallel for
    for (ptrdiff_t i = 0; i < nrequests_max; ++i) {
      assert (remote_ranks[i] >= 0 && remote_ranks[i] < size + 1);
// #pragma omp atomic
      ++send_counts[remote_ranks[i]];
    }
    send_offsets[0] = 0;
    for (int i = 0; i < size + 1; ++i) {
      send_offsets[i + 1] = send_offsets[i] + send_counts[i];
    }
    histogram_sort_inplace1<MPI_Aint>(remote_ranks, send_offsets, size + 1, remote_indices);
    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, comm);

    int max_to_one_dest = 0;
    for (int i = 0; i < size; ++i) max_to_one_dest = (std::max)(max_to_one_dest, send_counts[i]);
    MPI_Allreduce(MPI_IN_PLACE, &max_to_one_dest, 1, MPI_INT, MPI_MAX, comm);

    size_t group_limit = (std::min)(size_t(max_to_one_dest), 2 * size_t((CHUNKSIZE + size - 1) / size));
    int group_count = (group_limit == 0 ? 0 : int((max_to_one_dest + group_limit - 1) / group_limit));

    scoped_array<int> send_counts_for_mpi;
    send_counts_for_mpi.reset_to_new(size);
    scoped_array<int> recv_counts_for_mpi;
    recv_counts_for_mpi.reset_to_new(size);
    scoped_array<int> send_offsets_for_mpi;
    send_offsets_for_mpi.reset_to_new(size);
    scoped_array<MPI_Aint> recv_data;
    recv_data.reset_to_new(group_limit * size);

    for (int group = 0; group < group_count; ++group) {
      recv_offsets[0] = 0;
      for (int i = 0; i < size; ++i) {
        send_counts_for_mpi[i] = int((std::min)(group_limit, (std::max)(group_limit * group, size_t(send_counts[i])) - group_limit * group));
        recv_counts_for_mpi[i] = int((std::min)(group_limit, (std::max)(group_limit * group, size_t(recv_counts[i])) - group_limit * group));
        assert (recv_counts[i] >= 0);
        send_offsets_for_mpi[i] = send_offsets[i] + int((std::min)(group_limit * group, size_t(send_counts[i])));
        recv_offsets[i + 1] = recv_offsets[i] + recv_counts_for_mpi[i];
      }
      int recvtotal = recv_offsets[size];
      assert (recvtotal >= 0 && recvtotal <= int(group_limit * size));
      MPI_Alltoallv(remote_indices, send_counts_for_mpi.get(0, size), send_offsets_for_mpi.get(0, size), MPI_AINT, recv_data.get(0, recvtotal), recv_counts_for_mpi.get(0, size), recv_offsets, MPI_AINT, comm);
      for (int i = 0; i < recvtotal; ++i) {
        array.set(recv_data[i]);
      }
    }
    this->epoch_active = false;
  }
};

#if 0
template <typename Elt>
struct scatter {
  Elt* restrict output;
  size_t output_count;
  size_t nrequests_max;
  int epoch_active;
  MPI_Comm comm;
  MPI_Datatype elt_datatype;
  scoped_array<int> remote_ranks;
  scoped_array<MPI_Aint> remote_indices;
  scoped_array<Elt> values_to_send;
  int comm_size;
  scoped_array<int> send_counts;
  scoped_array<int> send_offsets;
  scoped_array<int> recv_counts;
  scoped_array<int> recv_offsets;

  scatter(Elt* output, size_t output_count, MPI_Comm user_comm, MPI_Datatype elt_datatype, size_t nrequests_max) 
    : output(output), output_count(output_count), nrequests_max(nrequests_max), epoch_active(false), elt_datatype(elt_datatype)
  {
    MPI_Comm_dup(user_comm, &this->comm);
    this->remote_ranks.reset_to_new(nrequests_max);
    this->remote_indices.reset_to_new(nrequests_max);
    this->values_to_send.reset_to_new(nrequests_max);
    MPI_Comm_size(this->comm, &this->comm_size);
    int size = this->comm_size;
    this->send_counts.reset_to_new(size + 1);
    this->send_offsets.reset_to_new(size + 2);
    this->recv_counts.reset_to_new(size);
    this->recv_offsets.reset_to_new(size + 1);
  }

  ~scatter() {
    assert (!this->epoch_active);
    MPI_Comm_free(&this->comm);
  }

  void begin_epoch() {
    assert (!this->epoch_active);
    // ptrdiff_t nr = ptrdiff_t(this->nrequests_max);
    // int size = this->comm_size;
    // int* restrict remote_ranks = this->remote_ranks.get(0, nr);
// #pragma omp parallel for
    // memset(remote_ranks, 0xFF, nr * sizeof(remote_ranks[0])); -- see note in end_epoch
    this->epoch_active = true;
  }

  void add_request(int remote_rank, size_t remote_idx, const Elt& val, size_t req_id) {
    assert (this->epoch_active);
    assert (remote_rank >= 0 && remote_rank < this->comm_size);
    assert (req_id < this->nrequests_max);
    assert (this->remote_ranks[req_id] == -1);
    this->remote_ranks[req_id] = remote_rank;
    this->remote_indices[req_id] = (MPI_Aint)remote_idx;
    this->values_to_send[req_id] = val;
  }

  void end_epoch(size_t nrequests) {
    // Assumes there are no unused slots before nrequests
    assert (this->epoch_active);
    int size = this->comm_size;
    int* restrict send_counts = this->send_counts.get(0, size + 1);
    int* restrict send_offsets = this->send_offsets.get(0, size + 2);
    int* restrict recv_counts = this->recv_counts.get(0, size);
    int* restrict recv_offsets = this->recv_offsets.get(0, size + 1);
    int* restrict remote_ranks = this->remote_ranks.get(0, this->nrequests_max);
    MPI_Aint* restrict remote_indices = this->remote_indices.get(0, this->nrequests_max);
    Elt* restrict values_to_send = this->values_to_send.get(0, this->nrequests_max);
    Elt* restrict output = this->output;
#ifndef NDEBUG
    size_t output_count = this->output_count;
#endif
    MPI_Comm comm = this->comm;
    MPI_Datatype elt_datatype = this->elt_datatype;
    assert (nrequests <= this->nrequests_max);
    memset(send_counts, 0, (size + 1) * sizeof(int));
// #pragma omp parallel for
    for (ptrdiff_t i = 0; i < ptrdiff_t(nrequests); ++i) {
      int rr = remote_ranks[i];
      assert (rr >= 0 && rr < size + 1);
      if (rr >= size) continue;
// #pragma omp atomic
      ++send_counts[rr];
    }
    send_offsets[0] = 0;
    for (int i = 0; i < size; ++i) {
      send_offsets[i + 1] = send_offsets[i] + send_counts[i];
    }
    send_offsets[size + 1] = int(nrequests);
    histogram_sort_inplace2<MPI_Aint, Elt>(remote_ranks, send_offsets, size + 1, remote_indices, values_to_send);
    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, comm);
    recv_offsets[0] = 0;
    for (int i = 0; i < size; ++i) {
      recv_offsets[i + 1] = recv_offsets[i] + recv_counts[i];
    }
    scoped_array<MPI_Aint> recv_indices;
    recv_indices.reset_to_new(recv_offsets[size]);
    scoped_array<Elt> recv_values;
    recv_values.reset_to_new(recv_offsets[size]);
    MPI_Alltoallv(remote_indices, send_counts, send_offsets, MPI_AINT, recv_indices.get(0, recv_offsets[size]), recv_counts, recv_offsets, MPI_AINT, comm);
    MPI_Alltoallv(values_to_send, send_counts, send_offsets, elt_datatype, recv_values.get(0, recv_offsets[size]), recv_counts, recv_offsets, elt_datatype, comm);
    for (int i = 0; i < recv_offsets[size]; ++i) {
      assert (recv_indices[i] >= 0 && recv_indices[i] < (int)output_count);
      output[recv_indices[i]] = recv_values[i];
    }
    recv_indices.reset();
    recv_values.reset();
    this->epoch_active = false;
  }
};
#endif

#if 0
template <typename Elt>
struct scatter {
  typedef std::pair<MPI_Aint, Elt> write_pair;
  static const unsigned int nelements_max = 256;
  static const unsigned int senddepth = 1;
  static const unsigned int recvdepth = 8;

  struct coalescing_buffer {
    write_pair data[nelements_max];
    unsigned int count;
    bool busy;
    MPI_Request send_req;

    coalescing_buffer(): count(0), busy(false), send_req(MPI_REQUEST_NULL) {}
    bool append(size_t idx, const Elt& e) {
      assert (count < nelements_max);
      assert (!busy);
      data[count++] = std::pair<size_t, Elt>(idx, e);
      return (count == nelements_max);
    }
    unsigned int send(unsigned int dest, MPI_Datatype write_pair_datatype, MPI_Comm comm) {
      assert (!busy);
      if (count == 0) return 0;
      busy = true;
#pragma omp critical (mpi)
      MPI_Isend(data, count, write_pair_datatype, int(dest), 0, comm, &send_req);
      return 1;
    }
    void test() {
      if (!busy) return;
      int flag;
#pragma omp critical (mpi)
      MPI_Test(&send_req, &flag, MPI_STATUS_IGNORE);
      if (flag) {
        busy = false;
        count = 0;
      }
    }
  };
  bool uninitialized;
  scoped_array<coalescing_buffer> buf;
  scoped_array<size_t> cur_send_buf;
  Elt* output;
  size_t output_count;
  MPI_Comm comm;
  MPI_Datatype elt_datatype;
  MPI_Datatype pair_datatype;
  unsigned int comm_size;
  unsigned int comm_rank;
  bool in_epoch;
  scoped_array<uint64_t> total_sent;
  uint64_t total_received;
  scoped_array<write_pair> recvbuf;
  checked_array<recvdepth, MPI_Request> recvreq;
  NBC_Request allreduce_req;
  bool allreduce_active;
  scoped_array<uint64_t> sends_to_rank;

  scatter(): uninitialized(true) {}

  scatter(Elt* output, size_t output_count, MPI_Comm user_comm,
          MPI_Datatype elt_datatype) 
    : uninitialized(true)
  {
    this->init(output, output_count, user_comm, elt_datatype);
  }

  void init(Elt* output, size_t output_count, MPI_Comm user_comm,
            MPI_Datatype elt_datatype) 
  {
#pragma omp critical (mpi)
    {
      assert (this->uninitialized);
      this->uninitialized = false;
      this->output = output;
      this->output_count = output_count;
      this->elt_datatype = elt_datatype;
      this->in_epoch = false;
      this->total_received = 0;
      MPI_Comm_dup(user_comm, &comm);
      int size;
      MPI_Comm_size(comm, &size);
      this->comm_size = (unsigned int)size;
      MPI_Comm_rank(this->comm, (int*)(&this->comm_rank));
      buf.reset_to_new(size_t(size) * senddepth);
      cur_send_buf.reset_to_new(size_t(size));
      total_sent.reset_to_new(size_t(size));
      recvbuf.reset_to_new(nelements_max * recvdepth);
      MPI_Aint displs[2];
      MPI_Datatype types[2] = {MPI_AINT, elt_datatype};
      int blocklengths[2] = {1, 1};
      {
        write_pair p;
        MPI_Aint base;
        MPI_Get_address(&p, &base);
        MPI_Get_address(&p.first, &displs[0]);
        MPI_Get_address(&p.second, &displs[1]);
        displs[0] -= base;
        displs[1] -= base;
        MPI_Type_create_struct(2, blocklengths, displs, types, &pair_datatype);
        MPI_Type_commit(&pair_datatype);
      }
      this->allreduce_active = false;
      sends_to_rank.reset_to_new(this->comm_size);
    }
  }

  ~scatter() {
    if (!uninitialized) {
      assert (!in_epoch);
#pragma omp critical (mpi)
      {
        MPI_Comm_free(&comm);
        MPI_Type_free(&pair_datatype);
      }
      buf.reset();
      total_sent.reset();
      recvbuf.reset();
      output = NULL;
      output_count = 0;
      sends_to_rank.reset();
    }
    uninitialized = true;
  }

  void begin_epoch() {
    assert (!this->in_epoch);
#pragma omp critical (mpi)
    {
      NBC_Request req;
      NBC_Ibarrier(this->comm, &req);
      NBC_Wait(&req);
      for (unsigned int i = 0; i < this->comm_size; ++i) {
        this->cur_send_buf[i] = i * senddepth;
      }
      memset(this->total_sent.get(0, this->comm_size), 0, this->comm_size * sizeof(uint64_t));
      this->total_received = 0;
      for (unsigned int i = 0; i < recvdepth; ++i) {
        MPI_Irecv(this->recvbuf.get(nelements_max * i,
                                    nelements_max),
                  nelements_max, pair_datatype,
                  MPI_ANY_SOURCE, 0, this->comm, &this->recvreq[i]);
      }
      this->allreduce_active = false;
      this->in_epoch = true;
    }
  }

  void add_request(int remote_rank, size_t remote_idx, const Elt& val) {
    assert (this->in_epoch);
    size_t orig_cur_send_buf = this->cur_send_buf[remote_rank]; // To tell if all buffers are full
    while (true) {
      size_t buf_num = this->cur_send_buf[remote_rank];
      coalescing_buffer& this_buf = this->buf[buf_num];
      if (this_buf.busy) {
        ++buf_num;
        if (buf_num == (remote_rank + 1) * senddepth) {
          buf_num = remote_rank * senddepth;
        }
        this->cur_send_buf[remote_rank] = buf_num;
        if (buf_num == orig_cur_send_buf) this->progress();
      } else {
        bool full = this_buf.append(remote_idx, val);
        if (full) {
          this->total_sent[remote_rank] += this_buf.send(remote_rank, pair_datatype, this->comm);
          this->progress();
        }
        return;
      }
    }
  }

  void end_epoch() {
    this->iend_epoch();
    while (!this->test_end_epoch()) this->progress();
  }

  void iend_epoch() {
    assert (in_epoch);
    for (unsigned int i = 0; i < this->comm_size * senddepth; ++i) {
      coalescing_buffer& this_buf = this->buf[i];
      if (!this_buf.busy) {
        total_sent[i / senddepth] += this_buf.send(i / senddepth, pair_datatype, comm);
      }
    }
    // This should really be MPI_Reduce_scatter_block
#pragma omp critical (mpi)
    NBC_Iallreduce(this->total_sent.get(0, this->comm_size), sends_to_rank.get(0, this->comm_size), this->comm_size, MPI_UINT64_T, MPI_SUM, comm, &allreduce_req);
    allreduce_active = true;
  }

  bool test_end_epoch() {
    if (!this->in_epoch) return true;
    if (this->allreduce_active) return false;
    int rank = this->comm_rank;
    if (this->total_received < this->sends_to_rank[rank]) return false;
    for (unsigned int i = 0; i < this->comm_size * senddepth; ++i) {
      coalescing_buffer& this_buf = this->buf[i];
      if (this_buf.busy) return false;
    }
#pragma omp critical (mpi)
    {
      for (unsigned int i = 0; i < recvdepth; ++i) {
        MPI_Cancel(&this->recvreq[i]);
        MPI_Wait(&this->recvreq[i], MPI_STATUS_IGNORE);
      }
    }
    this->in_epoch = false;
    return true;
  }

  void progress() {
    if (!in_epoch) return;
#pragma omp critical (mpi)
    {
      bool repeat = true;
      while (repeat) {
        repeat = false;
        int flag, index;
        MPI_Status st;
        MPI_Testany(recvdepth, this->recvreq.get(0, recvdepth), &index, &flag, &st);
        if (flag && index >= 0 && index < int(recvdepth)) {
          ++this->total_received;
          unsigned int recvbuf_offset = nelements_max * (unsigned int)(index);
          int count;
          MPI_Get_count(&st, this->pair_datatype, &count);
          for (unsigned int i = recvbuf_offset; i < recvbuf_offset + (unsigned int)(count); ++i) {
            assert (this->recvbuf[i].first >= 0 && this->recvbuf[i].first < MPI_Aint(this->output_count));
            this->output[this->recvbuf[i].first] = this->recvbuf[i].second;
          }
          MPI_Irecv(this->recvbuf.get(recvbuf_offset, nelements_max),
                    nelements_max, this->pair_datatype,
                    MPI_ANY_SOURCE, 0, this->comm, &this->recvreq[index]);
          repeat = true;
        }
      }
    }
    for (unsigned int i = 0; i < this->comm_size * senddepth; ++i) {
      this->buf[i].test();
    }
    if (this->allreduce_active) {
      int test_result;
#pragma omp critical (mpi)
      test_result = NBC_Test(&this->allreduce_req);
      if (test_result == NBC_OK) {
#pragma omp critical (mpi)
        NBC_Wait(&this->allreduce_req);
        this->allreduce_active = false;
      }
    }
  }
};
#endif

#endif /* ONESIDED_HPP */

