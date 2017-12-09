/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.4 -------------------------------------------------*/
/* date: 1/17/2014 ---------------------------------------------*/
/* authors: Aydin Buluc (abuluc@lbl.gov), Adam Lugowski --------*/
/* this file contributed by Scott Beamer of UC Berkeley --------*/
/****************************************************************/
/*
 Copyright (c) 2010-2014, The Regents of the University of California
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */


#ifndef BITMAPFRINGE_H
#define BITMAPFRINGE_H

// #include <algorithm>
#include "BitMap.h"
#include "CommGrid.h"

namespace combblas {

template <class IT, class VT>
class BitMapFringe {
 public:
  BitMapFringe(std::shared_ptr<CommGrid> grid, FullyDistSpVec<IT,VT> & x) {
    cg.reset(new CommGrid(*grid));   
    
	MPI_Comm World = x.getcommgrid()->GetWorld();
	MPI_Comm ColWorld = x.getcommgrid()->GetColWorld();
	MPI_Status status;

    // Find out how big local chunk will be after transpose
    long num_local_send = x.MyLocLength(), num_local_recv;
    diagneigh = cg->GetComplementRank();
	MPI_Sendrecv(&num_local_send, 1, MPI_LONG, diagneigh, TROST,
				 &num_local_recv, 1, MPI_LONG, diagneigh, TROST, World, &status); 

    // Calculate new local displacements
	MPI_Comm_size(ColWorld, &colneighs);
	MPI_Comm_rank(ColWorld, &colrank);
	  
  	int counts[colneighs];
  	counts[colrank] = num_local_recv;
	MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, counts, 1, MPI_INT, ColWorld);	

	int dpls[colneighs];
    dpls[0] = 0;
  	std::partial_sum(counts, counts+colneighs-1, dpls+1);
    long total_size = dpls[colneighs-1] + counts[colneighs-1];
    local_offset = x.LengthUntil();
    local_num_set = 0;

    // Compute word/byte displacements and send counts for gather
    word_dpls = new int[colneighs];
    byte_dpls = new int[colneighs];
    send_counts = new int[colneighs];
    for (int c=0; c<colneighs; c++) {
      word_dpls[c] = (dpls[c] + (64-(dpls[c]%64)))>>6;
      byte_dpls[c] = word_dpls[c]<<3;
      send_counts[c] = (counts[c] - (64-(dpls[c] % 64)) + 63)>>6;
    }

    // Compute subword displacements and transpose exchange details
    trans_subword_disp = dpls[colrank] % 64;
	MPI_Sendrecv(&trans_subword_disp, 1, MPIType<int32_t>(), diagneigh, TROST, 
				 &local_subword_disp, 1, MPIType<int32_t>(), diagneigh, TROST, World, &status); 
	  
    trans_words_send = (num_local_send + local_subword_disp + 63)>>6;
    trans_words_recv = (num_local_recv + trans_subword_disp + 63)>>6;

    // Allocate bitmaps
    local_bm = new BitMap(num_local_send + local_subword_disp);
    next_bm = new BitMap(num_local_send + local_subword_disp);
    trans_bm = new BitMap(num_local_recv + trans_subword_disp);
    gather_bm = new BitMap(total_size);
  }

  ~BitMapFringe() {
    delete local_bm;
    delete next_bm;
    delete trans_bm;
    delete gather_bm;
    delete[] send_counts;
    delete[] word_dpls;
    delete[] byte_dpls;
  }

  void LoadFromSpVec(FullyDistSpVec<IT,VT> & x) {
    local_bm->reset();
    for (SparseVectorLocalIterator<IT,VT> spit(x); spit.HasNext(); spit.Next())
      local_bm->set_bit(spit.GetLocIndex() + local_subword_disp);
    local_num_set = x.getlocnnz();
  }

  void LoadFromUpdates(IT* updates, long total_updates) {
    local_bm->reset();
    for (long i=0; i<total_updates; i++)
      local_bm->set_bit(updates[i] - local_offset + local_subword_disp);
    local_num_set = total_updates;
  }

  void LoadFromNext() {
    local_num_set = next_num_set;
  }

  void SetNext(IT local_index) {
    next_num_set++;
  }


  void IncrementNumSet(int num_updates) {
    next_num_set += num_updates;
  }


  BitMap* TransposeGather() 
  {
	MPI_Comm World = cg->GetWorld();
	MPI_Comm ColWorld = cg->GetColWorld();
	MPI_Status status;

    // Transpose bitmaps
	MPI_Sendrecv(local_bm->data(), trans_words_send, MPIType<uint64_t>(), diagneigh, TROST, 
				 trans_bm->data(), trans_words_recv, MPIType<uint64_t>(), diagneigh, TROST, World, &status); 
	  
    // Gather all but first words
#ifdef BOTTOMUPTIME
    double t1 = MPI_Wtime();
#endif
	MPI_Allgatherv(trans_bm->data()+1, send_counts[colrank], MPIType<uint64_t>(), gather_bm->data(), send_counts, word_dpls, MPIType<uint64_t>(), ColWorld);	
#ifdef BOTTOMUPTIME
    double t2 = MPI_Wtime();
    bottomup_allgather += (t2-t1);
#endif

    // Gather first words & add in
    gather_bm->data()[0] = 0;
    uint64_t firsts[colneighs];
    firsts[colrank] = trans_bm->data()[0];
    MPI_Allgather(MPI_IN_PLACE, 1, MPIType<uint64_t>(), firsts, 1, MPIType<uint64_t>(), ColWorld);
    for (int c=0; c<colneighs; c++)
      gather_bm->data()[word_dpls[c]-1] |= firsts[c];

    next_bm->reset();
    next_num_set = 0;
    return gather_bm;
  }


  void UpdateSpVec(FullyDistSpVec<IT,VT> & x) {
    IT *updates = new IT[local_num_set];
    IT bm_index=local_subword_disp, up_index=0;
	  
    if (local_bm->get_bit(bm_index))	// if the first valid bit is 1
      updates[up_index++] = bm_index - local_subword_disp;	// ABAB: local_subword_disp is NOT subtracted (as local_subword_disp is equal to bm_index)
	  
    bm_index = local_bm->get_next_bit(bm_index);
    while(bm_index != -1) {
      updates[up_index++] = bm_index - local_subword_disp;	// ABAB: local_subword_disp is subtracted
      bm_index = local_bm->get_next_bit(bm_index);
    }
    x.BulkSet(updates, local_num_set);
    delete[] updates;
  }


  IT GetNumSet() {
	  IT global_num_set = 0;
	  MPI_Allreduce(&local_num_set, &global_num_set, 1, MPIType<IT>(), MPI_SUM, cg->GetWorld());
	  return global_num_set;
	}


	int GetSubWordDisp() {
    return local_subword_disp;
	}


  BitMap* AccessBM() {
    return local_bm;
  }
 private:
 	std::shared_ptr<CommGrid> cg;
  BitMap* local_bm;
  BitMap* next_bm;
  BitMap* trans_bm;
  BitMap* gather_bm;
  int* send_counts;
  int* byte_dpls;
  int* word_dpls;
  int colneighs;
  int colrank;
  int diagneigh;
  IT local_offset;
  IT local_num_set;
  IT next_num_set;
  int local_subword_disp;
  int trans_subword_disp;
  long trans_words_send;
  long trans_words_recv;
};

}

#endif // BITMAPFRINGE_H
