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


#ifndef BITMAPCAROUSEL_H
#define BITMAPCAROUSEL_H

// #include <algorithm>
#include "BitMap.h"
#include "BitMapFringe.h"
#include "CommGrid.h"

namespace combblas {

template <class IT, class NT>
class BitMapCarousel {
 public:
  BitMapCarousel(std::shared_ptr<CommGrid> grid, IT glen, int local_subword_disp) {
    commGrid.reset(new CommGrid(*grid));   
    rotation_index = 0;
    global_size = glen;
    my_procrow = commGrid->GetRankInProcCol();
    my_proccol = commGrid->GetRankInProcRow();
    procrows = commGrid->GetGridRows();
    proccols = commGrid->GetGridCols();
	local_size = SizeOfChunk();
    rotlenuntil = RotLengthUntil();
    IT biggest_size = global_size - RotLengthUntil(procrows-1, proccols-1) + 63;
    bm = new BitMap(biggest_size);
    recv_buff = new BitMap(biggest_size);
    old_bm = new BitMap(biggest_size);
    sub_disps = new int[proccols];
    sub_disps[my_proccol] = local_subword_disp;	  
	MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, sub_disps, 1, MPI_INT, commGrid->GetRowWorld());
    curr_subword_disp = local_subword_disp;
  }
  
  ~BitMapCarousel() {
    delete bm;
    delete recv_buff;
    delete old_bm;
    delete[] sub_disps;
  }

  // Return the global index of the start of the local chunk (inclusive)
  IT GetGlobalStartOfLocal() const {
    return rotlenuntil;
  }

  IT GetGlobalStartOfLocal(int col) const {
    return RotLengthUntil(my_procrow,col);
  }

  // Return the global index of the end of the local chunk (exclusive)
  IT GetGlobalEndOfLocal() const {
    return rotlenuntil + local_size;
  }

  IT GetGlobalEndOfLocal(int col) const {
    return GetGlobalStartOfLocal(col) + local_size;
  }

  bool GetBit(IT index) const {
    return bm->get_bit(index - rotlenuntil + curr_subword_disp);
  }

  void SetBit(IT index) {
    bm->set_bit(index - rotlenuntil + curr_subword_disp);
  }

  template <class NT1>
  void LoadVec(FullyDistVec<IT,NT1> & x) {
    bm->reset();
    local_size = x.LocArrSize();
    for (DenseVectorLocalIterator<IT,NT> it(x); it.HasNext(); it.Next())
      if (it.GetValue() != -1)
        bm->set_bit(it.GetLocIndex() + curr_subword_disp);
  }

  IT RotLengthUntil() {
    int curr_col = (my_proccol + rotation_index) % proccols;
    return RotLengthUntil(my_procrow, curr_col);
  }

  IT RotLengthUntil(int my_procrow, int my_proccol) const {
  	IT n_perprocrow = global_size / procrows;
  	IT n_thisrow;
  	if(my_procrow == procrows-1)
  		n_thisrow = global_size - (n_perprocrow*(procrows-1));
  	else
  		n_thisrow = n_perprocrow;	
  	IT n_perproc = n_thisrow / proccols;
  	return ((n_perprocrow * my_procrow)+(n_perproc*my_proccol));
  }

  IT SizeOfChunk() const {
    int curr_col = (my_proccol + rotation_index) % proccols;
    return SizeOfChunk(my_procrow, curr_col);
  }

  IT SizeOfChunk(int my_procrow, int my_proccol) const {
    IT my_upper_limit;
    if (my_proccol == (proccols-1)) {
      if (my_procrow == (procrows-1)) {
        my_upper_limit = global_size;
      }else {
        my_upper_limit = RotLengthUntil(my_procrow+1, 0);
      }
    } else {
      my_upper_limit = RotLengthUntil(my_procrow, (my_proccol+1) % proccols);
    }
    return my_upper_limit - RotLengthUntil(my_procrow,my_proccol);
  }

  void RotateAlongRow() {
    int source = (my_proccol+1) % proccols;
    int dest = (my_proccol+(proccols-1)) % proccols;
    rotation_index = (rotation_index + 1) % proccols;
    long send_words = (local_size + 63 + curr_subword_disp)>>6;
    long recv_words = (SizeOfChunk() + 63 + sub_disps[(my_proccol+rotation_index)%proccols])>>6;

    MPI_Comm RowWorld = commGrid->GetRowWorld();
	MPI_Status status;
	MPI_Sendrecv(bm->data(), send_words, MPIType<uint64_t>(), dest, ROTATE,
				   recv_buff->data(), recv_words, MPIType<uint64_t>(), source, ROTATE, RowWorld, &status);

    local_size = SizeOfChunk();
    rotlenuntil = RotLengthUntil();
    std::swap(bm, recv_buff);
    curr_subword_disp = sub_disps[(my_proccol+rotation_index)%proccols];
  }

  void UpdateFringe(BitMapFringe<IT,NT> &bm_fringe) {
    uint64_t* dest = bm_fringe.AccessBM()->data();
    uint64_t* curr = bm->data();
    uint64_t* old = old_bm->data();
    IT num_words = (local_size + 63 + curr_subword_disp) / 64;
    for (IT i=0; i<num_words; i++) {
      dest[i] = curr[i] ^ old[i];
    }
  }

  void SaveOld() {
    old_bm->copy_from(bm);
  }

private:
  std::shared_ptr<CommGrid> commGrid;
  int rotation_index;
  int my_procrow;
  int my_proccol;
  int procrows;
  int proccols;
  int curr_subword_disp;
  IT rotlenuntil;
  IT global_size;
  IT local_size;
  BitMap* bm;
  BitMap* recv_buff;
  BitMap* old_bm;
  int* sub_disps;
};

}

#endif // BITMAPCAROUSEL_H
