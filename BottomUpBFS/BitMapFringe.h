#ifndef BITMAPFRINGE_H
#define BITMAPFRINGE_H

// #include <algorithm>
#include "BitMap.h"
#include "CommGrid.h"

template <class IT, class VT>
class BitMapFringe {
 public:
  BitMapFringe(shared_ptr<CommGrid> grid, FullyDistSpVec<IT,VT> & x) {
    cg.reset(new CommGrid(*grid));   
    
    MPI::Intracomm World = cg->GetWorld();
    MPI::Intracomm ColWorld = cg->GetColWorld();

    // Find out how big local chunk will be after transpose
    long num_local_send = x.MyLocLength(), num_local_recv;
    diagneigh = cg->GetComplementRank();
    World.Sendrecv(&num_local_send, 1, MPI::LONG, diagneigh, TROST,
                   &num_local_recv, 1, MPI::LONG, diagneigh, TROST);

    // Calculate new local displacements
  	colneighs = ColWorld.Get_size();
  	colrank = ColWorld.Get_rank();
  	int counts[colneighs];
  	counts[colrank] = num_local_recv;
    ColWorld.Allgather(MPI::IN_PLACE, 1, MPI::INT, counts, 1, MPI::INT);
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
    World.Sendrecv(&trans_subword_disp, 1, MPIType<int32_t>(), diagneigh, TROST, &local_subword_disp, 1, MPIType<int32_t>(), diagneigh, TROST);
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


  BitMap* TransposeGather() {
    MPI::Intracomm World = cg->GetWorld();
    MPI::Intracomm ColWorld = cg->GetColWorld();

    // Transpose bitmaps
    World.Sendrecv(local_bm->data(), trans_words_send, MPIType<uint64_t>(), diagneigh, TROST, trans_bm->data(), trans_words_recv, MPIType<uint64_t>(), diagneigh, TROST);

    // Gather all but first words
    double t1 = MPI_Wtime();
    ColWorld.Allgatherv(trans_bm->data()+1, send_counts[colrank], MPIType<uint64_t>(), gather_bm->data(), send_counts, word_dpls, MPIType<uint64_t>());
    double t2 = MPI_Wtime();
    bottomup_allgather += (t2-t1);

    // Gather first words & add in
    gather_bm->data()[0] = 0;
    uint64_t firsts[colneighs];
    firsts[colrank] = trans_bm->data()[0];
    ColWorld.Allgather(MPI::IN_PLACE, 1, MPIType<uint64_t>(), firsts, 1, MPIType<uint64_t>());
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
      updates[up_index++] = bm_index - local_subword_disp;	// ABAB: local_subword_disp is subtracted
	  
    bm_index = local_bm->get_next_bit(bm_index);
    while(bm_index != -1) {
      updates[up_index++] = bm_index - local_subword_disp;	// ABAB: local_subword_disp is subtracted
      bm_index = local_bm->get_next_bit(bm_index);
    }
    x.BulkSet(updates, local_num_set);
    delete[] updates;
  }


  IT GetNumSet() const {
		IT global_num_set = 0;
		(cg->GetWorld()).Allreduce(&local_num_set, &global_num_set, 1, MPIType<IT>(), MPI::SUM);
		return global_num_set;
	}


	int GetSubWordDisp() {
    return local_subword_disp;
	}


  BitMap* AccessBM() {
    return local_bm;
  }
 private:
 	shared_ptr<CommGrid> cg;
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

#endif // BITMAPFRINGE_H
