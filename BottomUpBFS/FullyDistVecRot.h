#ifndef _FULLY_DIST_VEC_ROT_H_
#define _FULLY_DIST_VEC_ROT_H_

#include "FullyDistVec.h"

template <class IT, class NT>
class FullyDistVecRot: public FullyDistVec<IT,NT>
{
 public:
  FullyDistVecRot(shared_ptr<CommGrid> grid, IT globallen, NT initval);

  // Return the global index of the start of the local chunk (inclusive)
  IT GetGlobalStartOfLocal() const;
  // Return the global index of the edb of the local chunk (exclusive)
  IT GetGlobalEndOfLocal() const;

  IT RotLengthUntil() const;

  IT RotLengthUntil(int my_procrow, int my_proccol) const;

  NT GetLocalElement(IT index) const;

  void SetLocalElement(IT index, NT value);

  IT SizeOfChunk() const;

  void RotateAlongRow();

 private:
  int rotation_offset;
  vector<NT> receive_buff;

  using FullyDistVec<IT,NT>::arr;
  using FullyDistVec<IT,NT>::commGrid;
  using FullyDistVec<IT,NT>::glen;
};

#include "FullyDistVecRot.cpp"

#endif //_FULLY_DIST_VEC_ROT_H_