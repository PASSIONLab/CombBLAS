#include "FullyDistVecRot.h"

template <class IT, class NT>
FullyDistVecRot<IT, NT>::FullyDistVecRot( shared_ptr<CommGrid> grid, IT globallen, NT initval)
: FullyDistVec<IT,NT>(grid, globallen, initval)
{
  rotation_offset = 0;
  int size_overestimate = glen / commGrid->GetSize() + 1;
  arr.reserve(size_overestimate);
  receive_buff.reserve(size_overestimate);
}

template <class IT, class NT>
IT FullyDistVecRot<IT,NT>::GetGlobalStartOfLocal() const
{
  return RotLengthUntil();
}

template <class IT, class NT>
IT FullyDistVecRot<IT,NT>::GetGlobalEndOfLocal() const
{
  return RotLengthUntil() + arr.size();
}


template <class IT, class NT>
IT FullyDistVecRot<IT,NT>::RotLengthUntil() const
{
  int my_procrow = commGrid->GetRankInProcCol();
  int proccols = commGrid->GetGridCols();
  int my_proccol = (commGrid->GetRankInProcRow() + rotation_offset) % proccols;
  return RotLengthUntil(my_procrow, my_proccol);
}

template <class IT, class NT>
IT FullyDistVecRot<IT,NT>::RotLengthUntil(int my_procrow, int my_proccol) const
{
	int procrows = commGrid->GetGridRows();
	IT n_perprocrow = glen / procrows;
	IT n_thisrow;
	if(my_procrow == procrows-1)
		n_thisrow = glen - (n_perprocrow*(procrows-1));
	else
		n_thisrow = n_perprocrow;	
	int proccols = commGrid->GetGridCols();
	IT n_perproc = n_thisrow / proccols;
	return ((n_perprocrow * my_procrow)+(n_perproc*my_proccol));
}

template <class IT, class NT>
IT FullyDistVecRot<IT,NT>::SizeOfChunk() const
{
  int procrows = commGrid->GetGridRows();
  int proccols = commGrid->GetGridCols();
  int my_procrow = commGrid->GetRankInProcCol();
  int my_proccol = (commGrid->GetRankInProcRow() + rotation_offset) % proccols;
  int my_upper_limit;
  if (my_proccol == (proccols-1)) {
    if (my_procrow == (procrows-1)) {
      my_upper_limit = glen;
    }else {
      my_upper_limit = RotLengthUntil(my_procrow+1, 0);
    }
  } else {
    my_upper_limit = RotLengthUntil(my_procrow, (my_proccol+1) % proccols);
  }
  return my_upper_limit - RotLengthUntil(my_procrow,my_proccol);
}

template <class IT, class NT>
NT FullyDistVecRot<IT,NT>:: GetLocalElement(IT index) const
{
  return arr[index - RotLengthUntil()];
}

template <class IT, class NT>
void FullyDistVecRot<IT,NT>::SetLocalElement(IT index, NT value) {
  arr[index - RotLengthUntil()] = value;
}

template <class IT, class NT>
void FullyDistVecRot<IT,NT>::RotateAlongRow()
{
	int proccols = commGrid->GetGridCols();
	int my_proccol = commGrid->GetRankInProcRow();
  int source = (my_proccol+1) % proccols;
  int dest = (my_proccol+(proccols-1)) % proccols;
  rotation_offset = (rotation_offset + 1) % commGrid->GetGridCols();
  int size_to_send = arr.size(), new_size;
  MPI::Intracomm RowWorld = commGrid->GetRowWorld();
	RowWorld.Sendrecv(&size_to_send, 1, MPI::INT, dest, ROTATE,
                    &new_size, 1, MPI::INT, source, ROTATE);
  receive_buff.resize(new_size);
  RowWorld.Sendrecv(arr.data(), arr.size(), MPI::LONG_LONG, dest, ROTATE,
                    receive_buff.data(), new_size, MPI::LONG_LONG, source, ROTATE);
  swap(arr, receive_buff);
}
