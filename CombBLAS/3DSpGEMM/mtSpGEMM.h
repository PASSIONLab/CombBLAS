#ifndef _mtSpGEMM_h
#define _mtSpGEMM_h

template <typename SR, typename NTO, typename IT, typename NT1, typename NT2>
SpTuples<IT, NTO> * LocalSpGEMM
(const SpDCCols<IT, NT1> & A,
 const SpDCCols<IT, NT2> & B,
 bool clearA, bool clearB);

#include "mtSpGEMM.cpp"
#endif
