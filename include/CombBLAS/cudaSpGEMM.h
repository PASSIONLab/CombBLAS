#ifndef _cudaSpGEMM_h
#define _cudaSpGEMM_h

#include <tuple>
    template < typename NTO, typename IT, typename NT1, typename NT2>
void transformColumn(IT A_nzc, size_t i, size_t nnzcolB, IT curptr, IT* A_Tran_CP,
    IT* A_Tran_IR,
    IT* A_Tran_JC,
    NT1* A_Tran_numx,
    IT* B_CP,
    IT* B_IR,
    IT* B_JC,
    NT2* B_numx,
    std::tuple<IT,IT,NTO> * tuplesC);

#endif