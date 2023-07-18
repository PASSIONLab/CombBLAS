#ifndef _cudaSpGEMM_h
#define _cudaSpGEMM_h

//#include "../GALATIC/include/CSR.h"
//#include "../GALATIC/include/CSR.cuh"
#include <tuple>
    template < typename NTO, typename IT, typename NT1, typename NT2>
void transformColumn(IT A_nzc, IT* A_Tran_CP,
    IT* A_Tran_IR,
    IT* A_Tran_JC,
    NT1* A_Tran_numx,
    IT* B_CP,
    IT* B_IR,
    IT* B_JC,
    NT2* B_numx,
     std::tuple<IT,IT,NTO> * tuplesC_d, IT* curptrC, IT B_nzc);

//template <typename Arith_SR, typename NTO, typename NT1, typename NT2, typename IT>
//CSR<NTO> LocalGalaticSPGEMM
//(CSR<NT1> input_A_CPU,
//CSR<NT2> input_B_CPU,
// bool clearA, bool clearB, Arith_SR semiring, IT * aux = nullptr);
#endif
