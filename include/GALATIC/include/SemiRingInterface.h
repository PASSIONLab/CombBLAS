//
// Created by Richard Lettich on 4/13/21.
//

#ifndef ACSPGEMM_FUNCTION_INTERFACE_H
#define ACSPGEMM_FUNCTION_INTERFACE_H

template <typename T, typename U, typename V>
struct SemiRing {
    typedef T leftInput_t;
    typedef U rightInput_t;
    typedef V output_t;

    V multiply(const T& a, const U& b);
    V add(const V& a, const V& b);

    V AdditiveIdentity();
};



#endif //ACSPGEMM_FUNCTION_INTERFACE_H











