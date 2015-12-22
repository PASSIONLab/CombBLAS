/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.4 -------------------------------------------------*/
/* date: 1/17/2014 ---------------------------------------------*/
/* authors: Aydin Buluc (abuluc@lbl.gov), Adam Lugowski --------*/
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

/**
 * Operations used in parallel reductions and scans
 **/

#ifndef _OPERATIONS_H_
#define _OPERATIONS_H_

#include <iostream>
#include <functional>
#include <cmath>
#include <limits>
#include "psort-1.0/driver/MersenneTwister.h"

using namespace std;

template<typename T1, typename T2>
struct equal_first
{
    bool operator()(pair<T1,T2> & lhs, pair<T1,T2> & rhs){
        return lhs.first == rhs.first;
    }
};



template<typename T>
struct myset: public std::unary_function<T, T>
{
  myset(T myvalue): value(myvalue) {};
  /** @returns value regardless of x */
  const T& operator()(const T& x) const
  {
    return value;
  } 
  T value;
};


template<typename T>
struct identity : public std::unary_function<T, T>
{
  /** @returns x itself */
  const T operator()(const T& x) const
  {
	return x;
  }
};


// Because identify reports ambiguity in PGI compilers
template<typename T>
struct myidentity : public std::unary_function<T, T>
{
  /** @returns x itself */
  const T operator()(const T& x) const
  {
	return x;
  }
};


template<typename T>
struct totality : public std::unary_function<T, bool>
{
  /** @returns true regardless */
  bool operator()(const T& x) const
  {
	return true;
  }
};
	
	
template<typename T>
struct safemultinv : public std::unary_function<T, T>
{
  const T operator()(const T& x) const
  {
	T inf = std::numeric_limits<T>::max();
    	return (x == 0) ? inf:(1/x);
  }
};


template<typename T>
struct sel2nd: public std::binary_function<T, T, T>
{
    const T& operator()(const T& x, const T & y) const
    {
        return y;
    }
};

template<typename T1, typename T2>
struct bintotality : public std::binary_function<T1, T2, bool>
{
  /** @returns true regardless */
  bool operator()(const T1& x, const T2 & y) const
  {
        return true;
  }
};



/**
 * binary_function<Arg1, Arg2, Result>
 * This is left untemplated because pow() only makes sense for 
 * <double, int, double> , <double, double, double> , <float, float, float>
 * and C++ can automatically upcast each case to <double, double, double>
 */
struct exponentiate : public std::binary_function<double, double, double> 
{
    double operator()(double x, double y) const { return std::pow(x, y); }
};


/**
 *  @brief Compute the maximum of two values.
 *
 *  This binary function object computes the maximum of the two values
 *  it is given. When used with MPI and a type @c T that has an
 *  associated, built-in MPI data type, translates to @c MPI_MAX.
 */
template<typename T>
struct maximum : public std::binary_function<T, T, T>
{
  /** @returns the maximum of x and y. */
  const T operator()(const T& x, const T& y) const
  {
    return x < y? y : x;
  }
};


/**
 *  @brief Compute the minimum of two values.
 *
 *  This binary function object computes the minimum of the two values
 *  it is given. When used with MPI and a type @c T that has an
 *  associated, built-in MPI data type, translates to @c MPI_MIN.
 */
template<typename T>
struct minimum : public std::binary_function<T, T, T>
{
  /** @returns the minimum of x and y. */
  const T operator()(const T& x, const T& y) const
  {
    return x < y? x : y;
  }
};

/**
 *  @brief With 50/50 chances, return a one of the operants
 */
template<typename T>
struct RandReduce : public std::binary_function<T, T, T>
{
    /** @returns the minimum of x and y. */
    const T operator()(const T& x, const T& y) 
    {
        return (M.rand() < 0.5)? x : y;
    }
    RandReduce()
    {
    #ifdef DETERMINISTIC
        M = MTRand(1);
    #else
        M = MTRand();	// generate random numbers with Mersenne Twister
    #endif
    }
    MTRand M;
};

/**
 *  @brief Returns a special value (passed to the constructor of the functor) when both operants disagree
 */
template<typename T>
struct SetIfNotEqual : public std::binary_function<T, T, T>
{
    const T operator()(const T& x, const T& y)
    {
        if(x != y)
        {
            return valuetoset;
        }
        else
        {
            return x;
        }
    }
    SetIfNotEqual(T value):valuetoset(value) { };
    T valuetoset;
};


/**
 *  @brief Compute the bitwise AND of two integral values.
 *
 *  This binary function object computes the bitwise AND of the two
 *  values it is given. When used with MPI and a type @c T that has an
 *  associated, built-in MPI data type, translates to @c MPI_BAND.
 */
template<typename T>
struct bitwise_and : public std::binary_function<T, T, T>
{
  /** @returns @c x & y. */
  T operator()(const T& x, const T& y) const
  {
    return x & y;
  }
};


/**
 *  @brief Compute the bitwise OR of two integral values.
 *
 *  This binary function object computes the bitwise OR of the two
 *  values it is given. When used with MPI and a type @c T that has an
 *  associated, built-in MPI data type, translates to @c MPI_BOR.
 */
template<typename T>
struct bitwise_or : public std::binary_function<T, T, T>
{
  /** @returns the @c x | y. */
  T operator()(const T& x, const T& y) const
  {
    return x | y;
  }
};

/**
 *  @brief Compute the logical exclusive OR of two integral values.
 *
 *  This binary function object computes the logical exclusive of the
 *  two values it is given. When used with MPI and a type @c T that has
 *  an associated, built-in MPI data type, translates to @c MPI_LXOR.
 */
template<typename T>
struct logical_xor : public std::binary_function<T, T, T>
{
  /** @returns the logical exclusive OR of x and y. */
  T operator()(const T& x, const T& y) const
  {
    return (x || y) && !(x && y);
  }
};

/**
 *  @brief Compute the bitwise exclusive OR of two integral values.
 *
 *  This binary function object computes the bitwise exclusive OR of
 *  the two values it is given. When used with MPI and a type @c T that
 *  has an associated, built-in MPI data type, translates to @c
 *  MPI_BXOR.
 */
template<typename T>
struct bitwise_xor : public std::binary_function<T, T, T>
{
  /** @returns @c x ^ y. */
  T operator()(const T& x, const T& y) const
  {
    return x ^ y;
  }
};




#endif


