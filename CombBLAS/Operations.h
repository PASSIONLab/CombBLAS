/****************************************************************/
/* Sequential and Parallel Sparse Matrix Multiplication Library */
/* version 2.3 --------------------------------------------------/
/* date: 01/18/2009 ---------------------------------------------/
/* author: Aydin Buluc (aydin@cs.ucsb.edu) ----------------------/
/****************************************************************/

/**
 * Operations used in parallel reductions and scans
 **/

#ifndef _OPERATIONS_H_
#define _OPERATIONS_H_

#include <iostream>
#include <functional>
#include "mpi.h"
using namespace std;

template<typename Op, typename T> struct is_mpi_op;

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
  const T& operator()(const T& x, const T& y) const
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
  const T& operator()(const T& x, const T& y) const
  {
    return x < y? x : y;
  }
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



/**************************************************************************
 * MPI_Op queries                                                         *
 **************************************************************************/

/**
 *  @brief Determine if a function object has an associated @c MPI_Op.
 *
 *  This trait determines if a function object type @c Op, when used
 *  with argument type @c T, has an associated @c MPI_Op. If so, @c
 *  is_mpi_op<Op,T> will derive from @c mpl::false_ and will
 *  contain a static member function @c op that takes no arguments but
 *  returns the associated @c MPI_Op value. For instance, @c
 *  is_mpi_op<std::plus<int>,int>::op() returns @c MPI_SUM.
 *
 *  Users may specialize @c is_mpi_op for any other class templates
 *  that map onto operations that have @c MPI_Op equivalences, such as
 *  bitwise OR, logical and, or maximum. However, users are encouraged
 *  to use the standard function objects in the @c functional and @c
 *  boost/mpi/operations.hpp headers whenever possible. For
 *  function objects that are class templates with a single template
 *  parameter, it may be easier to specialize @c is_builtin_mpi_op.
 */

// Everything that is derived from typetrait::false_ falls to this case which does not have an op() function
template<typename Op, typename T>
struct is_mpi_op : public typetrait::false_ { };

/// Everything from here is a valid instantiations as long as it is derived from typetrait::true_
template<typename T>
struct is_mpi_op<maximum<T>, T>
  : public boost::mpl::or_<is_mpi_integer_datatype<T>,
                           is_mpi_floating_point_datatype<T> >
{
  static MPI_Op op() { return MPI_MAX; }
};

template<typename T>
struct is_mpi_op<minimum<T>, T>
  : public boost::mpl::or_<is_mpi_integer_datatype<T>,
                           is_mpi_floating_point_datatype<T> >
{
  static MPI_Op op() { return MPI_MIN; }
};

template<typename T>
 struct is_mpi_op<std::plus<T>, T>
  : public boost::mpl::or_<is_mpi_integer_datatype<T>,
                           is_mpi_floating_point_datatype<T>,
                           is_mpi_complex_datatype<T> >
{
  static MPI_Op op() { return MPI_SUM; }
};


template<typename T>
 struct is_mpi_op<std::multiplies<T>, T>
  : public boost::mpl::or_<is_mpi_integer_datatype<T>,
                           is_mpi_floating_point_datatype<T>,
                           is_mpi_complex_datatype<T> >
{
  static MPI_Op op() { return MPI_PROD; }
};


template<typename T>
 struct is_mpi_op<std::logical_and<T>, T>
  : public boost::mpl::or_<is_mpi_integer_datatype<T>,
                           is_mpi_logical_datatype<T> >
{
  static MPI_Op op() { return MPI_LAND; }
};

template<typename T>
 struct is_mpi_op<std::logical_or<T>, T>
  : public boost::mpl::or_<is_mpi_integer_datatype<T>,
                           is_mpi_logical_datatype<T> >
{
  static MPI_Op op() { return MPI_LOR; }
};


template<typename T>
 struct is_mpi_op<logical_xor<T>, T>
  : public boost::mpl::or_<is_mpi_integer_datatype<T>,
                           is_mpi_logical_datatype<T> >
{
  static MPI_Op op() { return MPI_LXOR; }
};


template<typename T>
 struct is_mpi_op<bitwise_and<T>, T>
  : public boost::mpl::or_<is_mpi_integer_datatype<T>,
                           is_mpi_byte_datatype<T> >
{
  static MPI_Op op() { return MPI_BAND; }
};


template<typename T>
 struct is_mpi_op<bitwise_or<T>, T>
  : public boost::mpl::or_<is_mpi_integer_datatype<T>,
                           is_mpi_byte_datatype<T> >
{
  static MPI_Op op() { return MPI_BOR; }
};


template<typename T>
 struct is_mpi_op<bitwise_xor<T>, T>
  : public boost::mpl::or_<is_mpi_integer_datatype<T>,
                           is_mpi_byte_datatype<T> >
{
  static MPI_Op op() { return MPI_BXOR; }
};



#endif


