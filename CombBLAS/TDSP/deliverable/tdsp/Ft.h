#ifndef FT_H 
#define FT_H

#include <iostream>
#include <math.h>
#include <set>
#include <limits>
#include "../../trunk/CombBLAS/CombBLAS.h"

#define PERIOD (3600 * 24)

using namespace std;

struct Ft {
  //Customize this.
  double cost;

  Ft() {
  }

  Ft(int n) {
  }

  Ft& operator=(const Ft& _ft) {
    if (this != &_ft) 
      cost = _ft.cost;
    return *this; 
  }

  double arrivalTime(double t) const {
    // return t + (2 - 4 * (pow(((t - PERIOD * (int) (t / PERIOD)) / PERIOD - 0.5), 2))) * cost;
    return 2 * t;
  }
};

// hack not needed
bool operator< (const Ft &ft1, const Ft &ft2) {
  return true;
}

template <typename c, typename t>
inline std::basic_istream<c,t>& operator<<(std::basic_istream<c,t>& lhs, const Ft& rhs) { 
  lhs << rhs.cost;
  return lhs;
}

template <typename c, typename t>
inline std::basic_istream<c,t>& operator>>(std::basic_istream<c,t>& lhs, Ft& rhs) { 
  // Customize this.
  lhs >> rhs.cost;
  return lhs;
}

MPI::Datatype Ft_MPI_datatype;
template<> MPI::Datatype MPIType< Ft >( void ) {
  return Ft_MPI_datatype;
}

#endif
