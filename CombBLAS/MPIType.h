#ifndef _MPI_TYPE_H
#define _MPI_TYPE_H

#include <iostream>
#include <mpi.h>
#include <stdint.h>

using namespace std;

/**
  * C++ type to MPIType conversion is done through functions returning the mpi types
  * The templated function is explicitly instantiated for every C++ type 
  * that has a correspoinding MPI type. For all others, MPI::BYTE is returned
  **/


template <typename T> 
MPI::Datatype MPIType ( void )
{
	cerr << "Ops, that shouldn't happen, what type of data are you trying to send?" << endl;
	return MPI::BYTE;
};

template<> MPI::Datatype MPIType< signed char >( void );
template<> MPI::Datatype MPIType< signed short int >( void );
template<> MPI::Datatype MPIType< signed int >( void );
template<> MPI::Datatype MPIType< signed long int >( void );
template<> MPI::Datatype MPIType< unsigned char >( void );
template<> MPI::Datatype MPIType< unsigned short int >( void );
template<> MPI::Datatype MPIType< unsigned int >( void );
template<> MPI::Datatype MPIType< unsigned long int >( void );
template<> MPI::Datatype MPIType< int64_t >( void );
template<> MPI::Datatype MPIType< uint64_t >( void );
template<> MPI::Datatype MPIType< float >( void );
template<> MPI::Datatype MPIType< double >( void );
template<> MPI::Datatype MPIType< long double >( void );
template<> MPI::Datatype MPIType< bool >( void );



#endif
