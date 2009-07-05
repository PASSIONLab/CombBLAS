#ifndef _MPI_TYPE_H
#define _MPI_TYPE_H

#include <iostream>
#include <mpi.h>

using namespace std;


// These special cases are for MPI predefined datatypes for C
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
template<> MPI::Datatype MPIType< float >( void );
template<> MPI::Datatype MPIType< double >( void );
template<> MPI::Datatype MPIType< long double >( void );
template<> MPI::Datatype MPIType< bool >( void );

#endif
