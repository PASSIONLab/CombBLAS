#ifndef _MPI_TYPE_H
#define _MPI_TYPE_H

#include <iostream>
#include <mpi.h>

using namespace std;


// These special cases are for MPI predefined datatypes for C
template <typename T> 
MPI_Datatype MPIType ( void )
{
	cerr << "Ops, that shouldn't happen, what type of data are you trying to send?" << endl;
	return MPI_BYTE;
};

template<> MPI_Datatype MPIType< signed char >( void );
template<> MPI_Datatype MPIType< signed short int >( void );
template<> MPI_Datatype MPIType< signed int >( void );
template<> MPI_Datatype MPIType< signed long int >( void );
template<> MPI_Datatype MPIType< unsigned char >( void );
template<> MPI_Datatype MPIType< unsigned short int >( void );
template<> MPI_Datatype MPIType< unsigned int >( void );
template<> MPI_Datatype MPIType< unsigned long int >( void );
template<> MPI_Datatype MPIType< float >( void );
template<> MPI_Datatype MPIType< double >( void );
template<> MPI_Datatype MPIType< long double >( void );

#endif
