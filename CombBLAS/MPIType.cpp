#include <iostream>
#include <mpi.h>
#include "MPIType.h"

using namespace std;



template<> const MPI_Datatype MPIType< signed char >( void )
{
	return MPI_CHAR;
}; 

template<> const MPI_Datatype MPIType< signed short int >( void )
{
	return MPI_SHORT;
}; 

template<> const MPI_Datatype MPIType< signed int >( void )
{
	return MPI_INT;
};  

template<> const MPI_Datatype MPIType< signed long int >( void )
{
	return MPI_LONG;
}; 

template<> const MPI_Datatype MPIType< unsigned char >( void )
{
	return MPI_UNSIGNED_CHAR;
}; 

template<> const MPI_Datatype MPIType< unsigned short int >( void )
{
	return MPI_UNSIGNED_SHORT;
}; 
template<> const MPI_Datatype MPIType< unsigned int >( void )
{
	return MPI_UNSIGNED;
};

template<> const MPI_Datatype MPIType< unsigned long int >( void )
{
	return MPI_UNSIGNED_LONG;
}; 
template<> const MPI_Datatype MPIType< float >( void )
{
	return MPI_FLOAT;
}; 

template<> const MPI_Datatype MPIType< double >( void )
{
	return MPI_DOUBLE;
}; 

template<> const MPI_Datatype MPIType< long double >( void )
{
	return MPI_LONG_DOUBLE;
}; 

