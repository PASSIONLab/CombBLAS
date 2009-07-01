#include <iostream>
#include <mpi.h>
#include "MPIType.h"

using namespace std;



template<> MPI_Datatype MPIType< signed char >( void )
{
	return MPI_CHAR;
}; 

template<> MPI_Datatype MPIType< signed short int >( void )
{
	return MPI_SHORT;
}; 

template<> MPI_Datatype MPIType< signed int >( void )
{
	return MPI_INT;
};  

template<> MPI_Datatype MPIType< signed long int >( void )
{
	return MPI_LONG;
}; 

template<> MPI_Datatype MPIType< unsigned char >( void )
{
	return MPI_UNSIGNED_CHAR;
}; 

template<> MPI_Datatype MPIType< unsigned short int >( void )
{
	return MPI_UNSIGNED_SHORT;
}; 
template<> MPI_Datatype MPIType< unsigned int >( void )
{
	return MPI_UNSIGNED;
};

template<> MPI_Datatype MPIType< unsigned long int >( void )
{
	return MPI_UNSIGNED_LONG;
}; 
template<> MPI_Datatype MPIType< float >( void )
{
	return MPI_FLOAT;
}; 

template<> MPI_Datatype MPIType< double >( void )
{
	return MPI_DOUBLE;
}; 

template<> MPI_Datatype MPIType< long double >( void )
{
	return MPI_LONG_DOUBLE;
}; 

