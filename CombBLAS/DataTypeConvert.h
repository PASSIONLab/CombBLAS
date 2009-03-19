#ifndef _DATATYPE_TO_MPI_H
#define _DATATYPE_TO_MPI_H

#include <iostream>
#include <mpi.h>

using namespace std;



// These special cases are for MPI predefined datatypes for C
template <class T> 
const MPI_Datatype DataTypeToMPI ( void )
{
	cerr << "Ops, that shouldn't happen, what type of data are you trying to send?" << endl;
	return MPI_BYTE;
}; 

template<> const MPI_Datatype DataTypeToMPI< signed char >( void )
{
	return MPI_CHAR;
}; 

template<> const MPI_Datatype DataTypeToMPI< signed short int >( void )
{
	return MPI_SHORT;
}; 

template<> const MPI_Datatype DataTypeToMPI< signed int >( void )
{
	return MPI_INT;
};  

template<> const MPI_Datatype DataTypeToMPI< signed long int >( void )
{
	return MPI_LONG;
}; 

template<> const MPI_Datatype DataTypeToMPI< unsigned char >( void )
{
	return MPI_UNSIGNED_CHAR;
}; 

template<> const MPI_Datatype DataTypeToMPI< unsigned short int >( void )
{
	return MPI_UNSIGNED_SHORT;
}; 
template<> const MPI_Datatype DataTypeToMPI< unsigned int >( void )
{
	return MPI_UNSIGNED;
};

template<> const MPI_Datatype DataTypeToMPI< unsigned long int >( void )
{
	return MPI_UNSIGNED_LONG;
}; 
template<> const MPI_Datatype DataTypeToMPI< float >( void )
{
	return MPI_FLOAT;
}; 

template<> const MPI_Datatype DataTypeToMPI< double >( void )
{
	return MPI_DOUBLE;
}; 

template<> const MPI_Datatype DataTypeToMPI< long double >( void )
{
	return MPI_LONG_DOUBLE;
}; 
#endif
