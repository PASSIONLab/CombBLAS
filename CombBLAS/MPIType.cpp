#include <iostream>
#include <mpi.h>
#include "MPIType.h"

using namespace std;



template<> MPI::Datatype MPIType< signed char >( void )
{
	return MPI::CHAR;
}; 

template<> MPI::Datatype MPIType< signed short int >( void )
{
	return MPI::SHORT;
}; 
template<> MPI::Datatype MPIType< signed int >( void )
{
	return MPI::INT;
};  
template<> MPI::Datatype MPIType< signed long int >( void )
{
	return MPI::LONG;
};  
template<> MPI::Datatype MPIType< unsigned char >( void )
{
	return MPI::UNSIGNED_CHAR;
}; 

template<> MPI::Datatype MPIType< unsigned short int >( void )
{
	return MPI::UNSIGNED_SHORT;
}; 
template<> MPI::Datatype MPIType< unsigned int >( void )
{
	return MPI::UNSIGNED;
};
template<> MPI::Datatype MPIType< unsigned long int >( void )
{
	return MPI::UNSIGNED_LONG;
};
template<> MPI::Datatype MPIType<long long>(void)
{
	return MPI::LONG_LONG;
};
template<> MPI::Datatype MPIType< unsigned long long>(void)
{
	return MPI::UNSIGNED_LONG_LONG;
};
template<> MPI::Datatype MPIType< float >( void )
{
	return MPI::FLOAT;
}; 
template<> MPI::Datatype MPIType< double >( void )
{
	return MPI::DOUBLE;
}; 
template<> MPI::Datatype MPIType< long double >( void )
{
	return MPI::LONG_DOUBLE;
}; 
template<> MPI::Datatype MPIType< bool >( void )
{
	return MPI::BOOL;
};

