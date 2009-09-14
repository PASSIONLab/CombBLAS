#ifndef _MPI_TYPE_H
#define _MPI_TYPE_H

#include <iostream>
#include <mpi.h>

using namespace std;

/**
  * C++ type to MPIType conversion is done through functions returning the mpi types
  * The templated function is explicitly instantiated for every C++ type 
  * that has a correspoinding MPI type. For all others, MPI::BYTE is returned
  **/


namespace typetrait {
 	template< bool x > struct bool_
   	{
       		static bool const value = x;        
       		typedef bool_<x> type;              
       		typedef bool value_type;            
       		operator bool() const { return x; } 
   	};	

	typedef bool_<true>  true_;
	typedef bool_<false> false_;
}

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


/**
 *  @brief Type trait that determines if there exists a built-in
 *  integer MPI data type for a given C++ type.
 *
 *  This type trait determines when there is a direct mapping from a
 *  C++ type to an MPI data type that is classified as an integer data
 *  type. See @c is_mpi_builtin_datatype for general information about
 *  built-in MPI data types.
 */
template<typename T>
struct is_mpi_integer_datatype  : public typetrait::false_ { };

template<> struct is_mpi_integer_datatype< signed short int > : typetrait::true_ {}
template<> struct is_mpi_integer_datatype< signed int > : typetrait::true_ {}
template<> struct is_mpi_integer_datatype< signed long int > : typetrait::true_ {}
template<> struct is_mpi_integer_datatype< unsigned short int > : typetrait::true_ {}
template<> struct is_mpi_integer_datatype< unsigned int > : typetrait::true_ {}
template<> struct is_mpi_integer_datatype< unsigned long int > : typetrait::true_ {}

template<typename T>
struct is_mpi_floating_point_datatype	: public typetrait::false_ {};

template<> struct is_mpi_floating_point_datatype< float > : typetrait::true_ {}
template<> struct is_mpi_floating_point_datatype< double > : typetrait::true_ {}
template<> struct is_mpi_floating_point_datatype< long double > : typetrait::true_ {}




#endif
