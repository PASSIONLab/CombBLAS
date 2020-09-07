/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.6 -------------------------------------------------*/
/* date: 6/15/2017 ---------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc  --------------------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2017, The Regents of the University of California
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */


#include <iostream>
#include <mpi.h>
#include "CombBLAS/MPIType.h"

using namespace std;

namespace combblas {

MPIDataTypeCache mpidtc;	// global variable

template<> MPI_Datatype MPIType< signed char >( void )
{
	return MPI_CHAR;
}; 
template<> MPI_Datatype MPIType< unsigned char >( void )
{
	return MPI_UNSIGNED_CHAR;
}; 
template<> MPI_Datatype MPIType< signed short int >( void )
{
	return MPI_SHORT;
}; 
template<> MPI_Datatype MPIType< unsigned short int >( void )
{
	return MPI_UNSIGNED_SHORT;
}; 
template<> MPI_Datatype MPIType< int32_t >( void )
{
	return MPI_INT;
};  
template<> MPI_Datatype MPIType< uint32_t >( void )
{
	return MPI_UNSIGNED;
};
template<> MPI_Datatype MPIType<int64_t>(void)
{
	return MPI_LONG_LONG;
};
template<> MPI_Datatype MPIType< uint64_t>(void)
{
	return MPI_UNSIGNED_LONG_LONG;
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
template<> MPI_Datatype MPIType< bool >( void )
{
	return MPI_BYTE;  // usually  #define MPI_BOOL MPI_BYTE anyway
};

}
