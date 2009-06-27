#ifndef _MPI_TYPE_H
#define _MPI_TYPE_H

#include <iostream>
#include <mpi.h>

using namespace std;


// These special cases are for MPI predefined datatypes for C
template <class T> 
const MPI_Datatype MPIType ( void )
{
	cerr << "Ops, that shouldn't happen, what type of data are you trying to send?" << endl;
	return MPI_BYTE;
}; 


#endif
