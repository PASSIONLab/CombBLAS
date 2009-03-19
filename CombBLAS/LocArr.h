#ifndef _LOC_ARR_H_
#define _LOC_ARR_H_

#include <mpi.h>
#include "DataTypeToMPI.h"

template<class ST, class ET>
struct LocArr
{
	LocArr(ET * myaddr, ST mycount): addr((void*) myaddr ), 
					count(mycount),
					mpitype(DataTypeToMPI<NT>),
					eltsize(sizeof(NT)) {};
	
	void * addr;
	ST count;
	size_t eltsize;
	MPI_Datatype mpitype;
}

#endif

