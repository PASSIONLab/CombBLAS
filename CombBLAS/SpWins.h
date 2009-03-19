#ifndef _SP_WINS_H
#define _SP_WINS_H

#include <mpi.h>

struct SpWins
{
	MPI_Win maswin;
	MPI_Win jcwin;
	MPI_Win irwin;
	MPI_Win numwin;
};
#endif

