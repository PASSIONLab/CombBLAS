#ifndef _MPI_D_T_
#define _MPI_D_T_

MPI_Datatype gettype(long int fint)
{
	switch (fint) {
		case 2:
			return MPI_INT;
			break;
		case 3:
			return MPI_FLOAT;
			break;
		case 4:
			return MPI_DOUBLE;
			break;
		case 8:
			return MPI_BYTE;
			break;
		case 9:
			return MPI_BYTE;
			break;
		default:
			fprintf(stderr, "Unknown type\n");
			return NULL;
			break;
	}
}

MPI_Op getop(long int fint)
{
	switch(fint) {
		case 26:
			return MPI_SUM;
			break;
		case 27:
			return MPI_MAX;
			break;
		case 28:
			return MPI_MIN;
			break;
		default:
			fprintf(stderr, "Unknown op\n");
			return NULL;
			break;
	}
}

#endif

