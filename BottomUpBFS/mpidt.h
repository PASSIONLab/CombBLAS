#ifndef _MPI_D_T_
#define _MPI_D_T_

struct a {
  long int mpi_bottom__, mpi_integer__, mpi_real__, mpi_double_precision__,
           mpi_complex__, mpi_double_complex__, mpi_logical__,
           mpi_character__, mpi_byte__, mpi_2integer__, mpi_2real__,
           mpi_2double_precision__, mpi_2complex__, mpi_2double_complex__,
           mpi_integer1__, mpi_integer2__, mpi_integer4__, mpi_real2__,
           mpi_real4__, mpi_real8__, mpi_ub__, mpi_lb__, mpi_comm_world__,
           mpi_comm_self__, mpi_group_empty__, mpi_sum__, mpi_max__,
           mpi_min__, mpi_prod__, mpi_land__, mpi_band__, mpi_lor__,
           mpi_bor__, mpi_lxor__, mpi_bxor__, mpi_minloc__, mpi_maxloc__,
           mpi_op_null__, mpi_tag_ub__, mpi_host__, mpi_io__,
           mpi_errors_are_fatal__, mpi_errors_return__, mpi_packed__;
} mpipriv_ = {
  1, 2, 3, 4,
  5, 6, 7,
  8, 9, 10, 11,
  12, 13, 14,
  15, 16, 17, 18,
  19, 20, 21, 22, 23,
  24, 25, 26, 27,
  28, 29, 30, 31, 32,
  33, 34, 35, 36, 37,
  38, 39, 40, 41,
  42, 43, 44
};


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

