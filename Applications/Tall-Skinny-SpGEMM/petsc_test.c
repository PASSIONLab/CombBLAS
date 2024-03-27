#include <petscmat.h>
#include "omp.h"

static char help[] = "help yourself!";

int
main (int argc, char **argv)
{
	Mat A, B, C;
	PetscViewer fd;
	PetscInt m, n;
	MatInfo info;
	double nnz;
	int niters;

	PetscInitialize(&argc, &argv, (char*)0, help);

	int nthds, np;
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	#pragma omp parallel
	{
		nthds = omp_get_num_threads();
	}

	niters = atoi(argv[4]);
	PetscPrintf(PETSC_COMM_WORLD, "np %d nthds %d\n", np, nthds);


	double read_mat_beg = MPI_Wtime();
	
	PetscPrintf(PETSC_COMM_WORLD, "reading matrix %s (A)\n", argv[1]);
	PetscViewerBinaryOpen(PETSC_COMM_WORLD, argv[1] ,FILE_MODE_READ, &fd);
	MatCreate(PETSC_COMM_WORLD, &A);
	MatSetType(A, MATMPIAIJ);
	MatLoad(A, fd);
	PetscViewerDestroy(&fd);

	MatGetSize(A, &m, &n);
	MatGetInfo(A, MAT_GLOBAL_SUM, &info);
	PetscPrintf(PETSC_COMM_WORLD, "A matrix size %d %d %lld\n",
				m, n, (long long int)info.nz_used);

	PetscPrintf(PETSC_COMM_WORLD, "reading matrix %s (B)\n", argv[2]);
	PetscViewerBinaryOpen(PETSC_COMM_WORLD, argv[2] ,FILE_MODE_READ, &fd);
	MatCreate(PETSC_COMM_WORLD, &B);
	MatSetType(B, MATMPIAIJ);
	MatLoad(B, fd);
	PetscViewerDestroy(&fd);

	MatGetSize(B, &m, &n);
	MatGetInfo(B, MAT_GLOBAL_SUM, &info);
	PetscPrintf(PETSC_COMM_WORLD, "B matrix size %d %d %lld\n",
				m, n, (long long int)info.nz_used);

	double read_mat_end = MPI_Wtime();
	

	PetscPrintf(PETSC_COMM_WORLD, "Performing SpGEMM\n");
	int i;
	double start_time = MPI_Wtime();
	for (i = 0; i < niters; ++i)
	{
		MatMatMult(A, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C);
	}
	double end_time = MPI_Wtime();

	PetscPrintf(PETSC_COMM_WORLD, "IO %lf spgemm %lf\n",
				read_mat_end-read_mat_beg,
				(end_time-start_time)/niters);

	MatGetSize(C, &m, &n);
	MatGetInfo(C, MAT_GLOBAL_SUM, &info);
	PetscPrintf(PETSC_COMM_WORLD, "C matrix size %d %d %lld\n",
				m, n, (long long int)info.nz_used);


	/* PetscPrintf(PETSC_COMM_WORLD, "Writing the output matrix\n"); */
	/* PetscViewerBinaryOpen(PETSC_COMM_WORLD, argv[3], FILE_MODE_WRITE, &fd); */
	/* MatView(C, fd); */


	MatDestroy(&A);
	MatDestroy(&B);
	MatDestroy(&C);
	PetscFinalize();


	return 0;	
}
