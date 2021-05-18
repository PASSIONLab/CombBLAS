#include <ctf.hpp>
#include <float.h>

#include <iostream>

using namespace CTF;

typedef double REAL_T;



int
main (int argc, char **argv)
{
	int np, rank, nthds;
	int64_t m, n, k;
	m = atol(argv[1]);
	n = atol(argv[2]);
	k = atol(argv[3]);

	MPI_Init(&argc, &argv);
  	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  	MPI_Comm_size(MPI_COMM_WORLD, &np);

	World dw(argc, argv);

	#pragma omp parallel
	{
		nthds = omp_get_num_threads();
	}	

	if (dw.rank == 0)
	{
		std::cout << "np = " << np
				  << " nthds = " << nthds << std::endl;
		std::cout << m << " " << n << " " << k << std::endl;
	}

	Matrix<REAL_T> A(m, n, NS|SP, dw);
	Matrix<REAL_T> B(n, k, NS|SP, dw);
	Matrix<REAL_T> C(m, k, NS|SP, dw);

	double read_mat_beg = MPI_Wtime();
    if (dw.rank == 0)
		std::cout << "reading matrix A..." << std::endl;
	A.read_sparse_from_file(argv[4]);

	if (dw.rank == 0)
		std::cout << "reading matrix B..." << std::endl;
	B.read_sparse_from_file(argv[5]);

	double read_mat_end = MPI_Wtime();

	

	int niter = atoi(argv[7]);
	if (dw.rank == 0)
	{
		std::cout << "Starting " << niter << " benchmarking iterations of "
				  << "matrix multiplication with specified attributes."
				  << std::endl;
		initialize_flops_counter();
    }

	double min_time = DBL_MAX, max_time = 0.0, tot_time = 0.0, times[niter];
	Timer_epoch smatmul("specified matmul");
    smatmul.begin();
	for (int i = 0; i < niter; ++i)
	{
		if (dw.rank == 0)
			std::cout << "iteration " << i << std::endl;
		double start_time = MPI_Wtime();
		C["ik"] = A["ij"]*B["jk"];
		double end_time = MPI_Wtime();
      	double iter_time = end_time-start_time;
		times[i] = iter_time;
      	tot_time += iter_time;
		if (iter_time < min_time)
			min_time = iter_time;
      	if (iter_time > max_time)
			max_time = iter_time;
	}
	smatmul.end();

	if (dw.rank == 0)
	{
		std::sort(times, times+niter);
		std::cout << "iterations completed, did "
				  << (CTF::get_estimated_flops()/niter)
				  << " flops (per iteration)" << std::endl;
		std::cout << "Min time = " << min_time
				  << " Avg time = " <<  tot_time/niter
				  << " Med time = " << times[niter/2]
				  << " Max time = " << max_time << std::endl;
		std::cout << "I/O read " << read_mat_end-read_mat_beg << std::endl;
	}

	// if (dw.rank == 0)
	// 	std::cout << "writing matrix C..." << std::endl;
	// C.write_sparse_to_file(argv[6]);


	MPI_Finalize();

	return (EXIT_SUCCESS);
}
