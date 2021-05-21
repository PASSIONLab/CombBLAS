#include <chrono>
#include <iostream>
#include <string>

#include "cuda.h"
#include "cusparse.h"

#include "CombBLAS/CombBLAS.h"

using namespace std;
using namespace combblas;

typedef int64_t TEST_IDX_T;
typedef double TEST_NNZ_T;



void
wake_gpus (int myrank)
{
	int ndevices;
	cudaGetDeviceCount(&ndevices);
	if (myrank == 0)
	{
		std::cout << "Number of GPUs per node " << ndevices << "\n"
				  << std::flush;
		std::cout << "Waking the GPUs..." << std::flush;
	}
	int ts = 0;
	for (int i = 0; i < ndevices; ++i)
	{
		cudaSetDevice(i);
		int *array;
	 	int *dArray;
	 	int	 count = 7;
	 	int	 size  = count * sizeof(int);
	 	array	   = new int[count];
	 	for (int j = 0; j < count; j += 1)
	 		array[j] = j;
		cudaMalloc(&dArray, size);
		cudaMemcpy(dArray, array, size, cudaMemcpyHostToDevice);
		cudaFree(dArray);
		delete[] array;
	}
	if (myrank == 0)
		std::cout << " DONE!\n" << std::flush;
}



int main(int argc, char* argv[])
{
	

	{
		int nprocs, myrank, nthreads = 1;
		#ifdef _OPENMP
    	int provided, flag, claimed;
    	MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided );
    	MPI_Is_thread_main( &flag );
    	if (!flag)
        	SpParHelper::Print("This thread called init_thread but "
							   "Is_thread_main gave false\n");
    	MPI_Query_thread( &claimed );
    	if (claimed != provided)
        	SpParHelper::Print("Query thread gave different thread "
							   "level than requested\n");
		#else
		MPI_Init(&argc, &argv);
		#endif

		MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
		MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
		wake_gpus(myrank);

		int k = atoi(argv[2]);
		int nruns = atoi(argv[3]);

		#ifdef THREADED
		#pragma omp parallel
		{
			nthreads = omp_get_num_threads();
		}
		#endif

		string s_tmp;

		if (myrank == 0)
			cout << sqrt(nprocs) << " " << sqrt(nprocs)
				 << " " << nthreads << endl;

		shared_ptr<CommGrid> fullWorld;
		fullWorld.reset(new CommGrid(MPI_COMM_WORLD, 0, 0));

		SpParMat<TEST_IDX_T, TEST_NNZ_T,
				 SpCCols<TEST_IDX_T, TEST_NNZ_T> > A(fullWorld);
		A.ParallelReadMM(string(argv[1]), 1, maximum<TEST_NNZ_T>());

		TEST_IDX_T nr = A.getnrow(), nc = A.getncol(), nnz = A.getnnz();
		TEST_NNZ_T imb = A.LoadImbalance();
		if (myrank == 0)
		{
			cout << "Matrix A nr " << nr << " nc " << nc
				 << " nnz " << nnz << std::endl;
			cout << "load imb A: " << imb << std::endl;
		}

		TEST_IDX_T nnz_loc = A.seqptr()->getnnz();
		TEST_IDX_T nnzs[2];
		MPI_Reduce(&nnz_loc, &nnzs[0], 1,
				   sizeof(TEST_IDX_T) == 4 ? MPI_INT : MPI_LONG_LONG_INT,
				   MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&nnz_loc, &nnzs[1], 1,
				   sizeof(TEST_IDX_T) == 4 ? MPI_INT : MPI_LONG_LONG_INT,
				   MPI_MAX, 0, MPI_COMM_WORLD);
		if (myrank == 0)
			cout << "nnzs " << nnzs[0] << " " << (nnz/nprocs) << " "
				 << nnzs[1] << std::endl;
	
	
		DnParMat<TEST_IDX_T, TEST_NNZ_T> X(fullWorld, A.getncol(), k, 1.0);
		typedef PlusTimesSRing<TEST_NNZ_T, TEST_NNZ_T> PTSR;

		spmm_stats stats = {0};
		auto t_beg = std::chrono::high_resolution_clock::now();

		for (int i = 0; i < nruns; ++i)
		{
			SpParHelper::Print(".");
			auto Y = SpMM_sC<PTSR>(A, X, stats);
			// Y.PrintToFile("SpMM-Y-sC-2D-P");
		}

		auto t_end = std::chrono::high_resolution_clock::now();

		print_spmm_stats(stats, nruns);
		if (myrank == 0)
		{
			std::cout << "Overall SpMM time: " <<
				static_cast<std::chrono::duration<double> >(t_end-t_beg).count()
					  << std::endl;		
		}
	}
	
	MPI_Finalize();
	
	return 0;
}

