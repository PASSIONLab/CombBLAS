#include <cstdlib>
#include <climits>
#include <cusp/io/matrix_market.h>
#include <cusp/array2d.h>
#include <cusp/functional.h>
#include <cusp/multiply.h>
#include <cusp/print.h>
#include <random>

int main(int argc, char *argv[])
{
    // // create a simple example
    // cusp::array2d<float, cusp::host_memory> A(3,4);
    // A(0,0) = 10;  A(0,1) =  0;  A(0,2) = 20;  A(0,3) =  0;
    // A(1,0) =  0;  A(1,1) = 30;  A(1,2) =  0;  A(1,3) = 40;
    // A(2,0) = 50;  A(2,1) = 60;  A(2,2) = 70;  A(2,3) = 80;

    // // save A to disk in MatrixMarket format
    // cusp::io::write_matrix_market_file(A, "A.mtx");

    // // load A from disk into a coo_matrix
    // cusp::coo_matrix<int, float, cusp::device_memory> B;
    // cusp::io::read_matrix_market_file(B, "A.mtx");

    // // print B
    // cusp::print(B);

	std::default_random_engine gen;
	std::exponential_distribution<double> exp_dist(1.0);

	// long m = 822922;
	// long n = 1437547;
	long m = 219715;
	long n = 219715;
	int k = std::stoi(argv[1]);
	srand(7);
	
	cusp::array2d<double, cusp::host_memory> X(n, k);
	for (long i = 0; i < n; ++i)
	{
		for (int j = 0; j < k; ++j)
		{
			// X(i, j) = rand() / INT_MAX;
			X(i, j) = exp_dist(gen);
		}
	}
	cusp::array2d<double, cusp::device_memory> d_X(X);

	cusp::array2d<double, cusp::host_memory> Y(m, k);
	cusp::array2d<double, cusp::device_memory> d_Y(Y);

	cusp::constant_functor<double> initialize(std::numeric_limits<double>::max());
	thrust::minimum<double> reduce;
	thrust::project2nd<double, double> combine;

	cusp::csr_matrix<long, double, cusp::device_memory> d_A;
	cusp::io::read_matrix_market_file(d_A, "rank0-A.mtx");

	std::cout << "read complete" << std::endl;


	cusp::multiply(d_A, d_X, d_Y, initialize, combine, reduce);

	std::cout << "multiply complete" << std::endl;
	std::cout << "writing X" << std::endl;

	cusp::io::write_matrix_market_file(d_X, "rank0-X.mtx");

	std::cout << "writing Y" << std::endl;
	cusp::io::write_matrix_market_file(d_Y, "rank0-Y.mtx");

    return 0;
}

