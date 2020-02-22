#include <cuda.h>
#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/functional.h>
#include <cusp/multiply.h>
#include <cusp/print.h>



int main(void)
{
	cusp::csr_matrix<long,float,cusp::host_memory> A(4,3,6);

	// initialize matrix entries on host
    A.row_offsets[0] = 0;  // first offset is always zero
    A.row_offsets[1] = 2;
    A.row_offsets[2] = 2;
    A.row_offsets[3] = 3;
    A.row_offsets[4] = 6; // last offset is always num_entries

    A.column_indices[0] = 0; A.values[0] = 1;
    A.column_indices[1] = 1; A.values[1] = 1;
    A.column_indices[2] = 1; A.values[2] = 1;
    A.column_indices[3] = 0; A.values[3] = 1;
    A.column_indices[4] = 1; A.values[4] = 1;
    A.column_indices[5] = 1; A.values[5] = 1;


	cusp::array2d<float, cusp::host_memory> X(3,2);
	X(0,0) = 5; X(0,1) = 7;
	X(1,0) = 3; X(1,1) = 1;
	X(2,0) = 9; X(2,1) = 2;


	cusp::array2d<float, cusp::host_memory> Y(4,2);




	cusp::csr_matrix<long,float,cusp::device_memory> A_d(A);
	cusp::array2d<float,cusp::device_memory> X_d(X);
	cusp::array2d<float,cusp::device_memory> Y_d(Y);


	// define multiply functors
	cusp::constant_functor<float> initialize(std::numeric_limits<float>::max());
	// thrust::multiplies<float> combine;
	thrust::minimum<float>       reduce;
	thrust::project2nd<float,float> combine;
	
	// // initialize matrix
	// cusp::array2d<float, cusp::host_memory> A(2,2);
	// A(0,0) = 10;  A(0,1) = 20;
	// A(1,0) = 40;  A(1,1) = 50;

	// // initialize input vector
	// cusp::array1d<float, cusp::host_memory> x(2);
	// x[0] = 1;
	// x[1] = 2;

	// // allocate output vector
	// cusp::array1d<float, cusp::host_memory> y(2);

	// compute y = A * x
	cusp::multiply(A_d, X_d, Y_d, initialize, combine, reduce);

	// print y
	cusp::print(Y_d);

	return 0;
}
