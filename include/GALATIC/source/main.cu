//  Project AC-SpGEMM
//  https://www.tugraz.at/institute/icg/research/team-steinberger/
//
//  Copyright (C) 2018 Institute for Computer Graphics and Vision,
//                     Graz University of Technology
//
//  Author(s):  Martin Winter - martin.winter (at) icg.tugraz.at
//              Daniel Mlakar - daniel.mlakar (at) icg.tugraz.at
//              Rhaleb Zayer - rzayer (at) mpi-inf.mpg.de
//              Hans-Peter Seidel - hpseidel (at) mpi-inf.mpg.de
//              Markus Steinberger - steinberger ( at ) icg.tugraz.at
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//  THE SOFTWARE.
//

/*!/------------------------------------------------------------------------------
 * Main.cpp
 *
 * ac-SpGEMM
 *
 * Authors: Daniel Mlakar, Markus Steinberger, Martin Winter
 *------------------------------------------------------------------------------
*/

// Global includes
#include <fstream>
#include <iostream>
#include <string>
#include <random>
#include <algorithm>
#include <string>
#include <tuple>
#include <cuda_runtime.h>
#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>


// Local includes
#include "CSR.cuh"
#include "COO.cuh"
#include "Vector.h"
#include "dCSR.cuh"
#include "dVector.h"
#include "device/Multiply.cuh"
#include "Transpose.h"
#include "Compare.cuh"
#include "CPU_SpGEMM.h"
// CuSparse include
//#include "cusparse/include/cuSparseMultiply.h"

// // Nsparse include
// #ifndef NONSPARSE
// #include "nsparse/include/nsparseMultiply.h"
// #endif

// // RMerge include
// #ifndef NORMERGE
// #include "RMerge/include/rmergeMultiply.h"
// #endif
// const uint64_t max(uint64_t x, uint64_t y) {
// 	return x < y ? x :y;
// }
// // BhSparse include
// #ifndef NOBHSPARSE
// #include"bhSparse/include/bhSparseMultiply.h"
// #endif

//foo::foo(int x) {
//    this->a =x;
//
//}
template<typename T>
void log_good(T& s) {
	std::cout << "\033[1;31," << s << "\033[0m";
}

struct triv {};

struct mr2 {
    int16_t val;
    uint8_t  temp;
	uint8_t  temp2;

	uint8_t  temp3;

	uint8_t  temp4;

	uint8_t  temp5;

};

struct MinRing : SemiRing<MinRing, mr2 , triv> {
    int16_t val;
	int16_t val2;


    // __device__ __host__ MinRing(int32_t x, int32_t y) {
    //     val = x;
    // }
	// __device__ __host__ MinRing(int32_t x) {
    //     val = x;
    // }

    // __device__ __host__ ~MinRing() {
    // }

    // __device__ __host__ MinRing() {
    //      val = INT16_MIN;
    // }

	static MinRing Init(double x) {
		MinRing res;
		res.val = (short) x;
		return res;
	}
   __device__ __host__ mr2 multiply( MinRing & a,  MinRing & b) const {
        return mr2 { static_cast<short>(a.val == INT16_MAX || b.val == INT16_MAX ? INT16_MAX :  a.val + b.val ),0};
    }
    __device__ __host__ mr2  add(const mr2 & a, const mr2 & b)const  {

        return mr2 { a.val < b.val ? a.val : b.val,0} ;
    }

    __device__ bool operator==(const MinRing& rhs) const
    {
        return val == rhs.val;
    }

    static __host__ __device__ MinRing MultiplicativeIdentity() {
        MinRing result;
        result.val = 0;
        return result;
    }
    static __host__ __device__ mr2 AdditiveIdentity() {
        return mr2  { INT16_MAX ,0};
    }
};






unsigned int padding = 0;
template<typename T>
std::string typeext() {
    //FIXME not-C++ standard compliant
    return typeid(T).name();
}
template<> 
std::string typeext<float>()
{
	return std::string("");
}
template<> std::string typeext<uint32_t>()
{
    return std::string("i32_");
}
template<> 
std::string typeext<double>()
{
	return std::string("d_");
}

void printCheckMark()
{
	printf("\n        #\n       #\n      #\n #   #\n  # #\n   #\n\n");
}

void printCross()
{
	printf("\n #     # \n  #   #  \n   # #   \n    #    \n   # #   \n  #   #  \n #     # \n\n");
}

int main(int argc, char *argv[])
{


	std::cout << "########## ac-SpGEMM ##########" << std::endl;

	char  *filename;
	bool print_stats{ false };
	if (argc == 1)
	{
		std::cout << "Require filename of .mtx as first argument" << std::endl;
		return -1;
	}

	filename = argv[1];

	 int device = 0;
	// if (argc >= 3)
	// 	device = std::stoi(argv[2]);
	
	 bool testing = false;
	// if(argc >= 4)
	// 	testing = std::stoi(argv[3]) > 0 ? true : false;

	cudaSetDevice(device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	std::cout << "Going to use " << prop.name << " " << prop.major << "." << prop.minor << "\n";

	// CSR matrices on the device
	CSR<MinRing::input_t> csr_mat, csr_T_mat, result_mat, test_mat;
	dCSR<MinRing::input_t> dcsr_mat, dcsr_T_mat ;//, d_nsparse_result_mat, d_rmerge_result_mat, d_bhSparse_result_mat;

	dCSR<MinRing::output_t> d_result_mat_comp, d_result_mat;
	//try load csr file
	std::string csr_name = std::string(argv[1]) + typeext<MinRing::input_t>() + ".hicsr";
	try
	{
		std::cout << "trying to load csr file \"" << csr_name << "\"\n";
		csr_mat = loadCSR<MinRing::input_t>(csr_name.c_str());
		std::cout << "succesfully loaded: \"" << csr_name << "\"\n";
	}
	catch (std::exception& ex)
	{
		std::cout << "could not load csr file:\n\t" << ex.what() << "\n";
		try
		{
			std::cout << "trying to load mtx file \"" << argv[1] << "\"\n";
			COO<MinRing::input_t> coo_mat= loadMTX<MinRing::input_t>(argv[1]);
			// coo_mat.alloc(2,2,4);
            // coo_mat.data[0]= MinRing::Init(1);
            // coo_mat.data[1]= MinRing::Init(2);
            // coo_mat.data[2]= MinRing::Init(3);
            // coo_mat.data[3]= MinRing::Init(4);


            // coo_mat.row_ids[0] = 0;
            // coo_mat.col_ids[0] = 0;

            // coo_mat.row_ids[1] = 0;
            // coo_mat.col_ids[1] = 1;


            // coo_mat.row_ids[2] = 1;
            // coo_mat.col_ids[2] = 0;

            // coo_mat.row_ids[3] = 1;
            // coo_mat.col_ids[3] = 1;




            convert(csr_mat, coo_mat);
			std::cout << "succesfully loaded and converted: \"" << csr_name << "\"\n";
		}
		catch (std::exception& ex)
		{
			std::cout << ex.what() << std::endl;
			return -1;
		}
		try
		{
			std::cout << "write csr file for future use\n";
			storeCSR(csr_mat, csr_name.c_str());
		}
		catch (std::exception& ex)
		{
			std::cout << ex.what() << std::endl;
		}
	}

	// Convert host csr to device csr
	convert(dcsr_mat, csr_mat, padding);



	bool transpose = (dcsr_mat.rows != dcsr_mat.cols);
	if (transpose)
	{
		std::cout << "Matrix not square (" << dcsr_mat.rows << "x" << dcsr_mat.cols << ") - Calculate Transpose!\n";
		/*ACSpGEMM::Transpose(dcsr_mat, dcsr_T_mat);*/
		convert(csr_T_mat, dcsr_T_mat, padding);
	}

	printf("Input Matrix A: (%zu x %zu) - NNZ: %zu\n", dcsr_mat.rows, dcsr_mat.cols, dcsr_mat.nnz);
	if(transpose)
		printf("Input Matrix B: (%zu x %zu) - NNZ: %zu\n", dcsr_T_mat.rows, dcsr_T_mat.cols, dcsr_T_mat.nnz);




	const int Threads = 128;
	const int BlocksPerMP = 1;
	const int NNZPerThread = 2;
	const int InputElementsPerThreads = 2;
	const int RetainElementsPerThreads = 1;
	const int MaxChunksToMerge = 8;
	const int MaxChunksGeneralizedMerge = 512; // MAX: 865
	const int MergePathOptions = 8;

	GPUMatrixMatrixMultiplyTraits DefaultTraits(Threads, BlocksPerMP, NNZPerThread, InputElementsPerThreads, RetainElementsPerThreads, MaxChunksToMerge, MaxChunksGeneralizedMerge, MergePathOptions); // DefaultTraits(128, 2, 4, 1, 8, 128, 8);
	const bool Debug_Mode = true;
	bool checkBitStability{true};
	DefaultTraits.preferLoadBalancing = true;
	ExecutionStats stats, warmupstats, output_stats;
	stats.measure_all = false;
	output_stats.measure_all = false;

	uint32_t warmupiterations = testing ? checkBitStability ? 1 : 0: 20;
	uint32_t iterations = testing ? 1 : 20;

	

		// Multiplication
			/*if (testing)
				std::cout << "Iteration: " << i + 1 << "\n";*/
            MinRing j = MinRing { };

			std::cout << "Performing SpGEMM, GPU" << std::endl;
            ACSpGEMM::Multiply<MinRing>(dcsr_mat, transpose ? dcsr_T_mat : dcsr_mat, d_result_mat, DefaultTraits, stats, Debug_Mode,j);
			std::cout << "SpGEMM Done\n";

			CSR<MinRing::output_t> out;
			std::cout << "Performing SpGEMM, CPU" << std::endl;

			Mult_CPU(csr_mat, csr_mat,  out, j);
			std::cout << "CPU-SpGEMM Done\n";

			std::ofstream log_f;


			if(argc >= 3)
			{
				log_f.open(argv[2]);
			}

				

                CSR<MinRing::output_t> coo_mat;

                convert(coo_mat, d_result_mat,0);
                COO<MinRing::input_t> coo;
                cudaDeviceSynchronize();

				uint64_t err_count = 0;
				uint64_t checked = 0;
				if (coo_mat.nnz != out.nnz) {
					if (argc >= 3) {
						log_f  << "ERROR:" << "nonzeros GPU: " << coo_mat.nnz << " vs non-zeros cpu:" << out.nnz <<std::endl;
					}
				    	std::cout << red << "ERROR:" << "nonzeros GPU: " << coo_mat.nnz << " vs non-zeros cpu:" << out.nnz <<std::endl;

				}
				if (argc >= 3) {
					for (int i =0; i < coo_mat.nnz; i++) {
		
						if (coo_mat.data[i].val != out.data[i].val){
							log_f  << "ERROR, NNZ Entry#: " << i << " (" << coo_mat.row_offsets[i] << ", " << coo_mat.col_ids[i] << ") gpu: "  << coo_mat.data[i].val << " vs  CPU: " << out.data[i].val << std::endl;
							err_count++;
						} else {
							checked++;// this can be calulated from errocount, but I'm being paranoid to make sure we don't trivially pass
						}
					}


					log_f  << "Total errors: " <<err_count <<std::endl;


				} else {
					for (int i =0; i < coo_mat.nnz; i++) {
		
						if (coo_mat.data[i].val != out.data[i].val){
							std::cout  << red << "ERROR, NNZ Entry#: " << i << "  ("  << coo_mat.row_offsets[i] << ", " << coo_mat.col_ids[i] << ") gpu: "  << coo_mat.data[i].val << " vs  CPU: " << out.data[i].val << std::endl;
							err_count++;
						} else {
							checked++;
						}
					}

				}


				
			if(argc >= 3)
			{
				log_f << "output NNZ checked" << coo_mat.nnz << std::endl;
				std::cout << "NNZ correct / # of checked output:  " <<  checked << "/" << coo_mat.nnz  << std::endl;

				log_f.close();

			}

				std::cout << "Total errors: " << err_count <<std::endl;


				std::cout << "NNZ correct / # of checked output/ total:  " <<  checked << "/" << coo_mat.nnz  << "/" << max(coo.nnz,out.nnz ) << std::endl;

//
//				if (!ACSpGEMM::Compare<MinRing::input_t>(d_result_mat_comp, d_result_mat, true))
//				{
//					printf("NOT Bit-Identical\n");
//					printCross();
//					exit(-1);
//				}
//				else
//				{
//					printf("Bit-Identical\n");
//					printCheckMark();
//				}

		


	// output_stats.normalize();
	// std::cout << output_stats;
	// std::cout << "-----------------------------------------------\n";

	// if(checkBitStability)
	// 	return 0;

	return 0;
}


