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
* performTestCase.cpp
*
* ac-SpGEMM
*
* Authors: Daniel Mlakar, Markus Steinberger, Martin Winter
*------------------------------------------------------------------------------
*/

// Global includes
#include <fstream>
#include <iostream>
#include <ctime>
#include <iomanip>
#include <string>
#include <sstream> 
#include <random>
#include <algorithm>
#include <cuda_runtime.h>

#ifdef _WIN32
#include <intrin.h>
//surpress crash notification windows (close or debug program window)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <x86intrin.h>
#endif

// Local includes
#include "CSR.h"
#include "COO.h"
#include "Vector.h"
#include "dCSR.h"
#include "dVector.h"
#include "Multiply.h"
#include "Transpose.h"
#include "Compare.cuh"
#include "consistent_memory.h"
#include "CustomExceptions.h"

#ifdef _WIN32
#include <filesystem>
using namespace std::filesystem;
#else
#include <experimental/filesystem>
using namespace std::experimental::filesystem;
#endif

// CuSparse include
#include "cusparse/include/cuSparseMultiply.h"

// // Nsparse include
// #include "nsparse/include/nsparseMultiply.h"

// // RMerge include
// #include "RMerge/include/rmergeMultiply.h"

// // BhSparse include
// #include"bhSparse/include/bhSparseMultiply.h"

unsigned int padding = 0;
template<typename T>
std::string typeext();
template<>
std::string typeext<float>()
{
	return std::string("");
}
template<>
std::string typeext<double>()
{
	return std::string("d_");
}

template<typename Format>
std::string nameextension()
{
	return "";
}
template<>
std::string nameextension<double>()
{
	return "_d";
}
template<>
std::string nameextension<float>()
{
	return "_f";
}

template<typename Format>
bool isFloat()
{
	return false;
}

template<>
bool isFloat<float>()
{
	return true;
}

// #################################################################
//
uint32_t numTrailingBinaryZeros(uint32_t n)
{
    uint32_t mask = 1;
    for (uint32_t i = 0; i < 32; i++, mask <<= 1)
        if ((n & mask) != 0)
            return i;

    return 32;
}

// #################################################################
//
void writeDetailedInfo(const ExecutionStats& stats, std::ofstream& out)
{
	out << stats.shared_rows << ";";
	out << stats.simple_rows << ";";
	out << stats.simple_mergers << ";";
	out << stats.complex_rows << ";";
	out << stats.generalized_rows << ";";
	out << stats.duration << ";";
	out << stats.duration_blockstarts << ";";
	out << stats.duration_spgemm << ";";
	out << stats.duration_merge_case_computation << ";";
	out << stats.duration_merge_simple << ";";
	out << stats.duration_merge_max << ";";
	out << stats.duration_merge_generalized << ";";
	out << stats.duration_write_csr << ";";
	out << stats.mem_clear_return << ";";
	out << stats.mem_allocated_chunks << ";";
	out << stats.mem_used_chunks << ";";
	out << stats.restarts << ";";
	out << std::endl;
}

// #################################################################
//
void getNextMatrix(const char* foldername, const std::string& lastname, std::string& nextname)
{
	bool found_last = false;
	directory_iterator it{ foldername };
	for (; it != directory_iterator{}; ++it)
	{
		if (!is_regular_file(*it))
			continue;
		if (it->path().extension() != ".mtx")
			continue;
		if (!found_last)
		{
			if (it->path().filename() != lastname)
				continue;
			else
			{
				found_last = true;
				continue;
			}
		}
		else
		{
			nextname = it->path().filename().string();
			return;
		}
	}
	nextname = std::string("");
	return;
}

// #################################################################
//
std::string getColumnHeaders(uint32_t approaches, std::string prefix = "")
{
	std::string headers(prefix);

	if (approaches & (0x1 << 0))
		headers.append("cuSparse;");
	if (approaches & (0x1 << 1))
		headers.append("acSpGEMM;");
	// if (approaches & (0x1 << 2))
	// 	headers.append("nsparse;");
	// if (approaches & (0x1 << 3))
	// 	headers.append("RMerge;");
	// if (approaches & (0x1 << 4))
	// 	headers.append("bhSparse;");

	headers.append("\n");

	return headers;
}

// #################################################################
//
template<typename ValueType>
void writeMatrixStats(CSR<ValueType>& mat, const std::string matname, std::ofstream& outfs)
{
	typename CSR<ValueType>::Statistics stats = mat.rowStatistics();
	//"\nMatrix;rows;cols;nnz;r_mean;r_std_dev;r_min;r_max;
	outfs << matname << ";" << mat.rows << ";" << mat.cols << ";" << mat.nnz << ";"
		<< stats.mean << ";" << stats.std_dev << ";" << stats.min << ";" << stats.max << ";";
}

// #################################################################
//
template<typename ValueType>
size_t countFloatingPointOperations(CSR<ValueType>& matA, CSR<ValueType>& matB)
{
	size_t count = 0;
	for (auto nnzAiter = 0; nnzAiter < matA.nnz; ++nnzAiter)
		count += matB.row_offsets[matA.col_ids[nnzAiter] + 1] - matB.row_offsets[matA.col_ids[nnzAiter]];
	return count;
}

// #################################################################
//
std::ostream& writeGPUInfo(std::ostream& file)
{
	int cudaDevice;
	cudaGetDevice(&cudaDevice);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, cudaDevice);
	std::cout << "Going to use " << prop.name << " " << prop.major << "." << prop.minor << "\n";

	file << "name;cc;num_multiprocessors;warp_size;max_threads_per_mp;regs_per_mp;shared_memory_per_mp;total_constant_memory;total_global_memory;clock_rate;max_threads_per_block;max_regs_per_block;max_shared_memory_per_block\n"
		<< prop.name << ';'
		<< prop.major << '.'
		<< prop.minor << ';'
		<< prop.multiProcessorCount << ';'
		<< prop.warpSize<< ';'
		<< prop.maxThreadsPerMultiProcessor << ';'
		<< prop.regsPerMultiprocessor << ';'
		<< prop.sharedMemPerMultiprocessor << ';'
		<< prop.totalConstMem << ';'
		<< prop.totalGlobalMem << ';'
		<< prop.clockRate * 1000 << ';'
		<< prop.maxThreadsPerBlock << ';'
		<< prop.regsPerBlock << ';'
		<< prop.sharedMemPerBlock
		<< std::endl;
	return file;
}

// #################################################################
//
template<typename ValueType>
int performSpGEMMTests(int argc, char ** argv)
{
	std::string name_extension = "";

	bool runtests = true;
	if (argc > 2)
		runtests = std::string(argv[2]) != "0";

	int cudaDevice = 0;
	if (argc > 3)
		cudaDevice = std::atoi(argv[3]);

	bool continue_run = false;
	if (argc > 4)
		continue_run = std::string(argv[4]) != "0";

	std::vector<int> trait_init = { 256, 3, 2, 4, 4, 16, 256, 8 };
	if (argc > 5)
	{

		std::istringstream traitstream(argv[5]);
		std::vector<int> input_trait_init;
		std::string val;
		while (std::getline(traitstream, val, ','))
			input_trait_init.push_back(std::stoi(val));

		if (input_trait_init.size() != trait_init.size())
			printf("Malformed trait init input param; %zu params required; fallback to default\n", trait_init.size());
		else
			trait_init = input_trait_init;
	}

	uint32_t approach_selector = 0xFFFFFFFF;
	uint32_t first_approach = 0;
	if (argc > 6)
	{
		approach_selector = std::stoi(argv[6]);		
		first_approach = numTrailingBinaryZeros(approach_selector);
		if (approach_selector == 0)
		{
			printf("ERROR: No approaches selected for testing\n");
			return 0;
		}
	}

	cudaSetDevice(cudaDevice);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, cudaDevice);
	std::cout << "Going to use " << prop.name << " " << prop.major << "." << prop.minor << "\n";
	std::string gpuname = prop.name;

	GPUMatrixMatrixMultiplyTraits DefaultTraits(trait_init[0], trait_init[1], trait_init[2], trait_init[3], trait_init[4], trait_init[5], trait_init[6], trait_init[7]);
	DefaultTraits.preferLoadBalancing = true;

	std::ofstream results;
	std::ofstream stateout;
	std::ofstream statsout; //This will go horribly wrong: stateout vs statsout
	std::string trait_string =
		std::to_string(trait_init[0]) +
		"_" + std::to_string(trait_init[1]) +
		"_" + std::to_string(trait_init[2]) +
		"_" + std::to_string(trait_init[3]) +
		"_" + std::to_string(trait_init[4]) +
		"_" + std::to_string(trait_init[5]) +
		"_" + std::to_string(trait_init[6]) +
		"_" + std::to_string(trait_init[7]) + "_";
	std::string statefile = std::string("bit_") + trait_string + nameextension<ValueType>() + name_extension + gpuname + ".state";
	std::string lastname;
	std::string current_name;
	unsigned num_approaches = 6;
	unsigned current_approach = first_approach;
	bool finished_write = true;
	bool fresh_file = !continue_run;
	if (continue_run)
	{
		std::ifstream last(statefile.c_str());
		if (last)
		{
			std::getline(last, lastname);
			current_name = lastname;
			std::cout << "Continuing run after " << lastname << std::endl;
			results.open((std::string("bit_") + trait_string + nameextension<ValueType>() + name_extension + gpuname + ".csv").c_str(), std::ios_base::app);
			statsout.open("matrix_stats.csv", std::ios_base::app);
			std::cout << "After open" << std::endl;


			std::time_t now = std::time(NULL);
			std::tm * ptm = std::localtime(&now);
			char buffer[32];
			// Format: Mo, 15.06.2009 20:20:00
			std::strftime(buffer, 32, "%a, %d.%m.%Y %H:%M:%S", ptm);
			std::cout << buffer << std::endl;

			std::string lastapproach;
			std::getline(last, lastapproach);
			current_approach = (std::stoi(lastapproach) + 1) % num_approaches;
			std::string finished_write_string;
			std::getline(last, finished_write_string);
			finished_write = !finished_write_string.empty();

			if (!finished_write)
			{
				results << -3 << ";";
				finished_write = true;
			}

			last.close();

			if (!(approach_selector & (0x1 << current_approach)))
			{
				//this limits us to 31 approaches :-p
				uint32_t next_offset = numTrailingBinaryZeros((approach_selector & 0xEFFFFFFF) >> current_approach);
				if (next_offset < sizeof(uint32_t) * 8)
				{
					current_approach += next_offset;
				}
				else
				{
					current_approach = first_approach;

					results << std::endl;
					
					const char  *foldername = argc == 1 ? "." : argv[1];
					getNextMatrix(foldername, lastname, current_name);

					if (current_name.empty())
					{
						return 0;
					}
						
				}
			}
			else if (current_approach < std::stoi(lastapproach))
			{
				const char  *foldername = argc == 1 ? "." : argv[1];
				getNextMatrix(foldername, lastname, current_name);

				if (current_name.empty())
				{
					return 0;
				}

				results << std::endl;

				if (current_name.empty())
				{
					return 0;
				}
			}
		}
		else
		{
			fresh_file = true;
		}
		last.close();
		stateout.open(statefile.c_str());
	}

	if (fresh_file)
	{

		results.open((std::string("bit_") + trait_string + nameextension<ValueType>() + name_extension + gpuname + ".csv").c_str());
		results << "\"sep=;\"\n";
		writeGPUInfo(results);
		results << getColumnHeaders(approach_selector, "\nMatrix;rows;cols;nnz;r_mean;r_std_dev;r_min;r_max;Products;");

		statsout.open("matrix_stats.csv", std::ios_base::app);
		statsout << "\"sep=;\"\n";
		statsout << "\nMatrix; rows; cols; nnz; r_mean; r_std_dev; r_min; r_max;" << std::endl;
	}


	CSR<ValueType> csrmat, csrmat2, result_mat;
	
	char  *foldername;
	if (argc == 1)
	{
		foldername = const_cast<char*>(".");
	}
	else
		foldername = argv[1];

	bool found = fresh_file;
	directory_iterator it{ foldername };

	for (; it != directory_iterator{}; ++it)
	{
		if (!is_regular_file(*it))
		{
			continue;
		}			
		if (it->path().extension() != ".mtx")
		{
			continue;
		}
		if (!found && continue_run)
		{
			if (current_name.compare(it->path().filename().string()) != 0)
			{
				// std::cout << "Filename not current name\n";
				// std::cout << it->path().filename() << it->path().filename().string().length() <<  std::endl;
				// std::cout << current_name << current_name.length() << std::endl;
				continue;
			}
			else
				found = true;
		}

		std::string testname = it->path().filename().stem().string();
		std::cout << "\n\nrunning " << testname << std::endl;
		std::string mantname = it->path().string();
		std::string csr_name = mantname + typeext<ValueType>() + ".hicsr";

		if (approach_selector & (0x1 << current_approach))
		{
			try
			{
				std::cout << "trying to load csr file \"" << csr_name << "\"\n";
				csrmat = loadCSR<ValueType>(csr_name.c_str());
				std::cout << "succesfully loaded: \"" << csr_name << "\"\n";
			}
			catch (std::exception& ex)
			{
				std::cout << "could not load csr file:\n\t" << ex.what() << "\n";
				try
				{
					std::cout << "trying to load mtx file \"" << mantname << "\"\n";
					COO<ValueType> coo_mat = loadMTX<ValueType>(mantname.c_str());
					convert(csrmat, coo_mat);
					std::cout << "succesfully loaded and converted: \"" << csr_name << "\"\n";
				}
				catch (std::exception& ex)
				{
					std::cout << ex.what() << std::endl;
					std::cout << "Skipping matrix \"" << mantname.c_str() << "\"\n";
					continue;
				}
				try
				{
					std::cout << "write csr file for future use\n";
					storeCSR(csrmat, csr_name.c_str());
				}
				catch (std::exception& ex)
				{
					std::cout << ex.what() << std::endl;
				}
			}
		}

		if (current_approach == first_approach)
		{
			auto rowStats = csrmat.rowStatistics();

			results << testname << ";";
			results << csrmat.rows << ";" << csrmat.cols << ";" << csrmat.nnz << ";"
				<< rowStats.mean << ";" << rowStats.std_dev << ";" << rowStats.min << ";" << rowStats.max << ";";
		}

		if (continue_run)
			stateout << it->path().filename().string() << std::endl << current_approach << std::endl;

		if (runtests)
		{
			std::cout << "Matrix: " << csrmat.rows << "x" << csrmat.cols << ": " << csrmat.nnz << " nonzeros\n";

			int32_t iterations = 20;

			try
			{
				dCSR<ValueType> gpu_csrmat, gpu_csrmat2, d_csr_cuRes;
				convert(gpu_csrmat, csrmat, 0);
				cuSPARSE::CuSparseTest<ValueType> cusparse;

				//calculate the transpose if matrix is not square
				if (gpu_csrmat.rows != gpu_csrmat.cols)
				{
					cusparse.Transpose(gpu_csrmat, gpu_csrmat2);
					convert(csrmat2, gpu_csrmat2);
				}
				else
				{
					convert(gpu_csrmat2, csrmat, 0);
					convert(csrmat2, csrmat, 0);
				}

				//generate reference solution using cuSparse
				unsigned cuSubdiv_nnz = 0;
				if (current_approach != 0 || current_approach == first_approach)
				{
					cusparse.Multiply(gpu_csrmat, gpu_csrmat2, d_csr_cuRes, cuSubdiv_nnz);

					if (current_approach == first_approach)
					{
						//write out stats of result matrix
						CSR<ValueType> h_csr_cuRes;
						convert(h_csr_cuRes, d_csr_cuRes);
						writeMatrixStats(h_csr_cuRes, testname, statsout);
						size_t fpo = countFloatingPointOperations(csrmat, csrmat2);
						std::cout << "Multiplication Requires " << fpo << " Floating point operations" << std::endl;
						statsout << fpo << std::endl;
						results << fpo << ";";
						statsout.flush();
						statsout.close();
						}
				}

				switch (current_approach)
				{
				case 0:
				{
					dCSR<ValueType> d_csr_cuRes_comp;
					cuSPARSE::CuSparseTest<ValueType> cuSparseTest;
					bool bitstable{true};

					for (int i = 0; i < iterations; i++)
					{
						if(i == 0)
							cuSparseTest.Multiply(gpu_csrmat, gpu_csrmat2, d_csr_cuRes, cuSubdiv_nnz);
						else
						{
							cuSparseTest.Multiply(gpu_csrmat, gpu_csrmat2, d_csr_cuRes_comp, cuSubdiv_nnz);
							if (!(ACSpGEMM::Compare<ValueType>(d_csr_cuRes, d_csr_cuRes_comp, true)))
							{
								printf("cuSparse: ## NOT ## Bit-Identical\n");
								results << -999 << ";";
								bitstable = false;
								break;
							}
						}
					}
					if(bitstable)
					{
						printf("cuSparse: Bit-Identical\n");
						results << 0 << ";";
					}	
					stateout << 1 << std::endl;				
					break;
				}
				case 1:
				{
					dCSR<ValueType> d_csr_hiRes, d_csr_hiRes_comp;
					ExecutionStats stats;
					stats.measure_all = false;
					bool bitstable{true};

					// Multiplication
					for (int i = 0; i < iterations; ++i)
					{
						stats.reset();
						if(i == 0)
							ACSpGEMM::Multiply<ValueType>(gpu_csrmat, gpu_csrmat2, d_csr_hiRes, DefaultTraits, stats, false);
						else
						{
							ACSpGEMM::Multiply<ValueType>(gpu_csrmat, gpu_csrmat2, d_csr_hiRes_comp, DefaultTraits, stats, false);
							if (!(ACSpGEMM::Compare<ValueType>(d_csr_hiRes, d_csr_hiRes_comp, true)))
							{
								printf("AcSpGEMM: ## NOT ## Bit-Identical\n");
								results << -999 << ";";
								bitstable = false;
								break;
							}
						}
					}

					if(bitstable)
					{
						printf("AcSpGEMM: Bit-Identical\n");
						results << 0 << ";";
					}	
					stateout << 1 << std::endl;				
					break;
				}
				case 2:
				{
					// dCSR<ValueType> d_nsparse_result_mat, d_nsparse_result_mat_comp;
					// bool bitstable{true};
					// // Multiplication
					// for (int i = 0; i < iterations; ++i)
					// {
					// 	d_nsparse_result_mat_comp.reset();
					// 	if(i == 0)
					// 		NSparse::Multiply<ValueType>(gpu_csrmat, gpu_csrmat2, d_nsparse_result_mat);
					// 	else
					// 	{
					// 		NSparse::Multiply<ValueType>(gpu_csrmat, gpu_csrmat2, d_nsparse_result_mat_comp);
					// 		if (!(ACSpGEMM::Compare<ValueType>(d_nsparse_result_mat, d_nsparse_result_mat_comp, true)))
					// 		{
					// 			printf("Nsparse: ## NOT ## Bit-Identical\n");
					// 			results << -999 << ";";
					// 			bitstable = false;
					// 			break;
					// 		}
					// 	}
					// }

					// if(bitstable)
					// {
					// 	printf("Nsparse: Bit-Identical\n");
					// 	results << 0 << ";";
					// }
					
					// stateout << 1 << std::endl;
					printf("Nsparse not included in public repository\n");
					break;
				}
				case 3:
				{
					// bool bitstable{true};
					// uint32_t rmerge_nnz{ 0 };
					// HiSparse::Test::RMergeExecutionStats rmerge_stats;
					// HostVector<uint32_t> rmerge_offsets(csrmat.row_offsets.get(), csrmat.rows + 1);
					// rmerge_offsets[csrmat.rows] = csrmat.nnz;
					// HostVector<uint32_t> rmerge_indices(csrmat.col_ids.get(), csrmat.nnz);
					// HostVector<ValueType> rmerge_values(csrmat.data.get(), csrmat.nnz);
					// SparseHostMatrixCSR<ValueType> host_A(csrmat.cols, csrmat.rows, rmerge_values, rmerge_indices, rmerge_offsets);
					
					// HostVector<uint32_t> rmerge_offsets2(csrmat2.row_offsets.get(), csrmat2.rows + 1);
					// rmerge_offsets2[csrmat2.rows] = csrmat2.nnz;
					// HostVector<uint32_t> rmerge_indices2(csrmat2.col_ids.get(), csrmat2.nnz);
					// HostVector<ValueType> rmerge_values2(csrmat2.data.get(), csrmat2.nnz);
					// SparseHostMatrixCSR<ValueType> host_B(csrmat2.cols, csrmat2.rows, rmerge_values2, rmerge_indices2, rmerge_offsets2);

					// SparseDeviceMatrixCSR<ValueType> A = ToDevice(host_A);
					// SparseDeviceMatrixCSR<ValueType> B = ToDevice(host_B);
					// SparseDeviceMatrixCSR<ValueType> C, C_comp;
		

					// RMerge::Multiply<ValueType>(A, B, C);
					// dCSR<ValueType> d_rmerge_result_mat, d_rmerge_result_mat_comp;
					// d_rmerge_result_mat.nnz = rmerge_nnz;
					// d_rmerge_result_mat.rows = csrmat.rows;
					// d_rmerge_result_mat.cols = csrmat2.cols;
					// d_rmerge_result_mat.row_offsets = C.RowStarts().Data();
					// d_rmerge_result_mat.col_ids = C.ColIndices().Data();
					// d_rmerge_result_mat.data = C.Values().Data();

					// // Multiplication
					// for (uint32_t i = 0; i < iterations; ++i)
					// {
					// 	RMerge::Multiply<ValueType>(A, B, C_comp);
					// 	rmerge_nnz = C_comp.NonZeroCount();
					// 	d_rmerge_result_mat_comp.nnz = rmerge_nnz;
					// 	d_rmerge_result_mat_comp.rows = csrmat.rows;
					// 	d_rmerge_result_mat_comp.cols = csrmat2.cols;
					// 	d_rmerge_result_mat_comp.row_offsets = C_comp.RowStarts().Data();
					// 	d_rmerge_result_mat_comp.col_ids = C_comp.ColIndices().Data();
					// 	d_rmerge_result_mat_comp.data = C_comp.Values().Data();
					// 	if (!(ACSpGEMM::Compare<ValueType>(d_rmerge_result_mat, d_rmerge_result_mat, true)))
					// 	{
					// 		printf("RMerge: ## NOT ## Bit-Identical\n");
					// 		results << -999 << ";";
					// 		bitstable = false;
					// 		break;
					// 	}
					// }

					// // Let the other object destroy the memory
					// d_rmerge_result_mat.row_offsets = nullptr;
					// d_rmerge_result_mat.col_ids = nullptr;
					// d_rmerge_result_mat.data = nullptr;
					
					// if(bitstable)
					// {
					// 	printf("RMerge: Bit-Identical\n");
					// 	results << 0 << ";";
					// }
					// stateout << 1 << std::endl;
					printf("RMerge not included in public repository\n");
					break;
				}
				case 4:
				{
					// dCSR<ValueType> d_bhSparse_result_mat, d_bhSparse_result_mat_comp;
					// bool bitstable{true};
					// HiSparse::Test::bhSparseExecutionStats bhsparse_stats;

					// // Multiplication
					// for (int i = 0; i < iterations; ++i)
					// {
					// 	d_bhSparse_result_mat_comp.reset();
					// 	if(i == 0)
					// 		bhSparse::Multiply<ValueType>(gpu_csrmat, gpu_csrmat2, d_bhSparse_result_mat);
					// 	else
					// 	{
					// 		bhSparse::Multiply<ValueType>(gpu_csrmat, gpu_csrmat2, d_bhSparse_result_mat_comp);
					// 		if (!(ACSpGEMM::Compare<ValueType>(d_bhSparse_result_mat, d_bhSparse_result_mat_comp, true)))
					// 		{
					// 			printf("BhSparse: ## NOT ## Bit-Identical\n");
					// 			results << -999 << ";";
					// 			bitstable = false;
					// 			break;
					// 		}
					// 	}
					// }

					// if(bitstable)
					// {
					// 	printf("BhSparse: Bit-Identical\n");
					// 	results << 0 << ";";
					// }					
					// stateout << 1 << std::endl;
					printf("bhSparse not included in public repository\n");
					break;
				}
				default:
					std::cout << "error: wrong test state" << std::endl;
					break;
				}
			}
			catch (const SpGEMMException& e) {
				std::cout << "Error:\n" << e.what() << "\n";

				results << "-4;";

				stateout << 0 << std::endl;
			}
			catch (const MergeSimpleCaseException& e) {
				std::cout << "Error:\n" << e.what() << "\n";

				results << "-5;";

				stateout << 0 << std::endl;
			}
			catch (const MergeMaxChunksCaseException& e) {
				std::cout << "Error:\n" << e.what() << "\n";

				results << "-6;";

				stateout << 0 << std::endl;
			}
			catch (const MergeGeneralizedCaseException& e) {
				std::cout << "Error:\n" << e.what() << "\n";

				results << "-7;";

				stateout << 0 << std::endl;
			}
			catch (const MergeLoopingException& e) {
				std::cout << "Error:\n" << e.what() << "\n";

				results << "-8;";

				stateout << 0 << std::endl;
			}
			catch (const RestartOutOfMemoryException& e) {
				std::cout << "Error:\n" << e.what() << "\n";

				results << "-9;";

				stateout << 0 << std::endl;
			}
			catch (const RestartOutOfChunkPointerException& e) {
				std::cout << "Error:\n" << e.what() << "\n";

				results << "-10;";

				stateout << 0 << std::endl;
			}
			catch (const std::exception& e) {
				std::cout << "Error:\n" << e.what() << "\n";

				results << "-1;";

				stateout << 0 << std::endl;
			}
			results.flush();
			stateout.flush();
		}
		results.flush();
		results.close();
		stateout.flush();
		stateout.close();

		if (continue_run)
			return 1;
	}
	std::cout << "Test done\n";
	return 0;
}

// #################################################################
//
int main(int argc, char *argv[])
{
#ifdef _WIN32
	//surpress crash notification windows (close or debug program window)
	SetErrorMode(GetErrorMode() | SEM_NOGPFAULTERRORBOX);
#endif

	std::string value_type = argc > 7 ? argv[7] : "f";
	if (value_type.compare("f") == 0)
		return performSpGEMMTests<float>(argc, argv);
	else
		return performSpGEMMTests<double>(argc, argv);
}