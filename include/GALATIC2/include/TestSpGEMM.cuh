
#include <assert.h>
#include "CPU_SpGEMM.h"
#include "CSR.cuh"
#include "dCSR.cuh"
#include "../source/device/Multiply.cuh"

template<typename SEMIRING_T, typename F>
void TestSpGEMM( dCSR<typename SEMIRING_T::leftInput_t>& A, dCSR<typename SEMIRING_T::rightInput_t>& B, SEMIRING_T semiring, F equiv_rel, GPUMatrixMatrixMultiplyTraits& traits)
{

	//bool checkBitStability{true};
	ExecutionStats stats, warmupstats, output_stats;
	stats.measure_all = false;
	output_stats.measure_all = false;

	dCSR<typename SEMIRING_T::output_t> result_mat;

	std::cout << "starting GPU matrix multiply" << std::endl;

	ACSpGEMM::Multiply<SEMIRING_T>(A, B, result_mat, traits, warmupstats, true, semiring);
    cudaDeviceSynchronize();
    std::cout << "GPU matrix multiply Done"  << std::endl;



    // Convert input matrices

    CSR<typename SEMIRING_T::leftInput_t> A_cpu;
    CSR<typename SEMIRING_T::rightInput_t> B_cpu;

    convert(A_cpu, A);

    convert(B_cpu, B);

    cudaDeviceSynchronize();

    //convert gpu result to cpu
    CSR<typename SEMIRING_T::output_t> GPU_result_cpu;
    cudaDeviceSynchronize();

	convert(GPU_result_cpu, result_mat);

    cudaDeviceSynchronize();


    CSR<typename SEMIRING_T::output_t> CPU_result_cpu;
	Mult_CPU<SEMIRING_T>(A_cpu, B_cpu, CPU_result_cpu, semiring);

    std::cout << "Checking = # Rows, Cols, NNZ....";
    assert(CPU_result_cpu.rows == GPU_result_cpu.rows);
  std::cout << "Cpu "<< CPU_result_cpu.cols << "gpu " << GPU_result_cpu.cols;
    assert(CPU_result_cpu.cols == GPU_result_cpu.cols);
    assert(CPU_result_cpu.nnz == GPU_result_cpu.nnz);

    std::cout << " correct" << std::endl; 
   
    std::cout << "Checking Equivalency for non zeros...";


    int correct = 0;
    for (int i = 0; i < CPU_result_cpu.nnz; i++) {
        if (equiv_rel(CPU_result_cpu.data[i], GPU_result_cpu.data[i])) {
            correct++;
        } 
    }

    std::cout << "num correct  " << correct <<  "/ "  << CPU_result_cpu.nnz << std::endl;
    assert(correct == CPU_result_cpu.nnz);

    std::cout << " correct" << std::endl;


    std::cout << "Checking Equivalency for Column Id's...";


   

    int correct_col_ids = 0;
    for (int i = 0; i < CPU_result_cpu.nnz; i++) {
        if (CPU_result_cpu.col_ids[i] == GPU_result_cpu.col_ids[i]) {
            correct_col_ids++;
        }
    }

    assert(correct_col_ids == CPU_result_cpu.nnz);

    std::cout << " correct" << std::endl;

    std::cout << "Checking Equivalency for Row offsets's...";

   int cor_row_ids = 0;




    for (int i = 0; i < CPU_result_cpu.rows+1; i++) {
        if (CPU_result_cpu.row_offsets[i] == GPU_result_cpu.row_offsets[i]) {
            cor_row_ids++;

        } else {
            std::cout << " issue at " << i<< " with " <<  CPU_result_cpu.row_offsets[i] << " vs  "<< GPU_result_cpu.row_offsets[i]<< std::endl;
        }
    }

    std::cout << cor_row_ids << " correct out of " << CPU_result_cpu.rows+1 << std::endl;

    assert(cor_row_ids == CPU_result_cpu.rows+1);
    std::cout << " correct" << std::endl;
    std::cout << "correctness check complete" << std::endl;

}