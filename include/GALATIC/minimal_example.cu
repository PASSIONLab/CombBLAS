/*******************************************
#include "GALATIC/include/CSR.cuh"
#include "GALATIC/include/dCSR.cuh"
#include "GALATIC/include/SemiRingInterface.h"
#include "GALATIC/source/device/Multiply.cuh"

Your "includes" probably needs to look something like the above, rather than what's below. 
*******************************************/

//#include "include/CSR.cuh"
//#include "include/dCSR.cuh"
#include "include/SemiRingInterface.h"
#include "include/TestSpGEMM.cuh"
#include <chrono>

//#include "source/device/Multiply.cuh"

struct foo {
    double a;
};

struct foo2 {
    short h;
    double a;
    double b;
    double c;

    double d;
    short k;
};

struct Arith_SR : SemiRing<double, double, double>
{
  __host__ __device__ double multiply(const double& a, const double& b) const { return a * b; }
  __host__ __device__ double add(const double& a, const double& b)   const   { return a + b; }
   __host__ __device__  static double AdditiveIdentity()                  { return     0; }
};


int main(int argc, const char* argv[]) 
{
    CSR<Arith_SR::leftInput_t> input_A_CPU;
    CSR<Arith_SR::rightInput_t> input_B_CPU;

    COO<Arith_SR::leftInput_t> input_A_COO;
    COO<Arith_SR::rightInput_t> input_B_COO;

    CSR<Arith_SR::output_t> result_mat_CPU;
    
    

    

    printf("%s + %s", argv[1], argv[2]);
    input_A_COO = loadMTX<Arith_SR::leftInput_t>(argv[1]);
    input_B_COO =  loadMTX<Arith_SR::rightInput_t>(argv[2]);

    convert(input_A_CPU, input_A_COO);
    convert(input_B_CPU, input_B_COO);
    
     // [ [ 1,  2],
     //   [ 3 4 ] ]
     cudaDeviceSynchronize();

    
    // Transfer input matrices onto GPU
    

    // load data into semiring struct. For this one, we don't need to do anything
    Arith_SR semiring;
    
    
    // Setup execution options, we'll skip the details for now.
    
    const int Threads = 128;
    const int BlocksPerMP = 1;
    const int NNZPerThread = 2;
    const int InputElementsPerThreads = 2;
    const int RetainElementsPerThreads = 1;
    const int MaxChunksToMerge = 16;
    const int MaxChunksGeneralizedMerge = 256; // MAX: 865
    const int MergePathOptions = 8;
    
    
    GPUMatrixMatrixMultiplyTraits DefaultTraits(Threads, BlocksPerMP, NNZPerThread,
                                                 InputElementsPerThreads, RetainElementsPerThreads,
                                                 MaxChunksToMerge, MaxChunksGeneralizedMerge, MergePathOptions );
    
    const bool Debug_Mode = false;
    // DefaultTraits.preferLoadBalancing = true;
     ExecutionStats stats;
    // stats.measure_all = false;
    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::milliseconds ms;
    typedef std::chrono::duration<float> fsec;
    auto t0 = Time::now();
    
    for (int i =0; i < 10000; i++){
    // Actually perform the matrix multiplicaiton
    //if (i % 10 == 0) printf("%i\n",i);
    dCSR<Arith_SR::leftInput_t> input_A_GPU;
    dCSR<Arith_SR::rightInput_t> input_B_GPU;
    convert(input_A_GPU, input_A_CPU);
    convert(input_B_GPU, input_B_CPU);
    cudaDeviceSynchronize();
    dCSR<Arith_SR::output_t> result_mat_GPU;
        ACSpGEMM::Multiply<Arith_SR>(input_A_GPU, input_B_GPU, result_mat_GPU, DefaultTraits, stats, Debug_Mode, semiring);
         cudaDeviceSynchronize();
         //std::cout << result_mat_GPU.nnz << std::endl;
         convert(result_mat_CPU, result_mat_GPU);
         cudaDeviceSynchronize();
    }
    auto t1 = Time::now();
    fsec fs = t1 - t0;
    ms d = std::chrono::duration_cast<ms>(fs);
    dCSR<Arith_SR::output_t> result_mat_GPU;
    dCSR<Arith_SR::leftInput_t> input_A_GPU;
    dCSR<Arith_SR::rightInput_t> input_B_GPU;
    convert(input_A_GPU, input_A_CPU);
    convert(input_B_GPU, input_B_CPU);
    ACSpGEMM::Multiply<Arith_SR>(input_A_GPU, input_B_GPU, result_mat_GPU, DefaultTraits, stats, Debug_Mode, semiring);
         cudaDeviceSynchronize();
    printf("Took %d for 1000 tries, for an average of %d\n", d, (d / 1000));
    TestSpGEMM(input_A_GPU, input_B_GPU, semiring, [=] (const Arith_SR::output_t &a, const Arith_SR::output_t &b) { return std::abs(a-b) < 0.01; }, DefaultTraits);

    convert(result_mat_CPU, result_mat_GPU);

    cudaDeviceSynchronize();

    for (int i =0; i < 4; i++) {
        std::cout << "nnz: " << i <<   " val " <<  result_mat_CPU.data[i] << std::endl;
    }
    
}