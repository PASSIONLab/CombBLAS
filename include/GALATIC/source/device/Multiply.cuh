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
* Multiply.cpp
*
* ac-SpGEMM
*
* Authors: Daniel Mlakar, Markus Steinberger, Martin Winter
*------------------------------------------------------------------------------
*/
#pragma once

#include "memory.cuh"
// Global includes
#include <bitset>
#include <memory>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#ifdef _WIN32
#include <intrin.h>
#define LZCNT __lzcnt
#else
//#include <x86intrin.h>
#define LZCNT __builtin_clzll
#endif

// Local includes
#include "../../include/Multiply.h"
#include "../../include/device/MultiplyKernels.h"
#include "../../include/device/consistent_gpu_memory.h"
#include "../../include/devicetools/stream.h"
#include "../../include/meta_utils.h"
#include "../../include/device/acSpGEMM_DetermineBlockStarts.cuh"
#include "../../include/device/acSpGEMM_SpGEMM.cuh"
#include "../../include/device/acSpGEMM_MergeSimple.cuh"
#include "../../include/device/acSpGEMM_MergeMaxChunks.cuh"
#include "../../include/device/acSpGEMM_MergeGeneralized.cuh"
#include "../../include/device/acSpGEMM_ChunksToCSR.cuh"
#include "../../include/device/HelperFunctions.cuh"
#include "../../include/CustomExceptions.h"


#pragma once

#include "../../include/dCSR.cuh"
#include "../../include/execution_stats.h"
#include "../../include/default_scheduling_traits.h"

void startTimer(cudaEvent_t& start, CUstream stream = 0)
{
	HANDLE_ERROR(cudaEventRecord(start, stream));
}

float recordTimer(cudaEvent_t& start, cudaEvent_t& end, CUstream stream = 0)
{
	float time;
	HANDLE_ERROR(cudaEventRecord(end, stream));
	HANDLE_ERROR(cudaEventSynchronize(end));
	HANDLE_ERROR(cudaEventElapsedTime(&time, start, end));
	return time;
	return 0;
}

using IndexType = uint32_t;
using OffsetType = uint32_t;


namespace ACSpGEMM {

	template<typename T>
	__host__ __forceinline__ T divup(T a, T b)
	{
		return (a + b - 1) / b;
	}

	template<typename T>
	__host__ __forceinline__ T alignment(T size, size_t alignment)
	{
		return divup<T>(size, alignment) * alignment;
	}

	template <typename DataType, uint32_t threads, uint32_t blocks_per_mp, uint32_t nnz_per_thread, uint32_t input_elements_per_thread, uint32_t retain_elements_per_thread, uint32_t merge_max_chunks, uint32_t generalized_merge_max_path_options, uint32_t merge_max_path_options,  bool DEBUG_MODE,
            typename T, typename U, typename Label,
            typename SEMIRING_t>
            void MultiplyImplementation(const dCSR<typename SEMIRING_t::leftInput_t>& matA, const dCSR<typename SEMIRING_t::rightInput_t>& matB, dCSR<typename SEMIRING_t::output_t>& matOut, const GPUMatrixMatrixMultiplyTraits& traits, ExecutionStats& stats, SEMIRING_t semiring)
	{
		using ConsistentGPUMemory = ConsistentMemory<MemorySpace::device>;

		// the magic numbers to make it run smoother
		const float OverallocationFactor = 1.1f;
		const int ChunkPointerOverestimationFactor = 4;
		const float ChunkOverallocationFactor = 1.0f;
		using UintBitSet = std::bitset<sizeof(uint32_t)>;

		if(DEBUG_MODE)
		{
			std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
			std::cout << "THREADS: " << threads << " | NNZPerThread: " << nnz_per_thread << " | InputElementsPerThreads: " << input_elements_per_thread << " | RetainElementsPerThreads: " << retain_elements_per_thread;
			std::cout << " | MaxChunks: " << merge_max_chunks << " | MergePathOptions: " << merge_max_path_options << "| ChunkpointerOverestimationFactor: " << ChunkPointerOverestimationFactor << "\n";
			std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
		}

		// Helper variables
		size_t memory_usage_in_Bytes{ 0 };
		const size_t chunckAllocationsSize{ 256 };
		const size_t numFlags{ 128 };
		const size_t numCounters{ 3 };
		const size_t mergeTypeCounters{ 4 };
		static size_t maxExpectedNNZ{ 500000000 }; //limit allocation...
		static size_t minExpectedNNZ{ 10000000 }; //limit allocation...
								//	  10000000
		static float lastChunckBufferRequirementRatio{ 1.0f };
		const uint32_t nnzperblock{ threads * nnz_per_thread };
		size_t run{ 0 }, chunk_pointer_restart_run{ 0 };
		bool completed{ false };
		bool rowmerging{ false };
		MergeCaseOffsets mergeBlocks;
		uint32_t* currentCounters, *currentChunckAllocation, *currentFlag;
		uint32_t numSharedRows;
		size_t size_to_allocate;
		size_t upper_limit{ 3LL * 1024 * 1024 * 1024 };

		// Kernels
		AcSpGEMMKernels spgemm(threads);

		// Matrix information
		size_t Arows = matA.rows;
		size_t Acols = matA.cols;
		size_t Brows = matB.rows;
		size_t Bcols = matB.cols;
		size_t Crows = Arows;
		size_t Ccols = Bcols;

		 if (Acols != Brows)
		 	throw std::runtime_error("Unable to multiply matrix with matrix - invalid dimensions");

		// Matrix Output estimation
		double a_avg_row = matA.nnz / static_cast<double>(Arows);
		double b_avg_row = matB.nnz / static_cast<double>(Brows);
		double avg_row_overlap = b_avg_row / Bcols;
		// note geometric sequence
		double output_estimate = OverallocationFactor*Arows*b_avg_row * (1.0 - pow(1.0 - avg_row_overlap, a_avg_row)) / (avg_row_overlap);

		// chunks might get created earlier
		double single_chunk_estimate = b_avg_row;
		double current_overlap = avg_row_overlap;
		double merges;
		for (merges = 1; merges < static_cast<size_t>(a_avg_row + 1.0); ++merges)
		{
			if (single_chunk_estimate >= retain_elements_per_thread*threads)
				break;
			single_chunk_estimate += (1 - current_overlap)*b_avg_row;
			current_overlap = current_overlap + (1 - current_overlap)*avg_row_overlap;
		}
		double intermediate_estimate = OverallocationFactor * a_avg_row / std::min(merges, a_avg_row) * single_chunk_estimate * Arows;
		double mergepointer_estimate = std::max(intermediate_estimate, output_estimate) / (retain_elements_per_thread*threads) + 16 * 1024;
		size_t expectedNNZ = std::max(minExpectedNNZ, std::min(maxExpectedNNZ, static_cast<size_t>(lastChunckBufferRequirementRatio*std::max(intermediate_estimate, output_estimate))));
		size_to_allocate = (std::max(sizeof(typename SEMIRING_t::rightInput_t), sizeof(typename SEMIRING_t::output_t))+ sizeof(IndexType))*expectedNNZ*ChunkOverallocationFactor;
		size_t free, total;
		cudaMemGetInfo(&free, &total);
		upper_limit = std::min(upper_limit, free / 3);
		if (size_to_allocate > upper_limit)
			size_to_allocate = upper_limit;
		if(DEBUG_MODE)
		{
			std::cout << "A: " << Arows << "x" << Acols << " NNZ: " << matA.nnz << " avg row: " << a_avg_row << "  " << "B: " << Brows << "x" << Bcols << " NNZ: " << matB.nnz << " avg row: " << b_avg_row << "\n";
			std::cout << "expected row overlap: " << avg_row_overlap << " overallocation: " << OverallocationFactor << "\n";
			std::cout << "expected nnz: " << static_cast<size_t>(round(output_estimate)) << " expected temp: " << static_cast<size_t>(round(intermediate_estimate)) << " mem alloc: " << expectedNNZ << "\n";
			std::cout << "mergepointer alloc " << static_cast<size_t>(ChunkPointerOverestimationFactor*mergepointer_estimate) << " mergepointer estimate: " << mergepointer_estimate << "\n";
		}


		// CUDA variables
		CUstream stream = 0;
		int blockSize = 256;
		int gridSize(divup<int>(Arows + 1, blockSize));
		const int number_merge_streams = 3;
		static CUstream mergeStreams[number_merge_streams];
		for (int i = 0; i < number_merge_streams; ++i)
		{
			if(stats.measure_all)
				mergeStreams[i] = stream;
			else
				cudaStreamCreate(&mergeStreams[i]);
		}

		cudaEvent_t ce_start, ce_stop, individual_start, individual_stop;
		cudaEventCreate(&ce_start); cudaEventCreate(&ce_stop); cudaEventCreate(&individual_start); cudaEventCreate(&individual_stop);

		// GPU Memory Helper structures - general
		static ConsistentGPUMemory chunckPointers;
		static ConsistentGPUMemory combinedGeneralMemory;
		static ConsistentGPUMemory chunk_counter_cptr;
		uint32_t* chunckAllocations{ nullptr };
		uint32_t* blockStarts{ nullptr };
		uint32_t* sharedRowTracker{ nullptr };
		void** outputRowListHead{ nullptr };
		uint32_t* outputRowChunkCounter{ nullptr };
		uint32_t* completion_status{ nullptr };
		uint32_t* chunk_counter{ nullptr };
		void* prefixSumTemp{ nullptr };

		// GPU Memory Helper structures - merge stage allocation
		static ConsistentGPUMemory combineBlockOffsets; // SIZE: combineBlockOffsetsSize * sizeof(IndexType)

		static ConsistentGPUMemory chunk_indices_cptr; // SIZE:  ((mergeBlocks.shared_rows_max_chunks) * merge_max_chunks) * 8
		static ConsistentGPUMemory chunk_values_cptr; // SIZE: ((mergeBlocks.shared_rows_max_chunks) * merge_max_chunks) * 8       
				 //FIXME: RL - This is no longer *8, but sizeof(Either<typename SEMIRING_t::input_t*, typename SEMIRING_t::output_t*>). Probably *16 because alignment. this shoudln't matter?
			   	//FIXME:  till confirmed/tested irrelevant

		static ConsistentGPUMemory chunk_multiplier_cptr; // SIZE: ((mergeBlocks.shared_rows_max_chunks) * merge_max_chunks) * 8

		static ConsistentGPUMemory combinedMergeStageMemory;
		static uint32_t* shared_rows_handled{ nullptr };
		static uint32_t* restart_completion{ nullptr };
		static uint32_t* chunkElementConsumedAndPath{ nullptr };
		uint32_t* num_chunks{ nullptr };
		uint32_t* chunkElementCountDataOffset{ nullptr };
		uint32_t* sample_offset{ nullptr };
		static IndexType** chunk_indices{ nullptr };
		static Either<typename SEMIRING_t::rightInput_t*, typename SEMIRING_t::output_t*>* chunk_values{ nullptr };
		static typename SEMIRING_t::leftInput_t* chunk_multiplier{ nullptr };


		// CPU Memory Helper structures
		static RegisteredMemoryVar<size_t> chunkPointerSize(0);
		static RegisteredMemoryVar<size_t> outputRowInfoSize(0);
		static RegisteredMemoryVar<size_t> prefixSumTempMemSize;
		static RegisteredMemoryVar<size_t> combineBlockOffsetsSize(0);
		static RegisteredMemoryVar<size_t> mergeBlocksAlloc(0);
		static RegisteredMemoryVar<size_t> lastSharedRows(0);
		static RegisteredMemoryVar<size_t> merge_simple_rows(0);
		static RegisteredMemoryVar<size_t> merge_max_chunks_rows(0);
		static RegisteredMemoryVar<size_t> merge_generalized_rows(0);
		uint32_t flagsAndListAllocCounters[numFlags + numCounters];
		size_t tempChunkBufferSizes[256];
		CU::unique_ptr tempChunkBuffers[256];
		tempChunkBufferSizes[0] = alignment(size_to_allocate, 16);
		//
		// TSOPF_RS_b300_c2.mtx shows very weird results if this is done here??
		//
		// Allocate temporary memory for chunks
		tempChunkBuffers[0] = CU::allocMemory(tempChunkBufferSizes[0]);

		cudaDeviceSynchronize();
		// ##############################
		startTimer(ce_start, stream);
		// ##############################
		if(stats.measure_all)
			startTimer(individual_start, stream);


		// Allocate memory for block offsets
		uint32_t requiredBlocks = divup<uint32_t>(matA.nnz, nnzperblock);

		// Allocate memory for chunk and shared row tracker
		if (outputRowInfoSize < Crows)
		{
			//----------------------------------------------------------
			prefixSumTempMemSize = spgemm.tempMemSize<IndexType>(Crows);
			//----------------------------------------------------------
			outputRowInfoSize = Crows;
		}



		// Allocate combined general memory
		size_t combinedGeneralMemory_size =
			/*chunckAllocations*/alignment((chunckAllocationsSize + numFlags + numCounters + mergeTypeCounters) * sizeof(uint32_t), 8) +
			/*blockStarts*/ alignment((requiredBlocks + 2) * sizeof(uint32_t), 8) +
			/*completion_status*/ alignment((requiredBlocks + 2) * sizeof(uint32_t), 8) +
			///*chunk_counter*/ alignment((requiredBlocks + 2) * sizeof(uint32_t), 8) +
			/*outputRowListHead*/ alignment(Crows * sizeof(void*), 8) +
			/*outputRowChunkCounter*/ alignment(Crows * sizeof(uint32_t), 8) +
			/*sharedRowTracker*/ alignment(Crows * sizeof(uint32_t), 8) +
			/*prefixSumTemp*/ alignment(static_cast<size_t>(prefixSumTempMemSize), 8);
		combinedGeneralMemory.assure(combinedGeneralMemory_size);
		memory_usage_in_Bytes += combinedGeneralMemory_size;

		// Place pointers in correct positions
		outputRowListHead = combinedGeneralMemory.get<void*>();
		chunckAllocations = reinterpret_cast<uint32_t*>(outputRowListHead + (alignment(Crows * sizeof(void*), 8) / sizeof(void*)));
		completion_status = chunckAllocations + alignment((chunckAllocationsSize + numFlags + numCounters + mergeTypeCounters) * sizeof(uint32_t), 8) / sizeof(uint32_t);
		/*chunk_counter = completion_status + (alignment((requiredBlocks + 2) * sizeof(uint32_t), 8) / sizeof(uint32_t));*/
		blockStarts = completion_status + (alignment((requiredBlocks + 2) * sizeof(uint32_t), 8) / sizeof(uint32_t));
		outputRowChunkCounter = blockStarts + (alignment((requiredBlocks + 2) * sizeof(uint32_t), 8) / sizeof(uint32_t));
		sharedRowTracker = outputRowChunkCounter + (alignment(Crows * sizeof(uint32_t), 8) / sizeof(uint32_t));
		prefixSumTemp = reinterpret_cast<void*>(sharedRowTracker + (alignment(Crows * sizeof(uint32_t), 8) / sizeof(uint32_t)));

		// TODO: Move back in, currently sometimes produces crashes for whatever reason
		chunk_counter_cptr.assure((requiredBlocks + 2) * sizeof(uint32_t));
		chunk_counter = chunk_counter_cptr.get<uint32_t>();

		// Allocate memory for chunk pointers
		size_t targetChunkPointerSize =ChunkPointerOverestimationFactor*mergepointer_estimate; //fixme : rl
		if (chunkPointerSize < targetChunkPointerSize)
		{
			chunkPointerSize = targetChunkPointerSize;
			chunckPointers.assure((targetChunkPointerSize) * sizeof(void*));
			memory_usage_in_Bytes += (targetChunkPointerSize) * sizeof(void*);
		}

		// Allocate memory for offsets
		CU::unique_ptr newmat_offsets;
		if (matOut.rows != Crows)
		{
			newmat_offsets = CU::allocMemory((Crows + 1) * sizeof(OffsetType));

			memory_usage_in_Bytes += (Crows + 1) * sizeof(OffsetType);
		}
		else
		{
			newmat_offsets.consume(reinterpret_cast<CUdeviceptr>(matOut.row_offsets));
			matOut.row_offsets = nullptr;
		}


		spgemm.setLaunchDimensions(gridSize, stream, blockSize);
		//----------------------------------------------------------
		spgemm.h_DetermineBlockStarts<OffsetType, threads*nnz_per_thread>(
			Arows,
			matA.row_offsets,
			blockStarts,
			reinterpret_cast<uint64_t*>(outputRowListHead),
			outputRowChunkCounter,
			newmat_offsets.get<uint32_t>(),
			requiredBlocks,
			completion_status,
			(chunckAllocationsSize + numFlags + numCounters + mergeTypeCounters),
			chunckAllocations,
			(lastSharedRows),
			shared_rows_handled,
			restart_completion,
			chunk_counter,
			(lastSharedRows) * (generalized_merge_max_path_options + helper_overhead),
			chunkElementConsumedAndPath
			);
		//----------------------------------------------------------
		if(stats.measure_all)
			stats.duration_blockstarts = recordTimer(individual_start, individual_stop, stream);

		do
		{
			currentChunckAllocation = chunckAllocations + (2 * run);
			currentFlag = chunckAllocations + (chunckAllocationsSize + run + chunk_pointer_restart_run);
			currentCounters = chunckAllocations + (chunckAllocationsSize + numFlags);
			if (!rowmerging)
			{
				if(DEBUG_MODE)
				{
					std::cout << "################################################\n";
					std::cout << "Start spgemm stage with " << requiredBlocks<<  " and run: " << run << "\n";
				}
				if(stats.measure_all)
					startTimer(individual_start, stream);

				// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
				// Stage 2 - Compute SpGEMM
				// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
				spgemm.setLaunchDimensions(requiredBlocks, stream, threads);
				if (Arows < 0x10000 && Bcols < 0x10000)
				{
				if(DEBUG_MODE)
				{
					std::cout << "Case 1:\n";
				}
					//we can just use 16bit
					//----------------------------------------------------------
					spgemm.h_computeSpgemmPart<nnz_per_thread, threads, blocks_per_mp, input_elements_per_thread, retain_elements_per_thread, merge_max_path_options, typename SEMIRING_t::leftInput_t, typename SEMIRING_t::rightInput_t, typename SEMIRING_t::output_t, IndexType, OffsetType, 0, T,U,Label, SEMIRING_t>(
						matA.data, matA.col_ids, matA.row_offsets,
						matB.data, matB.col_ids, matB.row_offsets,
						blockStarts, matA.nnz, Arows,
						tempChunkBuffers[run].get<uint32_t>(), currentChunckAllocation, currentChunckAllocation + 1, tempChunkBufferSizes[run],
						chunckPointers.get<void*>(), currentCounters, chunkPointerSize,
						newmat_offsets.get<OffsetType>(), outputRowListHead, outputRowChunkCounter,
						sharedRowTracker, currentCounters + 1, avg_row_overlap, 1.0f / avg_row_overlap,
						currentFlag, completion_status, chunk_counter, currentCounters + 2, semiring);
					//----------------------------------------------------------
				}
				else if (Bcols < (1ull << LZCNT(nnz_per_thread*threads)) - 1)
				{
					if(DEBUG_MODE)
					{
						std::cout << "Case 2:\n";
					}
					//remap every local row to reduce bit count and use remaining for col ids
					//----------------------------------------------------------
                    spgemm.h_computeSpgemmPart<nnz_per_thread, threads, blocks_per_mp, input_elements_per_thread, retain_elements_per_thread, merge_max_path_options, typename SEMIRING_t::leftInput_t, typename SEMIRING_t::rightInput_t, typename SEMIRING_t::output_t, IndexType, OffsetType, true, T,U,Label, SEMIRING_t>(
                            matA.data, matA.col_ids, matA.row_offsets,
                            matB.data, matB.col_ids, matB.row_offsets,
						blockStarts, matA.nnz, Arows,
						tempChunkBuffers[run].get<uint32_t>(), currentChunckAllocation, currentChunckAllocation + 1, tempChunkBufferSizes[run],
						chunckPointers.get<void*>(), currentCounters, chunkPointerSize,
						newmat_offsets.get<OffsetType>(), outputRowListHead, outputRowChunkCounter,
						sharedRowTracker, currentCounters + 1, avg_row_overlap, 1.0f / avg_row_overlap,
						currentFlag, completion_status, chunk_counter, currentCounters + 2, semiring);
					//----------------------------------------------------------
				}
				else
				{
					if(DEBUG_MODE)
					{
						std::cout << "Case 3:\n";
					}
					//----------------------------------------------------------
					spgemm.h_computeSpgemmPart<nnz_per_thread, threads, blocks_per_mp, input_elements_per_thread, retain_elements_per_thread, merge_max_path_options, typename SEMIRING_t::leftInput_t, typename SEMIRING_t::rightInput_t, typename SEMIRING_t::output_t, IndexType, OffsetType, 2,T,U,Label, SEMIRING_t>(
						matA.data, matA.col_ids, matA.row_offsets,
						matB.data, matB.col_ids, matB.row_offsets,
						blockStarts, matA.nnz, Arows,
						tempChunkBuffers[run].get<uint32_t>(), currentChunckAllocation, currentChunckAllocation + 1, tempChunkBufferSizes[run],
						chunckPointers.get<void*>(), currentCounters, chunkPointerSize,
						newmat_offsets.get<OffsetType>(), outputRowListHead, outputRowChunkCounter,
						sharedRowTracker, currentCounters + 1, avg_row_overlap, 1.0f / avg_row_overlap,
						currentFlag, completion_status, chunk_counter, currentCounters + 2,semiring);
					//----------------------------------------------------------
				}
				// if (cudaDeviceSynchronize() != cudaSuccess) {
				// 	throw SpGEMMException();
				// }
				if(stats.measure_all)
					stats.duration_spgemm += recordTimer(individual_start, individual_stop, stream);
			}
			else
			{
		
				if(DEBUG_MODE)
				{
					std::cout << "################################################\n";
					std::cout << "Start Merge Stage\n";
				}
				uint32_t simple_restart_offset = 0;
				uint32_t max_chunks_restart_offset = mergeBlocks.shared_rows_simple;
				uint32_t generalized_restart_offset = mergeBlocks.shared_rows_simple + mergeBlocks.shared_rows_max_chunks;
				// Simple Case -> Output fits in shared
				if (mergeBlocks.shared_rows_simple)
				{
					if(stats.measure_all)
						startTimer(individual_start, mergeStreams[0]);

					spgemm.setLaunchDimensions(mergeBlocks.shared_rows_simple, mergeStreams[0], threads);
					if (Bcols < 1ull << LZCNT(threads - 1))
					{
						if (DEBUG_MODE)
						{
							std::cout << "Case: 1\n";
						}
						//----------------------------------------------------------
						spgemm.h_mergeSharedRowsSimple< nnz_per_thread, threads, blocks_per_mp, input_elements_per_thread, retain_elements_per_thread, merge_max_chunks, merge_max_path_options, typename SEMIRING_t::output_t, IndexType, OffsetType, false,T,U,Label, SEMIRING_t>(
							combineBlockOffsets.get<uint32_t>() + (3 * numSharedRows), combineBlockOffsets.get<uint32_t>(), outputRowListHead,
							newmat_offsets.get<OffsetType>(),
							tempChunkBuffers[run].get<uint32_t>(), currentChunckAllocation, NULL, tempChunkBufferSizes[run],
							chunckPointers.get<void*>(), currentCounters, chunkPointerSize,
							currentFlag, restart_completion, shared_rows_handled, simple_restart_offset, currentCounters + 2, semiring
							);
						//----------------------------------------------------------
					}
					else
					{
						if (DEBUG_MODE)
						{
							std::cout << "Case: 2\n";
						}
						//----------------------------------------------------------
						spgemm.h_mergeSharedRowsSimple< nnz_per_thread, threads, blocks_per_mp, input_elements_per_thread, retain_elements_per_thread, merge_max_chunks, merge_max_path_options, typename SEMIRING_t::output_t, IndexType, OffsetType, true,T,U,Label, SEMIRING_t>(
							combineBlockOffsets.get<uint32_t>() + (3 * numSharedRows), combineBlockOffsets.get<uint32_t>(), outputRowListHead,
							newmat_offsets.get<OffsetType>(),
							tempChunkBuffers[run].get<uint32_t>(), currentChunckAllocation, NULL, tempChunkBufferSizes[run],
							chunckPointers.get<void*>(), currentCounters, chunkPointerSize,
							currentFlag, restart_completion, shared_rows_handled, simple_restart_offset, currentCounters + 2,semiring
							);
						//----------------------------------------------------------
					}
					// if (cudaDeviceSynchronize() != cudaSuccess) {
					// 	throw MergeSimpleCaseException();
					// }
					if(stats.measure_all)
						stats.duration_merge_simple += recordTimer(individual_start, individual_stop, mergeStreams[0]);
				}

				// Complex Case -> Output gets merged through paths over MAX_CHUNKS
				if (mergeBlocks.shared_rows_max_chunks)
				{
					if (DEBUG_MODE)
					{
						std::cout << "Case: 4\n";
					}
					if(stats.measure_all)
						startTimer(individual_start, mergeStreams[1]);
					spgemm.setLaunchDimensions(mergeBlocks.shared_rows_max_chunks, mergeStreams[1], threads);
					//----------------------------------------------------------
					spgemm.h_mergeSharedRowsMaxChunks<nnz_per_thread, threads, blocks_per_mp, input_elements_per_thread, retain_elements_per_thread, merge_max_chunks, merge_max_path_options, typename SEMIRING_t::leftInput_t, IndexType, OffsetType,
                    typename SEMIRING_t::leftInput_t, typename SEMIRING_t::rightInput_t,Label,SEMIRING_t> (
						NULL, combineBlockOffsets.get<uint32_t>() + (1 * numSharedRows), outputRowListHead,
						newmat_offsets.get<OffsetType>(),
						tempChunkBuffers[run].get<uint32_t>(), currentChunckAllocation, NULL, tempChunkBufferSizes[run],
						chunckPointers.get<void*>(), currentCounters, chunkPointerSize,
						currentFlag, restart_completion, shared_rows_handled,
						chunk_indices, chunk_values, chunk_multiplier,
						chunkElementCountDataOffset, max_chunks_restart_offset, num_chunks, currentCounters + 2, semiring);
					//----------------------------------------------------------
					// if (cudaDeviceSynchronize() != cudaSuccess) {
					// 	throw MergeMaxChunksCaseException();
					// }
					if(stats.measure_all)
						stats.duration_merge_max += recordTimer(individual_start, individual_stop, mergeStreams[1]);
				}

				// General Case -> Handles cases with more than MAX_CHUNKS chunks
				if (mergeBlocks.shared_rows_generalized)
				{
					if (DEBUG_MODE)
					{
						std::cout << "Case: 5\n";
					}
					if(stats.measure_all)
						startTimer(individual_start, mergeStreams[2]);
					spgemm.setLaunchDimensions(mergeBlocks.shared_rows_generalized, mergeStreams[2], threads);
					//----------------------------------------------------------
					spgemm.h_mergeSharedRowsGeneralized<nnz_per_thread, threads, blocks_per_mp, input_elements_per_thread, retain_elements_per_thread,
					generalized_merge_max_path_options, merge_max_path_options, typename SEMIRING_t::leftInput_t, IndexType, OffsetType,
					  T,U,Label,SEMIRING_t>(
						NULL, combineBlockOffsets.get<uint32_t>() + (2 * numSharedRows), outputRowListHead,
						newmat_offsets.get<OffsetType>(),
						tempChunkBuffers[run].get<uint32_t>(), currentChunckAllocation, NULL, tempChunkBufferSizes[run],
						chunckPointers.get<void*>(), currentCounters, chunkPointerSize,
						currentFlag, restart_completion, shared_rows_handled,
						sample_offset, chunkElementConsumedAndPath, generalized_restart_offset, currentCounters + 2,
						semiring
						);
					//----------------------------------------------------------
					// if (cudaDeviceSynchronize() != cudaSuccess) {
					// 	throw MergeGeneralizedCaseException();
					// }
					if(stats.measure_all)
						stats.duration_merge_generalized += recordTimer(individual_start, individual_stop, mergeStreams[2]);
				}
			}

			// // Copy back flags
			HANDLE_ERROR(cudaMemcpy(&flagsAndListAllocCounters[0], chunckAllocations + chunckAllocationsSize, (numFlags + numCounters) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
			completed = flagsAndListAllocCounters[run + chunk_pointer_restart_run] == 0;

			if (!completed)
			{
				// if (stats.measure_all && stats.duration_merge_simple + stats.duration_merge_max + stats.duration_merge_generalized > 10000)
				// 	throw MergeLoopingException();


				uint32_t return_value = flagsAndListAllocCounters[run + chunk_pointer_restart_run];
				if (UintBitSet(return_value).test(0))
				{
					if (DEBUG_MODE)
					{
						std::cout << "Chunk Memory Restart allocating space for " << tempChunkBufferSizes[run] / (sizeof(typename SEMIRING_t::rightInput_t) + sizeof(IndexType)) << " elements\n";
					}
					// Get more chunk memory
					auto new_buffer_size = tempChunkBufferSizes[run];
					tempChunkBufferSizes[run+1] = new_buffer_size;
					tempChunkBuffers[run+1] = CU::allocMemory(new_buffer_size);
					if (++run == chunckAllocationsSize / 2) {
						std::cout << "Out of memory " << std::endl; 
						throw RestartOutOfMemoryException();
					}
				}
				if (UintBitSet(return_value).test(1))
				{
					if (DEBUG_MODE)
					{
						std::cout << "Chunk Pointer Restart allocating " << targetChunkPointerSize << " new pointers\n";
					}
					// Get more chunk pointers
					chunkPointerSize += targetChunkPointerSize;
					chunckPointers.increaseMemRetainData((targetChunkPointerSize) * 8);
					targetChunkPointerSize *= 2;
					if (++chunk_pointer_restart_run == chunckAllocationsSize / 2)
						throw RestartOutOfChunkPointerException();
					HANDLE_ERROR(cudaMemcpy(currentCounters, currentCounters + 2, sizeof(uint32_t), cudaMemcpyDeviceToDevice));
				}
			}
			if (completed && !rowmerging)
			{
				numSharedRows = flagsAndListAllocCounters[numFlags + 1];
				if (numSharedRows > 0)
				{
					if(stats.measure_all)
						startTimer(individual_start, stream);

					if (combineBlockOffsetsSize < 4 * (numSharedRows + 1))
					{
						combineBlockOffsetsSize = 4 * (numSharedRows + 1024);
						combineBlockOffsets.assure(combineBlockOffsetsSize * sizeof(IndexType));
						memory_usage_in_Bytes += combineBlockOffsetsSize * sizeof(IndexType);
					}
					CUdeviceptr mergeTypeCounters = reinterpret_cast<CUdeviceptr>(chunckAllocations) + 4 * (chunckAllocationsSize + numFlags + numCounters);

					//----------------------------------------------------------
					mergeBlocks = spgemm.assignCombineBlocks<IndexType, merge_max_chunks, 2 * threads * input_elements_per_thread, threads>(numSharedRows, prefixSumTemp, prefixSumTempMemSize, sharedRowTracker, newmat_offsets, outputRowChunkCounter, combineBlockOffsets, mergeTypeCounters, stream);
					//----------------------------------------------------------

					completed = false;
					rowmerging = true;

					if(DEBUG_MODE)
					{
						std::cout << "################################################\n";
						std::cout << "Assigned " << numSharedRows << " shared rows to blocks, starting \n\t\t"
							<< mergeBlocks.shared_rows_simple << " simple merges for " << mergeBlocks.shared_rows_simple_rows << " rows,\n\t\t"
							<< mergeBlocks.shared_rows_max_chunks << " max chunk mergers, and\n\t\t"
							<< mergeBlocks.shared_rows_generalized << " general mergers\n";
					}

					// Set merge stage row stats
					stats.shared_rows = numSharedRows;
					stats.simple_mergers = mergeBlocks.shared_rows_simple;
					stats.simple_rows = mergeBlocks.shared_rows_simple_rows;
					stats.complex_rows = mergeBlocks.shared_rows_max_chunks;
					stats.generalized_rows = mergeBlocks.shared_rows_generalized;
					merge_simple_rows = mergeBlocks.shared_rows_simple;
					merge_max_chunks_rows = mergeBlocks.shared_rows_max_chunks;
					merge_generalized_rows = mergeBlocks.shared_rows_generalized;

					// Allocate memory for all helper data structures
					size_t combinedMergeStageMemory_size =
						/*shared_rows_handled*/((numSharedRows) * sizeof(uint32_t)) +
						/*restart_completion*/((numSharedRows) * sizeof(uint32_t)) +
						/*chunkElementConsumedAndPath*/((numSharedRows) * (generalized_merge_max_path_options + helper_overhead) * sizeof(uint32_t)) +
						/*chunkElementCountDataOffset*/(((numSharedRows) * merge_max_chunks) * sizeof(uint32_t)) +
						/*num_chunks*/((numSharedRows) * sizeof(uint32_t)) +
						/*sample_offset*/(((numSharedRows) * (threads) * sizeof(uint32_t))); //+
						///* chunk_indices*/(((mergeBlocks.shared_rows_max_chunks) * merge_max_chunks) * sizeof(IndexType*)) +
						///*chunk_values*/(((mergeBlocks.shared_rows_max_chunks) * merge_max_chunks) * sizeof(typename SEMIRING_t::input_t*)) +
						///*chunk_multiplier*/(((mergeBlocks.shared_rows_max_chunks) * merge_max_chunks) * sizeof(typename SEMIRING_t::input_t));
					combinedMergeStageMemory.assure(combinedMergeStageMemory_size);
					memory_usage_in_Bytes += combinedMergeStageMemory_size;

					//// Place pointers in memory allocation
					shared_rows_handled = combinedMergeStageMemory.get<uint32_t>();
					restart_completion = shared_rows_handled + (numSharedRows);
					chunkElementConsumedAndPath = restart_completion + (numSharedRows);
					chunkElementCountDataOffset = chunkElementConsumedAndPath + (numSharedRows) * (generalized_merge_max_path_options + helper_overhead);
					num_chunks = chunkElementCountDataOffset + ((numSharedRows) * merge_max_chunks);
					sample_offset = num_chunks + (numSharedRows);

					// TODO: Why does this work??????????????????????????
					chunk_indices_cptr.assure(((mergeBlocks.shared_rows_max_chunks) * merge_max_chunks) * sizeof(IndexType*));
					chunk_indices = chunk_indices_cptr.get<IndexType*>();
					chunk_values_cptr.assure(((mergeBlocks.shared_rows_max_chunks) * merge_max_chunks) * sizeof( Either<typename SEMIRING_t::rightInput_t*, typename SEMIRING_t::output_t*>));
					chunk_values = chunk_values_cptr.get< Either<typename SEMIRING_t::rightInput_t*, typename SEMIRING_t::output_t*>>();
					chunk_multiplier_cptr.assure(((mergeBlocks.shared_rows_max_chunks) * merge_max_chunks) * sizeof(typename SEMIRING_t::leftInput_t));
					chunk_multiplier = chunk_multiplier_cptr.get<typename SEMIRING_t::leftInput_t>();


					// TODO: Why does this NOT work??????????????????????????
					/*chunk_indices = reinterpret_cast<IndexType**>(chunk_multiplier + ((mergeBlocks.shared_rows_max_chunks) * merge_max_chunks));*/
					/*chunk_values = reinterpret_cast<typename SEMIRING_t::input_t**>(chunk_indices + ((mergeBlocks.shared_rows_max_chunks) * merge_max_chunks));*/
					// chunk_multiplier = reinterpret_cast<typename SEMIRING_t::input_t*>(sample_offset + ((numSharedRows) * (threads)));

					memory_usage_in_Bytes += ((mergeBlocks.shared_rows_max_chunks) * merge_max_chunks) * sizeof(IndexType*);
					memory_usage_in_Bytes += ((mergeBlocks.shared_rows_max_chunks) * merge_max_chunks) * sizeof(Either<typename SEMIRING_t::rightInput_t*, typename SEMIRING_t::output_t*>);
					memory_usage_in_Bytes += ((mergeBlocks.shared_rows_max_chunks) * merge_max_chunks) * sizeof(typename SEMIRING_t::rightInput_t);

					if (numSharedRows > lastSharedRows)
					{
						cudaMemset(combinedMergeStageMemory.get(), 0,
							/*chunkElementConsumedAndPath*/((numSharedRows) * (generalized_merge_max_path_options + helper_overhead) * sizeof(uint32_t)) +
							/*shared_rows_handled*/((numSharedRows) * sizeof(uint32_t)) +
							/*restart_completion*/((numSharedRows) * sizeof(uint32_t))
						);
						lastSharedRows = numSharedRows;
					}
					if(stats.measure_all)
						stats.duration_merge_case_computation = recordTimer(individual_start, individual_stop, stream);
				}
			}
		} while (!completed);

		// Let's write the chunks out to a csr matrix
		if(stats.measure_all)
			startTimer(individual_start, stream);

		//----------------------------------------------------------
		spgemm.computeRowOffsets<IndexType>(Crows, prefixSumTemp, prefixSumTempMemSize, newmat_offsets, stream);
		//----------------------------------------------------------

		// Allocate output matrix
		IndexType matrix_elements;
		CUdeviceptr offs = newmat_offsets;
		offs += sizeof(IndexType) * Crows;
		HANDLE_ERROR(cudaMemcpy(&matrix_elements, reinterpret_cast<void*>(offs), sizeof(IndexType), cudaMemcpyDeviceToHost));

		if (matOut.nnz != matrix_elements)
		{
			//std::cout << "Reallocation HERE ################" << matOut.nnz << " | " << matrix_elements <<"\n";
			matOut.alloc(Crows, Ccols, matrix_elements, true);
		}
		matOut.row_offsets = std::move(newmat_offsets.getRelease<IndexType>());

		//----------------------------------------------------------
		spgemm.h_copyChunks<typename SEMIRING_t::output_t, IndexType, OffsetType>(chunckPointers.get<void*>(), currentCounters,
			matOut.data, matOut.col_ids, matOut.row_offsets);
		//----------------------------------------------------------
		if(stats.measure_all)
			stats.duration_write_csr = recordTimer(individual_start, individual_stop, stream);

		if (stats.measure_all)
		{
			stats.mem_allocated_chunks = tempChunkBufferSizes[0] * (run + 1);
			uint32_t* d_current_chunk_allocation = chunckAllocations + (2 * run);
			uint32_t h_current_chunk_allocation = 0;
			HANDLE_ERROR(cudaMemcpy(&h_current_chunk_allocation, d_current_chunk_allocation, sizeof(uint32_t), cudaMemcpyDeviceToHost));
			stats.mem_used_chunks = tempChunkBufferSizes[0] * run + h_current_chunk_allocation;
		}
		stats.restarts = run + chunk_pointer_restart_run;

		// ##############################
		stats.duration = recordTimer(ce_start, ce_stop, stream);
		// ##############################

		// Stream cleanup
		if (!(stats.measure_all))
		{
			for (int i = 0; i < number_merge_streams; ++i)
				cudaStreamDestroy(mergeStreams[i]);
		}

		return;
	}

	template<class CB, int... OPTIONS>
	struct Selection
	{
		CB& cb;
		Selection(CB& cb) : cb(cb) {}
	};

	template<class CB, int... OPTIONS>
	struct CallSelection
	{
		static void call(CB &cb)
		{
			cb. template call<OPTIONS...>();
		}
	};

	struct EnumFin
	{
		template<class CB, int... OPTIONS>
		static bool call(Selection<CB, OPTIONS...> cb)
		{
			CallSelection<CB, OPTIONS...>::call(cb.cb);
			return true;
		}
	};

	template<int CURRENT, int MAX, int STEP, class NEXT = EnumFin>
	struct EnumOption
	{
		template<class CB,int... OPTIONS, class... TYPES>
		static bool call(Selection<CB, OPTIONS...> cb, int value, TYPES... values)
		{
			if (value == CURRENT)
			{
				return NEXT::call(Selection<CB, OPTIONS..., CURRENT>(cb.cb), values...);
			}
			else
				return EnumOption<CURRENT + STEP, MAX, STEP, NEXT>::call(cb, value, values...);
		}
	};

	template<int MAX, int STEP, class NEXT>
	struct EnumOption<MAX, MAX, STEP, NEXT>
	{
		template<class CB, int... OPTIONS, class... TYPES>
		static bool call(Selection<CB, OPTIONS...> cb, int value, TYPES... values)
		{
			if (value == MAX)
			{
				return NEXT::call(Selection<CB, OPTIONS..., MAX>(cb.cb), values...);
			}
			else
				return false;
		}
	};


	template<typename DataType ,  typename T, typename U, typename Label,
            typename SEMIRING_t>
            struct MultiplyCall
	{
		const dCSR<typename SEMIRING_t::leftInput_t>& A;
		const dCSR<typename SEMIRING_t::rightInput_t>& B;
		dCSR<typename SEMIRING_t::output_t> &matOut;
		SEMIRING_t semiring;

		const GPUMatrixMatrixMultiplyTraits& scheduling_traits;
		ExecutionStats& exec_stats;

		MultiplyCall(const dCSR<typename SEMIRING_t::leftInput_t>& A, const dCSR<typename SEMIRING_t::rightInput_t>& B, dCSR<typename SEMIRING_t::output_t>& matOut, const GPUMatrixMatrixMultiplyTraits& scheduling_traits, ExecutionStats& exec_stats, SEMIRING_t semiring) :
			A(A), B(B), matOut(matOut), scheduling_traits(scheduling_traits), exec_stats(exec_stats), semiring(semiring)
		{

		}

		template<int Threads, int BlocksPerMP, int NNZPerThread, int InputPerThread, int RetainElements, int MaxChunkstoMerge, int MaxChunksGeneralizedMerge, int MergePathOptions, int Debug>
		void call()
		{
			const int RealBlocksPerMP = (256 * BlocksPerMP + Threads - 1) / Threads;
			ACSpGEMM::MultiplyImplementation<typename SEMIRING_t::leftInput_t, Threads, RealBlocksPerMP, NNZPerThread, InputPerThread, RetainElements, MaxChunkstoMerge, MaxChunksGeneralizedMerge, MergePathOptions, Debug == 0?false:true, T,U,Label, SEMIRING_t>(A, B, matOut, scheduling_traits, exec_stats,semiring);
		}
	};


	template < typename SEMIRING_t>
	        void Multiply(const dCSR<typename SEMIRING_t::leftInput_t>& A, const dCSR<typename SEMIRING_t::rightInput_t>& B, dCSR<typename SEMIRING_t::output_t>& matOut, const GPUMatrixMatrixMultiplyTraits& scheduling_traits, ExecutionStats& exec_stats, bool DEBUG_MODE, SEMIRING_t semiring)
	{
		MultiplyCall<typename SEMIRING_t::leftInput_t,typename SEMIRING_t::rightInput_t,typename SEMIRING_t::output_t,typename SEMIRING_t::output_t, SEMIRING_t> call(A, B, matOut, scheduling_traits, exec_stats, semiring);

	
	bool called = EnumOption<128, 128, 256,
	EnumOption<1, 1, 1,
	EnumOption<2, 2,1,
	EnumOption<2, 2, 2,
	EnumOption<1, 1, 1,
	EnumOption<16, 16, 8,
	EnumOption<256, 256, 128,
	EnumOption<8, 8, 8,
	EnumOption<0, 1, 1>>>>>>>>>
			::call(Selection<MultiplyCall<typename SEMIRING_t::leftInput_t, typename SEMIRING_t::rightInput_t, typename SEMIRING_t::output_t, typename SEMIRING_t::output_t, SEMIRING_t>>(call), scheduling_traits.Threads, scheduling_traits.BlocksPerMp, scheduling_traits.NNZPerThread, scheduling_traits.InputElementsPerThreads, scheduling_traits.RetainElementsPerThreads, scheduling_traits.MaxChunksToMerge, scheduling_traits.MaxChunksGeneralizedMerge, scheduling_traits.MergePathOptions, (int)DEBUG_MODE);
		if(!called)
		{
			std::cout << "Configuration not instantiated!\n";
		}
	};
}
 
