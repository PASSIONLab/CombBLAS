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
 * MultiplyKernels.h
 *
 * ac-SpGEMM
 *
 * Authors: Daniel Mlakar, Markus Steinberger, Martin Winter
 *------------------------------------------------------------------------------
*/

#pragma once

#include "../SemiRingInterface.h"
#include <stdint.h>
#include <tuple>
#include <cuda.h>
#include "../MergeCaseOffsets.h"

const int RESTART_OFF = 0;
const int RESTART_WRONG_CASE = 1;
const int RESTART_FIRST_ITERATION = 2;
const int RESTART_ITERATION_FINISH = 3;
const int RESTART_ITERATION_UNKNOWN =  4;
const int helper_overhead = 4;
#define WARP_SIZE 32
#define MAX_CHUNKS_CASE 0x80000000
#define GENERALIZED_CASE 0xC0000000
#define CASE_DISTINCTION 0x40000000 // MAX_CHUNKS_CASE - GENERALIZED_CASE

// Debugging
#define ROW_TO_INVESTIGATE 2579

#define ENABLE_SORTING



//###################################################
// Tagged unions  /  "enums"

template<typename T, typename U>
struct Either {
	union Data {
		T tee;
		U you;
	};

	Data data;
	unsigned char tag;

	__device__ __host__ bool isFirst() const {
		return tag == 0;
	}

	__device__ __host__ bool isSecond() const {
		return tag == 1;
	}

	__device__ __host__ const T& valFirst() const {
		return data.tee;
	}
	__device__ __host__ const U& valSecond() const  {
		return data.you;
	}

	static  __device__ __host__ Either First(T te) {
		Either result;
		result.data.tee = te;
		result.tag = 0;
		return result;
	}
	static __device__ __host__ Either Second(U u) {
		Either result;
		result.data.you = u;
		result.tag = 1;
		return result;
	}
	__device__ __host__ Either () {}
};





class AcSpGEMMKernels
{
public:
	AcSpGEMMKernels(uint32_t blockDim=128):
	blockDim{blockDim}
	{}

	void setLaunchDimensions(uint32_t _gridDim, cudaStream_t _stream = 0, uint32_t _blockDim = 128)
	{
		gridDim = _gridDim;
		blockDim = _blockDim;
		stream = _stream;
	}

	// #####################################################################
	// Determine Block Starts
	//
	template<typename OFFSET_TYPE, uint32_t NNZ_PER_BLOCK>
	void h_DetermineBlockStarts(int num_other, const uint32_t*__restrict offsets, uint32_t* startingIds, uint64_t* toClear, 
	uint32_t* toClear1, uint32_t* toClear2, int num3, uint32_t* toClear3, int num4, uint32_t* toClear4, 
		int num5, uint32_t* toClear5, uint32_t* toClear6, uint32_t* toClear7, int num8, uint32_t* toClear8);

	// #####################################################################
	// SpGEMM stage
	//
	template<uint32_t NNZ_PER_THREAD, uint32_t THREADS, uint32_t BLOCKS_PER_MP, uint32_t INPUT_ELEMENTS_PER_THREAD, uint32_t RETAIN_ELEMENTS_PER_THREAD, uint32_t MERGE_MAX_PATH_OPTIONS, typename VALUE_TYPE1, typename VALUE_TYPE2, typename VALUE_TYPE3, typename INDEX_TYPE, typename OFFSET_TYPE, int SORT_TYPE_MODE,
            typename T, typename U, typename Label,
            typename SEMIRING_t>
            void h_computeSpgemmPart(
	const typename SEMIRING_t::leftInput_t* valA, const INDEX_TYPE* indicesA, const OFFSET_TYPE* __restrict offsetsA,
	/*fixme const T2 -> */const typename SEMIRING_t::rightInput_t* __restrict valB, const INDEX_TYPE* __restrict indicesB, const OFFSET_TYPE* __restrict offsetsB,
	const uint32_t* __restrict startingIdsA, uint32_t nnz, uint32_t rows,
	uint32_t* chunks, uint32_t* chunk_alloc, uint32_t* chunk_worst_case, uint32_t chunk_size,
	void** chunks_pointers, uint32_t* chunk_pointer_alloc, uint32_t chunk_pointer_sizes,
	OFFSET_TYPE* output_row_count, void** output_row_list_heads, uint32_t* output_row_chunk_count,
	uint32_t* shared_rows_tracker, uint32_t* shared_rows_alloc, float expected_row_overlap, float expected_row_overlap_inv,
	uint32_t* run_flag, uint32_t* completion_status, uint32_t* chunk_counter, uint32_t* chunk_pointer_pos, SEMIRING_t semiring);
	// #####################################################################
	// Merge Chunks Simple
	//
	template<uint32_t NNZ_PER_THREAD, uint32_t THREADS, uint32_t BLOCKS_PER_MP, uint32_t INPUT_ELEMENTS_PER_THREAD, uint32_t RETAIN_ELEMENTS_PER_THREAD, uint32_t MERGE_MAX_CHUNKS, uint32_t MERGE_MAX_PATH_OPTIONS, typename VALUE_TYPE, typename INDEX_TYPE, typename OFFSET_TYPE, bool LONG_SORT,
            typename T, typename U, typename Label,
            typename SEMIRING_t>
            void h_mergeSharedRowsSimple(const uint32_t* __restrict blockOffsets, const uint32_t* __restrict sharedRows, void** output_row_list_heads,
	OFFSET_TYPE* output_row_count,
	uint32_t* chunks, uint32_t* chunk_alloc, uint32_t* chunk_pre_alloc, uint32_t chunk_size,
	void** chunks_pointers, uint32_t* chunk_pointer_alloc, uint32_t chunk_pointer_sizes,
	uint32_t* run_flag, uint32_t* restart_completion, uint32_t* shared_rows_handled, uint32_t restart_offset, uint32_t* chunk_pointer_pos, SEMIRING_t semiring);

	// #####################################################################
	// Merge Chunks Max Chunks
	//
	template<uint32_t NNZ_PER_THREAD, uint32_t THREADS, uint32_t BLOCKS_PER_MP, uint32_t INPUT_ELEMENTS_PER_THREAD,
	        uint32_t RETAIN_ELEMENTS_PER_THREAD, uint32_t MERGE_MAX_CHUNKS, uint32_t MERGE_MAX_PATH_OPTIONS, typename VALUE_TYPE,
	        typename INDEX_TYPE, typename OFFSET_TYPE,
	        typename T, typename U, typename Label,
	        typename SEMIRING_t>
	        void h_mergeSharedRowsMaxChunks(const uint32_t* __restrict blockOffsets, const uint32_t* __restrict sharedRows, void** output_row_list_heads,
	OFFSET_TYPE* output_row_count, uint32_t* chunks, uint32_t* chunk_alloc, uint32_t* chunk_pre_alloc, uint32_t chunk_size,
	void** chunks_pointers, uint32_t* chunk_pointer_alloc, uint32_t chunk_pointer_sizes,
	uint32_t* run_flag, uint32_t* restart_completion, uint32_t* shared_rows_handled,
	INDEX_TYPE** restart_chunkIndices, Either<typename SEMIRING_t::rightInput_t*, typename SEMIRING_t::output_t*>* restart_chunkValues, typename SEMIRING_t::leftInput_t* restart_multiplier, uint32_t* restart_chunkElementCount, uint32_t restart_offset, uint32_t* restart_num_chunks, uint32_t* chunk_pointer_pos, SEMIRING_t semiring);

	// #####################################################################
	// Merge Chunks Generalized
	//
	template<uint32_t NNZ_PER_THREAD, uint32_t THREADS, uint32_t BLOCKS_PER_MP, uint32_t INPUT_ELEMENTS_PER_THREAD, uint32_t RETAIN_ELEMENTS_PER_THREAD, uint32_t MERGE_MAX_CHUNKS, uint32_t MERGE_MAX_PATH_OPTIONS, typename VALUE_TYPE, typename INDEX_TYPE, typename OFFSET_TYPE,
            typename T, typename U, typename Label,
            typename SEMIRING_t>
	void h_mergeSharedRowsGeneralized(const uint32_t* __restrict blockOffsets, const uint32_t* __restrict sharedRows, void** output_row_list_heads,
	OFFSET_TYPE* output_row_count,
	uint32_t* chunks, uint32_t* chunk_alloc, uint32_t* chunk_pre_alloc, uint32_t chunk_size,
	void** chunks_pointers, uint32_t* chunk_pointer_alloc, uint32_t chunk_pointer_sizes,
	uint32_t* run_flag, uint32_t* restart_completion, uint32_t* shared_rows_handled,
	uint32_t* restart_sampleOffs, uint32_t* restart_chunkElementsConsumedAndPath, uint32_t restart_offset, uint32_t* chunk_pointer_pos, SEMIRING_t semiring);

	// #####################################################################
	// Copy Chunks into CSR format
	//
	template< typename VALUE_TYPE, typename INDEX_TYPE, typename OFFSET_TYPE>
	void h_copyChunks(void* const* __restrict chunks_pointers, const uint32_t* __restrict chunk_pointer_alloc, 
	VALUE_TYPE * value_out, INDEX_TYPE * index_out, const uint32_t* __restrict result_offets);

	// #####################################################################
	// Calculate temporary memory size
	//
	template<class INDEX_TYPE>
	size_t tempMemSize(size_t CRows);

	// #####################################################################
	// Merge Case assignment
	//
	 template<class INDEX_TYPE, INDEX_TYPE MaxMergeChunks, INDEX_TYPE MergeMaxElements, uint32_t SimpleMergeBlockSize>
	 MergeCaseOffsets assignCombineBlocks(size_t activeRows, void* tempMem, size_t tempMemSize, uint32_t* sharedRows, CUdeviceptr maxPerRowElements, uint32_t* chunckCounter, CUdeviceptr per_block_offsets, CUdeviceptr num_merge_blocks, CUstream stream = 0, CUstream overlapStream = 0);

	 // #####################################################################
	 // Compute CSR offsets
	 //
	 template<class INDEX_TYPE>
	 void computeRowOffsets(size_t Crows, void* tempMem, size_t tempMemSize, CUdeviceptr inout, CUstream stream = 0);

	
private:
	uint32_t blockDim;
	uint32_t gridDim;
	cudaStream_t stream;
};

