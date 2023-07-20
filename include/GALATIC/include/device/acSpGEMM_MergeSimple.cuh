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
 * MergeSimple.cuh
 *
 * ac-SpGEMM
 *
 * Authors: Daniel Mlakar, Markus Steinberger, Martin Winter
 *------------------------------------------------------------------------------
*/

#pragma once

#include "MultiplyKernels.h"

// #########################################################################################
//
//  Simple Case
//
// #########################################################################################
template<uint32_t NNZ_PER_THREAD, uint32_t THREADS, uint32_t BLOCKS_PER_MP, uint32_t INPUT_ELEMENTS_PER_THREAD, uint32_t RETAIN_ELEMENTS_PER_THREAD, uint32_t MERGE_MAX_CHUNKS, uint32_t MERGE_MAX_PATH_OPTIONS, typename VALUE_TYPE, typename INDEX_TYPE, typename OFFSET_TYPE, bool LONG_SORT,
        typename T, typename U, typename Label,
        typename SEMIRING_t>
__global__ void __launch_bounds__(THREADS, BLOCKS_PER_MP)
mergeSharedRowsSimple(const uint32_t* __restrict blockOffsets, const uint32_t* __restrict sharedRows, void** output_row_list_heads,
	OFFSET_TYPE* output_row_count,
	uint32_t* chunks, uint32_t* chunk_alloc, uint32_t* chunk_pre_alloc, uint32_t chunk_size,
	void** chunks_pointers, uint32_t* chunk_pointer_alloc, uint32_t chunk_pointer_sizes,
	uint32_t* run_flag, uint32_t* restart_completion, uint32_t* shared_rows_handled, uint32_t restart_offset, uint32_t* chunk_pointer_pos, SEMIRING_t semiring)
{
	using Chunk = ::Chunk<typename SEMIRING_t::output_t, INDEX_TYPE>;
	const uint32_t ELEMENTS_PER_THREAD = 2 * INPUT_ELEMENTS_PER_THREAD;
	using SortType = ChooseBitDataType<LONG_SORT ? 64 : 32>;
	const uint32_t SharedRowsShift = LONG_SORT ? 32 : count_clz<THREADS-1>::value;
	const uint32_t SharedRowsBits = 32 - count_clz<THREADS-1>::value;
	const SortType SharedRowsColMask = (SortType(1) << SharedRowsShift) - 1;
	const SortType SharedRowsMaskShifted = ~SharedRowsColMask;
	using LoadWorkDistribution = WorkDistribution<THREADS, 2>;
	using SortAndCombiner = SortAndCombine<SortType, typename SEMIRING_t::output_t, THREADS, ELEMENTS_PER_THREAD>;
	using ScanCombinerEntry = typename SortAndCombiner::ScanCombinerEntry;

	struct SMem
	{
    
		uint32_t runflag, chunk_pointer_position;
		uint32_t startSharedRow, numSharedRow;
		INDEX_TYPE minColumnId[THREADS];

		union
		{
			struct
			{
				const typename SEMIRING_t::output_t* dataPointer[2 * THREADS];
				union
				{
					ushort2 fromDataOffset[THREADS];
					uint16_t dataToIndexOffset[2 * THREADS];
				};
				struct {
					typename LoadWorkDistribution::SharedMemT workdistributionMem;
					typename LoadWorkDistribution::SharedTempMemT workdistributionTempMem;
					typename LoadWorkDistribution:: template SharedTempMemOutT<ELEMENTS_PER_THREAD> workdistributionTempMemOutFull;
				};
			};

			typename SortAndCombiner::SMem sAndCMem;

			struct
			{
				typename SEMIRING_t::output_t outDataBuffer[THREADS];
				INDEX_TYPE outIndexBuffer[THREADS];
				ushort2 outRowIdRowOffsetBuffer[THREADS];
				uint32_t outRowCounts[THREADS];
				uint32_t outChunkOffset[THREADS];
			};
		};
	};

	__shared__ SMem smem;

	//get my block's offset
	if (threadIdx.x == 0)
	{
		uint32_t bstart = blockOffsets[blockIdx.x];
		uint32_t shared_handled = shared_rows_handled[blockIdx.x + restart_offset];
		smem.startSharedRow = bstart + shared_handled;
		smem.numSharedRow = blockOffsets[blockIdx.x + 1] - (bstart + shared_handled);
		smem.runflag = *run_flag;
	}

	__syncthreads();

	if (smem.numSharedRow == 0)
		return;

	int count[2] = { 0, 0 };
	
	//load all chunk information
	if (threadIdx.x < smem.numSharedRow)
	{
		uint32_t idoffset[2] = { 0, 0 };
		uint32_t access_index[2] = { 0, 1 };
		uint64_t chunk = reinterpret_cast<uint64_t>(output_row_list_heads[sharedRows[smem.startSharedRow + threadIdx.x]]);
		// if (sharedRows[smem.startSharedRow + threadIdx.x] == ROW_TO_INVESTIGATE)
		// 	printf("Row %d in SIMPLE\n", sharedRows[smem.startSharedRow + threadIdx.x]);
		bool first_row = (chunk & 2) != 0;
		Chunk* __restrict pChunk = reinterpret_cast<Chunk*>(chunk & 0xFFFFFFFFFFFFFFFCULL);
		Chunk* __restrict second;
		if (first_row)
		{
			second = pChunk->readNextFront();
		}
		else
		{
			second = pChunk->readNextBack();
		}
		bool first_row2 = (reinterpret_cast<uint64_t>(second) & 2) != 0;
		second = reinterpret_cast<Chunk*>(reinterpret_cast<uint64_t>(second) & 0xFFFFFFFFFFFFFFFCULL);

#ifdef ENABLE_SORTING
		if (second->sort_key < pChunk->sort_key)
		{
			// Reverse access order
			access_index[0] = 1;
			access_index[1] = 0;
		}
#endif

		INDEX_TYPE minColumnId;
		
		const typename SEMIRING_t::output_t* pdata;
		idoffset[0] = pChunk->num_entries;
		if (first_row)
		{
			count[access_index[0]] = pChunk->firstCountCleared();
			pdata = pChunk->values_direct(idoffset[0]);
			minColumnId = pChunk->indices_direct(idoffset[0])[0];
			idoffset[0] = idoffset[0] * sizeof(typename SEMIRING_t::output_t);
			pChunk->setFirstConsumed();
		}
		else
		{
			count[access_index[0]] = pChunk->lastCountCleared();
			uint32_t baseoffset = idoffset[0] - count[access_index[0]];
			pdata = pChunk->values_direct(idoffset[0]) + baseoffset;
			minColumnId = pChunk->indices_direct(idoffset[0])[baseoffset];
			idoffset[0] = count[access_index[0]] * sizeof(typename SEMIRING_t::output_t) + baseoffset * sizeof(INDEX_TYPE);
			pChunk->setLastConsumed();
		}

		smem.dataPointer[2 * threadIdx.x + access_index[0]] = pdata;		

		idoffset[1] = second->num_entries;
		//we dont need to figure out whether the second pointer is front or back, as front follows back and vice versa
		if (first_row2)
		{
			count[access_index[1]] = second->firstCountCleared();
			minColumnId = min(minColumnId, second->indices_direct(idoffset[1])[0]);
			pdata = second->values_direct(idoffset[1]);
			idoffset[1] = idoffset[1] * sizeof(typename SEMIRING_t::output_t);
			second->setFirstConsumed();
		}
		else
		{
			count[access_index[1]] = second->lastCountCleared();
			uint32_t baseoffset = idoffset[1] - count[access_index[1]];
			minColumnId = min(minColumnId, second->indices_direct(idoffset[1])[baseoffset]);
			pdata = second->values_direct(idoffset[1]) + baseoffset;
			idoffset[1] = count[access_index[1]] * sizeof(typename SEMIRING_t::output_t) + baseoffset * sizeof(INDEX_TYPE);
			second->setLastConsumed();
		}

		smem.dataPointer[2 * threadIdx.x + access_index[1]] = pdata;
		smem.fromDataOffset[threadIdx.x] = make_ushort2(idoffset[access_index[0]], idoffset[access_index[1]]);
		smem.minColumnId[threadIdx.x] = minColumnId;
	}

	//use workdistribution to assign for loading
	LoadWorkDistribution::template initialize<true>(smem.workdistributionMem, smem.workdistributionTempMem, count);

	int rowPair[ELEMENTS_PER_THREAD];
	int element[ELEMENTS_PER_THREAD];

	int elements = LoadWorkDistribution:: template assignWorkAllThreads<false, ELEMENTS_PER_THREAD>(
		smem.workdistributionMem, smem.workdistributionTempMem, smem.workdistributionTempMemOutFull,
		rowPair, element);

	int numOut;
	ScanCombinerEntry combinedEntries[ELEMENTS_PER_THREAD];
	{
		SortType combIndex[ELEMENTS_PER_THREAD];
		typename SEMIRING_t::output_t data[ELEMENTS_PER_THREAD];
#pragma unroll
		for (int i = 0; i < ELEMENTS_PER_THREAD; ++i)
		{
			if (element[i] >= 0)
			{
				const typename SEMIRING_t::output_t* dp = smem.dataPointer[rowPair[i]];
				const INDEX_TYPE* colptr = reinterpret_cast<const INDEX_TYPE*>(reinterpret_cast<const char*>(dp) + smem.dataToIndexOffset[rowPair[i]]);
				INDEX_TYPE colid = colptr[element[i]];
				data[i] = dp[element[i]];
				uint32_t rowId = rowPair[i] / 2;
				SortType redcolid = colid - smem.minColumnId[rowId];
				/*if (redcolid >= (SortType(1) << SharedRowsShift))
					printf("data mix up happening: %d >= %d (shift %d, off %d)!\n", redcolid, 1 << SharedRowsShift, SharedRowsShift, smem.minColumnId[rowId]);*/
				combIndex[i] = (static_cast<SortType>(rowId) << SharedRowsShift) | redcolid;
			}
			else
			{
				data[i] = SEMIRING_t::AdditiveIdentity();
				combIndex[i] = static_cast<SortType>(-1);
			}
		}

		__syncthreads();

		numOut = SortAndCombiner::combine(smem.sAndCMem, combIndex, data, combinedEntries,
			[](auto a, auto b) {
			return a == b;
		},
			[SharedRowsMaskShifted](auto a, auto b) {
			return (a & SharedRowsMaskShifted) == (b & SharedRowsMaskShifted);
		}, semiring, LONG_SORT ? (32 + SharedRowsBits + 1) : 32);
	}

	__syncthreads();

	//write count for rows
	for (int i = 0; i < ELEMENTS_PER_THREAD; ++i)
	{
		if (combinedEntries[i].isRowend())
		{
			uint32_t row = combinedEntries[i].index >> SharedRowsShift;
			uint32_t rcount = combinedEntries[i].rowcount();
			smem.outRowCounts[row] = rcount;
		}
	}

	__syncthreads();

	// Let's see if we can go ahead
	if (threadIdx.x < smem.numSharedRow)
	{
		uint32_t chunkoff = 0xFFFFFFFF;
		int ignored;
		uint32_t elcount = smem.outRowCounts[threadIdx.x];
		if (!allocChunk<typename SEMIRING_t::output_t, INDEX_TYPE>(elcount, chunk_alloc, chunk_size, chunkoff, ignored, false))
		{
			// We have to restart for this block at this point, set run_flag and remember how many rows are left
			atomicOr(run_flag, 0x1);
			smem.runflag = 1;
		}
		else
		{
			smem.outChunkOffset[threadIdx.x] = chunkoff;
		}
	}
	__syncthreads();
	if (smem.runflag != 0)
	{
		return;
	}

	if (threadIdx.x == 0)
	{
		smem.chunk_pointer_position = atomicAdd(chunk_pointer_alloc, smem.numSharedRow);
		if (smem.chunk_pointer_position + smem.numSharedRow > chunk_pointer_sizes)
		{
			atomicOr(run_flag, 0x2);
			smem.runflag = 1;
			if (smem.chunk_pointer_position <= chunk_pointer_sizes)
				*chunk_pointer_pos = smem.chunk_pointer_position;
		}
	}
	__syncthreads();
	if (smem.runflag != 0)
	{
		return;
	}
		
	// Allocate chunk for each row and update count in global
	if (threadIdx.x < smem.numSharedRow)
	{
		uint32_t elcount = smem.outRowCounts[threadIdx.x];
		INDEX_TYPE actualrow = sharedRows[smem.startSharedRow + threadIdx.x];
		//write chunk pointer
		chunks_pointers[smem.chunk_pointer_position + threadIdx.x] = reinterpret_cast<void*>(Chunk::place(chunks, smem.outChunkOffset[threadIdx.x], elcount, actualrow, 0, 0));
		//write row count
		output_row_count[actualrow] = elcount;	
	}

	//loop over data and write out
	for (uint32_t written = 0; written < numOut; written += THREADS)
	{
		//store in shared for coalesced out
#pragma unroll
		for (int i = 0; i < ELEMENTS_PER_THREAD; ++i)
		{
			uint32_t poffset = combinedEntries[i].memoffset();
			if (combinedEntries[i].isResult() &&
				poffset >= written && poffset < written + THREADS)
			{
				uint32_t pwrite = poffset - written;
				uint32_t row = combinedEntries[i].index >> SharedRowsShift;
				smem.outDataBuffer[pwrite] = combinedEntries[i].value;
				smem.outIndexBuffer[pwrite] = static_cast<INDEX_TYPE>(combinedEntries[i].index & SharedRowsColMask) + smem.minColumnId[row];
				smem.outRowIdRowOffsetBuffer[pwrite] = make_ushort2(row, combinedEntries[i].rowcount() - 1);
			}
		}
		__syncthreads();

		//write out
		if (written + threadIdx.x < numOut)
		{
			ushort2 row_offset = smem.outRowIdRowOffsetBuffer[threadIdx.x];
			uint32_t chunkoffset = smem.outChunkOffset[row_offset.x];
			if (chunkoffset != 0xFFFFFFFF)
			{
				uint32_t count = smem.outRowCounts[row_offset.x];
				typename SEMIRING_t::output_t* valstart = Chunk::cast(chunks, chunkoffset)->values_direct(count);
				INDEX_TYPE* indexstart = Chunk::cast(chunks, chunkoffset)->indices_direct(count);
				valstart[row_offset.y] = smem.outDataBuffer[threadIdx.x];
				indexstart[row_offset.y] = smem.outIndexBuffer[threadIdx.x];
			}
		}
		__syncthreads();
	}

	// Indicator for restart
	if (threadIdx.x == 0)
		shared_rows_handled[blockIdx.x + restart_offset] += smem.numSharedRow;

	return;
}


template<uint32_t NNZ_PER_THREAD, uint32_t THREADS, uint32_t BLOCKS_PER_MP, uint32_t INPUT_ELEMENTS_PER_THREAD, uint32_t RETAIN_ELEMENTS_PER_THREAD, uint32_t MERGE_MAX_CHUNKS, uint32_t MERGE_MAX_PATH_OPTIONS, typename VALUE_TYPE, typename INDEX_TYPE, typename OFFSET_TYPE, bool LONG_SORT,
        typename T, typename U, typename Label,
        typename SEMIRING_t>
        void AcSpGEMMKernels::h_mergeSharedRowsSimple(const uint32_t* __restrict blockOffsets, const uint32_t* __restrict sharedRows, void** output_row_list_heads,
	OFFSET_TYPE* output_row_count,
	uint32_t* chunks, uint32_t* chunk_alloc, uint32_t* chunk_pre_alloc, uint32_t chunk_size,
	void** chunks_pointers, uint32_t* chunk_pointer_alloc, uint32_t chunk_pointer_sizes,
	uint32_t* run_flag, uint32_t* restart_completion, uint32_t* shared_rows_handled, uint32_t restart_offset, uint32_t* chunk_pointer_pos, SEMIRING_t semiring)
{
	mergeSharedRowsSimple<NNZ_PER_THREAD, THREADS, BLOCKS_PER_MP, INPUT_ELEMENTS_PER_THREAD, RETAIN_ELEMENTS_PER_THREAD, MERGE_MAX_CHUNKS, MERGE_MAX_PATH_OPTIONS, VALUE_TYPE, INDEX_TYPE, OFFSET_TYPE, LONG_SORT,  T,  U,  Label,SEMIRING_t><<<gridDim, blockDim>>>(
	blockOffsets, sharedRows, output_row_list_heads, output_row_count, chunks, chunk_alloc, chunk_pre_alloc, chunk_size,
	chunks_pointers, chunk_pointer_alloc, chunk_pointer_sizes, run_flag, restart_completion, shared_rows_handled, restart_offset, chunk_pointer_pos, semiring);
}

#define GPUCompressedMatrixMatrixMultiplyMergeSimple(TYPE, THREADS, BLOCKS_PER_MP, NNZPERTHREAD, INPUT_ELEMENTS_PER_THREAD, RETAIN_ELEMENTS_PER_THREAD, MERGE_MAX_CHUNKS, MERGE_MAX_PATH_OPTIONS) \
	template void AcSpGEMMKernels::h_mergeSharedRowsSimple<NNZPERTHREAD, THREADS, BLOCKS_PER_MP, INPUT_ELEMENTS_PER_THREAD, RETAIN_ELEMENTS_PER_THREAD, MERGE_MAX_CHUNKS, MERGE_MAX_PATH_OPTIONS, TYPE, uint32_t, uint32_t, false> \
	(const uint32_t* __restrict blockOffsets, const uint32_t* __restrict sharedRows, void** output_row_list_heads, \
	uint32_t* output_row_count, \
	uint32_t* chunks, uint32_t* chunk_alloc, uint32_t* chunk_pre_alloc, uint32_t chunk_size, \
	void** chunks_pointers, uint32_t* chunk_pointer_alloc, uint32_t chunk_pointer_sizes, \
	uint32_t* run_flag, uint32_t* restart_completion, uint32_t* shared_rows_handled, uint32_t restart_offset, uint32_t* chunk_pointer_pos); \
	template void AcSpGEMMKernels::h_mergeSharedRowsSimple<NNZPERTHREAD, THREADS, BLOCKS_PER_MP, INPUT_ELEMENTS_PER_THREAD, RETAIN_ELEMENTS_PER_THREAD, MERGE_MAX_CHUNKS, MERGE_MAX_PATH_OPTIONS, TYPE, uint32_t, uint32_t, true> \
	(const uint32_t* __restrict blockOffsets, const uint32_t* __restrict sharedRows, void** output_row_list_heads, \
	uint32_t* output_row_count, \
	uint32_t* chunks, uint32_t* chunk_alloc, uint32_t* chunk_pre_alloc, uint32_t chunk_size, \
	void** chunks_pointers, uint32_t* chunk_pointer_alloc, uint32_t chunk_pointer_sizes, \
	uint32_t* run_flag, uint32_t* restart_completion, uint32_t* shared_rows_handled, uint32_t restart_offset, uint32_t* chunk_pointer_pos);
	