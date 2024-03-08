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
 * SpGEMM.cuh
 *
 * ac-SpGEMM
 *
 * Authors: Daniel Mlakar, Markus Steinberger, Martin Winter
 *------------------------------------------------------------------------------
*/

#pragma once

#include <cuda.h>
#include <limits>
#include <cub/cub.cuh>
#include "MultiplyKernels.h"
#include "Chunk.cuh"
#include "HelperFunctions.cuh"
#include "WorkDistribution.cuh"
#include "ARowStorage.cuh"
#include "SortAndCombine.cuh"


//SORT_TYPE_MODE 0 .. 32bit direct, 1 32bit row remap, 2 64bit full
template<uint32_t NNZ_PER_THREAD, uint32_t THREADS, uint32_t BLOCKS_PER_MP, uint32_t INPUT_ELEMENTS_PER_THREAD, uint32_t RETAIN_ELEMENTS_PER_THREAD, uint32_t MERGE_MAX_PATH_OPTIONS, typename VALUE_TYPE1, typename VALUE_TYPE2, typename VALUE_TYPE3, typename INDEX_TYPE, typename OFFSET_TYPE, int SORT_TYPE_MODE,
        typename T, typename U, typename Label,
        typename SEMIRING_t>
        __global__ void __launch_bounds__(THREADS, BLOCKS_PER_MP)
computeSpgemmPart(
	const typename SEMIRING_t::leftInput_t* valA, const INDEX_TYPE* indicesA, const OFFSET_TYPE* __restrict offsetsA,
	const typename SEMIRING_t::rightInput_t *__restrict valB, const INDEX_TYPE* __restrict indicesB, const OFFSET_TYPE* __restrict offsetsB,
	const uint32_t* __restrict startingIdsA, uint32_t nnz, uint32_t rows,
	uint32_t* chunks, uint32_t* chunk_alloc, uint32_t* chunk_worst_case, uint32_t chunk_size,
	void** chunks_pointers, uint32_t* chunk_pointer_alloc, uint32_t chunk_pointer_sizes,
	OFFSET_TYPE* output_row_count, void** output_row_list_heads, uint32_t* output_row_chunk_count,
	uint32_t* shared_rows_tracker, uint32_t* shared_rows_alloc, float expected_row_overlap, float expected_row_overlap_inv,
	uint32_t* run_flag, uint32_t* completion_status, uint32_t* chunk_counter, uint32_t* chunk_pointer_pos, SEMIRING_t semiring)
{
	static_assert(RETAIN_ELEMENTS_PER_THREAD >= 1, "need at least one temporary element per thread to assure coalesced write out");
	// fetch A data
	//  tag with row and col ids
	//
	// fill work distribution
	//
	// fetch rows from B (each thread fetches one element)
	//  multiply and sort in (multiply, sort, prefix sum)
	//  run a scan to combine, compute row offset, and memory offset
	//
	// either write out chunk or keep all data/last row in shared memory to continue the combination

	const int NNZ_PER_BLOCK = NNZ_PER_THREAD*THREADS;
	const int TEMP_ITEMS_PER_BLOCK = (RETAIN_ELEMENTS_PER_THREAD*THREADS);

	using LEFT_T = typename SEMIRING_t::leftInput_t;
	using RIGHT_t = typename SEMIRING_t::rightInput_t;

	//SORT_TYPE_MODE 0 .. 32bit direct, 1 32bit row remap, 2 64bit full
	using SortType = ChooseBitDataType<(SORT_TYPE_MODE > 1 ) ? 64 : 32>;

	const uint32_t ChunkSortingBits = (sizeof(ChunkSortType) * 8) - count_clz<NNZ_PER_BLOCK>::value;

	// the number of elements each threads handles in registers
	const int CombineElements = INPUT_ELEMENTS_PER_THREAD + RETAIN_ELEMENTS_PER_THREAD;

	// cutoff for rows in B which will directly be forwarded to the merge stage
	const uint32_t LongRowCutOff = CombineElements * THREADS / 2;

	// used data types specialized for the setup
	using RowelementWorkDistribution = WorkDistribution<THREADS, NNZ_PER_THREAD>;
	using SortAndCombiner = SortAndCombine<SortType, typename SEMIRING_t::output_t, THREADS, CombineElements>;
	using ScanCombinerEntry = typename SortAndCombiner::ScanCombinerEntry;
	using SimpleScan = cub::BlockScan<uint32_t, THREADS>;
	using SimpleIntScan = cub::BlockScan<int32_t, THREADS>;
	using Chunk = Chunk<typename SEMIRING_t::output_t, INDEX_TYPE>;
	using DirectChunk = DirectChunk<typename SEMIRING_t::leftInput_t, typename SEMIRING_t::rightInput_t, INDEX_TYPE>;

	using ARowStorage = ARowStorage<INDEX_TYPE, NNZ_PER_BLOCK, THREADS, SORT_TYPE_MODE == 1>;
	struct SMem
	{
   
		// flattened out A data
		//INDEX_TYPE A_row_ids[NNZ_PER_BLOCK];
		uint32_t chunk_pointer_position, chunk_counter;
		ARowStorage A_row_ids;
		INDEX_TYPE A_col_ids[NNZ_PER_BLOCK];
        typename SEMIRING_t::leftInput_t A_indata[NNZ_PER_BLOCK];




		// comb data
		union {
			struct {
				INDEX_TYPE current_col_ids[TEMP_ITEMS_PER_BLOCK];
				typename ARowStorage::EncodedRowType current_row_ids[TEMP_ITEMS_PER_BLOCK < THREADS ? THREADS + 1 : TEMP_ITEMS_PER_BLOCK];
				typename SEMIRING_t::output_t current_output[TEMP_ITEMS_PER_BLOCK];
			};
			struct {
				uint32_t temp_work_storage_single[NNZ_PER_BLOCK];
			};
		};
		
		//TODO: temp mem and comb data could be overlapped!?

		// temp mem
		union {
			struct {
				typename RowelementWorkDistribution::SharedTempMemT workdistributionTempMem;
				typename RowelementWorkDistribution:: template SharedTempMemOutT<CombineElements> workdistributionTempMemOutFull;
			};
			struct {
				typename SimpleScan::TempStorage directChunkScanTempMem;
				typename SimpleScan::TempStorage nonDirectChunkScanTempMem;
			};
			typename SimpleIntScan::TempStorage intScanTempMem;
			typename SortAndCombiner::SMem sAndCMem;
			INDEX_TYPE rowCounts[TEMP_ITEMS_PER_BLOCK];
		};


		//work distribution
		typename RowelementWorkDistribution::SharedMemT workdistributionMem;

		INDEX_TYPE minCol, maxCol;
		typename ARowStorage::EncodedRowType minRow, maxRow;

		uint32_t chunkStartOffset;
		uint32_t firstRowCount;
		uint32_t lastRowCount;
		uint32_t runflag;
		uint32_t directChunkRows;
		uint32_t brokenChunkOffsetStart, brokenChunkOffsetEnd;

		typename ARowStorage::EncodedRowType minBrokenChunkRow, maxBrokenChunkRow;
	};

	__shared__ SMem smem;

	__shared__ uint32_t block_start_end[2];
	//__shared__ int currentStartElementIndex, currentEndElementIndex;
	//__shared__ uint32_t elem_handled_A, elem_handled_B, max_A, max_B, restart;
	//__shared__ float lastExpected;
	__shared__ int tempOffset, tempData, workavailable, consumedwork;

	// get block data
	if (threadIdx.x < 2)
	{
		block_start_end[threadIdx.x] = startingIdsA[blockIdx.x + threadIdx.x];
		//smem.A_row_ids[0] = static_cast<INDEX_TYPE>(-1);
		//currentEndElementIndex = completion_status[blockIdx.x];
		//lastExpected = 0.0f;

		// if we stopped globally, dont even start, otherwise consider restart
		//if (threadIdx.x == 0 && completion_status[blockIdx.x] != 0 && completion_status[blockIdx.x] != 0xFFFFFFFF)
		//	printf("%d restarting with %x %d\n", blockIdx.x, completion_status[blockIdx.x], completion_status[blockIdx.x] & (~0x80000000));
		smem.chunk_pointer_position = 0;
		smem.directChunkRows = 0;
		smem.runflag = *run_flag != 0 ? 0xFFFFFFFF : completion_status[blockIdx.x];
		smem.chunk_counter = chunk_counter[blockIdx.x];

		// for consume based restart, set consumedwork too
		consumedwork = (smem.runflag & 0x80000000) == 0 ? smem.runflag : 0;
	}

	smem.A_row_ids.clear();

	__syncthreads();
	if (smem.runflag == std::numeric_limits<uint32_t>::max())
		return;

	int worknnz = min(NNZ_PER_BLOCK, nnz - blockIdx.x * NNZ_PER_BLOCK);

	// Assign column ids of a
	//TODO: adjust num threads per row either dynamic (could be always pow 2) or a few preset static ones
	for (uint32_t r = block_start_end[0] + threadIdx.x; r <= block_start_end[1]; r += THREADS)
	{
		int ain = static_cast<int>(offsetsA[r] - blockIdx.x * NNZ_PER_BLOCK);
		int bin = offsetsA[min(rows, r + 1)] - blockIdx.x * NNZ_PER_BLOCK;

		int a = max(0, ain);
		int b = min(static_cast<int>(worknnz), bin);

		//iterate over all threads that start with that row
		if (a < b)
		{
			smem.A_row_ids.storeReference(a, r);
			int ra = a;
			smem.A_row_ids.storeRow(a, ra, r);
			for (++a; a < b; ++a)
				smem.A_row_ids.storeRow(a, ra, r);
		}
	}

	__syncthreads();

	bool directChunkRows = false;
	int workToDistribute[NNZ_PER_THREAD];

	// Read out lengths of rows from B for each element from A
	#pragma unroll
	for (uint32_t i = 0; i < NNZ_PER_THREAD; ++i)
	{
		uint32_t w = threadIdx.x + i * THREADS;
		INDEX_TYPE a_col = 0;
		uint32_t b_num = 0;
		

		if (w < worknnz)
		{
			// normal case or work element based restart
			bool load = true;
			
			if(load)
			{
				uint32_t l = w + blockIdx.x * NNZ_PER_BLOCK;
				a_col = indicesA[l];
				b_num = offsetsB[a_col + 1] - offsetsB[a_col];

				smem.A_col_ids[w] = indicesA[l];
				smem.A_indata[w] = valA[l];

				// Long rows are directly referred to the merge stage by only writing an identifier chunck info
				if (b_num >= LongRowCutOff)
				{
					// remember that we are now deadling with a dirct chunk row, which needs sorting
					b_num = b_num | 0x80000000;
					directChunkRows = true;
				}
				else if ((smem.runflag & 0x80000000) != 0)
				{
					// row based restart needs to set the consumed work too
					uint32_t to_start_row = smem.A_row_ids.restartRowDecode((smem.runflag & (~0x80000000)), block_start_end[0]);
					if (smem.A_row_ids.getEncodedRow(w) < to_start_row)
					{
						//printf("%d %d load  %x\n", blockIdx.x, threadIdx.x, completion_status[blockIdx.x]);
						atomicAdd(&consumedwork, b_num);
						b_num = 0;
					}
				}
			}
		}
		workToDistribute[i] = b_num;
	}

	// move all direct chunk rows to the front so we can quickly identify them later
	if (__syncthreads_or(directChunkRows))
	{
		// only write out during first run
		if (smem.runflag == 0)
		{
			uint32_t chunkoff[NNZ_PER_THREAD];
			bool success = true;

			#pragma unroll
			for (uint32_t i = 0; i < NNZ_PER_THREAD; ++i)
			{
				// alloc special chunk and write out
				if ((workToDistribute[i] & 0x80000000) != 0)
				{
					//FIXME: This is the wrong typez
				//	printf("%d %d allocating direct chunk for size %d\n", blockIdx.x, threadIdx.x, (workToDistribute[i] & (~0x80000000)));
					if (!allocDirectChunk<typename SEMIRING_t::leftInput_t, typename SEMIRING_t::rightInput_t, INDEX_TYPE>(chunk_alloc, chunk_size, chunkoff[i]))
					{
						success = false;
						atomicOr(run_flag, 0x1);
					}
					atomicAdd(&(smem.chunk_pointer_position), 1);
				}
			}
			if (__syncthreads_or(!success))
			{
				//re start with old state and alloc all chunks in next run
				return;
			}

			if (threadIdx.x == 0)
			{
				uint32_t num_chunks = smem.chunk_pointer_position;
				smem.chunk_pointer_position = atomicAdd(chunk_pointer_alloc, num_chunks);
				if (smem.chunk_pointer_position + num_chunks >= chunk_pointer_sizes)
				{
					success = false;
					atomicOr(run_flag, 0x2);
					if(smem.chunk_pointer_position < chunk_pointer_sizes)
						*chunk_pointer_pos = smem.chunk_pointer_position;
				}
			}
			if (__syncthreads_or(!success))
			{
				//re start with old state and alloc all chunks in next run
				return;
			}

			#pragma unroll
			for (uint32_t i = 0; i < NNZ_PER_THREAD; ++i)
			{
				if ((workToDistribute[i] & 0x80000000) != 0)
				{
				//	printf("%d %d added DirectChunk for row \n", blockIdx.x, threadIdx.x);

					// write chunk data
					DirectChunk * p_chunk = DirectChunk::cast(chunks, chunkoff[i]);
					chunks_pointers[atomicAdd(&(smem.chunk_pointer_position), 1)] = reinterpret_cast<void*>(p_chunk);

					uint32_t w = threadIdx.x + i * THREADS;
					auto encodedRow = smem.A_row_ids.getEncodedRow(w);
					INDEX_TYPE r = smem.A_row_ids.decodeRow(encodedRow);
					INDEX_TYPE a_col = smem.A_col_ids[w];
					uint32_t b_num = workToDistribute[i] & (~0x80000000);
					DirectChunk::place(chunks, chunkoff[i], b_num, r, indicesB + offsetsB[a_col], valB + offsetsB[a_col], smem.A_indata[w], (static_cast<ChunkSortType>(blockIdx.x) << ChunkSortingBits) | (threadIdx.x + i*THREADS + NNZ_PER_BLOCK));
					addPotentiallySharedRow(r, p_chunk, true, output_row_list_heads, shared_rows_tracker, shared_rows_alloc, true);

					// if ((r == 0))
						// printf("We have a direct chunk in row: %u with %u elements with col: %u\n", r, b_num, a_col);

					atomicAdd(output_row_chunk_count + r, 1);
					// mark so we do not go through simple merge
					if (INPUT_ELEMENTS_PER_THREAD * THREADS * MERGE_MAX_PATH_OPTIONS >= b_num)
					{
						// Set both top most bits if this can go to max chunks case
						atomicOr(output_row_chunk_count + r, MAX_CHUNKS_CASE);
					}						
					else
					{
						// Only set the topmost bit if this should go to the generalized case
						atomicOr(output_row_chunk_count + r, GENERALIZED_CASE);
					}
						

					//no need to set count, as we will go through max or general merge anyway
			//    	atomicAdd(output_row_count + r, CombineElements * THREADS);
					
				}
			}
		}
		
		
		#pragma unroll
		for (uint32_t i = 0; i < NNZ_PER_THREAD; ++i)
			smem.temp_work_storage_single[threadIdx.x + i * THREADS] = ((workToDistribute[i] & 0x80000000) != 0) ? 0xFFFFFFFF : workToDistribute[i];
		__syncthreads();

		// run a prefix sum to figure out where to place the direct chunk row ids and others
		uint32_t direct[NNZ_PER_THREAD], nonDirect[NNZ_PER_THREAD];
		for (uint32_t i = 0; i < NNZ_PER_THREAD; ++i)
		{
			// note stripped layout
			if (smem.temp_work_storage_single[threadIdx.x * NNZ_PER_THREAD + i] == 0xFFFFFFFF)
			{
				direct[i] = 1;
				nonDirect[i] = 0;
			}
			else
			{
				direct[i] = 0;
				nonDirect[i] = 1;
			}
		}
		uint32_t sum_direct;
		SimpleScan(smem.directChunkScanTempMem).ExclusiveSum(direct, direct, sum_direct);
		SimpleScan(smem.nonDirectChunkScanTempMem).ExclusiveSum(nonDirect, nonDirect);

		INDEX_TYPE a_col[NNZ_PER_THREAD];
		VALUE_TYPE1 a_vals[NNZ_PER_THREAD];
		typename ARowStorage::EncodedRowType a_rowIds[NNZ_PER_THREAD];

		//fetch the data
		#pragma unroll
		for (uint32_t i = 0; i < NNZ_PER_THREAD; ++i)
		{
			// note stripped layout
			int r = threadIdx.x * NNZ_PER_THREAD + i;
			a_col[i] = smem.A_col_ids[r];
			a_vals[i] = smem.A_indata[r];
			a_rowIds[i] = smem.A_row_ids.getEncodedRow(r);
			workToDistribute[i] = smem.temp_work_storage_single[r];
		}
		__syncthreads();

		//store shuffled and cleared workload
		#pragma unroll
		for (uint32_t i = 0; i < NNZ_PER_THREAD; ++i)
		{

			// note stripped layout
			uint32_t p = nonDirect[i] + sum_direct;
			if (workToDistribute[i] == 0xFFFFFFFF)
			{
				workToDistribute[i] = 0;
				p = direct[i];
			}

			smem.A_col_ids[p] = a_col[i];
			smem.A_indata[p] = a_vals[i];
			smem.A_row_ids.storeEncodedRow(p, a_rowIds[i]);
			smem.temp_work_storage_single[p] = workToDistribute[i];
		}

		smem.directChunkRows = sum_direct;
		__syncthreads();

		// load new work
		#pragma unroll
		for (uint32_t i = 0; i < NNZ_PER_THREAD; ++i)
			workToDistribute[i] = smem.temp_work_storage_single[threadIdx.x * NNZ_PER_THREAD + i];

		// Initialize the work distribution from stripped layout
		RowelementWorkDistribution:: template initialize<true>(smem.workdistributionMem, smem.workdistributionTempMem, workToDistribute);
	}
	else
	{
		// Initialize the work distribution from blocked layoyt and run while work is available
		RowelementWorkDistribution:: template initialize<false>(smem.workdistributionMem, smem.workdistributionTempMem, workToDistribute);
	}

	// now kept in shared
	tempData = 0;
	tempOffset = 0;


	// comsume based restart
	if (smem.runflag != 0 && (smem.runflag & 0x80000000) == 0)
	{
		RowelementWorkDistribution::removework(smem.workdistributionMem, smem.runflag);
	}
	

	// note: potential race condition with removework, however the entire work will never be removed and we only compare with > 0 -> fine?
	// TODO: -> we can remove syncthreads!?
	__syncthreads();
	
	workavailable = RowelementWorkDistribution::workAvailable(smem.workdistributionMem);
	while (workavailable > 0)
	{

		int localAEntry[CombineElements];
		int elementB[CombineElements];

		int elements = RowelementWorkDistribution:: template assignWorkAllThreads<false, CombineElements>(
			smem.workdistributionMem, smem.workdistributionTempMem, smem.workdistributionTempMemOutFull,
			localAEntry, elementB, CombineElements*THREADS - tempData);

		if(threadIdx.x == 0)
			consumedwork += CombineElements*THREADS - tempData;


		typename ARowStorage::EncodedRowType temp_row[CombineElements];
		INDEX_TYPE temp_col_id[CombineElements];
		typename SEMIRING_t::output_t temp_val[CombineElements];

		smem.minCol = smem.minRow = std::numeric_limits<INDEX_TYPE>::max();
		smem.maxCol = smem.maxRow = 0;

		// locel min/max row and col
		INDEX_TYPE minRow = std::numeric_limits<INDEX_TYPE>::max(), maxRow = 0;
		INDEX_TYPE minCol = std::numeric_limits<INDEX_TYPE>::max(), maxCol = 0;
		


		//fetch B data and set MIN/MAX values for how many rows in A and how many cols and B are touched
		#pragma unroll
		for (int i = 0; i < CombineElements; ++i)
		{
			if (i < elements)
			{
				uint32_t aentry = localAEntry[i];
				uint32_t fetch_row = smem.A_col_ids[aentry];
				temp_row[i] = smem.A_row_ids.getEncodedRow(aentry);
				minRow = min(minRow, temp_row[i]);
				maxRow = max(maxRow, temp_row[i]);
			

				//if (elementB[i] < 0 || aentry >= worknnz)
				//	printf("%d %d [%d]: max %d - nnz: %d - req: %d/%d - %d %d\n", blockIdx.x, threadIdx.x, i, elements, worknnz, CombineElements*THREADS - tempData, workavailable, localAEntry[i], elementB[i]);

				INDEX_TYPE elb = offsetsB[fetch_row] + elementB[i];
				temp_col_id[i] = indicesB[elb];
				temp_val[i] = semiring.multiply(smem.A_indata[aentry], valB[elb]);

				minCol = min(minCol, temp_col_id[i]);
				maxCol = max(maxCol, temp_col_id[i]);
			}
			else
			{
				// get from last iteration
				int t = i * THREADS + threadIdx.x - (CombineElements*THREADS - tempData);
				if (t >= 0)
				{
					int access = (tempOffset + t) % TEMP_ITEMS_PER_BLOCK;
					// offset tells us where the last row data is currently placed
					temp_row[i] = smem.current_row_ids[access];
					temp_col_id[i] = smem.current_col_ids[access];
					temp_val[i] = smem.current_output[access];

				//	printf("%d %d (%d %d %d): %d %d %f\n", blockIdx.x, threadIdx.x, access, t, tempData, temp_row[i], temp_col_id[i], temp_val[i]);

					minRow = min(minRow, temp_row[i]);
					maxRow = max(maxRow, temp_row[i]);

					minCol = min(minCol, temp_col_id[i]);
					maxCol = max(maxCol, temp_col_id[i]);

					//dummy value to indicate that we have something
					elementB[i] = 1;
				}
				else
					//indicate that we are empty
					elementB[i] = -1;
			}
		}

		//
		updateMinValue(smem.minCol, minCol);
		updateMinValue(smem.minRow, minRow);
		updateMaxValue(smem.maxCol, maxCol);
		updateMaxValue(smem.maxRow, maxRow);

		__syncthreads();


		INDEX_TYPE colRange = smem.maxCol - smem.minCol;
		INDEX_TYPE rowRange = smem.maxRow - smem.minRow + 1;
		INDEX_TYPE colBits = 32 - __clz(colRange);
		INDEX_TYPE rowBits = 32 - __clz(rowRange);


		if (colBits + rowBits > 32 && threadIdx.x == 0)
		{
			printf("colRange: %u rowRange: %u colBits: %u rowBits: %u | minCol: %u maxCol: %u | minRow: %u maxRow: %u\n", colRange, rowRange, colBits, rowBits, smem.minCol, smem.maxCol, smem.minRow, smem.maxRow);
			//return;
		}

		ScanCombinerEntry combinedEntries[CombineElements];
		{
			//TODO: if there are fewer items only, we want to only sort those...
			//TODO: if we can use uint32_t instead of uint64_t we want to use that...
			SortType combIndex[CombineElements];
			typename SEMIRING_t::output_t data[CombineElements];
			#pragma unroll
			for (int i = 0; i < CombineElements; ++i)
			{
				if (elementB[i] >= 0)
				{
					combIndex[i] = (static_cast<SortType>(temp_row[i] - smem.minRow) << colBits) | (temp_col_id[i] - smem.minCol);
					data[i] = temp_val[i];
				}
				else
				{
					combIndex[i] = ~SortType(0);
					data[i] = SEMIRING_t::AdditiveIdentity();
				}
			}

			tempData = SortAndCombiner::combine(smem.sAndCMem, combIndex, data, combinedEntries,
				[](auto a, auto b) {
				return a == b;
			},
				[colBits](auto a, auto b) {
				return (a >> colBits) == (b >> colBits);
			}, semiring,
				colBits + rowBits);

		}


		workavailable = RowelementWorkDistribution::workAvailable(smem.workdistributionMem);

		//we would like to know how many elements we have from the last row
		// TODO: check if that is right
		#pragma unroll
		for (int i = 0; i < CombineElements; ++i)
			if (combinedEntries[i].isRowend() && combinedEntries[i].memoffset() == tempData - 1)
				smem.lastRowCount = combinedEntries[i].rowcount();

		__syncthreads();

		// if (threadIdx.x == 0)
		// 	printf("%d decision to make: %d >= %d || !%d || 8 * %d < %d\n", blockIdx.x, tempData, TEMP_ITEMS_PER_BLOCK, workavailable, smem.lastRowCount, tempData);

		// TODO: check heuristic
		// if we must go out or if the last row is very small in comparison to the other data
		if (tempData >= TEMP_ITEMS_PER_BLOCK || workavailable <= 0 || 1* smem.lastRowCount < tempData)
		{
			// keep the last row around if we can so we reduce the amount of merging we have to perform
			int allocData = workavailable > 0 && smem.lastRowCount < TEMP_ITEMS_PER_BLOCK ? tempData - smem.lastRowCount : tempData;

			// determine how many chunks we need to generate (additional ones for single row chunks in between)
			bool multiChunk = false;

			if (smem.directChunkRows != 0)
			{
				for (uint32_t i = threadIdx.x; i < smem.directChunkRows; i += THREADS)
					multiChunk = multiChunk || (smem.minRow < smem.A_row_ids.getEncodedRow(i) && smem.A_row_ids.getEncodedRow(i) < smem.maxRow);

				multiChunk = __syncthreads_or(multiChunk);
			}


			if (multiChunk)
			{
				// we need to separate the output into multiple chunks
				//if (threadIdx.x == 0 && (smem.maxRow == 7094 || smem.maxRow == 6025 || smem.maxRow == 5086 || smem.maxRow == 5273 || smem.maxRow == 7350))
				//	printf("%d %d split chunk for %d-%d .. %d %d\n", blockIdx.x, threadIdx.x, smem.minRow, smem.maxRow, allocData, tempData);

				// init smem
				smem.brokenChunkOffsetStart = 0;
				smem.minBrokenChunkRow = smem.minRow;
				smem.maxBrokenChunkRow = smem.maxRow;


				// determine individual chunk ends
				// iterate over shared rows list and my data to see how many chunk boundaries i need to add
				// need access to the next element -> store in shared
				smem.current_row_ids[threadIdx.x+1] = (combinedEntries[CombineElements-1].index >> colBits) + smem.minRow;
				smem.current_row_ids[0] = smem.minRow;

				__syncthreads();
				uint32_t chunk_splitting_row_id = 0;
				uint32_t chunk_splitting_row = smem.A_row_ids.getEncodedRow(chunk_splitting_row_id);
				typename ARowStorage::EncodedRowType r = smem.current_row_ids[threadIdx.x];
				// search for the first chunk breaking row that is larger than the row handled by the previous thread
				// ie find the first chunk breaking row that can be relevant for my entries
				while (chunk_splitting_row <= r)
				{
					if (++chunk_splitting_row_id < smem.directChunkRows)
					{
						chunk_splitting_row = smem.A_row_ids.getEncodedRow(chunk_splitting_row_id);
					}
					else
					{
						// this threads entries are above all chunk breaking rows, so set it to max
						chunk_splitting_row = smem.maxRow + 1;
						break;
					}
				}

				//if (threadIdx.x == 0 && smem.maxRow == 7094)
				//{
				//	printf("Min: %u and Max: %u\n", smem.minRow, smem.maxRow);
				//}

				//if (/*threadIdx.x == 0 &&*/ smem.maxRow == 7094)
				//{
				//	printf("Chunk splitting row: %u with r: %u\n", chunk_splitting_row, r);
				//}

				// determine where to break
				static_assert(CombineElements <= 32, "can handle a maximum of 32 CombinedElements when performing multi chunk out");
				uint32_t chunk_breaks = 0;
				#pragma unroll
				for (int i = 0; i < CombineElements; ++i)
				{
					typename ARowStorage::EncodedRowType next_r = min(static_cast<typename ARowStorage::EncodedRowType>((combinedEntries[i].index >> colBits)) + smem.minRow, smem.maxRow);
					/*if (smem.maxRow == 7094 && r < 7100 && next_r < 7100 && r != next_r)
					{
						printf("Row given: %u | %u nextrow\n", r, next_r);
					}*/
					/*if (r != next_r && chunk_splitting_row <= next_r && (next_r == smem.maxRow) && chunk_splitting_row != smem.maxRow)
					{
						printf("R: %u | next_R: %u | chunk_splitting: %u | max: %u ----- directid: %u maxid: %u\n", r, next_r, chunk_splitting_row, smem.maxRow, chunk_splitting_row_id, smem.directChunkRows);
					}*/
					/*if (r != next_r && chunk_splitting_row <= next_r && next_r != smem.maxRow)*/
					/*if (r != next_r && chunk_splitting_row <= next_r && chunk_splitting_row != smem.maxRow && tempData == allocData)*/
					/*if (r != next_r && chunk_splitting_row <= next_r && (next_r != smem.maxRow || next_r == 7094 || next_r == 6025))*/
					if (r != next_r && chunk_splitting_row <= next_r && (next_r != smem.maxRow || (chunk_splitting_row != smem.maxRow && tempData == allocData)))
					{
						// we are at a chunk boundary
						chunk_breaks |= (1 << i);
						//if(smem.maxRow == 7094)
						//	printf("%d %d breaks chunk between %d %d\n", blockIdx.x, threadIdx.x, r, next_r);
						// find next
						do
						{
							if (++chunk_splitting_row_id < smem.directChunkRows)
								chunk_splitting_row = smem.A_row_ids.getEncodedRow(chunk_splitting_row_id);
							else
							{
								chunk_splitting_row = smem.maxRow + 1;
								break;
							}
						} while (chunk_splitting_row <= next_r);
					}
					r = next_r;
				}

				// run prefix sum to figure out how many chunk breaks to insert
				int num_broken_chunks[1] = { __popc(chunk_breaks) };
				int overall_broken_chunk, my_starting_offset[1];
				SimpleIntScan(smem.intScanTempMem).ExclusiveSum(num_broken_chunks, my_starting_offset, overall_broken_chunk);


				//if (threadIdx.x == 0 && smem.maxRow == 7094)
				//	printf("%d %d overall broken chunks: %d\n", blockIdx.x, threadIdx.x, overall_broken_chunk);

				// iterate over broken up chunks and write out in the typical manner
				for (int c = 0; c <= overall_broken_chunk; ++c)
				{
					__syncthreads();
					int local_chunk = c - my_starting_offset[0];
					if (local_chunk >= 0 && local_chunk < num_broken_chunks[0])
					{
						// it is our chunk - extract
						int handled_bits = 0;
						#pragma unroll
						for (int i = 0; i < CombineElements; ++i)
						{
							if ((chunk_breaks & (1 << i)) != 0)
							{
								if (handled_bits == local_chunk)
								{
									if (combinedEntries[i].isResult())
										smem.brokenChunkOffsetEnd = combinedEntries[i].memoffset();
									else
										smem.brokenChunkOffsetEnd = combinedEntries[i].memoffset() + 1;
									//printf("%d %d its my chunk time %d: %d\n", blockIdx.x, threadIdx.x, i, combinedEntries[i].memoffset());
								}
								++handled_bits;
							}
						}
					}
					__syncthreads();

					if(threadIdx.x == 0)
					{
						if (c == overall_broken_chunk)
						{
							// need to setup last chunk
							smem.brokenChunkOffsetEnd = smem.brokenChunkOffsetStart + tempData;
							//printf("%d %d its last chunk time: %d\n", blockIdx.x, threadIdx.x, smem.brokenChunkOffsetStart + tempData);
						}
						/*if (threadIdx.x == 0 && smem.maxRow == 1878)
							printf("We have allocData %u and other %u\n", allocData, smem.brokenChunkOffsetEnd - smem.brokenChunkOffsetStart);*/
						uint32_t chunkoff = completeChunkAlloc<typename SEMIRING_t::output_t, INDEX_TYPE>(min(smem.brokenChunkOffsetEnd - smem.brokenChunkOffsetStart, allocData), chunks, chunk_alloc, chunk_size, chunks_pointers, chunk_pointer_alloc, chunk_pointer_sizes, chunk_pointer_pos,
							[&]()
						{
							atomicOr(run_flag, 0x1);
							//if(threadIdx.x == 0)

							// Write out descriptor for restart into global
							//printf("%d going for restart: %x: %d -> %d -- block row range: %d<->%d\n", blockIdx.x, smem.runflag, (smem.runflag&(~0x80000000)), smem.A_row_ids.decodeRow(smem.A_row_ids.restartRowDecode((smem.runflag & (~0x80000000)), block_start_end[0])), block_start_end[0], block_start_end[1]);
							completion_status[blockIdx.x] = smem.runflag;
							chunk_counter[blockIdx.x] = smem.chunk_counter;
						}, [&]()
						{
							atomicOr(run_flag, 0x2);
							// Write out descriptor for restart into global
							//if(threadIdx.x == 0)
							//printf("%d going for restart: %x: %d -> %d -- block row range: %d<->%d\n", blockIdx.x, smem.runflag, (smem.runflag&(~0x80000000)), smem.A_row_ids.decodeRow(smem.A_row_ids.restartRowDecode((smem.runflag & (~0x80000000)), block_start_end[0])), block_start_end[0], block_start_end[1]);
							completion_status[blockIdx.x] = smem.runflag;
							chunk_counter[blockIdx.x] = smem.chunk_counter;
						});
						smem.chunkStartOffset = chunkoff;
					}

					__syncthreads();
					if (smem.chunkStartOffset == 0xFFFFFFFF)
						return;

					smem.firstRowCount = 0;

					int num = min(smem.brokenChunkOffsetEnd - smem.brokenChunkOffsetStart, allocData);

					allocData -= num;

					// write data for this chunk to smem and write out
					for (uint32_t written = smem.brokenChunkOffsetStart; written < smem.brokenChunkOffsetEnd; written += TEMP_ITEMS_PER_BLOCK)
					{
						//store in shared for coalesced out
						#pragma unroll
						for (int i = 0; i < CombineElements; ++i)
						{
							uint32_t poffset = combinedEntries[i].memoffset();
							if (combinedEntries[i].isResult() && poffset >= written && poffset < written + TEMP_ITEMS_PER_BLOCK)
							{
								uint32_t pwrite = poffset - written;
								INDEX_TYPE col = (combinedEntries[i].index & ((1u << colBits) - 1)) + smem.minCol;
								typename ARowStorage::EncodedRowType row = (combinedEntries[i].index >> colBits) + smem.minRow;
								smem.current_col_ids[pwrite] = col;
								smem.current_row_ids[pwrite] = row;
								smem.current_output[pwrite] = combinedEntries[i].value;
								smem.rowCounts[pwrite] = combinedEntries[i].isRowend() ? combinedEntries[i].rowcount() : 0;
							}
						}

						__syncthreads();

						#pragma unroll
						for (int i = 0; i < RETAIN_ELEMENTS_PER_THREAD; ++i)
						{
							//write out
							INDEX_TYPE rid;
							int writeout = written + i * THREADS + threadIdx.x - smem.brokenChunkOffsetStart;
							if (writeout < num)
							{
								typename SEMIRING_t::output_t* valstart = Chunk::cast(chunks, smem.chunkStartOffset)->values_direct(num);
								INDEX_TYPE* indexstart = Chunk::cast(chunks, smem.chunkStartOffset)->indices_direct(num);
								valstart[writeout] = smem.current_output[i * THREADS + threadIdx.x];
								indexstart[writeout] = smem.current_col_ids[i * THREADS + threadIdx.x];
								rid = smem.current_row_ids[i * THREADS + threadIdx.x];
								// if (rid >= rows) {
								// 	printf("%d %d rid bad row read %d \n",blockIdx.x, threadIdx.x , rid);
								// }
								if (smem.A_row_ids.decodeRow(rid) == 1878)
								{
									/*if(smem.current_col_ids[i * THREADS + threadIdx.x] == 0)
										printf("ChunkStartOffset: %u with num: %u\n", smem.chunkStartOffset, num);
									printf("Row %u: %u\n", smem.A_row_ids.decodeRow(rid), smem.current_col_ids[i * THREADS + threadIdx.x]);*/
								}
								if (writeout == num - 1)
								{
									smem.maxBrokenChunkRow = rid;
									smem.lastRowCount = smem.rowCounts[i * THREADS + threadIdx.x];
								}
							}
							else
								rid = std::numeric_limits<INDEX_TYPE>::max();

							uint32_t rcount = smem.rowCounts[i * THREADS + threadIdx.x];
							if (rcount != 0 && rid != std::numeric_limits<INDEX_TYPE>::max())
							{
								//write row count
								if (smem.firstRowCount == 0 && rid == smem.current_row_ids[0])
								{
									smem.minBrokenChunkRow = rid;
									smem.firstRowCount = rcount;
								}
								if ((smem.A_row_ids.decodeRow(rid) == 1878) /*|| (smem.A_row_ids.decodeRow(rid) == 11614) || (smem.A_row_ids.decodeRow(rid) == 14759) || (smem.A_row_ids.decodeRow(rid) == 14767) || (smem.A_row_ids.decodeRow(rid) == 11125)*/)
									printf("Adding count: %u to row %u\n", rcount, (smem.A_row_ids.decodeRow(rid)));
								atomicAdd(output_row_count + smem.A_row_ids.decodeRow(rid), rcount);
							}
						}
						__syncthreads();
					}

					// last is shared if we are in a broken chunk (allocData > 0) or if we write out the last completely
					bool shared_last = (allocData > 0 || tempData == num) && smem.minBrokenChunkRow != smem.maxBrokenChunkRow;
					if (threadIdx.x < (shared_last ? 2 : 1))
					{
						
						//write header
						/*if(smem.A_row_ids.decodeRow(smem.minBrokenChunkRow) <= 2605 && smem.A_row_ids.decodeRow(smem.maxBrokenChunkRow) >= 2605)*/
							/*printf("%d %d broken writing header: %d<->%d  .%d %d.  (%d/%d/%d)\n", blockIdx.x, threadIdx.x,
							smem.A_row_ids.decodeRow(smem.minBrokenChunkRow), smem.A_row_ids.decodeRow(smem.maxBrokenChunkRow), smem.firstRowCount, smem.lastRowCount, allocData, num, tempData);*/

						Chunk::place(chunks, smem.chunkStartOffset, num, smem.A_row_ids.decodeRow(smem.minBrokenChunkRow), smem.firstRowCount, smem.lastRowCount, (static_cast<ChunkSortType>(blockIdx.x) << ChunkSortingBits) | (smem.chunk_counter + threadIdx.x));

						bool minrow = threadIdx.x == 0 && smem.minBrokenChunkRow != smem.maxBrokenChunkRow;
						uint32_t r = smem.A_row_ids.decodeRow(minrow ? smem.minBrokenChunkRow : smem.maxBrokenChunkRow);
						Chunk* c = Chunk::cast(chunks, smem.chunkStartOffset);

						/*printf("%d %d adding shared row: %d first: %d - for encoded rows %d %d\n", blockIdx.x, threadIdx.x, r, minrow, smem.minBrokenChunkRow, smem.maxBrokenChunkRow);*/
						addPotentiallySharedRow(r, c, minrow, output_row_list_heads, shared_rows_tracker, shared_rows_alloc);
						atomicAdd(output_row_chunk_count + r, 1);

						// set new local restart information
						smem.runflag = tempData == num ? consumedwork : (0x80000000 | (smem.A_row_ids.restartRowEncode(smem.maxBrokenChunkRow, block_start_end[0]) + 1));

						//printf("%d %d updating tempData %d -= %d -> %d  and temp offset: %d\n", blockIdx.x, threadIdx.x, tempData, num, tempData - num, num % TEMP_ITEMS_PER_BLOCK);

						smem.brokenChunkOffsetStart = smem.brokenChunkOffsetEnd;

						//reset count
						tempData = tempData - num;
						tempOffset = num % TEMP_ITEMS_PER_BLOCK;
						if (threadIdx.x == 0)
							smem.chunk_counter += (shared_last ? 2 : 1);
					}
				}
			}
			else
			{
				//if (threadIdx.x == 0)
				//	printf("%d %d normal chunk for %d-%d\n", blockIdx.x, threadIdx.x, smem.minRow, smem.maxRow);
				if (threadIdx.x == 0)
				{
					uint32_t chunkoff = completeChunkAlloc<typename SEMIRING_t::output_t, INDEX_TYPE>(allocData, chunks, chunk_alloc, chunk_size, chunks_pointers, chunk_pointer_alloc, chunk_pointer_sizes, chunk_pointer_pos,
						[&]()
						{
							atomicOr(run_flag, 0x1);
							// Write out descriptor for restart into global
							completion_status[blockIdx.x] = smem.runflag;
							chunk_counter[blockIdx.x] = smem.chunk_counter;
						},
						[&]()
						{
							atomicOr(run_flag, 0x2);
							// Write out descriptor for restart into global
							completion_status[blockIdx.x] = smem.runflag;
							chunk_counter[blockIdx.x] = smem.chunk_counter;
						});

					smem.chunkStartOffset = chunkoff;
				}
				__syncthreads();
				if (smem.chunkStartOffset == 0xFFFFFFFF)
					return;

				//      every first element in row -> run prefix sum to determine number of entries in row
				//        not first or last, directly set count
				//        first and last row for potential overlap
				//          atomicMax at count
				//          if 0 before -> alloc list element and atomic exchange with head and write info + next pointer into list
				//            if the head was non-zero (second list element, add shared row entry:
				//              atomicAdd for alloc and write row
				//    : add first row in chunk to beginning of chunk
				//      add numentires to chunk
				//      add offset to data and column ids to chunk info
				//      this info can be updated for shared rows when we extract stuff :)

				smem.firstRowCount = 0;
				//RowCounter rc(smem.rowcounterMem);

				for (uint32_t written = 0; written < tempData; written += TEMP_ITEMS_PER_BLOCK)
				{
					//store in shared for coalesced out
					#pragma unroll
					for (int i = 0; i < CombineElements; ++i)
					{
						uint32_t poffset = combinedEntries[i].memoffset();
						if (combinedEntries[i].isResult() && poffset >= written && poffset < written + TEMP_ITEMS_PER_BLOCK)
						{
							uint32_t pwrite = poffset - written;
							INDEX_TYPE col = (combinedEntries[i].index & ((1u << colBits) - 1)) + smem.minCol;
							typename ARowStorage::EncodedRowType row = (combinedEntries[i].index >> colBits) + smem.minRow;
							//if (col > 21198119)
							//	printf("%d %d merge fucked up col: %d: %llx %d+d\n", blockIdx.x, threadIdx.x, col, combinedEntries[i].index, uint32_t(combinedEntries[i].index & ((1u << colBits) - 1)), smem.minCol);
							smem.current_col_ids[pwrite] = col;
							smem.current_row_ids[pwrite] = row;
							smem.current_output[pwrite] = combinedEntries[i].value;

							//printf("%d %d entry %d: %d/%d %f\n", blockIdx.x, threadIdx.x, poffset, row, col, combinedEntries[i].value);

							/*if (col < smem.minCol || col > smem.maxCol || row < smem.minRow || row > smem.maxRow || row >= rows || col >= rows)
							{
								printf("%d %d bad entry: %llx %d = %d + %d  (%d %d) %d (%d %d) - %d\n", blockIdx.x, threadIdx.x, combinedEntries[i].index, row, smem.minRow, (combinedEntries[i].index >> colBits), smem.minRow, smem.maxRow, col, smem.minCol, smem.maxCol, rows);
								__trap();
							}*/

							smem.rowCounts[pwrite] = combinedEntries[i].isRowend() ? combinedEntries[i].rowcount() : 0;
						}
					}

					__syncthreads();

					#pragma unroll
					for (int i = 0; i < RETAIN_ELEMENTS_PER_THREAD; ++i)
					{
						//write out
						INDEX_TYPE rid;
						uint32_t writeout = written + i * THREADS + threadIdx.x;
						if (writeout < allocData)
						{
							typename SEMIRING_t::output_t* valstart = Chunk::cast(chunks, smem.chunkStartOffset)->values_direct(allocData);
							INDEX_TYPE* indexstart = Chunk::cast(chunks, smem.chunkStartOffset)->indices_direct(allocData);
							valstart[writeout] = smem.current_output[i * THREADS + threadIdx.x];
							indexstart[writeout] = smem.current_col_ids[i * THREADS + threadIdx.x];
							rid = smem.current_row_ids[i * THREADS + threadIdx.x];
							//printf("row id %d", smem.current_row_ids[i * THREADS + threadIdx.x]);
							//fixme?
							// if ((rid >= rows || rid < 0) && rid != std::numeric_limits<INDEX_TYPE>::max() )
							// 	printf("%d %d fffffffffffitting rid: %d %d allocdata: %d, %d\n", blockIdx.x, threadIdx.x, rid, rows,allocData,  std::numeric_limits<INDEX_TYPE>::max() - rid );

						}
						else
						{
							rid = std::numeric_limits<INDEX_TYPE>::max();
							//fixme: suspicious  if theres an error, I thought I discarded these changes
							// if ((rid >= rows || rid < 0) && rid != std::numeric_limits<INDEX_TYPE>::max() )
							// 	printf("%d %d Eeeeeeeeeeeeeenonfitting rid: %d %d allocdata: %d\n", blockIdx.x, threadIdx.x, rid, rows,allocData);
						}

						uint32_t rcount = smem.rowCounts[i * THREADS + threadIdx.x];
						if (rcount != 0 && rid < rows)
						{
							//write row count
							//if (written + threadIdx.x == tempData - 1)
							//	smem.lastRowCount = rcount;
							if (smem.firstRowCount == 0 && rid == smem.current_row_ids[0])
								smem.firstRowCount = rcount;

							// if (rid >= rows || rid < 0)
								// printf("%d %d nonfitting rid: %d %d allocdata: %d\n", blockIdx.x, threadIdx.x, rid, rows,allocData);
							auto b = smem.A_row_ids.decodeRow(rid);
							atomicAdd(output_row_count + b, rcount);
						}
					}
					__syncthreads();
				}

				bool shared_last = tempData == allocData && smem.minRow != smem.maxRow;
				if (threadIdx.x < (shared_last ? 2 : 1))
				{
					////write header
					//if (smem.A_row_ids.decodeRow(smem.minRow) >= 2605 && smem.A_row_ids.decodeRow(smem.maxRow) <= 2605)
					//printf("%d %d writing header: %d<->%d  .%d %d.  (%d/%d)\n", blockIdx.x, threadIdx.x,
					//	smem.A_row_ids.decodeRow(smem.minRow), smem.A_row_ids.decodeRow(smem.maxRow), smem.firstRowCount, smem.lastRowCount, allocData, tempData);
					Chunk::place(chunks, smem.chunkStartOffset, allocData, smem.A_row_ids.decodeRow(smem.minRow), smem.firstRowCount, smem.lastRowCount, (static_cast<ChunkSortType>(blockIdx.x) << ChunkSortingBits) | (smem.chunk_counter + threadIdx.x));


					bool minrow = threadIdx.x == 0 && smem.minRow != smem.maxRow;
					uint32_t r = smem.A_row_ids.decodeRow(minrow ? smem.minRow : smem.maxRow);
					Chunk* c = Chunk::cast(chunks, smem.chunkStartOffset);

					//printf("%6d %4d adding shared row: %6d first: %d with %5d \n", blockIdx.x, threadIdx.x, r, minrow, minrow ? smem.firstRowCount : smem.lastRowCount);
					addPotentiallySharedRow(r, c, minrow, output_row_list_heads, shared_rows_tracker, shared_rows_alloc);
					atomicAdd(output_row_chunk_count + r, 1);

					// set new local restart information
					smem.runflag = tempData == allocData ? consumedwork : (0x80000000 | (smem.A_row_ids.restartRowEncode(smem.maxRow, block_start_end[0])));

					//printf("%d %d setting temp run flag to: %d == %d ? %d : (0x80000000 | %d) - %d -> %x %d\n", blockIdx.x, threadIdx.x, tempData, allocData, consumedwork, (smem.maxRow - block_start_end[0]), smem.maxRow, smem.runflag, smem.runflag & (~0x80000000));

					//reset count
					tempData = tempData - allocData;
					tempOffset = allocData % TEMP_ITEMS_PER_BLOCK;
					if (threadIdx.x == 0)
						smem.chunk_counter += (shared_last ? 2 : 1);
				}
			}
		}
		else
		{
			// directly store to shared
			#pragma unroll
			for (int i = 0; i < CombineElements; ++i)
			{
				if (combinedEntries[i].isResult())
				{
					uint32_t poffset = combinedEntries[i].memoffset();
					smem.current_col_ids[poffset] = (combinedEntries[i].index  & ((1u << colBits) - 1)) + smem.minCol;
					smem.current_row_ids[poffset] = (combinedEntries[i].index >> colBits) + smem.minRow;
					smem.current_output[poffset] = combinedEntries[i].value;
				}
			}

			//if (threadIdx.x == 0)
			//	printf("%d keep: %d->%d %d\n", blockIdx.x, smem.minRow, smem.maxRow, tempData);
			tempOffset = 0;
		}
		__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		// All done
		completion_status[blockIdx.x] = 0xFFFFFFFF;
	}
}


template<uint32_t NNZ_PER_THREAD, uint32_t THREADS, uint32_t BLOCKS_PER_MP, uint32_t INPUT_ELEMENTS_PER_THREAD, uint32_t RETAIN_ELEMENTS_PER_THREAD, uint32_t MERGE_MAX_PATH_OPTIONS, typename VALUE_TYPE1, typename VALUE_TYPE2, typename VALUE_TYPE3, typename INDEX_TYPE, typename OFFSET_TYPE, int SORT_TYPE_MODE,
        typename T, typename U, typename Label,
        typename SEMIRING_t>
        void AcSpGEMMKernels::h_computeSpgemmPart(
	const typename SEMIRING_t::leftInput_t* valA, const INDEX_TYPE* indicesA, const OFFSET_TYPE* __restrict offsetsA,
	/*fixme const T2 -> */const typename SEMIRING_t::rightInput_t* __restrict valB, const INDEX_TYPE* __restrict indicesB, const OFFSET_TYPE* __restrict offsetsB,
	const uint32_t* __restrict startingIdsA, uint32_t nnz, uint32_t rows,
	uint32_t* chunks, uint32_t* chunk_alloc, uint32_t* chunk_worst_case, uint32_t chunk_size,
	void** chunks_pointers, uint32_t* chunk_pointer_alloc, uint32_t chunk_pointer_sizes,
	OFFSET_TYPE* output_row_count, void** output_row_list_heads, uint32_t* output_row_chunk_count,
	uint32_t* shared_rows_tracker, uint32_t* shared_rows_alloc, float expected_row_overlap, float expected_row_overlap_inv,
	uint32_t* run_flag, uint32_t* completion_status, uint32_t* chunk_counter, uint32_t* chunk_pointer_pos, SEMIRING_t semiring)
{
	computeSpgemmPart< NNZ_PER_THREAD, THREADS, BLOCKS_PER_MP, INPUT_ELEMENTS_PER_THREAD, RETAIN_ELEMENTS_PER_THREAD, MERGE_MAX_PATH_OPTIONS, typename SEMIRING_t::leftInput_t, typename SEMIRING_t::rightInput_t, typename SEMIRING_t::output_t, INDEX_TYPE, OFFSET_TYPE, SORT_TYPE_MODE,  T,  U,  Label,SEMIRING_t> <<<gridDim, blockDim>>>
		(valA, indicesA, offsetsA, valB, indicesB, offsetsB, startingIdsA, nnz, rows, chunks, chunk_alloc, chunk_worst_case, chunk_size,
		chunks_pointers, chunk_pointer_alloc, chunk_pointer_sizes, output_row_count, output_row_list_heads, output_row_chunk_count,
		shared_rows_tracker, shared_rows_alloc, expected_row_overlap, expected_row_overlap_inv, run_flag, completion_status, chunk_counter, chunk_pointer_pos, semiring);
}


#define GPUCompressedMatrixMatrixMultiplyGEMM(TYPE, THREADS, BLOCKS_PER_MP, NNZPERTHREAD, INPUT_ELEMENTS_PER_THREAD, RETAIN_ELEMENTS_PER_THREAD, MERGE_MAX_CHUNKS, MERGE_MAX_PATH_OPTIONS) \
	template void h_computeSpgemmPart<NNZPERTHREAD, THREADS, BLOCKS_PER_MP, INPUT_ELEMENTS_PER_THREAD, RETAIN_ELEMENTS_PER_THREAD, MERGE_MAX_PATH_OPTIONS, TYPE, TYPE, TYPE, uint32_t, uint32_t, 0> \
	(const TYPE* valA, const uint32_t* indicesA, const uint32_t* __restrict offsetsA, \
	const TYPE* __restrict valB, const uint32_t* __restrict indicesB, const uint32_t* __restrict offsetsB, \
	const uint32_t* __restrict startingIdsA, uint32_t nnz, uint32_t rows,\
	uint32_t* chunks, uint32_t* chunk_alloc, uint32_t* chunk_worst_case, uint32_t chunk_size, \
	void** chunks_pointers, uint32_t* chunk_pointer_alloc, uint32_t chunk_pointer_sizes, \
	uint32_t* output_row_count, void** output_row_list_heads, uint32_t* output_row_chunk_count,\
	uint32_t* shared_rows_tracker, uint32_t* shared_rows_alloc, float expected_row_overlap, float expected_row_overlap_inv, \
	uint32_t* run_flag, uint32_t* completion_status, uint32_t* chunk_counter, uint32_t* chunk_pointer_pos); \
	template void h_computeSpgemmPart<NNZPERTHREAD, THREADS, BLOCKS_PER_MP, INPUT_ELEMENTS_PER_THREAD, RETAIN_ELEMENTS_PER_THREAD, MERGE_MAX_PATH_OPTIONS, TYPE, TYPE, TYPE, uint32_t, uint32_t, 1> \
	(const TYPE* valA, const uint32_t* indicesA, const uint32_t* __restrict offsetsA, \
	const TYPE* __restrict valB, const uint32_t* __restrict indicesB, const uint32_t* __restrict offsetsB, \
	const uint32_t* __restrict startingIdsA, uint32_t nnz, uint32_t rows, \
	uint32_t* chunks, uint32_t* chunk_alloc, uint32_t* chunk_worst_case, uint32_t chunk_size, \
	void** chunks_pointers, uint32_t* chunk_pointer_alloc, uint32_t chunk_pointer_sizes, \
	uint32_t* output_row_count, void** output_row_list_heads, uint32_t* output_row_chunk_count, \
	uint32_t* shared_rows_tracker, uint32_t* shared_rows_alloc, float expected_row_overlap, float expected_row_overlap_inv, \
	uint32_t* run_flag, uint32_t* completion_status, uint32_t* chunk_counter, uint32_t* chunk_pointer_pos); \
	template void h_computeSpgemmPart<NNZPERTHREAD, THREADS, BLOCKS_PER_MP, INPUT_ELEMENTS_PER_THREAD, RETAIN_ELEMENTS_PER_THREAD, MERGE_MAX_PATH_OPTIONS, TYPE, TYPE, TYPE, uint32_t, uint32_t, 2> \
	(const TYPE* valA, const uint32_t* indicesA, const uint32_t* __restrict offsetsA, \
	const TYPE* __restrict valB, const uint32_t* __restrict indicesB, const uint32_t* __restrict offsetsB, \
	const uint32_t* __restrict startingIdsA, uint32_t nnz, uint32_t rows, \
	uint32_t* chunks, uint32_t* chunk_alloc, uint32_t* chunk_worst_case, uint32_t chunk_size, \
	void** chunks_pointers, uint32_t* chunk_pointer_alloc, uint32_t chunk_pointer_sizes, \
	uint32_t* output_row_count, void** output_row_list_heads, uint32_t* output_row_chunk_count, \
	uint32_t* shared_rows_tracker, uint32_t* shared_rows_alloc, float expected_row_overlap, float expected_row_overlap_inv, \
	uint32_t* run_flag, uint32_t* completion_status, uint32_t* chunk_counter, uint32_t* chunk_pointer_pos);
