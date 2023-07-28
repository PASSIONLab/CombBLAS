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
 * MergeGeneralized.cuh
 *
 * ac-SpGEMM
 *
 * Authors: Daniel Mlakar, Markus Steinberger, Martin Winter
 *------------------------------------------------------------------------------
*/

#pragma once

#include "MultiplyKernels.h"

#define ELEMENT_TO_SEARCH 10198

//binary search for an element in an array; returns the number of elements that are smaller or equal than the one
// we are looking for
template<typename DATA_TYPE, typename SIZE_TYPE>
__device__ __forceinline__ SIZE_TYPE binarySearch(const DATA_TYPE* start, const SIZE_TYPE count, const DATA_TYPE target)
{
	if (count == 0)
		return  0;

	SIZE_TYPE lower_bound = 0;
	SIZE_TYPE upper_bound = count - 1;
	SIZE_TYPE tmp_loc;

	if (target < start[lower_bound])
		return 0;

	if (target > start[count - 1])
		return count;

	while (lower_bound <= upper_bound)
	{
		tmp_loc = (lower_bound + upper_bound) >> 1;

		if (target < start[tmp_loc])
		{
			upper_bound = tmp_loc - 1;
		}
		else if (target > start[tmp_loc])
		{
			lower_bound = tmp_loc + 1;
		}
		else
		{
			//we can have multiple target entries - let's skip them until we point after the last target
			while (tmp_loc < count && start[tmp_loc] == target)
				++tmp_loc;

			return tmp_loc;
		}
	}

	return lower_bound;	//element not found; return id of first element larger than target
}

// samples the interval [lower, upper] s.t. each of the num_samples sub intervals is approximately the same size
template<typename INDEX_TYPE>
__device__ __forceinline__ INDEX_TYPE getSample(INDEX_TYPE lower, INDEX_TYPE upper, uint32_t num_samples, uint32_t sample_point)
{
	float alpha = static_cast<float>(sample_point + 1) / num_samples;
	return (1 - alpha) * lower + alpha * upper;
}

template<uint32_t THREADS>
__device__ __forceinline__ uint32_t samplePosition(uint32_t minID, uint32_t maxID, int position = threadIdx.x + 1)
{
	return (divup<uint32_t>((maxID - minID), THREADS)) * (position);
}

template <typename INDEX_TYPE, uint32_t THREADS, uint32_t MERGE_MAX_CHUNKS>
__device__ __forceinline__ uint32_t sampling(typename cub::BlockScan<uint32_t, THREADS>::TempStorage& atomicMaxScanTemp, 
	INDEX_TYPE minID, INDEX_TYPE maxID, int numberChunks, uint32_t* max_sampling_category,
	uint32_t (&sample_offsets)[THREADS], const INDEX_TYPE *__restrict__(&chunkIndices)[MERGE_MAX_CHUNKS], uint32_t (&chunkElementCount)[MERGE_MAX_CHUNKS])
{
	uint32_t sampling_step = divup<uint32_t>((maxID - minID), THREADS);
	uint32_t my_sample_offset = 0;
	for (auto round = 0; round < numberChunks; ++round)
	{
		// Reset intermediary offset
		if (threadIdx.x == 0)
			*max_sampling_category = 0;
		sample_offsets[threadIdx.x] = 0;
		__syncthreads();
		uint32_t count = chunkElementCount[round];
		for (int i = threadIdx.x; i < count - 1; i += THREADS)
		{
			// Fetch column Ids
			INDEX_TYPE columnIndex = chunkIndices[round][i];
			INDEX_TYPE nextColumnIndex = chunkIndices[round][i + 1];
			INDEX_TYPE sampling_category = (columnIndex > 0) ? (columnIndex - 1) / sampling_step : 0;
			INDEX_TYPE next_sampling_category = (nextColumnIndex - 1) / sampling_step;
			if (sampling_category != next_sampling_category)
			{
				if (sampling_category < THREADS)
					sample_offsets[sampling_category] = i + 1;
				atomicMax(max_sampling_category, sampling_category);
			}
		}
		__syncthreads();

		// Set max
		if (*max_sampling_category < (THREADS - 1))
			sample_offsets[*max_sampling_category + 1] = count;
		__syncthreads();

		uint32_t sample_value[1] = { sample_offsets[threadIdx.x] };
		// Propagate Max
		cub::BlockScan<uint32_t, THREADS>(atomicMaxScanTemp).InclusiveScan(sample_value, sample_value, cub::Max());
		__syncthreads();
		// Write to global sample offsets
		my_sample_offset += sample_value[0];
		__syncthreads();
	}
	return my_sample_offset;
}


const int GlobalPathOffset = 0;
const int MinColumnOffset = 1;
const int MaxColumnOffset = 2;
const int ElementsHandledOffset = 3;

// #########################################################################################
//
//  Generalized Case
//
// #########################################################################################
template<uint32_t NNZ_PER_THREAD, uint32_t THREADS, uint32_t BLOCKS_PER_MP, uint32_t INPUT_ELEMENTS_PER_THREAD, uint32_t RETAIN_ELEMENTS_PER_THREAD, uint32_t MERGE_MAX_CHUNKS, uint32_t MERGE_MAX_PATH_OPTIONS, typename VALUE_TYPE, typename INDEX_TYPE, typename OFFSET_TYPE,  typename T, typename U, typename Label,
        typename SEMIRING_t>
__global__ void __launch_bounds__(THREADS, BLOCKS_PER_MP)
mergeSharedRowsGeneralized(const uint32_t* __restrict blockOffsets, const uint32_t* __restrict sharedRows, void** output_row_list_heads,
	OFFSET_TYPE* output_row_count,
	uint32_t* chunks, uint32_t* chunk_alloc, uint32_t* chunk_pre_alloc, uint32_t chunk_size,
	void** chunks_pointers, uint32_t* chunk_pointer_alloc, uint32_t chunk_pointer_sizes,
	uint32_t* run_flag, uint32_t* restart_completion, uint32_t* shared_rows_handled,
	uint32_t* restart_sampleOffs, uint32_t* restart_chunkElementsConsumedAndPath, uint32_t restart_offset, uint32_t* chunk_pointer_pos, SEMIRING_t semiring)
{
	static_assert(2 * INPUT_ELEMENTS_PER_THREAD * THREADS >= MERGE_MAX_CHUNKS, "Too many elements per column possible now!");

	using Chunk = ::Chunk<typename SEMIRING_t::output_t, INDEX_TYPE>;

    using DirectChunk = ::DirectChunk<typename SEMIRING_t::leftInput_t, typename SEMIRING_t::rightInput_t, INDEX_TYPE>;

    const uint32_t ELEMENTS_PER_THREAD = 2 * INPUT_ELEMENTS_PER_THREAD;
	using SortAndCombiner = SortAndCombine<uint32_t, typename SEMIRING_t::output_t, THREADS, ELEMENTS_PER_THREAD>;
	using ScanCombinerEntry = typename SortAndCombiner::ScanCombinerEntry;
	typedef cub::BlockScan<uint32_t, THREADS> SimpleScanT;
	const uint32_t LengthSamplesPerThread = (MERGE_MAX_CHUNKS + THREADS - 1) / THREADS;
	using SingleLoadWorkDistribution = WorkDistribution<THREADS, LengthSamplesPerThread>;
	using IndexSorter = cub::BlockRadixSort<ChunkSortType, THREADS, LengthSamplesPerThread, uint32_t>;
	
	using LEFT_T = typename SEMIRING_t::leftInput_t;
	using RIGHT_t = typename SEMIRING_t::rightInput_t;
	using OUT_t = typename SEMIRING_t::output_t;

	struct SMem
	{
		uint32_t runflag, restart/*, max_sampling_category*/;
		uint32_t numSharedRow;
		int numChunks;
		INDEX_TYPE maxColumnIdRow, currentMinColumnIdRow, currentMaxColumnIdRow;
		int sumOut;
		uint32_t completed;
		uint32_t longChunkOffset;
		INDEX_TYPE globalPath;
		INDEX_TYPE elementsHandled;



		const INDEX_TYPE* __restrict chunkIndices[MERGE_MAX_CHUNKS];
		Either<const RIGHT_t* ,  const OUT_t* >  chunkValues[MERGE_MAX_CHUNKS];
		LEFT_T multiplier[MERGE_MAX_CHUNKS];
		uint32_t chunkElementCount[MERGE_MAX_CHUNKS];
		INDEX_TYPE sample_offsets[THREADS];
		INDEX_TYPE elementsInChunkConsumed[MERGE_MAX_CHUNKS];
		uint32_t current_path_elements[MERGE_MAX_CHUNKS];
		
		// Used for sorting
		uint32_t indexing[MERGE_MAX_CHUNKS];

		union {
			struct
			{
				ChunkSortType sort_keys[MERGE_MAX_CHUNKS];
				typename IndexSorter::TempStorage indexptrtempmem;
			};
			
			struct {
				typename SingleLoadWorkDistribution::SharedMemT single_workdistributionMem;
				typename SingleLoadWorkDistribution::SharedTempMemT single_workdistributionTempMem;
				typename SingleLoadWorkDistribution:: template SharedTempMemOutT<ELEMENTS_PER_THREAD>  single_workdistributionTempMemOutFull;
			};

			typename SortAndCombiner::SMem single_sAndCMem;

			struct {
                typename SEMIRING_t::output_t longOutDataBuffer[THREADS];
				INDEX_TYPE longOutIndexBuffer[THREADS];
			};
		};
	
	};

	__shared__ SMem smem;

	//determine the block's offset
	if (threadIdx.x == 0)
	{
		uint32_t shared_handled = shared_rows_handled[(blockIdx.x + restart_offset)];
		smem.numSharedRow = 1 - shared_handled;
		smem.runflag = *run_flag;
		smem.restart = restart_completion[(blockIdx.x + restart_offset)];
		smem.sumOut = (smem.restart > RESTART_FIRST_ITERATION) ? output_row_count[sharedRows[blockIdx.x]] : 0;
	}
	__syncthreads();

	// Already handled
	if (smem.numSharedRow == 0)
		return;

	__syncthreads();

	if (threadIdx.x == 0)
	{
		//Get the one chunk that has elements of the block's row
		uint64_t chunk = reinterpret_cast<uint64_t>(output_row_list_heads[sharedRows[blockIdx.x]]);
		// DEBUG
		// if (sharedRows[blockIdx.x] == ROW_TO_INVESTIGATE)
		// 	printf("Row %d in Generalized\n", sharedRows[blockIdx.x]);
		// DEBUG
		uint32_t chunk_counter = 0;

		smem.currentMinColumnIdRow = std::numeric_limits<INDEX_TYPE>::max();
		smem.maxColumnIdRow = 0;

		//As long as we have some chunk that has elements of the block's row keep reading
		while (chunk != 0)
		{
			INDEX_TYPE minColumnId, maxColumnId;
			bool first_row = (chunk & 2) != 0;
			//get a pointer to the current chunk
			Chunk* __restrict pChunk = reinterpret_cast<Chunk*>(chunk & 0xFFFFFFFFFFFFFFFCULL);
			uint32_t count;
			const INDEX_TYPE* pIndices;
			Either<const RIGHT_t*, const OUT_t*> pValues;
			int32_t numentries = pChunk->num_entries;
			LEFT_T multiplier;

			smem.sort_keys[chunk_counter] = pChunk->sort_key;

			if (first_row)
			{
				// only first_row chunks can be direct ones
				if (pChunk->isDirect())
				{
					DirectChunk* __restrict pDirectChunk = reinterpret_cast<DirectChunk*>(pChunk);
					count = numentries;
					pIndices = pDirectChunk->indices_direct(numentries);
					pValues = Either<const RIGHT_t*, const OUT_t*>::First(pDirectChunk->values_direct(numentries));
					multiplier = pDirectChunk->getMultiplier();
					chunk = reinterpret_cast<uint64_t>(pChunk->readNextFront());
					pDirectChunk->setFirstConsumed();
					minColumnId = pIndices[0];
					maxColumnId = pIndices[count - 1];
				}
				else
				{
					count = pChunk->firstCountCleared();
					pChunk->setFirstConsumed();
					pIndices = pChunk->indices_direct(numentries);
					pValues = Either<const RIGHT_t*, const OUT_t*>::Second(pChunk->values_direct(numentries));
					minColumnId = pIndices[0];
					maxColumnId = pIndices[count - 1];
					chunk = reinterpret_cast<uint64_t>(pChunk->readNextFront());
				}
			}
			else
			{
				count = pChunk->lastCountCleared();
				pChunk->setLastConsumed();
				uint32_t baseoffset = numentries - count;
				pIndices = pChunk->indices_direct(numentries) + baseoffset;
				pValues = Either<const RIGHT_t*, const OUT_t*>::Second(pChunk->values_direct(numentries) + baseoffset);
				minColumnId = pIndices[0];
				maxColumnId = pIndices[count - 1];
				chunk = reinterpret_cast<uint64_t>(pChunk->readNextBack());
			}

			//Update global min/max column id
			smem.currentMinColumnIdRow = min(smem.currentMinColumnIdRow, minColumnId);
			smem.maxColumnIdRow = max(smem.maxColumnIdRow, maxColumnId);
			smem.currentMaxColumnIdRow = smem.maxColumnIdRow;

			// We do not have enough memory to store more chunk info
			if (chunk_counter >= MERGE_MAX_CHUNKS)
			{
				printf("ERROR: number of chunks (%d) exceeds maximum (%d) in block: %u;\n", chunk_counter, MERGE_MAX_CHUNKS, blockIdx.x);
				__trap();
				smem.runflag = 1;
				break;
			}
			else
			{
				smem.chunkIndices[chunk_counter] = pIndices;

				smem.chunkValues[chunk_counter] = pValues;
				smem.chunkElementCount[chunk_counter] = count;
				smem.multiplier[chunk_counter] = multiplier;
			}

			++chunk_counter;
		}
		smem.numChunks = chunk_counter;
	}
	__syncthreads();

	if (smem.runflag != 0)
		return;

	// Sort chunks
	{
		ChunkSortType key[LengthSamplesPerThread];
		uint32_t value[LengthSamplesPerThread];
		for (int i = threadIdx.x; i < MERGE_MAX_CHUNKS; i += THREADS)
		{
			value[i / THREADS] = i;
#ifdef ENABLE_SORTING
			if(i < smem.numChunks)
				key[i/THREADS] = smem.sort_keys[i];
			else
				key[i / THREADS] = 0xFFFFFFFF;
#endif
		}
#ifdef ENABLE_SORTING
		IndexSorter(smem.indexptrtempmem).Sort(key, value);
#endif
		for (int i = threadIdx.x; i < MERGE_MAX_CHUNKS; i += THREADS)
		{
			smem.indexing[(threadIdx.x*LengthSamplesPerThread) + (i / THREADS)] = value[i / THREADS];
			//smem.indexing[i] = i;
		}
	}
	__syncthreads();

	int chunkWorkElements[LengthSamplesPerThread];
	//Perform the sampling
	if (smem.restart < RESTART_FIRST_ITERATION)
	{
		//determine for each thread which column id he has to look for in the chunks
		uint32_t sample = getSample(smem.currentMinColumnIdRow, smem.currentMaxColumnIdRow, THREADS, threadIdx.x);

		//warp based sampling in rounds; in round r thread i works on chunk (i+r) % n
		INDEX_TYPE my_sample_offset = 0;
		int wid = threadIdx.x / 32;
		for (auto round = 0; round < smem.numChunks; ++round)
		{
			uint32_t count = smem.chunkElementCount[smem.indexing[(wid + round) % smem.numChunks]];
			const INDEX_TYPE* pIndices = smem.chunkIndices[smem.indexing[(wid + round) % smem.numChunks]];
			//perform binary search for sample in [pIndices, pIndices + count) and accumulate sample_locations
			my_sample_offset += binarySearch(pIndices, count, sample);
		}

		//uint32_t my_sample_offset = sampling<INDEX_TYPE, THREADS, MERGE_MAX_CHUNKS>(smem.atomicMaxScanTemp, smem.currentMinColumnIdRow, smem.currentMaxColumnIdRow, smem.numChunks, &(smem.max_sampling_category), smem.sample_offsets, smem.chunkIndices, smem.chunkElementCount);

		//write the threads sample offset to shared
		smem.sample_offsets[threadIdx.x] = my_sample_offset;
		restart_sampleOffs[blockIdx.x * THREADS + threadIdx.x] = my_sample_offset;
		if (threadIdx.x == 0)
		{
			restart_chunkElementsConsumedAndPath[blockIdx.x * (MERGE_MAX_CHUNKS + helper_overhead) + MERGE_MAX_CHUNKS + MinColumnOffset] = smem.currentMinColumnIdRow;
			restart_chunkElementsConsumedAndPath[blockIdx.x * (MERGE_MAX_CHUNKS + helper_overhead) + MERGE_MAX_CHUNKS + MaxColumnOffset] = smem.currentMaxColumnIdRow;
		}
	}
	//We already restarted at least once and have done at least one iteration in the last run, hence, we have values that we want to reuse
	else
	{
		smem.sample_offsets[threadIdx.x] = restart_sampleOffs[blockIdx.x * THREADS + threadIdx.x];
		if (threadIdx.x == 0)
		{
			smem.currentMinColumnIdRow = restart_chunkElementsConsumedAndPath[blockIdx.x * (MERGE_MAX_CHUNKS + helper_overhead) + MERGE_MAX_CHUNKS + MinColumnOffset];
			smem.currentMaxColumnIdRow = restart_chunkElementsConsumedAndPath[blockIdx.x * (MERGE_MAX_CHUNKS + helper_overhead) + MERGE_MAX_CHUNKS + MaxColumnOffset];
		}
	}

	for (int i = threadIdx.x; i < MERGE_MAX_CHUNKS; i += THREADS)
	{
		smem.elementsInChunkConsumed[i] = restart_chunkElementsConsumedAndPath[blockIdx.x * (MERGE_MAX_CHUNKS + helper_overhead) + i];
	}

	__syncthreads();

	if (threadIdx.x == 0)
	{
		smem.globalPath = restart_chunkElementsConsumedAndPath[blockIdx.x * (MERGE_MAX_CHUNKS + helper_overhead) + MERGE_MAX_CHUNKS + GlobalPathOffset];
		smem.elementsHandled = restart_chunkElementsConsumedAndPath[blockIdx.x * (MERGE_MAX_CHUNKS + helper_overhead) + MERGE_MAX_CHUNKS + ElementsHandledOffset];
		smem.restart = RESTART_FIRST_ITERATION;
	}			

	//we want to wait here s.t. e.g. smem.sample_offsets is available
	__syncthreads();

	bool sampling_required{ false };
	while (true)
	{
		// Maybe resampling is required
		if (sampling_required)
		{
			if (threadIdx.x == 0)
			{
				uint32_t minColumnIdRow = smem.currentMinColumnIdRow;

				if (smem.globalPath > 0 && smem.globalPath != static_cast<INDEX_TYPE>(-1))
				{
					smem.currentMinColumnIdRow = getSample(smem.currentMinColumnIdRow, smem.currentMaxColumnIdRow, THREADS, smem.globalPath - 1);
					//smem.currentMinColumnIdRow = samplePosition<THREADS>(smem.currentMinColumnIdRow, smem.currentMaxColumnIdRow, smem.globalPath - 1);
				}

				if (minColumnIdRow == smem.currentMinColumnIdRow && smem.globalPath != static_cast<INDEX_TYPE>(-1))
				{
					smem.currentMaxColumnIdRow = (smem.currentMinColumnIdRow + smem.currentMaxColumnIdRow) >> 1;	
				}

				smem.globalPath = 0;
				restart_chunkElementsConsumedAndPath[blockIdx.x * (MERGE_MAX_CHUNKS + helper_overhead) + MERGE_MAX_CHUNKS + MinColumnOffset] = smem.currentMinColumnIdRow;
				restart_chunkElementsConsumedAndPath[blockIdx.x * (MERGE_MAX_CHUNKS + helper_overhead) + MERGE_MAX_CHUNKS + MaxColumnOffset] = smem.currentMaxColumnIdRow;
				restart_chunkElementsConsumedAndPath[blockIdx.x * (MERGE_MAX_CHUNKS + helper_overhead) + MERGE_MAX_CHUNKS + GlobalPathOffset] = smem.globalPath;
			}
			__syncthreads();
			sampling_required = false;
			
			//determine for each thread which column id he has to look for in the chunks
			uint32_t sample = getSample(smem.currentMinColumnIdRow, smem.currentMaxColumnIdRow, THREADS, threadIdx.x);

			//warp based sampling in rounds; in round r thread i works on chunk (i+r) % n
			INDEX_TYPE my_sample_offset = 0;
			int wid = threadIdx.x / 32;
			for (auto round = 0; round < smem.numChunks; ++round)
			{
				uint32_t count = smem.chunkElementCount[smem.indexing[(wid + round) % smem.numChunks]];
				const INDEX_TYPE* pIndices = smem.chunkIndices[smem.indexing[(wid + round) % smem.numChunks]];
				//perform binary search for sample in [pIndices, pIndices + count) and accumulate sample_locations
				my_sample_offset += binarySearch(pIndices, count, sample);
			}
			//write the threads sample offset to shared
			smem.sample_offsets[threadIdx.x] = my_sample_offset;

			//uint32_t my_sample_offset = sampling<INDEX_TYPE, THREADS, MERGE_MAX_CHUNKS>(smem.atomicMaxScanTemp, smem.currentMinColumnIdRow, smem.currentMaxColumnIdRow, smem.numChunks, &(smem.max_sampling_category), smem.sample_offsets, smem.chunkIndices, smem.chunkElementCount);
			restart_sampleOffs[blockIdx.x * THREADS + threadIdx.x] = my_sample_offset;
		}
		__syncthreads();

		//Decide where to perform the next cut; how many elements/columns do we want to handle now?
		// after this the variables are updated to hold the new path [start sample id, end sample id)
		bool path_boundary = false;
		bool last_path = false;
		//check whether we can handle all remaining columns now; this would be the last path
		if (smem.sample_offsets[THREADS - 1] - smem.elementsHandled <= ELEMENTS_PER_THREAD * THREADS)
		{
			if (threadIdx.x == THREADS - 1)
				last_path = true;
		}
		else
		{
			path_boundary = threadIdx.x >= smem.globalPath && threadIdx.x < THREADS - 1 &&
				smem.sample_offsets[threadIdx.x] - smem.elementsHandled <= ELEMENTS_PER_THREAD * THREADS &&
				smem.sample_offsets[threadIdx.x + 1] - smem.elementsHandled > ELEMENTS_PER_THREAD * THREADS &&
				smem.sample_offsets[threadIdx.x] - smem.elementsHandled != 0;
		}

		// If no path can be chosen as any are too large to be handled -> resample
		sampling_required = __syncthreads_and(!path_boundary && !last_path);
		if (sampling_required)
			continue;

		//the thread with the id of the last column that should be handled updates the global path boundaries
		if (path_boundary || last_path)
		{
			smem.globalPath = threadIdx.x + 1; //first sample id *not* in the current path
			smem.completed = last_path;
		}
		__syncthreads(); 

		//For each chunk: determine cutoff id using a binary search aka. determine local path
		for(int i = 0; i < LengthSamplesPerThread; ++i)
			chunkWorkElements[i] = 0;
		for (int chunk = threadIdx.x; chunk < smem.numChunks; chunk += THREADS)
		{
			const INDEX_TYPE* pIndices = smem.chunkIndices[smem.indexing[chunk]];
			uint32_t count = smem.chunkElementCount[smem.indexing[chunk]];
			//how much of this chunk did we already consume? This is at the same time the start of the next local path;
			const uint32_t prev_cutoff = smem.elementsInChunkConsumed[chunk];

			//determine how many elements of this chunk are part of the current path
			uint32_t look_for = getSample(smem.currentMinColumnIdRow, smem.currentMaxColumnIdRow, THREADS, smem.globalPath - 1);
			//uint32_t look_for = samplePosition<THREADS>(smem.currentMinColumnIdRow, smem.currentMaxColumnIdRow, smem.globalPath - 1);

			smem.current_path_elements[chunk] = (count > prev_cutoff) ? binarySearch(pIndices + prev_cutoff, count - prev_cutoff, look_for) : 0;
			//update the number of consumed elements for each chunk
			smem.elementsInChunkConsumed[chunk] += smem.current_path_elements[chunk];
			//how many elements to handle in this chunk in the current path
			chunkWorkElements[chunk / THREADS] = smem.current_path_elements[chunk];
		}
		__syncthreads();

		SingleLoadWorkDistribution:: template initialize<false>(smem.single_workdistributionMem, smem.single_workdistributionTempMem, chunkWorkElements);

		int chunk[ELEMENTS_PER_THREAD];
		int element[ELEMENTS_PER_THREAD];

		int elements = SingleLoadWorkDistribution:: template assignWorkAllThreads<false, ELEMENTS_PER_THREAD>(
			smem.single_workdistributionMem, smem.single_workdistributionTempMem, smem.single_workdistributionTempMemOutFull,
			chunk, element);

		//combine entries of the current path in shared and write them into global
		int numOut;
		// Combine entries
		ScanCombinerEntry combinedEntries[ELEMENTS_PER_THREAD];
		{
			uint32_t combIndex[ELEMENTS_PER_THREAD];
			typename SEMIRING_t::output_t data[ELEMENTS_PER_THREAD];
#pragma unroll
			for (int i = 0; i < ELEMENTS_PER_THREAD; ++i)
			{
				if (element[i] >= 0)
				{
					const INDEX_TYPE* __restrict ip = smem.chunkIndices[smem.indexing[chunk[i]]] + smem.elementsInChunkConsumed[chunk[i]] - smem.current_path_elements[chunk[i]];
					combIndex[i] = ip[element[i]];

					if ( smem.chunkValues[smem.indexing[chunk[i]]].isFirst()) {
						const RIGHT_t* dp = smem.chunkValues[smem.indexing[chunk[i]]].valFirst() + smem.elementsInChunkConsumed[chunk[i]] - smem.current_path_elements[chunk[i]];
						data[i] = semiring.multiply(smem.multiplier[smem.indexing[chunk[i]]], dp[element[i]]);

					} else {
						const OUT_t* dp = smem.chunkValues[smem.indexing[chunk[i]]].valSecond() + smem.elementsInChunkConsumed[chunk[i]] - smem.current_path_elements[chunk[i]];
						data[i] = dp[element[i]];
					}					
                }
				else
				{
					data[i] = SEMIRING_t::AdditiveIdentity();
					combIndex[i] = static_cast<INDEX_TYPE>(-1);
				}
			}
			__syncthreads();

			numOut = SortAndCombiner::combine(smem.single_sAndCMem, combIndex, data, combinedEntries,
				[](auto a, auto b) {
				return a == b;
			},
				[](auto a, auto b) {
				return true;
			}, semiring);
			// ######## DEBUG
			//if (numOut == 0 && threadIdx.x == 0)
			//{
			//	printf("%d %d oops in generalized\n", blockIdx.x, threadIdx.x);
			//}
			// ######## DEBUG
		}

		// create new chunk (could also reuse old ones if completely used up...?)
		if (threadIdx.x == 0)
		{
			uint32_t chunkoff;
			int ignored;
			if (!allocChunk<typename SEMIRING_t::output_t, INDEX_TYPE>(numOut, chunk_alloc, chunk_size, chunkoff, ignored, false))
			{
				chunkoff = static_cast<uint32_t>(-1);
				atomicOr(run_flag, 0x1);
				// Write restart state
				restart_completion[(blockIdx.x + restart_offset)] = smem.restart;
			}
			else
			{
				//need to add flag and offset for copy later (offset = s)
				uint32_t s = smem.sumOut;
				INDEX_TYPE actualrow = sharedRows[blockIdx.x];
				//write chunk pointer
				uint32_t chunk_pointer_position = atomicAdd(chunk_pointer_alloc, 1);
				if (chunk_pointer_position >= chunk_pointer_sizes)
				{
					chunkoff = static_cast<uint32_t>(-1);
					atomicOr(run_flag,0x2);
					if (chunk_pointer_position == chunk_pointer_sizes)
					{
						*chunk_pointer_pos = chunk_pointer_sizes;
					}
					restart_completion[(blockIdx.x + restart_offset)] = smem.restart;
				}
				else
				{
					//FIXME SUSPICIOUS LINE april 25 
					chunks_pointers[chunk_pointer_position] = reinterpret_cast<void*>(Chunk::place(chunks, chunkoff, numOut, actualrow, Chunk::StartingOffsetFlag | s, 0));
					//write row count
					s += numOut;
					smem.sumOut = s;
					output_row_count[actualrow] = s;
				}				
			}
			smem.longChunkOffset = chunkoff;
		}
		__syncthreads();

		if (smem.longChunkOffset == static_cast<uint32_t>(-1))
		{
			return;
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

					smem.longOutDataBuffer[pwrite] = combinedEntries[i].value;
					smem.longOutIndexBuffer[pwrite] = combinedEntries[i].index;
				}
			}
			__syncthreads();

			//write outg
			if (written + threadIdx.x < numOut)
			{
				typename SEMIRING_t::output_t* valstart = Chunk::cast(chunks, smem.longChunkOffset)->values_direct(numOut);
				INDEX_TYPE* indexstart = Chunk::cast(chunks, smem.longChunkOffset)->indices_direct(numOut);

				valstart[written + threadIdx.x] = smem.longOutDataBuffer[threadIdx.x];
				indexstart[written + threadIdx.x] = smem.longOutIndexBuffer[threadIdx.x];
			}
			__syncthreads();
		}
		for (int i = threadIdx.x; i < MERGE_MAX_CHUNKS; i += THREADS)
		{
			restart_chunkElementsConsumedAndPath[blockIdx.x * (MERGE_MAX_CHUNKS + helper_overhead) + i] = smem.elementsInChunkConsumed[i];
		}
		if (threadIdx.x == 0)
		{
			smem.elementsHandled = smem.sample_offsets[smem.globalPath - 1]; //update path start (first sample id in the path) 
			restart_chunkElementsConsumedAndPath[blockIdx.x * (MERGE_MAX_CHUNKS + helper_overhead) + MERGE_MAX_CHUNKS + GlobalPathOffset] = smem.globalPath;
			restart_chunkElementsConsumedAndPath[blockIdx.x * (MERGE_MAX_CHUNKS + helper_overhead) + MERGE_MAX_CHUNKS + ElementsHandledOffset] = smem.elementsHandled;
		}
		__syncthreads();

		// Work is done, we can stop now
		if (smem.completed)
		{
			if(smem.currentMaxColumnIdRow == smem.maxColumnIdRow)
				break;

			__syncthreads();

			if (threadIdx.x == 0)
			{
				smem.globalPath = static_cast<uint32_t>(-1);
				smem.currentMinColumnIdRow = smem.currentMaxColumnIdRow + 1;
				smem.currentMaxColumnIdRow = smem.maxColumnIdRow;
			}
			sampling_required = true;
		}

		smem.restart = RESTART_ITERATION_UNKNOWN;
		__syncthreads();
	}

	// This row is done
	if (threadIdx.x == 0)
	{
		shared_rows_handled[(blockIdx.x + restart_offset)] = 1;
	}
}

template<uint32_t NNZ_PER_THREAD, uint32_t THREADS, uint32_t BLOCKS_PER_MP, uint32_t INPUT_ELEMENTS_PER_THREAD, uint32_t RETAIN_ELEMENTS_PER_THREAD, uint32_t MERGE_MAX_CHUNKS, uint32_t MERGE_MAX_PATH_OPTIONS, typename VALUE_TYPE, typename INDEX_TYPE, typename OFFSET_TYPE,
        typename T, typename U, typename Label,
        typename SEMIRING_t>
        void  AcSpGEMMKernels::h_mergeSharedRowsGeneralized(const uint32_t* __restrict blockOffsets, const uint32_t* __restrict sharedRows, void** output_row_list_heads,
	OFFSET_TYPE* output_row_count,
	uint32_t* chunks, uint32_t* chunk_alloc, uint32_t* chunk_pre_alloc, uint32_t chunk_size,
	void** chunks_pointers, uint32_t* chunk_pointer_alloc, uint32_t chunk_pointer_sizes,
	uint32_t* run_flag, uint32_t* restart_completion, uint32_t* shared_rows_handled,
	uint32_t* restart_sampleOffs, uint32_t* restart_chunkElementsConsumedAndPath, uint32_t restart_offset, uint32_t* chunk_pointer_pos, SEMIRING_t semiring)
{
	mergeSharedRowsGeneralized<NNZ_PER_THREAD, THREADS, BLOCKS_PER_MP, INPUT_ELEMENTS_PER_THREAD, RETAIN_ELEMENTS_PER_THREAD, MERGE_MAX_CHUNKS, MERGE_MAX_PATH_OPTIONS, VALUE_TYPE, INDEX_TYPE, OFFSET_TYPE, T, U,  Label,SEMIRING_t><<<gridDim, blockDim>>>(
		blockOffsets, sharedRows, output_row_list_heads, output_row_count, chunks, chunk_alloc, chunk_pre_alloc, chunk_size,
		chunks_pointers, chunk_pointer_alloc, chunk_pointer_sizes, run_flag, restart_completion, shared_rows_handled,
		restart_sampleOffs, restart_chunkElementsConsumedAndPath, restart_offset, chunk_pointer_pos, semiring);
}


#define GPUCompressedMatrixMatrixMultiplyMergeGeneralized(TYPE, THREADS, BLOCKS_PER_MP, NNZPERTHREAD, INPUT_ELEMENTS_PER_THREAD, RETAIN_ELEMENTS_PER_THREAD, MERGE_MAX_CHUNKS, MERGE_MAX_PATH_OPTIONS) \
	template void AcSpGEMMKernels::h_mergeSharedRowsGeneralized<NNZPERTHREAD, THREADS, BLOCKS_PER_MP, INPUT_ELEMENTS_PER_THREAD, RETAIN_ELEMENTS_PER_THREAD, MERGE_MAX_CHUNKS, MERGE_MAX_PATH_OPTIONS, TYPE, uint32_t, uint32_t> \
	(const uint32_t* __restrict blockOffsets, const uint32_t* __restrict sharedRows, void** output_row_list_heads, \
	uint32_t* output_row_count, \
	uint32_t* chunks, uint32_t* chunk_alloc, uint32_t* chunk_pre_alloc, uint32_t chunk_size, \
	void** chunks_pointers, uint32_t* chunk_pointer_alloc, uint32_t chunk_pointer_sizes, \
	uint32_t* run_flag, uint32_t* restart_completion, uint32_t* shared_rows_handled, \
	uint32_t* restart_sampleOffs, uint32_t* restart_chunkElementsConsumedAndPath, uint32_t restart_offset, uint32_t* chunk_pointer_pos);

