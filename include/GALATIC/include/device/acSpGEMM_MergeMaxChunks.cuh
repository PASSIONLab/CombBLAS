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
 * MergeMaxChunks.cuh
 *
 * ac-SpGEMM
 *
 * Authors: Daniel Mlakar, Markus Steinberger, Martin Winter
 *------------------------------------------------------------------------------
*/

#pragma once

// Local includes
#include "MultiplyKernels.h"
#include "../meta_utils.h"
#include <typeinfo>       // operator typeid


#include <iostream>

#define DIVISION_FACTOR 2





// #########################################################################################
// Resampling
//
	template <typename INDEX_TYPE, uint32_t MERGE_MAX_CHUNKS, uint32_t MERGE_MAX_PATH_OPTIONS>
__device__ __forceinline__ void printSampling(const uint32_t* __restrict sharedRows, int numChunks, INDEX_TYPE (&id_samples)[MERGE_MAX_CHUNKS][MERGE_MAX_PATH_OPTIONS],
	int row_index)
{
	if (sharedRows[blockIdx.x] == row_index && threadIdx.x == 0)
	{
		for (int i = 0; i < numChunks*MERGE_MAX_PATH_OPTIONS; ++i)
		{
			if (i % MERGE_MAX_PATH_OPTIONS == 0)
				printf("\n");
			printf("%u ", id_samples[i / MERGE_MAX_PATH_OPTIONS][i % MERGE_MAX_PATH_OPTIONS]);
		}
		printf("\n");
	}
}

__device__ __forceinline__ void printInvalidPath(const uint32_t* __restrict sharedRows)
{
	if (threadIdx.x == 0)
	{
		printf("%u\n", sharedRows[blockIdx.x]);
	}
}

__device__ __forceinline__ void printCountPerSampling(const uint32_t* __restrict sharedRows, uint32_t outputCount, uint32_t sampleID, uint32_t UpperBound, uint32_t row)
{
	if (outputCount < UpperBound && sharedRows[blockIdx.x] == row)
	{
		printf("Thread: %u  --  Outputcount: %u -- SampleID: %u\n", threadIdx.x, outputCount, sampleID);
	}
}





// #########################################################################################
//
//  Max Chunks Case
//
// #########################################################################################
template<uint32_t NNZ_PER_THREAD, uint32_t THREADS, uint32_t BLOCKS_PER_MP, uint32_t INPUT_ELEMENTS_PER_THREAD, uint32_t RETAIN_ELEMENTS_PER_THREAD, uint32_t MERGE_MAX_CHUNKS, uint32_t MERGE_MAX_PATH_OPTIONS, typename VALUE_TYPE, typename INDEX_TYPE, typename OFFSET_TYPE,
         typename T, typename U, typename Label,
        typename SEMIRING_t>
        __global__ void __launch_bounds__(THREADS, BLOCKS_PER_MP)
mergeSharedRowsMaxChunks(const uint32_t* __restrict blockOffsets, const uint32_t* __restrict sharedRows, void** output_row_list_heads,
	OFFSET_TYPE* output_row_count,
	uint32_t* chunks, uint32_t* chunk_alloc, uint32_t* chunk_pre_alloc, uint32_t chunk_size,
	void** chunks_pointers, uint32_t* chunk_pointer_alloc, uint32_t chunk_pointer_sizes,
	uint32_t* run_flag, uint32_t* restart_completion, uint32_t* shared_rows_handled,
	INDEX_TYPE** restart_chunkIndices, Either<typename SEMIRING_t::rightInput_t*, typename SEMIRING_t::output_t*>* restart_chunkValues, typename SEMIRING_t::leftInput_t* restart_multiplier, uint32_t* restart_chunkElementCount, uint32_t restart_offset, uint32_t* restart_num_chunks, uint32_t* chunk_pointer_pos,
	SEMIRING_t  semiring)
{
	using LEFT_T = typename SEMIRING_t::leftInput_t;
	using RIGHT_t = typename SEMIRING_t::rightInput_t;

	using OUT_t = typename SEMIRING_t::output_t;

	using Chunk = ::Chunk<typename SEMIRING_t::output_t, INDEX_TYPE>;

    using DirectChunk = ::DirectChunk<LEFT_T,RIGHT_t, INDEX_TYPE>;
	
	const uint32_t ELEMENTS_PER_THREAD = 2 * INPUT_ELEMENTS_PER_THREAD;
	using SingleLoadWorkDistribution = WorkDistribution<THREADS>;
	using SortAndCombiner = SortAndCombine<uint32_t, typename SEMIRING_t::output_t, THREADS, ELEMENTS_PER_THREAD>;
	using ScanCombinerEntry = typename SortAndCombiner::ScanCombinerEntry;
	const uint32_t PathEncodingBits = 32 - count_clz<MERGE_MAX_PATH_OPTIONS + 1>::value;
	using PathEncoding = ChooseBitDataType<static_max<32,MERGE_MAX_CHUNKS * PathEncodingBits>::value>;


	constexpr const uint32_t LengthSamplesPerThread = ((MERGE_MAX_PATH_OPTIONS + 1)*MERGE_MAX_CHUNKS + THREADS - 1) / THREADS;
	
	constexpr bool problem  = LengthSamplesPerThread >= 1;
	static_assert(problem, "LengthSamplesPerThread must  be >= 1");

	using SampleSorter = cub::BlockRadixSort<INDEX_TYPE, THREADS, LengthSamplesPerThread, ushort2>;
	using PathMergeScan = cub::BlockScan<PathEncoding, THREADS>;
	using IndexSorter = cub::BlockRadixSort<ChunkSortType, THREADS, 1, uint32_t>;

	struct SMem
	{

    
		uint32_t runflag, restart, halveStep;
		uint32_t startSharedRow, numSharedRow;
		int numChunks;
		int sumOut;
		uint32_t completed;
		PathEncoding usePath;
		union {
			INDEX_TYPE useMaxId;
			uint32_t remCounter;
		};
		uint32_t longChunkOffset;
		const INDEX_TYPE* __restrict chunkIndices[MERGE_MAX_CHUNKS]; 
		Either<const RIGHT_t* ,  const OUT_t* >  chunkValues[MERGE_MAX_CHUNKS]; //RL FIXME : add restrict back to internal pointer types?
		T multiplier[MERGE_MAX_CHUNKS];
		uint32_t chunkElementCount[MERGE_MAX_CHUNKS];
		volatile uint32_t chunkTakeElements[MERGE_MAX_CHUNKS];

		// Used for sorting
		uint32_t indexing[MERGE_MAX_CHUNKS];

		union {
			struct
			{
				ChunkSortType sort_keys[MERGE_MAX_CHUNKS];
				typename IndexSorter::TempStorage indexptrtempmem;
			};
			struct
			{
				union {
					INDEX_TYPE id_samples[MERGE_MAX_CHUNKS][MERGE_MAX_PATH_OPTIONS];
					struct {
						typename SampleSorter::TempStorage sorterTempMem;
						typename PathMergeScan::TempStorage pathmergeTempMem;
					};
					struct {
						uint32_t downStreamCount[THREADS + 1];
						INDEX_TYPE downStreamIndices[THREADS + 1];
					};
				};
			};
			struct {
				typename SingleLoadWorkDistribution::SharedMemT single_workdistributionMem;
				typename SingleLoadWorkDistribution::SharedTempMemT single_workdistributionTempMem;
				typename SingleLoadWorkDistribution:: template SharedTempMemOutT<ELEMENTS_PER_THREAD>  single_workdistributionTempMemOutFull;
			};
			typename SortAndCombiner::SMem single_sAndCMem;
			struct {
				OUT_t longOutDataBuffer[THREADS];
				INDEX_TYPE longOutIndexBuffer[THREADS];
			};
		};
	};

	__shared__ SMem smem;

	//get my block's offset
	if (threadIdx.x == 0)
	{
		uint32_t shared_handled = shared_rows_handled[blockIdx.x + restart_offset];
		smem.numSharedRow = 1 - shared_handled;
		smem.runflag = *run_flag;
		smem.restart = restart_completion[blockIdx.x + restart_offset];
		smem.sumOut = (smem.restart > RESTART_FIRST_ITERATION) ? output_row_count[sharedRows[blockIdx.x]] : 0;
		smem.halveStep = 0;
	}
	__syncthreads();

	if (smem.numSharedRow == 0)
		return;



	// Read in chunks (maximum MERGE_MAX_CHUNKS)
	if (threadIdx.x == 0 && smem.restart < RESTART_FIRST_ITERATION)
	{
		uint64_t chunk = reinterpret_cast<uint64_t>(output_row_list_heads[sharedRows[blockIdx.x]]);
		// if (sharedRows[blockIdx.x] == ROW_TO_INVESTIGATE)
		// 	printf("Row %d in MAX CHUNKS\n", sharedRows[blockIdx.x]);
		uint32_t chunk_counter = 0;
		uint32_t outsum = 0;

		while (chunk != 0)
		{
			bool first_row = (chunk & 2) != 0;
			Chunk* __restrict pChunk = reinterpret_cast<Chunk*>(chunk & 0xFFFFFFFFFFFFFFFCULL);
			uint32_t count;
			const INDEX_TYPE* pIndices;
			Either<const RIGHT_t*, const OUT_t*> pValues;
			int32_t numentries = pChunk->num_entries;
			typename SEMIRING_t::leftInput_t multiplier;

			smem.sort_keys[chunk_counter] = pChunk->sort_key;

			if (first_row)
			{
				//only first rows can be direct
				if (pChunk->isDirect())
				{
					DirectChunk* __restrict pDirectChunk = reinterpret_cast<DirectChunk*>(pChunk);
					count = numentries;
					pIndices = pDirectChunk->indices_direct(numentries);
					pValues = Either<const RIGHT_t*, const OUT_t*>::First(pDirectChunk->values_direct(numentries));
					multiplier = pDirectChunk->getMultiplier();
					pDirectChunk->setFirstConsumed();
					chunk = reinterpret_cast<uint64_t>(pChunk->readNextFront());
				}
				else
				{
					count = pChunk->firstCountCleared();
					pChunk->setFirstConsumed();
					pIndices = pChunk->indices_direct(numentries);
					pValues =Either<const RIGHT_t*, const OUT_t*>::Second( pChunk->values_direct(numentries));
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
				chunk = reinterpret_cast<uint64_t>(pChunk->readNextBack());
			}

			if (chunk_counter >= MERGE_MAX_CHUNKS)
			{
				printf("%d %d too many chunks: %d %d : count is : %u and should not be more than: %u\n", blockIdx.x, threadIdx.x, chunk_counter + 1, outsum + count, output_row_count[sharedRows[blockIdx.x]], ELEMENTS_PER_THREAD *THREADS * (MERGE_MAX_CHUNKS - 1));
				smem.runflag = 1;
			}
			else
			{
				smem.chunkIndices[chunk_counter] = pIndices;
				smem.chunkValues[chunk_counter] = pValues;
				smem.chunkElementCount[chunk_counter] = count;
				smem.multiplier[chunk_counter] = multiplier;
			}
			// DEBUG
			//if(sharedRows[blockIdx.x] == ROW_TO_INVESTIGATE)
			//	printf("Chunk %d : Count: %d Row: %u\n", chunk_counter, count, sharedRows[blockIdx.x]);
			// DEBUG
			outsum += count;
			++chunk_counter;
		}

		smem.numChunks = min(chunk_counter, MERGE_MAX_CHUNKS);
		smem.completed = (outsum < ELEMENTS_PER_THREAD*THREADS) ? 1 : 0;
		if (smem.restart == RESTART_OFF)
			restart_num_chunks[(blockIdx.x)] = smem.numChunks;
	}
	else if (threadIdx.x == 0)
	{
		smem.numChunks = restart_num_chunks[(blockIdx.x)];
		smem.completed = 0;
	}
	__syncthreads();

	if (smem.runflag != 0)
		return;

	// Sorting only if >= RESTART_FIRST_ITERATION
	{
		uint32_t value[1]{threadIdx.x};
		if (smem.restart < RESTART_FIRST_ITERATION)
		{
			ChunkSortType key[1];

			if (threadIdx.x < smem.numChunks)
				key[0] = smem.sort_keys[threadIdx.x];
			else
				key[0] = 0xFFFFFFFF;
#ifdef ENABLE_SORTING
			IndexSorter(smem.indexptrtempmem).Sort(key, value);
#endif
		}

		for (int i = threadIdx.x; i < MERGE_MAX_CHUNKS; i += THREADS)
		{
			smem.indexing[threadIdx.x] = value[0];
		}
	}
	__syncthreads();

	// If elements can't be held in temp, load samples (MERGE_MAX_PATH_OPTIONS per chunk)
	if (!smem.completed)
	{
		if (smem.restart >= RESTART_FIRST_ITERATION)
		{
			// Load values from last restart
			for (int wip = threadIdx.x / MERGE_MAX_PATH_OPTIONS; wip < smem.numChunks; wip += THREADS / MERGE_MAX_PATH_OPTIONS)
			{
				uint32_t lid = threadIdx.x % MERGE_MAX_PATH_OPTIONS;
				if (lid == 0)
				{
					// Do not use indexing here as we write the chunks out in correct order
					smem.chunkElementCount[wip] = restart_chunkElementCount[((blockIdx.x) * MERGE_MAX_CHUNKS) + wip];
					smem.chunkIndices[wip] = restart_chunkIndices[((blockIdx.x) * MERGE_MAX_CHUNKS) + wip];
					//fixme: RL bad practice.......
					smem.chunkValues[wip] = *reinterpret_cast<Either<const RIGHT_t* ,  const OUT_t* >*> (&restart_chunkValues[((blockIdx.x) * MERGE_MAX_CHUNKS) + wip]);
					smem.multiplier[wip] = restart_multiplier[((blockIdx.x) * MERGE_MAX_CHUNKS) + wip];
				}
			}
			if (threadIdx.x == 0 && smem.restart == RESTART_ITERATION_FINISH)
			{
				// We want to finish in the next iteration
				smem.completed = 1;
			}
		}
		else
		{
			__syncthreads();
			// We start our first iteration soon
			if (threadIdx.x == 0)
			{
				smem.restart = RESTART_FIRST_ITERATION;
			}
		}
		__syncthreads();

		//load samples from each list for column offset (warp based in parallel)
		for (int wip = threadIdx.x / MERGE_MAX_PATH_OPTIONS; wip < smem.numChunks; wip += THREADS / MERGE_MAX_PATH_OPTIONS)
		{
			uint32_t lid = threadIdx.x % MERGE_MAX_PATH_OPTIONS;
			uint32_t count = smem.chunkElementCount[smem.indexing[wip]];
			uint32_t step = (count + MERGE_MAX_PATH_OPTIONS - 1) / MERGE_MAX_PATH_OPTIONS;
			uint32_t test = min(count - 1, step * lid);
			INDEX_TYPE id = count > 0 ? smem.chunkIndices[smem.indexing[wip]][test] : 0xFFFFFFFF;
			smem.id_samples[wip][lid] = id;
		}
	}
	else if (threadIdx.x == 0)
	{
		// We are in the wrong case, remember that here
		smem.restart = RESTART_WRONG_CASE;
	}
	__syncthreads();

	// DEBUG
	//printSampling<INDEX_TYPE, MERGE_MAX_CHUNKS, MERGE_MAX_PATH_OPTIONS>(sharedRows, smem.numChunks, smem.id_samples, ROW_TO_INVESTIGATE);
	// DEBUG

	while (true)
	{
		int chunkWorkElements[1];
		if (!smem.completed)
		{
			INDEX_TYPE mySampledIds[LengthSamplesPerThread];
			ushort2 mySamplePayload[LengthSamplesPerThread];

#pragma unroll
			for (uint32_t i = 0; i < LengthSamplesPerThread; ++i)
			{
				uint32_t lid = i*THREADS + threadIdx.x;
				uint32_t chunk = lid / (MERGE_MAX_PATH_OPTIONS + 1);
				uint32_t sample = lid - chunk * (MERGE_MAX_PATH_OPTIONS + 1);
				if (chunk < smem.numChunks)
				{
					mySampledIds[i] = sample == 0 ? 0 : smem.id_samples[chunk][sample - 1];
					mySamplePayload[i] = make_ushort2(chunk, sample);
				}
				else
				{
					mySampledIds[i] = 0xFFFFFFFF;
					mySamplePayload[i] = make_ushort2(MERGE_MAX_CHUNKS, MERGE_MAX_PATH_OPTIONS + 1);
				}
			}
			__syncthreads();

			//sort according to index
			SampleSorter(smem.sorterTempMem).Sort(mySampledIds, mySamplePayload);

			//construct bitmask
			PathEncoding paths[LengthSamplesPerThread];
#pragma unroll
			for (uint32_t i = 0; i < LengthSamplesPerThread; ++i)
				paths[i] = static_cast<PathEncoding>(mySamplePayload[i].y) << static_cast<PathEncoding>(mySamplePayload[i].x * PathEncodingBits);
			//merge up
			PathMergeScan(smem.pathmergeTempMem).InclusiveScan(paths, paths, PathMergerOp<MERGE_MAX_CHUNKS, PathEncodingBits>());

			// reset and then compute output count
			uint32_t outputCount[LengthSamplesPerThread];
			for (uint32_t i = 0; i < LengthSamplesPerThread; ++i)
				outputCount[i] = 0;

			const PathEncoding Mask = (1 << PathEncodingBits) - 1;
#pragma unroll
			for (uint32_t chunk = 0; chunk < MERGE_MAX_CHUNKS; ++chunk)
			{
				if (chunk < smem.numChunks)
				{
					uint32_t count = smem.chunkElementCount[smem.indexing[chunk]];
					uint32_t step = (count + MERGE_MAX_PATH_OPTIONS - 1) / MERGE_MAX_PATH_OPTIONS;
#pragma unroll
					for (uint32_t i = 0; i < LengthSamplesPerThread; ++i)
					{
						uint32_t chunkPath = static_cast<uint32_t>((paths[i] >> (PathEncodingBits * chunk)) & Mask);
						outputCount[i] += min(count, step * chunkPath);
					}
				}
			}
			__syncthreads();

			// ######## DEBUG
			//printCountPerSampling(sharedRows, outputCount[0], mySampledIds[0], 2 * ELEMENTS_PER_THREAD*THREADS, ROW_TO_INVESTIGATE);
			// ######## DEBUG

			//publish so next can check it
			smem.downStreamCount[THREADS] = 0xFFFFFFFF;
			smem.downStreamIndices[THREADS] = 0;
			smem.downStreamIndices[threadIdx.x] = mySampledIds[0];

			smem.usePath = 0;
			smem.useMaxId = 0;
			__syncthreads();

			// Propagate outputcount locally first such that first element per array is correct
#pragma unroll
			for (uint32_t i = LengthSamplesPerThread - 1; i > 0; --i)
				if (mySampledIds[i - 1] == mySampledIds[i])
					outputCount[i - 1] = outputCount[i];

			smem.downStreamCount[threadIdx.x] = outputCount[0];
			__syncthreads();

			//propagate count over equal ids over arrays
			bool prop = mySampledIds[0] == smem.downStreamIndices[threadIdx.x + 1] &&
				mySampledIds[0] != 0xFFFFFFFF;
			bool changed;
			do
			{
				changed = prop && smem.downStreamCount[threadIdx.x + 1] != outputCount[0];
				if (changed)
					smem.downStreamCount[threadIdx.x] = outputCount[0] = smem.downStreamCount[threadIdx.x + 1];
				changed = __syncthreads_or(changed);
			} while (changed);

			//propagate count locally again
			if (mySampledIds[LengthSamplesPerThread - 1] == smem.downStreamIndices[threadIdx.x + 1])
				outputCount[LengthSamplesPerThread - 1] = smem.downStreamCount[threadIdx.x + 1];
#pragma unroll
			for (uint32_t i = LengthSamplesPerThread - 1; i > 0; --i)
				if (mySampledIds[i - 1] == mySampledIds[i])
					outputCount[i - 1] = outputCount[i];

			// ######## DEBUG
			//printCountPerSampling(sharedRows, outputCount[0], mySampledIds[0], 2 * ELEMENTS_PER_THREAD*THREADS, ROW_TO_INVESTIGATE);
			// ######## DEBUG

			//find the first that goes over the threshold
			if (outputCount[LengthSamplesPerThread - 1] <= ELEMENTS_PER_THREAD*THREADS && smem.downStreamCount[threadIdx.x + 1] > ELEMENTS_PER_THREAD*THREADS)
			{
				// ######## DEBUG
				/*if (sharedRows[blockIdx.x] == ROW_TO_INVESTIGATE)
					printf("THREAD: %u Outputcount: %u Next Count %u | path: %llu maxid: %u\n", threadIdx.x, outputCount[LengthSamplesPerThread - 1], smem.downStreamCount[threadIdx.x + 1], paths[LengthSamplesPerThread - 1], smem.downStreamIndices[threadIdx.x + 1]);*/
				// ######## DEBUG
				smem.usePath = paths[LengthSamplesPerThread - 1];
				smem.useMaxId = smem.downStreamIndices[threadIdx.x + 1];
			}

#pragma unroll
			for (uint32_t i = 0; i < LengthSamplesPerThread - 1; ++i)
			{
				if (outputCount[i] <= ELEMENTS_PER_THREAD*THREADS && outputCount[i + 1] > ELEMENTS_PER_THREAD*THREADS)
				{
					smem.usePath = paths[i];
					smem.useMaxId = mySampledIds[i + 1];
				}
			}

			smem.completed = 1;
			__syncthreads();

			if (smem.usePath == 0)
			{
				//if (sharedRows[blockIdx.x] != ROW_TO_INVESTIGATE)
				//	return;

				// ######## DEBUG
				/*if(sharedRows[blockIdx.x] == ROW_TO_INVESTIGATE)
					printInvalidPath(sharedRows);*/
				// ######## DEBUG

				if (threadIdx.x == 0)
				{
					smem.useMaxId = UINT32_MAX;
					smem.halveStep = 1;
					// ######## DEBUG
					/*if (sharedRows[blockIdx.x] == ROW_TO_INVESTIGATE)
						printf("-----------------------------------------------------------------------------------\n");*/
					// ######## DEBUG
				}
				__syncthreads();

				// Go one half step -> get smallest ID
				// -> all chunks should reach this with now at most half the workload
				if (threadIdx.x < smem.numChunks)
				{
					uint32_t count = smem.chunkElementCount[smem.indexing[threadIdx.x]];
					int step = ((count + (MERGE_MAX_PATH_OPTIONS * DIVISION_FACTOR) - 1) / (MERGE_MAX_PATH_OPTIONS * DIVISION_FACTOR));
					if (count > 1)
					{
						INDEX_TYPE id = smem.chunkIndices[smem.indexing[threadIdx.x]][step];
						// ######## DEBUG
						//if (sharedRows[blockIdx.x] == ROW_TO_INVESTIGATE)
						//	printf("Chunk: %d with Count: %d - step: %d| Check out ID for chunk: %u\n", threadIdx.x, count, step, id);
						// ######## DEBUG
						atomicMin(&(smem.useMaxId), id);
					}
				}
				__syncthreads();

				// Select all chunks that are below this ID
				if (threadIdx.x == 0)
				{
					// ######## DEBUG
					/*if (sharedRows[blockIdx.x] == ROW_TO_INVESTIGATE)
						printf("MaxID chosen: %u\n", smem.useMaxId);*/
					// ######## DEBUG
					for (int i = 0; i < smem.numChunks; ++i)
					{
						if (smem.chunkElementCount[smem.indexing[i]] > 0 && smem.chunkIndices[smem.indexing[i]][0] < smem.useMaxId)
						{
							// Take these chunks -> for each chunk set the path to 1
							smem.usePath |= static_cast<PathEncoding>(1) << static_cast<PathEncoding>(i * PathEncodingBits);
						}
					}
				}
				__syncthreads();
			}
			// ######################################################################################################################################################


			//determine actual chunk ends to use
			for (int wip = threadIdx.x / WARP_SIZE; wip < smem.numChunks; wip += THREADS / WARP_SIZE)
			{
				const PathEncoding PathCodingMask = (1 << PathEncodingBits) - 1;
				int lpos = static_cast<int>((smem.usePath >> (wip*PathEncodingBits)) & PathCodingMask);
				int count = smem.chunkElementCount[smem.indexing[wip]];
				int step;
				if (smem.halveStep)
					step = ((count + (MERGE_MAX_PATH_OPTIONS * DIVISION_FACTOR) - 1) / (MERGE_MAX_PATH_OPTIONS * DIVISION_FACTOR));
				else
					step = (count + MERGE_MAX_PATH_OPTIONS - 1) / MERGE_MAX_PATH_OPTIONS;
				int startpos = max(0, step * (lpos - 1));
				int endpos = min(count, step * lpos);

				smem.chunkTakeElements[wip] = endpos;
				int current = endpos;

				for (int i = startpos + laneid(); i < endpos; i += WARP_SIZE)
				{
					INDEX_TYPE next = static_cast<uint32_t>(-1);
					if (i < count - 1)
						next = smem.chunkIndices[smem.indexing[wip]][i + 1];
					if (smem.chunkIndices[smem.indexing[wip]][i] < smem.useMaxId && smem.useMaxId <= next)
						current = i + 1;
				}

				uint32_t found = __ballot_sync(0xFFFFFFFF, current != endpos);
				if (found != 0)
				{
					current = __shfl_sync(0xFFFFFFF, current, __ffs(found) - 1);
					smem.chunkTakeElements[wip] = current;
				}

				//not reduced to 0 -> set completed false
				if (current != count)
					smem.completed = 0;
			}
			__syncthreads();


			chunkWorkElements[0] = 0;
			if (threadIdx.x < smem.numChunks)
			{
				chunkWorkElements[0] = smem.chunkTakeElements[threadIdx.x];
			}
		}
		else
		{
			//we can combine all at once!
			chunkWorkElements[0] = 0;
			if (threadIdx.x < smem.numChunks)
				chunkWorkElements[0] = smem.chunkElementCount[smem.indexing[threadIdx.x]];
		}

		//use workdistribution to assign for loading
		SingleLoadWorkDistribution:: template initialize<true>(smem.single_workdistributionMem, smem.single_workdistributionTempMem, chunkWorkElements);

		int chunk[ELEMENTS_PER_THREAD];
		int element[ELEMENTS_PER_THREAD];

		int elements = SingleLoadWorkDistribution:: template assignWorkAllThreads<false, ELEMENTS_PER_THREAD>(
			smem.single_workdistributionMem, smem.single_workdistributionTempMem, smem.single_workdistributionTempMemOutFull,
			chunk, element);

		// ######## DEBUG
		if (threadIdx.x == 0 && elements == 0 /*&& sharedRows[blockIdx.x] == ROW_TO_INVESTIGATE*/)
		{
			//printf("Row: %u got 0 elements with maxID: %u\n", sharedRows[blockIdx.x], smem.useMaxId);
		}
		// ######## DEBUG

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
					const INDEX_TYPE* __restrict ip = smem.chunkIndices[smem.indexing[chunk[i]]];
					combIndex[i] = ip[element[i]];

					const Either<const RIGHT_t* ,  const OUT_t* > dp = smem.chunkValues[smem.indexing[chunk[i]]];

					if (dp.isFirst()) {
						auto idx_ = element[i];
						RIGHT_t right_ = dp.valFirst()[idx_];
						auto idx_r_ = chunk[i];
						auto idx_r_2_ = smem.indexing[idx_r_];
						auto left_ =  smem.multiplier[idx_r_2_];
						data[i] = semiring.multiply(left_ , right_);
					} else {
						auto idx_ = element[i];
						data[i] =  dp.valSecond()[idx_];
					}
				}
				else
				{
					data[i] = SEMIRING_t::AdditiveIdentity();
					combIndex[i] = static_cast<uint32_t>(-1);
				}
			}
			__syncthreads();

			
			auto & j  =smem.single_sAndCMem;

			auto fo = 		[](auto a, auto b) {
				return a == b;
			};

			auto bq = [](auto a, auto b) {
				return true;
			};

			numOut = 2;
			numOut =  SortAndCombiner::combine(j,
				 combIndex, 
				 data, 
				 combinedEntries,
				fo,bq
				, semiring);


			__syncthreads();
			// ######## DEBUG
			/*if (numOut == 0 && threadIdx.x == 0)
			{
				printf("%d %d oops in max chunks\n", blockIdx.x, threadIdx.x);
			}*/
			//if (numOut == 0)
			//	return;
			// ######## DEBUG
		}

		// create new chunk (could also reuse old ones if completely used up...?)
		if (threadIdx.x == 0)
		{
			// Try to allocate chunk
			uint32_t chunkoff;
			int ignored;
			// Update pre alloc before the actual allocation
			if (!allocChunk<OUT_t, INDEX_TYPE>(numOut, chunk_alloc, chunk_size, chunkoff, ignored, false))
			{
				chunkoff = static_cast<uint32_t>(-1);
				atomicOr(run_flag, 0x1);
				// Write restart state
				restart_completion[blockIdx.x + restart_offset] = smem.restart;
			}
			else
			{
				//need to add flag and offset for copy later (offset = s)
				uint32_t s = smem.sumOut;
				//write chunk header
				INDEX_TYPE actualrow = sharedRows[blockIdx.x];
				//write chunk pointer
				uint32_t chunk_pointer_position = atomicAdd(chunk_pointer_alloc, 1);
				if (chunk_pointer_position >= chunk_pointer_sizes)
				{
					chunkoff = static_cast<uint32_t>(-1);
					atomicOr(run_flag,0x2);
					if(chunk_pointer_position == chunk_pointer_sizes)
						*chunk_pointer_pos = chunk_pointer_sizes;
					// Write restart state
					restart_completion[blockIdx.x + restart_offset] = smem.restart;
				}
				else
				{
					chunks_pointers[chunk_pointer_position] = reinterpret_cast<void*>(Chunk::place(chunks, chunkoff, numOut, actualrow, Chunk::StartingOffsetFlag | s, 0));
					//write row count
					s += numOut;
					smem.sumOut = s;
					output_row_count[actualrow] = s;
				}
			}
			smem.longChunkOffset = chunkoff;
		}

		smem.remCounter = 0;
		__syncthreads();

		if (smem.longChunkOffset == static_cast<uint32_t>(-1))
		{
			// Write out current state and return
			for (int wip = threadIdx.x / MERGE_MAX_PATH_OPTIONS; wip < smem.numChunks; wip += THREADS / MERGE_MAX_PATH_OPTIONS)
			{
				uint32_t lid = threadIdx.x % MERGE_MAX_PATH_OPTIONS;
				if (lid == 0)
				{
					restart_chunkElementCount[((blockIdx.x) * MERGE_MAX_CHUNKS) + wip] = smem.chunkElementCount[smem.indexing[wip]];
					restart_multiplier[((blockIdx.x) * MERGE_MAX_CHUNKS) + wip] = smem.multiplier[smem.indexing[wip]];
					restart_chunkIndices[((blockIdx.x) * MERGE_MAX_CHUNKS) + wip] = const_cast<INDEX_TYPE*>(smem.chunkIndices[smem.indexing[wip]]);
					// FIXME: RL  - casting like this is a sin
					restart_chunkValues[((blockIdx.x) * MERGE_MAX_CHUNKS) + wip] = *reinterpret_cast<Either< RIGHT_t* ,   OUT_t* >*>(&smem.chunkValues[smem.indexing[wip]]);
				}
			}
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

			//write out
			if (written + threadIdx.x < numOut)
			{
				typename SEMIRING_t::output_t* valstart = Chunk::cast(chunks, smem.longChunkOffset)->values_direct(numOut);
				INDEX_TYPE* indexstart = Chunk::cast(chunks, smem.longChunkOffset)->indices_direct(numOut);

				valstart[written + threadIdx.x] = smem.longOutDataBuffer[threadIdx.x];
				indexstart[written + threadIdx.x] = smem.longOutIndexBuffer[threadIdx.x];
			}
			__syncthreads();
		}

		// Work is done, we can stop now
		if (smem.completed)
			break;

		//reduce all counts and adjust pointers
		for (int wip = threadIdx.x / MERGE_MAX_PATH_OPTIONS; wip < smem.numChunks; wip += THREADS / MERGE_MAX_PATH_OPTIONS)
		{
			uint32_t lid = threadIdx.x % MERGE_MAX_PATH_OPTIONS;
			uint32_t count = smem.chunkElementCount[smem.indexing[wip]];
			uint32_t rem = smem.chunkTakeElements[wip];

			uint32_t newcount = count - rem;
			smem.chunkElementCount[smem.indexing[wip]] = newcount;
			const INDEX_TYPE* __restrict newchunkIndices = smem.chunkIndices[smem.indexing[wip]] + rem;
			smem.chunkIndices[smem.indexing[wip]] = newchunkIndices;
			Either<const RIGHT_t* ,  const OUT_t* >   newchunkValues; //fixme RL  :  add restrict on interior types?

			if (smem.chunkValues[smem.indexing[wip]].isFirst()) {
				newchunkValues = Either<const RIGHT_t* ,  const OUT_t* >::First(smem.chunkValues[smem.indexing[wip]].valFirst() + rem);
			} else {
				newchunkValues = Either<const RIGHT_t* ,  const OUT_t* >::Second(smem.chunkValues[smem.indexing[wip]].valSecond() + rem);
			}
			
			smem.chunkValues[smem.indexing[wip]] = newchunkValues;

			uint32_t step = (newcount + MERGE_MAX_PATH_OPTIONS - 1) / MERGE_MAX_PATH_OPTIONS;
			uint32_t test = min(newcount - 1, step * lid);
			INDEX_TYPE id =  newcount > 0 ? newchunkIndices[test] : 0xFFFFFFFF;
			smem.id_samples[wip][lid] = id;

			if (lid == 0)
				atomicAdd(&smem.remCounter, newcount);
		}
		__syncthreads();

		// ######## DEBUG
		//printSampling<INDEX_TYPE, MERGE_MAX_CHUNKS, MERGE_MAX_PATH_OPTIONS>(sharedRows, smem.numChunks, smem.id_samples, ROW_TO_INVESTIGATE);
		// ######## DEBUG

		smem.completed = smem.remCounter < ELEMENTS_PER_THREAD*THREADS ? 1 : 0;
		if (threadIdx.x == 0)
		{
			smem.restart = smem.completed ? RESTART_ITERATION_FINISH : RESTART_ITERATION_UNKNOWN;
			smem.halveStep = 0;
		}
		__syncthreads();
	}

	// This row is done
	if (threadIdx.x == 0)
	{
		shared_rows_handled[blockIdx.x + restart_offset] = 1;
	}

	return;
}


template<uint32_t NNZ_PER_THREAD, uint32_t THREADS, uint32_t BLOCKS_PER_MP, uint32_t INPUT_ELEMENTS_PER_THREAD, uint32_t RETAIN_ELEMENTS_PER_THREAD, uint32_t MERGE_MAX_CHUNKS, uint32_t MERGE_MAX_PATH_OPTIONS, typename VALUE_TYPE, typename INDEX_TYPE, typename OFFSET_TYPE,
        typename T, typename U, typename Label,
        typename SEMIRING_t>
        	        void AcSpGEMMKernels::h_mergeSharedRowsMaxChunks(const uint32_t* __restrict blockOffsets, const uint32_t* __restrict sharedRows, void** output_row_list_heads,
	OFFSET_TYPE* output_row_count, uint32_t* chunks, uint32_t* chunk_alloc, uint32_t* chunk_pre_alloc, uint32_t chunk_size,
	void** chunks_pointers, uint32_t* chunk_pointer_alloc, uint32_t chunk_pointer_sizes,
	uint32_t* run_flag, uint32_t* restart_completion, uint32_t* shared_rows_handled,
	INDEX_TYPE** restart_chunkIndices, Either<typename SEMIRING_t::rightInput_t*, typename SEMIRING_t::output_t*>* restart_chunkValues, typename SEMIRING_t::leftInput_t* restart_multiplier, uint32_t* restart_chunkElementCount, uint32_t restart_offset, uint32_t* restart_num_chunks, uint32_t* chunk_pointer_pos, SEMIRING_t semiring)
{

	mergeSharedRowsMaxChunks<NNZ_PER_THREAD, THREADS, BLOCKS_PER_MP, INPUT_ELEMENTS_PER_THREAD, RETAIN_ELEMENTS_PER_THREAD, MERGE_MAX_CHUNKS, MERGE_MAX_PATH_OPTIONS, VALUE_TYPE, INDEX_TYPE, OFFSET_TYPE,   T,  U,  Label,
             SEMIRING_t><<<gridDim, blockDim>>>(
		blockOffsets, sharedRows, output_row_list_heads, output_row_count, chunks, chunk_alloc, chunk_pre_alloc, chunk_size,
		chunks_pointers, chunk_pointer_alloc, chunk_pointer_sizes, run_flag, restart_completion, shared_rows_handled,
		restart_chunkIndices, restart_chunkValues, restart_multiplier, restart_chunkElementCount, restart_offset, restart_num_chunks, chunk_pointer_pos, semiring);
}


#define GPUCompressedMatrixMatrixMultiplyMergeMaxChunks(TYPE, THREADS, BLOCKS_PER_MP, NNZPERTHREAD, INPUT_ELEMENTS_PER_THREAD, RETAIN_ELEMENTS_PER_THREAD, MERGE_MAX_CHUNKS, MERGE_MAX_PATH_OPTIONS) \
	template void AcSpGEMMKernels::h_mergeSharedRowsMaxChunks<NNZPERTHREAD, THREADS, BLOCKS_PER_MP, INPUT_ELEMENTS_PER_THREAD, RETAIN_ELEMENTS_PER_THREAD, MERGE_MAX_CHUNKS, MERGE_MAX_PATH_OPTIONS, TYPE, uint32_t, uint32_t> \
	(const uint32_t* __restrict blockOffsets, const uint32_t* __restrict sharedRows, void** output_row_list_heads, \
	uint32_t* output_row_count, \
	uint32_t* chunks, uint32_t* chunk_alloc, uint32_t* chunk_pre_alloc, uint32_t chunk_size, \
	void** chunks_pointers, uint32_t* chunk_pointer_alloc, uint32_t chunk_pointer_sizes, \
	uint32_t* run_flag, uint32_t* restart_completion, uint32_t* shared_rows_handled, \
	uint32_t** restart_chunkIndices, TYPE** restart_chunkValues, TYPE* restart_multiplier, uint32_t* restart_chunkElementCountDataOffset2, uint32_t restart_offset, uint32_t* restart_num_chunks, uint32_t* chunk_pointer_pos);
