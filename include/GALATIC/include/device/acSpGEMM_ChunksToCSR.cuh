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
 * ChunkstoCSR.cuh
 *
 * ac-SpGEMM
 *
 * Authors: Daniel Mlakar, Markus Steinberger, Martin Winter
 *------------------------------------------------------------------------------
*/

#pragma once

#include "MultiplyKernels.h"
#include "Chunk.cuh"

template< typename VALUE_TYPE, typename INDEX_TYPE, typename OFFSET_TYPE>
__global__ void copyChunks(void* const* __restrict chunks_pointers, const uint32_t* __restrict chunk_pointer_alloc, 
	VALUE_TYPE * value_out, INDEX_TYPE * index_out, const OFFSET_TYPE* __restrict result_offets)
{
	using Chunk = ::Chunk<VALUE_TYPE, INDEX_TYPE>;

	struct Smem
	{
		uint32_t chunksize;
		uint32_t writeoffset;
		const VALUE_TYPE* in_values;
		const INDEX_TYPE* in_indices;
	};

	__shared__ Smem smem;

	uint32_t counter = blockIdx.x;

	while (counter < *chunk_pointer_alloc)
	{
		if(threadIdx.x == 0)
		{
			const Chunk* chunk = reinterpret_cast<const Chunk*>(chunks_pointers[counter]);
			uint32_t chunksize = chunk->num_entries;
			const VALUE_TYPE* in_values = chunk->values_direct(chunksize);
			const INDEX_TYPE* in_indices = chunk->indices_direct(chunksize);
			uint32_t firstrow = chunk->firstrow;

			uint32_t startingOffset = chunk->startingoffset();
			if(startingOffset == 0)
			{
				if (chunk->firstConsumed())
				{
					uint32_t firstoffset = chunk->firstCountCleared();
					chunksize -= firstoffset;
					in_values += firstoffset;
					in_indices += firstoffset;
					++firstrow;
				}
				if (chunk->lastConsumed() && !chunk->isDirect())
					chunksize -= chunk->lastCountCleared();
			}

			smem.chunksize = chunksize;
			smem.in_values = in_values;
			smem.in_indices = in_indices;

			//special case for multiple chunk rows (need offset for writing!)
			smem.writeoffset = startingOffset + result_offets[firstrow];
		}
		__syncthreads();

		//write out
		for (uint32_t i = threadIdx.x; i < smem.chunksize; i += blockDim.x)
		{
			value_out[smem.writeoffset + i] = smem.in_values[i];
			index_out[smem.writeoffset + i] = smem.in_indices[i];
		}

		counter += gridDim.x;
	}

}

template<typename VALUE_TYPE, typename INDEX_TYPE, typename OFFSET_TYPE>
void AcSpGEMMKernels::h_copyChunks(void* const* __restrict chunks_pointers, const uint32_t* __restrict chunk_pointer_alloc, VALUE_TYPE * value_out, INDEX_TYPE * index_out, const uint32_t* __restrict result_offets)
{
	int blockSize(256);

	static size_t copyBlocksOnGPU = 0;
	if (copyBlocksOnGPU == 0)
	{
		CUdevice dev;
		cudaGetDevice(&dev);
		int occ, sm;
		void(*ptr)(void* const* __restrict, const uint32_t* __restrict, VALUE_TYPE *, INDEX_TYPE * index_out, const uint32_t* __restrict) = copyChunks< VALUE_TYPE, INDEX_TYPE, OFFSET_TYPE>;
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, ptr, blockSize, 0);
		cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev);
		copyBlocksOnGPU = sm*occ;
	}
	copyChunks<VALUE_TYPE, INDEX_TYPE, OFFSET_TYPE> <<<copyBlocksOnGPU, blockSize >>>(chunks_pointers, chunk_pointer_alloc, value_out, index_out, result_offets);
}
