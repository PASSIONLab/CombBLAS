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
 * Chunk.cuh
 *
 * ac-SpGEMM
 *
 * Authors: Daniel Mlakar, Markus Steinberger, Martin Winter
 *------------------------------------------------------------------------------
*/

#pragma once
#include "../common.h"

using ChunkSortType = uint32_t;
const int chunk_member_offset = alignment(sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(ChunkSortType), 8);


template<typename VALUE_TYPE, typename INDEX_TYPE>
struct alignas(16) Chunk
{
	// with which row does the chunk start
	INDEX_TYPE firstrow;
	// the number of matrix entries and column offsets in the chunk
	uint32_t num_entries;
	// where does the last (uncompleted) row in the chunk start
	uint32_t last_row_count;
	// how many elements are in the first row
	uint32_t first_row_count;
	// sortkey
	ChunkSortType sort_key;



	__device__ __forceinline__ Chunk(uint32_t num, INDEX_TYPE firstrow, uint32_t firstrowCount = 0, uint32_t lastrowCount = 0, ChunkSortType sortkey = 0) :
		firstrow(firstrow), num_entries(num), last_row_count(lastrowCount), first_row_count(firstrowCount), sort_key(sortkey)
	{

	}

	__device__ __forceinline__ static uint32_t size(uint32_t count, bool nextPointers)
	{
		uint32_t s = (nextPointers ? 16 : 0) + count*(sizeof(VALUE_TYPE) + sizeof(INDEX_TYPE)) + sizeof(Chunk);
		return (s + 15) & 0xFFFFFFF0;

	}
	__device__ __forceinline__
		static Chunk* place(void* chunks, uint32_t offset, uint32_t num, INDEX_TYPE firstrow, uint32_t firstrowCount = 0, uint32_t lastrowCount = 0, ChunkSortType sortkey = 0)
	{
		return new(reinterpret_cast<char*>(chunks) + offset) Chunk(num, firstrow, firstrowCount, lastrowCount, sortkey);
	}
	__device__ __forceinline__
		static Chunk* cast(void* chunks, uint32_t offset)
	{
		return reinterpret_cast<Chunk*>(reinterpret_cast<char*>(chunks) + offset);
	}
	//__device__ __forceinline__ void write(void* location) const
	//{
	//	*reinterpret_cast<uint4*>(location) = *reinterpret_cast<const uint4*>(this);
	//}

	__device__ __forceinline__ VALUE_TYPE* values_direct(uint32_t count)
	{
		return reinterpret_cast<VALUE_TYPE*>(reinterpret_cast<char*>(this) + chunk_member_offset);
	}
	__device__ __forceinline__ INDEX_TYPE* indices_direct(uint32_t count)
	{
		return reinterpret_cast<INDEX_TYPE*>(reinterpret_cast<char*>(this) + chunk_member_offset + sizeof(VALUE_TYPE)*count);
	}

	__device__ __forceinline__ const VALUE_TYPE* values_direct(uint32_t count) const
	{
		return reinterpret_cast<const VALUE_TYPE*>(reinterpret_cast<const char*>(this) + chunk_member_offset);
	}
	__device__ __forceinline__ const INDEX_TYPE* indices_direct(uint32_t count) const
	{
		return reinterpret_cast<const INDEX_TYPE*>(reinterpret_cast<const char*>(this) + chunk_member_offset + sizeof(VALUE_TYPE)*count);
	}

	__device__ __forceinline__ void writeNextFront(Chunk* next)
	{
		*reinterpret_cast<Chunk**>(reinterpret_cast<char*>(this) - 16) = next;
	}

	__device__ __forceinline__ void writeNextBack(Chunk* next)
	{
		*reinterpret_cast<Chunk**>(reinterpret_cast<char*>(this) - 8) = next;
	}

	__device__ __forceinline__ void writeNextPointer(Chunk* next, bool front)
	{
		*reinterpret_cast<Chunk**>(reinterpret_cast<char*>(this) - 16 + (front ? 0 : 8)) = next;
	}

	__device__ __forceinline__ Chunk* readNextFront() const
	{
		return *reinterpret_cast<Chunk* const *>(reinterpret_cast<char const *>(this) - 16);
	}

	__device__ __forceinline__ Chunk* readNextBack() const
	{
		return *reinterpret_cast<Chunk* const *>(reinterpret_cast<char const *>(this) - 8);
	}

	__device__ __forceinline__ void setLastConsumed()
	{
		last_row_count = last_row_count | 0x80000000;
	}
	__device__ __forceinline__ void setFirstConsumed()
	{
		first_row_count = first_row_count | 0x80000000;
	}

	static const uint32_t StartingOffsetFlag = 0x40000000;

	__device__ __forceinline__ uint32_t startingoffset() const
	{
		if ((first_row_count & StartingOffsetFlag) == StartingOffsetFlag)
			return first_row_count & 0x3FFFFFFF;
		return 0;
	}

	__device__ __forceinline__ bool lastConsumed() const
	{
		return (last_row_count & 0x80000000) != 0;
	}
	__device__ __forceinline__ bool firstConsumed() const
	{
		return (first_row_count & 0x80000000) != 0;
	}

	__device__ __forceinline__ uint32_t lastCountCleared() const
	{
		return last_row_count & (~0x80000000);
	}
	__device__ __forceinline__ uint32_t firstCountCleared() const
	{
		return first_row_count & (~0xC0000000);
	}

	__device__ __forceinline__ VALUE_TYPE getMultiplier() const
	{
		return 1;
	}
	__device__ __forceinline__ bool isDirect() const
	{
		return last_row_count == 0xFFFFFFFF;
	}
};

template<typename VALUE_TYPE, typename INDEX_TYPE>
__device__ __forceinline__ bool allocChunk(uint32_t count, uint32_t* chunk_alloc, uint32_t chunk_size, uint32_t& offset, int& worstcaseRem, bool nextPointers = true)
{
	uint32_t s = Chunk<VALUE_TYPE, INDEX_TYPE>::size(count, nextPointers);
	worstcaseRem -= s;
	offset = atomicAdd(chunk_alloc, s) + (nextPointers ? 16 : 0);
	return offset + s <= chunk_size;
}

template<typename VALUE_TYPE, typename INDEX_TYPE, typename OUT_OF_MEM_CALLBACK, typename OUT_OF_CHUNK_POINTER_CALLBACK>
__device__ __forceinline__ uint32_t completeChunkAlloc(uint32_t count, uint32_t* chunks, uint32_t* chunk_alloc, uint32_t chunk_size, void** chunks_pointers, uint32_t* chunk_pointer_alloc, uint32_t chunk_pointer_sizes, uint32_t* chunk_pointer_pos, OUT_OF_MEM_CALLBACK cb, OUT_OF_CHUNK_POINTER_CALLBACK ccb)
{
	//alloc chunk
	uint32_t chunkoff;
	int unused_worstCaseRemainder;
	if (!allocChunk<VALUE_TYPE, INDEX_TYPE>(count, chunk_alloc, chunk_size, chunkoff, unused_worstCaseRemainder))
	{
		chunkoff = 0xFFFFFFFF;
		cb();
	}
	else
	{
		//write chunk pointer
		uint32_t chunk_pointer_position = atomicAdd(chunk_pointer_alloc, 1);
		if (chunk_pointer_position >= chunk_pointer_sizes)
		{
			chunkoff = 0xFFFFFFFF;
			if (chunk_pointer_position == chunk_pointer_sizes)
				*chunk_pointer_pos = chunk_pointer_sizes;
			ccb();
		}
		else
		{
			chunks_pointers[chunk_pointer_position] = reinterpret_cast<void*>(Chunk<VALUE_TYPE, INDEX_TYPE>::cast(chunks, chunkoff));
		}		
	}
	return chunkoff;
}



template<typename LEFT_T, typename VALUE_TYPE, typename INDEX_TYPE>
struct alignas(16) DirectChunk : public Chunk<VALUE_TYPE, INDEX_TYPE>
{
	using Chunk<VALUE_TYPE, INDEX_TYPE>::sort_key;
	const INDEX_TYPE* indices;
	const VALUE_TYPE* values;
	LEFT_T multiplier;

	__device__ __forceinline__ DirectChunk(uint32_t num, INDEX_TYPE firstrow, const INDEX_TYPE* indices, const VALUE_TYPE* values, LEFT_T multiplier, ChunkSortType sortkey = 0) :
		Chunk<VALUE_TYPE, INDEX_TYPE>(num, firstrow, num, 0xFFFFFFFF, sortkey),
		indices(indices), 
		values(values), 
		multiplier(multiplier)
	{

	}

	__device__ __forceinline__ static uint32_t size(bool nextPointers)
	{
		uint32_t s = (nextPointers ? 16 : 0) + sizeof(DirectChunk);
		return (s + 15) & 0xFFFFFFF0;
	}

	__device__ __forceinline__
		static DirectChunk* place(void* chunks, uint32_t offset, uint32_t num, INDEX_TYPE firstrow, const INDEX_TYPE* indices, const VALUE_TYPE* values, LEFT_T multiplier, ChunkSortType sortkey = 0)
	{
		return new(reinterpret_cast<char*>(chunks) + offset) DirectChunk(num, firstrow, indices, values, multiplier, sortkey);
	}
	__device__ __forceinline__
		static DirectChunk* cast(void* chunks, uint32_t offset)
	{
		return reinterpret_cast<DirectChunk*>(reinterpret_cast<char*>(chunks) + offset);
	}
	//__device__ __forceinline__ void write(void* location) const
	//{
	//	*reinterpret_cast<uint4*>(location)[0] = *reinterpret_cast<const uint4*>(this)[0];
	//	*reinterpret_cast<uint4*>(location)[1] = *reinterpret_cast<const uint4*>(this)[1];
	//}

	__device__ __forceinline__ const VALUE_TYPE* values_direct(uint32_t count)
	{
		return values;
	}
	__device__ __forceinline__ const INDEX_TYPE* indices_direct(uint32_t count)
	{
		return indices;
	}

	__device__ __forceinline__ const VALUE_TYPE* values_direct(uint32_t count) const
	{
		return values;
	}
	__device__ __forceinline__ const INDEX_TYPE* indices_direct(uint32_t count) const
	{
		return indices;
	}

	__device__ __forceinline__ LEFT_T getMultiplier() const
	{
		return multiplier;
	}
};

template<typename LEFT_T, typename VALUE_TYPE, typename INDEX_TYPE>
__device__ __forceinline__ bool allocDirectChunk(uint32_t* chunk_alloc, uint32_t chunk_size, uint32_t& offset, bool nextPointers = true)
{
	uint32_t s = DirectChunk<LEFT_T, VALUE_TYPE, INDEX_TYPE>::size(nextPointers);
	offset = atomicAdd(chunk_alloc, s) + (nextPointers ? 16 : 0);
	return offset + s <= chunk_size;
}
