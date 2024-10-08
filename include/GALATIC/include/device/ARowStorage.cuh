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
 * ARowStorage.cuh
 *
 * ac-SpGEMM
 *
 * Authors: Daniel Mlakar, Markus Steinberger, Martin Winter
 *------------------------------------------------------------------------------
*/

#pragma once

#include "../meta_utils.h"


template<typename INDEX_TYPE, uint32_t NNZ_PER_BLOCK, uint32_t THREADS, bool ENCODING>
class ARowStorage;

template<typename INDEX_TYPE, uint32_t NNZ_PER_BLOCK, uint32_t THREADS>
class ARowStorage<INDEX_TYPE, NNZ_PER_BLOCK, THREADS, false>
{
	INDEX_TYPE row_ids[NNZ_PER_BLOCK];

public:
	using EncodedRowType = INDEX_TYPE;

	__device__ __forceinline__
		void clear()
	{
		#pragma unroll
		for (uint32_t i = 0; i < NNZ_PER_BLOCK; i += THREADS)
			row_ids[i + threadIdx.x] = 0;
	}

	__device__ __forceinline__
	void storeReference(uint32_t id, INDEX_TYPE row)
	{
	}

	__device__ __forceinline__
	void storeRow(uint32_t id, uint32_t ref, INDEX_TYPE row)
	{
		row_ids[id] = row;
		//printf("direct %d stores row: %d %d %d -> %d gets row %d\n", threadIdx.x, id, ref, row, id, row);
	}
	__device__ __forceinline__
		void storeEncodedRow(uint32_t id, INDEX_TYPE row)
	{
		row_ids[id] = row;
	}

	__device__ __forceinline__
	INDEX_TYPE getEncodedRow(uint32_t id)
	{
		//printf("direct %d req encoded row: %d (which is -> %d)\n", threadIdx.x, id, row_ids[id]);
		return row_ids[id];
	}

	__device__ __forceinline__
	INDEX_TYPE decodeRow(INDEX_TYPE row)
	{
		//printf("direct %d decodes row: %d -> %d\n", threadIdx.x, row, row);
		return row;
	}

	__device__ __forceinline__
	static INDEX_TYPE restartRowDecode(uint32_t restart_row, INDEX_TYPE first_row)
	{
		return first_row + restart_row;
	}
	__device__ __forceinline__
	static uint32_t restartRowEncode(INDEX_TYPE row, INDEX_TYPE first_row)
	{
		return row - first_row;
	}
};

template<typename INDEX_TYPE, uint32_t NNZ_PER_BLOCK, uint32_t THREADS>
class ARowStorage<INDEX_TYPE, NNZ_PER_BLOCK, THREADS, true>
{
	using ReferenceType = ChooseBitDataType<static_max<16,32 - count_clz<NNZ_PER_BLOCK - 1>::value>::value>;
	INDEX_TYPE row_ids[NNZ_PER_BLOCK];
	ReferenceType references[NNZ_PER_BLOCK];

	
public:

	using EncodedRowType = uint32_t;

	__device__ __forceinline__
	void clear()
	{
		#pragma unroll
		for (uint32_t i = 0; i < NNZ_PER_BLOCK; i += THREADS)
			references[i + threadIdx.x] = 0;
	}


	__device__ __forceinline__
		void storeReference(EncodedRowType id, INDEX_TYPE row)
	{
		row_ids[id] = row;
		//printf("%d stores ref: %d %d -> %d gets real row %d\n", threadIdx.x, id, row, id, row);
	}

	__device__ __forceinline__
		void storeRow(uint32_t id, EncodedRowType ref, INDEX_TYPE row)
	{
		references[id] = static_cast<ReferenceType>(ref);
		//printf("%d stores row: %d %d %d -> %d gets ref %d\n", threadIdx.x, id, ref, row, id, ref);
	}

	__device__ __forceinline__
	void storeEncodedRow(uint32_t id, EncodedRowType ref)
	{
		references[id] = static_cast<ReferenceType>(ref);
	}

	__device__ __forceinline__
		EncodedRowType getEncodedRow(uint32_t id)
	{
		//printf("%d req encoded row: %d (which is %d -> %d)\n", threadIdx.x, id, references[id], row_ids[references[id]]);
		return references[id];
	}

	__device__ __forceinline__
	INDEX_TYPE decodeRow(EncodedRowType row)
	{
		//printf("%d decodes row: %d -> %d\n", threadIdx.x, row, row_ids[row]);
		return row_ids[row];
	}

	__device__ __forceinline__
		static INDEX_TYPE restartRowDecode(EncodedRowType restart_row, INDEX_TYPE first_row)
	{
		return restart_row;
	}
	__device__ __forceinline__
	static uint32_t restartRowEncode(EncodedRowType row, INDEX_TYPE first_row)
	{
		return row;
	}
};