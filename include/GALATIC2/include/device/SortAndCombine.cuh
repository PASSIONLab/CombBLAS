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
 * SortAndCombine.cuh
 *
 * ac-SpGEMM
 *
 * Authors: Daniel Mlakar, Markus Steinberger, Martin Winter
 *------------------------------------------------------------------------------
*/

#pragma once
#include <stdio.h>
#include <cub/cub.cuh>

template<typename SORTINDEX_TYPE, typename VALUE_TYPE, uint32_t THREADS, uint32_t ELEMENTS_PER_THREAD>
class SortAndCombine
{
	template<typename SameElement, typename SameRow, typename SEMIRING_t>
            class CombinerOp
	{
		SameElement sameElement;
		SameRow sameRow;
		SEMIRING_t semiring;
	public:
		__device__ __forceinline__ CombinerOp(SameElement sameElement, SameRow sameRow, SEMIRING_t semiring) :
			sameElement(sameElement), 
			sameRow(sameRow),
			semiring(semiring)
		{ }
		template <typename T>
		__device__ __forceinline__  T operator()(const  T &a, const  T &b) const
		{
			//T comb;
			//uint32_t ca = a.key.get();

			//////0x1 means we have to add over
			////if (ca & 0x1)
			////	comb.value = b.value;
			////else
			////	comb.value = a.value + b.value;

			////same as above, just without conditional

			//float amul = 1.0f - __int_as_float((ca & 0x1) * __float_as_int(1.0f));
			//comb.value = a.value * amul + b.value;

			//// in case we are at the end of the combine elements, we want to increase both by one
			//uint32_t modca = ca + ((ca & 0x1) * 0x00020002);
			//// we need to add the parts that are outside of the mask
			//uint32_t amask = ((ca & 0x10000) * 0xFFFE) ^ 0xFFFEFFFE;
			//// in case a new row starts, we need to reset the front part
			//uint32_t res = (modca & amask) + b.key.get();

			//comb.key = decltype(comb.key)(res);
			//return comb;


			uint32_t newastate = (!sameRow(a.index, b.index)) ? (a.getState() & 0xFFFE) : (a.getState() & 0xFFFEFFFE);
			//decltype(a.value) amul = sameElement(a.index, b.index) ? SEMIRING_t::MultiplicativeIdentity() : SEMIRING_t::AdditiveIdentity()  ;
			return  T(b.index, semiring.add( sameElement(a.index, b.index) ?  a.value :  (SEMIRING_t::AdditiveIdentity())  , b.value), newastate + b.getState());

		}
	};
public:
	class CombResult
	{
		uint32_t state;
	public:
		SORTINDEX_TYPE index;
		VALUE_TYPE value;

		__device__ __forceinline__ CombResult() = default;

		__device__ __forceinline__ CombResult(SORTINDEX_TYPE index, VALUE_TYPE value, uint32_t state = 0) : 
			index(index), value(value), state(state)
		{ }

		__device__ __forceinline__ CombResult(SORTINDEX_TYPE index, VALUE_TYPE value, bool endElement, bool endRow) :
			index(index), value(value), state((endRow ? 0x10000 : 0) | (endElement ? 0x20003 : 0))
		{ }

		__device__ __forceinline__ uint32_t getState() const
		{
			return state;
		}
		__device__ __forceinline__ uint32_t memoffset() const
		{
			return ((state >> 1) & 0x7FFF) -1;
		}
		__device__ __forceinline__ uint32_t rowcount() const
		{
			return state >> 17;
		}
		__device__ __forceinline__ bool isResult() const
		{
			return (state & 0x1) != 0;
		}
		__device__ __forceinline__ bool isRowend() const
		{
			return ((state >> 16) & 0x1) != 0;
		}
	};


	using CUBCombIndexValueSort = cub::BlockRadixSort<SORTINDEX_TYPE, THREADS, ELEMENTS_PER_THREAD, VALUE_TYPE>;
	using ScanCombinerEntry = CombResult;
	using CUBScanCombiner = cub::BlockScan<ScanCombinerEntry, THREADS>;

	union SMem
	{
		typename CUBCombIndexValueSort::TempStorage combIndexValueSortTempMem;
		typename CUBScanCombiner::TempStorage combinerScanTempMem;
		SORTINDEX_TYPE threadFirstElementIdentifier[THREADS + 1];
	};
	
	template<typename SameElement, typename SameRow, typename SEMIRING_t>
	__device__ __forceinline__ 
	static uint32_t combine(SMem& smem,
		SORTINDEX_TYPE (&combIndex)[ELEMENTS_PER_THREAD], typename SEMIRING_t::output_t (&data)[ELEMENTS_PER_THREAD], ScanCombinerEntry(&combinedEntries)[ELEMENTS_PER_THREAD],
		SameElement sameElement, SameRow sameRow, SEMIRING_t semiring, uint32_t sortbits = sizeof(SORTINDEX_TYPE)*8)
	{

		//sort according to RowA/ColumnB (together with shared content)
		CUBCombIndexValueSort(smem.combIndexValueSortTempMem).Sort(combIndex, data, 0, sortbits);
		__syncthreads();
		
		
		//figure out who has the last element to be combined
		smem.threadFirstElementIdentifier[THREADS] = static_cast<SORTINDEX_TYPE>(-1);
		smem.threadFirstElementIdentifier[threadIdx.x] = combIndex[0];
		__syncthreads();


		SORTINDEX_TYPE c = combIndex[ELEMENTS_PER_THREAD - 1];
		SORTINDEX_TYPE oc = smem.threadFirstElementIdentifier[threadIdx.x + 1];
	
		combinedEntries[ELEMENTS_PER_THREAD - 1] = CombResult(combIndex[ELEMENTS_PER_THREAD - 1], data[ELEMENTS_PER_THREAD - 1], !sameElement(c, oc), !sameRow(c, oc));


		#pragma unroll
		for (int i = 0; i < ELEMENTS_PER_THREAD - 1; ++i)
		{
			SORTINDEX_TYPE c = combIndex[i];
			SORTINDEX_TYPE oc = combIndex[i + 1];

			combinedEntries[i] = CombResult(combIndex[i], data[i], !sameElement(c, oc), !sameRow(c, oc));
		}

		__syncthreads();


		//segmented prefix sum to add up / get mem offset for new data
		ScanCombinerEntry accumulate;
		CUBScanCombiner(smem.combinerScanTempMem).InclusiveScan(combinedEntries, combinedEntries, CombinerOp<SameElement, SameRow, SEMIRING_t>(sameElement, sameRow,semiring), accumulate);
		//uint32_t outputData = tempData + min(TEMP_PER_THREAD * THREADS, TEMP_PER_THREAD * THREADS + RowelementWorkDistribution::workAvailable(smem.workdistributionMem));
		uint32_t count = accumulate.memoffset() + 1;


		return count;
	}
};

template<uint32_t MaxChunks, size_t PerChunkBits>
struct PathMergerOp
{
	template <typename T>
	__device__ __forceinline__ T operator()(const T &a, const T &b) const
	{
		const T Mask = (1 << PerChunkBits) - 1;
		T res = 0;
		#pragma unroll
		for (uint32_t i = 0; i < MaxChunks; ++i)
		{
			T tmask = Mask << static_cast<T>(i*PerChunkBits);
			res = res | (max(a & tmask, b & tmask));
		}
		return res;
	}
};
