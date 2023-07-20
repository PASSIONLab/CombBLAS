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
 * HelperFunctions.cuh
 *
 * ac-SpGEMM
 *
 * Authors: Daniel Mlakar, Markus Steinberger, Martin Winter
 *------------------------------------------------------------------------------
*/

#pragma once

#include <cub/cub.cuh>
#include <cub/iterator/transform_input_iterator.cuh>

#include <thrust/device_ptr.h>

#include <memory>
#include <algorithm>

#include <stdint.h>
#include "../meta_utils.h"
#include "../devicetools/event.h"
#include "../MergeCaseOffsets.h"

namespace
{
	template <
		typename IndexType,
		typename ConversionOp,
		typename OffsetT = ptrdiff_t>
	class CustomGeneratorIterator
	{
	public:

		// Required iterator traits
		typedef CustomGeneratorIterator             self_type;              ///< My own type
		typedef OffsetT                             difference_type;        ///< Type to express the result of subtracting one iterator from another
		typedef typename ConversionOp::value_type   value_type;             ///< The type of the element the iterator can point to
		//typedef value_type*                        pointer;               ///< pointer not supported
		typedef value_type                           reference;              ///< The type of a reference to an element the iterator can point to

#if (THRUST_VERSION >= 100700)
																																				// Use Thrust's iterator categories so we can use these iterators in Thrust 1.7 (or newer) methods
		typedef typename thrust::detail::iterator_facade_category<
			thrust::any_system_tag,
			thrust::random_access_traversal_tag,
			value_type,
			reference
		>::type iterator_category;                                        ///< The iterator category
#else
		typedef std::random_access_iterator_tag     iterator_category;      ///< The iterator category
#endif  // THRUST_VERSION

	private:

		ConversionOp    conversion_op;
		IndexType  id;

	public:

		/// Constructor
		__host__ __device__ __forceinline__ CustomGeneratorIterator(
			ConversionOp        conversion_op,      ///< Conversion functor to wrap
			IndexType           base_id = 0)          ///< Input id to start at
			:
			conversion_op(conversion_op),
			id(base_id)
		{}

		/// Postfix increment
		__host__ __device__ __forceinline__ self_type operator++(int)
		{
			self_type retval = *this;
			++id;
			return retval;
		}

		/// Prefix increment
		__host__ __device__ __forceinline__ self_type operator++()
		{
			++id;
			return *this;
		}

		/// Indirection
		__host__ __device__ __forceinline__ reference operator*() const
		{
			return conversion_op(id);
		}

		/// Addition
		template <typename Distance>
		__host__ __device__ __forceinline__ self_type operator+(Distance n) const
		{
			self_type retval(conversion_op, id + n);
			return retval;
		}

		/// Addition assignment
		template <typename Distance>
		__host__ __device__ __forceinline__ self_type& operator+=(Distance n)
		{
			id += n;
			return *this;
		}

		/// Subtraction
		template <typename Distance>
		__host__ __device__ __forceinline__ self_type operator-(Distance n) const
		{
			self_type retval(conversion_op, id - n);
			return retval;
		}

		/// Subtraction assignment
		template <typename Distance>
		__host__ __device__ __forceinline__ self_type& operator-=(Distance n)
		{
			id -= n;
			return *this;
		}

		/// Distance
		__host__ __device__ __forceinline__ difference_type operator-(self_type other) const
		{
			return id - other.id;
		}

		/// Array subscript
		template <typename Distance>
		__host__ __device__ __forceinline__ reference operator[](Distance n) const
		{
			return conversion_op(id + n);
		}


		/// Equal to
		__host__ __device__ __forceinline__ bool operator==(const self_type& rhs)
		{
			return (id == rhs.id);
		}

		/// Not equal to
		__host__ __device__ __forceinline__ bool operator!=(const self_type& rhs)
		{
			return (id != rhs.id);
		}
	};


	template <
		typename            ValueType,
		typename            Consume,
		typename            OffsetT = ptrdiff_t>
		class CustomOutputConsumerIterator
	{
	private:

		// Proxy object
		struct Reference
		{
			ValueType* ptr;
			Consume consume;

			/// Constructor
			__host__ __device__ __forceinline__ Reference(ValueType* ptr, Consume consume) : ptr(ptr), consume(consume) {}

			/// Assignment
			__device__ __forceinline__ ValueType operator = (ValueType val)
			{
				consume(ptr, val);
				return val;
			}
		};

	public:

		// Required iterator traits
		typedef CustomOutputConsumerIterator        self_type;              ///< My own type
		typedef OffsetT                             difference_type;        ///< Type to express the result of subtracting one iterator from another
		typedef ValueType                           value_type;             ///< The type of the element the iterator can point to
		typedef ValueType*                          pointer;                ///< The type of a pointer to an element the iterator can point to
		typedef Reference                           reference;              ///< The type of a reference to an element the iterator can point to

#if (THRUST_VERSION >= 100700)
																																				// Use Thrust's iterator categories so we can use these iterators in Thrust 1.7 (or newer) methods
		typedef typename thrust::detail::iterator_facade_category<
			thrust::device_system_tag,
			thrust::random_access_traversal_tag,
			value_type,
			reference
		>::type iterator_category;                                        ///< The iterator category
#else
		typedef std::random_access_iterator_tag     iterator_category;      ///< The iterator category
#endif  // THRUST_VERSION

	private:

		Consume consume;
		pointer ptr;
		

	public:

		/// Constructor
		__host__ __device__ __forceinline__ CustomOutputConsumerIterator(
			Consume consume,
			pointer ptr = nullptr)     ///< Native pointer to wrap
			:
			consume(consume),
			ptr(ptr)
		{}

		/// Postfix increment
		__host__ __device__ __forceinline__ self_type operator++(int)
		{
			self_type retval = *this;
			ptr++;
			return retval;
		}


		/// Prefix increment
		__host__ __device__ __forceinline__ self_type operator++()
		{
			ptr++;
			return *this;
		}

		/// Indirection
		__host__ __device__ __forceinline__ reference operator*() const
		{
			return Reference(ptr, consume);
		}

		/// Addition
		template <typename Distance>
		__host__ __device__ __forceinline__ self_type operator+(Distance n) const
		{
			self_type retval(consume, ptr + n);
			return retval;
		}

		/// Addition assignment
		template <typename Distance>
		__host__ __device__ __forceinline__ self_type& operator+=(Distance n)
		{
			ptr += n;
			return *this;
		}

		/// Subtraction
		template <typename Distance>
		__host__ __device__ __forceinline__ self_type operator-(Distance n) const
		{
			self_type retval(consume, ptr - n);
			return retval;
		}

		/// Subtraction assignment
		template <typename Distance>
		__host__ __device__ __forceinline__ self_type& operator-=(Distance n)
		{
			ptr -= n;
			return *this;
		}

		/// Distance
		__host__ __device__ __forceinline__ difference_type operator-(self_type other) const
		{
			return ptr - other.ptr;
		}

		/// Array subscript
		template <typename Distance>
		__host__ __device__ __forceinline__ reference operator[](Distance n) const
		{
			return Reference(ptr + n, consume);
		}

		/// Equal to
		__host__ __device__ __forceinline__ bool operator==(const self_type& rhs)
		{
			return (ptr == rhs.ptr);
		}

		/// Not equal to
		__host__ __device__ __forceinline__ bool operator!=(const self_type& rhs)
		{
			return (ptr != rhs.ptr);
		}
	};


	template<class INDEX_TYPE, INDEX_TYPE MergeMaxElements, INDEX_TYPE MaxMergeChunks, class OUT_TYPE, int BITS, int NUM, int OFFSET>
	class CaseCombinerConverter
	{
		const INDEX_TYPE* const maxPerRowElements;
		const uint32_t* const sharedRows;
		const uint32_t* const chunkCounter;
	public:

		typedef OUT_TYPE value_type;

		__host__ __device__ __forceinline__
			CaseCombinerConverter(const uint32_t* sharedRows, const INDEX_TYPE* maxPerRowElements, const uint32_t* chunkCounter) :
			maxPerRowElements(maxPerRowElements),
			sharedRows(sharedRows),
			chunkCounter(chunkCounter)
		{ }

		__host__ __device__ __forceinline__
		OUT_TYPE operator()(const uint32_t &id) const
		{
			uint32_t row = sharedRows[id];
			uint32_t chunks = chunkCounter[row];
			//INDEX_TYPE elementCounter = maxPerRowElements[row];
			int type = 2;
			if (chunks == 2 && maxPerRowElements[row] < MergeMaxElements)
				type = 0;
			else if ((chunks & (~MAX_CHUNKS_CASE)) < MaxMergeChunks && (chunks & CASE_DISTINCTION) == 0 /*&& false*/)
				type = 1;

			if (type < OFFSET || type >= OFFSET + NUM)
				return 0;
			OUT_TYPE	res = ((OUT_TYPE(1) << (BITS-1)) | OUT_TYPE(1)) << (BITS*(type - OFFSET));
			/*if (blockIdx.x == 0)
				printf("%d (%d) convert %d (%d) to %llx\n", id, sharedRows[id], elementCounter, type, (uint64_t)res);*/
			return res;
		}
	};

	template<class INDEX_TYPE, class IN_TYPE, int BITS, int NUM, int OFFSET>
	class CaseSeparatorConsumer
	{
		INDEX_TYPE* const outputPointers;
		INDEX_TYPE* const counters;
		INDEX_TYPE* const row_counts;
		const uint32_t* const sharedRows;
		const uint32_t activeRows;
	public:
		__host__ __device__ __forceinline__
			CaseSeparatorConsumer(const uint32_t* sharedRows, INDEX_TYPE* outputPointers, INDEX_TYPE* counters, uint32_t activeRows, INDEX_TYPE* row_counts) :
			outputPointers(outputPointers),
			counters(counters),
			sharedRows(sharedRows),
			row_counts{ row_counts },
			activeRows(activeRows)
		{ }

		__host__ __device__ __forceinline__
		void operator()(IN_TYPE* virtualOffset, const IN_TYPE sumresult) const
		{
			int type = -1;
			INDEX_TYPE offset = 0;
			const IN_TYPE mask = IN_TYPE(1) << (BITS - 1);
			const IN_TYPE select = mask - 1;

			IN_TYPE* virtualBase = nullptr;
			uint32_t dist = virtualOffset - virtualBase;
			if (dist == activeRows - 1)
			{
				// final writes counts
				#pragma unroll
				for (int i = 0; i < NUM; ++i)
				{
					if (i + OFFSET < 3)
					{ 
						INDEX_TYPE sum = static_cast<INDEX_TYPE>((sumresult >> (i*BITS)) & select);
						counters[OFFSET + i] = sum;
					}
				}
				counters[3] = 0;
				outputPointers[(activeRows) * 3] = 0;
			}

			#pragma unroll
			for (int i = 0; i < NUM; ++i)
			{
				if ((sumresult & (mask << (i*BITS))) != 0)
				{
					type = i;
					offset = static_cast<INDEX_TYPE>((sumresult >> (i*BITS)) & select);
				}
			}
			if (type == -1)
			{
				//if(blockIdx.x == 0)
				//	printf("%d %d: %llx would not write\n", blockIdx.x, threadIdx.x, (uint64_t)sumresult);
				return;
			}

			type += OFFSET;
			//if (blockIdx.x == 0)
			//	printf("%d %d: %llx would write %d to %d(%d) (%llx, %llx, %d   + %d)\n", blockIdx.x, threadIdx.x, (uint64_t)sumresult, offset, dist, type, outputPointers, counters, activeRows, activeRows*type + dist + 1);
			//printf("%d %d: %llx writinting %d (%d) to %d (%d*%d + %d -1)\n", blockIdx.x, threadIdx.x, (uint64_t)sumresult, sharedRows[dist], dist, 
			//	(activeRows)*type + offset - 1, activeRows, type,offset);
			/*if (type == 0 && row_counts[sharedRows[dist]] > 1024)
				printf("RowCount at position %u is : %u\n", sharedRows[dist], row_counts[sharedRows[dist]]);*/
			outputPointers[(activeRows)*type + offset-1] = sharedRows[dist];
		}
	};

	template<class TYPE, TYPE MASK>
	struct CombinedAdd
	{
		/// Boolean max operator, returns <tt>(a > b) ? a : b</tt>
		template <typename T>
		__host__ __device__ __forceinline__ T operator()(const T &a, const T &b) const
		{
			return (a & MASK) + b;
		}
	};


	struct BlockOffsetRange
	{
		uint32_t begin, end;
		uint32_t count;
	};

	template<class INDEX_TYPE>
	class BlockOffsetCreator
	{
		const INDEX_TYPE* const maxPerRowElements;
		const INDEX_TYPE* const sharedRows;
	public:

		typedef BlockOffsetRange value_type;

		__host__ __device__ __forceinline__
			BlockOffsetCreator(const INDEX_TYPE* sharedRows, const INDEX_TYPE* maxPerRowElements) :
			maxPerRowElements(maxPerRowElements),
			sharedRows(sharedRows)
		{ }

		__host__ __device__ __forceinline__
		BlockOffsetRange operator()(const uint32_t &id) const
		{
			/*if(maxPerRowElements[sharedRows[id]] > 1024)*/
			//printf("%d creating range: (row %d) %d-%d with %d\n", id, sharedRows[id], id, id + 1, maxPerRowElements[sharedRows[id]]);
			return BlockOffsetRange{ id, id + 1, maxPerRowElements[sharedRows[id]] };
		}
	};

	template<uint32_t SimpleMergeThreads, uint32_t MaxComb>
	struct BlockOffsetCombiner
	{
		template <typename T>
		__host__ __device__ __forceinline__ T operator()(const T &a, const T &b) const
		{
			//we have to consider the maximum number elements the block can hold (Max Comb)
			//and we have to consider the maximum number shared rows the block can handle (SimpleMergeThreads - 1) -> < SimpleMergeThreads
			if (a.end == b.begin && a.count + b.count < MaxComb && b.end - a.begin < SimpleMergeThreads)
			{
				//printf("merging: %d<>%d %d<>%d %d + %d < %d\n", a.begin, a.end, b.begin, b.end, a.count, b.count, MaxComb);
				return BlockOffsetRange{ a.begin, b.end, a.count + b.count};
			}
			else
			{
				//printf("not merging: %d<>%d %d<>%d %d + %d < %d\n", a.begin, a.end, b.begin, b.end, a.count, b.count, MaxComb);
			}
			//if (b.count >= 1024)
			//	printf("B.count: %u is too large\n", b.count);
			return b;
		}
	};

	class BlockOffsetExtractor
	{
		uint2* const rangeOut;
	public:
		__host__ __device__ __forceinline__
			BlockOffsetExtractor(uint2* rangeOut) :
			rangeOut(rangeOut)
		{ }

		__host__ __device__ __forceinline__
			void operator()(BlockOffsetRange* virtualOffset, const BlockOffsetRange result) const
		{
			BlockOffsetRange* virtualBase = nullptr;
			uint32_t dist = virtualOffset - virtualBase;
			/*if(result.count > 1024)*/
				//printf("%d writing range (%d<>%d) %d | %u\n", dist, result.begin, result.end, result.count, result.test);
			rangeOut[dist] = uint2{ result.begin, result.end };
		}
	};


	class RangeStartTranslator
	{
		const uint2* __restrict__ ranges;
		const uint32_t activeRows;
	public:

		typedef uint32_t value_type;

		__host__ __device__ __forceinline__
			RangeStartTranslator(const uint2* ranges, uint32_t activeRows) :
			ranges(ranges),
			activeRows(activeRows)
		{ }

		__host__ __device__ __forceinline__
			uint32_t operator()(const uint32_t &id) const
		{
			uint32_t res = 0x80000001;
			if (id < activeRows - 1)
			{
				if (ranges[id].x == ranges[id + 1].x)
					res = 0;
			}
			
			//if(res != 0)
			//	printf("%d is a block end (%d<>%d)\n", id, ranges[id].x, ranges[id].y);

			return res;
		}
	};

	template<class INDEX_TYPE>
	class BlockStartWriter
	{
		INDEX_TYPE* const blockOffsets;
		INDEX_TYPE* counter;
		const uint32_t activeRows;
	public:
		__host__ __device__ __forceinline__
			BlockStartWriter(INDEX_TYPE* blockOffsets, INDEX_TYPE* counter, uint32_t activeRows) :
			blockOffsets(blockOffsets),
			counter(counter),
			activeRows(activeRows)
		{ }

		__host__ __device__ __forceinline__
			void operator()(uint32_t* virtualOffset, const uint32_t result) const
		{
			uint32_t* z = nullptr;
			INDEX_TYPE d = virtualOffset - z;
			if (d == activeRows - 1)
			{
				*counter = (result & (~0x80000000));
			}
			if (result & 0x80000000)
			{
				blockOffsets[result & (~0x80000000)] = d+1;
				//printf("writing block offset %d : %d\n", result & (~0x80000000), d+1);
			}
		}
	};	

	struct PinnedHostMemDeleter
	{
		void operator()(void* p) const noexcept
		{
			cudaFreeHost(p);
		}
	};

	template<typename T>
	inline auto allocHostMemory(size_t elements = 1)
	{
		void* p;
		cudaMallocHost(&p, elements * sizeof(T));
		return std::unique_ptr<T, PinnedHostMemDeleter>(static_cast<T*>(p));
	}
}

namespace std
{
	template<typename IndexType, typename ConversionOp, typename OffsetT>
	struct iterator_traits<CustomGeneratorIterator<IndexType, ConversionOp, OffsetT>>
	{
		typedef typename CustomGeneratorIterator<IndexType, ConversionOp, OffsetT>::value_type value_type;
	};
}

template<class INDEX_TYPE>
size_t AcSpGEMMKernels::tempMemSize(size_t CRows)
{
	void *d_temp_storage = nullptr;
	size_t temp_storage_bytes = 0;
	size_t temp_storage_bytes2 = 0;
	INDEX_TYPE *in = nullptr, *out = nullptr;
	uint64_t *in64 = nullptr, *out64 = nullptr;
	cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, in, out, CRows + 1);
	cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes2, in64, out64, CRows + 1);
	temp_storage_bytes = std::max(temp_storage_bytes, temp_storage_bytes2);

	CustomGeneratorIterator<uint32_t, BlockOffsetCreator<INDEX_TYPE>> initr(BlockOffsetCreator<INDEX_TYPE>(nullptr, nullptr));
	CustomOutputConsumerIterator<BlockOffsetRange, BlockOffsetExtractor> outitr(BlockOffsetExtractor(nullptr));
	cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes2, initr, outitr, BlockOffsetCombiner<1024, 1024>(), CRows);
	temp_storage_bytes = std::max(temp_storage_bytes, temp_storage_bytes2);

	CustomGeneratorIterator<uint32_t, RangeStartTranslator> initr2(RangeStartTranslator(nullptr, 0));
	CustomOutputConsumerIterator<uint32_t, BlockStartWriter<INDEX_TYPE>> outitr2(BlockStartWriter<INDEX_TYPE>(nullptr, nullptr, 0));
	cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes2, initr2, outitr2, CombinedAdd<uint32_t, ~0x80000000>(), CRows);

	return std::max(temp_storage_bytes, temp_storage_bytes2) + (CRows * sizeof(uint2) + 15) / 16 * 16 + 16;
}

 template<class INDEX_TYPE, INDEX_TYPE MaxMergeChunks, INDEX_TYPE MergeMaxElements, uint32_t SimpleMergeBlockSize>
 MergeCaseOffsets AcSpGEMMKernels::assignCombineBlocks(size_t activeRows, void* tempMem, size_t tempMemSize, uint32_t* sharedRows, CUdeviceptr maxPerRowElements, uint32_t* chunckCounter, CUdeviceptr per_block_offsets, CUdeviceptr num_merge_blocks, CUstream stream, CUstream overlapStream)
 {
 	//const INDEX_TYPE MaxBlockLoad = DoubleMaxBlockLoad;
 	size_t outtempsize = (activeRows * sizeof(uint2) + 15) / 16 * 16;
 	size_t adjtempsize = tempMemSize - outtempsize;
	void* temporaryMem = reinterpret_cast<void*>(reinterpret_cast<uint64_t>(tempMem) + outtempsize);

 	static auto blockCounter = allocHostMemory<INDEX_TYPE>(4);

 	//sum and offsets of merges
 	if (activeRows < ((1u<<9)-1))
 	{
 		// use 8 bits
 		CaseCombinerConverter<INDEX_TYPE, MergeMaxElements, MaxMergeChunks, uint32_t, 10, 3, 0> inConv(sharedRows, reinterpret_cast<INDEX_TYPE*>(maxPerRowElements), chunckCounter);
 		CustomGeneratorIterator<uint32_t, CaseCombinerConverter<INDEX_TYPE, MergeMaxElements, MaxMergeChunks, uint32_t, 10, 3, 0>> initr(inConv);

 		CaseSeparatorConsumer<INDEX_TYPE, uint32_t, 10, 3, 0> outConv(sharedRows, reinterpret_cast<INDEX_TYPE*>(per_block_offsets), reinterpret_cast<INDEX_TYPE*>(num_merge_blocks), activeRows, reinterpret_cast<INDEX_TYPE*>(maxPerRowElements));
 		CustomOutputConsumerIterator<uint32_t, CaseSeparatorConsumer<INDEX_TYPE, uint32_t, 10, 3, 0>> outitr(outConv);

 		CombinedAdd<uint32_t, (((1u << 9) - 1)<<20) | (((1u << 9) - 1)<<10) | ((1u << 9) - 1)> comb;
 		cub::DeviceScan::InclusiveScan(temporaryMem, adjtempsize, initr, outitr, comb, activeRows, stream);
 	}
 	else if (activeRows < ((1u << 20) - 1))
 	{
 		// use 16 bit
 		CaseCombinerConverter<INDEX_TYPE, MergeMaxElements, MaxMergeChunks, uint64_t, 21, 3, 0> inConv(sharedRows, reinterpret_cast<INDEX_TYPE*>(maxPerRowElements), chunckCounter);
 		CustomGeneratorIterator<uint64_t, CaseCombinerConverter<INDEX_TYPE, MergeMaxElements, MaxMergeChunks, uint64_t, 21, 3, 0>> initr(inConv);

 		CaseSeparatorConsumer<INDEX_TYPE, uint64_t, 21, 3, 0> outConv(sharedRows, reinterpret_cast<INDEX_TYPE*>(per_block_offsets), reinterpret_cast<INDEX_TYPE*>(num_merge_blocks), activeRows, reinterpret_cast<INDEX_TYPE*>(maxPerRowElements));
 		CustomOutputConsumerIterator<uint64_t, CaseSeparatorConsumer<INDEX_TYPE, uint64_t, 21, 3, 0>> outitr(outConv);

 		CombinedAdd<uint64_t, (((1ull << 20) - 1) << 42) | (((1ull << 20) - 1) << 21) | ((1ull << 20) - 1)> comb;
 		cub::DeviceScan::InclusiveScan(temporaryMem, adjtempsize, initr, outitr, comb, activeRows, stream);
 	}
 	else
 	{
 		// use 32 bit triple call
 		CombinedAdd<uint32_t, ~0x80000000> comb;

 		CaseCombinerConverter<INDEX_TYPE, MergeMaxElements, MaxMergeChunks, uint32_t, 32, 1, 0> inConv0(sharedRows, reinterpret_cast<INDEX_TYPE*>(maxPerRowElements), chunckCounter);
 		CustomGeneratorIterator<uint32_t, CaseCombinerConverter<INDEX_TYPE, MergeMaxElements, MaxMergeChunks, uint32_t, 32, 1, 0>> initr0(inConv0);

 		CaseSeparatorConsumer<INDEX_TYPE, uint32_t, 32, 1, 0> outConv0(sharedRows, reinterpret_cast<INDEX_TYPE*>(per_block_offsets), reinterpret_cast<INDEX_TYPE*>(num_merge_blocks), activeRows, reinterpret_cast<INDEX_TYPE*>(maxPerRowElements));
 		CustomOutputConsumerIterator<uint32_t, CaseSeparatorConsumer<INDEX_TYPE, uint32_t, 32, 1, 0>> outitr0(outConv0);
 		cub::DeviceScan::InclusiveScan(temporaryMem, adjtempsize, initr0, outitr0, comb, activeRows, stream);

 		CaseCombinerConverter<INDEX_TYPE, MergeMaxElements, MaxMergeChunks, uint32_t, 32, 1, 1> inConv1(sharedRows, reinterpret_cast<INDEX_TYPE*>(maxPerRowElements), chunckCounter);
 		CustomGeneratorIterator<uint32_t, CaseCombinerConverter<INDEX_TYPE, MergeMaxElements, MaxMergeChunks, uint32_t, 32, 1, 1>> initr1(inConv1);

 		CaseSeparatorConsumer<INDEX_TYPE, uint32_t, 32, 1, 1> outConv1(sharedRows, reinterpret_cast<INDEX_TYPE*>(per_block_offsets), reinterpret_cast<INDEX_TYPE*>(num_merge_blocks), activeRows, reinterpret_cast<INDEX_TYPE*>(maxPerRowElements));
 		CustomOutputConsumerIterator<uint32_t, CaseSeparatorConsumer<INDEX_TYPE, uint32_t, 32, 1, 1>> outitr1(outConv1);
 		cub::DeviceScan::InclusiveScan(temporaryMem, adjtempsize, initr1, outitr1, comb, activeRows, stream);

 		CaseCombinerConverter<INDEX_TYPE, MergeMaxElements, MaxMergeChunks, uint32_t, 32, 1, 2> inConv2(sharedRows, reinterpret_cast<INDEX_TYPE*>(maxPerRowElements), chunckCounter);
 		CustomGeneratorIterator<uint32_t, CaseCombinerConverter<INDEX_TYPE, MergeMaxElements, MaxMergeChunks, uint32_t, 32, 1, 2>> initr2(inConv2);

 		CaseSeparatorConsumer<INDEX_TYPE, uint32_t, 32, 1, 2> outConv2(sharedRows, reinterpret_cast<INDEX_TYPE*>(per_block_offsets), reinterpret_cast<INDEX_TYPE*>(num_merge_blocks), activeRows, reinterpret_cast<INDEX_TYPE*>(maxPerRowElements));
 		CustomOutputConsumerIterator<uint32_t, CaseSeparatorConsumer<INDEX_TYPE, uint32_t, 32, 1, 2>> outitr2(outConv2);
 		cub::DeviceScan::InclusiveScan(temporaryMem, adjtempsize, initr2, outitr2, comb, activeRows, stream);
 	}

 	cudaMemcpy(blockCounter.get(), reinterpret_cast<void*>(num_merge_blocks), 3 * sizeof(INDEX_TYPE), cudaMemcpyDeviceToHost);
 	uint32_t combSharedRows = blockCounter.get()[0];

 	{
 		BlockOffsetCreator<INDEX_TYPE> rangeCreator(reinterpret_cast<INDEX_TYPE*>(per_block_offsets), reinterpret_cast<INDEX_TYPE*>(maxPerRowElements));
 		CustomGeneratorIterator<uint32_t, BlockOffsetCreator<INDEX_TYPE>> initr(rangeCreator);

 		BlockOffsetCombiner<SimpleMergeBlockSize, MergeMaxElements> comb;

 		BlockOffsetExtractor rangeExtractor(reinterpret_cast<uint2*>(tempMem));
 		CustomOutputConsumerIterator<BlockOffsetRange, BlockOffsetExtractor> outitr(rangeExtractor);
 		cub::DeviceScan::InclusiveScan(temporaryMem, adjtempsize, initr, outitr, comb, combSharedRows, stream);
 	}

 	{
 		RangeStartTranslator rangeCreator(reinterpret_cast<uint2*>(tempMem), combSharedRows);
 		CustomGeneratorIterator<uint32_t, RangeStartTranslator> initr(rangeCreator);

 		CombinedAdd<uint32_t, ~0x80000000> comb;

 		BlockStartWriter<INDEX_TYPE> rangeExtractor(reinterpret_cast<INDEX_TYPE*>(per_block_offsets) + 3 * activeRows, reinterpret_cast<INDEX_TYPE*>(num_merge_blocks) + 3, combSharedRows);
 		CustomOutputConsumerIterator<uint32_t, BlockStartWriter<INDEX_TYPE>> outitr(rangeExtractor);
 		cub::DeviceScan::InclusiveScan(temporaryMem, adjtempsize, initr, outitr, comb, combSharedRows, stream);
 	}

	cudaMemcpy(blockCounter.get(), reinterpret_cast<void*>(num_merge_blocks), 4 * sizeof(INDEX_TYPE), cudaMemcpyDeviceToHost);

 	return MergeCaseOffsets(blockCounter.get()[3], blockCounter.get()[1], blockCounter.get()[2], blockCounter.get()[0]);
 }

 template<class INDEX_TYPE>
 void AcSpGEMMKernels::computeRowOffsets(size_t Crows, void* tempMem, size_t tempMemSize, CUdeviceptr inout, CUstream stream)
 {
 	INDEX_TYPE* workmem = reinterpret_cast<INDEX_TYPE*>(inout);
 	cub::DeviceScan::ExclusiveSum(tempMem, tempMemSize, workmem, workmem, Crows + 1, stream);
 }

__forceinline__ __device__ unsigned laneid()
{
    unsigned ret; 
    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(ret));
    return ret;
}

template<class T, int N>
__device__ __forceinline__ void updateMinValue(T &sv, T(&values)[N], int num = N)
{
	typename cub::WarpReduce< T >::TempStorage nosmem;
	T v = sv;
	#pragma unroll
	for (int i = 0; i < N; ++i)
		if (i < num)
			v = min(v, values[i]);

	T res = cub::WarpReduce< T >(nosmem).Reduce(v, cub::Min());
	if (laneid() == 0)
		atomicMin(&sv, res);
}

template<class T>
__device__ __forceinline__ void updateMinValue(T &sv, T v)
{
	typename cub::WarpReduce< T >::TempStorage nosmem;
	T res = cub::WarpReduce< T >(nosmem).Reduce(v, cub::Min());
	if (laneid() == 0)
		atomicMin(&sv, res);
}

template<class T>
__device__ __forceinline__ void updateMaxValue(T &sv, T v)
{
	typename cub::WarpReduce< T >::TempStorage nosmem;

	T res = cub::WarpReduce< T >(nosmem).Reduce(v, cub::Max());
	if (laneid() == 0)
		atomicMax(&sv, res);
}

template<class T, int N>
__device__ __forceinline__ void updateMaxValue(T &sv, T(&values)[N], int num = N)
{
	typename cub::WarpReduce< T >::TempStorage nosmem;
	T v = sv;
	#pragma unroll
	for (int i = 0; i < N; ++i)
		if (i < num)
			v = max(v, values[i]);

	T res = cub::WarpReduce< T >(nosmem).Reduce(v, cub::Max());
	if (laneid() == 0)
		atomicMax(&sv, res);
}

template<uint32_t X, int Completed = 0>
struct count_clz
{
	static const uint32_t value = (X & 0x80000000) ? Completed : static_clz< (X << 1), Completed + 1 >::value;
};
template<uint32_t X>
struct count_clz<X, 32>
{
	static const uint32_t value = 32;
};

template<uint32_t BITS>
struct ChooseBitDataTypeRounded;
template<>
struct ChooseBitDataTypeRounded<8>
{
	using type = uint8_t;
};
template<>
struct ChooseBitDataTypeRounded<16>
{
	using type = uint16_t;
};
template<>
struct ChooseBitDataTypeRounded<32>
{
	using type = uint32_t;
};
template<>
struct ChooseBitDataTypeRounded<64>
{
	using type = uint64_t;
};

template<uint32_t BITS>
struct ChooseBitDataTypeRounding
{
	using type = typename ChooseBitDataTypeRounded<BITS <= 8 ? 8 : BITS <= 16 ? 16 : BITS <= 32 ? 32 : BITS <= 64 ? 64 : BITS <= 96 ? 96 : 128>::type;
};

template<uint32_t BITS>
using  ChooseBitDataType = typename ChooseBitDataTypeRounding<BITS>::type;


template<typename VALUE_TYPE, typename INDEX_TYPE>
__device__ __forceinline__
void addPotentiallySharedRow(uint32_t row, Chunk<VALUE_TYPE, INDEX_TYPE> * chunk, bool first_row,
	void** output_row_list_heads, uint32_t* shared_rows_tracker, uint32_t* shared_rows_alloc, bool force_addlist = false)
{
	unsigned long long* rlh = reinterpret_cast<unsigned long long*>(output_row_list_heads);
	unsigned long long c = reinterpret_cast<unsigned long long>(chunk) | (first_row?2:0);
	uint64_t next = atomicExch(rlh + row, c);
	bool addlist = false;
	if (next == 0)
	{
		if(force_addlist)
			addlist = true;
		else
		{
			//we are first, so mark that next needs to add to list
			uint64_t set = atomicCAS(rlh + row, c, (c | 0x1));
			if (set != c)
				//someone else added to the list before we could mark for setting shared list, so we have to do it
				addlist = true;
		}
	}
	else if ((next & 0x1) != 0)
	{
		addlist = true;
		next = next & 0xFFFFFFFFFFFFFFFEULL;
	}

	chunk->writeNextPointer(reinterpret_cast<Chunk<VALUE_TYPE, INDEX_TYPE>*>(next), first_row);

	uint32_t p = static_cast<uint32_t>(-1);
	if (addlist)
	{
		p = atomicAdd(shared_rows_alloc, 1);
		shared_rows_tracker[p] = row;
	}
}

// ########################################################################
// Explicit instantiations
// ########################################################################
template size_t AcSpGEMMKernels::tempMemSize<uint32_t>(size_t CRows);
template void AcSpGEMMKernels::computeRowOffsets<uint32_t>(size_t Crows, void* tempMem, size_t tempMemSize, CUdeviceptr inout, CUstream stream);
#define GPUCompressedMatrixMatrixMultiplyHelper(THREADS, TEMPPERTHREAD, MERGEMAXCHUNKS) \
template MergeCaseOffsets AcSpGEMMKernels::assignCombineBlocks<uint32_t, MERGEMAXCHUNKS, (TEMPPERTHREAD * 2) * THREADS, THREADS>(size_t activeRows, void* tempMem, size_t tempMemSize, uint32_t* sharedRows, CUdeviceptr maxPerRowElements, uint32_t* chunckCounter, CUdeviceptr per_block_offsets, CUdeviceptr num_merge_blocks, CUstream stream, CUstream overlapStream);

