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

#include <cub/cub.cuh>


template<int THREADS, int ELEMENTS_PER_THREAD_IN = 1>
class WorkDistribution
{
public:
	typedef cub::BlockScan<int, THREADS> SimpleScanT;

	struct SharedMemT
	{
		int work_sum[THREADS*ELEMENTS_PER_THREAD_IN + 1];
	};

	using SharedTempMemT = typename SimpleScanT::TempStorage;

	template<int MAX_ELEMENTS_PER_THREAD_OUT = 1>
	struct SharedTempMemOutT
	{
		int work_offsets[THREADS*MAX_ELEMENTS_PER_THREAD_OUT];
	};
	

	template<bool BLOCKIN>
	__device__ __forceinline__
	static void initialize(SharedMemT& smem, SharedTempMemT& sum_space, int (&thread_work_count)[ELEMENTS_PER_THREAD_IN])
	{
		int* work_sum = smem.work_sum;
		
		if (!BLOCKIN && ELEMENTS_PER_THREAD_IN > 1)
		{
			//change from interleaved to blocked
			#pragma unroll
			for (int i = 0; i < ELEMENTS_PER_THREAD_IN; ++i)
				work_sum[threadIdx.x + i * THREADS + 1] = thread_work_count[i];
			__syncthreads();
			#pragma unroll
			for (int i = 0; i < ELEMENTS_PER_THREAD_IN; ++i)
				thread_work_count[i] = work_sum[threadIdx.x * ELEMENTS_PER_THREAD_IN + i + 1];
		}
		SimpleScanT(sum_space).InclusiveSum(thread_work_count, thread_work_count);
		#pragma unroll
		for(int i = 0; i < ELEMENTS_PER_THREAD_IN; ++i)
		{
			work_sum[threadIdx.x * ELEMENTS_PER_THREAD_IN + i + 1] = thread_work_count[i];
		}
		work_sum[0] = 0;
		__syncthreads();
	}

	template<bool BLOCKOUT, int MAX_ELEMENTS_PER_THREAD_OUT>
	__device__ __forceinline__
	static int assignWorkAllThreads(SharedMemT& smem, SharedTempMemT& sum_space, SharedTempMemOutT<MAX_ELEMENTS_PER_THREAD_OUT>& tempmem, 
		int (&work_element_out)[MAX_ELEMENTS_PER_THREAD_OUT], int(&within_element_id)[MAX_ELEMENTS_PER_THREAD_OUT], 
		int num_distribute = MAX_ELEMENTS_PER_THREAD_OUT*THREADS)
	{
		int* work_sum = smem.work_sum;
		int* work_offsets = tempmem.work_offsets;

		// clear work offsets
		#pragma unroll
		for (int i = 0; i < MAX_ELEMENTS_PER_THREAD_OUT; ++i)
			work_offsets[i*THREADS + threadIdx.x] = 0;
		
		__syncthreads();

		// compute which thread should start with a given work element
		#pragma unroll
		for (int i = 0; i < ELEMENTS_PER_THREAD_IN; ++i)
		{
			int v = work_sum[i*THREADS + threadIdx.x];
			int vn = work_sum[i*THREADS + threadIdx.x + 1];
			if (v < MAX_ELEMENTS_PER_THREAD_OUT*THREADS && v != vn)
				work_offsets[v] = i*THREADS + threadIdx.x;
		}

		__syncthreads();
		
		//compute max per thread elements
		num_distribute = min(num_distribute, work_sum[THREADS*ELEMENTS_PER_THREAD_IN]);

		// read my offset (can be the right offset or zero as only the first one will have the right per input element)
		#pragma unroll
		for (int i = 0; i < MAX_ELEMENTS_PER_THREAD_OUT; ++i)
		{
			//if (MAX_ELEMENTS_PER_THREAD_OUT*threadIdx.x + i < num_distribute)
				work_element_out[i] = work_offsets[MAX_ELEMENTS_PER_THREAD_OUT*threadIdx.x + i];
			//else
			//work_element_out[i] = 0;
		}


		SimpleScanT(sum_space).InclusiveScan(work_element_out, work_element_out, cub::Max());

		int outElements = MAX_ELEMENTS_PER_THREAD_OUT;
		if (!BLOCKOUT)
		{

			__syncthreads();

			//stripped layout requires another trip through shared..
			#pragma unroll
			for (int i = 0; i < MAX_ELEMENTS_PER_THREAD_OUT; ++i)
					work_offsets[MAX_ELEMENTS_PER_THREAD_OUT*threadIdx.x + i] = work_element_out[i];

			__syncthreads();

			// run from back to front so we can just decrese the count iif elements cross thread boundaries (same as below, just with different indices)
			#pragma unroll
			for (int i = MAX_ELEMENTS_PER_THREAD_OUT-1; i >= 0; --i)
			{
				if (i*THREADS + threadIdx.x < num_distribute)
				{
					work_element_out[i] = work_offsets[threadIdx.x + i*THREADS];
					int workoffset = (threadIdx.x + i*THREADS);
					within_element_id[i] = work_sum[work_element_out[i] + 1] - workoffset - 1;

					//if ((within_element_id[i] < 0 && i + 1 < outElements) || (workoffset >= num_distribute))
					//	outElements = i + 1;
				}
				else
				{
					outElements = i;
					work_element_out[i] = -1;
					within_element_id[i] = -1;
				}
			}
		}
		else
		{ 
			// run from back to front so we can just decrese the count iif elements cross thread boundaries
			#pragma unroll
			for (int i = MAX_ELEMENTS_PER_THREAD_OUT - 1; i >= 0; --i)
			{
				int workoffset = (MAX_ELEMENTS_PER_THREAD_OUT*threadIdx.x+i);
				within_element_id[i] = work_sum[work_element_out[i] + 1] - workoffset - 1;
				if (workoffset >= num_distribute)
					outElements = i;
			}
		}

		__syncthreads();

		// update counts
		#pragma unroll
		for (int i = 0; i < ELEMENTS_PER_THREAD_IN; ++i)
		{
			work_sum[threadIdx.x + i*THREADS + 1] = max(0,work_sum[threadIdx.x + i*THREADS + 1] - num_distribute);
			//	printf("nwork: %d %d : %d\n", blockIdx.x, threadIdx.x + i*THREADS + 1, work_sum[threadIdx.x + i*THREADS + 1]);
		}

		__syncthreads();

		return outElements;
	}

	template<bool BLOCKOUT, int MAX_ELEMENTS_PER_THREAD_OUT>
	__device__ __forceinline__
		static int assignWorkAllThreads_depricated(SharedMemT& smem, SharedTempMemT& sum_space, SharedTempMemOutT<MAX_ELEMENTS_PER_THREAD_OUT>& tempmem,
			int(&work_element_out)[MAX_ELEMENTS_PER_THREAD_OUT], int(&within_element_id)[MAX_ELEMENTS_PER_THREAD_OUT],
			uint32_t* max_A_entry, uint32_t* max_B_for_max_A_entry, int num_distribute = MAX_ELEMENTS_PER_THREAD_OUT*THREADS)
	{
		int* work_sum = smem.work_sum;
		int* work_offsets = tempmem.work_offsets;

		// clear work offsets
#pragma unroll
		for (int i = 0; i < MAX_ELEMENTS_PER_THREAD_OUT; ++i)
			work_offsets[i*THREADS + threadIdx.x] = 0;

		__syncthreads();

		// compute which thread should start with a given work element
#pragma unroll
		for (int i = 0; i < ELEMENTS_PER_THREAD_IN; ++i)
		{
			int v = work_sum[i*THREADS + threadIdx.x];
			int vn = work_sum[i*THREADS + threadIdx.x + 1];
			if (v < MAX_ELEMENTS_PER_THREAD_OUT*THREADS && v != vn)
				work_offsets[v] = i*THREADS + threadIdx.x;
		}

		__syncthreads();

		//compute max per thread elements
		num_distribute = min(num_distribute, work_sum[THREADS*ELEMENTS_PER_THREAD_IN]);

		// read my offset (can be the right offset or zero as only the first one will have the right per input element)
#pragma unroll
		for (int i = 0; i < MAX_ELEMENTS_PER_THREAD_OUT; ++i)
		{
			//if (MAX_ELEMENTS_PER_THREAD_OUT*threadIdx.x + i < num_distribute)
			work_element_out[i] = work_offsets[MAX_ELEMENTS_PER_THREAD_OUT*threadIdx.x + i];
			//else
			//work_element_out[i] = 0;
		}


		SimpleScanT(sum_space).InclusiveScan(work_element_out, work_element_out, cub::Max());

		int outElements = MAX_ELEMENTS_PER_THREAD_OUT;
		if (!BLOCKOUT)
		{

			__syncthreads();

			//stripped layout requires another trip through shared..
#pragma unroll
			for (int i = 0; i < MAX_ELEMENTS_PER_THREAD_OUT; ++i)
				work_offsets[MAX_ELEMENTS_PER_THREAD_OUT*threadIdx.x + i] = work_element_out[i];

			__syncthreads();

			// run from back to front so we can just decrese the count iif elements cross thread boundaries (same as below, just with different indices)
#pragma unroll
			for (int i = MAX_ELEMENTS_PER_THREAD_OUT - 1; i >= 0; --i)
			{
				if (i*THREADS + threadIdx.x < num_distribute)
				{
					work_element_out[i] = work_offsets[threadIdx.x + i*THREADS];
					int workoffset = (threadIdx.x + i*THREADS);
					within_element_id[i] = work_sum[work_element_out[i] + 1] - workoffset - 1;

					//TODO: needs adjustment for num_distribute
					if (max_A_entry && (threadIdx.x == THREADS - 1) && (i == (MAX_ELEMENTS_PER_THREAD_OUT - 1)))
					{
						// Set max element in A and corresponding max element in B
						*max_A_entry = work_element_out[i];
						*max_B_for_max_A_entry = within_element_id[i];
					}
					//if ((within_element_id[i] < 0 && i + 1 < outElements) || (workoffset >= num_distribute))
					//	outElements = i + 1;
				}
				else
				{
					outElements = i;
					work_element_out[i] = -1;
					within_element_id[i] = -1;
				}
			}
		}
		else
		{
			// run from back to front so we can just decrese the count iif elements cross thread boundaries
#pragma unroll
			for (int i = MAX_ELEMENTS_PER_THREAD_OUT - 1; i >= 0; --i)
			{
				int workoffset = (MAX_ELEMENTS_PER_THREAD_OUT*threadIdx.x + i);
				within_element_id[i] = work_sum[work_element_out[i] + 1] - workoffset - 1;
				if (workoffset >= num_distribute)
					outElements = i;
			}
		}

		__syncthreads();

		// update counts
#pragma unroll
		for (int i = 0; i < ELEMENTS_PER_THREAD_IN; ++i)
		{
			work_sum[threadIdx.x + i*THREADS + 1] = max(0, work_sum[threadIdx.x + i*THREADS + 1] - num_distribute);
			//	printf("nwork: %d %d : %d\n", blockIdx.x, threadIdx.x + i*THREADS + 1, work_sum[threadIdx.x + i*THREADS + 1]);
		}

		__syncthreads();

		return outElements;
	}

	__device__ __forceinline__
	static int workAvailable(SharedMemT& smem)
	{
		//if (threadIdx.x == 0)
		//	printf("%d work available: %d\n", blockIdx.x, smem.work_sum[ELEMENTS_PER_THREAD_IN*THREADS]);
		return const_cast<volatile int*>(smem.work_sum)[ELEMENTS_PER_THREAD_IN*THREADS];
	}
	__device__ __forceinline__
	static void removework(SharedMemT& smem, int amount)
	{
		#pragma unroll
		for (int i = 0; i < ELEMENTS_PER_THREAD_IN; ++i)
		{
			smem.work_sum[threadIdx.x + i*THREADS + 1] = max(0, smem.work_sum[threadIdx.x + i*THREADS + 1] - amount);
		}
	}
};