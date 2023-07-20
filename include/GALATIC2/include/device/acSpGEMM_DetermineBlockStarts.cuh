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
 * DetermineBlockStarts.cuh
 *
 * ac-SpGEMM
 *
 * Authors: Daniel Mlakar, Markus Steinberger, Martin Winter
 *------------------------------------------------------------------------------
*/

#pragma once

#include "MultiplyKernels.h"
#include "../common.h"


template<typename OFFSET_TYPE, uint32_t NNZ_PER_BLOCK>
__global__ void DetermineBlockStarts(int num_other, const OFFSET_TYPE*__restrict offsets, uint32_t* startingIds, 
	uint64_t* toClear, uint32_t* toClear1, uint32_t* toClear2, int num3, uint32_t* toClear3, int num4, uint32_t* toClear4,
	int num5, uint32_t* toClear5, uint32_t* toClear6, uint32_t* toClear7, int num8, uint32_t* toClear8)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id > num_other)
		return;

	int a = offsets[id];
	int b = offsets[min(id + 1, num_other)];

	int blocka = divup<int>(a, NNZ_PER_BLOCK);
	int blockb = (b - 1) / static_cast<int>(NNZ_PER_BLOCK);

	//iterate over all blocks that start with that row
	for (; blocka <= blockb; ++blocka)
		startingIds[blocka] = id;

	//write last
	if (id == num_other)
		startingIds[divup<int>(b, NNZ_PER_BLOCK)] = id - 1;
	else
	{
		toClear[id] = 0,
		toClear1[id] = 0;
	}	
	toClear2[id] = 0;

	for (int i = id; i < num3; i+=num_other)
	{
		toClear3[i] = 0;
	}
	
	for (int i = id; i < num4; i += num_other)
	{
		toClear4[i] = 0;
	}

	for (int i = id; i < num5; i += num_other)
	{
		toClear5[i] = 0;
		toClear6[i] = 0;
		toClear7[i] = 0;
	}

	for (int i = id; i < num8; i += num_other)
	{
		toClear8[i] = 0;
	}
}

template<typename OFFSET_TYPE, uint32_t NNZ_PER_BLOCK>
void AcSpGEMMKernels::h_DetermineBlockStarts(int num_other, const uint32_t*__restrict offsets, uint32_t* startingIds, uint64_t* toClear, uint32_t* toClear1, uint32_t* toClear2, int num3, uint32_t* toClear3, int num4, uint32_t* toClear4,
	int num5, uint32_t* toClear5, uint32_t* toClear6, uint32_t* toClear7, int num8, uint32_t* toClear8)
{
	DetermineBlockStarts <OFFSET_TYPE, NNZ_PER_BLOCK> <<<gridDim, blockDim, 0 , stream>>>(num_other, offsets, startingIds, toClear, toClear1, toClear2, num3, toClear3,
		num4, toClear4,
		num5, toClear5, toClear6, toClear7, 
		num8, toClear8);
}


#define GPUCompressedMatrixMatrixMultiplyBlockStarts(THREADS, NNZPERTHREAD) \
	template void AcSpGEMMKernels::h_DetermineBlockStarts<uint32_t, THREADS*NNZPERTHREAD>(int num_other, const uint32_t*__restrict offsets, uint32_t* startingIds, uint64_t* toClear, uint32_t* toClear1, uint32_t* toClear2, int num3, uint32_t* toClear3, int num4, uint32_t* toClear4, int num5, uint32_t* toClear5, uint32_t* toClear6, uint32_t* toClear7, int num8, uint32_t* toClear8);

