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


#ifndef INCLUDED_CUDA_MEMORY
#define INCLUDED_CUDA_MEMORY

#pragma once

#include <cstddef>

#include <cuda_runtime.h>

#include "../../include/devicetools/unique_handle.h"


namespace CU
{
	struct MemFreeDeleter
	{
		void operator ()(CUdeviceptr ptr) const
		{
			cudaFree(reinterpret_cast<void*>(ptr));
		}
	};
	
	using unique_ptr = unique_handle<CUdeviceptr, 0ULL, MemFreeDeleter>;
	
	
	struct pitched_memory
	{
		pitched_memory(const pitched_memory&) = delete;
		pitched_memory& operator =(const pitched_memory&) = delete;
		
		unique_ptr memory;
		std::size_t pitch;
		
		pitched_memory() {}
		
		pitched_memory(unique_ptr memory, std::size_t pitch)
			: memory(std::move(memory)),
			  pitch(pitch)
		{
		}
		
		pitched_memory(pitched_memory&& m)
			: memory(std::move(m.memory)),
			  pitch(m.pitch)
		{
		}
		
		pitched_memory& operator =(pitched_memory&& m)
		{
			using std::swap;
			swap(memory, m.memory);
			pitch = m.pitch;
			return *this;
		}
	};
	
	
	unique_ptr allocMemory(std::size_t size);
	unique_ptr allocMemoryPitched(std::size_t& pitch, std::size_t row_size, std::size_t num_rows, unsigned int element_size);
	pitched_memory allocMemoryPitched(std::size_t row_size, std::size_t num_rows, unsigned int element_size);
}

#endif  // INCLUDED_CUDA_MEMORY
