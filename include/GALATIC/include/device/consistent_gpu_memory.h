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

#pragma once

#include <utility>
#include "../../include/devicetools/memory.h"
#include "../memory_space.h"
#include "../consistent_memory.h"

namespace ACSpGEMM {
	template<>
	class ConsistentMemory<MemorySpace::device> : RegisteredMemory
	{
		size_t _size;
		CU::unique_ptr _ptr;

		size_t clear() override
		{
			auto s = _size;
			reset(0);
			return s;
		}
	public:
		ConsistentMemory() : _size(0)
		{
			register_consistent_memory(this);
		}

		~ConsistentMemory()
		{
			unregister_consistent_memory(this);
		}

		operator CUdeviceptr() const noexcept { return _ptr; }

		template <typename T = void>
		T* get() const noexcept { return reinterpret_cast<T*>(_ptr.operator long long unsigned int()); }

		void increaseMemRetainData(size_t size)
		{
			CU::unique_ptr tmp_ptr = CU::allocMemory(_size + size);
			cudaMemcpy(tmp_ptr.get(), _ptr.get(), _size, cudaMemcpyDeviceToDevice);
			_ptr.reset();
			_ptr = std::move(tmp_ptr);
			_size += size;
		}

		void assure(size_t size)
		{
			if (size > _size)
			{
				_ptr.reset();
				_ptr = CU::allocMemory(size);
				_size = size;
			}
		}
		void reset(size_t size = 0)
		{
			_ptr.reset();
			_size = 0;
			assure(size);
		}
	};
}
