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

#include <vector>
#include <algorithm>
#include "memory_space.h"

namespace ACSpGEMM {
	class RegisteredMemory
	{
	public:
		virtual size_t clear() = 0;
	};

	inline std::vector<RegisteredMemory*>& getRegMemories()
	{
		static std::vector<RegisteredMemory*> m;
		return m;
	}

	inline void register_consistent_memory(RegisteredMemory* memory)
	{
		getRegMemories().push_back(memory);
	}
	inline void unregister_consistent_memory(RegisteredMemory* memory)
	{
		auto &m = getRegMemories();
		std::remove(begin(m), end(m), memory);
	}
	inline size_t clear_consistentMemory()
	{
		size_t s = 0;
		for (auto m : getRegMemories())
			s += m->clear();
		return s;
	}

	template<MemorySpace>
	class ConsistentMemory;

	template<class T>
	class RegisteredMemoryVar : RegisteredMemory
	{
		T v;
		size_t clear() override
		{
			v = 0;
			return 0;
		}
	public:
		RegisteredMemoryVar() : v(0)
		{
			register_consistent_memory(this);
		}
		explicit RegisteredMemoryVar(T v) : v(v)
		{
			register_consistent_memory(this);
		}
		~RegisteredMemoryVar()
		{
			unregister_consistent_memory(this);
		}

		RegisteredMemoryVar& operator+= (T add)
		{
			v += add;
			return *this;
		}

		void operator = (T other)
		{
			v = other;
		}
		operator T() const noexcept
		{
			return v;
		}
	};
}
