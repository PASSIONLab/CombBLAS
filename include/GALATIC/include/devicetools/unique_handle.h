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


#ifndef INCLUDED_CUDA_UNIQUE_HANDLE
#define INCLUDED_CUDA_UNIQUE_HANDLE

#pragma once

#include <utility>


namespace CU
{
	template <typename T, T NULL_VALUE, typename Deleter>
	class unique_handle : Deleter
	{
		T h;
		
		void free(T handle) noexcept
		{
			if (handle != NULL_VALUE)
				Deleter::operator ()(handle);
		}
		
	public:
		unique_handle(const unique_handle&) = delete;
		unique_handle& operator =(const unique_handle&) = delete;
		
		using handle_type = T;
		using deleter_type = Deleter;
		
		static constexpr T null_value = NULL_VALUE;
		
		explicit unique_handle(T handle = NULL_VALUE) noexcept
			: h(handle)
		{
		}

		void consume(T handle) noexcept { h = handle; }

		
		unique_handle(T handle, const Deleter& d) noexcept
			: Deleter(d),
			  h(handle)
		{
		}
		
		unique_handle(T handle, Deleter&& d) noexcept
			: Deleter(std::move(d)),
			  h(handle)
		{
		}
		
		unique_handle(unique_handle&& h) noexcept
			: Deleter(std::move(static_cast<Deleter&&>(h))),
			  h(h.h)
		{
			h.h = NULL_VALUE;
		}
		
		~unique_handle()
		{
			free(h);
		}
		
		operator T() const noexcept { return h; }

		template <typename DataType = void>
		DataType* get() const noexcept { return reinterpret_cast<DataType*>(h); }

		template <typename DataType = void>
		DataType* getRelease() noexcept { DataType* tmp = reinterpret_cast<DataType*>(h); h = 0ULL; return tmp; }
		
		unique_handle& operator =(unique_handle&& h) noexcept
		{
			using std::swap;
			swap(*this, h);
			return *this;
		}
		
		T release() noexcept
		{
			T temp = h;
			h = NULL_VALUE;
			return temp;
		}
		
		void reset(T handle = null_value) noexcept
		{
			using std::swap;
			swap(this->h, handle);
			free(handle);
		}
		
		friend void swap(unique_handle& a, unique_handle& b) noexcept
		{
			using std::swap;
			swap(a.h, b.h);
		}
	};
}

#endif  // INCLUDED_CUDA_UNIQUE_HANDLE
