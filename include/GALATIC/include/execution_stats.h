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

#include <stdint.h>
#include <iostream>

struct ExecutionStats
{
	//timings
	bool measure_all;
	float duration;
	float duration_blockstarts;
	float duration_spgemm;
	float duration_merge_case_computation;
	float duration_merge_simple;
	float duration_merge_max;
	float duration_merge_generalized;
	float duration_write_csr;


	//merge cases
	uint32_t shared_rows;
	uint32_t simple_mergers;
	uint32_t simple_rows;
	uint32_t complex_rows;
	uint32_t generalized_rows;

	//memory consumption
	size_t mem_allocated_chunks;
	size_t mem_used_chunks;
	size_t mem_clear_return;

	//misc
	size_t restarts;
	int called{ 0 };
	friend std::ostream& operator<<(std::ostream&, const ExecutionStats&);

	ExecutionStats() : measure_all(false),
		duration(0), duration_blockstarts(0), duration_spgemm(0), duration_merge_case_computation(0),
		duration_merge_simple(0), duration_merge_max(0), duration_merge_generalized(0), duration_write_csr(0),		
		shared_rows(0), simple_mergers(0), simple_rows(0), complex_rows(0), generalized_rows(0),
		mem_allocated_chunks(0), mem_used_chunks(), mem_clear_return(0),
		restarts(0) { }

	ExecutionStats& operator+=(const ExecutionStats& stats)
	{
		this->duration += stats.duration;
		this->duration_blockstarts += stats.duration_blockstarts;
		this->duration_spgemm += stats.duration_spgemm;
		this->duration_merge_case_computation += stats.duration_merge_case_computation;
		this->duration_merge_simple += stats.duration_merge_simple;
		this->duration_merge_max += stats.duration_merge_max;
		this->duration_merge_generalized += stats.duration_merge_generalized;
		this->duration_write_csr += stats.duration_write_csr;
		this->shared_rows += stats.shared_rows;
		this->simple_mergers += stats.simple_mergers;
		this->simple_rows += stats.simple_rows;
		this->complex_rows += stats.complex_rows;
		this->generalized_rows += stats.generalized_rows;
		this->mem_allocated_chunks += stats.mem_allocated_chunks;
		this->mem_used_chunks += stats.mem_used_chunks;
		this->mem_clear_return += stats.mem_clear_return;
		this->restarts += stats.restarts;
		++called;
		// printf("Overall: %f and added up: %f\n", stats.duration, (stats.duration_blockstarts + stats.duration_spgemm + stats.duration_merge_case_computation +
		// 	stats.duration_merge_simple + stats.duration_merge_max + stats.duration_merge_generalized + stats.duration_write_csr));
		return *this;
	}

	void reset()
	{
		this->duration = 0.0f;
		this->duration_blockstarts = 0.0f;
		this->duration_spgemm = 0.0f;
		this->duration_merge_case_computation = 0.0f;
		this->duration_merge_simple = 0.0f;
		this->duration_merge_max = 0.0f;
		this->duration_merge_generalized = 0.0f;
		this->duration_write_csr = 0.0f;
		this->shared_rows = 0;
		this->simple_mergers = 0;
		this->simple_rows = 0;
		this->complex_rows = 0;
		this->generalized_rows = 0;
		this->mem_allocated_chunks = 0;
		this->mem_used_chunks = 0;
		this->mem_clear_return = 0;
		this->restarts = 0;
	}

	void normalize()
	{
		if (called)
		{
			float division_factor = static_cast<float>(called);
			this->duration /= division_factor;
			this->duration_blockstarts /= division_factor;
			this->duration_spgemm /= division_factor;
			this->duration_merge_case_computation /= division_factor;
			this->duration_merge_simple /= division_factor;
			this->duration_merge_max /= division_factor;
			this->duration_merge_generalized /= division_factor;
			this->duration_write_csr /= division_factor;
			this->shared_rows /= called;
			this->simple_mergers /= called;
			this->simple_rows /= called;
			this->complex_rows /= called;
			this->generalized_rows /= called;
			this->mem_allocated_chunks /= called;
			this->mem_used_chunks /= called;
			this->mem_clear_return /= called;
			this->restarts /= called;
		}
	}
};

inline std::ostream& operator<<(std::ostream& os, const ExecutionStats& obj) {
	os << "Overall Duration: " << obj.duration << " ms\n";
	os << "Restarts: " << obj.restarts << std::endl;
	if (obj.measure_all)
	{
		os << "Sum individual timings: " << obj.duration_blockstarts + obj.duration_spgemm + obj.duration_merge_case_computation + obj.duration_merge_simple + obj.duration_merge_max + obj.duration_merge_generalized + obj.duration_write_csr << " ms\n";
		os << std::string("Duration BlockStarts: ") << obj.duration_blockstarts << " ms | Duration SpGEMM: " << obj.duration_spgemm << " ms\n";
		os << "Duration MergeCase: " << obj.duration_merge_case_computation << " ms | Duration Merge Simple: " << obj.duration_merge_simple << " ms\n";
		os << "Duration Merge Max: " << obj.duration_merge_max << " ms | Duration Merge Generalized: " << obj.duration_merge_generalized << " ms\n";
		os << "Duration Merge Write CSR: " << obj.duration_write_csr << " ms\n";
	}
	return os;
}
