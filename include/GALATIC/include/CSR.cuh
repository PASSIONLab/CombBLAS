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

#include <memory>
#include <algorithm>
#include <math.h>
#include <cstring>

#include "COO.cuh"

#include <stdint.h>
#include <string>
#include <fstream>
#include <stdexcept>
#include <iterator>
#include <vector>
#include <algorithm>
#include <memory>
#include <iostream>

#pragma once

template<typename T>
struct COO;

template<typename T>
struct DenseVector;

template<typename T>
struct CSR
{
	struct Statistics
	{
		double mean;
		double std_dev;
		size_t max;
		size_t min;
	};

	void computeStatistics(double& mean, double& std_dev, size_t& max, size_t& min)
	{
		// running variance by Welford
		size_t count = 0;
		mean = 0;
		double M2 = 0;
		max = 0;
		min = cols;
		for (size_t i = 0; i < rows; ++i)
		{
			size_t r_length = row_offsets[i + 1] - row_offsets[i];
			min = std::min(min, r_length);
			max = std::max(max, r_length);
			++count;
			double newValue = static_cast<double>(r_length);
			double delta = newValue - mean;
			mean = mean + delta / count;
			double delta2 = newValue - mean;
			M2 = M2 + delta * delta2;
		}
		if (count < 2)
			std_dev = 0;
		else
			std_dev = sqrt(M2 / (count - 1));
	}

	Statistics rowStatistics()
	{
		Statistics stats;
		computeStatistics(stats.mean, stats.std_dev, stats.max, stats.min);
		return stats;
	}

	size_t rows, cols, nnz;

	std::unique_ptr<T[]> data;
	std::unique_ptr<unsigned int[]> row_offsets;
	std::unique_ptr<unsigned int[]> col_ids;

	CSR() : rows(0), cols(0), nnz(0), data(std::unique_ptr<T[]>(new T[0])) {
	}
	void alloc(size_t rows, size_t cols, size_t nnz);

	// CSR<T>& operator=(CSR<T> other)
	// {
	// 	this->rows = other.rows;
	// 	this->cols = other.cols;
	// 	this->nnz = other.nnz;
	// 	this->data = std::move(other.data);
	// 	this->row_offsets = std::move(other.row_offsets);
	// 	this->col_ids = std::move(other.col_ids);
	// 	return *this;
	// }

	// CSR(const CSR<T>& other)
	// {
	// 	this->rows = other.rows;
	// 	this->cols = other.cols;
	// 	this->nnz = other.nnz;
	// 	this->data = std::make_unique<T[]>(other.nnz);
	// 	memcpy(this->data.get(), other.data.get(), sizeof(T) * other.nnz);
	// 	this->col_ids = std::make_unique<unsigned int[]>(other.nnz);
	// 	memcpy(this->col_ids.get(), other.col_ids.get(), sizeof(unsigned int) * other.nnz);
	// 	this->row_offsets = std::make_unique<unsigned int[]>(other.rows + 1);
	// 	memcpy(this->row_offsets.get(), other.row_offsets.get(), sizeof(unsigned int) * (other.rows + 1));
	// }

};



namespace {
    template<typename VALUE_TYPE>
    struct State
    {
        typedef VALUE_TYPE ValueType;

        bool transpose;

        State() :  transpose(false) { }
        State(bool transpose) :  transpose(transpose) { }
    };

    struct CSRIOHeader
    {
        static constexpr char Magic[] = { 'H','i', 1, 'C','o','m','p','s','d' };

        char magic[sizeof(Magic)];
        uint64_t typesize;
        uint64_t compresseddir;
        uint64_t indexsize;
        uint64_t fixedoffset;
        uint64_t offsetsize;
        uint64_t num_rows, num_columns;
        uint64_t num_non_zeroes;

        CSRIOHeader() = default;


        template<typename T>
        static uint64_t typeSize()
        {
            return sizeof(T);
        }

        template<typename T>
        CSRIOHeader(const CSR<T>& mat)
        {
            for (size_t i = 0; i < sizeof(Magic); ++i)
                magic[i] = Magic[i];
            typesize = typeSize<T>();
            compresseddir = 0;
            indexsize = typeSize<uint32_t>();
            fixedoffset = 0;
            offsetsize = typeSize<uint32_t>();

            num_rows = mat.rows;
            num_columns = mat.cols;
            num_non_zeroes = mat.nnz;
        }

        bool checkMagic() const
        {
            for (size_t i = 0; i < sizeof(Magic); ++i)
                if (magic[i] != Magic[i])
                    return false;
            return true;
        }
    };
    constexpr char CSRIOHeader::Magic[];
}

template<typename T>
void CSR<T>::alloc(size_t r, size_t c, size_t n)
{
    rows = r;
    cols = c;
    nnz = n;

    data = std::make_unique<T[]>(n);
    col_ids = std::make_unique<unsigned int[]>(n);
    row_offsets = std::make_unique<unsigned int[]>(r+1);
}

template<typename T>
CSR<T> loadCSR(const char * file)
{
    std::ifstream fstream(file, std::fstream::binary);
    if (!fstream.is_open())
        throw std::runtime_error(std::string("could not open \"") + file + "\"");

    CSRIOHeader header;
    State<T> state;
    fstream.read(reinterpret_cast<char*>(&header), sizeof(CSRIOHeader));
    if (!fstream.good())
        throw std::runtime_error("Could not read CSR header");
    if (!header.checkMagic())
        throw std::runtime_error("File does not appear to be a CSR Matrix");

    fstream.read(reinterpret_cast<char*>(&state), sizeof(state));
    if (!fstream.good())
        throw std::runtime_error("Could not read CompressedMatrix state");
    if (header.typesize != CSRIOHeader::typeSize<T>())
        throw std::runtime_error("File does not contain a CSR matrix with matching type");

    CSR<T> res;
    res.alloc(header.num_rows, header.num_columns, header.num_non_zeroes);

    fstream.read(reinterpret_cast<char*>(&res.data[0]), res.nnz * sizeof(T));
    fstream.read(reinterpret_cast<char*>(&res.col_ids[0]), res.nnz * sizeof(unsigned int));
    fstream.read(reinterpret_cast<char*>(&res.row_offsets[0]), (res.rows+1) * sizeof(unsigned int));

    if (!fstream.good())
        throw std::runtime_error("Could not read CSR matrix data");

    return res;
}

template<typename T>
void storeCSR(const CSR<T>& mat, const char * file)
{
    std::ofstream fstream(file, std::fstream::binary);
    if (!fstream.is_open())
        throw std::runtime_error(std::string("could not open \"") + file + "\"");

    CSRIOHeader header(mat);
    State<T> state;
    fstream.write(reinterpret_cast<char*>(&header), sizeof(CSRIOHeader));
    fstream.write(reinterpret_cast<const char*>(&state), sizeof(state));
    fstream.write(reinterpret_cast<char*>(&mat.data[0]), mat.nnz * sizeof(T));
    fstream.write(reinterpret_cast<char*>(&mat.col_ids[0]), mat.nnz * sizeof(unsigned int));
    fstream.write(reinterpret_cast<char*>(&mat.row_offsets[0]), (mat.rows + 1) * sizeof(unsigned int));

}

template<typename T>
void spmv(DenseVector<T>& res, const CSR<T>& m, const DenseVector<T>& v, bool transpose)
{
    if (transpose && v.size != m.rows)
        throw std::runtime_error("SPMV dimensions mismatch");
    if (!transpose && v.size != m.cols)
        throw std::runtime_error("SPMV dimensions mismatch");

    size_t outsize = transpose ? m.cols : m.rows;
    if (res.size < outsize)
        res.data = std::make_unique<T[]>(outsize);
    res.size = outsize;

    if (transpose)
    {
        std::fill(&res.data[0], &res.data[0] + m.cols, 0);
        for (size_t i = 0; i < m.rows; ++i)
        {
            for (unsigned int o = m.row_offsets[i]; o < m.row_offsets[i+1]; ++o)
                res.data[m.col_ids[o]] += m.data[o] * v.data[i];
        }
    }
    else
    {
        for (size_t i = 0; i < m.rows; ++i)
        {
            T val = 0;
            for (unsigned int o = m.row_offsets[i]; o < m.row_offsets[i+1]; ++o)
                val += m.data[o] * v.data[m.col_ids[o]];
            res.data[i] = val;
        }
    }
}

template<typename T>
void convert(CSR<T>& res, const COO<T>& coo)
{
    struct Entry
    {
        unsigned int r, c;
        T v;
        bool operator < (const Entry& other)
        {
            if (r != other.r)
                return r < other.r;
            return c < other.c;
        }
    };

    std::vector<Entry> entries;
    std::cout << coo.nnz << std::endl;
    entries.reserve(coo.nnz);
    for (size_t i = 0; i < coo.nnz; ++i)
        entries.push_back(Entry{ coo.row_ids[i], coo.col_ids[i], coo.data[i] });
    std::sort(std::begin(entries), std::end(entries));

    res.alloc(coo.rows, coo.cols, coo.nnz);
    std::fill(&res.row_offsets[0], &res.row_offsets[coo.rows], 0);
    for (size_t i = 0; i < coo.nnz; ++i)
    {
        res.data[i] = entries[i].v;
        res.col_ids[i] = entries[i].c;
        ++res.row_offsets[entries[i].r];
    }

    unsigned int off = 0;
    for (size_t i = 0; i < coo.rows; ++i)
    {
        unsigned int n = off + res.row_offsets[i];
        res.row_offsets[i] = off;
        off = n;
    }
    res.row_offsets[coo.rows] = off;
}
