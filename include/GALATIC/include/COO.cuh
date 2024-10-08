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

#include "Vector.h"
#include <iostream>
#include <memory>

#include "Vector.h"

#include <string>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <iterator>
#include <vector>
template<typename T>
struct COO
{
	size_t rows, cols, nnz;

	std::unique_ptr<T[]> data;
	std::unique_ptr<unsigned int[]> row_ids;
	std::unique_ptr<unsigned int[]> col_ids;

	COO() : rows(0), cols(0), nnz(0) { }
	void alloc(size_t rows, size_t cols, size_t nnz);
};


namespace {
    struct DataTypeValidator {
        static bool validate(std::string type) {
            return false;
        }
    };
/*
    template<>
    struct DataTypeValidator<float> {
        static const bool validate(std::string type) {
            return type.compare("real") == 0 || type.compare("integer") == 0;
        }
    };
    template<typename VALUE_TYPE>
    struct DataTypeValidator  {
        static const bool validate(std::string type) {
           std::cout << "type: " << type << std::endl;
            return type.compare("real") == 0 || type.compare("integer") == 0;;
        }
    };

    template<>
    struct DataTypeValidator<int> {
        static const bool validate(std::string type) {
            return type.compare("integer") == 0;
        }
    };

    template<>
    struct DataTypeValidator<uint64_t> {
        static const bool validate(std::string type) {
            return type.compare("integer") == 0;
        }
    };*/
}

template<typename T>
void COO<T>::alloc(size_t r, size_t c, size_t n)
{
    rows = r;
    cols = c;
    nnz = n;

    data = std::make_unique<T[]>(n);
    row_ids = std::make_unique<unsigned int[]>(n);
    col_ids = std::make_unique<unsigned int[]>(n);
}

template<typename T>
COO<T> loadMTX(const char * file)
{
    std::ifstream fstream(file);
    if (!fstream.is_open())
        throw std::runtime_error(std::string("could not open \"") + file + "\"");

    COO<T> resmatrix;
    size_t num_rows = 0, num_columns = 0, num_non_zeroes = 0;

    size_t line_counter = 0;
    std::string line;
    bool pattern = false;
    bool hermitian = false;
    // read header;
    std::getline(fstream, line);
    if (line.compare(0, 32, "%%MatrixMarket matrix coordinate") != 0)
        throw std::runtime_error("Can only read MatrixMarket format that is in coordinate form");
    std::istringstream iss(line);
    std::vector<std::string> tokens{ std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{} };
    bool complex = false;

    if (tokens[3] == "pattern")
        pattern = true;
    else if (tokens[3] == "complex")
        complex = true;
    else if (tokens[3] != "real")
        throw std::runtime_error("MatrixMarket data type does not match matrix format");
    bool symmetric = false;
//    if (tokens[4].compare("general") == 0)
        symmetric = false;
  //  else if (tokens[4].compare("symmetric") == 0)
//        symmetric = true;
   // else if (tokens[4].compare("Hermitian") == 0)
   //     hermitian = true;
  //  else
    //    throw std::runtime_error("Can only read MatrixMarket format that is either symmetric, general or hermitian");

    while (std::getline(fstream, line))
    {
        ++line_counter;
        if (line[0] == '%')
            continue;
        std::istringstream liness(line);
        liness >> num_rows >> num_columns >> num_non_zeroes;
        if (liness.fail())
            throw std::runtime_error(std::string("Failed to read matrix market header from \"") + file + "\"");
        //std::cout << "Read matrix header" << std::endl;
        //std::cout << "rows: " << rows << " columns: " << columns << " nnz: " << nnz << std::endl;
        break;
    }

    size_t reserve = num_non_zeroes;
    if (symmetric || hermitian)
        reserve *= 2;

    resmatrix.alloc(num_rows, num_columns, reserve);

    //read data
    size_t read = 0;
    while (std::getline(fstream, line))
    {
        ++line_counter;
        if (line[0] == '%')
            continue;

        std::istringstream liness(line);


        do
        {
            char ch;
            liness.get(ch);
            if (!isspace(ch))
            {
                liness.putback(ch);
                break;
            }

        } while (!liness.eof());
        if (liness.eof() || line.length() == 0)
            continue;

        uint32_t r, c;
        T d;
        liness >> r >> c;
        if (pattern)
            d = 0;// T::Init(1);
        else {
            double a;
            liness >> a;
            d =0;// T::Init(a);
        }
        if (liness.fail())
            throw std::runtime_error(std::string("Failed to read data at line ") + std::to_string(line_counter) + " from matrix market file \"" + file + "\"");
        if (r > num_rows)
            throw std::runtime_error(std::string("Row index out of bounds at line  ") + std::to_string(line_counter) + " in matrix market file \"" + file + "\"");
        if (c > num_columns)
            throw std::runtime_error(std::string("Column index out of bounds at line  ") + std::to_string(line_counter) + " in matrix market file \"" + file + "\"");

        resmatrix.row_ids[read] = r - 1;
        resmatrix.col_ids[read] = c - 1;
        resmatrix.data[read] = d;
        ++read;
        if ((symmetric || hermitian) && r != c)
        {
            resmatrix.row_ids[read] = c - 1;
            resmatrix.col_ids[read] = r - 1;
            resmatrix.data[read] = d;
            ++read;
        }
    }

    resmatrix.nnz = read;
    return resmatrix;
}



template<typename T>
COO<T> loadCOO(const char * file)
{
    return COO<T>();
}

template<typename T>
void storeCOO(const COO<T>& mat, const char * file)
{

}

template<typename T>
void spmv(DenseVector<T>& res, const COO<T>& m, const DenseVector<T>& v, bool transpose)
{
    if (transpose && v.size != m.rows)
        throw std::runtime_error("SPMV dimensions mismatch");
    if (!transpose && v.size != m.cols)
        throw std::runtime_error("SPMV dimensions mismatch");

    size_t outsize = transpose ? m.cols : m.rows;
    if (res.size < outsize)
        res.data = std::make_unique<T[]>(outsize);
    res.size = outsize;

    std::fill(&res.data[0], &res.data[0] + outsize, 0);


    if(transpose)
        for (size_t i = 0; i < m.nnz; ++i)
            res.data[m.col_ids[i]] += m.data[i] * v.data[m.row_ids[i]];
    else
        for (size_t i = 0; i < m.nnz; ++i)
            res.data[m.row_ids[i]] += m.data[i] * v.data[m.col_ids[i]];
}

