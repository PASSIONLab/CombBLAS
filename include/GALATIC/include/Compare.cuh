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
* Compare.h
*
* ac-SpGEMM
*
* Authors: Daniel Mlakar, Markus Steinberger, Martin Winter
*------------------------------------------------------------------------------
*/

#pragma once

#include "dCSR.cuh"
#include <stdio.h>
#include "common.h"


namespace ACSpGEMM {

    template<typename DataType>
    __global__ void d_compare(int in_rows, int in_cols, const uint32_t *__restrict reference_offset,
                              const uint32_t *__restrict reference_indices, const DataType *__restrict reference_values,
                              const uint32_t *__restrict compare_offset, const uint32_t *__restrict compare_indices,
                              const DataType *__restrict compare_values, bool compare_data, double epsilon,
                              uint32_t *verification) {
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        if (tid >= in_rows)
            return;

        uint32_t ref_offset = reference_offset[tid];
        uint32_t comp_offset = compare_offset[tid];
        uint32_t ref_number_entries = reference_offset[tid + 1] - ref_offset;
        uint32_t comp_number_entries = compare_offset[tid + 1] - comp_offset;

        if (ref_number_entries != comp_number_entries) {
#ifdef VERIFICATION_TEXT
            printf("---------- Row: %u | Row length not identical: (Ref|Comp) : (%u|%u)\n",tid, ref_number_entries, comp_number_entries);
#endif
            *verification = 1;
        }

        uint32_t num_entries = min(ref_number_entries, comp_number_entries);

        for (uint32_t i = 0; i < num_entries; ++i) {
            if (reference_indices[ref_offset + i] != compare_indices[comp_offset + i]) {
#ifdef VERIFICATION_TEXT
                printf("Row: %u | Row indices do NOT match: (Ref|Comp) : (%u|%u) - pos: %u/%u\n", tid, reference_indices[ref_offset + i], compare_indices[comp_offset + i], i, num_entries);
#endif
                *verification = 1;
            }
            if (compare_data) {
                if (!(reference_values[ref_offset + i] == compare_values[comp_offset + i])) {
#ifdef VERIFICATION_TEXT
                    printf("Row: %u | Values do NOT match: (Ref|Comp) : (%f|%f) - pos: %u/%u\n", tid, reference_values[ref_offset + i], compare_values[comp_offset + i], i, num_entries);
#endif
                    *verification = 1;
                }
            }
        }
    }

    template<typename DataType>
    bool Compare(const dCSR<DataType> &reference_mat, const dCSR<DataType> &compare_mat, bool compare_data) {
        int blockSize(256);
        int gridSize(divup<int>(reference_mat.rows + 1, blockSize));
        double epsilon = 0.1;
        uint32_t *verification, h_verification;
        cudaMalloc(&verification, sizeof(uint32_t));
        cudaMemset(verification, 0, sizeof(uint32_t));

        d_compare<DataType> <<< gridSize, blockSize >>> (reference_mat.rows, reference_mat.cols,
                reference_mat.row_offsets, reference_mat.col_ids, reference_mat.data,
                compare_mat.row_offsets, compare_mat.col_ids, compare_mat.data,
                compare_data, epsilon, verification);

        cudaMemcpy(&h_verification, verification, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        return (h_verification == 0);
    }
}