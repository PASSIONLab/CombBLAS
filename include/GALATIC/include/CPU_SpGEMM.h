#include <algorithm>

#include <vector>
#include "CSR.cuh"
#include "dCSR.cuh"

#pragma once

 template<typename T>
 using Vec = std::vector<T>;

     template <typename T> 
     struct CSR_Tuple {
         uint64_t col;
         T value;
         CSR_Tuple(uint64_t col, T value) : col(col), value(value) {}
     };


 template<typename SEMIRING_t,
          typename LEFT_T  = typename SEMIRING_t::leftInput_t,        // input type alias  for mul
          typename RIGHT_T  = typename SEMIRING_t::rightInput_t,      // input type alias  for mul
          typename OUT_t = typename SEMIRING_t::output_t              // output type alias for mul
 >
 void Mult_CPU( CSR<LEFT_T> &A,  CSR<RIGHT_T> &B, CSR<OUT_t>& C, SEMIRING_t& sr)
 {
     	

     Vec<CSR_Tuple<OUT_t>> result = Vec<CSR_Tuple<OUT_t>>();
     Vec<uint64_t> row_starts = Vec<uint64_t>();

    int last_percent = 0;

    Vec<CSR_Tuple<OUT_t>> temp_buffer = Vec<CSR_Tuple<OUT_t>>();


     for (uint64_t A_row_idx = 0; A_row_idx < A.rows; A_row_idx++)
     {
         if (A_row_idx*10 / A.rows > last_percent) {
             std::cout  << "CPU Done%: " << A_row_idx*100 / A.rows  <<std::endl;
             last_percent = A_row_idx*10 / A.rows;
         }
         const uint64_t A_row_start =  A.row_offsets[A_row_idx];
         const uint64_t A_row_end   =  A_row_idx + 1 >= A.rows ? A.nnz :  A.row_offsets[A_row_idx+1];
        
        temp_buffer.clear();
         // for every element A_r,k in row A_row_idx
         for (uint64_t A_element_idx = A_row_start; A_element_idx < A_row_end; A_element_idx++)
         {
             const LEFT_T &A_element = A.data[A_element_idx];

             // for every element B_k,c


             uint64_t A_col_idx = A.col_ids[A_element_idx];

             uint64_t B_row_start = B.row_offsets[A_col_idx];
             uint64_t B_row_end = A_col_idx + 1 >= B.rows ? B.nnz :  B.row_offsets[A_col_idx+1]; 


             for (uint64_t c_star = B_row_start; c_star < B_row_end; c_star++){
                 const RIGHT_T & B_element = B.data[c_star];
                 uint64_t b_col = B.col_ids[c_star];                
                 auto jq  =sr.multiply(A_element, B_element);
                 temp_buffer.push_back(CSR_Tuple<OUT_t>(b_col, jq ));
             }


         }

         std::sort(
             temp_buffer.begin(),
             temp_buffer.end(),
            [] (const CSR_Tuple<OUT_t> &a, const CSR_Tuple<OUT_t> &b)  {  return a.col < b.col; }
         );


         int64_t last_col = -1;
         row_starts.push_back(result.size());
         for (auto & ele : temp_buffer) {
             if (ele.col != last_col) {
                 result.push_back(ele);
             } else {
                 result[result.size() -1] = CSR_Tuple<OUT_t>(ele.col, sr.add(result[result.size() -1].value, ele.value));
             }
             last_col = ele.col;
         }
     }

     C.alloc(A.rows,B.cols, result.size());

     for (int i = 0; i < result.size(); i++) {
         C.data[i] = result.at(i).value;
         C.col_ids[i] = result.at(i).col;
     }

     row_starts.push_back(result.size());

     C.row_offsets[0] =0;
     for (int i = 0; i < A.rows+1; i++) {
         C.row_offsets[i] = row_starts.at(i);
     }

 } 