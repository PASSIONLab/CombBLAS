
#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>


namespace GALATIC {
template<typename T>
void convert(CSR<T>& dst, const dCSR<T>& src, unsigned int padding=0)
{
    dst.alloc(src.rows + padding, src.cols, src.nnz + 8 * padding);
    dst.rows = src.rows; dst.nnz = src.nnz; dst.cols = src.cols;
    cudaMemcpy(dst.data.get(), src.data, dst.nnz * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(dst.col_ids.get(), src.col_ids, dst.nnz * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(dst.row_offsets.get(), src.row_offsets, (dst.rows + 1) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
}

};

template<typename IDX_t, typename VALUE_t> 
cusp::csr_matrix< IDX_t,  VALUE_t, cusp::device_memory> to_cusp_csr( CSR<VALUE_t>& orig_mat)
{
    cusp::csr_matrix<IDX_t, VALUE_t, cusp::host_memory> result_cpu(orig_mat.rows, orig_mat.cols, orig_mat.nnz);

    for (int i = 0; i < orig_mat.rows; i++) {
        result_cpu.row_offsets[i] = orig_mat.row_offsets[i];
    }

    for (int i = 0; i < orig_mat.nnz; i++) {
        result_cpu.column_indices[i] = orig_mat.col_ids[i];
        result_cpu.values[i] = orig_mat.data[i];
    }

    cusp::csr_matrix<IDX_t, VALUE_t, cusp::device_memory> result(result_cpu);
    return result;
}




template<typename IDX_t, typename SEMIRING_t> 
void CuspMultiplyWrapper(cusp::csr_matrix< IDX_t, typename SEMIRING_t::input_t, cusp::device_memory>& A,
                         cusp::csr_matrix< IDX_t, typename SEMIRING_t::input_t, cusp::device_memory>& B, 
                         cusp::csr_matrix< IDX_t, typename SEMIRING_t::output_t, cusp::device_memory>& C,
                         SEMIRING_t sr) {
    cusp::multiply(A,B,C,  __device__ [] (auto a) { return a;}, 
             __device__  [sr] (thrust::device_reference<typename SEMIRING_t::input_t>& a const, thrust::device_reference<typename SEMIRING_t::input_t>& b const)  {return sr.multiply(a,b); },
              __device__  [sr](typename SEMIRING_t::output_t& a const, typename SEMIRING_t::output_t & b const ) {return sr.add(a,b); } );

    
}