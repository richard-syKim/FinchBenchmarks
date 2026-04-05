#include <iostream>
#include <vector>
#include <chrono>
#include <limits>
#include <cstring>
#include <mkl.h>
#include <mkl_spblas.h>

extern "C" {
    double mkl_add(const double* a, const double* b, int rows, int cols, 
        int** c_outer_out, int** c_inner_out, double** c_vals_out, int* c_nnz_out, int n_threads) {

        mkl_set_num_threads(n_threads);

        int job_to_csr[6] = {0, 0, 0, 2, 0, 1};  // dense to CSR, 0-based rows, 1-based output

        int a_nnz = 0, b_nnz = 0;
        std::vector<double> a_vals(rows * cols), b_vals(rows * cols);
        std::vector<int>    a_col(rows * cols),  b_col(rows * cols);
        std::vector<int>    a_row(rows + 1),     b_row(rows + 1);
        int info = 0;

        // Convert A
        mkl_ddnscsr(job_to_csr,
                    &rows, &cols,
                    const_cast<double*>(a), &cols,   // dense input, leading dimension
                    a_vals.data(), a_col.data(), a_row.data(),
                    &info);
        a_nnz = a_row[rows] - 1;   // 1-based: last entry minus 1-offset

        // Convert B
        mkl_ddnscsr(job_to_csr,
                    &rows, &cols,
                    const_cast<double*>(b), &cols,
                    b_vals.data(), b_col.data(), b_row.data(),
                    &info);
        b_nnz = b_row[rows] - 1;


        sparse_matrix_t A_mkl, B_mkl, C_mkl;
        mkl_sparse_d_create_csr(&A_mkl, SPARSE_INDEX_BASE_ONE,
                                rows, cols,
                                a_row.data(), a_row.data() + 1,
                                a_col.data(), a_vals.data());

        mkl_sparse_d_create_csr(&B_mkl, SPARSE_INDEX_BASE_ONE,
                                rows, cols,
                                b_row.data(), b_row.data() + 1,
                                b_col.data(), b_vals.data());

        // mkl_sparse_d_add: compute C = alpha*A + B for sparse matrices.
        const double alpha = 1.0;

        const int n_warmup   = 5;
        const int n_trials   = 100;
        const int batch_size = 1000;

        for (int w = 0; w < n_warmup; ++w) {
            mkl_sparse_d_add(SPARSE_OPERATION_NON_TRANSPOSE, A_mkl, alpha, B_mkl, &C_mkl);
            mkl_sparse_destroy(C_mkl);
        }

        double min_time = std::numeric_limits<double>::max();
        for (int t = 0; t < n_trials; ++t) {
            auto start = std::chrono::steady_clock::now();
            for (int b = 0; b < batch_size; ++b) {
                mkl_sparse_d_add(SPARSE_OPERATION_NON_TRANSPOSE, A_mkl, alpha, B_mkl, &C_mkl);
                mkl_sparse_destroy(C_mkl);
            }
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            min_time = std::min(min_time, elapsed.count() / batch_size);
        }

        mkl_sparse_d_add(SPARSE_OPERATION_NON_TRANSPOSE, A_mkl, alpha, B_mkl, &C_mkl);

        // mkl_sparse_d_export_csr: retrieves the internal CSR arrays from the handle without copying
        sparse_index_base_t indexing;
        int    c_rows, c_cols;
        int   *c_row_start, *c_row_end, *c_inner;
        double *c_vals;
        mkl_sparse_d_export_csr(C_mkl, &indexing,
                                &c_rows, &c_cols,
                                &c_row_start, &c_row_end,
                                &c_inner, &c_vals);

        int nnz = c_row_start[rows];   // 1-based: last row_start entry = total nnz+1
        int* outer = new int[rows + 1];
        for (int i = 0; i < rows; ++i) outer[i] = c_row_start[i];
        outer[rows] = c_row_end[rows - 1];

        int*    inner_out = new int[nnz];
        double* vals_out  = new double[nnz];
        std::memcpy(inner_out, c_inner, sizeof(int)    * nnz);
        std::memcpy(vals_out,  c_vals,  sizeof(double) * nnz);

        *c_outer_out = outer;
        *c_inner_out = inner_out;
        *c_vals_out  = vals_out;
        *c_nnz_out   = nnz;

        mkl_sparse_destroy(A_mkl);
        mkl_sparse_destroy(B_mkl);
        mkl_sparse_destroy(C_mkl);

        return min_time;
    }
}