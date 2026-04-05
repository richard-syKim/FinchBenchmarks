#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <omp.h>
#include <chrono>
#include <limits>

typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMat;

extern "C" {
    double eigen_add(const double* a, const double* b, int rows, int cols,
        int** c_outer_out, int** c_inner_out, double** c_vals_out, int* c_nnz_out, int n_threads) {

        // Eigen dense matrices to sparse
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
            A_dense(a, rows, cols);
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
            B_dense(b, rows, cols);

        SpMat A = A_dense.sparseView();
        SpMat B = B_dense.sparseView();

        omp_set_num_threads(n_threads);

        SpMat C;

        // warmup
        for (int w = 0; w < n_warmup; ++w) {
            C = A + B;
            // C.setZero();
        }

        // actual
        const int n_warmup   = 5;
        const int n_trials   = 100;
        const int batch_size = 1000;

        double min_time = std::numeric_limits<double>::max();
        for (int t = 0; t < n_trials; ++t) {
            // C.setZero();
            
            auto start = std::chrono::steady_clock::now();
            for (int b = 0; b < batch_size; ++b) {
                C = A + B;
            }
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            min_time = std::min(min_time, elapsed.count() / batch_size);
        }

        // CSR
        C.makeCompressed();
        int nnz = C.nonZeros();

        int*    outer = new int[rows + 1];
        int*    inner = new int[nnz];
        double* vals  = new double[nnz];

        std::memcpy(outer, C.outerIndexPtr(), sizeof(int) * (rows + 1));
        std::memcpy(inner, C.innerIndexPtr(), sizeof(int) * nnz);
        std::memcpy(vals,  C.valuePtr(),      sizeof(double) * nnz);

        *c_outer_out = outer;
        *c_inner_out = inner;
        *c_vals_out  = vals;
        *c_nnz_out   = nnz;

        return min_time;
    }
}