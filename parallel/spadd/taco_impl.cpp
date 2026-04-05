#include "taco.h"
#include <omp.h>
#include <chrono>
#include <cstring>

using namespace taco;

extern "C" {
    double taco_add(const double** a, const double** b, int rows, int cols, 
            int** c_pos_out, int** c_crd_out, double** c_vals_out, int n_threads) {

        Format ds({Dense, Sparse});

        Tensor<double> A("A", {rows, cols}, ds);
        Tensor<double> B("B", {rows, cols}, ds);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                if (a[i][j] != 0.0) A.insert({i, j}, a[i][j]);
                if (b[i][j] != 0.0) B.insert({i, j}, b[i][j]);
            }
        }
        A.pack();
        B.pack();

        Tensor<double> C("C", {rows, cols}, ds);

        IndexVar i, j;
        C(i, j) = A(i, j) + B(i, j);

        IndexStmt stmt = C.getAssignment().concretize();
        stmt = stmt.parallelize(
            i,
            ParallelUnit::CPUThread,
            OutputRaceStrategy::NoRaces
        );

        omp_set_num_threads(n_threads);

        C.compile(stmt);

        // extract result for correctness
        C.assemble();
        C.compute();

        taco_tensor_t* ct = C.getStorage().getTacoTensorT();
        int*    c_pos  = reinterpret_cast<int*>   (ct->indices[1][0]);
        int*    c_crd  = reinterpret_cast<int*>   (ct->indices[1][1]);
        double* c_vals = reinterpret_cast<double*>(ct->vals);

        if (c_pos_out)  *c_pos_out  = c_pos;
        if (c_crd_out)  *c_crd_out  = c_crd;
        if (c_vals_out) *c_vals_out = c_vals;

        // Benchmarking
        const int n_warmup   = 5;
        const int n_trials   = 100;
        const int batch_size = 1000;

        for (int w = 0; w < n_warmup; ++w) {
            C.assemble();
            C.compute();
        }

        double min_time = std::numeric_limits<double>::max();
        for (int t = 0; t < n_trials; ++t) {
            auto start = std::chrono::steady_clock::now();
            for (int b = 0; b < batch_size; ++b) {
                C.assemble();
                C.compute();
            }
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            min_time = std::min(min_time, elapsed.count() / batch_size);
        }

        return min_time;
    }
}
