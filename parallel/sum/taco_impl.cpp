#include "taco.h"
#include <omp.h>
#include <chrono>
#include <cstring>

using namespace taco;

extern "C" {
    double taco_sum(const double* dense_v, int dim, double* out_sum, int n_threads) {
        Tensor<double> v("v", {dim}, Format({Sparse}));
        for (int k = 0; k < dim; ++k) {
            if (dense_v[k] != 0.0) {
                v.insert({k}, dense_v[k]);
            }
        }
        v.pack();

        Tensor<double> sum("sum", {}, Format());
        
        IndexVar i;
        
        sum() += v(i);

        IndexStmt stmt = sum.getAssignment().concretize();
        stmt = stmt.parallelize(
            i,
            ParallelUnit::CPUThread,
            OutputRaceStrategy::Atomics // or Temporary if performance is better
        );

        sum.compile(stmt);

        sum.assemble();
        sum.compute();
        *out_sum = sum.begin()->second;

        const int n_warmup = 5;
        const int n_trials = 100;
        const int batch_size = 1000; // total time > ~1ms

        // warmup
        for (int w = 0; w < n_warmup; ++w) {
            sum.assemble();
            sum.compute();
        }

        // actual
        double min_time = std::numeric_limits<double>::max();
        for (int t = 0; t < n_trials; ++t) {
            auto start = std::chrono::steady_clock::now();
            for (int b = 0; b < batch_size; ++b) {
                sum.assemble();
                sum.compute();
            }
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            min_time = std::min(min_time, elapsed.count() / batch_size);
        }

        return min_time;
    }
}