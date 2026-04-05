#include <iostream>
#include <vector>
#include <chrono>
#include <mkl.h>

extern "C" {
    double mkl_sum(const double* dense_v, int dim, double* out_sum, int n_threads) {
        mkl_set_num_threads(n_threads);

        const int N = static_cast<int>(dim);
        std::vector<double> ones(N, 1.0);

        // Warmup
        const int n_warmup = 5;
        const int n_trials = 100;
        for (int w = 0; w < n_warmup; ++w) {
            cblas_ddot(N, dense_v, 1, ones.data(), 1);
        }

        // Timed trials
        double min_time = std::numeric_limits<double>::max();
        double sum = 0.0;
        for (int t = 0; t < n_trials; ++t) {
            auto start = std::chrono::steady_clock::now();
            sum = cblas_ddot(N, dense_v, 1, ones.data(), 1);
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            min_time = std::min(min_time, elapsed.count());
        }

        *out_sum = sum;
        return min_time;
    }
}
