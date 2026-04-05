#include <Eigen/Dense>
#include <omp.h>
#include <chrono>

extern "C" {
    double eigen_sum(const double* dense_v, int dim, double* out_sum, int n_threads) {
        Eigen::Map<const Eigen::VectorXd> v(dense_v, dim);

        omp_set_num_threads(n_threads);

        const int n_warmup = 5;
        const int n_trials = 100;
        const int batch_size = 1000; // total time > ~1ms
        double sum = 0.0;

        // warmup
        for (int w = 0; w < n_warmup; ++w) {
            sum = 0.0;
            #pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < v.size(); ++i) sum += v(i);
        }

        // actual
        double min_time = std::numeric_limits<double>::max();
        for (int t = 0; t < n_trials; ++t) {
            auto start = std::chrono::steady_clock::now();
            for (int b = 0; b < batch_size; ++b) {
                sum = 0.0;
                #pragma omp parallel for reduction(+:sum)
                for (int i = 0; i < v.size(); ++i) {
                    sum += v(i);
                }
            }
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            min_time = std::min(min_time, elapsed.count() / batch_size);
        }

        *out_sum = sum;

        return min_time;
    }
}
