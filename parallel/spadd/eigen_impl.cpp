#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <omp.h>
#include <chrono>
#include <sys/stat.h>
#include <iostream>
#include <cstdint>
#include "../../deps/SparseRooflineBenchmark/src/benchmark.hpp"

typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMat;

int main(int argc, char **argv){
    auto params = parse(argc, argv);

    int n_threads = params.max_threads;
    omp_set_num_threads(n_threads);
    Eigen::setNbThreads(n_threads);

    SpMat A, B;
    Eigen::loadMarket(A, params.input + "/A.ttx");
    Eigen::loadMarket(B, params.input + "/B.ttx");
    SpMat C;

    auto time = benchmark(
      [&C, &A, &B]() {
        C = SpMat(A.rows(), A.cols());
      },
      [&C, &A, &B]() {
        C = A + B;
      }
    );
    
    Eigen::saveMarket(C, params.input + "/C.ttx");
    json measurements;
    measurements["time"] = time.first;
    measurements["memory"] = 0;
    std::ofstream measurements_file(params.output + "/measurements.json");
    measurements_file << measurements;
    measurements_file.close();
    return 0;
}