#include <intrin.h>
#include <omp.h>

#include <chrono>
#include <iostream>
#include <random>

#include "matrix.h"

namespace lab5 {
class ElapsedTimer {
  using ClockType = std::chrono::high_resolution_clock;

 public:
  ElapsedTimer() : _startTime(ClockType::now()) {}

  [[nodiscard]] std::chrono::nanoseconds elapsed() const {
    return ClockType::now() - _startTime;
  }

 private:
  ClockType::time_point _startTime;
};

class Measurement {
 public:
  explicit Measurement(std::string label) : _label(std::move(label)) {}

  [[nodiscard]] ElapsedTimer timer() const { return _timer; }

  ~Measurement() {
    std::chrono::nanoseconds elapsed = _timer.elapsed();
    std::cout << _label << ':' << " time taken " << elapsed << '\n';
  }

 private:
  std::string _label;
  ElapsedTimer _timer;
};

void MatrixAddSeq(const double *lhs, const double *rhs, double *result,
                  size_t dim) {
  for (long long i = 0; i < dim; ++i) {
    for (long long j = 0; j < dim; ++j) {
      const long long index = i * dim + j;
      result[index] = lhs[index] + rhs[index];
    }
  }
}

void MatrixAddSeqSimd(const double *lhs, const double *rhs, double *result,
                      size_t dim) {
  auto pA = reinterpret_cast<const __m256i *>(lhs);
  auto pB = reinterpret_cast<const __m256i *>(rhs);
  auto pC = reinterpret_cast<__m256i *>(result);

  constexpr auto blockSize = sizeof(__m256i) / sizeof(result[0]);

  for (int i = 0; i < dim / blockSize; ++i) {
    for (int j = 0; j < dim / blockSize; ++j) {
      pC[i * j] = _mm256_or_si256(pA[i * j], pB[i * j]);
    }
  }
}

void MatrixAddPar(const double *lhs, const double *rhs, double *result,
                  size_t dim) {
#pragma omp parallel for
  for (long long i = 0; i < dim; ++i) {
    for (long long j = 0; j < dim; ++j) {
      const long long index = i * dim + j;
      result[index] = lhs[index] + rhs[index];
    }
  }
}

void MatrixMultiplySeq(const double *lhs, const double *rhs, double *result,
                       size_t dim) {
  for (long long i = 0; i < dim; ++i) {
    for (long long j = 0; j < dim; ++j) {
      double total{};
      for (long long k = 0; k < dim; ++k) {
        total += lhs[i * dim + k] * rhs[k * dim + j];
      }
      result[i * dim + j] = total;
    }
  }
}

void MatrixMultiplyPar(const double *lhs, const double *rhs, double *result,
                       size_t dim) {
#pragma omp parallel for
  for (long long i = 0; i < dim; ++i) {
    for (long long j = 0; j < dim; ++j) {
      double total{};
      for (long long k = 0; k < dim; ++k) {
        total += lhs[i * dim + k] * rhs[i * k + j];
      }
      result[i * dim + j] = total;
    }
  }
}

void BitMatrixAddSeq(const bool *lhs, const bool *rhs, bool *result,
                     size_t dim) {
  for (long long i = 0; i < dim; ++i) {
    for (long long j = 0; j < dim; ++j) {
      const long long index = i * dim + j;
      result[index] = lhs[index] | rhs[index];
    }
  }
}

void BitMatrixAddPar(const bool *lhs, const bool *rhs, bool *result,
                     size_t dim) {
#pragma omp parallel for
  for (long long i = 0; i < dim; ++i) {
    for (long long j = 0; j < dim; ++j) {
      const long long index = i * dim + j;
      result[index] = lhs[index] | rhs[index];
    }
  }
}

void BitMatrixMultiplySeq(const bool *lhs, const bool *rhs, bool *result,
                          size_t dim) {
  for (long long i = 0; i < dim; ++i) {
    for (long long j = 0; j < dim; ++j) {
      int total{};
      for (long long k = 0; k < dim; ++k) {
        total += int(lhs[i * dim + k]) * int(rhs[k * dim + j]);
      }
      result[i * dim + j] = total & 1;
    }
  }
}

void BitMatrixMultiplyPar(const bool *lhs, const bool *rhs, bool *result,
                          size_t dim) {
#pragma omp parallel for
  for (long long i = 0; i < dim; ++i) {
    for (long long j = 0; j < dim; ++j) {
      int total{};
      for (long long k = 0; k < dim; ++k) {
        total += int(lhs[i * dim + k]) * int(rhs[k * dim + j]);
      }
      result[i * dim + j] = total & 1;
    }
  }
}

}  // namespace lab5

int main() {
#ifndef _OPENMP
#error "OpenMP must be enabled"
#endif

  constexpr auto dim = 1000;
  constexpr auto size = 1000 * 1000;
  auto arrA = new double[size];
  std::uniform_int_distribution dist(0, 1000);
  std::mt19937 generator(std::random_device{}());

  std::generate(arrA, arrA + size, [&] { return dist(generator); });

  auto arrB = new double[size];

  std::generate(arrB, arrB + size, [&] { return dist(generator); });

  auto arrResult = new double[size];

  auto bitA = new bool[size];
  std::generate(bitA, bitA + size, [&] { return bool(dist(generator) & 1); });

  auto bitB = new bool[size];
  std::generate(bitB, bitB + size, [&] { return bool(dist(generator) & 1); });

  auto bitResult = new bool[size];

  {
    lab5::Measurement m("ArrayMatrix: sequential addition");
    lab5::MatrixAddSeq(arrA, arrB, arrResult, dim);
  }

  {
    lab5::Measurement m("ArrayMatrix: parallel addition");
    lab5::MatrixAddPar(arrA, arrB, arrResult, dim);
  }

  {
    lab5::Measurement m("ArrayMatrix: sequential multiplication");
    lab5::MatrixMultiplySeq(arrA, arrB, arrResult, dim);
  }

  {
    lab5::Measurement m("ArrayMatrix: parallel multiplication");
    lab5::MatrixMultiplyPar(arrA, arrB, arrResult, dim);
  }

  {
    lab5::Measurement m("BitMatrix: sequential addition");
    lab5::BitMatrixAddSeq(bitA, bitB, bitResult, dim);
  }

  {
    lab5::Measurement m("BitMatrix: parallel addition");
    lab5::BitMatrixAddPar(bitA, bitB, bitResult, dim);
  }

  {
    lab5::Measurement m("BitMatrix: sequential multiplication");
    lab5::BitMatrixMultiplySeq(bitA, bitB, bitResult, dim);
  }

  {
    lab5::Measurement m("BitMatrix: parallel multiplication");
    lab5::BitMatrixMultiplyPar(bitA, bitB, bitResult, dim);
  }

  std::cout << "\n\n----- SIMD commands ------\n\n";

  {
    lab5::Measurement m("ArrayMatrix: SIMD-powered addition");
    lab5::MatrixAddSeqSimd(arrA, arrB, arrResult, dim);
  }

  delete[] bitResult;
  delete[] bitB;
  delete[] bitA;
  delete[] arrResult;
  delete[] arrB;
  delete[] arrA;
}
