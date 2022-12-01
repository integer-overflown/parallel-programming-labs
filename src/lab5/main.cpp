#include <intrin.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <random>

#include "matrix.h"

#ifndef _OPENMP
#error "OpenMP must be enabled"
#endif

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
  auto pA = reinterpret_cast<const __m256 *>(lhs);
  auto pB = reinterpret_cast<const __m256 *>(rhs);
  auto pC = reinterpret_cast<__m256 *>(result);

  constexpr auto blockSize = sizeof(__m256) / sizeof(double);

  for (int i = 0; i < dim / blockSize; ++i) {
    for (int j = 0; j < dim / blockSize; ++j) {
      pC[i * j] = _mm256_add_ps(pA[i * j], pB[i * j]);
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

void BitMatrixAddSeqSimd(const bool *lhs, const bool *rhs, bool *result,
                         size_t dim) {
  const auto pA = reinterpret_cast<const __m256i *>(lhs);
  const auto pB = reinterpret_cast<const __m256i *>(rhs);
  const auto pC = reinterpret_cast<__m256i *>(result);

  constexpr auto blockSize = sizeof(__m256i) / sizeof(bool);

  for (int i = 0; i < dim / blockSize; ++i) {
    for (int j = 0; j < dim / blockSize; ++j) {
      pC[i * j] = _mm256_or_si256(pA[i * j], pB[i * j]);
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
  constexpr auto dim = 1000;
  constexpr auto size = 1000 * 1000;

  std::uniform_int_distribution dist(0, 1000);
  std::mt19937 generator(std::random_device{}());

  auto generateRandomDouble = [&] { return dist(generator); };
  auto generateRandomBool = [&] { return bool(dist(generator) & 1); };

  auto arrA = std::make_unique<double[]>(size);
  std::generate(arrA.get(), arrA.get() + size, generateRandomDouble);

  auto arrB = std::make_unique<double[]>(size);
  std::generate(arrB.get(), arrB.get() + size, generateRandomDouble);

  auto arrResult = std::make_unique<double[]>(size);

  auto bitA = std::make_unique<bool[]>(size);
  std::generate(bitA.get(), bitA.get() + size, generateRandomBool);

  auto bitB = std::make_unique<bool[]>(size);
  std::generate(bitB.get(), bitB.get() + size, generateRandomBool);

  auto bitResult = std::make_unique<bool[]>(size);

  {
    lab5::Measurement m("ArrayMatrix: sequential addition");
    lab5::MatrixAddSeq(arrA.get(), arrB.get(), arrResult.get(), dim);
  }

  {
    lab5::Measurement m("ArrayMatrix: parallel addition");
    lab5::MatrixAddPar(arrA.get(), arrB.get(), arrResult.get(), dim);
  }

  {
    lab5::Measurement m("ArrayMatrix: sequential multiplication");
    lab5::MatrixMultiplySeq(arrA.get(), arrB.get(), arrResult.get(), dim);
  }

  {
    lab5::Measurement m("ArrayMatrix: parallel multiplication");
    lab5::MatrixMultiplyPar(arrA.get(), arrB.get(), arrResult.get(), dim);
  }

  {
    lab5::Measurement m("BitMatrix: sequential addition");
    lab5::BitMatrixAddSeq(bitA.get(), bitB.get(), bitResult.get(), dim);
  }

  {
    lab5::Measurement m("BitMatrix: parallel addition");
    lab5::BitMatrixAddPar(bitA.get(), bitB.get(), bitResult.get(), dim);
  }

  {
    lab5::Measurement m("BitMatrix: sequential multiplication");
    lab5::BitMatrixMultiplySeq(bitA.get(), bitB.get(), bitResult.get(), dim);
  }

  {
    lab5::Measurement m("BitMatrix: parallel multiplication");
    lab5::BitMatrixMultiplyPar(bitA.get(), bitB.get(), bitResult.get(), dim);
  }

  std::cout << "\n\n----- SIMD commands ------\n\n";

  {
    lab5::Measurement m("ArrayMatrix: SIMD-powered addition");
    lab5::MatrixAddSeqSimd(arrA.get(), arrB.get(), arrResult.get(), dim);
  }

  {
    lab5::Measurement m("BitMatrix: SIMD-powered addition");
    lab5::BitMatrixAddSeqSimd(bitA.get(), bitB.get(), bitResult.get(), dim);
  }
}
