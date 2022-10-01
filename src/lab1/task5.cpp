#include <omp.h>

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <variant>

#include "matrix.h"

#define BENCHMARK(times, timeFun, body)                         \
  ([&] {                                                        \
    auto val = std::numeric_limits<decltype(timeFun())>::max(); \
    for (int i = 0; i < times; ++i) {                           \
      auto start = timeFun();                                   \
      { body }                                                  \
      auto end = timeFun() - start;                             \
      val = std::min(val, end);                                 \
    }                                                           \
    return val;                                                 \
  }())

namespace lab1 {
namespace {

template <typename T>
class Array2DFree {
 public:
  explicit Array2DFree(size_t rows) : _rows{rows} {}

  void operator()(T **ptr) {
    std::for_each(ptr, ptr + _rows, [](T *sub) { delete[] sub; });
    delete[] ptr;
  }

 private:
  size_t _rows{};
};

struct uninitialized_t {
  explicit uninitialized_t() = default;
};

template <typename T>
using Array2DAutoPtr = std::unique_ptr<T *, Array2DFree<T>>;

namespace detail {

template <typename T, typename AllocRow>
std::unique_ptr<T *, Array2DFree<T>> Allocate2DArray(size_t rows, size_t cols,
                                                     AllocRow &&alloc) {
  T **array = new T *[rows];
  std::generate(
      array, array + rows,
      [cols, alloc = std::forward<AllocRow>(alloc)] { return alloc(cols); });
  return {array, Array2DFree<T>(rows)};
}

}  // namespace detail

template <typename T>
std::unique_ptr<T *, Array2DFree<T>> Allocate2DArray(size_t rows, size_t cols) {
  return detail::Allocate2DArray<T>(rows, cols,
                                    [](size_t size) { return new T[size]{}; });
}

template <typename T>
std::unique_ptr<T *, Array2DFree<T>> Allocate2DArray(size_t rows, size_t cols,
                                                     uninitialized_t) {
  return detail::Allocate2DArray<T>(rows, cols,
                                    [](size_t size) { return new T[size]; });
}

template <typename T>
void SquareMatrixMultiply(T **lhs, T **rhs, T **result, size_t size) {
  for (ptrdiff_t i = 0; i < size; ++i) {
    for (ptrdiff_t j = 0; j < size; ++j) {
      T total{};
      for (ptrdiff_t k = 0; k < size; ++k) {
        total += lhs[i][k] * rhs[k][j];
      }
      result[i][j] = total;
    }
  }
}

template <typename T>
void Generate2D(lab1::Array2DAutoPtr<T> &array, size_t rows, size_t cols,
                T (*gen)(ptrdiff_t, ptrdiff_t)) {
  std::for_each(
      array.get(), array.get() + rows, [=, i = ptrdiff_t(0)](T *row) mutable {
        std::generate(row, row + cols, [=, col = ptrdiff_t(0)]() mutable {
          return gen(i, col++);
        });
        ++i;
      });
}

template <typename T>
void Print2D(lab1::Array2DAutoPtr<T> &array, size_t rows, size_t cols) {
  std::for_each(array.get(), array.get() + rows, [=](T *row) {
    std::for_each(row, row + cols,
                  [](T elem) { std::cout << std::setw(5) << elem << ' '; });
    std::cout << '\n';
  });
}

enum class ParseError { ValueOutOfBounds, InvalidInteger };

std::variant<uint64_t, ParseError> ParseUInt64(std::string_view string) {
  char *endPtr;
  errno = 0;
  uint64_t value = std::strtoull(string.data(), &endPtr, 10);

  if (errno == ERANGE) {
    return ParseError::ValueOutOfBounds;
  }

  if (*endPtr != '\0') {
    return ParseError::InvalidInteger;
  }

  return value;
}

void BenchmarkNoObjects(size_t size)
{
  constexpr auto label = "(no objects) ";

  lab1::Array2DAutoPtr<float> a =
      lab1::Allocate2DArray<float>(size, size, lab1::uninitialized_t{});
  lab1::Generate2D(
      a, size, size, +[](ptrdiff_t i, ptrdiff_t j) { return float(i + j); });

  lab1::Array2DAutoPtr<float> b =
      lab1::Allocate2DArray<float>(size, size, lab1::uninitialized_t{});
  lab1::Generate2D(
      b, size, size, +[](ptrdiff_t i, ptrdiff_t j) { return float(i == j); });

  lab1::Array2DAutoPtr<float> result =
      lab1::Allocate2DArray<float>(size, size, lab1::uninitialized_t{});

  std::cout << label << "Starting benchmark...\n";

  double execTime = BENCHMARK(1, omp_get_wtime, {
    lab1::SquareMatrixMultiply(a.get(), b.get(), result.get(), size);
  });

  std::cout << label << "time taken: " << execTime << " seconds\n";
}

void BenchmarkUsingObjects(size_t size)
{
  constexpr auto label = "(using objects) ";

  Matrix<float> a(size, size, [](ptrdiff_t i, ptrdiff_t j) { return float(i + j); });
  Matrix<float> b(size, size, [](ptrdiff_t i, ptrdiff_t j) { return float(i == j); });

  std::cout << label << "Starting benchmark...\n";

  double execTime = BENCHMARK(1, omp_get_wtime, {
    (void) a.multiply(b);
  });

  std::cout << label << "time taken: " << execTime << " seconds\n";
}

}  // namespace
}  // namespace lab1

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "usage: " << (argc == 1 ? argv[0] : "<executable>") << ' '
              << "[matrix size]\n";
    return EXIT_FAILURE;
  }

  uint64_t size;

  if (auto parseResult = lab1::ParseUInt64(argv[1]);
      std::holds_alternative<lab1::ParseError>(parseResult)) {
    std::cerr << argv[1] << ' '
              << (std::get<lab1::ParseError>(parseResult) ==
                          lab1::ParseError::InvalidInteger
                      ? "is not a valid integer"
                      : "is too large")
              << '\n';
    return EXIT_FAILURE;
  } else {
    size = std::get<uint64_t>(parseResult);
  }

  std::cout << "Running benchmarks on matrices with size " << size << '\n';

  lab1::BenchmarkNoObjects(size);

  lab1::BenchmarkUsingObjects(size);
}
