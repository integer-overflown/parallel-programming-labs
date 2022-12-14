#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <span>
#include <vector>

namespace lab2 {
namespace {

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

  ~Measurement() {
    std::chrono::nanoseconds elapsed = _timer.elapsed();
    std::cout << _label << ':' << " time taken " << elapsed << '\n';
  }

 private:
  std::string _label;
  ElapsedTimer _timer;
};

namespace unoptimized {

void BubbleSort(std::span<int> array) {
  const size_t size = std::size(array);

  for (size_t p = size - 1; p >= 1; p--) {
    for (size_t i = 0; i < p; i++) {
      if (array[i] > array[i + 1]) {
        std::swap(array[i], array[i + 1]);
      }
    }
  }
}

int CountPositiveNumbers(std::span<const int> numbers) {
  int total{};
  for (int value : numbers) {
    if (value > 0) {
      ++total;
    }
  }
  return total;
}

template <typename T, size_t Extent>
  requires(std::is_floating_point_v<T>)
void RoundAll(std::span<T, Extent> array) {
  std::transform(array.begin(), array.end(), array.begin(),
                 [](T &val) { return std::round(val); });
}

template <typename T, size_t Extent1, size_t Extent2>
std::vector<T> PolynomialMultiply(std::span<T, Extent1> lhs,
                                  std::span<T, Extent2> rhs) {
  std::vector<T> result(std::size(lhs) + std::size(rhs) - 1);

  for (ptrdiff_t i = 0; i < std::size(lhs); ++i) {
    for (ptrdiff_t j = 0; j < std::size(rhs); ++j) {
      result[i + j] += lhs[i] * rhs[j];
    }
  }

  return result;
}

}  // namespace unoptimized

namespace optimized {

void BubbleSort(std::span<int> array) {
  const size_t size = std::size(array);

  for (int p = 1; p < size; p++) {
    for (int i = p - 1; i >= 0; i--) {
      if (array[i] > array[i + 1]) {
        std::swap(array[i], array[i + 1]);
      }
    }
  }
}

int CountPositiveNumbers(std::span<const int> numbers) {
  int total{};
  for (int value : numbers) {
    // this is going to fail for std::numeric_limits<int>::min(), because of -1
    total += !bool(static_cast<unsigned int>(value - 1) &
                   ~std::numeric_limits<int>::max());
  }
  return total;
}

template <typename T, size_t Extent>
  requires(std::is_floating_point_v<T>)
void RoundAll(std::span<T, Extent> array) {
  std::transform(array.begin(), array.end(), array.begin(),
                 [](T val) { return int64_t(val + T{0.5}); });
}

template <typename T, size_t Extent>
std::vector<T> PolynomialMultiply(std::span<T, Extent> lhs,
                                  std::span<T, Extent> rhs) {
  const size_t size = std::size(lhs);
  std::vector<T> result(2 * size);

  for (size_t j = 0; j < size; j++) {
    if (rhs[j] != 0) {
      auto pz = result.begin() + j;

      if (rhs[j] == 1) {
        for (size_t i = 0; i < size; i++) pz[i] += lhs[i];
      } else {
        for (size_t i = 0; i < size; i++) pz[i] -= lhs[i];
      }
    }
  }

  return result;
}

}  // namespace optimized

template <typename Container>
void Print(const Container &container) {
  std::for_each(begin(container), end(container),
                [](auto item) { std::cout << item << ' '; });
  std::cout << '\n';
}

}  // namespace
}  // namespace lab2

int main() {
  const std::array numbers{1, -2, -1, -2, 0, 2000, -20, 23, -23};
  int numPositive;

  {
    lab2::Measurement m("unoptimized::CountPositiveNumbers");
    numPositive = lab2::unoptimized::CountPositiveNumbers(numbers);
  }

  std::cout << "unoptimized::CountPositiveNumbers" << ':' << ' ' << numPositive
            << '\n';

  {
    lab2::Measurement m("optimized::CountPositiveNumbers");
    numPositive = lab2::optimized::CountPositiveNumbers(numbers);
  }

  std::cout << "optimized::CountPositiveNumbers" << ':' << ' ' << numPositive
            << '\n';

  {
    std::array test(numbers);
    lab2::Measurement m("unoptimized::BubbleSort");
    lab2::unoptimized::BubbleSort(test);
  }
  lab2::Print(numbers);

  {
    std::array test(numbers);
    lab2::Measurement m("optimized::BubbleSort");
    lab2::optimized::BubbleSort(test);
  }
  lab2::Print(numbers);

  std::array handfulOfDoubles{1.3, 2.1, 6.3, 2.2};
  {
    std::array test(handfulOfDoubles);
    lab2::Measurement m("lab2::unoptimized::RoundAll");
    lab2::unoptimized::RoundAll(std::span{test});
  }

  {
    std::array test(handfulOfDoubles);
    lab2::Measurement m("lab2::optimized::RoundAll");
    lab2::optimized::RoundAll(std::span{test});
  }

  std::array lhs{5, 0, 10, 1, 3, 2, 1, 3, 12, 21, 33, 11};
  std::array rhs{0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1};

  {
    lab2::Measurement m("unoptimized::PolynomialMultiply");
    lab2::unoptimized::PolynomialMultiply(std::span{lhs}, std::span{rhs});
  }

  {
    lab2::Measurement m("optimized::PolynomialMultiply");
    lab2::optimized::PolynomialMultiply(std::span{lhs}, std::span{rhs});
  }

  return EXIT_SUCCESS;
}
