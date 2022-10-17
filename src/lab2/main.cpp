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
    std::cout << _label << ':' << " time taken " << _timer.elapsed() << '\n';
  }

 private:
  std::string _label;
  ElapsedTimer _timer;
};

namespace unoptimized {

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


}  // namespace unoptimized

namespace optimized {
int CountPositiveNumbers(std::span<const int> numbers) {
  int total{};
  for (int value : numbers) {
    total += !bool(static_cast<unsigned int>(value) &
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


}  // namespace optimized

void BubbleSort(std::span<int> array) {
  const size_t size = std::size(array);
  bool swapped;

  do {
    swapped = false;

    for (size_t i = 0; i < size - 1; i++) {
      if (array[i] > array[i + 1]) {
        std::swap(array[i], array[i + 1]);
        swapped = true;
      }
    }

  } while (swapped);
}

template <typename Container>
void Print(const Container &container) {
  std::for_each(begin(container), end(container),
                [](auto item) { std::cout << item << ' '; });
  std::cout << '\n';
}

template <typename T, size_t Extent1, size_t Extent2>
std::vector<T> polynomialMultiply(std::span<T, Extent1> lhs,
                                  std::span<T, Extent2> rhs) {
  std::vector<T> result(std::size(lhs) + std::size(rhs) - 1);

  for (ptrdiff_t i = 0; i < std::size(lhs); ++i) {
    for (ptrdiff_t j = 0; j < std::size(rhs); ++j) {
      result[i + j] += lhs[i] * rhs[j];
    }
  }

  return result;
}

}  // namespace
}  // namespace lab2

int main() {
  std::array numbers{1, 2, -1, -2, 0, 2000};
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
    lab2::Measurement m("lab2::BubbleSort");
    lab2::BubbleSort(numbers);
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

  std::array lhs{5, 0, 10, 6};
  std::array rhs{1, 2, 4};
  {
    lab2::Measurement m("lab2::polynomialMultiply");
    lab2::polynomialMultiply(std::span{lhs}, std::span{rhs});
  }

  return EXIT_SUCCESS;
}
