#include <intrin.h>
#include <omp.h>
#include <Windows.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <ranges>
#include <vector>

#undef max  // picked up from Windows.h, clashes with std definitions
#undef min

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
T SumUp(std::span<const T> numbers) {
  return std::accumulate(numbers.begin(), numbers.end(), T{0});
}

template <auto start, auto end>
auto randomInRange() {
  static std::uniform_int_distribution<int> dist{start, end};
  static std::random_device device;
  static std::mt19937 engine{device()};
  return dist(engine);
}

[[nodiscard]] uint64_t GetPerformanceFrequency() {
  static uint64_t cached = []() -> uint64_t {
    LARGE_INTEGER performanceFrequency;
    QueryPerformanceFrequency(&performanceFrequency);
    return performanceFrequency.QuadPart;
  }();
  return cached;
}

std::chrono::nanoseconds ConvertTicksToNs(uint64_t ticks) {
  using ldouble = long double;
  auto ns = ldouble(ticks) * 1e+9L / ldouble(GetPerformanceFrequency());
  return std::chrono::nanoseconds{static_cast<long long>(std::floor(ns))};
}

uint64_t GetPerformanceCounterValue() {
  LARGE_INTEGER counterValue;
  QueryPerformanceCounter(&counterValue);
  return counterValue.QuadPart;
}

template <typename Type, size_t Size>
std::array<Type, Size> GenerateAscendingSequence() {
  std::array<Type, Size> values{};
  std::ranges::generate(values.begin(), values.end(),
                        [n = uint32_t(0)]() mutable { return n++; });
  return values;
}

void RunTask3() {
  auto values = GenerateAscendingSequence<uint32_t, 1000>();

  // cache the performance frequency
  (void)lab1::GetPerformanceFrequency();

  constexpr auto cMeasurementsCount = 10;

  {
    uint64_t result = BENCHMARK(cMeasurementsCount, __rdtsc,
                                { lab1::SumUp<uint32_t>(values); });
    std::cout << result << " ticks\n";
  }

  {
    uint64_t result =
        BENCHMARK(cMeasurementsCount, lab1::GetPerformanceCounterValue,
                  { lab1::SumUp<uint32_t>(values); });
    std::cout << lab1::ConvertTicksToNs(result) << " nanoseconds\n";
  }
}

void RunTask4() {
  auto testSequenceWithSizeAbsolute = [](size_t arraySize) {
    std::vector<uint32_t> values;
    values.reserve(arraySize);
    std::generate_n(std::back_inserter(values), arraySize,
                    [n = uint32_t(0)]() mutable { return n++; });

    double result =
        BENCHMARK(1, omp_get_wtime, { lab1::SumUp<uint32_t>(values); });

    std::cout << "[Absolute] " << result << " seconds" << '\n';
  };

  auto testSequenceWithSizeRelative = [](size_t arraySize) {
    constexpr DWORD maxTime = 2'000;  // 2ms
    DWORD remainingTime = maxTime;
    DWORD cycles{};

    std::vector<uint32_t> values;
    values.reserve(arraySize);
    std::generate_n(std::back_inserter(values), arraySize,
                    [n = uint32_t(0)]() mutable { return n++; });

    do {
      DWORD result =
          BENCHMARK(1, GetTickCount, { lab1::SumUp<uint32_t>(values); });
      if (remainingTime > result) {  // would underflow otherwise
        remainingTime -= result;
      } else {
        break;
      }
      ++cycles;
    } while (remainingTime > 0);

    std::cout << "[Relative] function runs " << cycles
              << " times in 2 seconds\n";
  };

  for (int c : {100'000, 200'000, 300'000}) {
    std::cout << "Test for " << c << " elements\n";
    testSequenceWithSizeAbsolute(c);
    testSequenceWithSizeRelative(c);
    std::cout << "------------------------------\n";
  }
}

}  // namespace
}  // namespace lab1

int main() {
  lab1::RunTask3();
  lab1::RunTask4();
}
