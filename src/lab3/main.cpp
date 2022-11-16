#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#include <array>

#include "cpuinfo.h"

#define CHECK_CPU_FEATURE(name) lab3::CpuInfo::Instance().Has##name()
#define PRINT_CPU_FEATURE_SUPPORT(name)                                  \
  std::cout << "Feature " << #name << " is "                             \
            << (CHECK_CPU_FEATURE(name) ? "supported" : "not supported") \
            << '\n'

namespace lab3 {
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

template <typename T>
std::vector<T> SumUp(std::vector<T> const &a, std::vector<T> const &b) {
  const size_t size = std::min(a.size(), b.size());
  std::vector<T> res(size);

  for (size_t i = 0; i < size; ++i) {
    res[i] = std::abs(a[i]) + std::abs(b[i]);
  }

  return res;
}

namespace simd {
template <typename T>
std::vector<T> SumUp(std::vector<T> const &a, std::vector<T> const &b) {
  const size_t size = std::min(a.size(), b.size());
  std::vector<T> res(size);

#pragma omp simd
  for (size_t i = 0; i < size; ++i) {
    res[i] = std::abs(a[i]) + std::abs(b[i]);
  }

  return res;
}
}  // namespace simd

template<typename T>
const char *typeName() noexcept;

template<>
const char *typeName<int8_t>() noexcept { return "int8"; }

template<>
const char *typeName<int16_t>() noexcept { return "int16"; }

template<>
const char *typeName<int32_t>() noexcept { return "int32"; }

template<>
const char *typeName<int64_t>() noexcept { return "int64"; }

template<>
const char *typeName<float>() noexcept { return "float"; }

template<>
const char *typeName<double>() noexcept { return "double"; }

}  // namespace
}  // namespace lab3

int main() {
  const lab3::CpuInfo &info = lab3::CpuInfo::Instance();
  std::cout << info.Vendor() << '\n';

  PRINT_CPU_FEATURE_SUPPORT(SSE3);
  PRINT_CPU_FEATURE_SUPPORT(SSSE3);
  PRINT_CPU_FEATURE_SUPPORT(SSE41);
  PRINT_CPU_FEATURE_SUPPORT(SSE42);
  PRINT_CPU_FEATURE_SUPPORT(AVX);
  PRINT_CPU_FEATURE_SUPPORT(AVX2);

  std::array a{1, 2, 3, 4, 5, 6};
  std::array b{2, 3, 4, 5, 6, 7};
  std::vector<int> c(a.size());

//  {
//    lab3::Measurement m("Simple vector addition");
//    lab3::SumUp(a, b);
//  }

  [=]<typename... Args>
  {
    auto measure = [&]<typename T>() {
      using namespace std::string_literals;
      std::vector<T> lhs(a.begin(), a.end());
      std::vector<T> rhs(b.begin(), b.end());

      lab3::Measurement m("SIMD-powered vector addition: type "s + lab3::typeName<T>());
      lab3::simd::SumUp(lhs, rhs);
    };

    (measure.template operator()<Args>(), ...);
  }.
  operator()<int8_t, int16_t, int32_t, int64_t, float, double>();
}
