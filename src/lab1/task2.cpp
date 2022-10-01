#include <intrin.h>
#include <omp.h>
#include <Windows.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <stdexcept>

#define LAB1_ALWAYS_INLINE inline __forceinline

#define LAB1_TASK2_MEASURE_TIME_FUNCTION_PRECISION(function, functionName, \
                                                   units)                  \
  do {                                                                     \
    auto getTime = function;                                               \
    auto val = getTime();                                                  \
    for (;;) {                                                             \
      if (auto newVal = getTime(); newVal != val) {                        \
        std::cout << "Function " << functionName << " precision is" << ' ' \
                  << (newVal - val) << ' ' << units << '\n';               \
        break;                                                             \
      }                                                                    \
    }                                                                      \
  } while (0)

namespace lab1 {
namespace {

LAB1_ALWAYS_INLINE uint64_t
GetSystemTimeGenericWrapper(void (*getTimeFunction)(LPFILETIME)) {
  FILETIME fileTime;
  getTimeFunction(&fileTime);
  ULARGE_INTEGER largeIntRepresentation;
  largeIntRepresentation.LowPart = fileTime.dwLowDateTime;
  largeIntRepresentation.HighPart = fileTime.dwHighDateTime;
  return largeIntRepresentation.QuadPart;
}

LAB1_ALWAYS_INLINE uint64_t GetSystemTimeAsFileTime() {
  return GetSystemTimeGenericWrapper(::GetSystemTimeAsFileTime);
}

LAB1_ALWAYS_INLINE uint64_t GetSystemTimePreciseAsFileTime() {
  return GetSystemTimeGenericWrapper(::GetSystemTimePreciseAsFileTime);
}

LAB1_ALWAYS_INLINE uint64_t ReadTimestampCounterNative() {
#ifdef LAB1_CXX_INLINE_ASSEMBLY_SUPPORTED
  uint64_t c;

  asm volatile(
      "rdtsc\n\t"           // Returns the time in EDX:EAX.
      "shl $32, %%rdx\n\t"  // Shift the upper bits left.
      "or %%rdx, %0"        // 'Or' in the lower bits.
      : "=a"(c)
      :
      : "rdx");

  return c;
#else
  throw std::runtime_error(
      "This function is not supported on current compiler");
#endif
}

}  // namespace
}  // namespace lab1

int main() {
  LAB1_TASK2_MEASURE_TIME_FUNCTION_PRECISION([] { return time(nullptr); },
                                             "time", "seconds");
  LAB1_TASK2_MEASURE_TIME_FUNCTION_PRECISION(clock, "clock", "milliseconds");
  LAB1_TASK2_MEASURE_TIME_FUNCTION_PRECISION(lab1::GetSystemTimeAsFileTime,
                                             "GetSystemTimeAsFileTime",
                                             "hundreds of nanoseconds");
  LAB1_TASK2_MEASURE_TIME_FUNCTION_PRECISION(
      lab1::GetSystemTimePreciseAsFileTime, "GetSystemTimePreciseAsFileTime",
      "hundreds of nanoseconds");
  LAB1_TASK2_MEASURE_TIME_FUNCTION_PRECISION(GetTickCount, "GetTickCount",
                                             "milliseconds");
  LAB1_TASK2_MEASURE_TIME_FUNCTION_PRECISION([] { return __rdtsc(); },
                                             "__rdtsc", "ticks");
#ifdef LAB1_CXX_INLINE_ASSEMBLY_SUPPORTED
  LAB1_TASK2_MEASURE_TIME_FUNCTION_PRECISION(
      lab1::ReadTimestampCounterNative, "rdtsc (x64 inline assembly)", "ticks");
#else
  std::cerr << "Skipping test: rdtsc (x64 inline assembly)" << '\n'
            << "reason: function not supported" << std::endl;
#endif

  LAB1_TASK2_MEASURE_TIME_FUNCTION_PRECISION(
      [] {
        LARGE_INTEGER value;
        QueryPerformanceCounter(&value);
        return value.QuadPart;
      },
      "QueryPerformanceCounter", "units");

  LARGE_INTEGER performanceFrequency;
  QueryPerformanceFrequency(&performanceFrequency);
  std::cout << "* where unit is: "
            << 1e+9 / static_cast<long double>(performanceFrequency.QuadPart)
            << ' ' << "nanoseconds" << '\n';

  LAB1_TASK2_MEASURE_TIME_FUNCTION_PRECISION(
      std::chrono::high_resolution_clock::now,
      "std::chrono::high_resolution_clock", "nanoseconds");

#ifdef LAB1_HAVE_OMP
  LAB1_TASK2_MEASURE_TIME_FUNCTION_PRECISION(omp_get_wtime, "omp_get_wtime",
                                             "seconds");
#else
  std::cerr << "Skipping OMP test: reason: not supported\n";
#endif
}
