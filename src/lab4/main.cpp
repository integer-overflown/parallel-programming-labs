#include <omp.h>
#include <Windows.h>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>

#undef min
#undef max

namespace lab4 {
namespace {

typedef struct {
  double h, s;
  int begin, end, step;
} THREADSDATA, *PTHREADSDATA;

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

DWORD GetNumberOfCores() {
  SYSTEM_INFO sysInfo;
  GetSystemInfo(&sysInfo);
  return sysInfo.dwNumberOfProcessors;
}

uint32_t Factorial(uint32_t x) {
  uint32_t res{1};
  while (x > 0) {
    res *= x;
    x--;
  }
  return res;
}

double CalculatePiRamanujan() {
  double sum{};
  uint32_t n{};
  double i = (std::sqrt(8)) / 9801;

  for (;;) {
    double term = (Factorial(4 * n) / std::pow(Factorial(n), 4)) *
                  ((26390.0 * n + 1103.0) / std::pow(396.0, 4.0 * n));
    sum += term;

    if (std::abs(term) < 1e-15) {
      break;
    }

    n++;
  }

  return 1.0 / (i * sum);
}

double CalculatePiIntegral(const uint32_t numIterations = 10'000'000) {
  double h = 1.0 / numIterations, pi = 0;
  for (int i = 0; i < numIterations; i++) {
    double x = h * i;
    pi += (4.0 / (1 + x * x));
  }
  pi = h * pi;
  return pi;
}

double CalculatePiLeibniz(const uint32_t numIterations = 10'000'000) {
  double s{};
  for (int i = 1; i < numIterations; i += 4) {
    s += 1.0 / i;
    s -= 1.0 / (i + 2);
  }
  return 4.0 * s;
}

DWORD WINAPI ThreadFun(PVOID pData) {
  auto p = static_cast<PTHREADSDATA>(pData);
  double s = 0, x, h = p->h;
  for (int i = p->begin; i < p->end; i += p->step) {
    x = i * h;
    s += 4 / (1 + x * x);
  }
  p->s = s * h;
  return 0;
}

double CalculatePiIntegralParallel() {
  constexpr int32_t numIterations = 10'000'000;
  const int nThreads = omp_get_max_threads();
  double h = 1.0 / numIterations;
  auto *hThread = new HANDLE[nThreads - 1];
  auto pDatas = new THREADSDATA[nThreads];
  const int delta = numIterations / nThreads;

  for (int i = 0; i < nThreads; i++) {
    pDatas[i].begin = i * delta;
    pDatas[i].end = pDatas[i].begin + delta;
    pDatas[i].step = 1;
    pDatas[i].h = h;

    if (i != nThreads - 1) {
      hThread[i] = CreateThread(nullptr, 0, ThreadFun, &pDatas[i], 0, nullptr);
    }
  }
  ThreadFun(&pDatas[nThreads - 1]);
  WaitForMultipleObjects(nThreads - 1, hThread, TRUE, INFINITE);

  double pi = 0;
  for (int i = 0; i < nThreads - 1; i++) {
    CloseHandle(hThread[i]);
    pi += pDatas[i].s;
  }

  pi += pDatas[nThreads - 1].s;

  delete[] hThread;
  delete[] pDatas;
  return pi;
}

double CalculatePiIntegralOmp(const uint32_t numIterations = 10'000'000) {
  double h = 1.0 / numIterations;
  double pi = 0;

#pragma omp parallel for reduction(+ : pi)
  for (int i = 0; i < numIterations; i++) {
    double x = h * i;
    pi += 4.0 / (1 + x * x);
  }
  pi = h * pi;
  return pi;
}

}  // namespace
}  // namespace lab4

int main() {
  constexpr auto cDelim = "----------------------\n";

  std::cout << "OpenMP is "
            <<
#ifdef _OPENMP
      "enabled"
#else
      "disabled"
#endif
            << '\n';

  std::cout << "omp_get_max_threads: " << omp_get_max_threads() << '\n';
  std::cout << "WinAPI: " << lab4::GetNumberOfCores() << '\n';

  double res;

  {
    lab4::Measurement m("Value of PI (Ramanujan)");
    res = lab4::CalculatePiRamanujan();
  }

  std::cout << "Value of PI (Ramanujan): " << std::setprecision(16) << res
            << '\n';

  {
    lab4::Measurement m("Value of PI (CalculatePiLeibniz)");
    res = lab4::CalculatePiLeibniz();
  }

  std::cout << "Value of PI (CalculatePiLeibniz): " << std::setprecision(16)
            << res << '\n';

  std::chrono::nanoseconds integralSequentialExecTime;

  {
    lab4::Measurement m("Value of PI (CalculatePiIntegral)");
    res = lab4::CalculatePiIntegral();
    integralSequentialExecTime = m.timer().elapsed();
  }

  std::cout << "Value of PI (CalculatePiIntegral): " << std::setprecision(16)
            << res << '\n';

  std::cout << "Task #5 -- Iterations accuracy\n";
  {
    double minOffset = std::numeric_limits<double>::max();
    uint32_t it;

    for (const uint32_t iterations :
         {1'000, 10'000, 100'000, 1'000'000, 10'000'000}) {
      double offset = lab4::CalculatePiIntegral(iterations) - std::numbers::pi;

      if (std::abs(offset) < std::abs(minOffset)) {
        it = iterations;
        minOffset = offset;
      }
    }

    std::cout
        << "lab4::CalculatePiIntegral: the most exact value is produced by "
        << it << " iterations with offset " << minOffset << '\n';
  }

  {
    double minOffset = std::numeric_limits<double>::max();
    uint32_t it = 0;

    for (const uint32_t iterations :
         {1'000, 10'000, 100'000, 1'000'000, 10'000'000}) {
      double offset = lab4::CalculatePiLeibniz(iterations) - std::numbers::pi;

      if (std::abs(offset) < std::abs(minOffset)) {
        it = iterations;
        minOffset = offset;
      }
    }

    std::cout
        << "lab4::CalculatePiLeibniz: the most exact value is produced by "
        << it << " iterations with offset " << minOffset << '\n';
  }
  std::cout << cDelim;

  const double sequentialOffset =
      lab4::CalculatePiIntegral() - std::numbers::pi;
  std::cout << "Sequential offset: " << sequentialOffset << '\n';

  double parallelOffsetWinAPIThreads;
  std::chrono::nanoseconds integralWinAPIThreadsExecTime;

  std::cout << cDelim;

  std::cout << "Task #7 - Parallel integral function efficiency\n";
  {
    {
      lab4::Measurement m("Value of PI (CalculatePiIntegralParallel)");
      res = lab4::CalculatePiIntegralParallel();
      parallelOffsetWinAPIThreads = res - std::numbers::pi;
      integralWinAPIThreadsExecTime = m.timer().elapsed();
    }

    std::cout << "Value of PI (CalculatePiIntegralParallel): " << res << '\n';
    std::cout << "(WinAPI threads) Parallel offset: "
              << parallelOffsetWinAPIThreads << '\n';
    std::cout << "More precise is: "
              << (sequentialOffset < parallelOffsetWinAPIThreads
                      ? "sequential"
                      : "parallel (WinAPI)")
              << '\n';
    std::cout << cDelim;
  }

  double parallelOffsetOmp;
  std::chrono::nanoseconds parallelOmpExecTime;

  std::cout << "Task #8 - Parallel integral function efficiency\n";
  {
    {
      lab4::Measurement m("Value of PI (CalculatePiIntegralOmp)");
      res = lab4::CalculatePiIntegralOmp();
      parallelOffsetOmp = res - std::numbers::pi;
      parallelOmpExecTime = m.timer().elapsed();
    }

    std::cout << "Value of PI (CalculatePiIntegralOmp): " << res << '\n';
    std::cout << "(OMP) Parallel offset: " << parallelOffsetOmp << '\n';
    std::cout << "More precise is: "
              << (sequentialOffset < parallelOffsetOmp ? "sequential"
                                                       : "parallel (OMP)")
              << '\n';
    std::cout << "Among parallel methods, more precise is: "
              << (parallelOffsetWinAPIThreads < parallelOffsetOmp
                      ? "parallel (WinAPI)"
                      : "parallel (OMP)")
              << '\n';
  }
  std::cout << cDelim;

  std::cout << "Task #9 - Speedup comparison\n";
  {
    std::cout << "WinAPI threads vs sequential: "
              << static_cast<long double>(integralSequentialExecTime.count()) /
                     integralWinAPIThreadsExecTime.count()
              << '\n';

    std::cout << "OMP vs sequential: "
              << static_cast<long double>(integralSequentialExecTime.count()) /
                     parallelOmpExecTime.count()

              << '\n';
  }
  std::cout << cDelim;
}
