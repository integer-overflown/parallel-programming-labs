#include <iostream>

#include "cpuinfo.h"

#define CHECK_CPU_FEATURE(name) lab3::CpuInfo::Instance().Has##name()
#define PRINT_CPU_FEATURE_SUPPORT(name) std::cout << "Feature " << #name << " is " << (CHECK_CPU_FEATURE(name) ? "supported" : "not supported") << '\n'

int main() {
  const lab3::CpuInfo &info = lab3::CpuInfo::Instance();
  std::cout << info.Vendor() << '\n';

  PRINT_CPU_FEATURE_SUPPORT(SSE3);
  PRINT_CPU_FEATURE_SUPPORT(SSSE3);
  PRINT_CPU_FEATURE_SUPPORT(SSE41);
  PRINT_CPU_FEATURE_SUPPORT(SSE42);
  PRINT_CPU_FEATURE_SUPPORT(AVX);
  PRINT_CPU_FEATURE_SUPPORT(AVX2);
}
