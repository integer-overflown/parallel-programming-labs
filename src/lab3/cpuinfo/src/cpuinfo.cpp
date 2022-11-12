#include "cpuinfo.h"

#include <intrin.h>

#include <array>

namespace lab3 {

static constexpr size_t cVendorStringLength = 12;

CpuInfo::CpuInfo() {
  std::array<int, 4> info;
  __cpuid(info.data(), 0);

  const int maxId = info[0];

  char vendor[cVendorStringLength + 1]{};
  *reinterpret_cast<int *>(vendor) = info[1];
  *reinterpret_cast<int *>(vendor + sizeof(int)) = info[3];
  *reinterpret_cast<int *>(vendor + 2 * sizeof(int)) = info[2];

  _vendorString = vendor;

  if (maxId < 1)
  {
    return;
  }

  __cpuid(info.data(), 1);
  _f1ecx = info[2];

  if (maxId < 7)
  {
    return;
  }

  __cpuid(info.data(), 7);
  _f7ebx = info[1];
  _f7ecx = info[2];
}

const CpuInfo& CpuInfo::Instance() {
  static CpuInfo instance;
  return instance;
}

const std::string& CpuInfo::Vendor() const { return _vendorString; }

bool CpuInfo::HasAVX() const noexcept { return _f1ecx[28]; }

bool CpuInfo::HasAVX2() const noexcept { return _f7ebx[5]; }

bool CpuInfo::HasSSE3() const noexcept { return _f1ecx[0]; }

bool CpuInfo::HasSSSE3() const noexcept { return _f1ecx[9]; }

bool CpuInfo::HasSSE41() const noexcept { return _f1ecx[19]; }

bool CpuInfo::HasSSE42() const noexcept { return _f1ecx[20]; }

}  // namespace lab3
