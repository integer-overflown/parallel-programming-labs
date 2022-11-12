#ifndef PARALLEL_PROGRAMMING_CPUINFO_H
#define PARALLEL_PROGRAMMING_CPUINFO_H
#include <string>
#include <bitset>
#include <climits>

namespace lab3 {

class CpuInfo {
 public:
  CpuInfo();

  static const CpuInfo &Instance();

  [[nodiscard]] const std::string &Vendor() const;

  [[nodiscard]] bool HasAVX() const noexcept;

  [[nodiscard]] bool HasAVX2() const noexcept;

  [[nodiscard]] bool HasSSE3() const noexcept;

  [[nodiscard]] bool HasSSSE3() const noexcept;

  [[nodiscard]] bool HasSSE41() const noexcept;

  [[nodiscard]] bool HasSSE42() const noexcept;

 private:
  std::string _vendorString;
  std::bitset<sizeof(int) * CHAR_BIT> _f1ecx;
  std::bitset<sizeof(int) * CHAR_BIT> _f7ebx;
  std::bitset<sizeof(int) * CHAR_BIT> _f7ecx;
};

}  // namespace lab3

#endif  // PARALLEL_PROGRAMMING_CPUINFO_H
