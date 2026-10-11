// PCIe topology from sysfs: the path from a device up to its root port, each hop's link, and
// the bottleneck along it.
//
// Two traps this exists to avoid: the R9700 has an internal switch, so its own endpoint reports
// "32 GT/s x16" while the real slot link (the root port) is Gen4 x4; and idle links sit at
// 2.5 GT/s under ASPM, so the *current* link speed says nothing about capacity. The bottleneck
// is the minimum over every hop of the *maximum* link (both ends of a link report it; the
// negotiated rate is never above either end's maximum).
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace unholy::hw {

struct PciHop {
  std::string bdf;          // e.g. "0000:c1:00.0"
  uint32_t class_code = 0;  // 0x060400 = PCI bridge, 0x030000 = display, ...
  double max_gts = 0;       // GT/s, 0 if the attribute is absent
  int max_width = 0;
  double cur_gts = 0;
  int cur_width = 0;
};

struct PciPath {
  std::vector<PciHop> hops;  // endpoint first, root port last

  // Bytes/s the slowest link on the path can carry at its negotiated maximum
  // (encoding overhead included: 8b/10b below 8 GT/s, 128b/130b from Gen3).
  double bottleneck_bytes_per_s() const;
  // The hop that sets the bottleneck (index into hops), -1 if none has link attributes.
  int bottleneck_hop() const;
};

// Normalizes "c3:00.0" / "0000:c3:00.0" / upper case to "0000:c3:00.0".
std::string normalize_bdf(const std::string& bdf);

std::optional<PciPath> pci_path(const std::string& bdf);

// Effective unidirectional bytes/s of one link.
double link_bytes_per_s(double gts, int width);

// Where two devices' paths meet. `via_root_complex` means the first shared element is the host
// bridge (today's layout: each card on its own root port). Otherwise `common_bdf` is the
// lowest shared bridge (a switch port: the planned PLX layout), and traffic between the two
// devices can stay below it.
struct PathJoin {
  bool via_root_complex = true;
  std::string common_bdf;
};
PathJoin join_paths(const PciPath& a, const PciPath& b);

}  // namespace unholy::hw
