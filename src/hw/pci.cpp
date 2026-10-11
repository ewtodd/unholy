#include "hw/pci.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>

namespace fs = std::filesystem;

namespace unholy::hw {

namespace {

std::optional<std::string> read_attr(const fs::path& dir, const char* name) {
  std::ifstream f(dir / name);
  if (!f) return std::nullopt;
  std::string s;
  std::getline(f, s);
  return s;
}

// "16.0 GT/s PCIe" -> 16.0; "Unknown" -> 0.
double parse_gts(const std::optional<std::string>& s) {
  if (!s) return 0;
  return std::strtod(s->c_str(), nullptr);
}

int parse_int(const std::optional<std::string>& s, int base = 10) {
  if (!s) return 0;
  return static_cast<int>(std::strtol(s->c_str(), nullptr, base));
}

bool is_bdf(const std::string& s) {
  // dddd:bb:dd.f
  return s.size() == 12 && s[4] == ':' && s[7] == ':' && s[10] == '.';
}

}  // namespace

std::string normalize_bdf(const std::string& bdf) {
  std::string s = bdf;
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
  if (s.size() == 7) s = "0000:" + s;  // "c3:00.0"
  return s;
}

double link_bytes_per_s(double gts, int width) {
  if (gts <= 0 || width <= 0) return 0;
  const double enc = gts < 8.0 ? 0.8 : 128.0 / 130.0;
  return gts * 1e9 * enc * width / 8.0;
}

std::optional<PciPath> pci_path(const std::string& bdf_in) {
  const std::string bdf = normalize_bdf(bdf_in);
  std::error_code ec;
  fs::path p = fs::canonical(fs::path("/sys/bus/pci/devices") / bdf, ec);
  if (ec) return std::nullopt;

  PciPath path;
  // Walk up while the directory is a PCI function (has a BDF name); stop at pciDDDD:BB.
  while (is_bdf(p.filename().string())) {
    PciHop hop;
    hop.bdf = p.filename().string();
    hop.class_code = static_cast<uint32_t>(parse_int(read_attr(p, "class"), 16));
    hop.max_gts = parse_gts(read_attr(p, "max_link_speed"));
    hop.max_width = parse_int(read_attr(p, "max_link_width"));
    hop.cur_gts = parse_gts(read_attr(p, "current_link_speed"));
    hop.cur_width = parse_int(read_attr(p, "current_link_width"));
    path.hops.push_back(hop);
    p = p.parent_path();
  }
  if (path.hops.empty()) return std::nullopt;
  return path;
}

int PciPath::bottleneck_hop() const {
  int best = -1;
  double best_bw = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < hops.size(); ++i) {
    const double bw = link_bytes_per_s(hops[i].max_gts, hops[i].max_width);
    if (bw > 0 && bw < best_bw) {
      best_bw = bw;
      best = static_cast<int>(i);
    }
  }
  return best;
}

double PciPath::bottleneck_bytes_per_s() const {
  const int i = bottleneck_hop();
  return i < 0 ? 0 : link_bytes_per_s(hops[i].max_gts, hops[i].max_width);
}

PathJoin join_paths(const PciPath& a, const PciPath& b) {
  // Compare from the root end; the last shared bridge is the join point.
  PathJoin j;
  auto ia = a.hops.rbegin();
  auto ib = b.hops.rbegin();
  for (; ia != a.hops.rend() && ib != b.hops.rend() && ia->bdf == ib->bdf; ++ia, ++ib) {
    j.via_root_complex = false;
    j.common_bdf = ia->bdf;
  }
  return j;
}

}  // namespace unholy::hw
